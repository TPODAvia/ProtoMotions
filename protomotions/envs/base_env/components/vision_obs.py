"""VisionObs component: integrates CenterPose for image->pose observations.

This implementation assumes that
- NVLabs/CenterPose is cloned locally (environment variable ``CENTERPOSE_ROOT`` or
  config entry ``centerpose_config.repo_root`` points to the repo root).
- The DCNv2 CUDA extension inside CenterPose was built (``./make.sh``).
- A trained model checkpoint (``.pth``) is available and specified in
  ``centerpose_config.model_path``.
- Your environment implements ``env.get_images(env_ids)`` which returns a list of
  RGB **numpy** images (``uint8``) shaped ``[H, W, 3]`` in either RGB or BGR.  If
  the channel order is RGB, set ``centerpose_config.bgr`` to ``False``.

The code loads CenterPose *once* and re‑uses it for all subsequent calls. The
output is a 10‑D vector per environment consisting of
  [tx, ty, tz, qx, qy, qz, qw, sx, sy, sz]
where ``t`` is the object translation (metres), ``q`` is a unit quaternion
(rotation from object to camera frame) and ``s`` are the 3D side lengths of the
object's axis‑aligned cuboid as predicted by CenterPose.  If no object is
found, the vector is all zeros.

If you need a different encoding (keypoints, multiple objects, etc.) adapt the
``_encode_detection`` method.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import cv2  # type: ignore
import numpy as np
import torch

from protomotions.envs.base_env.components.base_component import BaseComponent


class VisionObs(BaseComponent):
    """Observation component that can provide a vision‑based context.

    When ``vision_obs.enabled`` and ``vision_obs.use_centerpose`` are both
    ``True`` in the YAML config, this component runs a *single* forward pass of
    NVLabs CenterPose on the current camera images and encodes the highest‑
    scoring detection as a 10‑D pose vector per environment.
    """

    # -------------------------- public API ---------------------------------

    def __init__(self, config, env):
        super().__init__(config, env)
        self._vision_context: torch.Tensor | None = None

        if self.config.vision_obs.enabled and self.config.vision_obs.use_centerpose:
            self._init_centerpose()
        else:
            self._centerpose_ready = False

    def compute_observations(self, env_ids: List[int]):
        """Populate ``self._vision_context`` for the given env_ids."""
        if not self.config.vision_obs.enabled:
            self._vision_context = None
            return

        if self.config.vision_obs.use_centerpose and self._centerpose_ready:
            images = self.env.get_images(env_ids)  # type: ignore[attr-defined]
            if len(images) != len(env_ids):
                raise ValueError(
                    "env.get_images(env_ids) must return len(env_ids) images, got "
                    f"{len(images)}"
                )
            with torch.no_grad():
                self._vision_context = self._run_centerpose(images)
        else:
            # Fall back to ground‑truth (direct) object observations.
            self._vision_context = self._get_direct_observation(env_ids)

    def get_obs(self) -> Dict[str, torch.Tensor]:
        if self._vision_context is not None:
            return {"vision_context": self._vision_context.clone()}
        return {}

    # ------------------------ internal helpers -----------------------------

    # ---- CenterPose initialisation ----

    def _init_centerpose(self):
        cp_cfg = self.config.vision_obs.centerpose_config
        repo_root = cp_cfg.get("repo_root", os.getenv("CENTERPOSE_ROOT", ""))
        if not repo_root:
            raise RuntimeError(
                "CenterPose repo root not specified. Either set CENTERPOSE_ROOT "
                "or provide centerpose_config.repo_root in the YAML config."
            )
        repo_src = os.path.join(repo_root, "src")
        if repo_src not in sys.path:
            sys.path.insert(0, repo_src)

        try:
            from opts import opts  # type: ignore
            from detector import Detector  # type: ignore
        except ImportError as exc:  # pragma: no cover – informative error path
            raise ImportError(
                "Failed to import CenterPose. Make sure the repo is cloned and "
                "its 'src' directory is on PYTHONPATH, and that DCNv2 was "
                "compiled successfully."
            ) from exc

        model_path = cp_cfg.get("model_path")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                "CenterPose model checkpoint (.pth) not found. "
                "Set centerpose_config.model_path in the YAML config."
            )

        arch = cp_cfg.get("arch", "dla_34")
        dataset = cp_cfg.get("dataset", "objectron")
        top_k = str(cp_cfg.get("topK", 1))
        gpu_id = "0" if self.env.device.type == "cuda" else "-1"

        # Build the CLI‑style arg list expected by opts().init().
        default_args = [
            "--demo", "",  # dummy placeholder required by opts
            "--arch", arch,
            "--dataset", dataset,
            "--K", top_k,
            "--load_model", model_path,
            "--gpu", gpu_id,
        ]
        # allow user to pass custom extra_args list in YAML
        default_args.extend(cp_cfg.get("extra_args", []))

        # Parse options and build the detector.
        self._cp_opt = opts().init(default_args)  # type: ignore[attr-defined]
        self._cp_detector = Detector(self._cp_opt)  # type: ignore[call-arg]
        self._cp_detector.model.to(self.env.device)
        self._cp_detector.model.eval()

        # channel order: assume BGR (OpenCV) unless explicitly stated otherwise
        self._bgr = cp_cfg.get("bgr", True)
        # Normalisation constants – fall back to CenterNet defaults.
        self._mean = np.array(cp_cfg.get("mean", [0.407, 0.447, 0.470]), dtype=np.float32)
        self._std = np.array(cp_cfg.get("std", [0.274, 0.272, 0.281]), dtype=np.float32)

        self._centerpose_ready = True

    # ---- CenterPose inference ----

    def _run_centerpose(self, images: List[np.ndarray]) -> torch.Tensor:
        """Run CenterPose on a batch of images and return encoded pose vectors.

        Parameters
        ----------
        images : list of np.ndarray
            Each image should be ``uint8`` in RGB or BGR according to self._bgr.

        Returns
        -------
        torch.Tensor  # shape [B, 10]
        """
        batch_vectors: List[torch.Tensor] = []
        for img in images:
            if img.dtype != np.uint8:
                raise ValueError("Images must be uint8 numpy arrays (0‑255)")
            if img.shape[-1] != 3:
                raise ValueError("Images must have 3 channels")

            # If we received RGB but CenterPose expects BGR, convert.
            if not self._bgr:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # The Detector object performs its own resizing and normalisation.
            # We therefore pass the raw cv2 BGR image to detector.run().
            result = self._cp_detector.run(img)  # type: ignore[arg-type]
            vec = self._encode_detection(result)
            batch_vectors.append(vec)

        return torch.stack(batch_vectors, dim=0)

    # ---- Encoding helper ----

    def _encode_detection(self, detector_output: Dict) -> torch.Tensor:
        """Map CenterPose raw output -> 10‑D pose vector.

        detector_output (dict) has the structure returned by ``Detector.run``.
        We extract the *first* detection (highest confidence) and build a 10‑D
        vector [tx,ty,tz, qx,qy,qz,qw, sx,sy,sz]. If no detection is present we
        return zeros.
        """
        results = detector_output.get("results", [])
        if len(results) == 0:
            return torch.zeros(10, device=self.env.device, dtype=torch.float32)

        det = results[0]  # highest score
        # The keys below correspond to CenterPose's ``Detector.run`` output.
        # They may vary depending on the model/dataset.  Adjust if needed.
        translation = det.get("T", det.get("t"))  # list/np of length 3
        quat = det.get("q", det.get("quat"))      # list/np of length 4
        size = det.get("dim", det.get("size"))     # list/np of length 3

        if translation is None or quat is None or size is None:
            # One or more fields missing – fall back to zeros.
            return torch.zeros(10, device=self.env.device, dtype=torch.float32)

        vec = np.concatenate([translation, quat, size]).astype(np.float32)
        return torch.from_numpy(vec).to(self.env.device)

    # ---- Ground‑truth fallback ----

    def _get_direct_observation(self, env_ids: List[int]) -> torch.Tensor:
        # This fallback retains the original dummy behaviour: a tensor of ones.
        # Replace / extend with your simulator's ground‑truth pose if desired.
        num_envs = len(env_ids)
        return torch.ones(num_envs, 128, device=self.env.device, dtype=torch.float32)
