import torch
import time
from torch import Tensor
from typing import Dict, Tuple
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from rich.progress import track
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.masked_mimic.agent import MaskedMimic
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.masked_mimic.model import VaeDeterministicOutputModel
from protomotions.agents.common.weight_init import weight_init
from protomotions.agents.ppo.utils import discount_values, bounds_loss
from protomotions.agents.utils.data_utils import ExperienceBuffer


def deep_clone(x):
    if isinstance(x, dict):
        return {k: deep_clone(v) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.clone().contiguous()
    else:
        return x


class DistillationAgent(PPO):
    """
    Teacher-Student Distillation Agent for creating lightweight models.
    
    This agent learns from multiple teacher models:
    1. Path Follower Teacher
    2. Goal-Directed Teacher  
    3. Steering Teacher
    4. MaskedMimic Teacher
    
    The student model is a lightweight version that combines all capabilities.
    """
    
    def __init__(self, fabric, env, config):
        super().__init__(fabric, env, config)
        self.teacher_models = {}
        self.teacher_weights = {}
        
    def setup(self):
        """Setup student model and load teacher models."""
        # Setup student model with completely fresh instantiation
        student_model = instantiate(self.config.model)
        student_model.apply(weight_init)
        
        # Ensure model is properly initialized for distributed training
        student_model.train()
        
        optimizer = instantiate(
            self.config.model.config.optimizer,
            params=list(student_model.parameters()),
        )
        
        self.model, self.optimizer = self.fabric.setup(student_model, optimizer)
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")
        
        # Load teacher models
        self._load_teacher_models()
        
    def _load_teacher_models(self):
        """Load all teacher models from their checkpoints."""
        teacher_configs = self.config.teacher_models
        
        for teacher_name, teacher_config in teacher_configs.items():
            print(f"Loading teacher model: {teacher_name}")
            
            # Load teacher model configuration
            teacher_model_config = OmegaConf.load(
                Path(teacher_config.checkpoint_path) / "config.yaml"
            )
            
            # Instantiate teacher model
            if teacher_config.model_type == "ppo":
                teacher_model: PPOModel = instantiate(
                    teacher_model_config.agent.config.model
                )
            elif teacher_config.model_type == "masked_mimic":
                teacher_model: VaeDeterministicOutputModel = instantiate(
                    teacher_model_config.agent.config.model
                )
            else:
                raise ValueError(f"Unknown teacher model type: {teacher_config.model_type}")
            
            # Setup teacher model
            teacher_model = self.fabric.setup(teacher_model)
            
            # Load checkpoint
            checkpoint_path = teacher_config.checkpoint_path + "/last.ckpt"
            if not Path(checkpoint_path).exists():
                checkpoint_path = teacher_config.checkpoint_path + "/score_based.ckpt"
            
            checkpoint = torch.load(checkpoint_path, map_location=self.fabric.device)
            teacher_model.load_state_dict(checkpoint["model"])
            
            # Freeze teacher parameters
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            
            # Store teacher model and weight
            self.teacher_models[teacher_name] = teacher_model
            self.teacher_weights[teacher_name] = teacher_config.weight
            
            print(f"Loaded teacher {teacher_name} with weight {teacher_config.weight}")
    
    def register_extra_experience_buffer_keys(self):
        """Register additional keys for distillation training."""
        self.experience_buffer.register_key(
            "teacher_actions", shape=(self.env.config.robot.number_of_actions,), dtype=torch.float32
        )
        self.experience_buffer.register_key(
            "teacher_values", shape=(1,), dtype=torch.float32
        )
        self.experience_buffer.register_key(
            "teacher_logprobs", shape=(), dtype=torch.float32
        )
        
    def fit(self):
        """Main training loop with distillation."""
        # Setup experience buffer like parent class
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                if key not in obs:
                    print(f"Warning: Key {key} not found in obs returned from env: {obs.keys()}")
                    continue
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Force reset on fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)
        
        # Training loop
        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()
            
            # Set networks in eval mode
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)
                
                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)
                    
                    # Store student observations
                    self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            if key in obs:
                                self.experience_buffer.update_data(key, step, obs[key])
                    
                    # Get student actions
                    obs_cloned = deep_clone(obs)
                    student_action, student_neglogp, student_value = self.model.get_action_and_value(obs_cloned)
                    self.experience_buffer.update_data("actions", step, student_action)
                    self.experience_buffer.update_data("neglogp", step, student_neglogp)
                    self.experience_buffer.update_data("values", step, student_value)
                    
                    # Get teacher actions (ensemble)
                    teacher_actions = []
                    teacher_values = []
                    teacher_logprobs = []
                    
                    for teacher_name, teacher_model in self.teacher_models.items():
                        if hasattr(teacher_model, 'get_action_and_value'):
                            # PPO teacher
                            teacher_action, teacher_neglogp, teacher_value = teacher_model.get_action_and_value(obs_cloned)
                            teacher_actions.append(teacher_action)
                            teacher_values.append(teacher_value)
                            teacher_logprobs.append(teacher_neglogp)
                        else:
                            # MaskedMimic teacher
                            teacher_action = teacher_model.act(obs_cloned)
                            teacher_actions.append(teacher_action)
                            # Use student values for MaskedMimic teachers
                            teacher_values.append(student_value)
                            teacher_logprobs.append(student_neglogp)
                    
                    # Ensemble teacher outputs (weighted average)
                    teacher_action = torch.stack(teacher_actions, dim=0)
                    teacher_value = torch.stack(teacher_values, dim=0)
                    teacher_logprob = torch.stack(teacher_logprobs, dim=0)
                    
                    # Weight by teacher weights
                    weights = torch.tensor([self.teacher_weights[name] for name in self.teacher_models.keys()], 
                                         device=self.device).unsqueeze(1).unsqueeze(2)
                    weights = weights / weights.sum()
                    
                    ensemble_teacher_action = (teacher_action * weights).sum(dim=0)
                    ensemble_teacher_value = (teacher_value * weights.unsqueeze(-1)).sum(dim=0)
                    ensemble_teacher_logprob = (teacher_logprob * weights).sum(dim=0)
                    
                    self.experience_buffer.update_data("teacher_actions", step, ensemble_teacher_action)
                    self.experience_buffer.update_data("teacher_values", step, ensemble_teacher_value)
                    self.experience_buffer.update_data("teacher_logprobs", step, ensemble_teacher_logprob)
                    
                    # Step environment with student action
                    next_obs, rewards, dones, terminated, extras = self.env_step(student_action)
                    
                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)
                    
                    # Update logging metrics
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)
                    
                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)
                    
                    next_value = self.model._critic(next_obs).flatten()
                    if self.config.normalize_values:
                        next_value = self.running_val_norm.normalize(next_value, un_norm=True)
                    next_value = next_value * (1 - terminated.float())
                    self.experience_buffer.update_data("next_values", step, next_value)
                    
                    self.step_count += self.get_step_count_increment()
                
                # Calculate advantages and returns
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)
                
                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)
                
                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)
            
            # Optimize model
            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)
            
            # Save model
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()
            
            # Evaluation
            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)
            
            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)
            
            if self.should_stop:
                self.save()
                return
        
        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)
    
    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Actor step with distillation loss."""
        dist = self.model._actor(batch_dict)
        logstd = self.model._actor.logstd
        std = torch.exp(logstd)
        neglogp = self.model.neglogp(batch_dict["actions"], dist.mean, std, logstd)
        
        # PPO loss
        ratio = torch.exp(batch_dict["neglogp"] - neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()
        
        # Bounds loss
        if self.config.bounds_loss_coef > 0:
            b_loss: Tensor = bounds_loss(dist.mean) * self.config.bounds_loss_coef
        else:
            b_loss = torch.zeros(self.num_envs, device=self.device)
        
        # Distillation loss (behavior cloning from teacher)
        teacher_actions = batch_dict["teacher_actions"]
        distillation_loss = torch.square(dist.mean - teacher_actions).mean()
        
        # KL divergence loss for action distribution
        teacher_logprobs = batch_dict["teacher_logprobs"]
        kl_loss = torch.square(neglogp - teacher_logprobs).mean()
        
        actor_ppo_loss = ppo_loss.mean()
        b_loss = b_loss.mean()
        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(batch_dict, dist)
        
        # Combine losses with distillation weights
        distillation_weight = self.config.distillation_weight
        kl_weight = self.config.kl_weight
        
        actor_loss = (actor_ppo_loss + b_loss + extra_loss + 
                     distillation_loss * distillation_weight + 
                     kl_loss * kl_weight)
        
        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "actor/distillation_loss": distillation_loss.detach(),
            "actor/kl_loss": kl_loss.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        log_dict.update(extra_actor_log_dict)
        
        return actor_loss, log_dict
    
    def critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Critic step with teacher value distillation."""
        student_values = self.model._critic(batch_dict).flatten()
        teacher_values = batch_dict["teacher_values"]
        
        # Value distillation loss
        value_distillation_loss = torch.square(student_values - teacher_values).mean()
        
        # Standard critic loss
        returns = batch_dict["returns"]
        if self.config.normalize_values:
            student_values = self.running_val_norm.normalize(student_values, un_norm=True)
        
        if self.config.clip_critic_loss:
            value_pred_clipped = batch_dict["values"] + torch.clamp(
                student_values - batch_dict["values"],
                -self.e_clip,
                self.e_clip,
            )
            value_losses = (student_values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = (student_values - returns) ** 2
        
        critic_loss = value_loss.mean()
        
        # Combine losses
        value_distillation_weight = self.config.value_distillation_weight
        total_critic_loss = critic_loss + value_distillation_loss * value_distillation_weight
        
        log_dict = {
            "critic/value_loss": critic_loss.detach(),
            "critic/value_distillation_loss": value_distillation_loss.detach(),
            "losses/critic_loss": total_critic_loss.detach(),
        }
        
        return total_critic_loss, log_dict 