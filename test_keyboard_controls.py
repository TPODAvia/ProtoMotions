#!/usr/bin/env python3
"""
Test script for keyboard controls in the steering environment.
This script demonstrates how to use the WASD keyboard controls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Import isaacgym before torch modules
import isaacgym  # noqa: F401

from protomotions.agents.ppo.agent import PPO


@hydra.main(config_path="protomotions/config")
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())
    
    # Use a simple steering configuration
    config = override_config
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = instantiate(config.env, device=device)
    
    # Create a dummy agent (we won't actually use it for control)
    fabric = instantiate(config.fabric)
    fabric.launch()
    agent = instantiate(config.agent, env=env, fabric=fabric)
    
    # Enable keyboard control
    if hasattr(env, 'enable_keyboard_control'):
        env.enable_keyboard_control(True)
        print("Keyboard control enabled!")
        print("Use WASD to control the robot:")
        print("  W/S: Forward/Backward")
        print("  A/D: Turn Left/Right")
        print("  K: Toggle keyboard control on/off")
        print("  Q: Quit")
    
    # Simple evaluation loop
    obs = env.reset()
    step = 0
    max_steps = 1000
    
    while step < max_steps:
        # Use zero actions (keyboard will control the target)
        actions = torch.zeros((env.num_envs, env.get_action_size()), device=device)
        
        # Step environment
        obs, rewards, dones, extras = env.step(actions)
        
        # Check if any environment is done
        if dones.any():
            env.reset(dones.nonzero(as_tuple=False).squeeze(-1))
        
        step += 1
        
        # Print current keyboard state every 100 steps
        if step % 100 == 0 and hasattr(env, 'keyboard_input'):
            print(f"Step {step}: Keyboard state: {env.keyboard_input}")
            if hasattr(env, 'keyboard_control_enabled'):
                print(f"Keyboard control enabled: {env.keyboard_control_enabled}")


if __name__ == "__main__":
    main() 