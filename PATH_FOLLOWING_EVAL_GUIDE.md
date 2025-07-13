# Path Following Evaluation Guide for MaskedMimic

This guide explains how to run evaluation with path following capabilities after training your MaskedMimic model.

## Prerequisites

1. **Trained Full Body Tracker**: You need a trained full body tracker checkpoint
2. **Trained MaskedMimic Model**: You need a trained MaskedMimic model with path following capabilities
3. **Motion File**: The motion data used during training

## Training Commands (Recap)

### Step 1: Train Full Body Tracker
```bash
PYTHON_PATH protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain +robot=smpl +simulator=isaaclab +experiment_name=smpl_tracker +motion_file=data/motions/smpl_humanoid_walk.npy
```

### Step 2: Train MaskedMimic with Path Following
```bash
PYTHON_PATH protomotions/train_agent.py +exp=masked_mimic/path_follower +robot=smpl +simulator=isaaclab +experiment_name=smpl_path_mimic +motion_file=data/motions/smpl_humanoid_walk.npy +agent.config.expert_model_path=results/smpl_tracker
```

## Evaluation Commands

### 1. Basic Path Following Evaluation

```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/path_following] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy
```

### 2. Goal-Directed Evaluation

```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/goal_directed] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy
```

### 3. Steering with Keyboard Controls

```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/steering] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy
```

## Evaluation Options

### Path Following Evaluation
- **Environment**: Uses path following environment with trajectory samples
- **Visualization**: Shows path markers and robot following the path
- **Controls**: Automatic path generation and following
- **Metrics**: Path following accuracy, distance to path, etc.

### Goal-Directed Evaluation
- **Environment**: Uses goal-directed environment with target positions
- **Visualization**: Shows goal markers and robot moving toward goals
- **Controls**: Automatic goal generation and navigation
- **Metrics**: Goal reaching success rate, path efficiency, etc.

### Steering Evaluation
- **Environment**: Uses steering environment with keyboard controls
- **Visualization**: Shows direction markers and robot following steering commands
- **Controls**: WASD keyboard controls for direction and speed
- **Metrics**: Steering accuracy, speed control, etc.

## Keyboard Controls (for Steering)

When running steering evaluation, you can use these keyboard controls:

- **W**: Increase target speed (forward)
- **S**: Decrease target speed (backward)
- **A**: Turn left (increase heading angle)
- **D**: Turn right (decrease heading angle)
- **K**: Toggle keyboard control on/off
- **R**: Reset environments
- **Q**: Quit the application

## Evaluation Parameters

### Common Parameters
- `headless: false` - Enable visualization
- `num_envs: 1` - Single environment for evaluation
- `max_eval_steps: 1000` - Maximum evaluation steps
- `vae.noise_type: "zeros"` - Deterministic VAE output

### Path Following Parameters
- `num_traj_samples: 10` - Number of trajectory samples
- `path_obs_size: 30` - Path observation size (3 per sample)
- `height_conditioned: true` - Include height information
- `enable_path_termination: false` - Don't terminate on path failure

### Goal-Directed Parameters
- `goal_distance_threshold: 2.0` - Success threshold for reaching goals
- `max_goal_distance: 20.0` - Maximum goal distance
- `goal_reward_weight: 1.0` - Weight for goal-reaching reward

### Steering Parameters
- `tar_speed_min: 1.2` - Minimum target speed
- `tar_speed_max: 6.0` - Maximum target speed
- `standard_heading_change: 1.57` - Standard heading change (90 degrees)

## Customizing Evaluation

### 1. Modify Path Parameters
```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/path_following] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy +env.config.path_follower_params.num_traj_samples=15 +env.config.path_follower_params.fail_dist=3.0
```

### 2. Modify Goal Parameters
```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/goal_directed] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy +env.config.goal_params.goal_distance_threshold=1.5 +env.config.goal_params.max_goal_distance=15.0
```

### 3. Modify Steering Parameters
```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/steering] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy +env.config.steering_params.tar_speed_min=0.8 +env.config.steering_params.tar_speed_max=8.0
```

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
   - Ensure the checkpoint path is correct
   - Check that training completed successfully

2. **Model Loading Errors**
   - Verify the model architecture matches the checkpoint
   - Check for missing terrain or path modules

3. **Environment Errors**
   - Ensure all required modules are included in the configuration
   - Check that motion file exists and is valid

4. **Performance Issues**
   - Reduce `num_envs` if running slowly
   - Close other applications to free GPU memory
   - Consider running in headless mode for faster evaluation

### Debug Mode

For debugging, you can run with additional logging:

```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/path_following] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy +headless=false +num_envs=1 +max_eval_steps=100
```

## Advanced Usage

### 1. Batch Evaluation
For evaluating multiple models:

```bash
# Create a script to evaluate multiple checkpoints
for checkpoint in results/smpl_path_mimic_*/last.ckpt; do
    echo "Evaluating $checkpoint"
    PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/path_following] checkpoint=$checkpoint +motion_file=data/motions/smpl_humanoid_walk.npy
done
```

### 2. Custom Path Generation
You can modify the path generation parameters:

```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/path_following] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy +env.config.path_follower_params.path_generator.sharp_turn_prob=0.1 +env.config.path_follower_params.path_generator.speed_max=3.0
```

### 3. Recording Evaluation
To record the evaluation:

```bash
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +opt=[masked_mimic/tasks/path_following] checkpoint=results/smpl_path_mimic/last.ckpt +motion_file=data/motions/smpl_humanoid_walk.npy +simulator.config.record_viewer=true +simulator.config.viewer_record_dir=output/recordings/path_following
```

## Expected Results

When running path following evaluation, you should see:

1. **Robot following a generated path** with red markers showing the trajectory
2. **Smooth motion** that maintains the quality of the original motion data
3. **Path following accuracy** displayed in the console
4. **Visual feedback** showing the robot's progress along the path

The robot should demonstrate:
- Natural walking motion from the training data
- Ability to follow curved paths
- Speed control based on path curvature
- Robust motion even on challenging paths 