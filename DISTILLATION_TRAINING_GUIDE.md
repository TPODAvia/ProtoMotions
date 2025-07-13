# ProtoMotions 4-Step Training Pipeline: Teacher-Student Distillation

This guide explains the complete 4-step training pipeline for creating a lightweight, unified agent that combines path following, goal-directed behavior, steering, and masked mimic capabilities.

## Overview

The 4-step training process creates a deployable, lightweight model through teacher-student distillation:

1. **Step 1**: Train Full Body Tracker (PPO)
2. **Step 2**: Train MaskedMimic with VAE
3. **Step 3**: Train Specialized Teachers (Path Follower, Goal-Directed, Steering)
4. **Step 4**: Train Lightweight Student via Distillation

## Step 1: Full Body Tracker Training

Train the base PPO agent to track motion sequences.

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=full_body_tracker/transformer_flat_terrain \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_tracker \
  +motion_file=data/motions/smpl_humanoid_walk.npy
```

**Expected Output**: `results/smpl_tracker/`

## Step 2: MaskedMimic Training

Train the VAE-based MaskedMimic model using the tracker as expert.

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=masked_mimic/path_follower \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_masked_mimic \
  +motion_file=data/motions/smpl_humanoid_walk.npy \
  +agent.config.expert_model_path=results/smpl_tracker
```

**Expected Output**: `results/smpl_masked_mimic/`

## Step 3: Specialized Teacher Training

Train specialized teachers for each capability.

### 3.1 Path Follower Teacher

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=full_body_tracker/transformer_flat_terrain \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_path_follower \
  +motion_file=data/motions/smpl_humanoid_walk.npy \
  +env.config.path_follower_params.enable_path_termination=true
```

### 3.2 Goal-Directed Teacher

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=full_body_tracker/transformer_flat_terrain \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_goal_directed \
  +motion_file=data/motions/smpl_humanoid_walk.npy \
  +env.config.goal_params.goal_reward_weight=1.0
```

### 3.3 Steering Teacher

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=full_body_tracker/transformer_flat_terrain \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_steering \
  +motion_file=data/motions/smpl_humanoid_walk.npy \
  +env.config.steering_params.obs_size=3
```

## Step 4: Teacher-Student Distillation

Train the lightweight student model using all teachers.

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=distillation/unified_lightweight \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_unified_lightweight \
  +motion_file=data/motions/smpl_humanoid_walk.npy
```

**Expected Output**: `results/smpl_unified_lightweight/`

## Model Architecture Comparison

### Teacher Models (Heavy)
- **Transformer Layers**: 4
- **Latent Dimension**: 512
- **Feedforward Size**: 1024
- **MLP Units**: 256-512
- **Total Parameters**: ~10M+

### Student Model (Lightweight)
- **Transformer Layers**: 2
- **Latent Dimension**: 256
- **Feedforward Size**: 512
- **MLP Units**: 64-256
- **Total Parameters**: ~2M (80% reduction)

## Distillation Loss Components

The student model learns through multiple loss terms:

1. **PPO Loss**: Standard policy optimization
2. **Distillation Loss**: Action imitation from teachers
3. **KL Loss**: Distribution matching
4. **Value Distillation**: Value function learning
5. **Bounds Loss**: Action constraint enforcement

## Evaluation

### Unified Evaluation

```bash
PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +opt=[distillation/tasks/unified_evaluation] \
  checkpoint=results/smpl_unified_lightweight/last.ckpt \
  +motion_file=data/motions/smpl_humanoid_walk.npy
```

### Individual Capability Testing

#### Path Following
```bash
PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +opt=[masked_mimic/tasks/path_following] \
  checkpoint=results/smpl_unified_lightweight/last.ckpt \
  +motion_file=data/motions/smpl_humanoid_walk.npy
```

#### Goal-Directed
```bash
PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +opt=[masked_mimic/tasks/goal_directed] \
  checkpoint=results/smpl_unified_lightweight/last.ckpt \
  +motion_file=data/motions/smpl_humanoid_walk.npy
```

#### Steering
```bash
PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +opt=[masked_mimic/tasks/steering] \
  checkpoint=results/smpl_unified_lightweight/last.ckpt \
  +motion_file=data/motions/smpl_humanoid_walk.npy
```

## Configuration Customization

### Teacher Weights

Adjust teacher influence in `protomotions/config/agent/distillation/agent.yaml`:

```yaml
teacher_models:
  path_follower:
    weight: 1.0    # Increase for better path following
  goal_directed:
    weight: 1.0    # Increase for better goal reaching
  steering:
    weight: 1.0    # Increase for better steering
  masked_mimic:
    weight: 1.0    # Increase for better motion quality
```

### Distillation Weights

```yaml
distillation_weight: 1.0      # Action imitation strength
kl_weight: 0.1                # Distribution matching strength
value_distillation_weight: 0.5 # Value learning strength
```

### Model Size

Adjust student model size in the configuration:

```yaml
latent_dim: 256    # Reduce for smaller model
ff_size: 512       # Reduce for smaller model
num_layers: 2      # Reduce for smaller model
```

## Deployment

The final lightweight model (`results/smpl_unified_lightweight/`) can be deployed with:

- **80% parameter reduction** compared to teachers
- **Unified capabilities** in a single model
- **Real-time inference** suitable for deployment
- **All input modalities** (path, goal, steering, motion)

## Troubleshooting

### Common Issues

1. **Teacher Loading Errors**: Ensure all teacher checkpoints exist
2. **Memory Issues**: Reduce batch size or model size
3. **Training Instability**: Adjust distillation weights
4. **Poor Performance**: Increase teacher weights or training epochs

### Performance Monitoring

Monitor these metrics during training:
- `actor/distillation_loss`: Teacher imitation quality
- `actor/kl_loss`: Distribution matching
- `critic/value_distillation_loss`: Value learning
- `cartesian_err`: Motion tracking accuracy
- `path_following_err`: Path following accuracy
- `goal_reaching_err`: Goal reaching accuracy

## Advanced Features

### Curriculum Learning

Gradually increase task difficulty:
1. Start with simple paths
2. Add complex goals
3. Include steering commands
4. Combine all capabilities

### Multi-Task Training

Train on multiple motion files simultaneously:
```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=distillation/unified_lightweight \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_unified_lightweight_multitask \
  +motion_file=data/motions/smpl_humanoid_walk.npy \
  +motion_file=data/motions/smpl_humanoid_run.npy \
  +motion_file=data/motions/smpl_humanoid_jump.npy
```

### Ensemble Distillation

Use multiple checkpoints from each teacher for more robust distillation. 