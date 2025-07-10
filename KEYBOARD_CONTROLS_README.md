# Keyboard Controls for Steering Environment

This document explains how to use the WASD keyboard controls that have been added to the steering environment in ProtoMotions.

## Overview

The keyboard controls allow you to directly control the robot's target direction and speed during evaluation, making it easy to test and demonstrate the robot's steering capabilities.

## Controls

### Movement Controls
- **W**: Increase target speed (forward)
- **S**: Decrease target speed (backward)
- **A**: Turn left (increase heading angle)
- **D**: Turn right (decrease heading angle)

### System Controls
- **K**: Toggle keyboard control on/off
- **Q**: Quit the application
- **R**: Reset environments
- **J**: Push robot (for testing)

## How to Use

### 1. During Evaluation
When you run the evaluation script, keyboard controls are automatically enabled:

```bash
python protomotions/eval_agent.py +robot=humanoid +simulator=isaacgym checkpoint=/path/to/checkpoint
```

The system will print:
```
Keyboard control enabled for evaluation!
Use WASD to control the robot:
  W/S: Forward/Backward
  A/D: Turn Left/Right
  K: Toggle keyboard control on/off
```

### 2. Test Script
You can also use the provided test script:

```bash
python test_keyboard_controls.py +robot=humanoid +simulator=isaacgym
```

## How It Works

### Architecture
1. **Keyboard Input**: IsaacGym captures keyboard events and updates the simulator's keyboard state
2. **State Propagation**: The base environment passes keyboard state to the steering environment
3. **Target Control**: The steering environment updates the target direction and speed based on keyboard input
4. **Robot Control**: The robot follows the keyboard-controlled targets using its trained policy

### Key Components

#### Steering Environment (`protomotions/envs/steering/env.py`)
- `enable_keyboard_control()`: Enable/disable keyboard control
- `update_keyboard_input()`: Process keyboard input
- `update_keyboard_controls()`: Update target direction and speed
- `keyboard_heading_change_rate`: Controls turning sensitivity
- `keyboard_speed_change_rate`: Controls speed change sensitivity

#### Simulator (`protomotions/simulator/isaacgym/simulator.py`)
- Keyboard event subscriptions for WASD and K keys
- Keyboard state tracking and propagation

#### Base Environment (`protomotions/envs/base_env/env.py`)
- Keyboard state propagation from simulator to environment

## Configuration

### Sensitivity Tuning
You can adjust the keyboard control sensitivity by modifying these parameters in the steering environment:

```python
self.keyboard_heading_change_rate = 2.0  # Radians per second
self.keyboard_speed_change_rate = 2.0    # Speed units per second
```

### Speed Limits
The keyboard-controlled speed is automatically clamped to the environment's speed limits:
- Minimum: `self._tar_speed_min`
- Maximum: `self._tar_speed_max`

## Troubleshooting

### Keyboard Not Responding
1. Make sure you're running in non-headless mode (viewer enabled)
2. Check that the window has focus
3. Verify that keyboard control is enabled (press K to toggle)

### Robot Not Following Commands
1. Ensure the robot has been trained for steering tasks
2. Check that the checkpoint file is valid
3. Verify that the environment is properly configured

### Performance Issues
1. Reduce the number of environments if running slowly
2. Close other applications to free up GPU memory
3. Consider running in headless mode for faster evaluation

## Example Usage

1. **Start evaluation**:
   ```bash
   python protomotions/eval_agent.py +robot=humanoid +simulator=isaacgym checkpoint=path/to/model.ckpt
   ```

2. **Control the robot**:
   - Press **W** to make the robot move forward faster
   - Press **A** to make the robot turn left
   - Press **D** to make the robot turn right
   - Press **S** to slow down or reverse

3. **Toggle control**:
   - Press **K** to enable/disable keyboard control
   - When disabled, the robot follows automatic targets

4. **Quit**:
   - Press **Q** to exit the application

## Technical Details

### Keyboard State Management
The keyboard state is tracked in the simulator and propagated through the environment hierarchy:
```
IsaacGym Simulator → Base Environment → Steering Environment
```

### Target Override
When keyboard control is enabled, the steering environment overrides the automatic target generation and uses keyboard input instead.

### Integration with Policy
The keyboard controls modify the target direction and speed, but the robot still uses its trained policy to achieve these targets. This means the robot will follow the keyboard commands using its learned locomotion skills. 