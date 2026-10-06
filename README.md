# 3D Robot Obstacle Avoidance with PyBullet

This project implements 3D obstacle avoidance for a mobile robot in the
[PyBullet](https://pybullet.org/) simulator. The environment contains both
static obstacles and moving obstacles, and is designed for training and
evaluating reinforcement-learning navigation policies.

The project includes several model variants:

- Recurrent PPO
- PPO with frame stacking
- PPO with attention
- SAC with attention

The main SAC experiment is implemented in
[`sac_train_v5_1.py`](./sac_train_v5_1.py). It uses a curriculum of maps with
increasing difficulty and supports both headless evaluation and graphical
playback.

## Features

- 3D robot navigation from a start position to a goal.
- Static and dynamic obstacles.
- PyBullet physics and visual simulation.
- LiDAR-based obstacle observations.
- Curriculum learning across multiple map difficulty levels.
- SAC policy with configurable obstacle-feature pooling.
- Training logs for TensorBoard.
- Best-model checkpoints saved during curriculum training.

## Requirements

Use Python 3.10 or newer and install the dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are PyBullet, Gymnasium, NumPy, PyTorch,
Stable-Baselines3, and SB3-Contrib.

## Main script

Run all commands from the project directory. The default command starts a new
SAC training run:

```bash
python sac_train_v5_1.py
```

Training uses eight parallel environments and runs for 4,500,000 timesteps.
The trained model is saved as `sac_nav_v5.2`, while intermediate and
curriculum-specific checkpoints are written to `best_models/`.

### Graphical playback

To visualize the trained policy in the PyBullet GUI:

```bash
python sac_train_v5_1.py --play
```

Playback automatically starts at the highest curriculum level and runs 20
episodes. The GUI displays the robot, obstacles, LiDAR beams, trajectory
history, camera view, and navigation information.

The playback policy is loaded from `sac_nav_v5.1`. Make sure that the
corresponding model file is available in the project directory before running
this command.

### Evaluation without the GUI

To evaluate the policy without opening the graphical interface:

```bash
python sac_train_v5_1.py --eval
```

By default, evaluation runs 100 episodes at the highest curriculum level and
prints:

- Success rate
- Collision rate
- Timeout rate
- Average episode length in steps

To select the number of evaluation episodes:

```bash
python sac_train_v5_1.py --eval --episodes 50
```

To evaluate a specific curriculum level:

```bash
python sac_train_v5_1.py --eval --level 5
```

If `--level` is omitted, the highest available curriculum level is selected.
The `--episodes` argument controls the number of evaluation episodes; it is
not a number of steps per episode. The maximum number of steps for an episode
is defined by the environment.

### Attention pooling

The SAC feature extractor supports two pooling modes:

```bash
python sac_train_v5_1.py --eval --pooling meanmax
python sac_train_v5_1.py --eval --pooling attention
```

`meanmax` is the default. Use `attention` when evaluating or training the
attention-based SAC variant. The same option can be used with `--play`:

```bash
python sac_train_v5_1.py --play --pooling attention
```

## Other training scripts

The repository also contains scripts and checkpoints for the other model
families:

- `train_recurrent_ppo.py` and `recurrent_ppo_nav_final.zip`:
  recurrent PPO with an LSTM policy.
- `train_ppo_framestack.py` and `ppo_framestack_nav_final.zip`:
  PPO using stacked observations.
- `ppo_train_v5_deepsets.py`, `train_ppo.py`, and related PPO checkpoints:
  feed-forward PPO variants, including obstacle-set feature processing.
- `sac_train_v5_1.py` and `sac_nav_v5.1` / `sac_nav_v5.2`:
  SAC with obstacle-feature pooling, including attention pooling.

The exact command-line options differ between these experimental scripts.
For a given script, inspect its `--help` output or its argument parser before
launching a run.

## Important files and directories

| Path | Description |
| --- | --- |
| `sac_train_v5_1.py` | Main SAC training, playback, and evaluation script |
| `requirements.txt` | Python dependencies |
| `robot.urdf` | Robot model used by PyBullet |
| `assets/` | Textures and visual assets |
| `best_models/` | Best checkpoints saved during curriculum learning |
| `tb_nav/` | TensorBoard logs for SAC experiments |
| `tb_nav_ppo/` | TensorBoard logs for PPO experiments |
| `tb_nav_recurrent_ppo/` | TensorBoard logs for recurrent PPO experiments |

## TensorBoard

To inspect training curves, launch TensorBoard from the project directory:

```bash
tensorboard --logdir tb_nav
```

Then open the local URL printed by TensorBoard in a browser.

## Notes

- A trained checkpoint must exist before using `--play` or `--eval`.
- Run evaluation with the same feature-extractor configuration used during
  training. For example, an attention-trained model should be evaluated with
  `--pooling attention`.
- Training can require substantial disk space and compute time because it uses
  parallel environments, a large replay buffer, and millions of timesteps.
- The project is intended for experimentation with reinforcement-learning
  navigation in simulated 3D environments; results depend on the selected
  checkpoint, random seeds, and curriculum level.
