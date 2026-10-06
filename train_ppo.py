"""
train.py — Entraînement PPO avec Frame Stacking (VecFrameStack) pour obstacles dynamiques.

Pourquoi Frame Stacking + PPO (MLP) ?
  L'observation à l'instant t ne contient pas la vitesse des obstacles.
  En empilant les 4 dernières observations (n_stack=4) via VecFrameStack, le réseau MLP
  obtient un historique temporel immédiat. Il peut ainsi calculer implicitement la vitesse
  et la direction des obstacles mobiles sans la lourdeur d'un LSTM (RecurrentPPO).

Usage :
    pip install stable-baselines3 gymnasium pybullet
    python train.py --timesteps 2000000 --n-envs 8 --n-stack 4
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from nav_env_ppo import RobotNavEnv


# --------------------------------------------------------------------------- #
# Curriculum : progression conjointe (obstacles statiques + dynamiques)
# --------------------------------------------------------------------------- #
class JointCurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, ramp_fraction=0.7, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.ramp_fraction = ramp_fraction
        self._last_state = None

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(1, self.total_timesteps)
        ramp = min(1.0, progress / self.ramp_fraction)
        
        # Progression de la densité statique (ex: 0.3 -> 1.0)
        current_density = 0.3 + ramp * (1.0 - 0.3)
        # Progression du nombre d'obstacles dynamiques (ex: 0 -> 6)
        current_dynamic = round(0 + ramp * 6)

        state = (round(current_density, 2), current_dynamic)
        if state != self._last_state:
            self.training_env.env_method("set_obstacle_density", current_density)
            self.training_env.env_method("set_dynamic_obstacle_count", current_dynamic)
            self._last_state = state
            if self.verbose:
                print(f"[Curriculum] step={self.num_timesteps:,} | Density={current_density:.2f} | Dynamic={current_dynamic}")
        return True


# --------------------------------------------------------------------------- #
def make_env_fn(rank, seed, obstacle_density, n_dynamic_obstacles_start, log_dir):
    def _init():
        env = RobotNavEnv(
            render_mode="none",
            obstacle_density=obstacle_density,
            n_dynamic_obstacles=n_dynamic_obstacles_start,
            seed=seed + rank,
        )
        env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with Frame Stacking on dynamic obstacles")
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-stack", type=int, default=4,
                         help="Nombre de frames empilées pour capturer la vitesse des obstacles")
    parser.add_argument("--n-steps", type=int, default=2048,
                         help="Longueur du rollout par env avant chaque update PPO")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--obstacle-density", type=float, default=1.0)
    parser.add_argument("--start-dynamic-obstacles", type=int, default=0)
    parser.add_argument("--end-dynamic-obstacles", type=int, default=6)
    parser.add_argument("--ramp-fraction", type=float, default=0.7,
                         help="Fraction de l'entraînement sur laquelle la rampe se déroule")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99,
                         help="Doit correspondre au gamma de nav_env.py pour le potential-based reward shaping")
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/ppo_framestack_dynamic")
    parser.add_argument("--log-dir", type=str, default="logs/dynamic_obstacles")
    parser.add_argument("--tensorboard-log", type=str, default="tb_logs/dynamic_obstacles")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                         help="Fréquence de sauvegarde en pas totaux d'environnement")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # 1. Création des environnements vectorisés parallèles
    env_fns = [
        make_env_fn(
            rank=i,
            seed=args.seed,
            obstacle_density=args.obstacle_density,
            n_dynamic_obstacles_start=args.start_dynamic_obstacles,
            log_dir=args.log_dir,
        )
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)

    # 2. Application du Frame Stacking (empilement de n_stack observations)
    vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)

    # 3. Initialisation du modèle PPO standard
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=args.learning_rate,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        tensorboard_log=args.tensorboard_log,
        verbose=1,
        seed=args.seed,
    )

    # 4. Callbacks (Curriculum + Checkpoints)
    curriculum_cb = JointCurriculumCallback(
        total_timesteps=args.timesteps,
        ramp_fraction=args.ramp_fraction,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix="ppo_framestack_ckpt",
    )

    # 5. Lancement de l'entraînement
    print(f"--- Lancement de l'entraînement PPO avec FrameStack (n_stack={args.n_stack}) ---")
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([curriculum_cb, checkpoint_cb]),
    )

    # Sauvegarde finale
    model.save(args.save_path)
    print(f"Modèle final sauvegardé : {args.save_path}.zip")


if __name__ == "__main__":
    main()