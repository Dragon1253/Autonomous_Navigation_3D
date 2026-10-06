"""
evaluate_ppo.py — Évaluation GUI pour PPO avec VecFrameStack et dimensions forcées.

Usage :
    python evaluate_ppo.py --model ./models/ppo_framestack_dynamic --episodes 20 --obstacle-density 1.0 --n-dynamic-obstacles 4
"""

import argparse
import io
import os
import time
import zipfile

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from nav_env import RobotNavEnv


def resolve_model_path(model_path):
    """Vérifie l'existence du fichier modèle avec ou sans l'extension .zip."""
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path


def load_model(model_path, env=None):
    """
    Charge le modèle PPO de manière robuste depuis le disque ou la mémoire.
    """
    try:
        return PPO.load(model_path, env=env)
    except RuntimeError as error:
        if "PytorchStreamReader failed reading file" not in str(error):
            raise

        print("Chargement direct impossible avec cette version de PyTorch.")
        print("Nouvelle tentative en lisant les poids depuis la mémoire...")

        original_torch_load = torch.load

        def torch_load_from_memory(file_obj, *args, **kwargs):
            if isinstance(file_obj, zipfile.ZipExtFile):
                return original_torch_load(io.BytesIO(file_obj.read()), *args, **kwargs)
            return original_torch_load(file_obj, *args, **kwargs)

        torch.load = torch_load_from_memory
        try:
            return PPO.load(model_path, env=env)
        finally:
            torch.load = original_torch_load


def make_env_fn(obstacle_density, n_dynamic_obstacles):
    """Génère l'environnement en forçant explicitement les dimensions du LiDAR."""
    def _init():
        return RobotNavEnv(
            render_mode="human",
            obstacle_density=obstacle_density,
            n_dynamic_obstacles=n_dynamic_obstacles,
            min_goal_dist=3.5,
            max_episode_steps=600,
            lidar_n_rays=36,  # <-- CORRECTION : Force 36 rayons pour obtenir une base de 43 et un total de 172 avec le stack
        )
    return _init


def main():
    parser = argparse.ArgumentParser(description="Évaluation GUI PPO + FrameStack")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/ppo_framestack_dynamic",
        help="Chemin vers le modèle .zip entraîné.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obstacle-density", type=float, default=1.0)
    parser.add_argument("--n-dynamic-obstacles", type=int, default=4)
    parser.add_argument("--n-stack", type=int, default=4, help="Nombre de frames empilées")
    parser.add_argument("--delay", type=float, default=0.02, help="Délai d'affichage (s)")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fichier modèle introuvable : '{model_path}'")

    # 1. Création de l'environnement vectorisé avec FrameStack
    vec_env = DummyVecEnv([make_env_fn(args.obstacle_density, args.n_dynamic_obstacles)])
    vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)

    # 2. Chargement robuste du modèle
    print(f"--- Chargement du modèle depuis : {model_path} ---")
    model = load_model(model_path, env=vec_env)

    successes, collisions, timeouts = 0, 0, 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, infos = vec_env.step(action)

            done = done_vec[0]
            ep_reward += reward[0]
            step += 1

            if args.delay > 0:
                time.sleep(args.delay)

            if done:
                info = infos[0]
                if info.get("collided", False):
                    collisions += 1
                    outcome = "COLLISION"
                elif info.get("dist_to_goal", 999) < 0.25:
                    successes += 1
                    outcome = "SUCCÈS"
                else:
                    timeouts += 1
                    outcome = "TIMEOUT"

                print(f"Épisode {ep + 1:>2}/{args.episodes} - {outcome:<12} - Pas={step} - Reward={ep_reward:.1f}")

    print("\n" + "=" * 40)
    print(f"Bilan Global ({args.episodes} épisodes) :")
    print(f"  - Succès     : {successes}/{args.episodes} ({successes/args.episodes*100:.1f}%)")
    print(f"  - Collisions : {collisions}/{args.episodes} ({collisions/args.episodes*100:.1f}%)")
    print(f"  - Timeouts   : {timeouts}/{args.episodes} ({timeouts/args.episodes*100:.1f}%)")
    print("=" * 40)

    vec_env.close()


if __name__ == "__main__":
    main()