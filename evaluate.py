"""
evaluate.py — Visualisation GUI, contrôleur aléatoire ou modèle RecurrentPPO chargé.
"""

import argparse
import io
import os
import time
import zipfile

import numpy as np
import torch

from nav_env_ppo import RobotNavEnv


def resolve_model_path(model_path):
    """Ajoute l'extension .zip si elle n'est pas spécifiée dans le chemin."""
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path


def load_recurrent_ppo_model(model_path):
    """
    Charge le modèle RecurrentPPO en gérant le bogue de lecture PyTorchStreamReader
    via la lecture en mémoire (BytesIO).
    """
    from sb3_contrib import RecurrentPPO

    model_path = resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle est introuvable : {model_path}")

    try:
        return RecurrentPPO.load(model_path)
    except RuntimeError as error:
        if "PytorchStreamReader failed reading file" not in str(error):
            raise

        print("[INFO] Chargement direct impossible avec cette version de PyTorch.")
        print("[INFO] Nouvelle tentative en lisant les poids depuis la mémoire...")

        original_torch_load = torch.load

        def torch_load_from_memory(file_obj, *args, **kwargs):
            if isinstance(file_obj, zipfile.ZipExtFile):
                return original_torch_load(io.BytesIO(file_obj.read()), *args, **kwargs)
            return original_torch_load(file_obj, *args, **kwargs)

        torch.load = torch_load_from_memory
        try:
            return RecurrentPPO.load(model_path)
        finally:
            torch.load = original_torch_load


def run_random(env, episodes):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            time.sleep(1.0 / env.control_hz)
        outcome = "collision" if info.get("collided") else (
            "succès" if info["dist_to_goal"] < env.goal_threshold else "timeout"
        )
        print(f"[random] épisode {ep + 1}/{episodes} — reward={ep_reward:.1f} — {outcome}")


def run_model(env, episodes, model_path, deterministic=True):
    # Utilisation de la nouvelle fonction robuste de chargement
    model = load_recurrent_ppo_model(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        ep_reward = 0.0
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
            )
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            episode_start = np.array([done], dtype=bool)
            time.sleep(1.0 / env.control_hz)
        outcome = "collision" if info.get("collided") else (
            "succès" if info["dist_to_goal"] < env.goal_threshold else "timeout"
        )
        print(f"[model] épisode {ep + 1}/{episodes} — reward={ep_reward:.1f} — {outcome}")


def main():
    parser = argparse.ArgumentParser(description="Visualiser l'environnement en GUI PyBullet")
    parser.add_argument("--mode", choices=["random", "model"], default="model")
    parser.add_argument("--model-path", type=str, default=None,
                         help="Requis si --mode model (fichier .zip RecurrentPPO)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obstacle-density", type=float, default=1)
    parser.add_argument("--dynamic-obstacles", type=int, default=6)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    if args.mode == "model" and not args.model_path:
        parser.error("--model-path est requis quand --mode model")

    env = RobotNavEnv(
        render_mode="human",
        obstacle_density=args.obstacle_density,
        n_dynamic_obstacles=args.dynamic_obstacles,
    )

    try:
        if args.mode == "random":
            run_random(env, args.episodes)
        else:
            run_model(env, args.episodes, args.model_path, args.deterministic)
    finally:
        env.close()


if __name__ == "__main__":
    main()