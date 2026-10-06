"""Evaluation diagnostique du checkpoint SAC v4.

Exemples
--------
python analyze_sac_v4.py --episodes 100 --level 15
python analyze_sac_v4.py --episodes 20 --plot-episodes 6

Les seeds sont fixes par defaut, ce qui permet de comparer deux checkpoints ou
deux reglages d'environnement sur exactement les memes cartes.
"""

import argparse
import csv
import io
import json
import math
import os
import zipfile

import numpy as np
import torch
from stable_baselines3 import SAC

from sac_train_v4 import CurriculumManager, NavEnv


def load_sac_from_memory(model_path: str):
    """Evite le bug PyTorchStreamReader observe avec les .zip SB3 locaux."""
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        model_path = f"{model_path}.zip"
    original_load = torch.load

    def safe_load(file_obj, *args, **kwargs):
        if isinstance(file_obj, zipfile.ZipExtFile):
            return original_load(io.BytesIO(file_obj.read()), *args, **kwargs)
        return original_load(file_obj, *args, **kwargs)

    torch.load = safe_load
    try:
        return SAC.load(model_path)
    finally:
        torch.load = original_load


def clearance(env):
    """Clearance bord-a-bord minimale, sans dependance au LiDAR discret."""
    distances = [
        float(np.linalg.norm(env.pos - obstacle["pos"]) - env.R_ROBOT - obstacle["r"])
        for obstacle in env.obstacles
    ]
    return min(distances, default=float("inf"))


def run_episode(env, model, seed):
    obs, _ = env.reset(seed=seed)
    start = env.pos.copy()
    goal = env.goal.copy()
    path = [start.copy()]
    initial_obstacles = [
        (obstacle["pos"].copy(), float(obstacle["r"]), bool(obstacle["dynamic"]))
        for obstacle in env.obstacles
    ]
    dynamic_paths = {
        obstacle["id"]: [obstacle["pos"].copy()]
        for obstacle in env.obstacles if obstacle["dynamic"]
    }
    actions, rewards, clearances, dangers, ttc_penalties = [], [], [], [], []
    reverses = turns = 0
    done, info = False, {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action[0] < -0.05:
            reverses += 1
        if abs(action[1]) > 0.25:
            turns += 1

        obs, reward, terminated, truncated, info = env.step(action)
        # Le calcul ici est identique a celui de l'environnement et sert au log,
        # il n'influence jamais l'episode.
        v = action[0] * env.V_MAX if action[0] >= 0 else action[0] * env.V_MAX_REV
        robot_v = v * np.array([math.cos(env.theta), math.sin(env.theta)])
        ttc_penalty, danger = env._ttc_features(robot_v)
        actions.append(action)
        rewards.append(float(reward))
        clearances.append(clearance(env))
        dangers.append(danger)
        ttc_penalties.append(ttc_penalty)
        path.append(env.pos.copy())
        for obstacle in env.obstacles:
            if obstacle["dynamic"]:
                dynamic_paths[obstacle["id"]].append(obstacle["pos"].copy())
        done = terminated or truncated

    path = np.asarray(path)
    actions = np.asarray(actions)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    direct_distance = float(np.linalg.norm(goal - start))
    path_length = float(segment_lengths.sum())
    action_delta = np.linalg.norm(np.diff(actions, axis=0), axis=1) if len(actions) > 1 else np.array([0.0])
    if info.get("success"):
        outcome = "success"
    elif env.steps >= env.MAX_STEPS:
        outcome = "timeout"
    else:
        outcome = "collision"

    summary = {
        "seed": seed,
        "outcome": outcome,
        "steps": int(env.steps),
        "return": round(float(sum(rewards)), 3),
        "start_goal_distance_m": round(direct_distance, 3),
        "final_goal_distance_m": round(float(np.linalg.norm(env.goal - env.pos)), 3),
        "path_length_m": round(path_length, 3),
        "path_efficiency": round(direct_distance / path_length, 3) if path_length > 1e-6 else 0.0,
        "min_clearance_m": round(float(min(clearances)), 3),
        "mean_clearance_m": round(float(np.mean(clearances)), 3),
        "max_danger": round(float(max(dangers)), 3),
        "mean_ttc_penalty": round(float(np.mean(ttc_penalties)), 3),
        "mean_abs_turn": round(float(np.mean(np.abs(actions[:, 1]))), 3),
        "mean_action_change": round(float(np.mean(action_delta)), 3),
        "reverse_fraction": round(reverses / len(actions), 3),
        "turn_fraction": round(turns / len(actions), 3),
    }
    return summary, path, start, goal, initial_obstacles, dynamic_paths


def save_plot(episodes, output_path):
    # L'evaluation tourne souvent sans ecran (ou sans Tcl/Tk installe) : ne jamais
    # demander un backend GUI pour un simple fichier PNG.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3
    rows = math.ceil(len(episodes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for axis, (summary, path, start, goal, initial_obstacles, dynamic_paths) in zip(axes.flat, episodes):
        for position, radius, dynamic in initial_obstacles:
            axis.add_patch(plt.Circle(position, radius, color="#d32f2f" if dynamic else "#6d4c41",
                                      alpha=0.28, linewidth=0))
        for positions in dynamic_paths.values():
            positions = np.asarray(positions)
            axis.plot(positions[:, 0], positions[:, 1], color="#d32f2f", alpha=0.16, linewidth=0.5)
        axis.plot(path[:, 0], path[:, 1], color="#1769aa", linewidth=1.5, label="robot")
        axis.scatter(*start, c="#1565c0", s=35, label="depart")
        axis.scatter(*goal, c="#2e7d32", marker="*", s=120, label="but")
        axis.set(xlim=(-6, 6), ylim=(-6, 6), aspect="equal",
                 title=f"seed {summary['seed']} | {summary['outcome']} | {summary['steps']} pas")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(episodes):]:
        axis.axis("off")
    axes.flat[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic trajectoire SAC v4")
    parser.add_argument("--model", default="sac_nav_v4.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--level", type=int, default=15)
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--out-dir", default="analysis_sac_v4")
    parser.add_argument("--plot-episodes", type=int, default=9)
    args = parser.parse_args()

    # Pour une petite MLP, l'overhead des nombreux threads BLAS depasse largement
    # le calcul par step sur CPU. Ce choix ne change pas les actions deterministes.
    torch.set_num_threads(1)
    os.makedirs(args.out_dir, exist_ok=True)
    curriculum = CurriculumManager()
    curriculum.set_level(args.level)
    model = load_sac_from_memory(args.model)
    env = NavEnv(curriculum)
    summaries, plot_data = [], []
    try:
        for i in range(args.episodes):
            summary, path, start, goal, initial_obstacles, dynamic_paths = run_episode(env, model, args.base_seed + i)
            summaries.append(summary)
            if len(plot_data) < args.plot_episodes:
                plot_data.append((summary, path, start, goal, initial_obstacles, dynamic_paths))
            print(f"{i + 1:>3}/{args.episodes}: {summary['outcome']:<9} "
                  f"steps={summary['steps']:<3} clearance={summary['min_clearance_m']:.2f} m "
                  f"eff={summary['path_efficiency']:.2f}")
    finally:
        env.close()

    with open(os.path.join(args.out_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    counts = {name: sum(row["outcome"] == name for row in summaries) for name in ("success", "collision", "timeout")}
    report = {
        "episodes": args.episodes,
        "level": args.level,
        "base_seed": args.base_seed,
        "counts": counts,
        "rates": {name: round(value / args.episodes, 4) for name, value in counts.items()},
        "mean": {key: round(float(np.mean([row[key] for row in summaries])), 4)
                 for key in summaries[0] if key not in {"seed", "outcome"}},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    if plot_data:
        save_plot(plot_data, os.path.join(args.out_dir, "trajectories.png"))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
