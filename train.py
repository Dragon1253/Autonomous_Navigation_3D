"""
train.py — Entraînement RecurrentPPO (LSTM) pour l'étape "obstacles dynamiques".

Pourquoi RecurrentPPO et pas PPO ?
  L'observation reste un simple snapshot à l'instant t (lidar + goal + vel +
  prev_action). Elle ne contient AUCUNE information de vitesse des obstacles.
  Avec un MLP feed-forward (PPO stage 0), l'agent ne peut pas distinguer "obstacle
  qui s'approche" de "obstacle qui s'éloigne" à partir d'une seule frame -> il ne
  peut qu'apprendre une politique réactive dangereuse. RecurrentPPO ajoute une
  mémoire (LSTM) qui intègre l'historique récent d'observations, permettant à
  l'agent d'inférer implicitement la trajectoire des obstacles mobiles.

Important : on NE PART PAS des poids du modèle PPO statique (stage 0). L'ancienne
politique MlpPolicy et la nouvelle MlpLstmPolicy n'ont pas la même architecture
(pas de couche LSTM côté ancien modèle) -> les poids ne sont pas transférables,
il faut réentraîner de zéro.

Correctifs appliqués suite au diagnostic du run précédent (échec à 1.9M/2M steps) :

  1. n_steps/batch_size réellement câblés depuis argparse (avant : hardcodés dans
     le constructeur, les flags CLI ne servaient à rien). Défaut n_steps abaissé
     de 512 à 128 : avec une politique récurrente, n_steps = longueur de la
     fenêtre de BPTT. 512 pas de rétropropagation à travers le temps sur un LSTM
     tout juste initialisé est une source classique de gradients instables.
  2. target_kl=0.03 : garde-fou qui arrête les epochs PPO en cours de route si la
     divergence KL devient trop grande, pour éviter les updates destructeurs
     (critique avec n_epochs=10 sur une longue séquence récurrente).
  3. Learning rate en rampe linéaire (au lieu d'une valeur constante) pour que le
     pas d'optimisation diminue à mesure que la politique s'approche d'une bonne
     solution, au lieu de continuer à la faire osciller en fin d'entraînement.
  4. Curriculum en DEUX PHASES bien séparées au lieu d'un curriculum joint qui
     ramplait densité statique ET obstacles dynamiques en même temps dès le
     step 0 :
       - Phase warmup (0 -> warmup_fraction) : densité statique fixe (déjà
         validée en stage 0), AUCUN obstacle dynamique. Le LSTM apprend d'abord
         à stabiliser le contrôle moteur et sa mémoire sur un environnement
         stationnaire.
       - Phase dynamique (warmup_fraction -> 1.0) : rampe progressive du nombre
         d'obstacles dynamiques de start_dynamic_obstacles -> end_dynamic_obstacles.
     Cela évite de superposer trois sources de non-stationnarité (nouvelle
     architecture + optimisation + difficulté croissante) dès le début.

Usage :
    pip install stable-baselines3 sb3-contrib gymnasium pybullet
    python train.py --timesteps 2_000_000 --n-envs 8 --end-dynamic-obstacles 4
"""

import argparse
import os

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import RecurrentPPO

from nav_env_ppo import RobotNavEnv


# --------------------------------------------------------------------------- #
# Learning rate schedule
# --------------------------------------------------------------------------- #
def linear_schedule(initial_value: float):
    """
    Décroissance linéaire du learning rate : `initial_value` au début de
    l'entraînement, 0 à la toute fin. SB3 appelle cette fonction à chaque
    update avec `progress_remaining` qui va de 1.0 (début) à 0.0 (fin).
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


# --------------------------------------------------------------------------- #
# Curriculum en deux phases : warmup statique -> rampe dynamique
# --------------------------------------------------------------------------- #
class TwoPhaseCurriculumCallback(BaseCallback):
    """
    Phase 1 [0, warmup_fraction) :
        n_dynamic_obstacles = start_dynamic_obstacles (typiquement 0).
        La densité statique n'est PAS ramplée ici : elle est déjà validée en
        stage 0, donc on la garde fixe à sa valeur cible pendant toute
        l'entraînement (voir --obstacle-density).

    Phase 2 [warmup_fraction, 1.0] :
        Rampe linéaire de n_dynamic_obstacles de start_dynamic_obstacles
        jusqu'à end_dynamic_obstacles, atteinte exactement à la fin de
        l'entraînement.
    """

    def __init__(self, total_timesteps, warmup_fraction, start_dynamic, end_dynamic, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.warmup_fraction = warmup_fraction
        self.start_dynamic = start_dynamic
        self.end_dynamic = end_dynamic
        self._last_dynamic = None

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(1, self.total_timesteps)

        if progress < self.warmup_fraction:
            current_dynamic = self.start_dynamic
            phase_label = "WARMUP (statique uniquement)"
        else:
            ramp_progress = (progress - self.warmup_fraction) / max(1e-8, 1.0 - self.warmup_fraction)
            ramp_progress = min(1.0, ramp_progress)
            current_dynamic = round(
                self.start_dynamic + ramp_progress * (self.end_dynamic - self.start_dynamic)
            )
            phase_label = "DYNAMIQUE"

        if current_dynamic != self._last_dynamic:
            self.training_env.env_method("set_dynamic_obstacle_count", current_dynamic)
            self._last_dynamic = current_dynamic
            if self.verbose:
                print(f"[Curriculum] step={self.num_timesteps:,} | {phase_label} | Dynamic={current_dynamic}")

        return True


# --------------------------------------------------------------------------- #
def make_env_fn(rank, seed, obstacle_density, n_dynamic_obstacles_start, log_dir):
    def _init():
        env = RobotNavEnv(
            render_mode="none",
            obstacle_density=obstacle_density,       # fixe pendant tout l'entraînement (déjà validé stage 0)
            n_dynamic_obstacles=n_dynamic_obstacles_start,
            seed=seed + rank,
        )
        env = Monitor(env, filename=os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO on dynamic obstacles")
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=128,
                         help="Longueur du rollout par env avant chaque update PPO. "
                              "Pour une politique récurrente, c'est aussi la fenêtre de "
                              "BPTT : garder cette valeur raisonnable (<=256) pour éviter "
                              "les gradients instables sur le LSTM.")
    parser.add_argument("--batch-size", type=int, default=128,
                         help="Doit diviser (n_envs * n_steps) ET être un multiple de n_envs "
                              "(contrainte RecurrentPPO pour garder les séquences par env intactes).")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=0.03,
                         help="Arrête les epochs PPO en cours si la divergence KL dépasse ce "
                              "seuil, pour éviter les updates destructeurs.")
    parser.add_argument("--obstacle-density", type=float, default=1.0,
                         help="Densité d'obstacles statiques (déjà validée en stage 0), fixe "
                              "sur toute la durée de l'entraînement.")
    parser.add_argument("--start-dynamic-obstacles", type=int, default=0)
    parser.add_argument("--end-dynamic-obstacles", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.3,
                         help="Fraction de l'entraînement pendant laquelle AUCUN obstacle "
                              "dynamique n'est présent (LSTM apprend d'abord le contrôle de "
                              "base). La rampe vers end-dynamic-obstacles se déroule ensuite "
                              "sur le reste de l'entraînement.")
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                         help="Valeur initiale du learning rate ; décroît ensuite linéairement vers 0.")
    parser.add_argument("--gamma", type=float, default=0.99,
                         help="DOIT correspondre au gamma utilisé dans nav_env.py pour le "
                              "reward shaping potential-based")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/recurrent_ppo_dynamic")
    parser.add_argument("--log-dir", type=str, default="logs/dynamic_obstacles")
    parser.add_argument("--tensorboard-log", type=str, default="tb_logs/dynamic_obstacles")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                         help="En pas d'environnement TOTAUX (divisé par n_envs en interne)")
    args = parser.parse_args()

    # Garde-fous : évite de reproduire silencieusement le bug où n_steps/batch_size
    # étaient ignorés. On préfère planter tout de suite avec un message clair.
    buffer_size = args.n_envs * args.n_steps
    if buffer_size % args.batch_size != 0:
        parser.error(
            f"batch_size ({args.batch_size}) doit diviser n_envs * n_steps ({buffer_size})."
        )
    if args.batch_size % args.n_envs != 0:
        parser.error(
            f"batch_size ({args.batch_size}) doit être un multiple de n_envs ({args.n_envs}) "
            "pour RecurrentPPO (les séquences par environnement doivent rester entières)."
        )
    if not (0.0 <= args.warmup_fraction < 1.0):
        parser.error("warmup_fraction doit être dans [0.0, 1.0).")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

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

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        n_steps=args.n_steps,           # câblé depuis la CLI (avant : hardcodé à 512)
        batch_size=args.batch_size,     # câblé depuis la CLI (avant : hardcodé à 128)
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=args.target_kl,       # nouveau : garde-fou contre les updates destructeurs
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=args.learning_rate,  # rampe linéaire au lieu d'une valeur constante
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Feature extractor avant le LSTM
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=1,       # 1 couche suffit, 2 ralentissent trop sans gain notable
            shared_lstm=False,     # Séparer la mémoire de l'acteur (pi) et du critique (vf)
            enable_critic_lstm=True,
        ),
        tensorboard_log=args.tensorboard_log,
        verbose=1,
        seed=args.seed,
    )

    curriculum_cb = TwoPhaseCurriculumCallback(
        total_timesteps=args.timesteps,
        warmup_fraction=args.warmup_fraction,
        start_dynamic=args.start_dynamic_obstacles,
        end_dynamic=args.end_dynamic_obstacles,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix="recurrent_ppo_dynamic_ckpt",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([curriculum_cb, checkpoint_cb]),
    )

    model.save(args.save_path)
    print(f"Modèle final sauvegardé : {args.save_path}.zip")


if __name__ == "__main__":
    main()
