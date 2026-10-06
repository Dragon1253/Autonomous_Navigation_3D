"""
Navigation RL avec PyBullet : PPO classique + frame-stacking, pour isoler
l'effet de la memoire a court terme sur l'evitement d'obstacles dynamiques,
en comparaison avec train_ppo1.py (PPO sans stacking) et
train_recurrent_ppo.py (memoire via LSTM).

Reutilise NavEnv telle quelle. Tous les hyperparametres PPO sont IDENTIQUES a
train_ppo1.py (learning rate, n_steps, gSDE, target_kl, architecture...) :
seule difference = VecFrameStack ajoute avant VecNormalize. Le frame-stacking
est donc la SEULE variable testee par rapport au run PPO precedent.

Choix de N_STACK = 8 (~267 ms a 30 Hz) :
  - Vitesse des obstacles dynamiques (CurriculumManager.LEVELS) : 0.2-0.35 m/s
    a leur apparition (niveau 2) jusqu'a 0.8-0.95 m/s au niveau final.
  - Deplacement par step : 0.0067 a 0.0317 m. Le robot se deplace lui-meme
    jusqu'a 1.0 m/s (DT=1/30s) -> a vitesse lente, le mouvement propre du
    robot peut etre 3-5x plus grand que celui de l'obstacle a detecter :
    trop peu de frames noie le signal dans le mouvement propre.
  - Mais les obstacles rebondissent sur les murs et entre eux
    (_move_obstacles) : une fenetre trop longue peut englober un rebond et
    rendre la "vitesse apparente" (differences finies sur la pile) trompeuse.
  - A n_stack=8 : ~5.3 cm de deplacement cumule au niveau le plus lent
    (exploitable sur plusieurs differences successives), ~25 cm au niveau le
    plus rapide (comparable au rayon d'un obstacle, largement detectable),
    et la fenetre (267 ms) reste courte par rapport au temps typique entre
    deux rebonds. Cet ordre de grandeur correspond aussi au temps que met le
    robot a traverser sa zone de danger (0.5 m) a vitesse max (1 m/s).

N_STACK est une constante de module (pas un flag CLI) : la changer casse la
compatibilite entre un modele deja entraine et son VecNormalize sauvegarde
(dimension d'observation differente) -> il faudrait ré-entrainer from scratch.

Entrainement : python train_ppo_framestack.py [--n-envs 8] [--timesteps 5000000]
Test visuel  : python train_ppo_framestack.py --play

Prerequis : sac_train.py doit se trouver dans le meme dossier.
"""
import argparse
import zipfile
import numpy as np
import io
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecNormalize, VecFrameStack,
)
from stable_baselines3.common.utils import get_linear_fn
import torch

from sac_train import NavEnv, CurriculumManager, resolve_model_path

N_STACK = 8  # voir justification dans le docstring du module


# ----------------------- Niveau pousse depuis le process principal -----------------------
class LevelHolder:
    """Remplace CurriculumManager a l'interieur de chaque worker (voir
    train_ppo1.py pour l'explication complete). Plumbing dupliquee ici pour
    que ce fichier reste executable seul."""
    def __init__(self, level=0):
        self.level = level

    @property
    def params(self):
        return CurriculumManager.LEVELS[self.level]

    def report(self, success):
        pass  # agregation globale faite par le callback, pas par worker

    def set_level(self, level):
        self.level = level


class NavEnvPPO(NavEnv):
    """Meme physique/observation/reward que NavEnv, + hook set_level()
    appelable a distance via env_method(). Le frame-stacking est applique
    au niveau du VecEnv (VecFrameStack), pas ici -> NavEnv reste identique
    a celui utilise par SAC et RecurrentPPO."""
    def __init__(self, render_mode=None):
        super().__init__(LevelHolder(0), render_mode=render_mode)

    def set_level(self, level):
        self.curr.set_level(level)


def make_env(render_mode=None):
    def _init():
        return Monitor(NavEnvPPO(render_mode=render_mode))
    return _init


# ----------------------- Curriculum synchronise entre workers -----------------------
class SyncedCurriculumCallback(BaseCallback):
    """Identique a train_ppo1.py. VecFrameStack ne transforme que les
    observations et laisse passer infos/dones inchanges (VecEnvWrapper
    standard) -> aucune adaptation necessaire ici."""
    WINDOW = 50
    THRESHOLD = 0.8

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.level = 0
        self.history = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for done, info in zip(dones, infos):
            if done:
                self.history.append(1.0 if info.get("success", False) else 0.0)
        if len(self.history) > self.WINDOW:
            self.history = self.history[-self.WINDOW:]

        if (len(self.history) == self.WINDOW
                and np.mean(self.history) >= self.THRESHOLD
                and self.level < len(CurriculumManager.LEVELS) - 1):
            self.level += 1
            self.history = []
            self.training_env.env_method("set_level", self.level)
            print(f"\n=== CURRICULUM (PPO+framestack, {self.training_env.num_envs} envs "
                  f"synchronises): passage au niveau {self.level} ===\n")

        if self.n_calls % 10_000 == 0:
            sr = np.mean(self.history) if self.history else 0.0
            print(f"steps={self.num_timesteps} niveau={self.level} "
                  f"succes_fenetre={sr:.2f}")
        return True


# ----------------------- Construction du VecEnv (stack, sans normalisation) -----------------------
def build_stacked_env(env_fns):
    vec_env = SubprocVecEnv(env_fns) if len(env_fns) > 1 else DummyVecEnv(env_fns)
    # Stack AVANT normalisation : chaque frame empilee reste bornee dans
    # [-1,1] telle quelle, ce qui garde les valeurs interpretables et evite
    # de coupler la normalisation de recompense a la largeur de la pile.
    return VecFrameStack(vec_env, n_stack=N_STACK)


# ----------------------- Entrainement -----------------------
def train(n_envs=8, total_timesteps=5_000_000):
    env_fns = [make_env() for _ in range(n_envs)]
    vec_env = build_stacked_env(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", vec_env,
        # --- Hyperparametres strictement identiques a train_ppo1.py ---
        # (le frame-stacking est la seule variable testee ici)
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                            log_std_init=-1.0),
        verbose=1, tensorboard_log="./tb_nav_ppo_framestack",
    )
    model.learn(total_timesteps=total_timesteps, callback=SyncedCurriculumCallback())
    model.save("ppo_framestack_nav_final")
    vec_env.save("ppo_framestack_nav_vecnormalize.pkl")
    vec_env.close()


# ----------------------- Test visuel -----------------------
def resolve_model_path(model_path):
    """Ajoute l'extension .zip si elle n'est pas spécifiée dans le chemin."""
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path

def load_ppo(model_path):
    """
    Charge le modèle PPO en gérant le bogue de lecture PyTorchStreamReader
    via la lecture en mémoire (BytesIO).
    """

    model_path = resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle est introuvable : {model_path}")

    try:
        return PPO.load(model_path)
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
            return PPO.load(model_path)
        finally:
            torch.load = original_torch_load
def play():
    stacked_env = build_stacked_env([make_env(render_mode="human")])
    # VecFrameStack est deterministe (aucune stat a sauvegarder/charger) :
    # on le reconstruit a l'identique, seul VecNormalize doit recuperer ses
    # stats de normalisation sauvegardees pendant l'entrainement.
    env = VecNormalize.load("ppo_framestack_nav_vecnormalize.pkl", stacked_env)
    env.training = False
    env.norm_reward = False
    env.env_method("set_level", len(CurriculumManager.LEVELS) - 1)  # pleine difficulte

    model = load_ppo("ppo_framestack_nav_final")
    for ep in range(20):
        obs = env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = bool(done_arr[0])
            total += reward[0]
        print(f"Episode {ep}: reward={total:.1f} succes={info[0]['success']}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--n-envs", type=int, default=8,
                     help="reduire si CPU limite (SubprocVecEnv = 1 process par env)")
    ap.add_argument("--timesteps", type=int, default=5_000_000,
                     help="meme budget que train_ppo1.py pour une comparaison equitable")
    args = ap.parse_args()
    play() if args.play else train(n_envs=args.n_envs, total_timesteps=args.timesteps)