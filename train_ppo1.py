"""
Navigation RL avec PyBullet : PPO classique (Stable-Baselines3), pour
comparaison directe avec sac_train.py.

Reutilise NavEnv telle quelle (meme physique, meme observation, meme reward)
pour garantir une comparaison equitable avec le run SAC. Seule difference
structurelle : PPO tourne sur plusieurs environnements paralleles
(SubprocVecEnv) pour compenser sa moindre sample-efficiency, ce qui casse le
partage memoire du CurriculumManager d'origine -> voir SyncedCurriculumCallback
plus bas pour la resynchronisation entre workers.

Entrainement : python train_ppo.py [--n-envs 8] [--timesteps 5000000]
Test visuel  : python train_ppo.py --play

Prerequis : sac_train.py doit se trouver dans le meme dossier (import direct
de NavEnv/CurriculumManager, pas de duplication de code).
"""
import argparse
import os
import zipfile
import numpy as np
import io
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn
import torch

from sac_train_v3 import NavEnv, CurriculumManager, resolve_model_path


# ----------------------- Niveau pousse depuis le process principal -----------------------
class LevelHolder:
    """
    Remplace CurriculumManager a l'interieur de chaque worker.

    NavEnv n'appelle que `.params` (propriete) et `.report(success)` sur cet
    objet -> le duck-typing suffit, pas besoin de toucher a sac_train.py.
    Le niveau reel est calcule globalement par SyncedCurriculumCallback (dans
    le process principal) puis pousse ici via set_level().
    """
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
    """Meme physique/observation/reward que NavEnv (heritage direct), plus un
    hook set_level() appelable a distance via env_method() depuis le
    process principal."""
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
    """
    Avec SAC (1 seul env, 1 seul process), le CurriculumManager d'origine
    suffit : un historique, un niveau. Avec PPO on utilise plusieurs envs en
    parallele (SubprocVecEnv) -> chaque worker vit dans un process separe et
    NE PEUT PAS partager un objet Python en memoire avec les autres.

    Ce callback agrege les succes/echecs de TOUS les workers dans le process
    principal (via les `infos`/`dones` que SB3 remonte a chaque step), decide
    du changement de niveau exactement comme CurriculumManager.report(), puis
    diffuse le nouveau niveau a chaque worker via env_method("set_level", ...).
    """
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
            print(f"\n=== CURRICULUM (PPO, {self.training_env.num_envs} envs "
                  f"synchronises): passage au niveau {self.level} ===\n")

        if self.n_calls % 10_000 == 0:
            sr = np.mean(self.history) if self.history else 0.0
            print(f"steps={self.num_timesteps} niveau={self.level} "
                  f"succes_fenetre={sr:.2f}")
        return True


# ----------------------- Entrainement -----------------------
def train(n_envs=8, total_timesteps=2_000_000):
    env_fns = [make_env() for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    # Les recompenses vont de -60 (collision) a +100 (succes) alors que le
    # shaping par pas est de l'ordre de +-0.5 -> distribution tres bimodale.
    # SAC l'encaisse bien (replay buffer + double critique), PPO bootstrap sa
    # fonction de valeur sur des retours bruts et devient instable avec cet
    # ecart d'echelle : on normalise la recompense. Les observations sont
    # deja bornees dans [-1,1] -> inutile (et potentiellement nuisible) de
    # les normaliser en plus.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,  
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,          # exploration geree par gSDE, pas par un bonus fixe
        vf_coef=0.5,
        use_sde=True,          # exploration correlee dans le temps, adaptee
        sde_sample_freq=4,     # au controle continu robotique (vs iid gaussien)
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                            log_std_init=-1.0),
        verbose=1, tensorboard_log="./tb_nav_ppo",
    )
    model.learn(total_timesteps=total_timesteps, callback=SyncedCurriculumCallback())
    model.save("ppo_nav_final")
    vec_env.save("ppo_nav_vecnormalize.pkl")  # stats de normalisation, requises pour play()
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
    env = DummyVecEnv([make_env(render_mode="human")])
    env = VecNormalize.load("ppo_nav_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False  # on veut voir la vraie recompense a l'ecran
    env.env_method("set_level", len(CurriculumManager.LEVELS) - 1)  # pleine difficulte

    model = load_ppo("ppo_nav_final.zip")
    for ep in range(20):
        obs = env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = bool(done[0])
            total += reward[0]
        print(f"Episode {ep}: reward={total:.1f} succes={info[0]['success']}")
    env.close()
def evaluate(n_episodes=100, base_seed=10_000, level=None):
    """
    Evaluation a seeds fixes (pas le succes du curriculum, qui est auto-selectionne
    et bruite) : sert de reference reproductible pour comparer l'ancien modele
    (obs lidar seule) et le nouveau (obs enrichie + recompense TTC) sur exactement
    les memes episodes.
    """
    curr = CurriculumManager()
    curr.level = level if level is not None else len(CurriculumManager.LEVELS) - 1
    env = NavEnv(curr)
    model = load_ppo("ppo_nav_final.zip")
    successes, collisions, timeouts, steps_list = 0, 0, 0, []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        done, info = False, {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        if info.get("success"):
            successes += 1
        elif env.steps >= env.MAX_STEPS:
            timeouts += 1
        else:
            collisions += 1
        steps_list.append(env.steps)
    env.close()
    print(f"Niveau curriculum evalue : {curr.level}")
    print(f"Taux de succes : {successes / n_episodes:.2%}  ({successes}/{n_episodes})")
    print(f"Collisions     : {collisions / n_episodes:.2%}")
    print(f"Timeouts       : {timeouts / n_episodes:.2%}")
    print(f"Duree moyenne  : {np.mean(steps_list):.0f} steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--n-envs", type=int, default=8,
                     help="reduire si CPU limite (SubprocVecEnv = 1 process par env)")
    ap.add_argument("--timesteps", type=int, default=3_000_000,
                     help="PPO est on-policy -> generalement 3-5x plus de "
                          "steps que SAC pour atteindre un niveau comparable")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.play:
           play()
    elif args.eval:
        evaluate()
    else:
        train()