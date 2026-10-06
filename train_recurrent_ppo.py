"""
Navigation RL avec PyBullet : RecurrentPPO / PPO-LSTM (sb3-contrib), pour
comparaison directe avec sac_train.py et train_ppo1.py.

Reutilise NavEnv telle quelle (meme physique, meme observation, meme reward)
-> seule la politique change (LSTM au lieu de MLP feedforward), pour isoler
l'effet de la memoire temporelle sur l'evitement d'obstacles dynamiques.

Prerequis :
  pip install sb3-contrib
  sac_train.py doit se trouver dans le meme dossier (import direct de
  NavEnv/CurriculumManager, pas de duplication de code).

Entrainement : python train_recurrent_ppo.py [--n-envs 8] [--timesteps 3000000]
Test visuel  : python train_recurrent_ppo.py --play
"""
import argparse
import io
import os
import zipfile
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_linear_fn
import torch

from sac_train import NavEnv, CurriculumManager, resolve_model_path


# ----------------------- Niveau pousse depuis le process principal -----------------------
class LevelHolder:
    """Remplace CurriculumManager a l'interieur de chaque worker (voir
    train_ppo1.py pour l'explication complete). Plumbing algo-agnostique,
    dupliquee ici pour que ce fichier reste executable seul."""
    def __init__(self, level=0):
        self.level = level

    @property
    def params(self):
        return CurriculumManager.LEVELS[self.level]

    def report(self, success):
        pass  # agregation globale faite par le callback, pas par worker

    def set_level(self, level):
        self.level = level


class NavEnvRecurrent(NavEnv):
    """Meme physique/observation/reward que NavEnv, + hook set_level()
    appelable a distance via env_method()."""
    def __init__(self, render_mode=None):
        super().__init__(LevelHolder(0), render_mode=render_mode)

    def set_level(self, level):
        self.curr.set_level(level)


def make_env(render_mode=None):
    def _init():
        return Monitor(NavEnvRecurrent(render_mode=render_mode))
    return _init


# ----------------------- Curriculum synchronise entre workers -----------------------
class SyncedCurriculumCallback(BaseCallback):
    """Identique a celui de train_ppo1.py : agrege succes/echecs de tous les
    workers dans le process principal, decide du changement de niveau, le
    diffuse via env_method. Algo-agnostique (ne regarde que infos/dones)."""
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
            print(f"\n=== CURRICULUM (RecurrentPPO, {self.training_env.num_envs} envs "
                  f"synchronises): passage au niveau {self.level} ===\n")

        if self.n_calls % 10_000 == 0:
            sr = np.mean(self.history) if self.history else 0.0
            print(f"steps={self.num_timesteps} niveau={self.level} "
                  f"succes_fenetre={sr:.2f}")
        return True


# ----------------------- Entrainement -----------------------
def load_recurrent_ppo(model_path):
    """
    Charge le modèle RecurrentPPO en gérant le bogue de lecture PyTorchStreamReader
    via la lecture en mémoire (BytesIO).
    """

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
def train(n_envs=8, total_timesteps=3_000_000):
    env_fns = [make_env() for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=0.99)

    model = RecurrentPPO(
        "MlpLstmPolicy", vec_env,
        learning_rate=3e-4,
        # n_steps = longueur de BPTT, pas juste "taille du buffer" comme en
        # PPO feedforward. 128 pas (~4.3s a 30Hz) suffit a voir un obstacle
        # dynamique faire un rebond complet + une manoeuvre d'evitement,
        # sans faire exploser le cout de la retropropagation dans le temps.
        n_steps=128,
        batch_size=256,   # buffer = 128*8 = 1024, multiple propre de n_envs et de batch_size
        n_epochs=4,        # moins d'epochs qu'en feedforward : sequences BPTT
                           # courtes + memes donnees -> plus instable si on sur-optimise
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,     # exploration geree par l'entropie, pas par gSDE (voir plus bas)
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,    # resserre par rapport au PPO feedforward (0.03) :
                           # les updates recurrentes sont plus erratiques
        # PAS de use_sde ici : gSDE + memoire LSTM sont tous deux des
        # mecanismes d'exploration correles dans le temps. Les combiner
        # ajoute un facteur confondant difficile a diagnostiquer si
        # l'entrainement diverge -> on garde un seul levier (entropie) pour
        # cette comparaison.
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,      # LSTM separe acteur/critique : evite les
            enable_critic_lstm=True,  # gradients contradictoires entre les deux objectifs
            net_arch=dict(pi=[128], vf=[128]),  # tetes MLP legeres : le LSTM
                                                  # porte deja la representation temporelle
            log_std_init=-1.0,
        ),
        verbose=1, tensorboard_log="./tb_nav_recurrent_ppo",
    )
    model.learn(total_timesteps=total_timesteps, callback=SyncedCurriculumCallback())
    model.save("recurrent_ppo_nav_final")
    vec_env.save("recurrent_ppo_nav_vecnormalize.pkl")
    vec_env.close()


# ----------------------- Test visuel -----------------------
def play():
    env = DummyVecEnv([make_env(render_mode="human")])
    env = VecNormalize.load("recurrent_ppo_nav_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
    env.env_method("set_level", len(CurriculumManager.LEVELS) - 1)  # pleine difficulte

    model = load_recurrent_ppo("recurrent_ppo_nav_final.zip")
    n_envs = 1

    for ep in range(20):
        obs = env.reset()
        # CRITIQUE pour RecurrentPPO : sans reinitialiser lstm_states=None et
        # episode_starts=True au debut de chaque episode de test, l'etat
        # cache LSTM n'est jamais correctement propage/reinitialise -> le
        # modele se comporte de facon quasi aleatoire meme si l'entrainement
        # s'est bien passe. C'est le piege d'inference classique du LSTM-PPO.
        lstm_states = None
        episode_starts = np.ones((n_envs,), dtype=bool)
        done, total = False, 0.0
        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts,
                deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            done = bool(done_arr[0])
            episode_starts = done_arr
            total += reward[0]
        print(f"Episode {ep}: reward={total:.1f} succes={info[0]['success']}")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--n-envs", type=int, default=8,
                     help="reduire si CPU limite (le LSTM est plus couteux "
                          "par step que le MLP feedforward)")
    ap.add_argument("--timesteps", type=int, default=3_000_000,
                     help="a budget de steps egal, RecurrentPPO est plus "
                          "lent en temps reel que PPO (calcul LSTM sequentiel) "
                          "-> prevoir un temps d'entrainement plus long")
    args = ap.parse_args()
    play() if args.play else train(n_envs=args.n_envs, total_timesteps=args.timesteps)
