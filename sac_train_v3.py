"""
Navigation RL avec PyBullet : point A -> B, obstacles statiques + dynamiques,
curriculum adaptatif, SAC (Stable-Baselines3).

v2 - integre les ajouts valides sur le banc de test dynamique-only :
1. Observation enrichie : position + vitesse relative (repere robot) des
   K_NEAREST obstacles dynamiques les plus proches, en plus du lidar. Sans ca,
   le reseau doit inferer la vitesse des obstacles a partir de variations de
   distance lidar polluees par le mouvement propre du robot -> tres difficile
   a apprendre implicitement, quelle que soit l'architecture.
2. Recompense TTC (time-to-collision) basee sur la vraie vitesse de
   rapprochement (obstacle - robot), qui remplace/complete l'ancienne penalite
   de proximite brute. Cette formule s'applique correctement aussi bien aux
   obstacles statiques (vel=0, la vitesse de rapprochement vient alors
   uniquement du mouvement du robot) qu'aux obstacles dynamiques.
3. Mouvement des obstacles dynamiques moins previsible : perturbation
   aleatoire periodique de la direction (perturb_prob, en plus du rebond sur
   les murs), pour eviter une simple extrapolation balistique.
4. Curriculum revu : le champ perturb_prob est desormais precise a chaque
   palier. Les 2 premiers paliers avec obstacles dynamiques restent
   balistiques (perturb_prob=0) pour poser les bases, la perturbation
   n'apparait qu'ensuite - meme logique de prevute-avant-imprevu que sur le
   banc de test isole.

Entraînement : python sac_train.py
Test visuel  : python sac_train.py --play
"""
import argparse
import io
import math
import os
import zipfile
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


# ----------------------- Curriculum adaptatif -----------------------
class CurriculumManager:
    """Passe au niveau suivant si succes >= seuil sur une fenetre glissante."""
    LEVELS = [
        dict(n_static=(3, 3),   n_dyn=(0, 0),  speed=(0.0, 0.0),   perturb_prob=0.00,  min_goal=4.0),
        dict(n_static=(5, 7),   n_dyn=(0, 0),  speed=(0.0, 0.0),   perturb_prob=0.00,  min_goal=5.0),
        dict(n_static=(5, 7),   n_dyn=(1, 1),  speed=(0.2, 0.3),   perturb_prob=0.00,  min_goal=5.0),
        dict(n_static=(5, 7),   n_dyn=(2, 2),  speed=(0.2, 0.35),  perturb_prob=0.025, min_goal=5.0),
        dict(n_static=(7, 9),   n_dyn=(2, 3),  speed=(0.25, 0.4),  perturb_prob=0.10,  min_goal=5.5),
         dict(n_static=(7, 9),   n_dyn=(3, 4),  speed=(0.3, 0.5),   perturb_prob=0.125, min_goal=6.0),
         dict(n_static=(8, 10),  n_dyn=(3, 5),  speed=(0.3, 0.6),   perturb_prob=0.175, min_goal=6.0),
        dict(n_static=(8, 10),  n_dyn=(4, 6),  speed=(0.4, 0.65),  perturb_prob=0.175, min_goal=6.0),
        dict(n_static=(9, 11),  n_dyn=(5, 7),  speed=(0.45, 0.7),  perturb_prob=0.175, min_goal=6.0),
        dict(n_static=(9, 12),  n_dyn=(6, 8),  speed=(0.5, 0.75),  perturb_prob=0.225, min_goal=6.0),
        dict(n_static=(10, 12), n_dyn=(7, 9),  speed=(0.55, 0.8),  perturb_prob=0.225, min_goal=6.0),
        dict(n_static=(10, 13), n_dyn=(8, 10), speed=(0.6, 0.85),  perturb_prob=0.225, min_goal=6.0),
        dict(n_static=(12, 16), n_dyn=(9, 11), speed=(0.65, 0.85), perturb_prob=0.225, min_goal=6.0),
        dict(n_static=(14, 20), n_dyn=(10, 12), speed=(0.7, 0.9),  perturb_prob=0.25,  min_goal=6.0),
        # dict(n_static=(18, 25), n_dyn=(10, 13), speed=(0.75, 0.9), perturb_prob=0.25,  min_goal=6.0),
        dict(n_static=(20, 30), n_dyn=(10, 15), speed=(0.7, 0.8), perturb_prob=0.25,  min_goal=6.0),  # objectif
    ]
    WINDOW = 50
    THRESHOLD = 0.8

    def __init__(self):
        self.level = 0
        self.history = []

    def report(self, success: bool):
        self.history.append(1.0 if success else 0.0)
        if len(self.history) > self.WINDOW:
            self.history.pop(0)
        if (self.level < len(self.LEVELS) - 1
                and len(self.history) == self.WINDOW
                and np.mean(self.history) >= self.THRESHOLD):
            self.level += 1
            self.history = []
            print(f"\n=== CURRICULUM: passage au niveau {self.level} ===\n")

    @property
    def params(self):
        return self.LEVELS[self.level]

# ----------------------- Environnement -----------------------
class NavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ARENA = 6.0          # demi-taille de l'arene (12 x 12 m)
    R_ROBOT = 0.25
    V_MAX = 1.0          # m/s
    W_MAX = 2.0          # rad/s
    DT = 1.0 / 30.0
    N_RAYS = 24
    RAY_RANGE = 5.0
    GOAL_RADIUS = 0.45
    MAX_STEPS = 800
    K_NEAREST = 2         # nb d'obstacles dynamiques dont on donne pos+vitesse relatives
    V_OBS_NORM = 2.0       # normalisation des vitesses relatives (robot + obstacle <= ~2 m/s)

    def __init__(self, curriculum: CurriculumManager, render_mode=None):
        super().__init__()
        self.curr = curriculum
        self.render_mode = render_mode
        self.cid = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self._build_walls()
        self.robot_id = self._make_cylinder(self.R_ROBOT, 0.3, [0.1, 0.4, 0.9, 1])
        self.goal_vis = self._make_cylinder(self.GOAL_RADIUS, 0.02, [0.1, 0.9, 0.1, 0.6],
                                            collision=False)
        self.obstacles = []   # dicts: id, pos(2), vel(2), r, dynamic, perturb_prob
        self.OBS_BASE = 5 + 4 * self.K_NEAREST   # index ou commence le lidar dans l'obs
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            shape=(self.OBS_BASE + self.N_RAYS,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()

    # ---------- construction ----------
    def _make_cylinder(self, radius, height, rgba, collision=True):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                     physicsClientId=self.cid) if collision else -1
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=rgba, physicsClientId=self.cid)
        return p.createMultiBody(0, col, vis, [0, 0, height / 2],
                                 physicsClientId=self.cid)

    def _build_walls(self):
        t, h, L = 0.1, 0.5, self.ARENA
        for cx, cy, ex, ey in [(L, 0, t, L), (-L, 0, t, L), (0, L, L, t), (0, -L, L, t)]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ex, ey, h],
                                         physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[ex, ey, h],
                                      rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.cid)
            p.createMultiBody(0, col, vis, [cx, cy, h], physicsClientId=self.cid)

    # ---------- placement sans chevauchement ----------
    def _free(self, pos, r, others, clearance=None):
        # clearance par defaut : diametre robot (2 * R_ROBOT) + 0.15m de marge de passage
        if clearance is None:
            clearance = 2 * self.R_ROBOT + 0.15

        if np.max(np.abs(pos)) > self.ARENA - r - 0.15:
            return False
        for o in others:
            if np.linalg.norm(pos - o["pos"]) < r + o["r"] + clearance:
                return False
        return True

    def _spawn_obstacles(self, n_static, n_dyn, speed_rng, perturb_prob, start, goal):
        for o in self.obstacles:
            p.removeBody(o["id"], physicsClientId=self.cid)
        self.obstacles = []
        seg = goal - start
        total = n_static + n_dyn
        n_on_path = max(1, int(0.4 * total))  # garantit des obstacles sur la trajectoire

        # Espace minimum garanti entre la surface de deux obstacles
        min_pass_gap = 2 * self.R_ROBOT + 0.3

        for i in range(total):
            dynamic = i >= n_static
            r = float(self.rng.uniform(0.25, 0.5))
            for _ in range(200):  # augmente a 200 essais pour faciliter la recherche de place
                if i < n_on_path:  # le long du segment depart -> but
                    t = self.rng.uniform(0.25, 0.75)
                    perp = np.array([-seg[1], seg[0]])
                    perp /= (np.linalg.norm(perp) + 1e-8)
                    pos = start + t * seg + perp * self.rng.uniform(-1.2, 1.2)
                else:              # eparpilles dans l'arene
                    pos = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)

                ok = (self._free(pos, r, self.obstacles, clearance=min_pass_gap)
                      and np.linalg.norm(pos - start) > r + self.R_ROBOT + 1.0
                      and np.linalg.norm(pos - goal) > r + self.GOAL_RADIUS + 0.6)
                if ok:
                    break
            else:
                continue  # place introuvable -> on saute cet obstacle
            color = [0.9, 0.2, 0.2, 1] if dynamic else [0.4, 0.3, 0.2, 1]
            oid = self._make_cylinder(r, 0.6, color)
            p.resetBasePositionAndOrientation(oid, [*pos, 0.3], [0, 0, 0, 1],
                                              physicsClientId=self.cid)
            vel = np.zeros(2)
            if dynamic:
                ang = self.rng.uniform(0, 2 * np.pi)
                s = self.rng.uniform(*speed_rng)
                vel = s * np.array([np.cos(ang), np.sin(ang)])
            self.obstacles.append(dict(id=oid, pos=pos, vel=vel, r=r, dynamic=dynamic,
                                       perturb_prob=perturb_prob if dynamic else 0.0))

    # ---------- dynamique des obstacles : rebonds + perturbation aleatoire ----------
    def _move_obstacles(self):
        lim = self.ARENA - 0.15
        for o in self.obstacles:
            if not o["dynamic"]:
                continue
            # mouvement erratique type "pieton" : changement de direction aleatoire
            if o["perturb_prob"] > 0 and self.rng.random() < o["perturb_prob"] * self.DT:
                speed = np.linalg.norm(o["vel"])
                ang = self.rng.uniform(0, 2 * np.pi)
                o["vel"] = speed * np.array([np.cos(ang), np.sin(ang)])
            o["pos"] = o["pos"] + o["vel"] * self.DT
            for k in range(2):  # rebond frontieres
                if abs(o["pos"][k]) > lim - o["r"]:
                    o["vel"][k] *= -1
                    o["pos"][k] = np.clip(o["pos"][k], -(lim - o["r"]), lim - o["r"])
        # rebond obstacle <-> obstacle (statique ou dynamique)
        for i, a in enumerate(self.obstacles):
            if not a["dynamic"]:
                continue
            for j, b in enumerate(self.obstacles):
                if i == j:
                    continue
                d = a["pos"] - b["pos"]
                dist = np.linalg.norm(d)
                min_d = a["r"] + b["r"]
                if dist < min_d and dist > 1e-6:
                    n = d / dist
                    if np.dot(a["vel"], n) < 0:          # se dirige vers b -> reflexion
                        a["vel"] = a["vel"] - 2 * np.dot(a["vel"], n) * n
                    a["pos"] = b["pos"] + n * min_d       # separation
        for o in self.obstacles:
            if o["dynamic"]:
                p.resetBasePositionAndOrientation(o["id"], [*o["pos"], 0.3],
                                                  [0, 0, 0, 1], physicsClientId=self.cid)

    # ---------- lidar egocentrique ----------
    def _lidar(self):
        angles = self.theta + np.linspace(0, 2 * np.pi, self.N_RAYS, endpoint=False)
        starts, ends = [], []
        for a in angles:
            d = np.array([np.cos(a), np.sin(a)])
            s = self.pos + d * (self.R_ROBOT + 0.05)
            starts.append([*s, 0.2])
            ends.append([*(s + d * self.RAY_RANGE), 0.2])
        res = p.rayTestBatch(starts, ends, physicsClientId=self.cid)
        return np.array([r[2] for r in res], dtype=np.float32)  # fractions [0,1]

    # ---------- features obstacles dynamiques (pos + vitesse relatives, repere robot) ----------
    def _dynamic_obstacle_features(self):
        dyn = [o for o in self.obstacles if o["dynamic"]]
        dyn.sort(key=lambda o: np.linalg.norm(o["pos"] - self.pos))
        c, s = math.cos(-self.theta), math.sin(-self.theta)
        R = np.array([[c, -s], [s, c]])   # rotation monde -> repere robot
        feats = []
        for o in dyn[:self.K_NEAREST]:
            rel_pos = R @ (o["pos"] - self.pos)
            rel_vel = R @ o["vel"]
            feats.extend([
                float(np.clip(rel_pos[0] / self.RAY_RANGE, -1, 1)),
                float(np.clip(rel_pos[1] / self.RAY_RANGE, -1, 1)),
                float(np.clip(rel_vel[0] / self.V_OBS_NORM, -1, 1)),
                float(np.clip(rel_vel[1] / self.V_OBS_NORM, -1, 1)),
            ])
        while len(feats) < self.K_NEAREST * 4:   # zero-padding si moins de K obstacles dynamiques
            feats.append(0.0)
        return feats

    def _obs(self):
        to_goal = self.goal - self.pos
        d = np.linalg.norm(to_goal)
        ang = math.atan2(to_goal[1], to_goal[0]) - self.theta
        base = [min(d / (2 * self.ARENA * 1.5), 1.0), np.cos(ang), np.sin(ang),
               self.last_a[0], self.last_a[1]]
        return np.concatenate([
            base, self._dynamic_obstacle_features(), self._lidar()
        ]).astype(np.float32)

    # ---------- penalite time-to-collision (vitesse de rapprochement reelle) ----------
    def _ttc_penalty(self, v_robot_world, ttc_threshold=2.5, weight=0.4):
        """S'applique uniformement aux obstacles statiques (vel=0, la vitesse de
        rapprochement vient alors du seul mouvement du robot) et dynamiques."""
        penalty = 0.0
        for o in self.obstacles:
            rel_pos = o["pos"] - self.pos
            dist_surface = np.linalg.norm(rel_pos) - self.R_ROBOT - o["r"]
            if dist_surface <= 0:
                continue  # gere separement par la detection de collision
            rel_vel = o["vel"] - v_robot_world
            closing = -np.dot(rel_vel, rel_pos) / (np.linalg.norm(rel_pos) + 1e-6)
            if closing > 1e-3:
                ttc = dist_surface / closing
                if ttc < ttc_threshold:
                    penalty += weight * (ttc_threshold - ttc) / ttc_threshold
        return penalty

    # ---------- API Gym ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        prm = self.curr.params
        # depart / but : eloignes (80%), proches toleres (20%)
        min_goal = prm["min_goal"] if self.rng.random() < 0.8 else 2.5
        for _ in range(200):
            self.pos = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)
            self.goal = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)
            if np.linalg.norm(self.goal - self.pos) >= min_goal:
                break
        self.theta = float(self.rng.uniform(-np.pi, np.pi))
        n_s = int(self.rng.integers(prm["n_static"][0], prm["n_static"][1] + 1))
        n_d = int(self.rng.integers(prm["n_dyn"][0], prm["n_dyn"][1] + 1))
        self._spawn_obstacles(n_s, n_d, prm["speed"], prm["perturb_prob"], self.pos, self.goal)
        p.resetBasePositionAndOrientation(self.robot_id, [*self.pos, 0.15],
                                          p.getQuaternionFromEuler([0, 0, self.theta]),
                                          physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.goal_vis, [*self.goal, 0.01],
                                          [0, 0, 0, 1], physicsClientId=self.cid)
        self.prev_d = np.linalg.norm(self.goal - self.pos)
        self.steps = 0
        self.last_a = np.zeros(2, dtype=np.float32)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        v = (action[0] + 1) / 2 * self.V_MAX
        w = action[1] * self.W_MAX
        self.theta = math.atan2(math.sin(self.theta + w * self.DT),
                                math.cos(self.theta + w * self.DT))
        self.pos = self.pos + v * self.DT * np.array([np.cos(self.theta),
                                                      np.sin(self.theta)])
        v_robot_world = v * np.array([math.cos(self.theta), math.sin(self.theta)])
        lim = self.ARENA - 0.15 - self.R_ROBOT
        wall_hit = np.max(np.abs(self.pos)) > lim
        self.pos = np.clip(self.pos, -lim, lim)
        self._move_obstacles()
        p.resetBasePositionAndOrientation(self.robot_id, [*self.pos, 0.15],
                                          p.getQuaternionFromEuler([0, 0, self.theta]),
                                          physicsClientId=self.cid)
        self.last_a = action
        self.steps += 1

        d = np.linalg.norm(self.goal - self.pos)
        reward = 20.0 * (self.prev_d - d) - 0.05          # progression + temps
        self.prev_d = d
        obs = self._obs()
        min_ray = float(np.min(obs[self.OBS_BASE:])) * self.RAY_RANGE
        if min_ray < 0.5:                                  # zone de danger (marge generale)
            reward -= 0.15 * (0.5 - min_ray) / 0.5
        reward -= self._ttc_penalty(v_robot_world)          # terme specifique "se rapproche vite"

        terminated, truncated, success = False, False, False
        collision = wall_hit or any(
            np.linalg.norm(self.pos - o["pos"]) < self.R_ROBOT + o["r"]
            for o in self.obstacles)
        if collision:
            reward -= 60.0
            terminated = True
        elif d < self.GOAL_RADIUS:
            reward += 100.0
            terminated, success = True, True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        if terminated or truncated:
            self.curr.report(success)
        if self.render_mode == "human":
            import time; time.sleep(self.DT)
        return obs, float(reward), terminated, truncated, {"success": success ,
                                                             "level": self.curr.level,   # <- necessaire pour observer la progression depuis le process principal
                                                           }

    def close(self):
        p.disconnect(physicsClientId=self.cid)

# ----------------------- Callback de suivi -----------------------
# ----------------------- Callback de suivi (corrige) -----------------------
from collections import deque

class CurriculumLogCallback(BaseCallback):
    """Chaque worker a sa PROPRE instance de CurriculumManager (cf. train()) : il n'y a
    plus de niveau global unique a lire depuis le process principal. On agrege donc
    ce que chaque worker rapporte via son dict `info` a chaque step."""

    def __init__(self, log_every=2000, global_window=200):
        super().__init__()
        self.log_every = log_every
        self.global_outcomes = deque(maxlen=global_window)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        levels = [info["level"] for info in infos if "level" in info]
        if levels:
            self.logger.record("curriculum/level_min", float(np.min(levels)))
            self.logger.record("curriculum/level_mean", float(np.mean(levels)))
            self.logger.record("curriculum/level_max", float(np.max(levels)))

        for done, info in zip(dones, infos):
            if done and "success" in info:
                self.global_outcomes.append(1.0 if info["success"] else 0.0)
        if self.global_outcomes:
            self.logger.record("curriculum/global_success_rate",
                               float(np.mean(self.global_outcomes)))

        if self.n_calls % self.log_every == 0 and levels:
            print(f"steps={self.num_timesteps} "
                  f"niveaux(min/moy/max)={np.min(levels):.0f}/{np.mean(levels):.1f}/{np.max(levels):.0f} "
                  f"succes_global(~{len(self.global_outcomes)}ep)="
                  f"{np.mean(self.global_outcomes) if self.global_outcomes else 0:.2f}")
        return True

# ----------------------- Entrainement / test -----------------------

def resolve_model_path(model_path):
    """Ajoute l'extension .zip si elle n'est pas spécifiée dans le chemin."""
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path

def load_sac(model_path):
    """
    Charge le modèle SAC en gérant le bogue de lecture PyTorchStreamReader
    via la lecture en mémoire (BytesIO).
    """

    model_path = resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle est introuvable : {model_path}")

    try:
        return SAC.load(model_path)
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
            return SAC.load(model_path)
        finally:
            torch.load = original_torch_load

# ----------------------- train() corrige -----------------------
def make_env():
    def _init():
        return NavEnv(CurriculumManager())   # <- instance INDEPENDANTE par worker, pas partagee
    return _init

def train():
    N_ENVS = 8
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(N_ENVS)]))
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=64, gradient_steps=64,   # <- corrige aussi (etait reste a 1)
        learning_starts=10_000, ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1, tensorboard_log="./tb_nav",
    )
    model.learn(total_timesteps=4_000_000, callback=CurriculumLogCallback())
    model.save("sac_nav_final_64")
    env.close()

def play():
    curr = CurriculumManager()
    curr.level = len(CurriculumManager.LEVELS) - 1   # test en pleine difficulte
    env = NavEnv(curr, render_mode="human")
    model = load_sac("sac_nav_final_64")
    for ep in range(20):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        print(f"Episode {ep}: reward={total:.1f} succes={info['success']}")
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
    model = load_sac("sac_nav_final_64")
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
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--level", type=int, default=None,
                    help="Niveau de curriculum a evaluer (defaut: dernier niveau)")
    args = ap.parse_args()
    if args.play:
        play()
    elif args.eval:
        evaluate(level=args.level)
    else:
        train()