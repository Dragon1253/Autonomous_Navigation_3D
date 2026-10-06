"""
Navigation RL avec PyBullet : point A -> B, obstacles statiques + dynamiques,
curriculum adaptatif, PPO (Stable-Baselines3).

v5 - remplace la representation d'obstacles de v4 (K_NEAREST=4 obstacles
dynamiques concatenes, tries par distance) par un encodeur "deep sets" :

- L'observation contient desormais TOUS les obstacles (statiques ET
  dynamiques, jusqu'a MAX_OBSTACLES, zero-padded) avec un masque de
  presence, au lieu de seulement les 4 dynamiques les plus proches.
  Ca supprime la troncature qui laissait jusqu'a ~9 obstacles dynamiques
  sans info de position/vitesse aux paliers les plus difficiles (n_dyn
  jusqu'a 13, K_NEAREST=4).
- Chaque obstacle est passe dans un MLP partage (memes poids pour tous les
  slots), puis les embeddings sont agreges par pooling masque (mean+max,
  ou attention -- au choix via `pooling=`). Cette representation est
  invariante par permutation : contrairement a la concatenation triee de
  v4, un changement d'ordre entre deux obstacles proches en distance ne
  provoque plus de discontinuite dans les features.
- SAC -> PPO : plus rapide (pas de replay buffer / gradient_steps), et
  s'accommode bien du parallelisme SubprocVecEnv.

ATTENTION COMPATIBILITE : l'espace d'observation a completement change
(plus de K_NEAREST, obstacles complets + masque) et l'algorithme aussi
(PPO au lieu de SAC). Aucun checkpoint precedent (sac_nav_v4 etc.) n'est
compatible : reentrainement complet necessaire (model.save("ppo_nav_v5")).

Pour passer du pooling mean+max a l'attention : changer uniquement
`pooling="meanmax"` en `pooling="attention"` dans policy_kwargs (train()),
et reprendre l'entrainement depuis zero (l'architecture change).

Entraînement : python ppo_train_v5_deepsets.py
Test visuel  : python ppo_train_v5_deepsets.py --play
Evaluation   : python ppo_train_v5_deepsets.py --eval [--level N]
"""
import argparse
import csv
import io
import json
import math
import os
import zipfile
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ----------------------- Curriculum adaptatif -----------------------
class CurriculumManager:
    """Ne decide plus rien tout seul : tient juste le niveau courant et
    expose les parametres associes. Le niveau est pilote de facon
    centralisee par SyncedCurriculumCallback (cf plus bas), qui agrege le
    succes de TOUS les workers et diffuse le niveau via `set_level` /
    `env_method`."""
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
        dict(n_static=(18, 25), n_dyn=(10, 13), speed=(0.75, 0.9), perturb_prob=0.25,  min_goal=6.0),
        dict(n_static=(20, 30), n_dyn=(10, 15), speed=(0.7, 0.8), perturb_prob=0.25,  min_goal=6.0),  # objectif
    ]

    def __init__(self):
        self.level = 0

    @property
    def params(self):
        return self.LEVELS[self.level]

    def set_level(self, level):
        """Appele a distance par SyncedCurriculumCallback via env_method."""
        self.level = int(np.clip(level, 0, len(self.LEVELS) - 1))
        return self.level



def ensure_warehouse_floor_texture(path="assets/warehouse_floor.png", size=1024):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = np.random.default_rng(0).normal(150, 4, (size, size, 1))
    arr = np.clip(np.repeat(base, 3, axis=2), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    step = size // 6
    for i in range(0, size + 1, step):
        draw.line([(i, 0), (i, size)], fill=(120, 120, 120), width=2)
        draw.line([(0, i), (size, i)], fill=(120, 120, 120), width=2)
    band, cx = size // 14, size // 2
    draw.rectangle([cx - band, 0, cx - band + 10, size], fill=(225, 175, 30))
    draw.rectangle([cx + band - 10, 0, cx + band, size], fill=(225, 175, 30))
    img.save(path)
    return path
def ensure_drum_texture(path="assets/drum_band.png", size=256):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (size, size), (150, 95, 40))
    draw = ImageDraw.Draw(img)
    for y in (int(size * 0.15), int(size * 0.5), int(size * 0.85)):
        draw.rectangle([0, y - 6, size, y + 6], fill=(40, 40, 40))
    img.save(path)
    return path



# ----------------------- Environnement -----------------------
class NavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ARENA = 6.0          # demi-taille de l'arene (12 x 12 m)
    R_ROBOT = 0.25
    V_MAX = 1.0          # m/s, marche avant
    V_MAX_REV = 0.6      # m/s, marche arriere
    W_MAX = 2.0          # rad/s
    DT = 1.0 / 30.0
    N_RAYS = 24
    RAY_RANGE = 5.0
    GOAL_RADIUS = 0.45
    MAX_STEPS = 1300
    PASSAGE_MARGIN = 0.30  # marge de passage garantie entre 2 obstacles, en plus de 2*R_ROBOT

    MAX_OBSTACLES = 40    # >= max(n_static)+max(n_dyn) sur le curriculum ci-dessus (25+13=38), marge incluse
    OBS_FEAT_DIM = 6      # [rel_x, rel_y, rel_vx, rel_vy, rayon_norm, flag_dynamique] par obstacle
    OBS_POS_NORM = 10.0   # normalisation position relative (ordre de grandeur de l'arene)
    V_OBS_NORM = 2.0       # normalisation des vitesses relatives (robot + obstacle <= ~2 m/s)

    def __init__(self, curriculum: CurriculumManager, render_mode=None):
        super().__init__()
        self.curr = curriculum
        self.render_mode = render_mode
        self.cid = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        # dans NavEnv.__init__, juste après self.cid = p.connect(...)
        if render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)       # cache la UI de debug
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)   # ombres portées
            p.resetDebugVisualizerCamera(cameraDistance=9.0, cameraYaw=55, cameraPitch=-38,
                                        cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        tex_id = p.loadTexture(ensure_warehouse_floor_texture(), physicsClientId=self.cid)
        p.changeVisualShape(plane_id, -1, textureUniqueId=tex_id, physicsClientId=self.cid)
        self._build_walls()
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.R_ROBOT, height=0.3,
                                    physicsClientId=self.cid)
        self.robot_id = p.createMultiBody(0, col, self._make_robot_visual(), [0, 0, 0.15],
                                        physicsClientId=self.cid)
        # dans __init__, juste après la création du robot :
        self._drum_tex_id = p.loadTexture(ensure_drum_texture(), physicsClientId=self.cid)
        self._beacon_vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.06,
                                                rgbaColor=[1.0, 0.55, 0.0, 1],
                                                physicsClientId=self.cid)
        self.goal_vis = self._make_cylinder(self.GOAL_RADIUS, 0.02, [0.1, 0.9, 0.1, 0.6],
                                                    collision=False)
        self.obstacles = []   # dicts: id, pos(2), vel(2), r, dynamic, perturb_prob

        obs_dim = (5 + self.MAX_OBSTACLES * self.OBS_FEAT_DIM + self.MAX_OBSTACLES
                   + 1 + self.N_RAYS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self._last_danger = 0.0

    # ---------- construction ----------
    def _make_robot_visual(self):
        return p.createVisualShapeArray(
        shapeTypes=[p.GEOM_CYLINDER, p.GEOM_CYLINDER, p.GEOM_BOX],
        radii=[self.R_ROBOT, 0.035, 0],
        lengths=[0.30, 0.18, 0],
        halfExtents=[[0, 0, 0], [0, 0, 0], [0.07, 0.05, 0.03]],
        visualFramePositions=[[0, 0, 0], [0, 0, 0.24], [self.R_ROBOT * 0.85, 0, 0.05]],
        visualFrameOrientations=[[0, 0, 0, 1]] * 3,
        rgbaColors=[[0.10, 0.35, 0.85, 1], [0.05, 0.05, 0.05, 1], [1.0, 0.75, 0.0, 1]],
        physicsClientId=self.cid,
    )
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
            # dans _build_walls, remplace la rgbaColor du mur
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[ex, ey, h],
                                    rgbaColor=[0.55, 0.58, 0.62, 1], physicsClientId=self.cid)
            p.createMultiBody(0, col, vis, [cx, cy, h], physicsClientId=self.cid)

    # ---------- placement sans chevauchement ----------
    def _free(self, pos, r, others, clearance=None):
        if clearance is None:
            clearance = 2 * self.R_ROBOT + self.PASSAGE_MARGIN
        if np.max(np.abs(pos)) > self.ARENA - r - 0.15:
            return False
        for o in others:
            if np.linalg.norm(pos - o["pos"]) < r + o["r"] + clearance:
                return False
        return True

    def _spawn_obstacles(self, n_static, n_dyn, speed_rng, perturb_prob, start, goal):
        # tout en haut de _spawn_obstacles, dans la boucle de nettoyage :
        for o in self.obstacles:
            p.removeBody(o["id"], physicsClientId=self.cid)
            if o.get("beacon_id") is not None:
                p.removeBody(o["beacon_id"], physicsClientId=self.cid)
        self.obstacles = []
        seg = goal - start
        total = n_static + n_dyn
        n_on_path = max(1, int(0.4 * total))

        min_pass_gap = 2 * self.R_ROBOT + self.PASSAGE_MARGIN

        for i in range(total):
            dynamic = i >= n_static
            r = float(self.rng.uniform(0.25, 0.5))
            for _ in range(200):
                if i < n_on_path:
                    t = self.rng.uniform(0.25, 0.75)
                    perp = np.array([-seg[1], seg[0]])
                    perp /= (np.linalg.norm(perp) + 1e-8)
                    pos = start + t * seg + perp * self.rng.uniform(-1.2, 1.2)
                else:
                    pos = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)

                ok = (self._free(pos, r, self.obstacles, clearance=min_pass_gap)
                      and np.linalg.norm(pos - start) > r + self.R_ROBOT + 1.0
                      and np.linalg.norm(pos - goal) > r + self.GOAL_RADIUS + 0.6)
                if ok:
                    break
            else:
                continue
            color = [0.9, 0.2, 0.2, 1] if dynamic else [0.4, 0.3, 0.2, 1]
            # dans _spawn_obstacles, remplace TOUT le bloc depuis "if not dynamic:" (texture)
            # jusqu'à l'append final par ceci :
            oid = self._make_cylinder(r, 0.6, color)
            if not dynamic:
                p.changeVisualShape(oid, -1, textureUniqueId=self._drum_tex_id, rgbaColor=[1, 1, 1, 1],
                                    physicsClientId=self.cid)
            p.resetBasePositionAndOrientation(oid, [*pos, 0.3], [0, 0, 0, 1],
                                            physicsClientId=self.cid)

            vel = np.zeros(2)
            beacon_id = None
            if dynamic:
                ang = self.rng.uniform(0, 2 * np.pi)          # <- remise en place : vitesse aléatoire
                s = self.rng.uniform(*speed_rng)
                vel = s * np.array([np.cos(ang), np.sin(ang)])
                beacon_id = p.createMultiBody(0, -1, self._beacon_vis_id, [*pos, 0.6 + 0.12],
                                            physicsClientId=self.cid)

            self.obstacles.append(dict(id=oid, pos=pos, vel=vel, r=r, dynamic=dynamic,
                                    perturb_prob=perturb_prob if dynamic else 0.0,
                                    beacon_id=beacon_id))

    # ---------- dynamique des obstacles : rebonds + perturbation aleatoire ----------
    def _move_obstacles(self):
        lim = self.ARENA - 0.15
        for o in self.obstacles:
            if not o["dynamic"]:
                continue
            if o["perturb_prob"] > 0 and self.rng.random() < o["perturb_prob"] * self.DT:
                speed = np.linalg.norm(o["vel"])
                ang = self.rng.uniform(0, 2 * np.pi)
                o["vel"] = speed * np.array([np.cos(ang), np.sin(ang)])
            o["pos"] = o["pos"] + o["vel"] * self.DT
            for k in range(2):
                if abs(o["pos"][k]) > lim - o["r"]:
                    o["vel"][k] *= -1
                    o["pos"][k] = np.clip(o["pos"][k], -(lim - o["r"]), lim - o["r"])
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
                    if np.dot(a["vel"], n) < 0:
                        a["vel"] = a["vel"] - 2 * np.dot(a["vel"], n) * n
                    a["pos"] = b["pos"] + n * min_d
        # dans _move_obstacles, à la fin (remplace la boucle qui repositionne les dynamiques) :
        for o in self.obstacles:
            if not o["dynamic"]:
                continue
            p.resetBasePositionAndOrientation(o["id"], [*o["pos"], 0.3], [0, 0, 0, 1],
                                            physicsClientId=self.cid)
            if self.render_mode == "human":
                p.resetBasePositionAndOrientation(o["beacon_id"], [*o["pos"], 0.72],
                                                [0, 0, 0, 1], physicsClientId=self.cid)
                blink_on = (self.steps // 5) % 2 == 0
                color = [1.0, 0.55, 0.0, 1] if blink_on else [0.35, 0.2, 0.0, 1]
                p.changeVisualShape(o["beacon_id"], -1, rgbaColor=color, physicsClientId=self.cid)

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
        return np.array([r[2] for r in res], dtype=np.float32)

    # ---------- (v5) features "deep sets" : TOUS les obstacles + masque ----------
    def _obstacle_set_features(self):
        c, s = math.cos(-self.theta), math.sin(-self.theta)
        R = np.array([[c, -s], [s, c]])

        items = []
        for o in self.obstacles:
            rel_pos = R @ (o["pos"] - self.pos)
            rel_vel = R @ o["vel"]
            items.append((np.linalg.norm(rel_pos), rel_pos, rel_vel, o["r"], o["dynamic"]))
        items.sort(key=lambda t: t[0])
        items = items[:self.MAX_OBSTACLES]

        feats = np.zeros(self.MAX_OBSTACLES * self.OBS_FEAT_DIM, dtype=np.float32)
        mask = np.zeros(self.MAX_OBSTACLES, dtype=np.float32)
        for i, (_, rel_pos, rel_vel, r, dyn) in enumerate(items):
            base_i = i * self.OBS_FEAT_DIM
            feats[base_i + 0] = np.clip(rel_pos[0] / self.OBS_POS_NORM, -1, 1)
            feats[base_i + 1] = np.clip(rel_pos[1] / self.OBS_POS_NORM, -1, 1)
            feats[base_i + 2] = np.clip(rel_vel[0] / self.V_OBS_NORM, -1, 1)
            feats[base_i + 3] = np.clip(rel_vel[1] / self.V_OBS_NORM, -1, 1)
            feats[base_i + 4] = np.clip((r - 0.25) / 0.25, 0.0, 1.0)
            feats[base_i + 5] = 1.0 if dyn else -1.0
            mask[i] = 1.0
        return feats, mask

    def _obs(self):
        to_goal = self.goal - self.pos
        d = np.linalg.norm(to_goal)
        ang = math.atan2(to_goal[1], to_goal[0]) - self.theta
        base = [min(d / (2 * self.ARENA * 1.5), 1.0), np.cos(ang), np.sin(ang),
               self.last_a[0], self.last_a[1]]
        obst_feats, obst_mask = self._obstacle_set_features()
        return np.concatenate([
            obst_feats, obst_mask, base, [self._last_danger], self._lidar()
        ]).astype(np.float32)

    # ---------- TTC (time-to-collision) : penalite reward + scalaire de danger observable ----------
    def _ttc_features(self, v_robot_world, ttc_threshold=1.8, weight=0.4, cpa_margin=0.15):
        penalty = 0.0
        worst_ratio = 0.0
        for o in self.obstacles:
            rel_pos = o["pos"] - self.pos
            rel_vel = o["vel"] - v_robot_world
            vv = np.dot(rel_vel, rel_vel)
            if vv < 1e-6:
                continue
            t_cpa = max(-np.dot(rel_pos, rel_vel) / vv, 0.0)
            if t_cpa > ttc_threshold:
                continue
            closest = rel_pos + rel_vel * t_cpa
            cpa_dist = np.linalg.norm(closest) - self.R_ROBOT - o["r"]
            if cpa_dist < cpa_margin:
                urgency = 1.0 - t_cpa / ttc_threshold
                severity = np.clip((cpa_margin - cpa_dist) / cpa_margin, 0.0, 1.0)
                ratio = urgency * severity
                penalty += weight * ratio
                worst_ratio = max(worst_ratio, ratio)
        return penalty, float(np.clip(worst_ratio, 0.0, 1.0))

    # ---------- API Gym ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        prm = self.curr.params
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
        self._last_danger = 0.0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        v = action[0] * self.V_MAX if action[0] >= 0 else action[0] * self.V_MAX_REV
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
        to_goal_dir = (self.goal - self.pos) / (d + 1e-6)   # d = distance APRES déplacement
        vel_toward_goal = np.dot(v_robot_world, to_goal_dir)
        progress = 10.0 * vel_toward_goal * self.DT
        if progress < 0:
            progress *= 0.3   # s'éloigner temporairement (contourner, reculer) coûte moins cher qu'avant
        reward = progress - 0.05
        self.prev_d = d

        ttc_penalty, danger_scalar = self._ttc_features(v_robot_world)
        self._last_danger = danger_scalar

        obs = self._obs()
        min_ray = float(np.min(obs[-self.N_RAYS:])) * self.RAY_RANGE
        if min_ray < 0.4:
            eps = 0.05
            prox_penalty = 0.05 * (1.0 / (min_ray + eps) - 1.0 / (0.5 + eps))
            reward -= min(prox_penalty, 2.0)
        reward -= ttc_penalty

        terminated, truncated, success = False, False, False
        collision = wall_hit or any(
            np.linalg.norm(self.pos - o["pos"]) < self.R_ROBOT + o["r"]
            for o in self.obstacles)
        if collision:
            reward -= 80.0
            terminated = True
        elif d < self.GOAL_RADIUS:
            reward += 100.0
            terminated, success = True, True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        if self.render_mode == "human":
            import time; time.sleep(self.DT)
        return obs, float(reward), terminated, truncated, {"success": success,
                                                             "level": self.curr.level}

    def set_curriculum_level(self, level):
        return self.curr.set_level(level)

    def close(self):
        p.disconnect(physicsClientId=self.cid)


# ----------------------- Extracteur de features "deep sets" -----------------------
class ObstacleSetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_obstacles, obs_feat_dim,
                 obstacle_embed_dim=64, context_hidden=128,
                 pooling="meanmax", features_dim=256):
        super().__init__(observation_space, features_dim)
        assert pooling in ("meanmax", "attention")
        self.max_obstacles = max_obstacles
        self.obs_feat_dim = obs_feat_dim
        self.pooling = pooling

        total_dim = observation_space.shape[0]
        self.obstacle_block_size = max_obstacles * obs_feat_dim
        self.mask_block_size = max_obstacles
        self.context_size = total_dim - self.obstacle_block_size - self.mask_block_size
        assert self.context_size > 0, "observation_space incoherent avec max_obstacles/obs_feat_dim"

        self.obstacle_encoder = nn.Sequential(
            nn.Linear(obs_feat_dim, obstacle_embed_dim), nn.ReLU(),
            nn.Linear(obstacle_embed_dim, obstacle_embed_dim), nn.ReLU(),
        )

        if pooling == "attention":
            self.attn_query = nn.Linear(obstacle_embed_dim, 1)
            pooled_dim = obstacle_embed_dim
        else:
            pooled_dim = obstacle_embed_dim * 2

        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_size, context_hidden), nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(pooled_dim + context_hidden, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        bsz = observations.shape[0]
        obst = observations[:, :self.obstacle_block_size].reshape(
            bsz, self.max_obstacles, self.obs_feat_dim)
        mask_f = observations[:, self.obstacle_block_size:
                               self.obstacle_block_size + self.mask_block_size]
        context = observations[:, self.obstacle_block_size + self.mask_block_size:]

        embed = self.obstacle_encoder(obst)
        mask_f = mask_f.unsqueeze(-1)
        mask_b = mask_f > 0.5

        if self.pooling == "attention":
            scores = self.attn_query(embed).squeeze(-1)
            scores = scores.masked_fill(~mask_b.squeeze(-1), float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            pooled = (embed * weights).sum(dim=1)
        else:
            max_input = embed.masked_fill(~mask_b, -1e4)
            max_pool = max_input.max(dim=1).values
            max_pool = torch.nan_to_num(max_pool, neginf=0.0)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            mean_pool = (embed * mask_f).sum(dim=1) / denom
            pooled = torch.cat([mean_pool, max_pool], dim=1)

        ctx = self.context_encoder(context)
        return self.fusion(torch.cat([pooled, ctx], dim=1))


# ----------------------- Callbacks -----------------------
from collections import deque

class LevelBestModelCallback(BaseCallback):
    """Enregistre le meilleur modèle pour chaque niveau de curriculum en fonction du TAUX DE
    SUCCÈS moyen sur une fenêtre glissante (même métrique que celle tracée dans les courbes
    d'entrainement / SyncedCurriculumCallback), avec la récompense moyenne comme simple
    critère de départage. Se réinitialise à chaque changement de niveau.

    NB: la version précédente sélectionnait le "meilleur" modèle sur la récompense moyenne
    seule. Récompense et succès ne sont pas strictement corrélés ici (pénalités de temps/
    proximité, comportements prudents qui allongent l'épisode sans échouer, etc.), et avec
    une fenêtre courte (window_size=20 à l'appel) répartie sur plusieurs workers, un pic de
    récompense précoce et bruité pouvait verrouiller `best_mean_reward` pour le reste du
    palier (la comparaison stricte `>` ne se réinitialise qu'au changement de niveau) — sans
    jamais être rebattu même si le taux de succès réel grimpait ensuite à 90%. C'est
    exactement le symptôme observé : best_model (verrouillé tôt, ~80% de succès réel) pire
    que le modèle final (~85%, lui-même un simple instantané de fin d'entrainement, potentiellement
    déjà redescendu depuis le pic de 90%)."""
    def __init__(self, save_path="best_models", window_size=100, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.window_size = window_size
        self.check_freq = check_freq
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_successes = deque(maxlen=window_size)
        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.current_level = 0
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # Récupère le niveau actuel : moyenne sur TOUS les workers (et non infos[0] seul),
        # pour éviter un désync avec SubprocVecEnv juste après un changement de palier.
        levels_now = [info["level"] for info in infos if "level" in info]
        if levels_now:
            level = int(np.round(np.mean(levels_now)))
            if level != self.current_level:
                self.current_level = level
                self.best_success_rate = -np.inf
                self.best_mean_reward = -np.inf
                self.episode_rewards.clear()
                self.episode_successes.clear()
                if self.verbose > 0:
                    print(f"\n[BestModelCallback] Passage au niveau {self.current_level}. Réinitialisation.")

        # Enregistre récompenses ET succès de fin d'épisode
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if done and "success" in info:
                self.episode_successes.append(1.0 if info["success"] else 0.0)

        # Vérifie et sauvegarde si le taux de succès s'est amélioré (récompense = départage)
        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) >= self.window_size:
            success_rate = float(np.mean(self.episode_successes))
            mean_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else -np.inf

            improved = (success_rate > self.best_success_rate or
                        (success_rate == self.best_success_rate and mean_reward > self.best_mean_reward))

            if improved:
                self.best_success_rate = success_rate
                self.best_mean_reward = mean_reward
                model_name = f"best_model_level_{self.current_level}"
                full_path = os.path.join(self.save_path, model_name)
                self.model.save(full_path)
                if self.verbose > 0:
                    print(f"[BestModelCallback] Nouveau meilleur modèle au niveau {self.current_level} "
                          f"succès={success_rate:.1%} récompense_moy={mean_reward:.2f} -> Sauvégardé dans {full_path}.zip")

        return True


class SyncedCurriculumCallback(BaseCallback):
    WINDOW = 100
    THRESHOLD = 0.8
    MIN_STEPS_PER_LEVEL = 20_000
    MAX_STEPS_PER_LEVEL = 1_000_000

    def __init__(self, log_every=2000):
        super().__init__()
        self.log_every = log_every
        self.n_levels = len(CurriculumManager.LEVELS)
        self.level = 0
        self.level_start_step = 0
        self.global_outcomes = deque(maxlen=self.WINDOW)

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

        success_rate = float(np.mean(self.global_outcomes)) if self.global_outcomes else 0.0
        elapsed = self.num_timesteps - self.level_start_step
        self.logger.record("curriculum/global_success_rate", success_rate)
        self.logger.record("curriculum/global_level", float(self.level))
        self.logger.record("curriculum/steps_since_level_start", float(elapsed))

        if self.level < self.n_levels - 1:
            hit_ceiling = elapsed >= self.MAX_STEPS_PER_LEVEL
            hit_gate = (elapsed >= self.MIN_STEPS_PER_LEVEL
                        and len(self.global_outcomes) == self.WINDOW
                        and success_rate >= self.THRESHOLD)
            if hit_ceiling or hit_gate:
                self.level += 1
                self.level_start_step = self.num_timesteps
                self.global_outcomes.clear()
                self.training_env.env_method("set_curriculum_level", self.level)
                reason = "plafond de steps" if hit_ceiling and not hit_gate else "seuil de succes"
                print(f"\n=== CURRICULUM: passage GLOBAL au niveau {self.level} "
                      f"({reason}, apres {elapsed} steps a ce palier) ===\n")

        if self.n_calls % self.log_every == 0 and levels:
            print(f"steps={self.num_timesteps} niveau_global={self.level} "
                  f"niveaux_workers(min/moy/max)={np.min(levels):.0f}/{np.mean(levels):.1f}/{np.max(levels):.0f} "
                  f"succes_global(~{len(self.global_outcomes)}ep)={success_rate:.2f}")
        return True


class CurriculumAwareEvalCallback(BaseCallback):
    """Évaluation déterministe (sans bruit d'exploration) sur un env dédié, CONSCIENTE du
    curriculum. Objectif : éviter le piège d'un EvalCallback classique dans un cadre à
    curriculum, où comparer un score brut entre niveaux n'a aucun sens — un niveau plus
    facile aura structurellement un taux de succès plus haut qu'un niveau difficile, donc
    un simple "argmax(success_rate) sur toute la durée de l'entrainement" reste bloqué sur
    le meilleur score obtenu au niveau 0 (ex. 95%) même s'il ne sait rien faire au niveau 10.

    Deux mécanismes complémentaires :

    1. `best_per_level` : meilleur modèle PAR niveau (comparable seulement entre lui-même),
       basé sur une évaluation déterministe (policy sans exploration) sur `n_eval_episodes`
       épisodes à seeds fixes -> signal beaucoup plus stable que les rollouts d'entrainement.

    2. `best_overall` : sélection par ordre LEXICOGRAPHIQUE (niveau, taux_de_succès), mais en
       ne comptant un niveau comme "atteint" que si le taux de succès dépasse
       `min_success_to_count` (ex. 0.5). Un modèle qui réussit à un niveau plus difficile
       bat TOUJOURS un modèle plus fort sur un niveau plus facile, MAIS on n'accepte un
       nouveau niveau comme référence que s'il est déjà réellement compétent dessus — pas
       juste "vient d'y arriver, 10% de réussite". Ça évite le piège symétrique : passer un
       niveau ne doit pas suffire à écraser un best_overall solide obtenu sur un niveau
       antérieur si le nouveau niveau n'est pas encore maîtrisé.
    """
    def __init__(self, curriculum_ref, eval_freq=20_000, n_eval_episodes=100,
                 save_path="best_models", min_success_to_count=0.5, verbose=1):
        super().__init__(verbose)
        self.curriculum_ref = curriculum_ref  # instance de SyncedCurriculumCallback (lit .level)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.min_success_to_count = min_success_to_count
        os.makedirs(save_path, exist_ok=True)

        self.eval_env = NavEnv(CurriculumManager())
        self.best_per_level = {}            # level -> (success_rate, mean_reward)
        self.best_overall = (-1, -np.inf)   # (level, success_rate), ordre lexicographique
        self._last_eval_step = 0

    def _evaluate(self, level):
        self.eval_env.curr.set_level(level)
        successes, rewards = [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=100_000 + ep)  # seeds fixes -> évals comparables
            done, ep_reward, info = False, 0.0, {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            successes.append(1.0 if info.get("success") else 0.0)
            rewards.append(ep_reward)
        return float(np.mean(successes)), float(np.mean(rewards))

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        level = self.curriculum_ref.level
        success_rate, mean_reward = self._evaluate(level)
        self.logger.record("eval/level", float(level))
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        if self.verbose:
            print(f"[EvalCallback] Éval niveau {level} ({self.n_eval_episodes} ep, déterministe) "
                  f"succès={success_rate:.1%} récompense_moy={mean_reward:.2f}")

        # -- meilleur modèle DU niveau courant (toujours utile pour du diagnostic/rollback) --
        prev_sr, _ = self.best_per_level.get(level, (-np.inf, -np.inf))
        if success_rate > prev_sr:
            self.best_per_level[level] = (success_rate, mean_reward)
            path = os.path.join(self.save_path, f"eval_best_model_level_{level}")
            self.model.save(path)
            if self.verbose:
                print(f"[EvalCallback] Nouveau meilleur modèle niveau {level} -> {path}.zip")

        # -- meilleur modèle GLOBAL : ordre lexicographique (niveau, succès), gardé par seuil --
        if success_rate >= self.min_success_to_count:
            candidate = (level, success_rate)
            if candidate > self.best_overall:
                self.best_overall = candidate
                path = os.path.join(self.save_path, "best_overall_model")
                self.model.save(path)
                if self.verbose:
                    print(f"[EvalCallback] Nouveau meilleur modèle GLOBAL : niveau={level} "
                          f"succès={success_rate:.1%} -> {path}.zip")

        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


# ----------------------- Entrainement / test -----------------------

def resolve_model_path(model_path):
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path

def load_model(algo_cls, model_path):
    model_path = resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle est introuvable : {model_path}")
    try:
        return algo_cls.load(model_path)
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
            return algo_cls.load(model_path)
        finally:
            torch.load = original_torch_load


def make_env():
    def _init():
        return NavEnv(CurriculumManager())
    return _init


def make_policy_kwargs(pooling="meanmax"):
    return dict(
        features_extractor_class=ObstacleSetExtractor,
        features_extractor_kwargs=dict(
            max_obstacles=NavEnv.MAX_OBSTACLES,
            obs_feat_dim=NavEnv.OBS_FEAT_DIM,
            pooling=pooling,
            features_dim=256,
        ),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )


def train(pooling="meanmax"):
    N_ENVS = 8
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(N_ENVS)]))
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
        gamma=0.998, tau=0.005, train_freq=64, gradient_steps=64,
        learning_starts=10_000, ent_coef="auto",
        policy_kwargs=make_policy_kwargs(pooling=pooling),
        verbose=1, tensorboard_log="./tb_nav",
    )
    
    sync_curriculum_cb = SyncedCurriculumCallback()
    callbacks = CallbackList([
        sync_curriculum_cb,
        LevelBestModelCallback(save_path="best_models", window_size=50, check_freq=1000),
        CurriculumAwareEvalCallback(
            curriculum_ref=sync_curriculum_cb,
            eval_freq=20_000, n_eval_episodes=20,
            save_path="best_models", min_success_to_count=0.5,
        ),
    ])
    
    model.learn(total_timesteps=4_500_000, callback=callbacks)
    model.save("sac_nav_v5.2")
    env.close()


def play(pooling="meanmax"):
    curr = CurriculumManager()
    curr.level = len(CurriculumManager.LEVELS) - 1
    env = NavEnv(curr, render_mode="human")
    model = load_model(SAC, "sac_nav_v5.2")
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
    curr = CurriculumManager()
    curr.level = level if level is not None else len(CurriculumManager.LEVELS) - 1
    env = NavEnv(curr)
    model = load_model(SAC, "sac_nav_v5.2")
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


def evaluate_diagnostic(n_episodes=100, base_seed=10_000, level=None,
                        output_dir="analysis_sac_v5", plot_episodes=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(1)
    os.makedirs(output_dir, exist_ok=True)
    curr = CurriculumManager()
    curr.level = level if level is not None else len(CurriculumManager.LEVELS) - 1
    requested = curr.params
    model = load_model(SAC, "sac_nav_v5")
    env = NavEnv(curr)
    rows, trajectories = [], []
    try:
        for index in range(n_episodes):
            seed = base_seed + index
            obs, _ = env.reset(seed=seed)
            start, goal = env.pos.copy(), env.goal.copy()
            initial_obstacles = [(o["pos"].copy(), float(o["r"]), bool(o["dynamic"]))
                                 for o in env.obstacles]
            dynamic_paths = {o["id"]: [o["pos"].copy()] for o in env.obstacles if o["dynamic"]}
            path = [start.copy()]
            rewards, actions, clearances, dangers = [], [], [], []
            ttc_penalties, safe_stop_steps = [], 0
            done, info = False, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                rewards.append(float(reward))
                clearance_now = min((np.linalg.norm(env.pos - o["pos"]) - env.R_ROBOT - o["r"]
                                     for o in env.obstacles), default=float("inf"))
                clearances.append(clearance_now)
                dangers.append(float(env._last_danger))
                speed = action[0] * env.V_MAX if action[0] >= 0 else action[0] * env.V_MAX_REV
                robot_velocity = speed * np.array([math.cos(env.theta), math.sin(env.theta)])
                ttc_penalties.append(float(env._ttc_features(robot_velocity)[0]))
                if abs(action[0]) < 0.10 and clearance_now >= 0.60 and env._last_danger < 0.10:
                    safe_stop_steps += 1
                path.append(env.pos.copy())
                for obstacle in env.obstacles:
                    if obstacle["dynamic"]:
                        dynamic_paths[obstacle["id"]].append(obstacle["pos"].copy())
                done = terminated or truncated

            path = np.asarray(path)
            actions = np.asarray(actions)
            direct_distance = float(np.linalg.norm(goal - start))
            path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            actual_static = sum(not o["dynamic"] for o in env.obstacles)
            actual_dynamic = len(env.obstacles) - actual_static
            if info.get("success"):
                outcome = "success"
            elif env.steps >= env.MAX_STEPS:
                outcome = "timeout"
            else:
                outcome = "collision"
            action_delta = np.linalg.norm(np.diff(actions, axis=0), axis=1) if len(actions) > 1 else [0.0]
            rows.append({
                "seed": seed, "outcome": outcome, "steps": int(env.steps),
                "return": round(sum(rewards), 3),
                "start_goal_distance_m": round(direct_distance, 3),
                "final_goal_distance_m": round(float(np.linalg.norm(goal - env.pos)), 3),
                "path_length_m": round(path_length, 3),
                "path_efficiency": round(direct_distance / path_length, 3) if path_length else 0.0,
                "requested_static_range": f"{requested['n_static'][0]}-{requested['n_static'][1]}",
                "requested_dynamic_range": f"{requested['n_dyn'][0]}-{requested['n_dyn'][1]}",
                "actual_static": actual_static, "actual_dynamic": actual_dynamic,
                "actual_total": len(env.obstacles),
                "min_clearance_m": round(float(min(clearances)), 3),
                "mean_clearance_m": round(float(np.mean(clearances)), 3),
                "max_danger": round(float(max(dangers)), 3),
                "mean_abs_speed_action": round(float(np.mean(np.abs(actions[:, 0]))), 3),
                "mean_abs_turn": round(float(np.mean(np.abs(actions[:, 1]))), 3),
                "mean_action_change": round(float(np.mean(action_delta)), 3),
                "reverse_fraction": round(float(np.mean(actions[:, 0] < -0.05)), 3),
                "stop_fraction": round(float(np.mean(np.abs(actions[:, 0]) < 0.10)), 3),
                "safe_stop_fraction": round(float(safe_stop_steps / len(actions)), 3),
                "mean_ttc_penalty": round(float(np.mean(ttc_penalties)), 3),
            })
            if len(trajectories) < plot_episodes:
                trajectories.append((rows[-1], path, start, goal, initial_obstacles, dynamic_paths))
            print(f"{index + 1:>3}/{n_episodes}: {outcome:<9} steps={env.steps:<3} "
                  f"obs={len(env.obstacles):<2} clearance={min(clearances):.2f} m")
    finally:
        env.close()

    with open(os.path.join(output_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(row["outcome"] == name for row in rows)
              for name in ("success", "collision", "timeout")}
    report = {
        "episodes": n_episodes, "level": curr.level, "base_seed": base_seed,
        "requested": requested, "counts": counts,
        "rates": {name: round(value / n_episodes, 4) for name, value in counts.items()},
        "mean": {key: round(float(np.mean([row[key] for row in rows])), 4)
                 for key in rows[0]
                 if key not in {"seed", "outcome", "requested_static_range", "requested_dynamic_range"}},
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if trajectories:
        cols, n = 3, len(trajectories)
        fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(5 * cols, 5 * math.ceil(n / cols)), squeeze=False)
        for ax, (row, path, start, goal, obstacles, dynamic_paths) in zip(axes.flat, trajectories):
            for pos, radius, dynamic in obstacles:
                ax.add_patch(plt.Circle(pos, radius, color="#d32f2f" if dynamic else "#6d4c41", alpha=0.28))
            for positions in dynamic_paths.values():
                positions = np.asarray(positions)
                ax.plot(positions[:, 0], positions[:, 1], color="#d32f2f", alpha=0.18, linewidth=0.5)
            ax.plot(path[:, 0], path[:, 1], color="#1769aa", linewidth=1.5)
            ax.scatter(*start, c="#1565c0", s=35)
            ax.scatter(*goal, c="#2e7d32", marker="*", s=120)
            ax.set(xlim=(-6, 6), ylim=(-6, 6), aspect="equal",
                   title=f"seed {row['seed']} | {row['outcome']} | {row['steps']} pas")
            ax.grid(alpha=0.25)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "trajectories.png"), dpi=160)
        plt.close(fig)
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--diagnostic", action="store_true",
                    help="Evaluation tracee : CSV, resume JSON et cartes PNG reproductibles")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--output-dir", default="analysis_sac_v5")
    ap.add_argument("--level", type=int, default=None,
                    help="Niveau de curriculum a evaluer (defaut: dernier niveau)")
    ap.add_argument("--pooling", choices=["meanmax", "attention"], default="meanmax",
                    help="Mode de pooling de l'encodeur d'obstacles (defaut: meanmax)")
    args = ap.parse_args()
    if args.play:
        play(pooling=args.pooling)
    elif args.eval:
        evaluate(n_episodes=args.episodes, level=args.level)
    elif args.diagnostic:
        evaluate_diagnostic(n_episodes=args.episodes, level=args.level,
                            output_dir=args.output_dir)
    else:
        train(pooling=args.pooling)