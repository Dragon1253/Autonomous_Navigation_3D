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
import io
import math
import os
import zipfile
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
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
    MAX_STEPS = 800
    PASSAGE_MARGIN = 0.30  # marge de passage garantie entre 2 obstacles, en plus de 2*R_ROBOT

    # (v5) representation "deep sets" : tous les obstacles + masque, plus de K_NEAREST
    MAX_OBSTACLES = 40    # >= max(n_static)+max(n_dyn) sur le curriculum ci-dessus (25+13=38), marge incluse
    OBS_FEAT_DIM = 6      # [rel_x, rel_y, rel_vx, rel_vy, rayon_norm, flag_dynamique] par obstacle
    OBS_POS_NORM = 10.0   # normalisation position relative (ordre de grandeur de l'arene)
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

        # obstacles(MAX_OBSTACLES*OBS_FEAT_DIM) + masque(MAX_OBSTACLES)
        # + base(5) + danger(1) + lidar(N_RAYS)  <- ordre attendu par ObstacleSetExtractor
        obs_dim = (5 + self.MAX_OBSTACLES * self.OBS_FEAT_DIM + self.MAX_OBSTACLES
                   + 1 + self.N_RAYS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self._last_danger = 0.0

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
        if clearance is None:
            clearance = 2 * self.R_ROBOT + self.PASSAGE_MARGIN
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
        return np.array([r[2] for r in res], dtype=np.float32)

    # ---------- (v5) features "deep sets" : TOUS les obstacles + masque ----------
    def _obstacle_set_features(self):
        """Remplace l'ancien _dynamic_obstacle_features (K_NEAREST=4, dynamiques
        seulement). Ici : tous les obstacles (statiques + dynamiques), tries par
        distance seulement pour geler un ordre en cas de troncature defensive
        (ne devrait jamais arriver, MAX_OBSTACLES >= max du curriculum) -- la
        tri n'est pas requis par l'architecture (pooling invariant par
        permutation), juste une precaution."""
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
        # Ordre impose par ObstacleSetExtractor : [obstacles | masque | contexte]
        # ou contexte = base + danger + lidar. Ne pas reordonner sans mettre a
        # jour le slicing de l'extracteur (obstacle_block_size / mask_block_size).
        return np.concatenate([
            obst_feats, obst_mask, base, [self._last_danger], self._lidar()
        ]).astype(np.float32)

    # ---------- TTC (time-to-collision) : penalite reward + scalaire de danger observable ----------
    def _ttc_features(self, v_robot_world, ttc_threshold=1.8, weight=0.4):
        """S'applique uniformement aux obstacles statiques (vel=0) et dynamiques.
        Retourne un scalaire de danger (0=aucun risque, 1=collision quasi-
        imminente) = pire ratio de risque sur TOUS les obstacles. Garde comme
        prior explicite en plus de la representation deep-sets, en plus des
        features par-obstacle : ca ne coute rien et donne un signal agrege
        immediatement exploitable des le debut de l'entrainement."""
        penalty = 0.0
        worst_ratio = 0.0
        for o in self.obstacles:
            rel_pos = o["pos"] - self.pos
            dist_surface = np.linalg.norm(rel_pos) - self.R_ROBOT - o["r"]
            if dist_surface <= 0:
                continue
            rel_vel = o["vel"] - v_robot_world
            closing = -np.dot(rel_vel, rel_pos) / (np.linalg.norm(rel_pos) + 1e-6)
            if closing > 1e-3:
                ttc = dist_surface / closing
                if ttc < ttc_threshold:
                    ratio = (ttc_threshold - ttc) / ttc_threshold
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
        reward = 10.0 * (self.prev_d - d) - 0.05
        self.prev_d = d

        ttc_penalty, danger_scalar = self._ttc_features(v_robot_world)
        self._last_danger = danger_scalar

        obs = self._obs()
        min_ray = float(np.min(obs[-self.N_RAYS:])) * self.RAY_RANGE   # lidar = toujours les N_RAYS derniers
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


# ----------------------- (v5) Extracteur de features "deep sets" -----------------------
class ObstacleSetExtractor(BaseFeaturesExtractor):
    """Encode l'observation structuree [base | obstacles(MAX_OBS*FEAT) | masque(MAX_OBS)
    | danger | lidar] en :
      1) passant chaque obstacle (reel ou pad) dans un MLP PARTAGE (memes poids
         pour tous les slots) -> embedding par obstacle ;
      2) agregeant les embeddings par pooling MASQUE (les slots de padding ne
         contribuent pas), en mean+max ou en attention selon `pooling` ;
      3) fusionnant le vecteur agrege avec un encodage du "contexte" (base +
         danger + lidar) dans une couche finale.

    Pour passer a l'attention : `pooling="attention"` (l'agregation devient
    une somme ponderee par des scores softmax appris par obstacle, au lieu
    d'un mean+max fixe -- utile si certains obstacles doivent clairement
    dominer la decision plutot que d'etre moyennes/maximises independamment
    par dimension).
    """

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

        # (1) encodeur partage : mêmes poids appliques a chaque obstacle (reel ou pad)
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(obs_feat_dim, obstacle_embed_dim), nn.ReLU(),
            nn.Linear(obstacle_embed_dim, obstacle_embed_dim), nn.ReLU(),
        )

        if pooling == "attention":
            self.attn_query = nn.Linear(obstacle_embed_dim, 1)
            pooled_dim = obstacle_embed_dim
        else:  # meanmax
            pooled_dim = obstacle_embed_dim * 2

        # (3) encodeur du contexte (base + danger + lidar)
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
                               self.obstacle_block_size + self.mask_block_size]  # (B, M) in {0,1}
        context = observations[:, self.obstacle_block_size + self.mask_block_size:]

        embed = self.obstacle_encoder(obst)              # (B, M, E)
        mask_f = mask_f.unsqueeze(-1)                      # (B, M, 1)
        mask_b = mask_f > 0.5                              # (B, M, 1) bool, pour masked_fill

        if self.pooling == "attention":
            scores = self.attn_query(embed).squeeze(-1)           # (B, M)
            scores = scores.masked_fill(~mask_b.squeeze(-1), float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, M, 1)
            weights = torch.nan_to_num(weights, nan=0.0)          # garde-fou si masque entierement vide
            pooled = (embed * weights).sum(dim=1)                 # (B, E)
        else:
            max_input = embed.masked_fill(~mask_b, -1e4)
            max_pool = max_input.max(dim=1).values
            max_pool = torch.nan_to_num(max_pool, neginf=0.0)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            mean_pool = (embed * mask_f).sum(dim=1) / denom
            pooled = torch.cat([mean_pool, max_pool], dim=1)      # (B, 2E)

        ctx = self.context_encoder(context)
        return self.fusion(torch.cat([pooled, ctx], dim=1))


# ----------------------- Callback de curriculum (pilotage centralise) -----------------------
from collections import deque

class SyncedCurriculumCallback(BaseCallback):
    """Fenetre glissante GLOBALE (agregee sur tous les workers), plancher
    MIN_STEPS_PER_LEVEL, plafond MAX_STEPS_PER_LEVEL, diffusion du niveau via
    env_method. Inchange par rapport a v4 : generique a n'importe quel
    algorithme SB3 (on-policy ou off-policy), fonctionne tel quel avec PPO."""
    WINDOW = 50
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


# ----------------------- Entrainement / test -----------------------

def resolve_model_path(model_path):
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path

def load_model(algo_cls, model_path):
    """Charge un modele SB3 en gerant le bogue de lecture PyTorchStreamReader
    via la lecture en memoire (BytesIO), generalise a n'importe quel algo."""
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
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )


def train(pooling="meanmax"):
    N_ENVS = 8
    env = VecMonitor(SubprocVecEnv([make_env() for _ in range(N_ENVS)]))
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,          # par env -> 2048*8=16384 transitions par rollout
        batch_size=2048,
        n_epochs=10,
        gamma=0.998,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # PPO n'a pas de temperature auto comme SAC -> bonus d'entropie explicite
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=make_policy_kwargs(pooling),
        verbose=1, tensorboard_log="./tb_nav_v5",
    )
    model.learn(total_timesteps=5_000_000, callback=SyncedCurriculumCallback())
    model.save("ppo_nav_v5")
    env.close()


def play(pooling="meanmax"):
    curr = CurriculumManager()
    curr.level = len(CurriculumManager.LEVELS) - 1
    env = NavEnv(curr, render_mode="human")
    model = load_model(PPO, "ppo_nav_v5")
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
    model = load_model(PPO, "ppo_nav_v5")
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
    ap.add_argument("--pooling", choices=["meanmax", "attention"], default="meanmax",
                    help="Mode de pooling de l'encodeur d'obstacles (defaut: meanmax)")
    args = ap.parse_args()
    if args.play:
        play(pooling=args.pooling)
    elif args.eval:
        evaluate(level=args.level)
    else:
        train(pooling=args.pooling)
