"""
Navigation RL avec PyBullet : point A -> B, obstacles statiques + dynamiques,
curriculum adaptatif, SAC (Stable-Baselines3).

v4 - suite a l'audit de code du checkpoint sac_nav_final_64 (v3, 4M steps).
Trois causes racines identifiees et corrigees :

A. MARCHE ARRIERE ABSENTE. En v3, v = (a0+1)/2*V_MAX ne pouvait jamais etre
   negatif : le robot ne pouvait physiquement pas reculer, seulement pivoter
   sur place puis ravancer. Corrige : mapping asymetrique, v in
   [-V_MAX_REV, V_MAX], V_MAX_REV < V_MAX (recul plus lent, plus realiste).

B. PERCEPTION DYNAMIQUE LIMITEE. K_NEAREST=2 ne suffit pas au palier final
   (n_dyn jusqu'a 15) : les obstacles dynamiques au-dela des 2 plus proches
   n'avaient aucune info de vitesse (seulement le lidar, distance brute).
   Corrige : K_NEAREST=4, et surtout ajout d'un scalaire de "danger global"
   dans l'observation = pire ratio de risque TTC calcule sur TOUS les
   obstacles (statiques et dynamiques, au-dela des K_NEAREST explicites).
   Ce scalaire existait deja cote reward (_ttc_penalty) mais n'etait jamais
   donne a l'agent comme observation -> il ne pouvait pas anticiper, il
   subissait seulement la penalite a posteriori.

C. CALIBRAGE RISQUE/REWARD TROP PERMISSIF SUR LES PASSAGES ETROITS. Le
   clearance minimum garanti entre deux obstacles au spawn etait
   2*R_ROBOT+0.15 = 0.65m pour un robot de 0.5m de diametre, soit 15cm de
   marge totale (~7.5cm de chaque cote) : un couloir a la limite du
   franchissable pour une politique lidar sans planification geometrique.
   Corrige : marge de passage portee a 0.30m (PASSAGE_MARGIN), et la
   penalite de proximite (zone < 0.5m) passe d'une forme lineaire plafonnee
   a une forme en 1/distance, qui punit disproportionnellement le fait de
   "raser" un obstacle plutot que de garder de la marge.

ATTENTION COMPATIBILITE : l'espace d'observation (K_NEAREST, +1 scalaire de
danger) ET l'espace d'action (mapping vitesse asymetrique) ont change par
rapport a v3. Le checkpoint sac_nav_final_64 n'est PAS compatible : il faut
reentrainer depuis zero avec ce script (model.save("sac_nav_v4")).

D. CURRICULUM DESYNCHRONISE ENTRE WORKERS. En v3, chaque worker
   SubprocVecEnv avait sa PROPRE instance de CurriculumManager qui
   s'auto-avancait sur SA fenetre glissante locale de 50 episodes -> rien ne
   garantissait un temps d'entrainement suffisant au niveau final, et un
   worker pouvait rester bloque indefiniment sur un palier difficile.
   Corrige : CurriculumManager ne fait plus que tenir un niveau (set_level),
   pilote de facon centralisee par SyncedCurriculumCallback qui agrege le
   succes de TOUS les workers (fenetre globale), applique un plancher
   (MIN_STEPS_PER_LEVEL) et un plafond (MAX_STEPS_PER_LEVEL) de steps par
   palier, et diffuse le niveau via env_method. Le palier intermediaire
   n_static=(18,25) (commente en v3) est reinstaure pour lisser le saut vers
   le niveau final.

--- notes v2 (conservees pour l'historique) ---
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
    """(D) Ne decide plus rien tout seul : tient juste le niveau courant et
    expose les parametres associes. En v3, chaque worker SubprocVecEnv avait
    sa PROPRE instance qui s'auto-avancait sur SA fenetre glissante locale de
    50 episodes -> progression desynchronisee entre workers, et rien ne
    garantissait un temps d'entrainement suffisant au niveau final. Le niveau
    est maintenant pilote de facon centralisee par SyncedCurriculumCallback
    (cf plus bas), qui agrege le succes de TOUS les workers et diffuse le
    niveau via `set_level` / `env_method`."""
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
        # (D) reinstaure : le saut de 20 a 30 obstacles statiques directement
        # au niveau final etait trop brutal (commente en v3).
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

# ----------------------- Environnement -----------------------
class NavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ARENA = 6.0          # demi-taille de l'arene (12 x 12 m)
    R_ROBOT = 0.25
    V_MAX = 1.0          # m/s, marche avant
    V_MAX_REV = 0.6      # m/s, marche arriere (A : plus lente, plus realiste qu'un mapping symetrique)
    W_MAX = 2.0          # rad/s
    DT = 1.0 / 30.0
    N_RAYS = 24
    RAY_RANGE = 5.0
    GOAL_RADIUS = 0.45
    MAX_STEPS = 800
    K_NEAREST = 4         # (B) nb d'obstacles dynamiques dont on donne pos+vitesse relatives (etait 2)
    V_OBS_NORM = 2.0       # normalisation des vitesses relatives (robot + obstacle <= ~2 m/s)
    PASSAGE_MARGIN = 0.30  # (C) marge de passage garantie entre 2 obstacles, en plus de 2*R_ROBOT (etait 0.15)

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
        # (B) +1 : scalaire de danger global (pire ratio TTC, tous obstacles), avant le lidar
        self.OBS_BASE = 5 + 4 * self.K_NEAREST + 1   # index ou commence le lidar dans l'obs
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            shape=(self.OBS_BASE + self.N_RAYS,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self._last_danger = 0.0   # (B) mis a jour a chaque step(), lu par _obs()

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
        # (C) clearance par defaut : diametre robot (2 * R_ROBOT) + marge de passage
        # PASSAGE_MARGIN=0.30 (etait 0.15) -> couloirs difficiles mais franchissables
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
        n_on_path = max(1, int(0.4 * total))  # garantit des obstacles sur la trajectoire

        # (C) Espace minimum garanti entre la surface de deux obstacles
        min_pass_gap = 2 * self.R_ROBOT + self.PASSAGE_MARGIN

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
            base, self._dynamic_obstacle_features(), [self._last_danger], self._lidar()
        ]).astype(np.float32)

    # ---------- TTC (time-to-collision) : penalite reward + scalaire de danger observable ----------
    def _ttc_features(self, v_robot_world, ttc_threshold=1.8, weight=0.4):
        """S'applique uniformement aux obstacles statiques (vel=0, la vitesse de
        rapprochement vient alors du seul mouvement du robot) et dynamiques.

        (B) Retourne desormais AUSSI un scalaire de danger (0=aucun risque,
        1=collision quasi-imminente) = pire ratio de risque sur TOUS les
        obstacles, pas seulement les K_NEAREST dynamiques explicites. En v3
        ce risque n'existait que dans la reward (a posteriori) ; ici il est
        aussi expose dans l'observation (_obs) pour que l'agent puisse
        anticiper les obstacles au-dela de K_NEAREST."""
        penalty = 0.0
        worst_ratio = 0.0
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
        self._last_danger = 0.0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        # (A) mapping asymetrique : marche avant [0, V_MAX], marche arriere [-V_MAX_REV, 0].
        # En v3, v = (a0+1)/2*V_MAX ne pouvait jamais etre negatif -> impossible de
        # reculer, seul un pivot sur place puis une nouvelle approche etait possible.
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
        reward = 10.0 * (self.prev_d - d) - 0.05          # progression + temps
        self.prev_d = d

        # (B) calcule AVANT l'obs pour que le scalaire de danger reflete l'etat courant
        ttc_penalty, danger_scalar = self._ttc_features(v_robot_world)
        self._last_danger = danger_scalar

        obs = self._obs()
        min_ray = float(np.min(obs[self.OBS_BASE:])) * self.RAY_RANGE
        if min_ray < 0.4:                                  # zone de danger (marge generale)
            # (C) penalite en 1/distance (etait lineaire, plafonnee a 0.15) : punit
            # disproportionnellement le fait de raser un obstacle plutot que de
            # garder de la marge, pour decourager les raccourcis a la limite du
            # franchissable (cf. PASSAGE_MARGIN).
            eps = 0.05
            prox_penalty = 0.05 * (1.0 / (min_ray + eps) - 1.0 / (0.5 + eps))
            reward -= min(prox_penalty, 2.0)
        reward -= ttc_penalty          # terme specifique "se rapproche vite" (tous obstacles)

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
        # (D) plus de self.curr.report() ici : la decision de monter de niveau
        # est desormais centralisee dans SyncedCurriculumCallback, qui agrege
        # le succes de tous les workers plutot que la seule fenetre locale de
        # cet environnement. "success"/"level" restent dans info pour que le
        # callback puisse les lire depuis le process principal.
        if self.render_mode == "human":
            import time; time.sleep(self.DT)
        return obs, float(reward), terminated, truncated, {"success": success ,
                                                             "level": self.curr.level,   # <- necessaire pour observer la progression depuis le process principal
                                                           }

    def set_curriculum_level(self, level):
        """(D) Point d'entree appele a distance par SyncedCurriculumCallback
        via `training_env.env_method("set_curriculum_level", level)`, qui
        passe par IPC jusqu'a ce sous-processus SubprocVecEnv."""
        return self.curr.set_level(level)

    def close(self):
        p.disconnect(physicsClientId=self.cid)

# ----------------------- Callback de curriculum (pilotage centralise) -----------------------
from collections import deque

class SyncedCurriculumCallback(BaseCallback):
    """
    (D) Remplace CurriculumLogCallback (v3), qui se contentait de LIRE le
    niveau de chaque worker sans les piloter -- chaque worker avancait tout
    seul sur sa propre fenetre glissante locale (cf. l'ancienne docstring :
    "Chaque worker a sa PROPRE instance de CurriculumManager"). Consequences
    en v3 : progression desynchronisee entre workers, et rien ne garantissait
    un temps d'entrainement suffisant au niveau final (un worker malchanceux
    pouvait rester bloque, un worker chanceux grimper vite sur une fenetre
    de 50 episodes qui n'est pas forcement representative).

    Ici, un seul niveau "maitre" est tenu dans le process principal :
    - fenetre glissante GLOBALE (agregee sur tous les workers, donc plus
      stable statistiquement qu'une fenetre par worker) ;
    - plancher MIN_STEPS_PER_LEVEL : jamais de montee de niveau avant ce
      nombre de steps, meme en cas de succes precoce par chance ;
    - plafond MAX_STEPS_PER_LEVEL : force la montee si le seuil de succes
      n'est pas atteint apres ce nombre de steps, pour ne jamais rester
      bloque indefiniment et garantir un budget residuel pour les niveaux
      superieurs (dont le niveau final -- pas de plafond sur celui-ci,
      l'entrainement continue dessus jusqu'a total_timesteps) ;
    - diffuse le nouveau niveau a tous les workers via env_method (IPC vers
      chaque sous-processus SubprocVecEnv).

    Valeurs de MIN/MAX_STEPS_PER_LEVEL choisies par raisonnement (budget
    total / nb de niveaux), pas calibrees empiriquement -- a ajuster selon
    ce qu'on observe sur curriculum/global_level et curriculum/global_success_rate
    dans TensorBoard.
    """
    WINDOW = 50
    THRESHOLD = 0.8
    MIN_STEPS_PER_LEVEL = 20_000
    MAX_STEPS_PER_LEVEL = 500_000

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
        gamma=0.998, tau=0.005, train_freq=64, gradient_steps=64,   # <- corrige aussi (etait reste a 1)
        learning_starts=10_000, ent_coef="auto",
        policy_kwargs=dict(net_arch=[512, 512]),
        verbose=1, tensorboard_log="./tb_nav",
    )
    model.learn(total_timesteps=3_000_000, callback=SyncedCurriculumCallback())
    model.save("sac_nav_v4")
    env.close()

def play():
    curr = CurriculumManager()
    curr.level = len(CurriculumManager.LEVELS) - 1   # test en pleine difficulte
    env = NavEnv(curr, render_mode="human")
    model = load_sac("sac_nav_v4")
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
    model = load_sac("sac_nav_v4")
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