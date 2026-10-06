"""
Navigation RL avec PyBullet : point A -> B, obstacles statiques + dynamiques,
curriculum adaptatif, PPO/SAC avec visualisations avancees pour presentation.

v5.2 (Visuel) - Intégration complète des phases 1 à 5 :
- Robot détaillé avec LED d'état tricolore (danger TTC).
- Obstacles variés (caisses + barils) avec flèches d'orientation pour les dynamiques.
- Faisceaux LiDAR 3D en temps réel et traînée de trajectoire (40 derniers pas).
- HUD d'informations (Distance, Danger, Steps).
- Texture de mur industrielle à bandes de sécurité et caméra orbitale cinématique.

Entraînement : python sac_train_v5.py
Test visuel  : python sac_train_v5.py --play
Evaluation   : python sac_train_v5.py --eval [--level N]
"""
import argparse
import csv
import io
import json
import math
import os
import zipfile
from collections import deque
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
        dict(n_static=(20, 30), n_dyn=(10, 15), speed=(0.7, 0.8),  perturb_prob=0.25,  min_goal=6.0),
    ]

    def __init__(self):
        self.level = 0

    @property
    def params(self):
        return self.LEVELS[self.level]

    def set_level(self, level):
        self.level = int(np.clip(level, 0, len(self.LEVELS) - 1))
        return self.level


# ----------------------- Textures Phase 5 -----------------------
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

def ensure_wall_texture(path="assets/wall_stripes.png", size=512):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (size, size), (130, 135, 140))
    draw = ImageDraw.Draw(img)
    stripe_h = size // 6
    y0 = size // 2 - stripe_h // 2
    for x in range(-size, size * 2, 40):
        draw.polygon([(x, y0), (x + 20, y0), (x + 20 + stripe_h, y0 + stripe_h), (x + stripe_h, y0 + stripe_h)], fill=(230, 180, 20))
        draw.polygon([(x + 20, y0), (x + 40, y0), (x + 40 + stripe_h, y0 + stripe_h), (x + 20 + stripe_h, y0 + stripe_h)], fill=(30, 30, 30))
    img.save(path)
    return path


# ----------------------- Environnement -----------------------
class NavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ARENA = 6.0          
    R_ROBOT = 0.25
    V_MAX = 1.0          
    V_MAX_REV = 0.6      
    W_MAX = 2.0          
    DT = 1.0 / 30.0
    N_RAYS = 24
    RAY_RANGE = 5.0
    GOAL_RADIUS = 0.45
    MAX_STEPS = 1300
    PASSAGE_MARGIN = 0.30  

    MAX_OBSTACLES = 40    
    OBS_FEAT_DIM = 6      
    OBS_POS_NORM = 10.0   
    V_OBS_NORM = 2.0       

    def __init__(self, curriculum: CurriculumManager, render_mode=None):
        super().__init__()
        self.curr = curriculum
        self.render_mode = render_mode
        self.cid = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        
        if render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.cid)
            p.resetDebugVisualizerCamera(cameraDistance=9.0, cameraYaw=55, cameraPitch=-38,
                                        cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        tex_id = p.loadTexture(ensure_warehouse_floor_texture(), physicsClientId=self.cid)
        p.changeVisualShape(plane_id, -1, textureUniqueId=tex_id, physicsClientId=self.cid)
        
        self._build_walls()
        
        # Robot
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.R_ROBOT, height=0.3,
                                    physicsClientId=self.cid)
        self.robot_id = p.createMultiBody(0, col, self._make_robot_visual(), [0, 0, 0.15],
                                        physicsClientId=self.cid)
        
        self._drum_tex_id = p.loadTexture(ensure_drum_texture(), physicsClientId=self.cid)
        self._beacon_vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.06,
                                                rgbaColor=[1.0, 0.55, 0.0, 1],
                                                physicsClientId=self.cid)
        # Phase 2b: Chevron/Flèche d'orientation pour les obstacles dynamiques
        self._arrow_vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.04, 0.02],
                                                 rgbaColor=[1.0, 0.2, 0.2, 1],
                                                 physicsClientId=self.cid)
        
        self.goal_vis = self._make_cylinder(self.GOAL_RADIUS, 0.02, [0.1, 0.9, 0.1, 0.6],
                                            collision=False)
        self.obstacles = []   

        obs_dim = (5 + self.MAX_OBSTACLES * self.OBS_FEAT_DIM + self.MAX_OBSTACLES
                   + 1 + self.N_RAYS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self._last_danger = 0.0
        
        # Phase 3: Traînée de trajectoire
        self.trail_positions = deque(maxlen=40)

    # ---------- Phase 1: Robot Visuel ----------
    def _make_robot_visual(self):
        return p.createVisualShapeArray(
            shapeTypes=[
                p.GEOM_CYLINDER,             # Chassis
                p.GEOM_BOX, p.GEOM_BOX,     # 2 Roues laterales (cosmetiques)
                p.GEOM_CYLINDER,             # Mat Lidar
                p.GEOM_SPHERE,               # Tourelle Lidar (LED d'etat)
                p.GEOM_BOX                   # Fleche direction
            ],
            radii=[self.R_ROBOT, 0, 0, 0.03, 0.05, 0],
            lengths=[0.28, 0, 0, 0.15, 0, 0],
            halfExtents=[
                [0, 0, 0],
                [0.03, 0.10, 0.06],
                [0.03, 0.10, 0.06],
                [0, 0, 0],
                [0, 0, 0],
                [0.09, 0.045, 0.02]
            ],
            visualFramePositions=[
                [0, 0, 0],
                [0, self.R_ROBOT + 0.02, -0.05],
                [0, -self.R_ROBOT - 0.02, -0.05],
                [0, 0, 0.22],
                [0, 0, 0.30],
                [self.R_ROBOT * 0.8, 0, 0.06]
            ],
            visualFrameOrientations=[[0, 0, 0, 1]] * 6,
            rgbaColors=[
                [0.12, 0.14, 0.16, 1],       # Anthracite
                [0.05, 0.05, 0.05, 1],       # Roue gauche
                [0.05, 0.05, 0.05, 1],       # Roue droite
                [0.2, 0.2, 0.2, 1],          # Mat
                [0.15, 0.9, 0.3, 1],         # LED (vert initial)
                [1.0, 0.75, 0.0, 1]          # Fleche jaune
            ],
            physicsClientId=self.cid,
        )

    def _update_robot_status_led(self):
        if self._last_danger > 0.6:
            color = [0.9, 0.15, 0.15, 1]   # Rouge = Danger
        elif self._last_danger > 0.25:
            color = [1.0, 0.65, 0.0, 1]    # Orange = Prudence
        else:
            color = [0.15, 0.9, 0.3, 1]    # Vert = RAS
        p.changeVisualShape(self.robot_id, -1, shapeIndex=4, rgbaColor=color, physicsClientId=self.cid)

    # ---------- Phase 2: Formes d'obstacles ----------
    def _make_cylinder(self, radius, height, rgba, collision=True):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                     physicsClientId=self.cid) if collision else -1
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=rgba, physicsClientId=self.cid)
        return p.createMultiBody(0, col, vis, [0, 0, height / 2], physicsClientId=self.cid)

    def _make_box_obstacle(self, radius, height=0.6, rgba=[0.8, 0.5, 0.2, 1]):
        # Collision cylindrique constante pour preserver la physique
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.cid)
        half_side = radius / math.sqrt(2)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_side, half_side, height / 2],
                                  rgbaColor=rgba, physicsClientId=self.cid)
        return p.createMultiBody(0, col, vis, [0, 0, height / 2], physicsClientId=self.cid)

    def _build_walls(self):
        t, h, L = 0.1, 0.5, self.ARENA
        wall_tex = p.loadTexture(ensure_wall_texture(), physicsClientId=self.cid)
        for cx, cy, ex, ey in [(L, 0, t, L), (-L, 0, t, L), (0, L, L, t), (0, -L, L, t)]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ex, ey, h],
                                         physicsClientId=self.cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[ex, ey, h],
                                      rgbaColor=[0.8, 0.8, 0.8, 1], physicsClientId=self.cid)
            wid = p.createMultiBody(0, col, vis, [cx, cy, h], physicsClientId=self.cid)
            p.changeVisualShape(wid, -1, textureUniqueId=wall_tex, physicsClientId=self.cid)

    # ---------- Placement & mouvement ----------
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
            if o.get("beacon_id") is not None:
                p.removeBody(o["beacon_id"], physicsClientId=self.cid)
            if o.get("arrow_id") is not None:
                p.removeBody(o["arrow_id"], physicsClientId=self.cid)
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

            # Phase 2a: Alternance caisse (box) / baril (cylinder)
            is_box = (self.rng.random() > 0.5)
            color = [0.9, 0.2, 0.2, 1] if dynamic else [0.4, 0.3, 0.2, 1]
            
            if is_box:
                oid = self._make_box_obstacle(r, height=0.6, rgba=color)
            else:
                oid = self._make_cylinder(r, 0.6, color)
                if not dynamic:
                    p.changeVisualShape(oid, -1, textureUniqueId=self._drum_tex_id, rgbaColor=[1, 1, 1, 1],
                                        physicsClientId=self.cid)
            
            p.resetBasePositionAndOrientation(oid, [*pos, 0.3], [0, 0, 0, 1],
                                            physicsClientId=self.cid)

            vel = np.zeros(2)
            beacon_id, arrow_id = None, None
            if dynamic:
                ang = self.rng.uniform(0, 2 * np.pi)
                s = self.rng.uniform(*speed_rng)
                vel = s * np.array([np.cos(ang), np.sin(ang)])
                beacon_id = p.createMultiBody(0, -1, self._beacon_vis_id, [*pos, 0.72],
                                            physicsClientId=self.cid)
                # Phase 2b: Fleche d'orientation pour obstacle dynamique
                arrow_id = p.createMultiBody(0, -1, self._arrow_vis_id, [*pos, 0.65],
                                             physicsClientId=self.cid)

            self.obstacles.append(dict(id=oid, pos=pos, vel=vel, r=r, dynamic=dynamic,
                                    perturb_prob=perturb_prob if dynamic else 0.0,
                                    beacon_id=beacon_id, arrow_id=arrow_id))

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

                # Phase 2b: Orientation de la fleche selon le vecteur vitesse
                ang = math.atan2(o["vel"][1], o["vel"][0])
                p.resetBasePositionAndOrientation(o["arrow_id"], [*o["pos"], 0.65],
                                                p.getQuaternionFromEuler([0, 0, ang]),
                                                physicsClientId=self.cid)

    # ---------- Phase 3: Visualisation LiDAR Debug & Trainee ----------
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

    def _draw_lidar_debug(self, hits):
        if not hasattr(self, "_lidar_line_ids"):
            self._lidar_line_ids = [None] * self.N_RAYS
        angles = self.theta + np.linspace(0, 2 * np.pi, self.N_RAYS, endpoint=False)
        for i, (a, hit) in enumerate(zip(angles, hits)):
            hit_dist = hit * self.RAY_RANGE
            d = np.array([math.cos(a), math.sin(a)])
            start = [*(self.pos + d * (self.R_ROBOT + 0.05)), 0.2]
            end = [*(self.pos + d * hit_dist), 0.2]
            color = [0.9, 0.2, 0.2] if hit_dist < 1.5 else ([1.0, 0.6, 0.0] if hit_dist < 3.0 else [0.2, 0.9, 0.3])
            if self._lidar_line_ids[i] is None:
                self._lidar_line_ids[i] = p.addUserDebugLine(start, end, color, lineWidth=1.5,
                                                              physicsClientId=self.cid)
            else:
                p.addUserDebugLine(start, end, color, lineWidth=1.5,
                                  replaceItemUniqueId=self._lidar_line_ids[i], physicsClientId=self.cid)

    def _draw_trail(self):
        self.trail_positions.append(self.pos.copy())
        if len(self.trail_positions) < 2:
            return
        if not hasattr(self, "_trail_line_ids"):
            self._trail_line_ids = []
        needed = len(self.trail_positions) - 1
        while len(self._trail_line_ids) < needed:
            self._trail_line_ids.append(None)
        
        pts = list(self.trail_positions)
        for i in range(len(pts) - 1):
            p1 = [*pts[i], 0.05]
            p2 = [*pts[i+1], 0.05]
            alpha = (i + 1) / len(pts)
            color = [0.1 * alpha, 0.6 * alpha, 1.0 * alpha]
            if self._trail_line_ids[i] is None:
                self._trail_line_ids[i] = p.addUserDebugLine(p1, p2, color, lineWidth=2.0, physicsClientId=self.cid)
            else:
                p.addUserDebugLine(p1, p2, color, lineWidth=2.0, replaceItemUniqueId=self._trail_line_ids[i], physicsClientId=self.cid)

    # ---------- Phase 4: HUD Text ----------
    def _update_hud(self):
        dist = np.linalg.norm(self.goal - self.pos)
        txt = f"Dist. but: {dist:.1f}m | Danger: {self._last_danger:.2f} | Step: {self.steps}"
        if not hasattr(self, "_hud_id"):
            self._hud_id = p.addUserDebugText(txt, [0, 0, 3.2], textColorRGB=[1, 1, 1], textSize=1.4,
                                              physicsClientId=self.cid)
        else:
            p.addUserDebugText(txt, [0, 0, 3.2], textColorRGB=[1, 1, 1], textSize=1.4,
                               replaceItemUniqueId=self._hud_id, physicsClientId=self.cid)

    # ---------- Phase 5: Camera Cinématique ----------
    def _update_camera(self):
        yaw = 55 + 8 * math.sin(self.steps * 0.01)
        target = [*self.pos * 0.3, 0]
        p.resetDebugVisualizerCamera(cameraDistance=9.0, cameraYaw=yaw, cameraPitch=-38,
                                    cameraTargetPosition=target, physicsClientId=self.cid)

    # ---------- Deep Sets Features & TTC ----------
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

    # ---------- Gym API ----------
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
        self.trail_positions.clear()
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
        to_goal_dir = (self.goal - self.pos) / (d + 1e-6)
        vel_toward_goal = np.dot(v_robot_world, to_goal_dir)
        progress = 10.0 * vel_toward_goal * self.DT
        if progress < 0:
            progress *= 0.3
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

        # Render visuel dynamique
        if self.render_mode == "human":
            hits = obs[-self.N_RAYS:]
            self._draw_lidar_debug(hits)
            self._update_robot_status_led()
            self._draw_trail()
            self._update_hud()
            self._update_camera()
            import time; time.sleep(self.DT)

        return obs, float(reward), terminated, truncated, {"success": success,
                                                             "level": self.curr.level}

    def set_curriculum_level(self, level):
        return self.curr.set_level(level)

    def close(self):
        p.disconnect(physicsClientId=self.cid)


# ----------------------- Extracteur Deep Sets -----------------------
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
class LevelBestModelCallback(BaseCallback):
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

        levels_now = [info["level"] for info in infos if "level" in info]
        if levels_now:
            level = int(np.round(np.mean(levels_now)))
            if level != self.current_level:
                self.current_level = level
                self.best_success_rate = -np.inf
                self.best_mean_reward = -np.inf
                self.episode_rewards.clear()
                self.episode_successes.clear()

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if done and "success" in info:
                self.episode_successes.append(1.0 if info["success"] else 0.0)

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
        for done, info in zip(dones, infos):
            if done and "success" in info:
                self.global_outcomes.append(1.0 if info["success"] else 0.0)

        success_rate = float(np.mean(self.global_outcomes)) if self.global_outcomes else 0.0
        elapsed = self.num_timesteps - self.level_start_step

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
        return True


class CurriculumAwareEvalCallback(BaseCallback):
    def __init__(self, curriculum_ref, eval_freq=20_000, n_eval_episodes=20,
                 save_path="best_models", min_success_to_count=0.5, verbose=1):
        super().__init__(verbose)
        self.curriculum_ref = curriculum_ref
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.min_success_to_count = min_success_to_count
        os.makedirs(save_path, exist_ok=True)

        self.eval_env = NavEnv(CurriculumManager())
        self.best_per_level = {}            
        self.best_overall = (-1, -np.inf)   
        self._last_eval_step = 0

    def _evaluate(self, level):
        self.eval_env.curr.set_level(level)
        successes, rewards = [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=100_000 + ep)
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

        prev_sr, _ = self.best_per_level.get(level, (-np.inf, -np.inf))
        if success_rate > prev_sr:
            self.best_per_level[level] = (success_rate, mean_reward)
            path = os.path.join(self.save_path, f"eval_best_model_level_{level}")
            self.model.save(path)

        if success_rate >= self.min_success_to_count:
            candidate = (level, success_rate)
            if candidate > self.best_overall:
                self.best_overall = candidate
                path = os.path.join(self.save_path, "best_overall_model")
                self.model.save(path)

        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


# ----------------------- Helpers model loading & execution -----------------------
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
    model = load_model(SAC, "sac_nav_v5.1")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--pooling", choices=["meanmax", "attention"], default="meanmax")
    args = ap.parse_args()
    if args.play:
        play(pooling=args.pooling)
    elif args.eval:
        evaluate(n_episodes=args.episodes, level=args.level)
    else:
        train(pooling=args.pooling)