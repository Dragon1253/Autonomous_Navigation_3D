"""
Banc de test minimal : evitement d'obstacles DYNAMIQUES uniquement.
Adapte de sac_train.py pour isoler le probleme avant de reintroduire le statique.

Changements cles vs sac_train.py :
1. Zero obstacle statique (le curriculum ne porte que sur les obstacles dynamiques).
2. Arene reduite -> episodes plus courts, iteration plus rapide.
3. Observation enrichie : position + vitesse relative (repere robot) des K obstacles
   dynamiques les plus proches, EN PLUS du lidar.
   -> Sans cette info, le reseau doit inferer la vitesse des obstacles a partir de
   variations de distance lidar polluees par le mouvement propre du robot (rotation +
   translation a chaque step) : probleme de desambiguisation mouvement-propre /
   mouvement-obstacle, tres difficile a apprendre implicitement. C'est la raison la
   plus probable de l'echec commun a SAC/PPO/RecurrentPPO/FrameStack observe jusqu'ici.
4. Mouvement des obstacles moins previsible : perturbation aleatoire periodique de la
   direction (pas seulement rebond sur les murs), pour eviter que l'agent apprenne une
   simple extrapolation balistique au lieu d'une vraie reaction.
5. Recompense : penalite TTC (time-to-collision) basee sur la vitesse de rapprochement
   REELLE (produit scalaire vitesse relative . direction obstacle), pas seulement sur
   la distance brute -> ne penalise que les obstacles qui se rapprochent vraiment.

Les classes DynOnlyNavEnv / DynOnlyCurriculumManager respectent la meme interface
Gym que l'environnement original : tu peux les importer directement dans tes scripts
train_ppo.py / train_recurrent_ppo.py a la place de NavEnv/CurriculumManager pour
comparer les architectures sur exactement le meme banc de test.

Entrainement : python dyn_only_train.py
Test visuel  : python dyn_only_train.py --play
Eval fixe    : python dyn_only_train.py --eval
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


# ----------------------- Curriculum dynamique-only -----------------------
class DynOnlyCurriculumManager:
    """Passe au niveau suivant si succes >= seuil sur une fenetre glissante.
    Paliers ordonnes par PREVISIBILITE croissante, pas seulement par difficulte brute :
    on commence par du balistique pur (rebond mur uniquement) avant d'introduire des
    changements de direction aleatoires qui forcent une vraie reaction."""

    LEVELS = [
        # (n_dyn min/max, vitesse min/max, proba de perturbation direction /s, dist but min)
        dict(n_dyn=(1, 1), speed=(0.15, 0.25), perturb_prob=0.00, min_goal=3.0),
        dict(n_dyn=(1, 1), speed=(0.30, 0.45), perturb_prob=0.00, min_goal=3.5),
        dict(n_dyn=(1, 1), speed=(0.30, 0.50), perturb_prob=0.15, min_goal=3.5),
        dict(n_dyn=(2, 2), speed=(0.30, 0.50), perturb_prob=0.15, min_goal=4.0),
        dict(n_dyn=(2, 3), speed=(0.35, 0.60), perturb_prob=0.20, min_goal=4.0),
        # Niveau final : randomisation de domaine -> generalisation
        dict(n_dyn=(1, 4), speed=(0.20, 0.80), perturb_prob=0.25, min_goal=4.5),
        dict(n_dyn=(8, 10), speed=(0.20, 0.80), perturb_prob=0.25, min_goal=4.5),

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
class DynOnlyNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ARENA = 4.0          # demi-taille de l'arene (8 x 8 m) : reduite pour iterer vite
    R_ROBOT = 0.25
    V_MAX = 1.0           # m/s
    W_MAX = 2.0           # rad/s
    DT = 1.0 / 30.0
    N_RAYS = 24
    RAY_RANGE = 4.0
    GOAL_RADIUS = 0.4
    MAX_STEPS = 400
    K_NEAREST = 2         # nb d'obstacles dynamiques dont on donne pos+vitesse relatives
    V_OBS_NORM = 2.0       # normalisation des vitesses relatives (robot + obstacle <= ~2 m/s)

    def __init__(self, curriculum: DynOnlyCurriculumManager, render_mode=None):
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
        if clearance is None:
            clearance = 2 * self.R_ROBOT + 0.15
        if np.max(np.abs(pos)) > self.ARENA - r - 0.15:
            return False
        for o in others:
            if np.linalg.norm(pos - o["pos"]) < r + o["r"] + clearance:
                return False
        return True

    def _spawn_obstacles(self, n_dyn, speed_rng, perturb_prob, start, goal):
        for o in self.obstacles:
            p.removeBody(o["id"], physicsClientId=self.cid)
        self.obstacles = []
        seg = goal - start
        # ratio eleve sur le trajet : on veut forcer la reaction, pas des episodes vides
        n_on_path = max(1, int(0.7 * n_dyn))
        min_pass_gap = 2 * self.R_ROBOT + 0.15

        for i in range(n_dyn):
            r = float(self.rng.uniform(0.25, 0.45))
            for _ in range(200):
                if i < n_on_path:
                    t = self.rng.uniform(0.2, 0.8)
                    perp = np.array([-seg[1], seg[0]])
                    perp /= (np.linalg.norm(perp) + 1e-8)
                    pos = start + t * seg + perp * self.rng.uniform(-1.0, 1.0)
                else:
                    pos = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)
                ok = (self._free(pos, r, self.obstacles, clearance=min_pass_gap)
                      and np.linalg.norm(pos - start) > r + self.R_ROBOT + 1.0
                      and np.linalg.norm(pos - goal) > r + self.GOAL_RADIUS + 0.6)
                if ok:
                    break
            else:
                continue
            oid = self._make_cylinder(r, 0.6, [0.9, 0.2, 0.2, 1])
            p.resetBasePositionAndOrientation(oid, [*pos, 0.3], [0, 0, 0, 1],
                                              physicsClientId=self.cid)
            ang = self.rng.uniform(0, 2 * np.pi)
            s = self.rng.uniform(*speed_rng)
            vel = s * np.array([np.cos(ang), np.sin(ang)])
            self.obstacles.append(dict(id=oid, pos=pos, vel=vel, r=r, dynamic=True,
                                       perturb_prob=perturb_prob))

    # ---------- dynamique des obstacles : rebonds + perturbation aleatoire ----------
    def _move_obstacles(self):
        lim = self.ARENA - 0.15
        for o in self.obstacles:
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
        # rebond obstacle <-> obstacle
        for i, a in enumerate(self.obstacles):
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
        dyn = sorted(self.obstacles, key=lambda o: np.linalg.norm(o["pos"] - self.pos))
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
        while len(feats) < self.K_NEAREST * 4:   # zero-padding si moins de K obstacles
            feats.append(0.0)
        return feats

    def _obs(self):
        to_goal = self.goal - self.pos
        d = np.linalg.norm(to_goal)
        ang = math.atan2(to_goal[1], to_goal[0]) - self.theta
        base = [min(d / (2 * self.ARENA * 1.5), 1.0), math.cos(ang), math.sin(ang),
               self.last_a[0], self.last_a[1]]
        return np.concatenate([
            base, self._dynamic_obstacle_features(), self._lidar()
        ]).astype(np.float32)

    # ---------- penalite time-to-collision (vitesse de rapprochement reelle) ----------
    def _ttc_penalty(self, v_robot_world, ttc_threshold=2.5, weight=0.4):
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
        min_goal = prm["min_goal"] if self.rng.random() < 0.8 else 2.0
        for _ in range(200):
            self.pos = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)
            self.goal = self.rng.uniform(-self.ARENA + 1, self.ARENA - 1, 2)
            if np.linalg.norm(self.goal - self.pos) >= min_goal:
                break
        self.theta = float(self.rng.uniform(-np.pi, np.pi))
        n_d = int(self.rng.integers(prm["n_dyn"][0], prm["n_dyn"][1] + 1))
        self._spawn_obstacles(n_d, prm["speed"], prm["perturb_prob"], self.pos, self.goal)
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
        reward = 20.0 * (self.prev_d - d) - 0.05
        self.prev_d = d
        obs = self._obs()
        min_ray = float(np.min(obs[self.OBS_BASE:])) * self.RAY_RANGE
        if min_ray < 0.5:
            reward -= 0.15 * (0.5 - min_ray) / 0.5      # marge de securite generale
        reward -= self._ttc_penalty(v_robot_world)        # terme specifique "se rapproche vite"

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
        return obs, float(reward), terminated, truncated, {"success": success}

    def close(self):
        p.disconnect(physicsClientId=self.cid)


# ----------------------- Callback de suivi -----------------------
class CurriculumLogCallback(BaseCallback):
    def __init__(self, curriculum):
        super().__init__()
        self.curr = curriculum

    def _on_step(self):
        if self.n_calls % 10_000 == 0:
            sr = np.mean(self.curr.history) if self.curr.history else 0.0
            print(f"steps={self.num_timesteps} niveau={self.curr.level} "
                  f"succes_fenetre={sr:.2f}")
        return True


# ----------------------- Chargement modele (contournement bug PyTorch) -----------------------
def resolve_model_path(model_path):
    if not os.path.exists(model_path) and os.path.exists(f"{model_path}.zip"):
        return f"{model_path}.zip"
    return model_path

def load_sac(model_path):
    model_path = resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modele est introuvable : {model_path}")
    try:
        return SAC.load(model_path)
    except RuntimeError as error:
        if "PytorchStreamReader failed reading file" not in str(error):
            raise
        print("[INFO] Chargement direct impossible avec cette version de PyTorch.")
        print("[INFO] Nouvelle tentative en lisant les poids depuis la memoire...")
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


# ----------------------- Entrainement / test / eval -----------------------
def train(total_timesteps=500_000):
    curr = DynOnlyCurriculumManager()
    env = Monitor(DynOnlyNavEnv(curr))
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=500_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=5_000, ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1, tensorboard_log="./tb_dynonly",
    )
    model.learn(total_timesteps=total_timesteps, callback=CurriculumLogCallback(curr))
    model.save("sac_dynonly_final")
    env.close()

def play():
    curr = DynOnlyCurriculumManager()
    curr.level = len(DynOnlyCurriculumManager.LEVELS) - 1
    env = DynOnlyNavEnv(curr, render_mode="human")
    model = load_sac("sac_dynonly_final")
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

def evaluate(n_episodes=50, base_seed=10_000):
    """Evaluation a seeds fixes, niveau max : sert de reference commune pour
    comparer objectivement SAC/PPO/RecurrentPPO/FrameStack sur le meme banc de test."""
    curr = DynOnlyCurriculumManager()
    curr.level = len(DynOnlyCurriculumManager.LEVELS) - 1
    env = DynOnlyNavEnv(curr)
    model = load_sac("sac_dynonly_final")
    successes, steps_list = 0, []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        done, info = False, {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        successes += int(info.get("success", False))
        steps_list.append(env.steps)
    env.close()
    print(f"Taux de succes : {successes / n_episodes:.2%} sur {n_episodes} episodes "
          f"(seeds fixes, niveau max)")
    print(f"Duree moyenne : {np.mean(steps_list):.0f} steps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    args = ap.parse_args()
    if args.play:
        play()
    elif args.eval:
        evaluate()
    else:
        train(total_timesteps=args.timesteps)