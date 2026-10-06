"""
nav_env_ppo.py — Environnement de navigation (obstacles statiques + dynamiques).

Corrections apportées suite à l'audit du run précédent (échec à 1.9M/2M steps) :

  1. FIX MARGE DE SPAWN (double comptage) :
     L'ancienne fonction `_min_obstacle_clearance()` unique ajoutait `2*robot_radius`
     à CHAQUE paire d'objets, y compris pour la paire (obstacle, robot) où le rayon
     du robot était déjà compté séparément via `orad` -> le rayon du robot était
     compté deux fois, rendant la contrainte de spawn artificiellement sévère.
     On distingue maintenant deux marges :
       - `_robot_clearance()`   : entre le robot et un obstacle (juste la marge de
         sécurité, les rayons étant déjà comptés séparément).
       - `_passage_clearance()` : entre deux obstacles (on ajoute le diamètre du
         robot pour garantir qu'un passage reste toujours franchissable).
     De plus, l'ancien `raise RuntimeError(...)` en cas d'échec de placement (qui
     tue le sous-processus SubprocVecEnv en plein entraînement — cause probable du
     crash à 1.9M/2M steps, moment où le nombre d'obstacles dynamiques atteint son
     maximum via le curriculum) est remplacé par une dégradation gracieuse : on
     relâche progressivement la contrainte de clearance, et en dernier recours on
     conserve la meilleure position trouvée (chevauchement minimal) plutôt que de
     faire planter l'environnement.

  2. REFONTE DE LA FONCTION DE RÉCOMPENSE :
     - L'ancienne pénalité d'immobilité était conditionnée à
       `speed < seuil AND min_lidar_dist > safety_margin` (basée sur le lidar
       générique). Elle punissait à tort un arrêt volontaire et légitime devant un
       obstacle dynamique qui traverse la trajectoire sans être encore à moins de
       `safety_margin`. Elle est remplacée par un critère TTC (time-to-collision)
       géométrique, SANS mémoire (donc non contournable par une pulsation
       périodique, contrairement à une moyenne glissante) : l'arrêt est exempté de
       pénalité seulement si un obstacle dynamique franchira la marge de sécurité
       dans les `ttc_horizon` secondes à venir.
     - La pénalité de proximité utilise désormais la distance géométrique directe
       aux obstacles (plus précise que le lidar, pas de bruit de discrétisation
       des rayons), avec poids plein sur les obstacles DYNAMIQUES
       (`safety_margin_weight`) et poids résiduel optionnel sur les statiques
       (`static_safety_margin_weight`, à 0.0 par défaut : l'évitement des statiques
       a déjà été validé en stage 0, donc le signal dense de prudence est
       volontairement concentré sur le risque réellement nouveau).
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class DynamicObstacle:
    """Obstacle cylindrique cinématique (mis à jour centralement par l'environnement)."""

    def __init__(self, body_id, pos_xy, vel_xy, radius, arena_half_size, z_center):
        self.body_id = body_id
        self.pos = np.array(pos_xy, dtype=np.float32)
        self.vel = np.array(vel_xy, dtype=np.float32)
        self.radius = radius
        self.limit = arena_half_size - radius
        self._z = z_center


class RobotNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        render_mode="none",
        urdf_path=None,
        arena_size=6.0,
        wall_height=0.5,
        wall_thickness=0.1,
        n_static_obstacles=8,
        obstacle_density=1.0,
        n_dynamic_obstacles=0,
        dynamic_obstacle_speed=0.4,
        static_obstacle_radius=0.18,
        dynamic_obstacle_radius=0.18,
        robot_radius=0.22,
        lidar_n_rays=36,
        lidar_range=4.0,
        lidar_height=0.18,
        goal_threshold=0.25,
        min_goal_dist=3.5,
        safety_margin=0.35,
        safety_margin_weight=0.2,
        static_safety_margin_weight=0.0,
        ttc_horizon=1.5,
        idle_speed_threshold=0.03,
        idle_penalty_weight=0.25,
        angular_penalty_weight=0.02,
        collision_penalty=60.0,
        success_bonus=100.0,
        time_penalty=0.05,
        smoothness_weight=0.05,
        max_wheel_speed=8.0,
        max_episode_steps=600,
        physics_hz=240,
        control_hz=20,
        gamma=0.99,
        seed=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.urdf_path = urdf_path or os.path.join(os.path.dirname(__file__), "robot.urdf")

        self.arena_size = float(arena_size)
        self.arena_half = self.arena_size / 2.0
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness

        self.n_static_obstacles_base = n_static_obstacles
        self.obstacle_density = float(np.clip(obstacle_density, 0.0, 1.0))
        self.n_dynamic_obstacles = int(n_dynamic_obstacles)
        self.dynamic_obstacle_speed = dynamic_obstacle_speed
        self.static_obstacle_radius = static_obstacle_radius
        self.dynamic_obstacle_radius = dynamic_obstacle_radius
        self.robot_radius = robot_radius

        self.lidar_n_rays = lidar_n_rays
        self.lidar_range = lidar_range
        self.lidar_height = lidar_height
        self.goal_threshold = goal_threshold
        self.min_goal_dist = min_goal_dist

        self.safety_margin = safety_margin
        self.safety_margin_weight = safety_margin_weight              # poids plein : obstacles dynamiques
        self.static_safety_margin_weight = static_safety_margin_weight  # poids résiduel (0.0 = désactivé) : statiques
        self.ttc_horizon = ttc_horizon                                  # horizon (s) pour l'exemption d'immobilité

        self.idle_speed_threshold = idle_speed_threshold
        self.idle_penalty_weight = idle_penalty_weight
        self.angular_penalty_weight = angular_penalty_weight
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.smoothness_weight = smoothness_weight

        self.max_wheel_speed = max_wheel_speed
        self.max_episode_steps = max_episode_steps

        self.physics_hz = physics_hz
        self.control_hz = control_hz
        self.n_substeps = max(1, physics_hz // control_hz)
        self.control_dt = 1.0 / control_hz

        self.gamma = gamma

        self._np_random = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        low = np.array([0.0] * self.lidar_n_rays + [-1.0, -1.0, 0.0] + [-1.0, -1.0] + [-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0] * self.lidar_n_rays + [1.0, 1.0, 1.0] + [1.0, 1.0] + [1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        if render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)

        self.plane_id = None
        self.robot_id = None
        self._left_joint = None
        self._right_joint = None

        self.wall_ids = []
        self.static_obstacle_ids = []
        self.dynamic_obstacles = []
        self.static_obstacles_info = []  # Stocke les infos (x, y, rayon)
        self._active_obstacle_ids = set()

        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.goal_marker_id = None

        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_dist = None
        self._step_count = 0

    def set_obstacle_density(self, density):
        self.obstacle_density = float(np.clip(density, 0.0, 1.0))

    def set_dynamic_obstacle_count(self, n):
        self.n_dynamic_obstacles = int(max(0, n))

    # ------------------------------------------------------------------ #
    # Marges géométriques de spawn (FIX double comptage)
    # ------------------------------------------------------------------ #
    def _robot_clearance(self):
        """Marge entre le robot et un obstacle : les deux rayons sont déjà comptés
        séparément (own_radius + orad) dans l'appelant, donc on n'ajoute ici QUE la
        marge de sécurité — pas de second comptage du rayon du robot."""
        return self.safety_margin + 0.15

    def _passage_clearance(self):
        """Marge entre deux obstacles (aucun des deux n'étant le robot) : on ajoute
        le diamètre du robot pour garantir qu'un passage reste toujours
        franchissable entre deux obstacles adjacents."""
        return 2.0 * self.robot_radius + self.safety_margin + 0.15

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        p.resetSimulation(physicsClientId=self._client)
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)

        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / self.physics_hz, physicsClientId=self._client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        robot_spawn_limit = self.arena_half - self.robot_radius - self.wall_thickness - 0.2
        start_x = float(self._np_random.uniform(-robot_spawn_limit, robot_spawn_limit))
        start_y = float(self._np_random.uniform(-robot_spawn_limit, robot_spawn_limit))
        start_pos = np.array([start_x, start_y], dtype=np.float32)

        start_yaw = float(self._np_random.uniform(-math.pi, math.pi))
        quat = p.getQuaternionFromEuler([0, 0, start_yaw])
        self.robot_id = p.loadURDF(
            self.urdf_path, [start_pos[0], start_pos[1], 0.06], quat, physicsClientId=self._client
        )

        self._left_joint, self._right_joint = None, None
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self._client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self._client)
            joint_name = info[1].decode("utf-8")
            if joint_name == "wheel_left_joint":
                self._left_joint = j
            elif joint_name == "wheel_right_joint":
                self._right_joint = j

        p.setJointMotorControl2(self.robot_id, self._left_joint, p.VELOCITY_CONTROL, targetVelocity=0, force=5.0, physicsClientId=self._client)
        p.setJointMotorControl2(self.robot_id, self._right_joint, p.VELOCITY_CONTROL, targetVelocity=0, force=5.0, physicsClientId=self._client)

        self.wall_ids = self._create_arena_walls()

        # occupied : liste de tuples (x, y, rayon, is_robot) — le flag is_robot
        # détermine quelle marge appliquer (_robot_clearance vs _passage_clearance).
        occupied = [(start_pos[0], start_pos[1], self.robot_radius, True)]
        self.static_obstacles_info = []

        self.goal_pos = self._sample_free_position(occupied, 0.2, min_dist_from_first=self.min_goal_dist)
        self.goal_marker_id = None
        if self.render_mode == "human":
            vis_goal = p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.15, length=0.02,
                rgbaColor=[0.1, 0.9, 0.2, 0.9], physicsClientId=self._client,
            )
            self.goal_marker_id = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis_goal,
                basePosition=[self.goal_pos[0], self.goal_pos[1], 0.01],
                physicsClientId=self._client,
            )

        n_static = int(round(self.n_static_obstacles_base * self.obstacle_density))
        self.static_obstacle_ids = []
        if n_static > 0:
            col_static = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=self.static_obstacle_radius, height=0.5,
                physicsClientId=self._client,
            )
            vis_static = p.createVisualShape(
                p.GEOM_CYLINDER, radius=self.static_obstacle_radius, length=0.5,
                rgbaColor=[0.75, 0.25, 0.2, 1], physicsClientId=self._client,
            )

            n_on_path = int(round(n_static * 0.7))

            for i in range(n_static):
                if i < n_on_path:
                    pos = self._sample_obstacle_on_path(start_pos, self.goal_pos, self.static_obstacle_radius, occupied)
                else:
                    pos = self._sample_free_position(occupied, self.static_obstacle_radius)

                occupied.append((pos[0], pos[1], self.static_obstacle_radius, False))
                self.static_obstacles_info.append((pos[0], pos[1], self.static_obstacle_radius))

                oid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_static,
                    baseVisualShapeIndex=vis_static,
                    basePosition=[pos[0], pos[1], 0.25],
                    physicsClientId=self._client,
                )
                self.static_obstacle_ids.append(oid)

        self.dynamic_obstacles = []
        if self.n_dynamic_obstacles > 0:
            col_dyn = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=self.dynamic_obstacle_radius, height=0.5,
                physicsClientId=self._client,
            )
            vis_dyn = p.createVisualShape(
                p.GEOM_CYLINDER, radius=self.dynamic_obstacle_radius, length=0.5,
                rgbaColor=[0.95, 0.6, 0.1, 1], physicsClientId=self._client,
            )

            n_dyn_on_path = int(round(self.n_dynamic_obstacles * 0.7))

            for i in range(self.n_dynamic_obstacles):
                if i < n_dyn_on_path:
                    pos = self._sample_obstacle_on_path(start_pos, self.goal_pos, self.dynamic_obstacle_radius, occupied)
                else:
                    pos = self._sample_free_position(occupied, self.dynamic_obstacle_radius)

                occupied.append((pos[0], pos[1], self.dynamic_obstacle_radius, False))
                oid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_dyn,
                    baseVisualShapeIndex=vis_dyn,
                    basePosition=[pos[0], pos[1], 0.25],
                    physicsClientId=self._client,
                )

                if i % 2 == 0:
                    vec_to_goal = self.goal_pos - start_pos
                    angle = math.atan2(vec_to_goal[1], vec_to_goal[0]) + (math.pi / 2.0) + float(self._np_random.uniform(-0.3, 0.3))
                else:
                    angle = float(self._np_random.uniform(-math.pi, math.pi))

                vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * self.dynamic_obstacle_speed
                self.dynamic_obstacles.append(
                    DynamicObstacle(oid, pos, vel, self.dynamic_obstacle_radius, self.arena_half, z_center=0.25)
                )

        self._active_obstacle_ids = set(self.wall_ids) | set(self.static_obstacle_ids) | {
            o.body_id for o in self.dynamic_obstacles
        }

        self._prev_dist = float(np.linalg.norm(self.goal_pos - start_pos))

        obs = self._get_observation()
        info = {}
        return obs, info

    def _sample_obstacle_on_path(self, start_pos, goal_pos, own_radius, occupied, max_tries=250):
        vec = goal_pos - start_pos
        dist = float(np.linalg.norm(vec))
        if dist < 0.5:
            return self._sample_free_position(occupied, own_radius)

        unit_vec = vec / dist
        perp_vec = np.array([-unit_vec[1], unit_vec[0]], dtype=np.float32)
        spawn_limit = self.arena_half - own_radius - self.wall_thickness

        for _ in range(max_tries):
            alpha = float(self._np_random.uniform(0.2, 0.8))
            base_pos = start_pos + alpha * vec
            offset_perp = float(self._np_random.uniform(-0.5, 0.5))
            offset_parallel = float(self._np_random.uniform(-0.2, 0.2))

            pos = base_pos + offset_perp * perp_vec + offset_parallel * unit_vec
            pos[0] = float(np.clip(pos[0], -spawn_limit, spawn_limit))
            pos[1] = float(np.clip(pos[1], -spawn_limit, spawn_limit))

            ok = True
            for ox, oy, orad, is_robot in occupied:
                clearance = self._robot_clearance() if is_robot else self._passage_clearance()
                min_d = own_radius + orad + clearance
                if (pos[0] - ox) ** 2 + (pos[1] - oy) ** 2 < min_d ** 2:
                    ok = False
                    break
            if ok:
                return pos

        # Après épuisement des tentatives "sur le chemin", on délègue au placement
        # libre (qui gère lui-même la dégradation progressive, voir plus bas).
        return self._sample_free_position(occupied, own_radius)

    def _sample_free_position(self, occupied, own_radius, min_dist_from_first=0.0, max_tries=400):
        """
        Place un objet sans chevauchement, avec relâchement progressif de la marge
        de clearance (scale 1.0 -> 0.5 -> 0.15) si l'arène est trop dense pour la
        contrainte stricte. En dernier recours (toutes les tentatives échouent même
        à clearance très réduite), on conserve la MEILLEURE position trouvée
        (chevauchement minimal) plutôt que de lever une exception : un crash ici se
        propage au sous-processus SubprocVecEnv et interrompt tout l'entraînement.
        """
        spawn_limit = self.arena_half - own_radius - self.wall_thickness

        def clearance_for(is_robot):
            return self._robot_clearance() if is_robot else self._passage_clearance()

        best_pos = None
        best_margin = -float("inf")

        for scale in (1.0, 0.5, 0.15):
            for _ in range(max_tries):
                x = float(self._np_random.uniform(-spawn_limit, spawn_limit))
                y = float(self._np_random.uniform(-spawn_limit, spawn_limit))

                ok = True
                min_margin = float("inf")
                for i, (ox, oy, orad, is_robot) in enumerate(occupied):
                    min_d = own_radius + orad + clearance_for(is_robot) * scale
                    if i == 0 and min_dist_from_first > 0:
                        min_d = max(min_d, min_dist_from_first)
                    margin = math.hypot(x - ox, y - oy) - min_d
                    min_margin = min(min_margin, margin)
                    if margin < 0:
                        ok = False

                if ok:
                    return np.array([x, y], dtype=np.float32)
                if min_margin > best_margin:
                    best_margin = min_margin
                    best_pos = np.array([x, y], dtype=np.float32)

        print(
            f"[RobotNavEnv] Attention: placement en dégradé, chevauchement résiduel "
            f"(marge={best_margin:.3f} m). Arène probablement trop dense pour le "
            f"nombre d'obstacles demandé."
        )
        return best_pos if best_pos is not None else np.array([0.0, 0.0], dtype=np.float32)

    def _create_arena_walls(self):
        wall_ids = []
        t = self.wall_thickness
        h = self.wall_height
        half_len = self.arena_half + t

        wall_configs = [
            ([half_len, t / 2.0, h / 2.0], 0.0, self.arena_half + t / 2.0),
            ([half_len, t / 2.0, h / 2.0], 0.0, -self.arena_half - t / 2.0),
            ([t / 2.0, half_len, h / 2.0], self.arena_half + t / 2.0, 0.0),
            ([t / 2.0, half_len, h / 2.0], -self.arena_half - t / 2.0, 0.0),
        ]

        for half_extents, px, py in wall_configs:
            col_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._client)
            vis_box = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.4, 0.4, 0.45, 1.0], physicsClientId=self._client)
            wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box, baseVisualShapeIndex=vis_box, basePosition=[px, py, h / 2.0], physicsClientId=self._client)
            wall_ids.append(wall_id)
        return wall_ids

    def _update_dynamic_obstacles(self):
        for obs in self.dynamic_obstacles:
            obs.pos += obs.vel * self.control_dt

        for obs in self.dynamic_obstacles:
            for axis in range(2):
                if obs.pos[axis] > obs.limit:
                    obs.pos[axis] = obs.limit
                    obs.vel[axis] *= -1.0
                elif obs.pos[axis] < -obs.limit:
                    obs.pos[axis] = -obs.limit
                    obs.vel[axis] *= -1.0

        for obs in self.dynamic_obstacles:
            for sx, sy, s_rad in self.static_obstacles_info:
                spos = np.array([sx, sy], dtype=np.float32)
                dist = float(np.linalg.norm(obs.pos - spos))
                min_dist = obs.radius + s_rad

                if dist < min_dist:
                    n = (obs.pos - spos) / (dist + 1e-6)
                    obs.pos = spos + n * min_dist
                    obs.vel = obs.vel - 2.0 * np.dot(obs.vel, n) * n

        for i in range(len(self.dynamic_obstacles)):
            for j in range(i + 1, len(self.dynamic_obstacles)):
                obs1 = self.dynamic_obstacles[i]
                obs2 = self.dynamic_obstacles[j]
                dist = float(np.linalg.norm(obs1.pos - obs2.pos))
                min_dist = obs1.radius + obs2.radius

                if dist < min_dist:
                    n = (obs1.pos - obs2.pos) / (dist + 1e-6)
                    overlap = min_dist - dist

                    obs1.pos += n * (overlap / 2.0)
                    obs2.pos -= n * (overlap / 2.0)

                    v1n = np.dot(obs1.vel, n)
                    v2n = np.dot(obs2.vel, n)

                    if v1n - v2n < 0:
                        obs1.vel += (v2n - v1n) * n
                        obs2.vel += (v1n - v2n) * n

        for obs in self.dynamic_obstacles:
            p.resetBasePositionAndOrientation(
                obs.body_id,
                [float(obs.pos[0]), float(obs.pos[1]), obs._z],
                [0, 0, 0, 1],
                physicsClientId=self._client,
            )

    # ------------------------------------------------------------------ #
    # Distances géométriques directes (utilisées par la fonction de récompense)
    # ------------------------------------------------------------------ #
    def _nearest_dynamic_obstacle_dist(self, robot_pos_xy):
        """Distance bord-à-bord (négative si chevauchement) au dynamique le plus
        proche. +inf si aucun obstacle dynamique n'est actif."""
        if not self.dynamic_obstacles:
            return float("inf")
        return min(
            float(np.linalg.norm(robot_pos_xy - obs.pos)) - obs.radius - self.robot_radius
            for obs in self.dynamic_obstacles
        )

    def _nearest_static_obstacle_dist(self, robot_pos_xy):
        """Distance bord-à-bord au statique le plus proche. +inf si aucun."""
        if not self.static_obstacles_info:
            return float("inf")
        return min(
            math.hypot(robot_pos_xy[0] - sx, robot_pos_xy[1] - sy) - s_rad - self.robot_radius
            for sx, sy, s_rad in self.static_obstacles_info
        )

    def _dynamic_obstacle_incoming(self, robot_pos_xy):
        """
        Vrai si au moins un obstacle dynamique va franchir la marge de sécurité
        dans les `ttc_horizon` secondes à venir (extrapolation à vitesse constante,
        cohérente avec la physique des DynamicObstacle entre deux rebonds).

        Remplace l'ancien critère `min_lidar_dist > safety_margin` : c'est un
        calcul SANS mémoire (aucun état accumulé sur plusieurs steps), donc non
        contournable par une pulsation périodique (contrairement à une moyenne
        glissante de progrès) — l'agent ne peut pas "réinitialiser" un historique
        puisqu'il n'y en a pas.
        """
        min_dist = self.robot_radius + self.dynamic_obstacle_radius + self.safety_margin
        for obs in self.dynamic_obstacles:
            rel_pos = obs.pos - robot_pos_xy
            dist = float(np.linalg.norm(rel_pos))
            if dist <= min_dist:
                return True
            closing_speed = -float(np.dot(obs.vel, rel_pos)) / (dist + 1e-6)
            if closing_speed <= 0:
                continue  # l'obstacle s'éloigne : pas de danger imminent
            ttc = (dist - min_dist) / closing_speed
            if ttc <= self.ttc_horizon:
                return True
        return False

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        left_target = float(action[0]) * self.max_wheel_speed
        right_target = float(action[1]) * self.max_wheel_speed
        p.setJointMotorControl2(self.robot_id, self._left_joint, p.VELOCITY_CONTROL, targetVelocity=left_target, force=5.0, physicsClientId=self._client)
        p.setJointMotorControl2(self.robot_id, self._right_joint, p.VELOCITY_CONTROL, targetVelocity=right_target, force=5.0, physicsClientId=self._client)

        for _ in range(self.n_substeps):
            p.stepSimulation(physicsClientId=self._client)

        self._update_dynamic_obstacles()

        self._step_count += 1

        obs_vec = self._get_observation()
        dist_to_goal = self._current_goal_dist()
        collided = self._check_collision()

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        pos_xy = np.array(robot_pos[:2], dtype=np.float32)

        terminated = False
        truncated = False
        reward = 0.0

        reward += 2.0 * (self._prev_dist - dist_to_goal)
        self._prev_dist = dist_to_goal

        reward -= self.time_penalty
        reward -= self.smoothness_weight * float(np.linalg.norm(action - self._prev_action))
        self._prev_action = action.copy()

        # --- Marge de sécurité géométrique (FIX : ciblée dynamique, poids plein) --- #
        dyn_dist = self._nearest_dynamic_obstacle_dist(pos_xy)
        if dyn_dist < self.safety_margin:
            proximity = (self.safety_margin - dyn_dist) / self.safety_margin
            reward -= self.safety_margin_weight * proximity

        # Poids résiduel optionnel sur les statiques (0.0 par défaut : l'évitement
        # des statiques est déjà validé en stage 0, signal volontairement concentré
        # sur le risque dynamique). Mets static_safety_margin_weight > 0 si le
        # warmup se dégrade sans ce signal.
        if self.static_safety_margin_weight > 0.0:
            static_dist = self._nearest_static_obstacle_dist(pos_xy)
            if static_dist < self.safety_margin:
                proximity = (self.safety_margin - static_dist) / self.safety_margin
                reward -= self.static_safety_margin_weight * proximity

        min_lidar_dist = float(np.min(obs_vec[: self.lidar_n_rays])) * self.lidar_range  # conservé pour info/monitoring

        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        speed = float(np.linalg.norm(lin_vel[:2]))

        # --- Pénalité d'immobilité (FIX : critère TTC sans mémoire) --- #
        if speed < self.idle_speed_threshold and not self._dynamic_obstacle_incoming(pos_xy):
            reward -= self.idle_penalty_weight

        reward -= self.angular_penalty_weight * abs(ang_vel[2])

        if dist_to_goal < self.goal_threshold:
            reward += self.success_bonus
            terminated = True
        elif collided:
            reward -= self.collision_penalty
            terminated = True

        if self._step_count >= self.max_episode_steps:
            truncated = True

        info = {
            "dist_to_goal": dist_to_goal,
            "collided": collided,
            "min_lidar_dist": min_lidar_dist,
            "nearest_dynamic_dist": dyn_dist,
            "n_dynamic_obstacles": self.n_dynamic_obstacles,
        }
        return obs_vec, reward, terminated, truncated, info

    def _current_goal_dist(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        return float(np.linalg.norm(self.goal_pos - np.array(robot_pos[:2], dtype=np.float32)))

    def _check_collision(self):
        contacts = p.getContactPoints(bodyA=self.robot_id, physicsClientId=self._client)
        for c in contacts:
            if c[2] in self._active_obstacle_ids:
                return True
        return False

    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        pos_xy = np.array(pos[:2], dtype=np.float32)

        lidar = self._get_lidar(pos_xy, yaw)

        goal_vec_world = self.goal_pos - pos_xy
        dist = float(np.linalg.norm(goal_vec_world))
        angle_world = math.atan2(goal_vec_world[1], goal_vec_world[0])
        angle_robot = _wrap_angle(angle_world - yaw)
        goal_vec = np.array(
            [math.cos(angle_robot), math.sin(angle_robot), min(dist / self.lidar_range, 1.0)],
            dtype=np.float32,
        )

        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        forward_speed = float(np.dot(np.array(lin_vel[:2], dtype=np.float32), forward))
        vel_vec = np.array(
            [np.clip(forward_speed / 1.0, -1.0, 1.0), np.clip(ang_vel[2] / 3.0, -1.0, 1.0)],
            dtype=np.float32,
        )

        obs = np.concatenate([lidar, goal_vec, vel_vec, self._prev_action]).astype(np.float32)
        return obs

    def _get_lidar(self, pos_xy, yaw):
        z = self.lidar_height
        angles = yaw + np.linspace(0, 2 * math.pi, self.lidar_n_rays, endpoint=False)
        from_pts = [[pos_xy[0], pos_xy[1], z]] * self.lidar_n_rays
        to_pts = [
            [
                pos_xy[0] + self.lidar_range * math.cos(a),
                pos_xy[1] + self.lidar_range * math.sin(a),
                z,
            ]
            for a in angles
        ]
        results = p.rayTestBatch(from_pts, to_pts, physicsClientId=self._client)
        hit_fractions = np.array([r[2] for r in results], dtype=np.float32)
        return hit_fractions

    def render(self):
        pass

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


def _wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
