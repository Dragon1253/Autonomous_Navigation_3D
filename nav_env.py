"""
nav_env.py — Environnement Gymnasium 3D (PyBullet) pour navigation avec évitement
d'obstacles (Statiques + Dynamiques positionnés stratégiquement sur la trajectoire).
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class DynamicObstacle:
    """Obstacle cylindrique cinématique : translation à vitesse constante,
    rebond élastique sur les bords carrés de l'arène."""

    def __init__(self, body_id, pos_xy, vel_xy, radius, arena_half_size, z_center):
        self.body_id = body_id
        self.pos = np.array(pos_xy, dtype=np.float32)   # (x, y)
        self.vel = np.array(vel_xy, dtype=np.float32)   # (vx, vy) m/s
        self.radius = radius
        self.limit = arena_half_size - radius
        self._z = z_center

    def step(self, dt, client_id):
        self.pos += self.vel * dt
        for axis in range(2):
            if self.pos[axis] > self.limit:
                self.pos[axis] = self.limit
                self.vel[axis] *= -1.0
            elif self.pos[axis] < -self.limit:
                self.pos[axis] = -self.limit
                self.vel[axis] *= -1.0
        p.resetBasePositionAndOrientation(
            self.body_id,
            [float(self.pos[0]), float(self.pos[1]), self._z],
            [0, 0, 0, 1],
            physicsClientId=client_id,
        )


class RobotNavEnv(gym.Env):
    """Env Gymnasium : robot différentiel, lidar 36 rayons, obstacles statiques
    + dynamiques + murs d'enceinte, objectif ponctuel donné en coordonnées."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        render_mode="none",
        urdf_path=None,
        arena_size=6.0,                # arène carrée, côté = arena_size
        wall_height=0.5,               # Hauteur des murs
        wall_thickness=0.1,            # Épaisseur des murs
        n_static_obstacles=8,          # Augmenté à 8 par défaut
        obstacle_density=1.0,          
        n_dynamic_obstacles=0,         
        dynamic_obstacle_speed=0.4,    # m/s
        static_obstacle_radius=0.18,
        dynamic_obstacle_radius=0.18,
        robot_radius=0.22,             
        lidar_n_rays=24, 
        min_goal_dist=3.5,              # Augmenté à 36 (10° par rayon)
        lidar_range=4.0,
        lidar_height=0.18,             
        goal_threshold=0.25,
        safety_margin=0.35,            
        safety_margin_weight=1.0,      
        idle_speed_threshold=0.03,     
        idle_penalty_weight=0.25,      # Pénalité d'immobilité augmentée
        collision_penalty=30.0,        # Ajusté pour équilibrer l'exploration serrée
        success_bonus=100.0,
        time_penalty=0.01,
        smoothness_weight=0.05,
        max_wheel_speed=8.0,           # rad/s
        max_episode_steps=500,
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

        self.safety_margin = safety_margin
        self.safety_margin_weight = safety_margin_weight
        self.idle_speed_threshold = idle_speed_threshold
        self.idle_penalty_weight = idle_penalty_weight
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.smoothness_weight = smoothness_weight
        self.min_goal_dist = min_goal_dist

        self.max_wheel_speed = max_wheel_speed
        self.max_episode_steps = max_episode_steps

        self.physics_hz = physics_hz
        self.control_hz = control_hz
        self.n_substeps = max(1, physics_hz // control_hz)
        self.control_dt = 1.0 / control_hz

        self.gamma = gamma

        self._np_random = np.random.default_rng(seed)

        # --- Gymnasium spaces ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        low = np.array([0.0] * self.lidar_n_rays + [-1.0, -1.0, 0.0] + [-1.0, -1.0] + [-1.0, -1.0],
                        dtype=np.float32)
        high = np.array([1.0] * self.lidar_n_rays + [1.0, 1.0, 1.0] + [1.0, 1.0] + [1.0, 1.0],
                         dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- PyBullet client ---
        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        self.plane_id = None
        self.robot_id = None
        self._left_joint = None
        self._right_joint = None

        self.wall_ids = []
        self.static_obstacle_ids = []
        self.dynamic_obstacles = []  
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / self.physics_hz, physicsClientId=self._client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        # --- 1. Position de départ aléatoire du robot ---
        # Spawn limit pour le robot
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

        p.setJointMotorControl2(self.robot_id, self._left_joint, p.VELOCITY_CONTROL,
                                 targetVelocity=0, force=5.0, physicsClientId=self._client)
        p.setJointMotorControl2(self.robot_id, self._right_joint, p.VELOCITY_CONTROL,
                                 targetVelocity=0, force=5.0, physicsClientId=self._client)

        # --- Création des Murs d'enceinte ---
        self.wall_ids = self._create_arena_walls()

        occupied = [(start_pos[0], start_pos[1], self.robot_radius)]

        # --- 2. Objectif distant (garanti à au moins self.min_goal_dist du robot) ---
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

        # --- 3. Obstacles statiques générés sur le chemin départ -> goal ---
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
                    # Utilise start_pos au lieu de (0,0)
                    pos = self._sample_obstacle_on_path(start_pos, self.goal_pos, self.static_obstacle_radius, occupied)
                else:
                    pos = self._sample_free_position(occupied, self.static_obstacle_radius)

                occupied.append((pos[0], pos[1], self.static_obstacle_radius))
                oid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_static,
                    baseVisualShapeIndex=vis_static,
                    basePosition=[pos[0], pos[1], 0.25],
                    physicsClientId=self._client,
                )
                self.static_obstacle_ids.append(oid)

        # --- 4. Obstacles dynamiques ---
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

                occupied.append((pos[0], pos[1], self.dynamic_obstacle_radius))
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
                    DynamicObstacle(oid, pos, vel, self.dynamic_obstacle_radius,
                                     self.arena_half, z_center=0.25)
                )

        self._active_obstacle_ids = set(self.wall_ids) | set(self.static_obstacle_ids) | {
            o.body_id for o in self.dynamic_obstacles
        }

        self._prev_dist = float(np.linalg.norm(self.goal_pos - start_pos))

        obs = self._get_observation()
        info = {}
        return obs, info

    def _sample_obstacle_on_path(self, start_pos, goal_pos, own_radius, occupied, max_tries=100):
        """Génère un obstacle directement entre le point de départ et l'objectif."""
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
            for ox, oy, orad in occupied:
                if (pos[0] - ox) ** 2 + (pos[1] - oy) ** 2 < (own_radius + orad + self.safety_margin * 0.5) ** 2:
                    ok = False
                    break
            if ok:
                return pos

        return self._sample_free_position(occupied, own_radius)

    def _sample_free_position(self, occupied, own_radius, min_dist_from_first=0.0, max_tries=200):
        margin = max(0.08, self.robot_radius + self.safety_margin + 0.05)
        spawn_limit = self.arena_half - own_radius - self.wall_thickness

        for _ in range(max_tries):
            x = float(self._np_random.uniform(-spawn_limit, spawn_limit))
            y = float(self._np_random.uniform(-spawn_limit, spawn_limit))
            ok = True
            for i, (ox, oy, orad) in enumerate(occupied):
                min_d = own_radius + orad + margin
                if i == 0 and min_dist_from_first > 0:
                    min_d = max(min_d, min_dist_from_first)
                if (x - ox) ** 2 + (y - oy) ** 2 < min_d ** 2:
                    ok = False
                    break
            if ok:
                return np.array([x, y], dtype=np.float32)

        fallback_margin = self.robot_radius + (self.safety_margin * 0.5)
        for _ in range(max_tries):
            x = float(self._np_random.uniform(-spawn_limit, spawn_limit))
            y = float(self._np_random.uniform(-spawn_limit, spawn_limit))
            ok = True
            for i, (ox, oy, orad) in enumerate(occupied):
                min_d = own_radius + orad + fallback_margin
                if i == 0 and min_dist_from_first > 0:
                    min_d = max(min_d, min_dist_from_first)
                if (x - ox) ** 2 + (y - oy) ** 2 < min_d ** 2:
                    ok = False
                    break
            if ok:
                return np.array([x, y], dtype=np.float32)

        return np.array(
            [self._np_random.uniform(-spawn_limit, spawn_limit),
             self._np_random.uniform(-spawn_limit, spawn_limit)],
            dtype=np.float32,
        )

    def _create_arena_walls(self):
        """Génère 4 murs fermant le terrain carré."""
        wall_ids = []
        t = self.wall_thickness
        h = self.wall_height
        half_len = self.arena_half + t

        wall_configs = [
            ([half_len, t / 2.0, h / 2.0], 0.0, self.arena_half + t / 2.0),   # Nord
            ([half_len, t / 2.0, h / 2.0], 0.0, -self.arena_half - t / 2.0),  # Sud
            ([t / 2.0, half_len, h / 2.0], self.arena_half + t / 2.0, 0.0),   # Est
            ([t / 2.0, half_len, h / 2.0], -self.arena_half - t / 2.0, 0.0),  # Ouest
        ]

        for half_extents, px, py in wall_configs:
            col_box = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._client
            )
            vis_box = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.4, 0.4, 0.45, 1.0],
                physicsClientId=self._client
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_box,
                baseVisualShapeIndex=vis_box,
                basePosition=[px, py, h / 2.0],
                physicsClientId=self._client,
            )
            wall_ids.append(wall_id)
        return wall_ids

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        left_target = float(action[0]) * self.max_wheel_speed
        right_target = float(action[1]) * self.max_wheel_speed
        p.setJointMotorControl2(
            self.robot_id, self._left_joint, p.VELOCITY_CONTROL,
            targetVelocity=left_target, force=5.0, physicsClientId=self._client,
        )
        p.setJointMotorControl2(
            self.robot_id, self._right_joint, p.VELOCITY_CONTROL,
            targetVelocity=right_target, force=5.0, physicsClientId=self._client,
        )

        for _ in range(self.n_substeps):
            p.stepSimulation(physicsClientId=self._client)

        for obs in self.dynamic_obstacles:
            obs.step(self.control_dt, self._client)

        self._step_count += 1

        obs_vec = self._get_observation()
        dist_to_goal = self._current_goal_dist()

        collided = self._check_collision()

        terminated = False
        truncated = False
        reward = 0.0

        reward += (self._prev_dist - self.gamma * dist_to_goal)
        self._prev_dist = dist_to_goal

        reward -= self.time_penalty
        reward -= self.smoothness_weight * float(np.linalg.norm(action - self._prev_action))
        self._prev_action = action.copy()

        min_lidar_dist = float(np.min(obs_vec[: self.lidar_n_rays])) * self.lidar_range
        if min_lidar_dist < self.safety_margin:
            proximity = (self.safety_margin - min_lidar_dist) / self.safety_margin
            reward -= self.safety_margin_weight * proximity

        lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        speed = float(np.linalg.norm(lin_vel[:2]))
        if speed < self.idle_speed_threshold:
            reward -= self.idle_penalty_weight

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