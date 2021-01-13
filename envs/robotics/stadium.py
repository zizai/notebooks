import os
import time

import gym
import math
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

from neuroblast.sim.cheetah import MiniCheetah
from chaosbreaker.spaces import FloatBox

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
NUM_SIMULATION_ITERATION_STEPS = 300


class CheetahEnv(gym.Env):
    def __init__(self,
                 action_repeat=1,
                 control_latency=0.005,
                 control_time_step=None,
                 objective_weight_distance=1.0,
                 objective_weight_drift=0.0,
                 objective_weight_energy=0.005,
                 objective_weight_shake=0.0,
                 render=False,
                 max_num_steps=1000,
                 urdf_root=pybullet_data.getDataPath()):
        self.max_num_steps = max_num_steps
        self._control_latency = control_latency
        self._distance_limit = float("inf")
        self._is_render = render
        self._last_frame_time = 0.0
        self._objective_weights = [objective_weight_distance, objective_weight_energy, objective_weight_drift, objective_weight_shake]
        self._urdf_root = urdf_root

        # PD control needs smaller time step for stability.
        if control_time_step is not None:
            self.control_time_step = control_time_step
            self._action_repeat = action_repeat
            self._time_step = control_time_step / action_repeat
        else:
            # Default values for simulation time step and action repeat
            self._time_step = 0.002
            self._action_repeat = 5
            self.control_time_step = self._time_step * self._action_repeat

        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._objectives = []

        self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)

        if self._is_render:
            self._pybullet_server = p.connect(p.GUI_SERVER)
            self._pybullet_client = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
        else:
            self._pybullet_server = p.connect(p.SHARED_MEMORY_SERVER)
            self._pybullet_client = bc.BulletClient(connection_mode=p.SHARED_MEMORY)

        self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.scene = StadiumScene(self._pybullet_client)
        self.cheetah = MiniCheetah(self._pybullet_client,
                                   action_repeat=self._action_repeat,
                                   control_latency=self._control_latency,
                                   time_step=self._time_step)
        self.reset()

        observation_high = (self._get_observation_upper_bound() + OBSERVATION_EPS)
        observation_low = (self._get_observation_lower_bound() - OBSERVATION_EPS)
        action_dim = self.cheetah.num_motors
        action_high = np.array([self.cheetah.max_torque] * action_dim)
        self.action_space = FloatBox(-action_high, action_high)
        self.observation_space = FloatBox(observation_low, observation_high)

    def _get_observation(self):
        """Get observation of this environment, including noise and latency.
        """
        observation = []
        observation.extend(self.cheetah.get_motor_angles().tolist())
        observation.extend(self.cheetah.get_motor_velocities().tolist())
        observation.extend(self.cheetah.get_motor_torques().tolist())
        observation.extend(self.cheetah.get_base_orientation())
        self._observation = observation
        return self._observation

    def _get_true_observation(self):
        observation = []
        observation.extend(self.cheetah.get_true_motor_angles().tolist())
        observation.extend(self.cheetah.get_true_motor_velocities().tolist())
        observation.extend(self.cheetah.get_true_motor_torques().tolist())
        observation.extend(self.cheetah.get_true_base_orientation())
        self._observation = observation
        return self._observation

    def _get_observation_dim(self):
        return len(self._get_observation())

    def _get_observation_lower_bound(self):
        return -self._get_observation_upper_bound()

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

        Returns:
          The upper bound of an observation. See GetObservation() for the details
            of each element of an observation.
        """
        upper_bound = np.zeros(self._get_observation_dim())
        num_motors = self.cheetah.num_motors
        upper_bound[0:num_motors] = math.pi  # Joint angle.
        upper_bound[num_motors:2 * num_motors] = self.cheetah.motor_speed_limit  # Joint velocity.
        upper_bound[2 * num_motors:3 * num_motors] = self.cheetah.max_torque  # Joint torque.
        upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
        return upper_bound

    def _is_fallen(self):
        """Decide whether the cheetah has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the cheetah is considered fallen.

        Returns:
          Boolean value that indicates whether the cheetah has fallen.
        """
        orientation = self.cheetah.get_base_orientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = np.asarray(rot_mat[6:])
        up_axis = np.asarray([0, 0, 1])
        yaw_too_much = np.dot(up_axis, local_up) < 0.85

        pos = self.cheetah.get_base_position()
        # vel = self.cheetah.get_motor_velocities()
        too_low = pos[2] < 0.1
        return yaw_too_much or too_low

    def _reward(self):
        # Reward for running distance
        current_base_position = self.cheetah.get_base_position()
        forward_reward = current_base_position[0] - self._last_base_position[0]

        # Penalty for energy consumption
        energy_reward = -np.abs(
            np.dot(self.cheetah.get_motor_torques(),
                   self.cheetah.get_motor_velocities())) * self._time_step

        # Penalty for sideways translation.
        drift_reward = -abs(current_base_position[1] - self._last_base_position[1])

        # Penalty for sideways rotation of the body.
        orientation = self.cheetah.get_base_orientation()
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))

        objectives = [forward_reward, energy_reward, drift_reward, shake_reward]

        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = sum(weighted_objectives)
        self._objectives.append(objectives)
        return reward

    def _termination(self):
        position = self.cheetah.get_base_position()
        distance = math.sqrt(position[0] ** 2 + position[1] ** 2)
        return distance > self._distance_limit or self._env_step_counter >= self.max_num_steps

    def step(self, action):
        self._last_base_position = self.cheetah.get_base_position()

        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.cheetah.get_base_position()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

        self.cheetah.step(action)
        self._env_step_counter += 1

        obs = self._get_observation()
        reward = self._reward()
        done = self._termination()

        if done:
            self.cheetah.terminate()
        return obs, reward, done, {}

    def reset(self):
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)

        # self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)

        self.scene.reset()
        self.cheetah.reset()
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._objectives = []
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        return self._get_observation()

    def render(self, mode='human'):
        pass


class StadiumScene(object):
    def __init__(self,
                 pybullet_client,
                 urdf_root=pybullet_data.getDataPath(),
                 reflection=True):

        self._pybullet_client = pybullet_client
        self._reflection = reflection
        self._urdf_root = urdf_root

        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30

        # filename = os.path.expanduser("~/data/mar-saba-monastery-rawscan/source/MarSaba/MarSaba.obj")
        # collisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        # visualShapeIds = p.createVisualShape(p.GEOM_MESH, fileName=filename)
        # orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])

        # filename = os.path.expanduser("~/data/convent-nuestra-senora-de-los-angeles-de-la-hoz/source/La Hoz/Low_Poly/Convento_de_La_Hoz.obj")
        filename = os.path.expanduser("~/data/ancient-theatre-sagalassos-turkey/source/Saga_Theatre/Saga_Theatre.obj")
        # filename = os.path.expanduser("~/data/3d-city-porto-alegre-centro-historico-01/source/models/")
        collisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        visualShapeIds = p.createVisualShape(p.GEOM_MESH, fileName=filename)
        orn = p.getQuaternionFromEuler([0, 0, 0])

        self._ground_id = p.createMultiBody(0,
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeIds,
                                            baseOrientation=orn)

    def reset(self):
        self._pybullet_client.setGravity(0, 0, -10)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        # self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])


if __name__ == '__main__':
    env = CheetahEnv(render=True)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        env.cheetah.step(action)

        done = env._termination()

        time.sleep(0.01)
