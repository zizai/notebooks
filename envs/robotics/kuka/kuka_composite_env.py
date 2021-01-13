import glob
import os
from collections import namedtuple

import gym
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from chaosbreaker.envs.robotics.kuka.kuka import Kuka
from chaosbreaker.spaces.composite import Composite
from chaosbreaker.spaces.float_box import FloatBox
from chaosbreaker.spaces.int_box import IntBox
from pkg_resources import parse_version


POSITION_LIMIT = 100
RENDER_HEIGHT = 720
RENDER_WIDTH = 960

KukaCompositeObservation = namedtuple("KukaCompositeObservation", ["camera", "position"])


class KukaCompositeEnv(gym.Env):
    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=80,
                 enable_self_collision=True,
                 render=False,
                 max_num_steps=1000,
                 dv=0.06,
                 block_random=0.3,
                 camera_random=0,
                 width=48,
                 height=48,
                 num_objects=5,
                 eval_mode=False):
        """Initializes the KukaDiverseObjectEnv.

        Args:
          urdf_root: The diretory from which to load environment URDF's.
          action_repeat: The number of simulation steps to apply for each action.
          enable_self_collision: If true, enable self-collision.
          renders: If true, render the bullet GUI.
          discrete: If true, the action space is discrete. If False, the
            action space is continuous.
          max_num_steps: The maximum number of actions per episode.
          dv: The velocity along each dimension for each action.
          block_random: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
          camera_random: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
          width: The image width.
          height: The observation image height.
          num_objects: The number of objects in the bin.
          eval_mode: If true, use the test set of objects. If false, use the train
            set of objects.
        """
        self._timestep = 1. / 240.
        self._urdf_root = urdf_root
        self._action_repeat = action_repeat
        self._enable_self_collision = enable_self_collision
        self._observation = []
        self._env_step_counter = 0
        self._render = render
        self._max_num_steps = max_num_steps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dv = dv
        self._p = p
        self._block_random = block_random
        self._camera_random = camera_random
        self._width = width
        self._height = height
        self._num_objects = num_objects
        self._eval_mode = eval_mode

        if self._render:
            self.cid = p.connect(p.SHARED_MEMORY)
            if self.cid < 0:
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()
        self.reset()

        self.action_space = FloatBox(low=-1, high=1, shape=(4,))  # dx, dy, dz, da

        camera_space = IntBox(low=0,
                              high=255,
                              shape=(self._height, self._width, 4),
                              dtype=np.uint8)

        position_dim = len(self._get_position())
        position_high = np.array([POSITION_LIMIT] * position_dim)
        position_space = FloatBox(-position_high, position_high)

        self.observation_space = Composite((camera_space, position_space), KukaCompositeObservation)
        self.viewer = None
        self.max_num_steps = max_num_steps

    def _get_position(self):
        self._position = self._kuka.get_position()
        return self._position

    def _get_camera(self):
        """Return the observation as an image.
        """
        img_arr = p.getCameraImage(width=self._width,
                                   height=self._height,
                                   viewMatrix=self._view_matrix,
                                   projectionMatrix=self._proj_matrix)
        rgb = img_arr[2]
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        return np_img_arr[:, :, :3]

    def _get_observation(self):
        return {
            "position": self._get_position(),
            "cam": self._get_camera()
        }

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
          num_objects:
            Number of graspable objects.

        Returns:
          A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdf_root, "random_urdfs/*0/*.urdf")
        else:
            urdf_pattern = os.path.join(self._urdf_root, "random_urdfs/*[1-9]/*.urdf")
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames


    def _randomly_place_objects(self, urdfList):
        """Randomly places the objects in the bin.

        Args:
          urdfList: The list of urdf files to place in the bin.

        Returns:
          The list of object unique ID's.
        """

        # Randomize positions of each object urdf.
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0.4 + self._block_random * random.random()
            ypos = self._block_random * (random.random() - .5)
            angle = np.pi / 2 + self._block_random * np.pi * random.random()
            orn = p.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self._urdf_root, urdf_name)
            uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            # Let each object fall to the tray individual, to prevent object
            # intersection.
            for _ in range(500):
                p.stepSimulation()
        return objectUids

    def _step_continuous(self, action):
        """Applies a continuous velocity-control action.

        Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """
        # Perform commanded action.
        self._env_step += 1
        self._kuka.applyAction(action)
        for _ in range(self._action_repeat):
            p.stepSimulation()
            if self._render:
                time.sleep(self._timestep)
            if self._termination():
                break

        # If we are close to the bin, attempt grasp.
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        end_effector_pos = state[0]
        if end_effector_pos[2] <= 0.1:
            finger_angle = 0.3
            for _ in range(500):
                grasp_action = [0, 0, 0, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                #if self._renders:
                #  time.sleep(self._timeStep)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0
            for _ in range(500):
                grasp_action = [0, 0, 0.001, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                if self._render:
                    time.sleep(self._timestep)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0
            self._attempted_grasp = True
        observation = self._get_observation()
        done = self._termination()
        reward = self._reward()

        debug = {"grasp_success": self._grasp_success}
        return observation, reward, done, debug

    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the objects is above height .2 at the end of the
        episode.
        """
        reward = 0
        self._grasp_success = 0
        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            # If any block is above height, provide reward.
            if pos[2] > 0.2:
                self._grasp_success += 1
                reward = 1
                break
        return reward

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._attempted_grasp or self._env_step >= self._max_num_steps

    def reset(self):
        """Environment reset called at the beginning of an episode.
        """
        # Set the camera settings.
        look = [0.23, 0.2, 0.54]
        distance = 1.
        pitch = -56 + self._camera_random * np.random.uniform(-3, 3)
        yaw = 245 + self._camera_random * np.random.uniform(-3, 3)
        roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20. + self._camera_random * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        self._attempted_grasp = False
        self._env_step = 0
        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                   0.000000, 0.000000, 0.0, 1.0)

        p.setGravity(0, 0, -10)
        self._kuka = Kuka(urdfRootPath=self._urdf_root, timeStep=self._timestep)
        self._envStepCounter = 0
        p.stepSimulation()

        # Choose the objects in the bin.
        urdfList = self._get_random_object(self._num_objects, self._eval_mode)
        self._objectUids = self._randomly_place_objects(urdfList)
        self._observation = self._get_observation()
        return np.array(self._observation)

    def step(self, action):
        """Environment step.

        Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """
        dv = self._dv  # velocity per physics step.
        dx = dv * action[0]
        dy = dv * action[1]
        dz = dv * action[2]
        da = 0.25 * action[3]

        return self._step_continuous([dx, dy, dz, da, 0.3])

    def __del__(self):
        p.disconnect()


def main():
    env = KukaCompositeEnv(render=True)

    while True:
        env.step(env.action_space.sample())


if __name__ == "__main__":
    main()
