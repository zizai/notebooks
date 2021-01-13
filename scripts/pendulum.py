
import numpy as np
import time

from pybullet_envs import gym_pendulum_envs
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene


class InvertedPendulum(MJCFBasedRobot):
    swingup = False

    def __init__(self):
        MJCFBasedRobot.__init__(self, 'inverted_pendulum.xml', 'cart', action_dim=1, obs_dim=5)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        if not np.isfinite(a).all():
            print("a is inf")
            a[0] = 0
        self.slider.set_motor_torque(100 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        self.theta, theta_dot = self.j1.current_position()
        x, vx = self.slider.current_position()
        assert (np.isfinite(x))

        if not np.isfinite(x):
            print("x is inf")
            x = 0

        if not np.isfinite(vx):
            print("vx is inf")
            vx = 0

        if not np.isfinite(self.theta):
            print("theta is inf")
            self.theta = 0

        if not np.isfinite(theta_dot):
            print("theta_dot is inf")
            theta_dot = 0

        return np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])


class InvertedPendulumBulletEnv(MJCFBaseBulletEnv):

    def __init__(self):
        self.robot = InvertedPendulum()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            #print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            self._p.restoreState(self.stateId)
        r = MJCFBaseBulletEnv.reset(self)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            #print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.pos_x self.pos_y

        vel_penalty = 0
        if self.robot.swingup:
            reward = np.cos(self.robot.theta)
            done = False
        else:
            reward = 1.0
            done = np.abs(self.robot.theta) > .2

        rewards = [float(reward)]

        return state, sum(rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)


def main():
    env = gym_pendulum_envs.InvertedPendulumBulletEnv()
    env.render()
    obs = env.reset()

    while 1:
        action = np.random.randn(1)
        time.sleep(1. / 100.)
        obs = env.step(action)


if __name__ == '__main__':
    main()
