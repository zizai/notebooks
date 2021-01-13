import math

import gym
import numpy as np
import os
import pybullet
import pybullet_data
from pybullet_envs.env_bases import MJCFBaseBulletEnv

from chaosbreaker.envs.locomotion.locomotors import HumanoidFlagrun, HumanoidFlagrunHarder, Humanoid, Ant, HalfCheetah, Hopper, \
    Walker2D
from pybullet_envs.scene_abstract import Scene
from pybullet_utils import bullet_client


class StadiumScene(Scene):
    zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
        if self.stadiumLoaded == 0:
            self.stadiumLoaded = 1

            # stadium_pose = cpp_household.Pose()
            # if self.zero_at_running_strip_start_line:([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
            #	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

            filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            '''
            filename = os.path.expanduser("~/data/mar-saba-monastery-rawscan/source/MarSaba/MarSaba.obj")
            collisionShapeId = self._p.createCollisionShape(pybullet.GEOM_MESH, fileName=filename,
                                                      flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH)
            visualShapeIds = self._p.createVisualShape(pybullet.GEOM_MESH, fileName=filename)
            orn = self._p.getQuaternionFromEuler([math.pi / 2, 0, 0])
            self.ground_plane_mjcf = [self._p.createMultiBody(0, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeIds, baseOrientation=orn)]
            startHeight = 0
            linearDamping = 0.1
            '''

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                #self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
                #self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 1)

            #	for j in range(p.getNumJoints(i)):
            #		self._p.changeDynamics(i,j,lateralFriction=0)
            #despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground


class SinglePlayerStadiumScene(StadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False


class MultiplayerStadiumScene(StadiumScene):
    multiplayer = True
    players_count = 3

    def actor_introduce(self, robot):
        StadiumScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)


class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

    def __init__(self, robot, render=False, max_num_steps=1000):
        # print("WalkerBase::__init__ start")
        self.max_num_steps = max_num_steps
        self._step_counter = 0
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        MJCFBaseBulletEnv.__init__(self, robot, render)


    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                      gravity=9.8,
                                                      timestep=0.0165 / 4,
                                                      frame_skip=4)
        return self.stadium_scene


    def reset(self):
        if (self.stateId >= 0):
            #print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self._p, self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                                self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
            #print("saving state self.stateId:",self.stateId)

        self._step_counter = 0
        return r

    def _isDone(self):
        return self._alive < 0 or self._step_counter >= self.max_num_steps

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        self._step_counter += 1
        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                #see Issue 63: https://github.com/openai/roboschool/issues/63
                #feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    def camera_adjust(self):
        x, y, z = self.robot.body_real_xyz

        self.camera_x = x
        self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)


class HopperBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False, max_num_steps=1000):
        self.robot = Hopper()
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)


class Walker2DBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False, max_num_steps=1000):
        self.robot = Walker2D()
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)


class HalfCheetahBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False, max_num_steps=1000):
        self.robot = HalfCheetah()
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)

    def _isDone(self):
        return False


class AntBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False, max_num_steps=1000):
        self.robot = Ant()
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)


class HumanoidBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, robot=None, render=False, max_num_steps=1000):
        if robot is None:
            self.robot = Humanoid()
        else:
            self.robot = robot
        WalkerBaseBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)
        self.electricity_cost = 4.25 * WalkerBaseBulletEnv.electricity_cost
        self.stall_torque_cost = 4.25 * WalkerBaseBulletEnv.stall_torque_cost


class HumanoidFlagrunBulletEnv(HumanoidBulletEnv):
    random_yaw = True

    def __init__(self, render=False, max_num_steps=1000):
        self.robot = HumanoidFlagrun()
        HumanoidBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)

    def create_single_player_scene(self, bullet_client):
        s = HumanoidBulletEnv.create_single_player_scene(self, bullet_client)
        s.zero_at_running_strip_start_line = False
        return s


class HumanoidFlagrunHarderBulletEnv(HumanoidBulletEnv):
    random_lean = True  # can fall on start

    def __init__(self, render=False, max_num_steps=1000):
        self.robot = HumanoidFlagrunHarder()
        self.electricity_cost /= 4  # don't care that much about electricity, just stand up!
        HumanoidBulletEnv.__init__(self, self.robot, render=render, max_num_steps=max_num_steps)

    def create_single_player_scene(self, bullet_client):
        s = HumanoidBulletEnv.create_single_player_scene(self, bullet_client)
        s.zero_at_running_strip_start_line = False
        return s
