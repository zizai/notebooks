import math
import os

import pybullet as p
import pybullet_data
import time

import pygame

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#plane = p.loadSDF("plane_stadium.sdf")
#p.resetBasePositionAndOrientation(plane[0], [0, 0, -100], [0, 0, 0, 1])

filename = os.path.expanduser("~/data/mar-saba-monastery-rawscan/source/MarSaba/MarSaba.obj")
collisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
visualShapeIds = p.createVisualShape(p.GEOM_MESH, fileName=filename)
orn = p.getQuaternionFromEuler([math.pi/2, 0, 0])
p.createMultiBody(0, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeIds, baseOrientation=orn)
startHeight = 0
linearDamping = 0.1

'''
filename = os.path.expanduser("~/data/small-town-draft-modus/source/model.obj")
collisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
visualShapeIds = p.createVisualShape(p.GEOM_MESH, fileName=filename)
orn = p.getQuaternionFromEuler([math.pi/2, 0, 0])
p.createMultiBody(0, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeIds, baseOrientation=orn)

#filename = os.path.expanduser("~/data/convent-nuestra-senora-de-los-angeles-de-la-hoz/source/La Hoz/Low_Poly/Convento_de_La_Hoz.obj")
filename = os.path.expanduser("~/data/ancient-theatre-sagalassos-turkey/source/Saga_Theatre/Saga_Theatre.obj")
#filename = os.path.expanduser("~/data/3d-city-porto-alegre-centro-historico-01/source/models/")
collisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
visualShapeIds = p.createVisualShape(p.GEOM_MESH, fileName=filename)
orn = p.getQuaternionFromEuler([0, 0, 0])
p.createMultiBody(0, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeIds, baseOrientation=orn)

filename = os.path.expanduser("~/data/shuri-castle-shurijo-naha-okinawa-wip/source/Shujiro_Castle/Shujiro_Castle.obj")
collisionShapeId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
visualShapeIds = p.createVisualShape(p.GEOM_MESH, fileName=filename)
orn = p.getQuaternionFromEuler([0, 0, 0])
p.createMultiBody(0, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeIds, baseOrientation=orn)

'''

print(p.getBodyInfo(collisionShapeId))


sphere = p.loadURDF("sphere_small.urdf")
#objects = p.loadMJCF("mjcf/sphere.xml")
#sphere = objects[0]
p.resetBasePositionAndOrientation(sphere, [0, 0, startHeight], [0, 0, 0, 1])
p.changeDynamics(sphere, -1, linearDamping=linearDamping)
p.changeVisualShape(sphere, -1, rgbaColor=[1, 0, 0, 1])
p.setGravity(0, 0, -10)

forward = 0
turn = 0

forwardVec = [2, 0, 0]
cameraDistance = 1
cameraYaw = 35
cameraPitch = -35

joystick = None
pygame.init()
if pygame.joystick.get_count():
    print("[JOYSTICK]")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Get the name from the OS for the controller/joystick.
    name = joystick.get_name()
    print("Joystick name: {}".format(name))


while True:

    spherePos, orn = p.getBasePositionAndOrientation(sphere)

    cameraTargetPosition = spherePos
    p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
    camInfo = p.getDebugVisualizerCamera()
    camForward = camInfo[5]

    if joystick:
        event = pygame.event.get()
        if event:
            event = event[0]
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    turn = - event.value
                if event.axis == 1:
                    forward = - event.value

    keys = p.getKeyboardEvents()
    for k, v in keys.items():

        if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
            turn = -0.5
        if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
            turn = 0
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
            turn = 0.5
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
            turn = 0

        if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
            forward = 1
        if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
            forward = 0
        if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED):
            forward = -1
        if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED):
            forward = 0

    force = [forward * camForward[0], forward * camForward[1], 0]
    cameraYaw = cameraYaw + turn

    if forward:
        p.applyExternalForce(sphere, -1, force, spherePos, flags=p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(1. / 240.)
