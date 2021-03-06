#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import math
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))

# define the actor list, containing vehivle, camera, sensor
actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    # get carla.BlueprintLibrary
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    # define other actor_vehicles
    '''bp2 = blueprint_library.filter('model2')[0]
    print(bp2)
    bp3 = blueprint_library.filter('model1')[0]
    print(bp3)'''

    spectator = world.get_spectator()
    # spawn_point = random.choice(world.get_map().get_spawn_points()) # type: transform
    spawn_point = carla.Transform(random.choice(world.get_map().get_spawn_points()).location, carla.Rotation(yaw=90))

    vehicle = world.spawn_actor(bp, spawn_point) # type: carla.Actor
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=10, z=10))    
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    spectator.set_transform(camera.get_transform())
    # go forward
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # sleep for 3 seconds, then finish:
    time.sleep(5)
    # hand brake
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    vehicle.apply_control(carla.VehicleControl(hand_brake=False))
    # go backward
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0,reverse=True))
    # append the actor_vehivle to the actor list
    actor_list.append(vehicle)

    # sleep for 10 seconds, then finish:
    time.sleep(30)

finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')










