#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

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
    '''bp2 = blueprint_library.filter('model3')[0]
    print(bp2)
    bp3 = blueprint_library.filter('model3')[0]
    print(bp3)'''

    # spawn_point = random.choice(world.get_map().get_spawn_points()) # type: transform
    spawn_point = carla.Transform(carla.Location(4440,700,50),carla.Rotation(0,0,0))

    vehicle = world.spawn_actor(bp, spawn_point) # type: carla.Actor
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # sleep for 3 seconds, then finish:
    time.sleep(5)
    vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    vehicle.apply_control(carla.VehicleControl(hand_brake=False))
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0,reverse=True))
    actor_list.append(vehicle)

    # sleep for 10 seconds, then finish:
    time.sleep(10)

finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')










