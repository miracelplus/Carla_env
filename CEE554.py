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

# receive the velocity and receive acc 
def action(vehicle,acc,angle_old,angle_new,deltat):
    # angle_loc is the angle change in transform, angle_global is the angle in world
    # obtain the vel info of actor
    current_vel = vehicle.get_velocity()
    # the coming angular velocity in deltat
    angle_vel = (angle_new-angle_old)/deltat
    indicator = np.sign(acc)
    change_vel_x = indicator*acc*np.cos(angle_new)*deltat
    change_vel_y = indicator*acc*np.sin(angle_new)*deltat
    change_vel_z = acc*0

    change_vel = carla.Vector3D(change_vel_x,change_vel_y,change_vel_z)
    new_ang_vel = carla.Vector3D(0,0,angle_vel)
    new_vel = current_vel + change_vel
    vehicle.set_angular_velocity(new_ang_vel)
    vehicle.set_velocity(new_vel)

def vehicle_info(vehicle):
    current_vel = vehicle.get_velocity()
    current_acc = vehicle.get_acceleration()
    
    return current_vel, current_acc


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

    deltat = 0.1

    # spawn_point = random.choice(world.get_map().get_spawn_points()) # type: transform
    spawn_point = carla.Transform(carla.Location(4440,700,50),carla.Rotation(0,0,0))
    x=79.8
    y=40
    z=0
    new_point = carla.Location(79.8, 40, 0.0)
    # generate the vehicle actor
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_location(new_point)
    actor_list.append(vehicle)

    ##########################################################
    deltat=0.1
    action(vehicle,5,90,-90,deltat)

    time.sleep(deltat)

    ##########################################################
    '''for i in range(3000):
        x=x+0.3
        new_point = carla.Location(x,y,z)
        vehicle.set_location(new_point)
        time.sleep(0.01)'''
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # # sleep for 3 seconds, then finish:
    # time.sleep(5)
    # vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    # vehicle.apply_control(carla.VehicleControl(hand_brake=False))
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0,reverse=True))
    

    # sleep for 10 seconds, then finish:
    #time.sleep(1)

finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')










