#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import math
import random
from carla import Map
vehicles_list=[]

def get_view(cmd, inc = 60,world=0,vehicle=0,spectator=0):

 while True:

  os.system(cmd)
  print(vehicle.type_id)
  time.sleep(inc)
  angle = 0
  while angle < 356:
                timestamp = world.wait_for_tick().timestamp
                angle += timestamp.delta_seconds * 60.0
                spectator.set_transform(get_transform(vehicle.get_location(),  90))
             # go forward
            
         



def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    spectator = world.get_spectator()
    map_now=world.get_map()
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle')
   
    location = random.choice(world.get_map().get_spawn_points()).location
    transform=carla.Transform(location,carla.Rotation(yaw=0))
    blueprint=random.choice(vehicle_blueprints)
    ego_vehicle=world.spawn_actor(blueprint,transform)
  

    try:
            ego_vehicle.set_autopilot(True)
            self_transform = ego_vehicle.get_transform()
            nearest_waypoint = map_now.get_waypoint(self_transform.location, project_to_road=True)
            left_wp = nearest_waypoint.get_left_lane()
            right_wp = nearest_waypoint.get_right_lane()
            vehicle_npc_left = world.try_spawn_actor(blueprint, left_wp.transform)
            #vehicle_npc_left.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            #vehicle_npc_front = world.try_spawn_actor(blueprint, nearest_waypoint.transform)
            #vehicle_npc_front.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            vehicle_npc_right = world.try_spawn_actor(blueprint, right_wp.transform)
            #vehicle_npc_right.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            if vehicle_npc_left is not None:
                vehicles_list.append(vehicle_npc_left)
            if vehicle_npc_right is not None:
                vehicles_list.append(vehicle_npc_right)                
                
            for v in vehicles_list:
                print(v.type_id)
                v.set_autopilot(True)
                v.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
            get_view("echo %time%", 0.5,world,ego_vehicle,spectator)
    # sleep for 10 seconds, then finish:
            
    finally:

            ego_vehicle.destroy()
            #vehicle_npc_left.destroy()
            #vehicle_npc_right.destroy()


if __name__ == '__main__':

    main()
