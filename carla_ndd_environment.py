#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation using the Traffic Manager interface"""

import time
import random
import glob
import argparse
import logging
import sys
import os
import numpy as np
import math
from ndd_vehicle import *
import multiprocessing
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default

pool = multiprocessing.Pool(processes=cpus)


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


def calculate_turning_velocity(vehicle, direction, world):
    nearest_waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
    if direction == "Left":
        if str(nearest_waypoint.lane_change) == "Left" or str(nearest_waypoint.lane_change) == "Both":
            side_lane_waypoint = nearest_waypoint.get_left_lane()
        else:
            return None
            # raise ValueError("Turing left where no left lane exists")
    elif direction == "Right":
        if str(nearest_waypoint.lane_change) == "Right" or str(nearest_waypoint.lane_change) == "Both":
            side_lane_waypoint = nearest_waypoint.get_right_lane()
        else:
            return None
            # raise ValueError("Turing right where no right lane exists")
    running_distance = get_velocity_scalar(vehicle)*1
    if running_distance > side_lane_waypoint.lane_width:
        longitudinal_distance = np.sqrt(running_distance**2-side_lane_waypoint.lane_width**2)
        target_waypoint = side_lane_waypoint.next(longitudinal_distance)[0]
        location_difference = target_waypoint.transform.location - nearest_waypoint.transform.location
        new_velocity = location_difference
        # new_velocity = carla.Vector3D(location_difference.x/running_distance, location_difference.y/running_distance, location_difference.z/running_distance)
        if np.isnan(new_velocity.x):
            print("NAN")
        return new_velocity
    else:
        print("velocity too low!")
        return None

def vector_to_numpy_vector(vector):
    return np.array([vector.x, vector.y, vector.z])

def get_velocity_scalar(vehicle):
    velocity = vehicle.get_velocity()
    velocity_nparray = vector_to_numpy_vector(velocity)
    return np.linalg.norm(velocity_nparray)

def accelerate(vehicle, world, tm, acc, dt):
    """
        Apply acceleration for some period
        Only suits for vehicle on the straight road
    """
    velocity_nparray = vector_to_numpy_vector(vehicle.get_velocity())
    velocity_scalar = np.linalg.norm(velocity_nparray)
    velocity_scalar = velocity_scalar + acc*dt
    now_waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
    desired_length = velocity_scalar*dt

    new_waypoint = now_waypoint.next(desired_length)[0]
    new_velocity_direction = vector_to_numpy_vector(new_waypoint.transform.location) - vector_to_numpy_vector(vehicle.get_location())
    new_velocity = velocity_scalar * new_velocity_direction/np.linalg.norm(new_velocity_direction)
    new_velocity_vector = carla.Vector3D(new_velocity[0],new_velocity[1],new_velocity[2])
    vehicle.set_velocity(new_velocity_vector)
    # new_transform = carla.Transform(vehicle.get_lomtl
    # mtlcation(),new_waypoint.transform.rotation)
    vehicle.set_transform(new_waypoint.transform)

def get_spectator_transform(vehicle_location, angle, d=6.4, rotation=None):
    pitch = rotation.pitch
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location+carla.Location(z=50), rotation)

def is_straight_value(value):
    if abs(value)%90 < 3 or 90-abs(value)%90<3:
        return True
    else:
        return False

def is_straight_transform(transform):
    return is_straight_value(transform.rotation.yaw)
    
def apply_steer_velocity(vehicle, steer_velocity, dt):
    vehicle.set_velocity(steer_velocity)
    new_location = vehicle.get_location() + steer_velocity*dt
    vehicle.set_location(new_location)

def change_road_section(wp, world, length=10):
    """
        return whether after "length" meters there will be another road section
    """
    ego_wp = wp
    if ego_wp.next(length)[0].road_id == ego_wp.road_id:
        return False
    else:
        return True

def get_range(location1, location2):
    location_dif = location1-location2
    return get_absolute_value(location_dif)

def get_absolute_value(vector):
    return max(abs(vector.x),abs(vector.y),abs(vector.z))


def print_wp_info(wp):
    left_wp = wp.get_left_lane()
    right_wp = wp.get_right_lane()
    while(1):
        if left_wp and str(left_wp.lane_type) == "Driving":
            print("left type:", left_wp.lane_id)
        else:
            break
        left_wp = left_wp.get_left_lane()
    while(1):
        if right_wp and str(right_wp.lane_type) == "Driving":
            print("right type:", right_wp.lane_id)
        else:
            break
        right_wp = right_wp.get_right_lane()


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=100,
        type=int,
        help='number of vehicles (default: 10)')


    args = argparser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    client.load_world('Town04')
    
    try:
        
        SIMULATION_FREQUENCY = 20
        world = client.get_world()
        spectator = world.get_spectator()
        settings = world.get_settings()
        # settings.no_rendering_mode = True
        settings.fixed_delta_seconds = 1.0/float(SIMULATION_FREQUENCY)
        settings.synchronous_mode = True
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library().filter('model3')
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
        bp = blueprints.filter('model3')[0]
        bp.set_attribute('color','0,0,205')
        cav_bp = blueprints.filter('model3')[0]
        cav_bp.set_attribute('color','255,255,255')
        spawn_points = world.get_map().get_spawn_points()

        new_spawn_points = []
        for point in spawn_points:
            nearest_waypoint = world.get_map().get_waypoint(point.location, project_to_road=True)
            lc = nearest_waypoint.lane_change
            if str(lc) != "Left" and str(lc) != "NONE":
                new_spawn_points.append(point)
        spawn_points = new_spawn_points
        
        number_of_spawn_points = len(spawn_points)
        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor

        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            if len(vehicles_list)==0:
                blueprint = cav_bp
            else:
                blueprint = bp
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            vehicles_list.append(vehicle)
        
        

        print('Spawned %d vehicles, press Ctrl+C to exit.' % (len(vehicles_list)))
        cav = vehicles_list[0]
        tm = client.get_trafficmanager(8000)
        tm.global_percentage_speed_difference(-10)
        # tm.set_hybrid_physics_mode(True)
        for v in vehicles_list:
            v.set_autopilot(True)
            tm.ignore_lights_percentage(v, 100)
            tm.auto_lane_change(v, False)
        for i in range(20*SIMULATION_FREQUENCY):
            spectator.set_transform(carla.Transform(cav.get_location() + carla.Location(z=50), carla.Rotation(pitch=-90)))
            world.tick()
        client.start_recorder("recording04.log")
        

        

        for i in range(10000*SIMULATION_FREQUENCY):
            now_waypoint = world.get_map().get_waypoint(cav.get_location(), project_to_road=True)
        
            if i%SIMULATION_FREQUENCY == 0:
                print("New Second")
                # print(now_waypoint.lane_id)
                # print(cav.get_speed_limit())
                acc_flag_list = [False]*len(vehicles_list)
                steer_flag_list = [False]*len(vehicles_list)
                steer_velocity_list = [0]*len(vehicles_list)
                acc_list = [0]*len(vehicles_list)
                if is_straight_transform(now_waypoint.transform):
                # if 0:
                    cav_road_id = now_waypoint.road_id
                    cav_lane_id = now_waypoint.lane_id
                    # waypoint_list = pool.map(lambda x:world.get_map().get_waypoint(x.get_location(), project_to_road=True), vehicles_list)
                    waypoint_list = [world.get_map().get_waypoint(v.get_location(), project_to_road=True) for v in vehicles_list]
                    candidate_idx_list = [i for i in range(len(vehicles_list)) if waypoint_list[i].road_id == cav_road_id and get_range(waypoint_list[i].transform.location,now_waypoint.transform.location)<30 and cav_lane_id*waypoint_list[i].lane_id>0]
                    acc_flag = False 
                    steer_flag = False
                    ego_wp = now_waypoint
                    ego_road_id = ego_wp.road_id
                    env.generate_vehicle_from_carla(vehicles_list, waypoint_list, candidate_idx_list)
                    env.render()

                    # print("cav velocity:", cav.get_velocity().x, cav.get_velocity().y, cav.get_velocity().z)
                    # print("cav location:", cav.get_location().x, cav.get_location().y, cav.get_location().z)
                    for i in range(len(candidate_idx_list)):
                        if i == 0:
                            print("CAV")
                            # print_wp_info(waypoint_list[candidate_idx_list[i]])
                            continue
                        idx = candidate_idx_list[i]
                        # print_wp_info(waypoint_list[idx])
                        _, _, ndd_action = env.road.vehicles[i].act()
                        if ndd_action == "Left":
                            acc_flag_list[idx] = False
                            steer_flag_list[idx] = True
                            steer_velocity_list[idx] = calculate_turning_velocity(vehicle, "Left", world)
                        elif ndd_action == "Right":
                            acc_flag_list[idx] = False
                            steer_flag_list[idx] = True
                            steer_velocity_list[idx] = calculate_turning_velocity(vehicle, "Right", world)
                        elif type(ndd_action)== np.float64:
                            acc_flag_list[idx] = True
                            steer_flag_list[idx] = False
                            acc = ndd_action
                        else:
                            acc_flag_list[idx] = False
                            steer_flag_list[idx] = False

            for idx in range(len(vehicles_list)):
                if idx == 0:
                    continue
                if acc_flag_list[idx]:
                    accelerate(v, world, tm, acc_list[idx], 1.0/SIMULATION_FREQUENCY)
                elif steer_flag_list[idx]:
                    if steer_velocity_list[idx] is None:
                        accelerate(v, world, tm, 0, 1.0/SIMULATION_FREQUENCY)
                    else:
                        apply_steer_velocity(v, steer_velocity_list[idx], 1.0/SIMULATION_FREQUENCY)
                else:
                    v.set_autopilot(True)

            spectator.set_transform(carla.Transform(cav.get_location() + carla.Location(z=50), carla.Rotation(pitch=-90)))
            world.tick()
            
                
    finally:
        
        print('Destroying %d vehicles.\n' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        client.stop_recorder()
        


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.\n')
