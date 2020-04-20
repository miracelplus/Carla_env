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
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=200,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=00,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    client.load_world('Town04')

    try:
        SIMULATION_FREQUENCY = 30
        world = client.get_world()
        
        # settings = world.get_settings()
        # settings.no_rendering_mode = True
        # world.apply_settings(settings)
        # settings = world.get_settings()
        # settings.fixed_delta_seconds = 1.0/float(SIMULATION_FREQUENCY)
        # settings.synchronous_mode = True
        # world.apply_settings(settings)
        blueprints = world.get_blueprint_library().filter('model3')
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        spawn_points = world.get_map().get_spawn_points()
        print(len(spawn_points))
        new_spawn_points = []
        for point in spawn_points:
            nearest_waypoint = world.get_map().get_waypoint(point.location, project_to_road=True)
            lc = nearest_waypoint.lane_change
            # print(nearest_waypoint.lane_change)
            # left_wp = nearest_waypoint.get_left_lane()
            # right_wp = nearest_waypoint.get_right_lane()
            # print(right_wp)
            # print(str(lc))
            # if str(lc) == None:
            #     print("sdasd")
            if str(lc) != "Left" and str(lc) != "NONE":
                new_spawn_points.append(point)
        spawn_points = new_spawn_points
        number_of_spawn_points = len(spawn_points)
        print(number_of_spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor

        # --------------
        # Spawn vehicles
        # --------------

        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            vehicles_list.append(vehicle)

        print('Spawned %d vehicles, press Ctrl+C to exit.' % (len(vehicles_list)))

        time.sleep(1)
        tm = client.get_trafficmanager(port)
        for v in vehicles_list:
            v.set_autopilot(True)

        while True:
            time.sleep(1)
            # world.tick()

    finally:

        print('Destroying %d vehicles.\n' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.\n')
