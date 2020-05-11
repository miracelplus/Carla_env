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


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    client.load_world('Town04')
    client.replay_file("recording04.log", 1, 200, 0)

if __name__ == "__main__":
    main()
