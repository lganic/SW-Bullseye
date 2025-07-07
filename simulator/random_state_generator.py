from random import random, normalvariate, choice
from typing import Dict
import math

from tables import GUNS, MUZZLE_VELOCITY, PROJECTILE_DRAG, GRAVITY, APPROX_MAX_DISTANCE

# Some random wrapper functions to make code easier to read

def random_heading() -> float:
    # Return a random heading between 0 and 360

    return 360 * random()

def random_velocity(std_dev: float) -> float:
    # Return a random velocity of a target

    return normalvariate(0, std_dev)

def random_altitude(std_dev: float) -> float:
    # Return a random altitude of a target

    return normalvariate(0, std_dev)

def random_distance(maximum: float, std_dev_scalar: float) -> float:
    # Return a random distance based on a maximum distance, and a std dev scalar

    std_dev_in_use = maximum * std_dev_scalar

    return normalvariate(0, std_dev_in_use)

def random_wind():
    # Return a random amount of wind between 0 and 1

    return random()

class RandomStateGenerator:
    def __init__(self, distance_deviation_scalar = .1, velocity_std_dev = 30, altitude_std_dev = 100):

        self.distance_deviation_scalar = distance_deviation_scalar
        self.velocity_std_dev = velocity_std_dev
        self.altitude_std_dev = altitude_std_dev
    
    def generate(self) -> Dict[str, float]:

        gun_in_use              = choice(GUNS)

        muzzle_velocity         = MUZZLE_VELOCITY[gun_in_use]
        projectile_drag         = PROJECTILE_DRAG[gun_in_use]
        gravity                 = GRAVITY[gun_in_use]
        max_distance            = APPROX_MAX_DISTANCE[gun_in_use]

        heading                 = random_heading()
        target_velocity_heading = random_heading()
        my_velocity_heading     = random_heading()

        target_distance         = random_distance(max_distance, self.distance_deviation_scalar)
        target_altitude         = random_altitude(self.altitude_std_dev)

        target_velocity         = random_velocity(self.velocity_std_dev)
        my_velocity             = random_velocity(self.velocity_std_dev)

        wind_velocity           = random_wind()
        wind_direction          = random_heading()

        t_x = target_distance * math.cos(math.radians(heading))
        t_y = target_altitude
        t_z = target_distance * math.sin(math.radians(heading))


        return {
            'gun_name':                gun_in_use,
            'muzzle_velocity':         muzzle_velocity,
            'projectile_drag':         projectile_drag,
            'gravity':                 gravity,
            'heading':                 heading,
            'target_velocity_heading': target_velocity_heading,
            'my_velocity_heading':     my_velocity_heading,
            'target_distance':         target_distance,
            'target_altitude':         target_altitude,
            'target_velocity':         target_velocity,
            'my_velocity':             my_velocity,
            'wind_velocity':           wind_velocity,
            'wind_direction':          wind_direction,
            't_x':                     t_x,
            't_y':                     t_y,
            't_z':                     t_z,
            'target':                  (t_x, t_y, t_z)
        }