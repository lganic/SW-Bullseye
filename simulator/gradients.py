from typing import Dict

import math

from .base_sim import single_axis_positioning, split_to_vector
from .sim import offset_target_position
from .constants import WIND_SCALAR

class State:

    def __init__(self, state: Dict[str, float]):

        self.state = state

        self.muzzle_velocity         = state['muzzle_velocity']
        self.projectile_drag         = state['projectile_drag']
        self.gravity                 = state['gravity']
        self.heading                 = state['heading']
        self.target_velocity_heading = state['target_velocity_heading']
        self.my_velocity_heading     = state['my_velocity_heading']
        self.target_distance         = state['target_distance']
        self.target_altitude         = state['target_altitude']
        self.target_velocity         = state['target_velocity']
        self.my_velocity             = state['my_velocity']
        self.wind_velocity           = state['wind_velocity']
        self.wind_direction          = state['wind_direction']
        self.t_x                     = state['t_x']
        self.t_y                     = state['t_y']
        self.t_z                     = state['t_z']

        w = WIND_SCALAR * self.wind_velocity

        self.w_x = -w * math.cos(math.radians(self.wind_direction))
        self.w_z = -w * math.sin(math.radians(self.wind_direction))

    # The next parts involve a bit of calculus to arrive at. But its
    # just partial differentials of the squared error (distance to target)
    # with respect to each variable (az, el, time), so looks complicated, 
    # but its actually just basic setup for gradient descent

    def partial_az(self, az: float, el: float, time: float):

        myvel_x, myvel_z = split_to_vector(self.my_velocity_heading, self.my_velocity)

        i_x = self.muzzle_velocity * math.cos(el) * math.cos(az) + myvel_x
        i_z = self.muzzle_velocity * math.cos(el) * math.sin(az) + myvel_z
        k = 1 - self.projectile_drag
        a = k * (1 - k ** time) / self.projectile_drag

        target_offset_x, target_offset_z = offset_target_position(self.state, time)

        total = 0

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_x, self.w_x, self.projectile_drag, time)

        base -= self.t_x + target_offset_x

        base *= (-a / 60) * self.muzzle_velocity * math.cos(el) * math.sin(az)

        total += base

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_z, self.w_z, self.projectile_drag, time)

        base -= self.t_z + target_offset_z

        base *= (a / 60) * self.muzzle_velocity * math.cos(el) * math.cos(az)

        total += base

        return total

    def partial_el(self, az: float, el: float, time: float):

        myvel_x, myvel_z = split_to_vector(self.my_velocity_heading, self.my_velocity)

        i_x = self.muzzle_velocity * math.cos(el) * math.cos(az) + myvel_x
        i_z = self.muzzle_velocity * math.cos(el) * math.sin(az) + myvel_z

        i_y = self.muzzle_velocity * math.sin(el)
        k = 1 - self.projectile_drag
        a = k * (1 - k ** time) / self.projectile_drag

        target_offset_x, target_offset_z = offset_target_position(self.state, time)

        total = 0

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_x, self.w_x, self.projectile_drag, time)

        base -= self.t_x + target_offset_x

        base *= (-a / 60) * self.muzzle_velocity * math.sin(el) * math.cos(az)

        total += base

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_y, self.gravity, self.projectile_drag, time)

        base -= self.t_y

        base *= (a / 60) * self.muzzle_velocity * math.cos(el)

        total += base

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_z, self.w_z, self.projectile_drag, time)

        base -= self.t_z + target_offset_z

        base *= (-a / 60) * self.muzzle_velocity * math.sin(el) * math.sin(az)

        total += base

        return total
    
    def partial_time(self, az: float, el: float, time: float):

        myvel_x, myvel_z = split_to_vector(self.my_velocity_heading, self.my_velocity)

        i_x = self.muzzle_velocity * math.cos(el) * math.cos(az) + myvel_x
        i_z = self.muzzle_velocity * math.cos(el) * math.sin(az) + myvel_z

        i_y = self.muzzle_velocity * math.sin(el)

        k = 1 - self.projectile_drag
        a = k * (1 - k ** time) / self.projectile_drag

        target_offset_x, target_offset_z = offset_target_position(self.state, time)
        target_velocity_x, target_velocity_z = split_to_vector(self.target_velocity_heading, self.target_velocity)

        def part(i, q):
            return (-k * (i + q / self.projectile_drag) * math.log(k) * math.pow(k, time) - q) / (60 * self.projectile_drag)

        total = 0

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_x, self.w_x, self.projectile_drag, time)

        base -= self.t_x + target_offset_x

        base *= part(i_x, self.w_x) - target_velocity_x

        total += base

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_y, self.gravity, self.projectile_drag, time)

        base -= self.t_y

        base *= part(i_y, self.gravity)

        total += base

        # Re-use single axis positioning, since its the same function due to the chain rule
        base = single_axis_positioning(a, i_z, self.w_z, self.projectile_drag, time)

        base -= self.t_z + target_offset_z

        base *= part(i_z, self.w_z) - target_velocity_z

        total += base

        return total
    