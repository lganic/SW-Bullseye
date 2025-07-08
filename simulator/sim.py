from typing import Dict, Tuple
import math

from .constants import WIND_SCALAR
from .base_sim import single_axis_positioning, split_to_vector

# The forward kinematics equations are based on the work of smithy3141 on the Stormworks Discord
# Also big thanks to Trapdoor on the Stormworks Discord for their help working wind forces in

def generate_forward_ballistics(firing_state: Dict[str, float], az: float, el: float, time: float, v = False) -> Tuple[float, float, float]:

    '''
    Generate a ballistics prediction, using the azimuth, elevation, and time, as well as the firing parameters. 
    Az, and El should be in radians.
    '''

    v = firing_state['muzzle_velocity']
    d = firing_state['projectile_drag']
    g = firing_state['gravity']

    k = 1 - d

    a = k * (1 - k ** time) / d

    w = WIND_SCALAR * firing_state['wind_velocity']
    wind_angle = math.radians(firing_state['wind_direction'])

    w_x = -w * math.cos(wind_angle)
    w_z = -w * math.sin(wind_angle)

    myvel_x, myvel_z = split_to_vector(firing_state['my_velocity_heading'], firing_state['my_velocity'])

    i_x = v * math.cos(el) * math.cos(az) + myvel_x
    i_y = v * math.sin(el)
    i_z = v * math.cos(el) * math.sin(az) + myvel_z

    # if v:
    #     print("CJHEC")
    #     print(az, el, time)
    #     print(v, myvel_x, myvel_z, firing_state['my_velocity'])
    #     print(split_to_vector(firing_state['my_velocity_heading'], firing_state['my_velocity']))
    #     print(i_x, i_y, i_z)

    x_pos = single_axis_positioning(a, i_x, w_x, d, time)
    y_pos = single_axis_positioning(a, i_y, g  , d, time)
    z_pos = single_axis_positioning(a, i_z, w_z, d, time)

    return (x_pos, y_pos, z_pos)

def offset_target_position(firing_state: Dict[str, float], time: float):

    velocity_x, velocity_z = split_to_vector(firing_state['target_velocity_heading'], firing_state['target_velocity'])

    return velocity_x * time / 60, velocity_z * time / 60

if __name__ == '__main__':

    state = {
        'muzzle_velocity': 900,
        'projectile_drag': .005,
        'gravity': .5,
        'wind_velocity': .5,
        'wind_direction': 162,
    }

    el = math.radians(4.5)
    az = -1.30954084853

    print(generate_forward_ballistics(state, az, el, 237))