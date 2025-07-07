from typing import Dict, Tuple
import math

from constants import WIND_SCALAR
from base_sim import single_axis_positioning

# The forward kinematics equations are based on the work of smithy3141 on the Stormworks Discord
# Also big thanks to Trapdoor on the Stormworks Discord for their help working wind forces in

def generate_forward_ballistics(firing_state: Dict[str, float], az: float, el: float, time: float) -> Tuple[float, float, float]:

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

    i_x = v * math.cos(el) * math.cos(az)
    i_y = v * math.sin(el)
    i_z = v * math.cos(el) * math.sin(az)

    x_pos = single_axis_positioning(a, i_x, w_x, d, time)
    y_pos = single_axis_positioning(a, i_y, g  , d, time)
    z_pos = single_axis_positioning(a, i_z, w_z, d, time)

    return (x_pos, y_pos, z_pos)


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