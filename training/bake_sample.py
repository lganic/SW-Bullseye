import math
import configparser
import os
import json

CONFIG_NAME = '../params.conf'

CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_NAME)

# Controls mapping launch conditions to and from ranges where they can be ingested / output by a neural network. 

def map_single_field_to_nn(value, scale):

    if value < -scale:
        raise ValueError(f'Attempt to encode value less than expected bounds: {value} < {-scale}')
    
    if value > scale:
        raise ValueError(f'Attempt to encode value greated than expected bounds: {value} > {scale}')

    return value / scale

    # return math.log(1 + value) / math.log(1 + scale)

def map_single_field_from_nn(value, scale):

    # maybe remove these checks later? Not sure if they are really required.
    if value < -1:
        raise ValueError(f'Attempt to decode value less than expected bounds: {value} < -1')
    
    if value > 1:
        raise ValueError(f'Attempt to decode value greater than expected bounds: {value} > 1')

    return value * scale

    # return math.exp(value * math.log(1 + scale)) - 1

def dual_axis_angle_encoding(angle_in_degrees):

    # Use positional encoding trick from typical transformer architectures.
    # (probably used elsewhere, thats just where I know it from)
    return math.cos(math.radians(angle_in_degrees)), math.sin(math.radians(angle_in_degrees))

def wrap_value(value, maximum):

    return value - 2 * maximum * math.ceil(value / (2 * maximum) - .5)

def map_from_degrees(value):

    value = wrap_value(value, 180)

    return map_single_field_to_nn(value, 180)

def map_to_degrees(value):

    return map_single_field_from_nn(value, 180)

def map_from_radians(value):

    value = wrap_value(value, math.pi)
    
    return map_single_field_to_nn(value, math.pi)

def map_to_radians(value):

    return map_single_field_from_nn(value, math.pi)

class Baker:
    def __init__(self):

        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)

        # Read scalar values from the 'Scalars' section
        self.velocity_scalar        = config.getfloat('Scalars', 'velocity_scalar')
        self.distance_scalar        = config.getfloat('Scalars', 'distance_scalar')
        self.altitude_scalar        = config.getfloat('Scalars', 'altitude_scalar')
        self.muzzle_velocity_scalar = config.getfloat('Scalars', 'muzzle_velocity_scalar')
        self.projectile_drag_scalar = config.getfloat('Scalars', 'projectile_drag_scalar')
        self.time_scalar            = config.getfloat('Scalars', 'time_scalar')
    
    def bake_file(self, file_path: str):

        with open(file_path, 'r') as file_obj:

            data = json.load(file_obj)
        
        # Load specific fields needed
        target_distance         = data['target_distance']
        target_altitude         = data['target_altitude']
        target_heading          = data['heading']
        target_velocity         = data['target_velocity']
        target_velocity_heading = data['target_velocity_heading']
        my_velocity             = data['my_velocity']
        my_velocity_heading     = data['my_velocity_heading']
        wind_velocity           = data['wind_velocity']
        wind_heading            = data['wind_direction']
        projectile_velocity     = data['muzzle_velocity']
        projectile_drag         = data['projectile_drag']

        solution_az             = math.degrees(data['solution_az']) - target_heading
        solution_el             = data['solution_el']
        solution_time           = data['solution_time']

        processed_target_distance     = map_single_field_to_nn(target_distance, self.distance_scalar)
        processed_target_altitude     = map_single_field_to_nn(target_altitude, self.altitude_scalar)
        processed_target_velocity     = map_single_field_to_nn(target_velocity, self.velocity_scalar)
        processed_my_velocity         = map_single_field_to_nn(my_velocity, self.velocity_scalar)
        processed_projectile_velocity = map_single_field_to_nn(projectile_velocity, self.muzzle_velocity_scalar)
        processed_projectile_drag     = map_single_field_to_nn(projectile_drag, self.projectile_drag_scalar)

        processed_target_heading_x, processed_target_heading_y                   = dual_axis_angle_encoding(target_heading)
        processed_target_velocity_heading_x, processed_target_velocity_heading_y = dual_axis_angle_encoding(target_velocity_heading)
        processed_my_velocity_heading_x, processed_my_velocity_heading_y         = dual_axis_angle_encoding(my_velocity_heading)
        processed_wind_heading_x, processed_wind_heading_y                       = dual_axis_angle_encoding(wind_heading)

        processed_wind_velocity = wind_velocity

        processed_solution_az   = map_from_degrees(solution_az)
        processed_solution_el   = map_from_radians(solution_el)
        processed_solution_time = map_single_field_to_nn(solution_time, self.time_scalar)

        row = {
            "target_distance": processed_target_distance,
            "target_altitude": processed_target_altitude,
            "target_velocity": processed_target_velocity,
            "my_velocity": processed_my_velocity,
            "projectile_velocity": processed_projectile_velocity,
            "projectile_drag": processed_projectile_drag,
            "target_heading_x": processed_target_heading_x,
            "target_heading_y": processed_target_heading_y,
            "target_velocity_heading_x": processed_target_velocity_heading_x,
            "target_velocity_heading_y": processed_target_velocity_heading_y,
            "my_velocity_heading_x": processed_my_velocity_heading_x,
            "my_velocity_heading_y": processed_my_velocity_heading_y,
            "wind_heading_x": processed_wind_heading_x,
            "wind_heading_y": processed_wind_heading_y,
            "wind_velocity": processed_wind_velocity,
            "solution_az": processed_solution_az,
            "solution_el": processed_solution_el,
            "solution_time": processed_solution_time
        }

        return row


        