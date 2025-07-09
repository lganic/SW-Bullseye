from typing import Dict
import math

from .gradients import State
from .sim import generate_forward_ballistics, offset_target_position

def _vprint(v, p):
    if v:
        print(p)

        # return {
        #     'gun_name':                gun_in_use,
        #     'muzzle_velocity':         muzzle_velocity,
        #     'projectile_drag':         projectile_drag,
        #     'gravity':                 gravity,
        #     'heading':                 heading,
        #     'target_velocity_heading': target_velocity_heading,
        #     'my_velocity_heading':     my_velocity_heading,
        #     'target_distance':         target_distance,
        #     'target_altitude':         target_altitude,
        #     'target_velocity':         target_velocity,
        #     'my_velocity':             my_velocity,
        #     'wind_velocity':           wind_velocity,
        #     'wind_direction':          wind_direction
        # }

def calculate_inital_guess(firing_state):

    # Suuuuper lazy. But we just want a ballpark.

    az = math.radians(firing_state['heading'])

    # Remap az to +- pi
    az = az - 2 * math.pi * (1 + math.floor(az / (2 * math.pi) - .5))

    # Estimate elevation, based on twice the ballistic prediction
    el = math.asin(60 * firing_state['gravity'] * firing_state['target_distance'] / math.pow(firing_state['muzzle_velocity'], 2))

    # Assume straight line trajectory
    time = 1.75 * 60 * (firing_state['target_distance'] / firing_state['muzzle_velocity'])

    return az, el, time

def calculate_error(firing_state, current_solution):

    position = generate_forward_ballistics(firing_state, *current_solution)

    # calculate distance between calculated position, and actual position

    target_offset_x, target_offset_z = offset_target_position(firing_state, current_solution[2])

    offset_target = (firing_state['target'][0] + target_offset_x, firing_state['target'][1], firing_state['target'][2] + target_offset_z)

    return math.sqrt(sum([math.pow(a - b, 2) for a, b in zip(position, offset_target)]))

def calculate_firing_solution(firing_state: Dict[str, float], learning_rate = .0001, tolerance = .1, max_iterations = 10000, verbose = False):

    current_state = State(firing_state)

    _vprint(verbose, f'Calculating solution for: {firing_state["gun_name"]}')

    current_guess = calculate_inital_guess(firing_state)

    _vprint(verbose, f'Starting error: {calculate_error(firing_state, current_guess)}')

    error = math.inf

    count = 0

    error_increased_count = 0
    flipped_time_gradient = False
    time_gradient_scalar = 1

    while error > tolerance:

        count += 1
        if count > max_iterations:
            _vprint(verbose, firing_state)
            _vprint(verbose, "Failed to calculate for conditions:")
            _vprint(verbose, f'For gun: {firing_state["gun_name"]}')
            _vprint(verbose, f'Velocity: {firing_state["muzzle_velocity"]}')
            _vprint(verbose, f'Drag: {firing_state["projectile_drag"]}')
            _vprint(verbose, '')
            _vprint(verbose, f'Wind: {firing_state["wind_velocity"]} at direction: {firing_state["wind_direction"]}')
            _vprint(verbose, '')
            _vprint(verbose, f'Target is at:')
            _vprint(verbose, f'X: {firing_state["t_x"]}')
            _vprint(verbose, f'Y: {firing_state["t_y"]}')
            _vprint(verbose, f'Z: {firing_state["t_z"]}')
            _vprint(verbose, '')
            _vprint(verbose, f'Error: {error}')
            return 

        # Calculate gradients
        gradient_az   = .01 * current_state.partial_az(*current_guess)
        gradient_el   = .01 * current_state.partial_el(*current_guess)
        gradient_time = current_state.partial_time(*current_guess)

        # Descend the gradient
        new_az   = current_guess[0] - learning_rate * gradient_az
        new_el   = current_guess[1] - learning_rate * gradient_el
        new_time = current_guess[2] - learning_rate * gradient_time * time_gradient_scalar

        current_guess = (new_az, new_el, new_time)

        new_error = calculate_error(firing_state, current_guess)

        if not flipped_time_gradient and new_error > error:
            error_increased_count += 1

            if error_increased_count > 10:
                _vprint(verbose, 'Increasing error detected, flipped time gradient')
                flipped_time_gradient = True
                time_gradient_scalar = -1
        else:
            error_increased_count = 0
        
        error = new_error

        _vprint(verbose, f'Current error: {error} @ {current_guess}, Iterations: {count}')
    
    return current_guess