from typing import Dict
import math

from gradients import State
from simulator import generate_forward_ballistics

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

    # Estimate elevation, based on twice the ballistic prediction
    el = math.asin(60 * firing_state['gravity'] * firing_state['target_distance'] / math.pow(firing_state['muzzle_velocity'], 2))

    # Assume straight line trajectory
    time = 2 * 60 * (firing_state['target_distance'] / firing_state['muzzle_velocity'])

    return az, el, time

def calculate_error(firing_state, current_solution):

    position = generate_forward_ballistics(firing_state, *current_solution)

    # calculate distance between calculated position, and actual position

    return math.sqrt(sum([math.pow(a - b, 2) for a, b in zip(position, firing_state['target'])]))

def calculate_firing_solution(firing_state: Dict[str, float], learning_rate = .0001, tolerance = .1, max_iterations = 10, verbose = False):

    current_state = State(firing_state)

    _vprint(verbose, f'Calculating solution for: {firing_state["gun_name"]}')

    current_guess = calculate_inital_guess(firing_state)

    print(f'Starting error: {calculate_error(firing_state, current_guess)}')

    error = math.inf

    count = 0

    while error > tolerance:

        count += 1
        if count > max_iterations:
            print("Failed to calculate for conditions:")
            print(f'For gun: {firing_state["gun_name"]}')
            print(f'Velocity: {firing_state["muzzle_velocity"]}')
            print(f'Drag: {firing_state["projectile_drag"]}')
            print()
            print(f'Wind: {firing_state["wind_velocity"]} at direction: {firing_state["wind_direction"]}')
            print()
            print(f'Target is at:')
            print(f'X: {firing_state["t_x"]}')
            print(f'Y: {firing_state["t_y"]}')
            print(f'Z: {firing_state["t_z"]}')
            return 

        # Calculate gradients
        gradient_az   = current_state.partial_az(*current_guess)
        gradient_el   = current_state.partial_el(*current_guess)
        gradient_time = current_state.partial_time(*current_guess)

        # Descend the gradient
        new_az   = current_guess[0] - learning_rate * gradient_az
        new_el   = current_guess[1] - learning_rate * gradient_el
        new_time = current_guess[2] - learning_rate * gradient_time

        current_guess = (new_az, new_el, new_time)

        error = calculate_error(firing_state, current_guess)

        _vprint(verbose, f'Current error: {error} @ {current_guess}')
    
    return current_guess