from random_state_generator import RandomStateGenerator
from descent import calculate_firing_solution

sg = RandomStateGenerator()

state = sg.generate()

solution = calculate_firing_solution(state, verbose = True)


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
        #     '':          wind_direction,
        #     't_x':                     t_x,
        #     't_y':                     t_y,
        #     't_z':                     t_z,
        #     'target':                  (t_x, t_y, t_z)
        # }

print(f'For gun: {state["gun_name"]}')
print(f'Velocity: {state["muzzle_velocity"]}')
print(f'Drag: {state["projectile_drag"]}')
print()
print(f'Wind: {state["wind_velocity"]} at direction: {state["wind_direction"]}')
print()
print(f'Target is at:')
print(f'X: {state["t_x"]}')
print(f'Y: {state["t_y"]}')
print(f'Z: {state["t_z"]}')
print()
print(f'Solution:')
print(f'Az: {solution[0]}')
print(f'El: {solution[1]}')
print(f'T : {solution[2]}')