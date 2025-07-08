from random_state_generator import RandomStateGenerator
from descent import calculate_firing_solution

sg = RandomStateGenerator()

state = sg.generate()

state = {'gun_name': 'Battle Cannon', 'muzzle_velocity': 800, 'projectile_drag': 0.002, 'gravity': 0.5, 'heading': 58.813656417270586, 'target_velocity_heading': 61.704354123771786, 'my_velocity_heading': 335.13713457716153, 'target_distance': 421.64314668249864, 'target_altitude': -139.22460035645983, 'target_velocity': -60.93023704486718, 'my_velocity': -19.87134126120039, 'wind_velocity': 0.20509200199067734, 'wind_direction': 252.1816922063915, 't_x': 218.33656934682224, 't_y': -139.22460035645983, 't_z': 360.7105288596097, 'target': (218.33656934682224, -139.22460035645983, 360.7105288596097)}

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