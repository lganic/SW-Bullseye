from simulator.random_state_generator import RandomStateGenerator
from simulator.descent import calculate_firing_solution
from simulator.sim import generate_forward_ballistics, offset_target_position

sg = RandomStateGenerator()

state = sg.generate()

state = {'gun_name': 'Heavy Autocannon', 'muzzle_velocity': 600, 'projectile_drag': 0.005, 'gravity': 0.5, 'heading': 169.10733290548166, 'target_velocity_heading': 173.49205598270166, 'my_velocity_heading': 85.9008154952466, 'target_distance': 1173.1859158016475, 'target_altitude': 26.652052635533007, 'target_velocity': 20.556013648336034, 'my_velocity': 18.00741582270858, 'wind_velocity': 0.2914951205177915, 'wind_direction': 85.00509565656975, 't_x': -1152.0485145504038, 't_y': 26.652052635533007, 't_z': 221.69666925228853, 'target': (-1152.0485145504038, 26.652052635533007, 221.69666925228853)}

solution = calculate_firing_solution(state, verbose = False)

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
print('My Velocity:')
print(f'Mag: {state["my_velocity"]}')
print(f'Dir: {state["my_velocity_heading"]}')
print()
print(f'Solution:')
print(f'Az: {solution[0]}')
print(f'El: {solution[1]}')
print(f'T : {solution[2]}')

print('Double checking solution:')
x, y, z = generate_forward_ballistics(state, *solution, v=True)

print(x,y,z)

o_x, o_z = offset_target_position(state, solution[2])

print(f'Target travelled: {o_x},{o_z} during flight time')

print('Final target position at predicted time of impact:')

print(f'{state["t_x"] + o_x},{state["t_y"]},{state["t_z"] + o_z}')