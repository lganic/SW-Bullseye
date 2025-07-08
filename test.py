from simulator.random_state_generator import RandomStateGenerator
from simulator.descent import calculate_firing_solution
from simulator.sim import generate_forward_ballistics, offset_target_position

sg = RandomStateGenerator()

state = sg.generate()

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
print(f'Solution:')
print(f'Az: {solution[0]}')
print(f'El: {solution[1]}')
print(f'T : {solution[2]}')

print('Double checking solution:')
x, y, z = generate_forward_ballistics(state, *solution)

print(x,y,z)

o_x, o_z = offset_target_position(state, solution[2])

print(f'Target travelled: {o_x},{o_z} during flight time')

print('Final target position at predicted time of impact:')

print(f'{state["t_x"] + o_x},{state["t_y"]},{state["t_z"] + o_z}')