import os
import json
import math
import torch
from neural_net.model import FlexibleMLP
from training.bake_sample import Baker
from data_generation.simulator.sim import generate_forward_ballistics, offset_target_position
from tqdm import tqdm

def find_best_launch_time(state, az, el, min_time = 0, max_time = 10000, tolerance = .0001):

    # Given an azimuth, an elevation, and a firing state, find the time of impact which minizmizes the distance
    target_position = state['target']

    def f(t):

        new_position = generate_forward_ballistics(state, az, el, t)
        target_offset = offset_target_position(state, t)

        distance = math.sqrt(sum([math.pow(a - b, 2) for a, b in zip((target_position[0] + target_offset[0], target_position[1], target_position[2] + target_offset[1]), new_position)]))

        return distance

    def golden_search(function, min_range, max_range, tolerance):

        if max_range - min_range < tolerance:

            return (max_range + min_range) / 2

        golden_ratio = (1 + math.sqrt(5)) / 2

        x1 = max_range - (max_range - min_range) / golden_ratio
        x2 = min_range + (max_range - min_range) / golden_ratio

        fx1 = function(x1)
        fx2 = function(x2)

        if fx1 < fx2:

            return golden_search(function, min_range, x2, tolerance)
        
        return golden_search(function, x1, max_range, tolerance)
    
    return golden_search(f, min_time, max_time, tolerance)
    

# Initialize model and load weights
model = FlexibleMLP(input_size=15, hidden_layers=[35, 45, 55, 45, 25,  15, 6], output_size=3)
# model = FlexibleMLP(input_size=13, hidden_layers=[20, 25, 35, 20, 14,  8, 4], output_size=2)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

testing_directory = 'test_database'
file_paths = os.listdir(testing_directory)

# Initialize baker and generate a single test row
b = Baker()

total = 0
amount = 0

for file_path in tqdm(file_paths, desc = 'Processing...'):

    file_path = os.path.join(testing_directory, file_path)

    with open(file_path, 'r') as f:
        data = json.load(f)


    row = b.bake_file(file_path)

    # Define input/output columns
    input_cols = [
        'target_distance','target_altitude','target_velocity',
        'my_velocity','projectile_velocity','projectile_drag',
        'target_heading_x','target_heading_y',
        'target_velocity_heading_x','target_velocity_heading_y',
        'my_velocity_heading_x','my_velocity_heading_y',
        'wind_heading_x','wind_heading_y','wind_velocity'
    ]

    # Convert input to tensor
    x = torch.tensor([row[col] for col in input_cols], dtype=torch.float32).unsqueeze(0)

    # Move to device
    device = next(model.parameters()).device
    x = x.to(device)

    # Run inference
    with torch.no_grad():
        y_pred = model(x).cpu().numpy().flatten()

    # print("Predicted:", y_pred)

    solution_az_x, solution_az_y, solution_el = y_pred

    actual_solution = b.reverse_bake(data, solution_az_x, solution_az_y, solution_el)

    solution_time = find_best_launch_time(data, *actual_solution)

    # print("Decompiled:", actual_solution)

    # print("Simualting ballistics...")
    position = generate_forward_ballistics(data, *actual_solution, time = solution_time)


    # print(f'Simulated position: {position}')

    offset_position = offset_target_position(data, solution_time)

    x, y, z = data['target']
    x += offset_position[0]
    z += offset_position[1]

    # print(f'Actual position at predicted impact time: {x},{y},{z}')

    d = math.sqrt(sum([math.pow(a - b, 2) for a, b in zip(position, (x, y, z))]))

    total += d
    amount += 1
    # print(f'Distance: {d}')

print(f'Average distance: {total / amount}')