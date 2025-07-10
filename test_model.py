import os
import json
import math
import torch
from neural_net.model import FlexibleMLP
from training.bake_sample import Baker
from data_generation.simulator.sim import generate_forward_ballistics, offset_target_position

# Initialize model and load weights
model = FlexibleMLP(input_size=15, hidden_layers=[24, 48, 24, 12], output_size=3)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

testing_directory = 'test_database'
file_path = os.path.join(testing_directory, os.listdir(testing_directory)[2])

with open(file_path, 'r') as f:
    data = json.load(f)

# Initialize baker and generate a single test row
b = Baker()
row = b.bake_file(file_path)  # Replace with actual path

# Define input/output columns
input_cols = [
    'target_distance','target_altitude','target_velocity',
    'my_velocity','projectile_velocity','projectile_drag',
    'target_heading_x','target_heading_y',
    'target_velocity_heading_x','target_velocity_heading_y',
    'my_velocity_heading_x','my_velocity_heading_y',
    'wind_heading_x','wind_heading_y','wind_velocity'
]
output_cols = ['solution_az', 'solution_el', 'solution_time']

# Convert input to tensor
x = torch.tensor([row[col] for col in input_cols], dtype=torch.float32).unsqueeze(0)

# Move to device
device = next(model.parameters()).device
x = x.to(device)

# Run inference
with torch.no_grad():
    y_pred = model(x).cpu().numpy().flatten()

print("Predicted:", y_pred)

solution_az, solution_el, solution_time = y_pred
actual_solution = b.reverse_bake(data, solution_az, solution_el, solution_time)

print("Decompiled:", actual_solution)

print("Simualting ballistics...")
position = generate_forward_ballistics(data, *actual_solution)


print(f'Simulated position: {position}')

offset_position = offset_target_position(data, actual_solution[2])

x, y, z = data['target']
x += offset_position[0]
z += offset_position[1]

print(f'Actual position at predicted impact time: {x},{y},{z}')

d = math.sqrt(sum([math.pow(a - b, 2) for a, b in zip(position, (x, y, z))]))

print(f'Distance: {d}')