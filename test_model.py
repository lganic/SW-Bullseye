import pandas as pd
import torch
from neural_net.model import FlexibleMLP

model = FlexibleMLP(input_size=15, hidden_layers=[32, 16, 8], output_size=3)
model.load_state_dict(torch.load("best-model.pth"))
model.eval()  # sets dropout/batchnorm to eval mode

# Load test CSV and pick one row
df = pd.read_csv("test.csv")
example_row = df.iloc[0]

# Separate input and expected output
input_cols = [
    'target_distance','target_altitude','target_velocity',
    'my_velocity','projectile_velocity','projectile_drag',
    'target_heading_x','target_heading_y',
    'target_velocity_heading_x','target_velocity_heading_y',
    'my_velocity_heading_x','my_velocity_heading_y',
    'wind_heading_x','wind_heading_y','wind_velocity'
]

output_cols = ['solution_az', 'solution_el', 'solution_time']

x = torch.tensor(example_row[input_cols].values, dtype=torch.float32).unsqueeze(0)  # shape: [1, 15]

# Send to model device
device = next(model.parameters()).device
x = x.to(device)

# Make prediction
model.eval()
with torch.no_grad():
    y_pred = model(x).cpu().numpy().flatten()

print("Predicted:", y_pred)

# Optional: Compare to ground truth
y_true = example_row[output_cols].values
print("Actual:   ", y_true)
