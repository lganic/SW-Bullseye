import torch
from torch.utils.data import Dataset
import pandas as pd

class BallisticsDataset(Dataset):
    def __init__(self, csv_file, input_cols=None, output_cols=None, transform=None):
        df = pd.read_csv(csv_file)
        
        self.input_cols = input_cols or [
            'target_distance','target_altitude','target_velocity',
            'my_velocity','projectile_velocity','projectile_drag',
            'target_heading_x','target_heading_y',
            'target_velocity_heading_x','target_velocity_heading_y',
            'my_velocity_heading_x','my_velocity_heading_y',
            'wind_heading_x','wind_heading_y','wind_velocity'
        ]
        self.output_cols = output_cols or ['solution_az', 'solution_el']#, 'solution_time']
        self.transform = transform

        self.inputs = torch.tensor(df[self.input_cols].values, dtype=torch.float32)
        self.outputs = torch.tensor(df[self.output_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
