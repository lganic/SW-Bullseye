import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class BallisticsDataset(Dataset):
    def __init__(self, csv_file, input_cols=None, output_cols=None, transform=None):
        self.data = pd.read_csv(csv_file)
        
        self.input_cols = input_cols or [
            'target_distance','target_altitude','target_velocity',
            'my_velocity','projectile_velocity','projectile_drag',
            'target_heading_x','target_heading_y',
            'target_velocity_heading_x','target_velocity_heading_y',
            'my_velocity_heading_x','my_velocity_heading_y',
            'wind_heading_x','wind_heading_y','wind_velocity'
        ]
        self.output_cols = output_cols or ['solution_az', 'solution_el', 'solution_time']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data.loc[idx, self.input_cols].values, dtype=torch.float32)
        outputs = torch.tensor(self.data.loc[idx, self.output_cols].values, dtype=torch.float32)

        if self.transform:
            inputs = self.transform(inputs)
        return inputs, outputs
