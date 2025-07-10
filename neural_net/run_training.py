from torch.utils.data import DataLoader

from training import train_model
from model import FlexibleMLP
from loader import BallisticsDataset

train_dataset = BallisticsDataset("training_files/train.csv")
val_dataset   = BallisticsDataset("training_files/val.csv")

model = FlexibleMLP(input_size=15, hidden_layers=[32, 16, 8], output_size=3)

trained_model = train_model(
    model,
    train_dataset,
    val_dataset,
    batch_size=32,
    lr=1e-3,
    max_epochs=100,
    patience=5
)