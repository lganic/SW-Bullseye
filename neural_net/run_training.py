from torch.utils.data import DataLoader

from training import train_model
from model import FlexibleMLP
from loader import BallisticsDataset

train_dataset = BallisticsDataset("training_files/train.csv")
val_dataset   = BallisticsDataset("training_files/val.csv")

model = FlexibleMLP(input_size=15, hidden_layers=[35, 45, 55, 45, 25,  15, 6], output_size=2)

trained_model = train_model(
    model,
    train_dataset,
    val_dataset,
    batch_size=1024,
    lr=1e-5,
    max_epochs=1000000,
    patience=5
)