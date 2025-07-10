import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def train_model(
    model,
    train_dataset,
    val_dataset,
    batch_size=32,
    lr=1e-3,
    max_epochs=100,
    patience=10,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(10000 * loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                val_loss = criterion(preds, y_val)
                val_losses.append(10000 * val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        e_string = ''
        if epochs_no_improve > 0:
            e_string = str(epochs_no_improve)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}", e_string)

    torch.save(best_model_state, 'best_model.pth')

    # Restore best weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model
