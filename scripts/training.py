import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from .config import (
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
    MODEL_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    DATA_FILE_PATH,
)
from .data_processing import load_and_prepare_data, create_sequences
from .models import MLPRegressor


def train_specialist_model():
    (
        train_df,
        validation_df,
        feature_columns,
        f_scaler,
        t_scaler,
        train_runs,
        test_runs,
    ) = load_and_prepare_data(DATA_FILE_PATH)
    X_train, y_train = create_sequences(train_df, feature_columns)
    X_val, y_val = create_sequences(validation_df, feature_columns)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(-1))
    validation_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val).unsqueeze(-1)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE)

    model = MLPRegressor(len(feature_columns)).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss/len(val_loader):.6f}")

    os.makedirs("./results", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    dump(f_scaler, FEATURE_SCALER_PATH)
    dump(t_scaler, TARGET_SCALER_PATH)
    print("Model and scalers saved.")
    return train_runs, test_runs
