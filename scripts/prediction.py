import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from joblib import load
from .config import (
    DEVICE,
    TARGET_NEURON,
    MODEL_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    DATA_FILE_PATH,
)
from .data_processing import create_sequences
from .models import MLPRegressor


def run_dynamic_hybrid_prediction(RUN_TO_TEST=25):
    df = pd.read_csv(DATA_FILE_PATH)
    feature_scaler = load(FEATURE_SCALER_PATH)
    target_scaler = load(TARGET_SCALER_PATH)
    feature_columns = df.columns.drop(["run", "time"])

    model = MLPRegressor(len(feature_columns)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_run_df = df[df["run"] == RUN_TO_TEST].copy()
    test_run_df_scaled = test_run_df.copy()
    test_run_df_scaled[feature_columns] = feature_scaler.transform(
        test_run_df[feature_columns]
    )

    X_single_run, _ = create_sequences(test_run_df_scaled, feature_columns)
    predictions_scaled = []

    for i in range(len(X_single_run)):
        current_sequence = (
            torch.tensor(X_single_run[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )
        with torch.no_grad():
            next_pred_scaled = model(current_sequence).item()
        predictions_scaled.append(next_pred_scaled)

    predictions_rescaled = target_scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    )
    plt.figure(figsize=(18, 8))
    plt.plot(
        test_run_df["time"],
        test_run_df[TARGET_NEURON],
        label="Ground Truth",
        color="blue",
    )
    plt.plot(
        test_run_df["time"],
        predictions_rescaled,
        label="Prediction",
        color="red",
        linestyle="--",
    )
    plt.legend()
    plt.show()
