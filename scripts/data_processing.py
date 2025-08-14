import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .config import TARGET_NEURON, TEST_RUNS


def load_and_prepare_data(DATA_FILE_PATH):
    df = pd.read_csv(DATA_FILE_PATH)

    # Spike threshold
    target_mean = df[TARGET_NEURON].mean()
    target_std = df[TARGET_NEURON].std()
    spike_threshold = target_mean + 3 * target_std
    print(f"Calculated spike threshold: {spike_threshold:.2f} mV")

    # Split runs
    all_runs = df["run"].unique()
    train_val_runs, test_runs = train_test_split(
        all_runs, test_size=TEST_RUNS, random_state=42
    )
    train_runs, validation_runs = train_test_split(
        train_val_runs, test_size=TEST_RUNS, random_state=42
    )

    train_df = df[df["run"].isin(train_runs)]
    validation_df = df[df["run"].isin(validation_runs)]

    feature_columns = df.columns.drop(["run", "time"])
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_df_scaled = train_df.copy()
    train_df_scaled[feature_columns] = feature_scaler.fit_transform(
        train_df[feature_columns]
    )
    train_df_scaled[[TARGET_NEURON]] = target_scaler.fit_transform(
        train_df[[TARGET_NEURON]]
    )

    validation_df_scaled = validation_df.copy()
    validation_df_scaled[feature_columns] = feature_scaler.transform(
        validation_df[feature_columns]
    )
    validation_df_scaled[[TARGET_NEURON]] = target_scaler.transform(
        validation_df[[TARGET_NEURON]]
    )

    return (
        train_df_scaled,
        validation_df_scaled,
        feature_columns,
        feature_scaler,
        target_scaler,
        train_runs,
        test_runs,
    )


def create_sequences(df, feature_columns):
    X, y = [], []
    for run_id in df["run"].unique():
        run_features = df[df["run"] == run_id][feature_columns].values
        run_target = df[df["run"] == run_id][TARGET_NEURON].values
        X.append(run_features)
        y.append(run_target)

    X = np.vstack(X).astype(np.float32)
    y = np.hstack(y).astype(np.float32)
    X = X[:, np.newaxis, :]  # Add sequence dim

    return X, y
