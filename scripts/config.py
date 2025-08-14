import torch

# General settings
TARGET_NEURON = "Neuron_0"
SEQUENCE_LENGTH = 1
TEST_RUNS = 10
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_FILE_PATH = "./data/spike_dataset.csv"

# Model and scaler paths
MODEL_PATH = "./data/aval_predictor_spike.pth"
FEATURE_SCALER_PATH = "./data/feature_scaler_sub_spike.joblib"
TARGET_SCALER_PATH = "./data/target_scaler_sub_spike.joblib"
