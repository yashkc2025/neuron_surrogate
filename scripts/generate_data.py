import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Simulation parameters
N_NEURONS = 5  # Number of neurons
TIMESTEPS = 1000  # Number of time steps
DT = 0.1  # ms per time step
N_RUNS = 300  # Number of simulated runs
DATA_FILE_PATH = (
    Path(__file__).resolve().parent / "../data/spike_dataset.csv"
).resolve()

# Izhikevich neuron parameters (regular spiking)
a = 0.02
b = 0.2
c = -65
d = 8


def simulate_izhikevich(I, timesteps):
    """Simulate a single neuron using Izhikevich model."""
    v = -65.0  # Membrane potential
    u = b * v  # Recovery variable
    V = np.zeros(timesteps)

    for t in range(timesteps):
        if v >= 30:  # spike threshold
            V[t] = 30
            v = c
            u += d
        else:
            V[t] = v
            dv = 0.04 * v**2 + 5 * v + 140 - u + I[t]
            du = a * (b * v - u)
            v += dv * DT
            u += du * DT
    return V


all_dfs = []

for run_id in range(N_RUNS):
    time = np.arange(0, TIMESTEPS * DT, DT)
    df = pd.DataFrame(index=range(TIMESTEPS))

    for neuron_id in range(N_NEURONS):
        # Random input current with some baseline noise
        I = 5 * np.random.randn(TIMESTEPS) + 10
        V = simulate_izhikevich(I, TIMESTEPS)
        df[f"Neuron_{neuron_id}"] = V

    df["run"] = run_id
    df["time"] = time
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv(DATA_FILE_PATH, index=False)
print(f"Generated dataset saved to {DATA_FILE_PATH}")

# plot one neuron for visualization
plt.plot(final_df["time"][:TIMESTEPS], final_df["Neuron_0"][:TIMESTEPS])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Example Neuron Spike Train")
plt.show()
