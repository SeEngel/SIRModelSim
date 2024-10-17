import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
from utils import simulate_with_noise_on_params


st.markdown("# Simulation with Noise on Parameters")
st.markdown("""
```python
dSdt = mu * (N - S) - eta * k * I * S
dIdt = eta * k * I * S - (gamma + mu) * I
dRdt = gamma * I - mu * R

S(0) = S0
I(0) = I0
R(0) = R0

simulation = z(parameter+noise)

Noise model:

noisy_params = {
    "S0": np.random.normal(S0, noise_level * 0.1 * abs(S0)),
    "I0": np.random.normal(I0, noise_level * 0.1 * abs(I0)),
    "R0": np.random.normal(R0, noise_level * 0.1 * abs(R0)),
    "muh": np.random.normal(muh, noise_level * 0.1 * abs(muh)),
    "eta": np.random.normal(eta, noise_level * 0.1 * abs(eta)),
    "gamma": np.random.normal(gamma, noise_level * 0.1 * abs(gamma)),
    "N": round(np.random.normal(N, noise_level * 0.1 * abs(N))),
    "k": np.random.normal(k, noise_level * 0.1 * abs(k)),
}   
```
""")

# Input fields for parameters
S0 = st.number_input("Initial susceptible population (S0)", value=0.99)
I0 = st.number_input("Initial infected population (I0)", value=0.01)
R0 = st.number_input("Initial recovered population (R0)", value=0.0)
muh = st.number_input("Birth and death rate (muh)", value=0.01)
eta = st.number_input("Infection coefficient (eta)", value=0.1)
gamma = st.number_input("Recovery rate (gamma)", value=0.05)
N = st.number_input("Total population (N)", value=100)
k = st.number_input("Interaction rate (k)", value=0.1)
# Input fields for time parameters
t_end = st.number_input("End time for the simulation (t_end)", value=160.0)
t_steps = st.number_input("Steps the time points at which to store the computed solution (t_steps)", value=300.0)
noise_level = st.number_input("Noise Level Variance", value=2.0, min_value=0.001)
N_samples = st.number_input("Number of Samples (N_samples)", value=100, min_value=1)


# Collect parameters
params = {
    "S0": S0,
    "I0": I0,
    "R0": R0,
    "muh": muh,
    "eta": eta,
    "gamma": gamma,
    "N": N,
    "k": k,
    "t_end": t_end,
    "t_steps": t_steps
}

# Simulate and display results
if st.button("Run Simulation"):
    t_samples_paramsSampled = simulate_with_noise_on_params(**params, noise_level=noise_level, N_samples=N_samples)
    t = t_samples_paramsSampled[0]
    samples = t_samples_paramsSampled[1]
    paramsSampled = t_samples_paramsSampled[2]
    # Convert samples to DataFrame for S, I, and R
    S_samples = pd.DataFrame([sample[0] for sample in samples]).T
    I_samples = pd.DataFrame([sample[1] for sample in samples]).T
    R_samples = pd.DataFrame([sample[2] for sample in samples]).T

    S_samples["t"] = t
    I_samples["t"] = t
    R_samples["t"] = t

    # Calculate mean for S, I, and R
    mean_S = S_samples.drop(columns=["t"]).mean(axis=1)
    mean_I = I_samples.drop(columns=["t"]).mean(axis=1)
    mean_R = R_samples.drop(columns=["t"]).mean(axis=1)

    # Plot mean of S, I, and R in one plot
    fig_mean, ax_mean = plt.subplots()
    ax_mean.plot(S_samples["t"], mean_S, color='blue', label='Mean S')
    ax_mean.plot(I_samples["t"], mean_I, color='green', label='Mean I')
    ax_mean.plot(R_samples["t"], mean_R, color='red', label='Mean R')
    ax_mean.set_xlabel('Time')
    ax_mean.set_ylabel('Population')
    ax_mean.legend()
    st.pyplot(fig_mean)

    # Plot S samples
    fig_S, ax_S = plt.subplots()
    for column in S_samples.drop(columns=["t"]).columns:
        ax_S.plot(S_samples["t"], S_samples[column], color='grey', alpha=0.5)
    ax_S.plot(S_samples["t"], mean_S, color='blue', label='Mean S')
    ax_S.set_xlabel('Time')
    ax_S.set_ylabel('S')
    ax_S.legend()
    st.pyplot(fig_S)

    # Plot I samples
    fig_I, ax_I = plt.subplots()
    for column in I_samples.drop(columns=["t"]).columns:
        ax_I.plot(I_samples["t"], I_samples[column], color='grey', alpha=0.5)
    ax_I.plot(I_samples["t"], mean_I, color='green', label='Mean I')
    ax_I.set_xlabel('Time')
    ax_I.set_ylabel('I')
    ax_I.legend()
    st.pyplot(fig_I)

    # Plot R samples
    fig_R, ax_R = plt.subplots()
    for column in R_samples.drop(columns=["t"]).columns:
        ax_R.plot(R_samples["t"], R_samples[column], color='grey', alpha=0.5)
    ax_R.plot(R_samples["t"], mean_R, color='red', label='Mean R')
    ax_R.set_xlabel('Time')
    ax_R.set_ylabel('R')
    ax_R.legend()
    st.pyplot(fig_R)

