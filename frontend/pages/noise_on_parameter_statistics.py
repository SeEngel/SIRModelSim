import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from utils import simulate_with_noise_on_params, VAE
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
# Markdown hinzufügen
st.markdown("# Simulation with Noise on Parameters Reconstruction")
st.markdown("""
```python
dSdt = mu * (N - S) - eta * k * I * S
dIdt = eta * k * I * S - (gamma + mu) * I
dRdt = gamma * I - mu * R

S(0) = S0
I(0) = I0
R(0) = R0

simulation = z(parameter+noise)

IE(XGBoost(simulation)->parameters) -> parameters

Noise model:

noisy_params = {
    "S0": np.random.normal(S0, noise_level * 0.1 * abs(S0)),
    "I0": np.random.normal(I0, noise_level * 0.1 * abs(I0)),
    "R0": np.random.normal(R0, noise_level * 0.1 * abs(R0)),
    "muh": np.random.normal(muh, noise_level * 0.1 * abs(muh)),
    "eta": np.random.normal(eta, noise_level * 0.1 * abs(eta)),
    "gamma": np.random.normal(gamma, noise_level * 0.1 * abs(gamma)),
    "N": round(np.random.normal(N, noise_level * 0.1 * abs(N))),
    "k": np.random.normal(k, noise_level * 0.1 * abs(k))
}   
```
""")
st.markdown("""
This page simulates an SIR (Susceptible-Infected-Recovered) model with noise on the infection and recovery rates (beta and gamma) and displays the results.
XGBoost is used to train from randomized parameters (labels) and corresponding infection curves (features).
Then, the trained XGBoost model is used as a prediction model to estimate the parameters for each sample based on the infection curves with added Gaussian noise.
A variational autoencoder (VAE) is used to embed samples into a 2-dimensional latent space, and these embeddings are used for a final XGBoost training to map given measured SIR simulations to parameters.
The results are statistically presented, and the mean estimated parameters are compared to the true parameter infection curves.
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
M = st.number_input("Artifical Trainset size (M)", value=100, min_value=1)


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
    
    # Erstelle DataFrames für S, I und R
    all_samples_S = pd.DataFrame([sample[0] for sample in samples]).T
    all_samples_S["t"] = t

    all_samples_I = pd.DataFrame([sample[1] for sample in samples]).T
    all_samples_I["t"] = t

    all_samples_R = pd.DataFrame([sample[2] for sample in samples]).T
    all_samples_R["t"] = t
    
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

    with st.expander("Show S samples"):
        fig_S, ax_S = plt.subplots()
        for column in S_samples.drop(columns=["t"]).columns:
            ax_S.plot(S_samples["t"], S_samples[column], color='grey', alpha=0.5)
        ax_S.plot(S_samples["t"], mean_S, color='blue', label='Mean S')
        ax_S.set_xlabel('Time')
        ax_S.set_ylabel('S')
        ax_S.legend()
        st.pyplot(fig_S)

    with st.expander("Show I samples"):
        fig_I, ax_I = plt.subplots()
        for column in I_samples.drop(columns=["t"]).columns:
            ax_I.plot(I_samples["t"], I_samples[column], color='grey', alpha=0.5)
        ax_I.plot(I_samples["t"], mean_I, color='green', label='Mean I')
        ax_I.set_xlabel('Time')
        ax_I.set_ylabel('I')
        ax_I.legend()
        st.pyplot(fig_I)

    with st.expander("Show R samples"):
        fig_R, ax_R = plt.subplots()
        for column in R_samples.drop(columns=["t"]).columns:
            ax_R.plot(R_samples["t"], R_samples[column], color='grey', alpha=0.5)
        ax_R.plot(R_samples["t"], mean_R, color='red', label='Mean R')
        ax_R.set_xlabel('Time')
        ax_R.set_ylabel('R')
        ax_R.legend()
        st.pyplot(fig_R)



    # Infer and display parameter reconstruction
    with st.spinner("Running Reconstruction..."):
        # Collect parameters
        t_samples_paramsSampled = simulate_with_noise_on_params(**params, noise_level=noise_level, N_samples=M)
        t = t_samples_paramsSampled[0]
        samples = t_samples_paramsSampled[1]
        paramsSampled = t_samples_paramsSampled[2]
        S_samples = pd.DataFrame([sample[0] for sample in samples]).T
        I_samples = pd.DataFrame([sample[1] for sample in samples]).T
        R_samples = pd.DataFrame([sample[2] for sample in samples]).T
        # Concatenate S_samples, I_samples, and R_samples to create SIR_samples
        SIR_samples = pd.concat([S_samples, I_samples, R_samples], axis=0).T
                
        # Convert SIR_samples and paramsSampled to DataFrames
        samples_df = SIR_samples
        # Ensure unique column names for samples_df
        samples_df.columns = [f"column_{i}" for i in range(samples_df.shape[1])]
        params_df = pd.DataFrame(paramsSampled)
        
        # Show statistics of the data set
        with st.expander("Show Statistics of the Simulated Train Data Set"):
            st.write("Statistics of the simulated train data set:")
            st.write(samples_df.describe())

        
        # Ensure params_df has the same number of rows as samples_df
        if len(params_df) != len(samples_df):
            st.error("The number of rows in params_df does not match the number of rows in samples_df.")
        else:


            # Prepare data
            samples_tensor = torch.tensor(samples_df.values, dtype=torch.float32)
            dataset = TensorDataset(samples_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Model parameters
            input_dim = samples_df.shape[1]
            hidden_dim = 64
            latent_dim = 2

            # Initialize model, optimizer
            model = VAE(input_dim, hidden_dim, latent_dim)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # Train VAE
            num_epochs = 50
            model.train()
            for epoch in range(num_epochs):
                train_loss = 0
                for batch in dataloader:
                    data = batch[0]
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(data)
                    loss = VAE.vae_loss(recon_batch, data, mu, logvar)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}')

            # Encode the samples to 2D latent space
            model.eval()
            with torch.no_grad():
                mu, _ = model.encode(samples_tensor)
                encoded_samples = mu.numpy()

            # Plot the 2D embedding space
            fig_embedding, ax_embedding = plt.subplots()
            scatter = ax_embedding.scatter(encoded_samples[:, 0], encoded_samples[:, 1], cmap='viridis')
            legend1 = ax_embedding.legend(*scatter.legend_elements(), title="Parameters")
            ax_embedding.add_artist(legend1)
            ax_embedding.set_xlabel('Latent Dimension 1')
            ax_embedding.set_ylabel('Latent Dimension 2')
            ax_embedding.set_title('2D Embedding Space')
            st.pyplot(fig_embedding)

            # Train XGBoost model on the 2D embeddings
            model_xgb = XGBRegressor()
            model_xgb.fit(encoded_samples, params_df)

        # Display the model's parameters
        st.write("Model trained successfully!")
        
        all_samples_df = pd.concat([all_samples_S, all_samples_I, all_samples_R], axis=0).T

        # Encode all_samples_df using the VAE model
        all_samples_tensor = torch.tensor(all_samples_df.values, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(all_samples_tensor)
            encoded_all_samples = mu.numpy()

        # Predict parameters for each encoded sample
        predicted_params = model_xgb.predict(encoded_all_samples)

        # Convert predicted parameters to DataFrame
        predicted_params_df = pd.DataFrame(predicted_params, columns=params_df.columns)

        # Show boxplots of the predicted parameters
        st.write("Boxplots of the predicted parameters:")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Create boxplots for each parameter
        predicted_params_df.boxplot(ax=ax)

        # Add stars for the user-set parameters
        user_params = [params[col] for col in predicted_params_df.columns]
        for i, param in enumerate(user_params):
            ax.plot(i + 1, param, 'r*', markersize=15, label='User-set parameter' if i == 0 else "")

        ax.set_title("Boxplots of Predicted Parameters with User-set Parameters")
        ax.legend()

        st.pyplot(fig)

        # Plot the solution curves
        st.write("Plotting solution curves...")

        # Simulate with user parameters
        response = requests.post("http://localhost:8000/simulate", json=params)
        task_id = response.json()["task_id"]

        # Poll for results
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                st.write("Processing...")
                time.sleep(1)

        # Display results
        if status == "completed":
            result = result["result"]
            st.write("Simulation with user parameters completed!")
            user_solution = pd.DataFrame({
            "t": result["t"],
            "S": result["S"],
            "I": result["I"],
            "R": result["R"]
            })

        # Simulate with mean of reconstructed parameters
        mean_reconstructed_params = predicted_params_df.mean().to_dict()
        mean_reconstructed_params["N"] = round(mean_reconstructed_params["N"])
        mean_reconstructed_params["t_end"] = t_end
        mean_reconstructed_params["t_steps"] = t_steps

        response = requests.post("http://localhost:8000/simulate", json=mean_reconstructed_params)
        task_id = response.json()["task_id"]

        # Poll for results
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                st.write("Processing...")
                time.sleep(1)

        # Display results
        if status == "completed":
            result = result["result"]
            st.write("Simulation with mean reconstructed parameters completed!")
            mean_solution = pd.DataFrame({
            "t": result["t"],
            "S": result["S"],
            "I": result["I"],
            "R": result["R"]
            })

        # Plot all sample curves for S in grey
        fig_S, ax_S = plt.subplots()
        for column in all_samples_S.drop(columns=["t"]).columns:
            ax_S.plot(all_samples_S["t"], all_samples_S[column], color='grey', alpha=0.5)
        ax_S.plot(user_solution["t"], user_solution["S"], color='red', label='User Solution S')
        ax_S.plot(mean_solution["t"], mean_solution["S"], color='blue', label='Mean Reconstructed Solution S')
        ax_S.set_xlabel('Time')
        ax_S.set_ylabel('S')
        ax_S.legend()
        st.pyplot(fig_S)

        # Plot all sample curves for I in grey
        fig_I, ax_I = plt.subplots()
        for column in all_samples_I.drop(columns=["t"]).columns:
            ax_I.plot(all_samples_I["t"], all_samples_I[column], color='grey', alpha=0.5)
        ax_I.plot(user_solution["t"], user_solution["I"], color='red', linestyle='--', label='User Solution I')
        ax_I.plot(mean_solution["t"], mean_solution["I"], color='blue', linestyle='--', label='Mean Reconstructed Solution I')
        ax_I.set_xlabel('Time')
        ax_I.set_ylabel('I')
        ax_I.legend()
        st.pyplot(fig_I)

        # Plot all sample curves for R in grey
        fig_R, ax_R = plt.subplots()
        for column in all_samples_R.drop(columns=["t"]).columns:
            ax_R.plot(all_samples_R["t"], all_samples_R[column], color='grey', alpha=0.5)
        ax_R.plot(user_solution["t"], user_solution["R"], color='red', linestyle=':', label='User Solution R')
        ax_R.plot(mean_solution["t"], mean_solution["R"], color='blue', linestyle=':', label='Mean Reconstructed Solution R')
        ax_R.set_xlabel('Time')
        ax_R.set_ylabel('R')
        ax_R.legend()
        st.pyplot(fig_R)