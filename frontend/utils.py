


import time
import numpy as np
import requests
import torch
import torch.nn as nn



def simulate_with_noise_on_params(S0, I0, R0, muh, eta, gamma, N, k, t_end, t_steps, noise_level=0.1, N_samples=10):
    samples = []
    noisy_params_list = []
    for _ in range(N_samples):
        para_list = ["S0", "I0", "R0", "muh", "eta", "gamma", "N", "k"]
        noisy_params = {
            "S0": np.random.normal(S0, noise_level * 0.1 * abs(S0)),
            "I0": np.random.normal(I0, noise_level * 0.1 * abs(I0)),
            "R0": np.random.normal(R0, noise_level * 0.1 * abs(R0)),
            "muh": np.random.normal(muh, noise_level * 0.1 * abs(muh)),
            "eta": np.random.normal(eta, noise_level * 0.1 * abs(eta)),
            "gamma": np.random.normal(gamma, noise_level * 0.1 * abs(gamma)),
            "N": round(np.random.normal(N, noise_level * 0.1 * abs(N))),
            "k": np.random.normal(k, noise_level * 0.1 * abs(k)),
            "t_end": t_end,
            "t_steps": t_steps
        }   
        new_noisy_params = {k: v for k, v in noisy_params.items() if k in para_list}
        noisy_params_list.append(new_noisy_params)
        response = requests.post("http://localhost:8000/simulate", json=noisy_params)
        task_id = response.json()["task_id"]
        
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                time.sleep(1)
        
        if status == "completed":
            result = result["result"]
            t = result["t"]
            S = np.array(result["S"])
            I = np.array(result["I"])
            R = np.array(result["R"])
            samples.append([S, I, R])
        else:
            raise Exception(result["message"])
    
    return [t, samples, noisy_params_list]




# Define VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decode(z)
        return recon_batch, mu, logvar

    # Loss function
    def vae_loss(recon_x, x, mu, logvar):
        # Ensure the target values are between 0 and 1
        x = torch.clamp(x, 0, 1)
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD