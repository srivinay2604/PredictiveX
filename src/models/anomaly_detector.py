import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: list = [32, 16]):
        super(PyTorchAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class AnomalyDetectorAutoencoder:
    """
    Unsupervised Anomaly Detection using a PyTorch Autoencoder.
    Trained exclusively on healthy early-life operational sensor cycles.
    """
    def __init__(self, latent_dim: int = 8, hidden_dims: list = [32, 16], percentile_threshold: float = 95.0):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.percentile_threshold = percentile_threshold
        self.scaler = StandardScaler()
        self.model = None
        self.anomaly_threshold = None
        self.feature_cols = []
        
    def fit(self, healthy_df: pd.DataFrame, feature_cols: list, epochs: int = 35, lr: float = 0.001, batch_size: int = 64):
        self.feature_cols = feature_cols
        X_healthy = healthy_df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X_healthy)
        
        input_dim = X_scaled.shape[1]
        self.model = PyTorchAutoencoder(input_dim=input_dim, latent_dim=self.latent_dim, hidden_dims=self.hidden_dims)
        self.model.train()
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            print(f"Autoencoder Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(X_healthy):.4f}", flush=True)
                
        # Determine anomaly MSE threshold on healthy training data
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
            reconstructed = self.model(tensor_x)
            mse_per_sample = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()
            self.anomaly_threshold = float(np.percentile(mse_per_sample, self.percentile_threshold))
            
        print(f"Autoencoder Training Completed. Anomaly Threshold ({self.percentile_threshold}th percentile MSE): {self.anomaly_threshold:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (reconstruction_mse_anomaly_scores, is_anomaly_binary_flags).
        """
        if self.model is None or self.anomaly_threshold is None:
            raise ValueError("Autoencoder model is not trained yet.")
            
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
            reconstructed = self.model(tensor_x)
            mse_scores = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()
            
        is_anomaly = (mse_scores > self.anomaly_threshold).astype(int)
        return mse_scores, is_anomaly

    def save(self, path: str = "models/autoencoder.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "threshold": self.anomaly_threshold,
            "feature_cols": self.feature_cols,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims
        }, path)
        print(f"Saved PyTorch Autoencoder model to {path}")

    def load(self, path: str = "models/autoencoder.pth"):
        checkpoint = torch.load(path, weights_only=False)
        self.scaler = checkpoint["scaler"]
        self.anomaly_threshold = checkpoint["threshold"]
        self.feature_cols = checkpoint["feature_cols"]
        self.latent_dim = checkpoint["latent_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        
        input_dim = len(self.feature_cols)
        self.model = PyTorchAutoencoder(input_dim=input_dim, latent_dim=self.latent_dim, hidden_dims=self.hidden_dims)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print(f"Loaded PyTorch Autoencoder model from {path}")
        return self
