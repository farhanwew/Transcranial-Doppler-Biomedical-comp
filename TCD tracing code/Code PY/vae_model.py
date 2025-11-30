import torch
import torch.nn as nn
import torch.nn.functional as F

class TCD_VAE(nn.Module):
    """
    1D Convolutional Variational Autoencoder for TCD signal segments.
    Input: (Batch, 1, 1024)
    Latent Dim: 32 (default)
    """
    def __init__(self, input_length=1024, latent_dim=32):
        super(TCD_VAE, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim

        # --- Encoder ---
        # Layer 1: 1024 -> 512
        self.enc_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Layer 2: 512 -> 256
        self.enc_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Layer 3: 256 -> 128
        self.enc_conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Flatten size: 64 channels * 128 spatial size = 8192
        self.flatten_dim = 64 * 128
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Layer 1: 128 -> 256
        self.dec_convT1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        
        # Layer 2: 256 -> 512
        self.dec_convT2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm1d(16)
        
        # Layer 3: 512 -> 1024
        self.dec_convT3 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1) 

    def encode(self, x):
        x = F.leaky_relu(self.bn1(self.enc_conv1(x)))
        x = F.leaky_relu(self.bn2(self.enc_conv2(x)))
        x = F.leaky_relu(self.bn3(self.enc_conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 128) # Unflatten
        x = F.leaky_relu(self.bn4(self.dec_convT1(x)))
        x = F.leaky_relu(self.bn5(self.dec_convT2(x)))
        # Output layer: No activation (linear) or Sigmoid/Tanh depending on normalization.
        # Assuming data is normalized standard scalar or minmax later. 
        # For now, linear output is safer for general signal reconstruction unless [0,1] is strictly enforced.
        x = self.dec_convT3(x) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    """
    Computes VAE loss: MSE + KLD
    """
    # 1. Reconstruction Loss (MSE)
    # Sum over features, mean over batch is usually better for stability, 
    # but standard VAE uses sum over everything. Let's use sum and normalize by batch size implicitly in optimizer.
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + kld_weight * KLD
