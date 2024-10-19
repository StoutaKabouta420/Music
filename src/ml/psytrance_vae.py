import torch
import torch.nn as nn
import torch.nn.functional as F

class PsytranceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PsytranceVAE, self).__init__()
        
        self.latent_dim = latent_dim  # Store latent_dim as an attribute
        
        # Encoder
        self.enc1 = nn.Linear(input_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc_mu = nn.Linear(256, latent_dim)
        self.enc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.dec1 = nn.Linear(latent_dim, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        return self.enc_mu(h2), self.enc_logvar(h2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.dec1(z))
        h4 = F.relu(self.dec2(h3))
        return self.dec3(h4)  # Remove sigmoid activation
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Use MSE loss instead of binary cross entropy
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD