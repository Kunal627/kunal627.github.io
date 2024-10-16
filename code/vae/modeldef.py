import torch
import torch.nn as nn
import os

# Convolutional VAE Model
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        
        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 14, 14)
#            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: (64, 7, 7)
#            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: (128, 4, 4)
#            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # Fully connected layers for mean and log variance
        self.fc1_mean = nn.Linear(128 * 4 * 4, latent_dim)  # Flattened input to latent_dim output
        self.fc1_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # Flattened input to latent_dim output
        
        # Decoder: Transposed Convolutional layers
        self.fc2 = nn.Linear(latent_dim, 128 * 4 * 4)  # Map latent_dim back to 128 * 4 * 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: (64, 8, 8)
#            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2, output_padding=1), # Output: (32, 16, 16)
#            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (1, 28, 28)
#            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.apply(self.initialize_weights)

    def initialize_weights(self, layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)   

    def encode(self, x):
        h = self.encoder(x)
        #print("h.shape: ", h.shape)
        h = h.view(h.size(0), -1)  # Flatten the output
        mean = self.fc1_mean(h)
        logvar = self.fc1_logvar(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        h = self.fc2(z)
        h = h.view(h.size(0), 128, 4, 4)  # Reshape to (batch_size, 128, 4, 4) for decoding
        return self.decoder(h)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar

# Loss Function: Reconstruction + KL Divergence Loss
def loss_function(recon_x, x, mean, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence Loss
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD, KLD


def save_checkpoint(model, optimizer, epoch, latent_dim, checkpoint_dir="checkpoints", filename="vae_checkpoint.pth"):
    """
    Saves a checkpoint of the model and optimizer state.
    
    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer used for training the model.
        epoch: The current epoch number.
        checkpoint_dir: The directory where to save the checkpoint.
        filename: The filename of the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint_path = os.path.join(checkpoint_dir, f"{filename}_epoch_{epoch}_ld_{latent_dim}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x