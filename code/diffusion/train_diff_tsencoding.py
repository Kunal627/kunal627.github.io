## training time per epoch: 30 minutes
## intel i7-vpro 9th gen 16GB RAM

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from diffmodel import DiffModel
import torch.nn.functional as F
from tqdm import tqdm
from common import linear_beta_schedule, cosine_beta_schedule

batch_size = 128
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta_start = 0.0001
beta_end = 0.02
num_steps = 500


# Variance schedule
beta_linear = False
beta = linear_beta_schedule(num_steps, beta_start, beta_end) if beta_linear else cosine_beta_schedule(num_steps)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, 0)


def fwd_diff(x_0, t, noise = None):
    """
    Forward diffusion process
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)  # Reshape for broadcasting

    x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
    return x_t, noise

# Define transformations for the training dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load the CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load the CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = DiffModel(in_channels=3, out_channels=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Iterate through the training dataset
    for i, (images, _) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        #print(images.shape)
        # Sample a random timestep for each image in the batch
        t = torch.randint(0, num_steps, (images.size(0),), device=device).long()

        # Generate random Gaussian noise
        noise = torch.randn_like(images).to(device)
        
        # Perform forward diffusion
        noisy_images, noise = fwd_diff(images, t, noise=noise)
        
        # Predict the noise using the model
        predicted_noise = model(noisy_images, t)
        
        # Compute the loss (Mean Squared Error between the actual and predicted noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model checkpoint
    print("Saving model checkpoint...")
    torch.save(model.state_dict(), f"./out/checkpoints/diffusion/diffmodel_diff_model_epoch_{epoch+1}.pt")  