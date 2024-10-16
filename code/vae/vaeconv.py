import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from modeldef import ConvVAE, loss_function, save_checkpoint

# Hyperparameters
checkpoint_interval = 3
batch_size = 256
learning_rate = 1e-3
num_epochs = 100
step_size = 20
gamma = 0.5
latent_dim = 30  # Dimension of the latent space
latent_interval = 2

train_loss_values = []
val_loss_values = []
kl_divergence_values = []

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()
                                ])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)


# Model, Optimizer
model = ConvVAE(latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training Loop
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    train_kl = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data)
        loss, kl_div = loss_function(recon_batch, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        train_kl += kl_div.item()
        optimizer.step()

    #scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader.dataset):.4f} KL: {train_kl/len(train_loader.dataset):.4f}')
    # Store values for plotting
    train_loss_values.append(train_loss/len(train_loader.dataset))
    kl_divergence_values.append(train_kl/len(train_loader.dataset))

    #if epoch % 2 == 0:
    #    generate_images(model, num_rows=4, num_cols=5)
    #
    if epoch % checkpoint_interval == 0:
        save_checkpoint(model, optimizer, epoch, latent_dim, checkpoint_dir="checkpoints", filename="vae_checkpoint.pth")

    #if epoch % latent_interval == 0:
    #    latent_vectors = get_latent_space(model, val_loader)
    #    latent_vectors = torch.cat(latent_vectors, dim=0)
    #    print(f"Latent space shape: {latent_vectors.shape}")
    #    visualize_latent_space(latent_vectors)
    #    model.train()

# Plotting after training
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_values, 'b', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot KL divergence
plt.subplot(1, 2, 2)
plt.plot(epochs, kl_divergence_values, 'g', label='KL Divergence')
plt.title('KL Divergence Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
