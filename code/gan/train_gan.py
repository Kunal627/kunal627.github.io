import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import GeneratorGAN, DiscriminatorGAN
import torch.nn as nn
import torchvision
from tqdm import tqdm
from common import save_checkpoint


# Define a custom weights initialization function
def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # For fully connected layers (Linear), initialize weights from a normal distribution
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Hyperparameters
dim_latent = 100      # Dimension of the latent space
lr_gen = 0.0002   
lr_dis = 0.0002
batch_size = 64
epochs = 50
checkpoint_interval = 5
num_generator_updates = 1

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize generator and discriminator
generator = GeneratorGAN(dim_latent)
discriminator = DiscriminatorGAN()

# initialize weights
generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr_dis, betas=(0.5, 0.999))

# Lists to store epoch losses
g_losses = []
d_losses = []

for epoch in range(epochs):
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", leave=True)   
    for batch_idx, (real, _) in progress_bar:
        real = real.view(-1, 28 * 28)
        batch_size = real.size(0)

        # Labels for real and fake images
        #real_labels = torch.ones(batch_size, 1)
        #fake_labels = torch.zeros(batch_size, 1)
        # apply label smoothing to real labels
        real_labels = torch.full((batch_size, 1), 0.9)
        fake_labels = torch.full((batch_size, 1), 0.0)

        # ====================
        # Train Discriminator
        # ====================
        # Real images
        outputs = discriminator(real)
        d_loss_real = criterion(outputs, real_labels)

        # Fake images
        z = torch.randn(batch_size, dim_latent)
        fake = generator(z)
        outputs = discriminator(fake.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake

        # Backpropagation and optimization
        optim_d.zero_grad()
        d_loss.backward()
        optim_d.step()
        d_loss_epoch += d_loss.item()

        # ====================
        # Train Generator
        # ====================
        for _ in range(num_generator_updates):
            #print("Training generator")
            z = torch.randn(batch_size, dim_latent) 
            fake = generator(z)
            outputs = discriminator(fake)
            g_loss = criterion(outputs, real_labels)  # Trick discriminator to think fake images are real

            # Backpropagation and optimization
            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()
            g_loss_epoch += g_loss.item()



        progress_bar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        #break

    # Calculate average losses for the epoch
    avg_g_loss = g_loss_epoch / (len(dataloader) * num_generator_updates)
    avg_d_loss = d_loss_epoch / len(dataloader)
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)

    # Print losses and visualize generated images
    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")

    if epoch % checkpoint_interval == 0:
        save_checkpoint(generator, optim_g, epoch, dim_latent, checkpoint_dir="./out/checkpoints/gan/", filename="gan_genupd1_checkpoint")

    # Generate images for visualization
    with torch.no_grad():
        plt.figure(figsize=(10, 5))
        z = torch.randn(64, dim_latent)
        fake = generator(z).view(-1, 1, 28, 28).cpu()
        grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.title(f"Epoch {epoch + 1}") 
        plt.savefig(f"./images/gan_genupd1_epoch_{epoch + 1}.png")
        #plt.show()
    
    #break

    # Final plot
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Loss During Training')
    plt.savefig('./images/gan_genupd1_losses.png')  # Save the final figure
    #plt.show()