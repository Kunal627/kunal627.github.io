import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from common import save_checkpoint
from model  import Generator, Discriminator

# Hyperparameters
z_dim = 100
#lr = 0.0002
gen_lr = 0.0002
disc_lr = 0.0001
batch_size = 128
epochs = 50
checkpoint_interval = 3
num_generator_updates = 3

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Instantiate the generator and discriminator
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers and loss function
optim_g = optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# DataLoader
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True),
    batch_size=batch_size, shuffle=True
)

# Lists to store epoch losses
g_losses = []
d_losses = []

# Training Loop
for epoch in range(epochs):
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
    for batch_idx, (real, _) in progress_bar:
        real = real.to(device)
        batch_size = real.size(0)

        # Labels for real and fake images
        #real_labels = torch.ones(batch_size, 1).to(device)
        #fake_labels = torch.zeros(batch_size, 1).to(device)

        # apply label smoothing to real labels
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)

        # ====================
        # Train Discriminator
        # ====================
        # Real images
        outputs = discriminator(real)
        d_loss_real = criterion(outputs, real_labels)

        # Fake images
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
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
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = generator(z)
            outputs = discriminator(fake)
            g_loss = criterion(outputs, real_labels)  # Trick discriminator to think fake images are real

            # Backpropagation and optimization
            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()
            g_loss_epoch += g_loss.item()

        # Update tqdm with loss values
        progress_bar.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        #break
    
    # Calculate average losses for the epoch
    avg_g_loss = g_loss_epoch / (len(dataloader) * num_generator_updates)
    avg_d_loss = d_loss_epoch / len(dataloader)
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)

    # Print losses and visualize generated images after each epoch
    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")

    if epoch % checkpoint_interval == 0:
        save_checkpoint(generator, optim_g, epoch, z_dim, checkpoint_dir="./out/checkpoints/dcgan/", filename="dcgan_genupd3_checkpoint")

    # Generate images for visualization
    with torch.no_grad():
        # hardcode number of images to generate = btach_size
        z = torch.randn(128, z_dim, 1, 1).to(device)
        fake = generator(z).cpu()
        print(fake.shape)
        grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.title(f"Epoch {epoch + 1}")
        plt.savefig(f"./images/dcgan_genupd3_epoch_{epoch + 1}.png")
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
    plt.savefig('./images/dcgan_genupd3_losses.png')  # Save the final figure
    #plt.show()