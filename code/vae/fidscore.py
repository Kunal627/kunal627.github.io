import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from modeldef import ConvVAE

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 30  # Dimension of the latent space
batch_size = 256
# load the checkpoint
ckpt = 'vae_checkpoint.pth_epoch_81.pth'

checkpoint = torch.load(f'./checkpoints/{ckpt}')
model = ConvVAE(latent_dim=latent_dim).to(device)

# Transformations for MNIST dataset
transform = transforms.Compose([
    transforms.Grayscale(3), # Convert to 3 channels
    transforms.Resize(299),  # Resize to 299x299 (Inception-v3 input size)
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(torch.uint8))
#    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1] range,
#    transforms.Lambda(lambda x: (x * 255).clamp(0, 255).to(torch.uint8))
])

transform_gen = transforms.Compose([
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    transforms.Resize(299),
    transforms.Lambda(lambda x: x.to(torch.uint8))
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#    transforms.Lambda(lambda x: (x * 255).clamp(0, 255).to(torch.uint8))
])

# Load the MNIST dataset
mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_data, batch_size=batch_size, shuffle=False)

fid = FrechetInceptionDistance(feature=2048).to(device)

# Extract features from real images and random noise (as "fake" images)
for real_images, _ in dataloader:
    real_images = real_images.to(device)
    with torch.no_grad():
        z = torch.randn(batch_size, model.fc2.in_features)  # Sample from the latent space
        generated = model.decode(z).view(-1, 1, 28, 28).cpu()  # Generate images from the latent vectors
        fake_images = torch.stack([transform_gen(img) for img in generated])
    # Update the FID metric with real and fake images
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

# Compute the FID score
fid_score = fid.compute().item()
print(f"FID score: {fid_score}")
