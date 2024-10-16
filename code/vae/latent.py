import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from modeldef import ConvVAE


# Hyperparameters
checkpoint_interval = 3
batch_size = 256
learning_rate = 1e-3
num_epochs = 60
step_size = 20
gamma = 0.5
latent_dim = 30  # Dimension of the latent space
latent_interval = 2

def generate_images(model, num_rows=2, num_cols=5):
    """
    Generates and visualizes images using a trained VAE model.
    
    Args:
        model: The trained VAE model.
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
    """
    model.eval()
    num_images = num_rows * num_cols
    with torch.no_grad():
        z = torch.randn(num_images, model.fc2.in_features)  # Sample from the latent space
        generated = model.decode(z).view(-1, 1, 28, 28).cpu().numpy()

        # Plot the generated images in a grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
        for i in range(num_images):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].imshow(generated[i][0], cmap='gray')
            axes[row, col].axis('off')
        plt.tight_layout()
    return plt

def get_latent_space(model, data_loader):
    print("get_latent_space")
    latent_vectors = [] 
    labels = []
    model.eval()
    with torch.no_grad():  # Disable gradient computation

        for batch in data_loader:
            # Get the input data
            inputs, label = batch  # Assuming the data loader returns (data, labels)

            # Forward pass through the encoder to get mean and log variance
            mu, log_var = model.encode(inputs)

            # Get the latent vector (z) by sampling from the Gaussian distribution
            z = model.reparameterize(mu, log_var)

            # Store the latent vectors
            latent_vectors.append(z)
            labels.append(label)

        return latent_vectors, labels

def visualize_latent_space(latent_vectors, labels=None):
    # Convert latent vectors to numpy for compatibility with t-SNE
    latent_vectors_np = latent_vectors.cpu().numpy() if isinstance(latent_vectors, torch.Tensor) else latent_vectors

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors_np)

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=5)
    
    # If labels are provided, add a colorbar
    if labels is not None:
        plt.colorbar(scatter, label='Labels', ticks=range(10))  # Assuming 10 classes (0-9 for MNIST)
        plt.clim(-0.5, 9.5)  # Set limits for better color mapping

    plt.title("2D t-SNE visualization of VAE latent space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    return plt

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# load the checkpoint
ckpt = 'vae_checkpoint.pth_epoch_81.pth'

checkpoint = torch.load(f'./checkpoints/{ckpt}')
model = ConvVAE(latent_dim=latent_dim)

# Restore the model and optimizer states
model.load_state_dict(checkpoint['model_state_dict'])

# Get the latent space for the validation set
latent_vectors, labels = get_latent_space(model, val_loader)
latent_vectors = torch.cat(latent_vectors, dim=0)
labels = torch.cat(labels).numpy()  # Ensure labels are a NumPy array for plotting
print(f"Latent space shape: {latent_vectors.shape}")
plt1 = visualize_latent_space(latent_vectors, labels=labels)
plt.savefig(f'{ckpt}.png')

pltimg = generate_images(model, num_cols=10, num_rows=8)
pltimg.savefig(f'{ckpt}_images.png')