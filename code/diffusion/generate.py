import torch
from common import linear_beta_schedule, cosine_beta_schedule
import matplotlib.pyplot as plt
from model_def import Unet

# Load the trained model from the checkpoint
model = Unet(in_channels=3, out_channels=3)
model.load_state_dict(torch.load('unet_diff_model_epoch_1.pt'))


# Set the number of diffusion steps
num_steps = 1000

# Variance schedule
beta_start = 0.001
beta_end = 0.2
beta_linear = False
betas = linear_beta_schedule(num_steps, beta_start, beta_end) if beta_linear else cosine_beta_schedule(num_steps)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, 0)


def generate_images(model, num_images=1, steps=1000):
    # Initialize with random noise
    x_t = torch.randn(num_images, 3, 32, 32)  # Change dimensions according to your image size (e.g., 32x32 for CIFAR-10)
    
    # Set model to evaluation mode
    model.eval()

    # Iterate over the diffusion steps
    with torch.no_grad():
        for t in reversed(range(steps)):
            # Compute the noise prediction from the model
            predicted_noise = model(x_t, t)

            # Calculate the current alpha and alpha_hat
            alpha = alphas[t]
            alpha_hat_t = alpha_hat[t]

            # Denoising step
            if t > 0:  # Only add noise if not at the last step
                x_t = (1 / torch.sqrt(alpha)) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_hat_t) * predicted_noise)
                # Add noise according to the schedule
                noise = torch.randn_like(x_t)
                x_t += torch.sqrt(betas[t]) * noise
            else:
                x_t = (1 / torch.sqrt(alpha)) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_hat_t) * predicted_noise)

    return x_t

# Function to plot images
def plot_images(images):
    num_images = images.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axes[i].imshow(images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy())  # Convert to HWC format
        axes[i].axis('off')
    plt.show()

# Generate and plot images
generated_images = generate_images(model, num_images=5)
plot_images(generated_images)