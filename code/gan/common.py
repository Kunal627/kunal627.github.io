import torch
import os

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
