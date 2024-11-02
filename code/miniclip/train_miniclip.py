import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10  # Using CIFAR10 for simplicity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from tqdm import tqdm
from model import MiniClip
import torch.nn.functional as F

batch_size = 128
sequence_length = 10
learning_rate = 1e-3
num_epochs= 50
num_heads = 8
embed_dim = 512

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Synthetic text labels based on CIFAR-10 classes
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
def generate_text_labels(labels):
    return [f"A photo of a {CIFAR_CLASSES[label]}" for label in labels]

# Contrastive Loss Function
def contrastive_loss(image_features, text_features, temperature=0.07):
    image_features = nn.functional.normalize(image_features, dim=-1)
    text_features = nn.functional.normalize(text_features, dim=-1)
    # Compute cosine similarity matrix
    logits_per_image = image_features @ text_features.T  # (batch_size, batch_size)
    logits_per_image /= temperature
    # Labels for contrastive learning
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size, device=image_features.device)

    # Cross-entropy loss for image-to-text and text-to-image
    loss_i2t = F.cross_entropy(logits_per_image, labels)  # Image-to-text loss
    loss_t2i = F.cross_entropy(logits_per_image.T, labels)  # Text-to-image loss
    
    # Average the two losses
    loss = (loss_i2t + loss_t2i) / 2
    return loss


# Initialize the CLIP model, tokenizer, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniClip(img_size=32, patch_size=8, in_channels=3, vocab_size=30522, embed_dim=embed_dim, max_seq_length=sequence_length, num_heads=num_heads).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
    for batch_idx , (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Generate simple text labels based on CIFAR classes
        text_labels = [f"A photo of a {CIFAR_CLASSES[label]}" for label in labels]
        #print(text_labels)
        
        # Tokenize text (assuming a tokenizer compatible with text encoder)
        text_tokens = tokenizer(text_labels, padding="max_length", max_length=10, return_tensors="pt").to(device)
        input_ids = text_tokens['input_ids'].to(device)
        attention_mask = text_tokens['attention_mask'].to(device)
        new_column = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype)
        modified_attention_mask = torch.cat((new_column, attention_mask), dim=1)        # Add a column of ones for CLS token

        #print("modified_attention_mask", modified_attention_mask, modified_attention_mask.shape)
        self_attention_mask = modified_attention_mask.unsqueeze(1).unsqueeze(2)
        #print("self_attention_mask", self_attention_mask, self_attention_mask.shape)

        # Forward pass
        img_f, txt_f = model(images, input_ids, attention_mask=self_attention_mask) 

        loss = contrastive_loss(img_f, txt_f)
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Validation Loop
    model.eval()
    total_correct = 0
    total_samples = 0
    val_total_loss = 0.0

    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", unit="batch")
        
        for images, labels in val_progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Generate simple text labels based on CIFAR classes
            text_labels = [f"A photo of a {CIFAR_CLASSES[label]}" for label in labels]
            text_tokens = tokenizer(text_labels, padding="max_length", max_length=10, return_tensors="pt").to(device)
            input_ids = text_tokens['input_ids'].to(device)
            attention_mask = text_tokens['attention_mask'].to(device)
            new_column = torch.ones(attention_mask.size(0), 1, dtype=attention_mask.dtype)
            modified_attention_mask = torch.cat((new_column, attention_mask), dim=1)        # Add a column of ones for CLS token

            self_attention_mask = modified_attention_mask.unsqueeze(1).unsqueeze(2)

            # Forward pass
            img_embd, txt_embd = model(images, input_ids, attention_mask=self_attention_mask) 

            # Compute cosine similarity
            similarity = torch.matmul(img_embd, txt_embd.T)  # Shape: (batch_size, batch_size)

            # Get the predicted indices by finding the max similarity for each image
            preds = similarity.argmax(dim=1)

            # Assuming that text indices correspond to the image indices in the dataset
            total_correct += (preds == torch.arange(len(labels), device=device)).sum().item()  # Count correct predictions
            total_samples += len(labels)

            loss = contrastive_loss(img_f, txt_f)
            val_total_loss += loss.item()

            # Update validation progress bar
            val_progress_bar.set_postfix(val_loss=loss.item())

    accuracy = total_correct / total_samples
    val_avg_loss = val_total_loss / len(val_loader)
    print(f"Validation Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {val_avg_loss:.4f} , Accuracy: {accuracy * 100:.2f}%")