from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Contrastive Loss (InfoNCE Loss)
def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    batch_size = image_embeddings.size(0)
    logits = (image_embeddings @ text_embeddings.T) / temperature
    labels = torch.arange(batch_size, device=image_embeddings.device)
    loss_i2t = F.cross_entropy(logits, labels)  # Image-to-text
    loss_t2i = F.cross_entropy(logits.T, labels)  # Text-to-image
    return (loss_i2t + loss_t2i) / 2

# Validation Loop
def validate_clip(model, val_loader, tokenizer):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            # Prepare image and text inputs
            images = images.to(device)
            texts = [f"a photo of a {cifar10_classes[label]}" for label in labels]
            
            # Tokenize text inputs
            text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)

            # Forward pass to get embeddings
            image_embeddings, text_embeddings = model(images, input_ids, attention_mask)

            # Calculate similarity scores between all pairs of image and text embeddings
            similarity_matrix = image_embeddings @ text_embeddings.T

            # For each image, find the text with the highest similarity and check if it matches the label
            predicted_labels = similarity_matrix.argmax(dim=1)
            correct += (predicted_labels == torch.arange(len(labels), device=device)).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")



def train_clip(model, train_loader, optimizer, tokenizer, epoch):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        # Prepare image and text inputs
        images = images.to(device)
        texts = [f"a photo of a {cifar10_classes[label]}" for label in labels]
        
        # Tokenize text inputs
        text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward pass
        image_embeddings, text_embeddings = model(images, input_ids, attention_mask)

        # Compute contrastive loss
        loss = contrastive_loss(image_embeddings, text_embeddings)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# CIFAR-10 class names for generating text descriptions
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Tokenizer, Optimizer
model = CLIP(image_output_dim=512, text_output_dim=512).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    train_clip(model, train_loader, optimizer, tokenizer, epoch)
    validate_clip(model, val_loader, tokenizer)