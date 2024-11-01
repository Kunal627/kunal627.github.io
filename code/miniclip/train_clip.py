import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10  # Using CIFAR10 for simplicity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from model import VisionEncoder, TextEncoder

batch_size = 32
sequence_length = 10
## Image Encoder using ViT
#class ImageEncoder(nn.Module):
#    def __init__(self):
#        super(ImageEncoder, self).__init__()
#        # Use a simple Vision Transformer (ViT) model for image encoding
#        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#        self.flatten = nn.Flatten()
#        self.fc1 = nn.Linear(16 * 32 * 32, 256)  # For CIFAR10 images (32x32)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = torch.relu(x)
#        x = self.flatten(x)
#        x = self.fc1(x)
#        return x
#
## Text Encoder using BERT
#class TextEncoder(nn.Module):
#    def __init__(self):
#        super(TextEncoder, self).__init__()
#        self.bert = BertModel.from_pretrained('bert-base-uncased')
#
#    def forward(self, input_ids, attention_mask):
#        outputs = self.bert(input_ids, attention_mask=attention_mask)
#        return outputs.pooler_output  # Get the pooled output
#
## CLIP Model
## vocab size = 30522 for BERT tokenizer
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.image_encoder = VisionEncoder(img_size=32, patch_size=8, in_channels=3, num_classes=10, embed_dim=128, num_heads=1, num_layers=1, mlp_dim=4, dropout=0.1)
        self.text_encoder = TextEncoder(vocab_size=30522, embed_dim=128, num_heads=1, num_layers=1, mlp_dim=4, max_seq_length=10, dropout=0.1)

    def forward(self, images, input_ids, attention_mask=None):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        return image_features, text_features

# Contrastive Loss Function
def contrastive_loss(image_features, text_features, temperature=0.07):
    image_features = nn.functional.normalize(image_features, dim=-1)
    text_features = nn.functional.normalize(text_features, dim=-1)
    logits = torch.matmul(image_features, text_features.t()) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return nn.CrossEntropyLoss()(logits, labels)

# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the CLIP model, tokenizer, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIP().to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images = images.to(device)
        
        # Create text descriptions for the images (dummy text for example)
        text_data = [f"This is a {train_dataset.classes[label]}" for label in labels]
        encoding = tokenizer(text_data, padding='max_length', truncation=True, return_tensors='pt', max_length = sequence_length).to(device)

        #print(encoding['input_ids'].shape)
        #print(encoding['input_ids'])

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        image_features, text_features = model(images, encoding['input_ids'], encoding['attention_mask'])
        
        # Calculate loss
        loss = contrastive_loss(image_features, text_features)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

print("Training completed.")
