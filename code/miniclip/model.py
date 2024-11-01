import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # input: (B, C, H, W)
        x = self.projection(x) # output: (B, E, N, N), where E=emb_size, N=img_size//patch_size
        x = x.flatten(2)    # output: (B, E, N*N)
        x = x.transpose(1, 2) # output: (B, N*N, E)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=128, num_heads=1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # input: (B, N, E), where N is number of patches , E is the embedding dimension
        x, _ = self.attn(x, x, x)  # output: (B, N, E)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=128, num_heads=1, mlp_ratio=4.0, qkv_bias=False, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, int(emb_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(emb_size * mlp_ratio), emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # input: (B, N, E), where N is the number of patches, E is the embedding dimension
        # output: (B, N, E)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, mlp_dim, dropout):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# the only difference between VisionEncoder and VisionTransformer is the head layer
class VisionEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, num_layers, mlp_dim, dropout):
        super(VisionEncoder, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        #self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        #x = self.head(x[:, 0])
        return x[:,0]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, mlp_dim, max_seq_length, dropout):
        super(TextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.max_seq_length = max_seq_length

    def forward(self, x, attention_mask=None):
        x = self.token_embedding(x)
        print("IN Text Encoder x shape", x.shape)
        print("IN Text Encoder pos_embedding shape", self.pos_embedding.shape)
        print("Attention mask shape", attention_mask.shape)
        x = x + self.pos_embedding


# Create attention mask for transformer
        if attention_mask is not None:
            # Convert mask to shape (seq_length, seq_length) with values 0 for attendable tokens
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.max_seq_length, 1)
            attention_mask = (1.0 - attention_mask) * -1e9  # Masking with -inf where mask is 0        
#        #print("Attention mask", attention_mask)
#        if attention_mask is not None:
#            x = x.masked_fill(attention_mask == 0, 0)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        return x
    
#model = VisionEncoder(img_size=32, patch_size=8, in_channels=1, num_classes=10, embed_dim=128, num_heads=1, num_layers=1, mlp_dim=4, dropout=0.1)
##summary(model, (1,32,32))
#x = torch.randn(32, 1, 32, 32)
#output = model(x)
#print(output.shape)
#
#model = TextEncoder(vocab_size=100, embed_dim=128, num_heads=1, num_layers=1, mlp_dim=4, max_seq_length=10, dropout=0.1)
#x = torch.randint(0, 100, (32, 10))
#output = model(x)
#print(output.shape)