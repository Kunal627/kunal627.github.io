import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer


# output mask should be (batch_size, num_heads, seq_len, seq_len)
# for 2d mask (seq_len, seq_len) --> (1, 1, seq_len, seq_len)
# for 3d mask (B, seq_len, seq_len) --> (B, 1, seq_len, seq_len)
def expand_mask(mask):
    assert mask.dim() >= 2, "Mask should have at least two dimensions"

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, seq_len, seq_len)  for 3d mask
    
    while mask.dim() < 4:
        mask =  mask.unsqueeze(0) # (1, 1, seq_len, seq_len) for 2d mask

    return mask


# Patch embedding layer converts our image into a sequence of patches
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # input is an image with dimensions (batch_size, in_channels, img_size, img_size)
        x = self.projection(x) # output: (batc_size, emb_size, n, n) where n = img_size//patch_size
        x = x.flatten(2)    # output: (B, E, n * n)
        x = x.transpose(1, 2) # output: (B, n * n, E)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embd_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        #print("embd_dim", embd_dim)
        #print("num_heads", num_heads)
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads

        self.qkv_proj = nn.Linear(embd_dim, 3*embd_dim, bias=False) # project input to query, key, value
        self.fc_out = nn.Linear(embd_dim, embd_dim) # final output layer

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)


    def forward(self, x, mask=None, return_attention=False):
        # input: (B, N, E), where N is number of patches or sequence length , E is the embedding dimension
        # pathces for images and sequences for text
        B, N, _ = x.shape

        #if mask is not None:
        #    
        #    mask = expand_mask(mask)

        qkv = self.qkv_proj(x)  # (B, N, 3*E)

        #print("qkv", qkv.shape)

        qkv = qkv.reshape(B, N, self.num_heads, 3* self.head_dim)  # (B, N, 3*E) -> (B, N, num_heads, head_dim*3)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, N, num_heads, head_dim*3) -> (B, num_heads, N, head_dim*3)
        q, k, v = qkv.chunk(3, dim=-1)  # split q, k, v (B, num_heads, N, head_dim)

        #print("q", q.shape)
        #print("k", k.shape)
        #print("v", v.shape)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5 # (B, num_heads, N, N)

        #print("attn", attn.shape)
        if mask is not None:
            #print("mask in MultiHeadAttention", mask.shape)
            mask = expand_mask(mask)
            #print("mask in MultiHeadAttention after expand", mask.shape)
            #print("attention shape", attn.shape)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(attn, dim=-1) # (B, num_heads, N, N)
        values = torch.matmul(attention, v) # (B, num_heads, N, N) x (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)

        #print("values", values.shape)
        values = values.permute(0, 2, 1, 3).reshape(B, N, self.embd_dim) # (B, N, num_heads, head_dim) -> (B, N, E)

        if return_attention:
            return values, attention
        else:
            return values

class TransformerEncoder(nn.Module):
    def __init__(self, embd_dim, num_heads, dropout=0.1, mlp_multi=4):
        super(TransformerEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn = MultiHeadAttention(embd_dim, num_heads)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embd_dim, embd_dim * mlp_multi),
            nn.GELU(),
            nn.Linear(int(embd_dim * mlp_multi), embd_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask=None):
        #if attention_mask is not None:
            #print("attention_mask in TransformerEncoder", attention_mask.shape)
        #print("in transformer encoder", mask.shape)
        # input: (B, N, E), where N is the number of patches, E is the embedding dimension
        # output: (B, N, E)
        x = x + self.attn(self.ln1(x), mask=attention_mask) 
        #print("after attn", x.shape)
        x = x + self.mlp(self.ln2(x))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes=10, embed_dim=128, num_heads=1, num_layers=1, mlp_multi=4, dropout=0.1):
        super(ImageEncoder, self).__init__()
        #print("in image encoder")
        #print("num_heads", num_heads)
        assert (img_size % patch_size) == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        # CLS token for global representation from the whole image (Vision Transformer paper)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learn the position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, dropout, mlp_multi) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        #print("in forward", x.shape)
        B = x.shape[0]  # batch size
        x = self.patch_embedding(x)                     # (batch_size, num_patches, embed_dim)
        #print("after patch embedding", x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (batch_size, 1, embed_dim)
        #print("cls_tokens", cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)           # (batch_size, num_patches + 1, embed_dim)
        #print("after cat", x.shape)
        x = x + self.pos_embedding                      # (batch_size, num_patches + 1, embed_dim)
        #print("after pos embedding", x.shape)
        for layer in self.encoder:
            x = layer(x)                                # (batch_size, num_patches + 1, embed_dim)
        #print("after encoder", x.shape)
        x = self.norm(x)                                # (batch_size, num_patches + 1, embed_dim)
        return x[:, 0, :]  # only return the CLS token output for classification


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=1, num_layers=1, mlp_multi=4, max_seq_length=10, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = self._sinusoidal_positional_encoding(max_seq_length + 1, embed_dim)     # +1 for the CLS token
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, dropout, mlp_multi) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    @staticmethod
    def _sinusoidal_positional_encoding(max_seq_len, emb_size, theta = 10000.0):

        pe = torch.zeros(max_seq_len, emb_size)                # initialize the positional encoding matrix with zeros

        pos = torch.arange(0, max_seq_len).unsqueeze(1)        # create a tensor with values from 0 to max_seq_len
        dim =  torch.arange(emb_size // 2).unsqueeze(0)        # create a tensor with values from 0 to emb_size//2

        thetas = pos / torch.pow(theta , (2 * dim) / emb_size)  # calculate the theta values for each position

        pe[:, 0::2] = torch.sin(thetas)                        # apply sin to even indices
        pe[:, 1::2] = torch.cos(thetas)                        # apply cos to odd indices

        return pe.unsqueeze(0)                                 # add a batch dimension to the positional encoding matrix


    def forward(self, input_ids, attention_mask=None):
        #if attention_mask is not None:
        #    print("attention_mask in TextEncoder", attention_mask.shape)
        #print("in text encoder", attention_mask.shape)
        B = input_ids.shape[0]
        x = self.token_embedding(input_ids)                     # (batch_size, max_seq_length, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)           # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)                   # (batch_size, max_seq_length + 1, embed_dim)

        x = x + self.pos_embedding                              # (batch_size, max_seq_length, embed_dim)
        for layer in self.encoder:
            x = layer(x, attention_mask)                                        # (batch_size, max_seq_length, embed_dim)
        return x[:, 0, :]  # only return the CLS token output for classification



class MiniClip(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, vocab_size=30522, num_classes=None, embed_dim=128, max_seq_length=10, num_heads=1):
        super(MiniClip, self).__init__()
        self.image_encoder = ImageEncoder(img_size, patch_size, in_channels, embed_dim=embed_dim, num_heads=num_heads)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim, max_seq_length = max_seq_length, num_heads=num_heads)

    def forward(self, images, input_ids, attention_mask=None):
        #if attention_mask is not None:
        #    print("attention_mask in MiniClip", attention_mask.shape)
        
        image_embd = self.image_encoder(images)
        # use attention mask to mask the padding tokens
        text_embd = self.text_encoder(input_ids, attention_mask)

        return image_embd, text_embd



#########################################################################################################################################
#
# Use pretrained models for testing
# 
#########################################################################################################################################
# Pretrained Image Encoder (ResNet50)
class ImageEncoderT(nn.Module):
    def __init__(self, output_dim=512):
        super(ImageEncoderT, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, images):
        return F.normalize(self.resnet(images), dim=-1)

# Pretrained Text Encoder (BERT)
class TextEncoderT(nn.Module):
    def __init__(self, output_dim=512):
        super(TextEncoderT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return F.normalize(self.fc(pooled_output), dim=-1)

# CLIP Model combining Image and Text Encoder
class CLIP(nn.Module):
    def __init__(self, image_output_dim=512, text_output_dim=512):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoderT(output_dim=image_output_dim)
        self.text_encoder = TextEncoderT(output_dim=text_output_dim)

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        return image_embeddings, text_embeddings


#batch_size = 32
#sequence_length = 10
#image_size = 32
#patch_size = 8
#in_channels= 3
#vocab_size = 100
#embed_dim  = 128
#
#model = ImageEncoder(image_size, patch_size, in_channels, embed_dim=embed_dim)
#x = torch.randn(batch_size, in_channels, image_size, image_size)
#y = model(x)
#print("Image Encoder loaded successfully")
#print(y.shape)
#
#
#model = TextEncoder(vocab_size, embed_dim)
#x = torch.randint(0, 100, (batch_size, sequence_length))
#y = model(x)
#print("Text Encoder loaded successfully")
#print(y.shape)
