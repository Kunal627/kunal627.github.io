import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

torch.manual_seed(42)


class EmbeddingT5(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingT5, self).__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight)
        self.d_model = d_model

    def forward(self, x):
        out = self.weight[x]
        return out

class RelativePosEnc(nn.Module):
    def __init__(self, max_rel_pos=2, num_heads=4):
        super(RelativePosEnc, self).__init__()
        self.max_rel_pos = max_rel_pos
        self.num_heads = num_heads
        self.rel_pos = nn.Parameter(torch.randn(num_heads, 2*max_rel_pos + 1))

    def forward(self, seq_len):
        positions = torch.arange(seq_len, dtype=torch.long) 
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        pos_clip = torch.clamp(rel_positions, -self.max_rel_pos, self.max_rel_pos)
        pos_idx = pos_clip + self.max_rel_pos

        rel_bias = self.rel_pos[:, pos_idx]
        return rel_bias


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x * self.weight
        return x


class FeedFwd(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedFwd, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_rel_pos=2, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.relative_position_embedding = RelativePosEnc(max_rel_pos, num_heads)

    def forward(self, k, q, v, mask=None):
        print("###############query shape", q.shape)
        batch_size, seq_len, d_model = q.shape

        Q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        relative_bias = self.relative_position_embedding(seq_len)  # Shape: [num_heads, seq_len, seq_len]
        attention_scores = attention_scores + relative_bias.unsqueeze(0)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = F.softmax(attention_scores, dim=-1)
        attn_out = torch.matmul(attention_scores, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(attn_out)
        return out


class T5EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_rel_pos, d_ff, dropout=0.1):
        super(T5EncoderBlock, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, max_rel_pos, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedFwd(d_model, d_ff)
    
    def forward(self, x, mask=None):
        x = self.norm1(self.attn(x, x, x, mask) + x)
        x = self.norm2(self.ff(x) + x)
        return x


class T5Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_rel_pos=2):
        super(T5Encoder, self).__init__()
        self.encoderblocks = nn.ModuleList([T5EncoderBlock(d_model, num_heads, max_rel_pos, d_ff, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.encoderblocks:
            x = layer(x, mask)    
        return x
    
class T5DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_rel_pos=2):
        super(T5DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, max_rel_pos)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, max_rel_pos)  
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1)
        )

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ln3 = RMSNorm(d_model)
        self.dout = nn.Dropout(0.1)

    def forward(self, x, enc_out, self_mask=None, cross_mask = None):
        self_attn_out = self.self_attention(x, x, x, self_mask)
        x = x + self.ln1(self_attn_out)

        cross_attn_out = self.cross_attention(x, enc_out, enc_out, cross_mask)
        x = x + self.ln2(cross_attn_out)

        ff_out = self.feedforward(x)
        x = x + self.ln3(ff_out)

        return x

class T5Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, max_rel_pos=2):
        super(T5Decoder, self).__init__()
        self.decoderblocks = nn.ModuleList([T5DecoderBlock(d_model, num_heads, d_ff, max_rel_pos) for _ in range(num_layers)])


    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        for layer in self.decoderblocks:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x


class T5(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_rel_pos=2):
        super(T5, self).__init__()
        self.embed = EmbeddingT5(vocab_size, d_model)
        self.encoder = T5Encoder(d_model, num_heads, d_ff, num_layers, max_rel_pos)
        self.decoder = T5Decoder(d_model, num_heads, d_ff, num_layers, max_rel_pos)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inp_ids, decoder_ids, enc_mask=None, dec_mask=None, cross_mask=None):
        inp_embed = self.embed(inp_ids)
        print("inp_embed", inp_embed.shape)
        enc_out = self.encoder(inp_embed, enc_mask)

        dec_embed = self.embed(decoder_ids)
        dec_out = self.decoder(dec_embed, enc_out, dec_mask , cross_mask)
        logits = self.out_proj(dec_out)

        return logits




# Model Summary Function
def model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
        print(f"{name}: {param.size()} ({param.numel()})")
    
    print(f"\nTotal trainable parameters: {total_params}")

# Test

# Hyperparameters
vocab_size = 32128  # Vocabulary size of T5
d_model = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
dropout = 0.1
max_position = 128
seq_len = 10
batch_size = 2



t5_model = T5(vocab_size, d_model, num_heads, ff_dim, num_layers, max_position)

# Input tensors (batch_size, seq_len)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
print("input_ids:", input_ids.shape)
decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Optional masks (for example purposes, no masking applied here)
encoder_mask = torch.ones(batch_size, 1, seq_len, seq_len)
decoder_mask = torch.ones(batch_size, 1, seq_len, seq_len)
cross_mask = torch.ones(batch_size, 1, seq_len, seq_len)

# Forward pass
logits = t5_model(input_ids, decoder_input_ids, encoder_mask, decoder_mask, cross_mask)

# Model Summary
model_summary(t5_model)

print("\nOutput shape:", logits.shape)  # Expected: (batch_size, seq_len, vocab_size)