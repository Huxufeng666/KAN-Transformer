import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(FlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        output = self.o_proj(attn_output)
        
        return output

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.flash_attention = FlashAttention(embed_dim, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_output = self.flash_attention(x, mask)
        x = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x

# Example usage:
batch_size = 32
seq_len = 128
embed_dim = 512
num_heads = 8
num_layers = 6

x = torch.randn(batch_size, seq_len, embed_dim)
mask = torch.ones(batch_size, seq_len).bool()

transformer = Transformer(num_layers, embed_dim, num_heads)
output = transformer(x, mask)
print(output.shape)  # Expected output: (batch_size, seq_len, embed_dim)
