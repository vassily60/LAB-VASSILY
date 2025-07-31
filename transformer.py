import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

# A simple transformer-like block with NVTX annotations
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim):
        super(SimpleTransformerBlock, self).__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # LayerNorm on input
        nvtx.range_push("LayerNorm1")
        x_norm = self.norm1(x)
        nvtx.range_pop()
        
        # Q, K, V projection
        nvtx.range_push("QKV Projection")
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        nvtx.range_pop()
        
        # Attention: compute scaled dot-product
        nvtx.range_push("Attention Scoring")
        # Transpose last two dimensions for dot-product: (batch, seq_len, dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        nvtx.range_pop()
        
        # Softmax over attention scores
        nvtx.range_push("Softmax")
        attn_weights = torch.softmax(attn_scores, dim=-1)
        nvtx.range_pop()
        
        # Attention output: multiply weights by V then project
        nvtx.range_push("Attention Output")
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.attn_out(attn_output)
        nvtx.range_pop()
        
        # Residual connection 1
        nvtx.range_push("Residual Connection 1")
        x = x + attn_output
        nvtx.range_pop()
        
        # LayerNorm after attention
        nvtx.range_push("LayerNorm2")
        x_norm2 = self.norm2(x)
        nvtx.range_pop()
        
        # Feed Forward network
        nvtx.range_push("Feed Forward")
        ffn_out = self.ffn(x_norm2)
        nvtx.range_pop()
        
        # Residual connection 2
        nvtx.range_push("Residual Connection 2")
        out = x + ffn_out
        nvtx.range_pop()
        
        return out

if __name__ == '__main__':
    # Check that a CUDA device is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    # Set bigger input sizes here:
    batch_size, seq_len, dim = 8, 128, 1024  # Bigger sizes for better GPU utilization

    # Create random input tensor on GPU
    x = torch.randn(batch_size, seq_len, dim, device=device)

    # Create model and move it to GPU
    model = SimpleTransformerBlock(dim).to(device)

    # Warm up GPU (optional)

    # Run forward pass with synchronization for profiling
    nvtx.range_push("Transformer Full Forward Pass")
    output = model(x)
    torch.cuda.synchronize()  # make sure all GPU work finished before profiling ends
    nvtx.range_pop()

    print("Output shape:", output.shape)


