import torch
import math

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          N: int, d_model: int, h: int):

    head_dim = d_model // h
    scale = 1.0 / math.sqrt(head_dim)

    # Output buffer
    # output: (N, d_model)

    # We will fill "output" directly; do not allocate extra big buffers
    for head in range(h):
        # Slice for this head
        qs = Q[:, head*head_dim:(head+1)*head_dim]   # (N, head_dim)
        ks = K[:, head*head_dim:(head+1)*head_dim]   # (N, head_dim)
        vs = V[:, head*head_dim:(head+1)*head_dim]   # (N, head_dim)

        # Compute attention logits
        # (N, head_dim) @ (head_dim, N) -> (N, N)
        scores = torch.matmul(qs, ks.transpose(0, 1)) * scale

        # Softmax row-wise
        attn = torch.softmax(scores, dim=1)          # (N, N)

        # Weighted sum: (N, N) @ (N, head_dim) → (N, head_dim)
        out_h = torch.matmul(attn, vs)

        # Write the result for this head into output buffer
        output[:, head*head_dim:(head+1)*head_dim] = out_h
