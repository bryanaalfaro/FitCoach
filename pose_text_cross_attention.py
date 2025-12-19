import torch
import torch.nn as nn

class PoseCrossAttention(nn.Module):
    """
    Toy cross attention module:

    Inputs: pose_tokens [T, d_model] or [B, T, d_model]
    Has a small set of learnable query embeddings (e.g., "form query", "speed query")
    Uses MultiheadAttention to attend over pose tokens.
    Returns:
        logits: [B, n_queries, n_out]
        attn_weights: [B, n_queries, T] attention over time
    """
    def __init__(self, d_model=512, n_heads=4, n_queries=2, n_out=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_out = n_out

        # Learnable query embeddings: [n_queries, d_model]
        self.query_embed = nn.Parameter(torch.randn(n_queries, d_model))

        # Cross-attention from queries --> pose_tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # Small MLP head mapping each query output to some scores
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_out),
        )

    def forward(self, pose_tokens, pose_mask=None, return_attn=False):
        """
        pose_tokens: [T, d_model] or [B, T, d_model]
        pose_mask: optional [B, T] bool tensor, True = pad / ignore

        Returns:
            logits: [B, n_queries, n_out]
            attn_weights: [B, n_queries, T]  (if return_attn=True)
        """
        if pose_tokens.dim() == 2:   # [T, D]
            pose_tokens = pose_tokens.unsqueeze(0)  # [1, T, D]

        B, T, D = pose_tokens.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"

        # Expanding learnable queries to batch: [B, n_queries, D]
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        out, attn_weights = self.attn(
            query=q,
            key=pose_tokens,
            value=pose_tokens,
            key_padding_mask=pose_mask,
        )
        # out: [B, n_queries, D]

        logits = self.mlp_head(out)  # [B, n_queries, n_out]

        # Normalizing attention shape to [B, n_queries, T]
        if attn_weights.dim() == 4:
            # [n_heads, B, n_queries, T] or [B, n_heads, n_queries, T]
            if attn_weights.shape[0] == B:
                attn_mean = attn_weights.mean(dim=1)  # [B, n_queries, T]
            else:
                attn_mean = attn_weights.mean(dim=0)  # [B, n_queries, T]
        else:
            attn_mean = attn_weights

        if return_attn:
            return logits, attn_mean

        return logits
