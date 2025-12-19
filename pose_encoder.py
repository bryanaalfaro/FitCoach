
import torch
import torch.nn as nn
import numpy as np

class SimplePoseEncoder(nn.Module):
    """
    simple pose encoder:
     Input: poses [T, 33, 4] or numpy array
     Output: pose tokens [T, d_model]
    """
    def __init__(self, d_model: int = 512, use_visibility: bool = True):
        super().__init__()
        self.use_visibility = use_visibility
        in_dim = 33 * (4 if use_visibility else 3)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, poses):
        # Accepting numpy or torch
        if not isinstance(poses, torch.Tensor):
            poses = torch.from_numpy(poses)

        poses = poses.float()

        if not self.use_visibility:
            poses = poses[..., :3]  # dropping visibility

        T = poses.shape[0]
        flat = poses.view(T, -1)   # [T, 33 * D_joint]
        tokens = self.mlp(flat)    # [T, d_model]
        return tokens
