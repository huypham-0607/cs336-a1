import torch
import math

from torch import nn
from einops import rearrange, reduce, repeat, einsum

class Linear(nn.Module):
    """Perform a single linear transformation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype:torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std = math.sqrt(2.0/(in_features + out_features))
        self.A = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(size=(self.out_features, self.in_features), dtype=self.dtype, device=self.device),
                mean=0.0, std=std, a=-3*std, b=3*std,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return einsum(x, self.A, "... j, k j -> ... k")

class Embedding(nn.Module):
    """Map a index (token id) to a dense vector representation
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        std = 1.0
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    size=(self.num_embeddings, self.embedding_dim),
                    dtype=self.dtype,
                    device=self.device
                ),
                mean=0.0, std=std, a=-3*std, b=3*std,
            )
        )

    def forward(self, x: torch.Tensor):
        return self.weights[x]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g = nn.Parameter(
            torch.ones(size=[self.d_model], device=self.device, dtype=self.dtype)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = (x.square().sum(dim=-1)*(1.0/self.d_model) + self.eps).sqrt()
        res = einsum(x, rms, self.g, "... d_model, ..., d_model -> ... d_model")

        res = res.to(x.dtype)
        return res