import torch
import math
from einops import rearrange, reduce, repeat, einsum

class Linear(torch.nn.Module):
    """Perform a single linear transformation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype:torch.dtype | None = None,
    ):
        self.super.__init__()
        std = math.sqrt(1.0/(in_features + out_features))
        self.A = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.tensor((out_features, in_features)),
                mean=0.0,
                std=std,
                a=-3*std,
                b=3*std,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return einsum(x, self.A, "... j, k j -> ... k")


