import torch
from torch import nn
from .tf_encoder_layer import TFEncoderLayer


class TFEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        d_qkv: int,
        n_layers: int,
        n_heads: int,
        dropout_p: float,
        mlp_dim: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_layer = n_layers
        self.transformer = nn.ModuleList(
            [
                TFEncoderLayer(
                    emb_dim=emb_dim,
                    d_qkv=d_qkv,
                    n_heads=n_heads,
                    dropout_p=dropout_p,
                    mlp_dim=mlp_dim,
                    attn_norm_eps=layer_norm_eps if i != 0 else 0,
                    ffn_norm_eps=layer_norm_eps,
                    norm_first=True,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.LongTensor, **kwargs):
        for block in self.transformer[:-1]:
            x = block(x)
        # special operation for last transformer layer.
        # Only use the cls embedding for query in attention
        last_block = self.transformer[-1]
        x = last_block(x, q_idx=0)
        return x
