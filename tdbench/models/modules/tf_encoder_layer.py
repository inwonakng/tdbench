import torch
from torch import nn
from .multihead_attention import MultiHeadAttention


class TFEncoderLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        attn_norm_eps: float,
        ffn_norm_eps: float,
        mlp_dim: int,
        d_qkv: int = None,
        dropout_p: float = None,
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.attention = MultiHeadAttention(
            d_emb=emb_dim,
            n_heads=n_heads,
            d_qkv=d_qkv,
            dropout_p=dropout_p,
        )

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, emb_dim),
        )

        self.norm_attn = nn.LayerNorm(
            emb_dim,
            eps=attn_norm_eps,
        )

        self.norm_ffn = nn.LayerNorm(
            emb_dim,
            eps=ffn_norm_eps,
        )

        self.drop = nn.Dropout(dropout_p)

    def forward(
        self,
        src: torch.FloatTensor,
        # this is for the trick used in FT. Use only the CLS token as query in last block of tf.
        q_idx: int = None,
    ):
        if self.norm_first:
            x = self.norm_attn(src)

            if q_idx is None:
                q = x
            else:
                q = x[:, q_idx : q_idx + 1, :]

            attn = self.attention(q, x, x)
            x = src + self.drop(attn)
            x = x + self.drop(self.ffn(self.norm_ffn(x)))
        else:
            attn = self.attention(src, src, src)
            x = self.norm_attn(src + self.drop(attn))
            x = self.norm_ffn(x + self.drop(self.ffn(x)))
        return x
