import torch
from torch import nn
import torch.nn.functional as F
import math


def scaled_dot_prod(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    dropout_p: int = 0,
    mask: torch.LongTensor = None,
):
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e15)
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    attn_out = attn @ v
    return attn_out


"""
Original:
                        |  Input: (d_emb)   |
                        ---------------------
Linear(d_emb, d_emb) => | Q1 | Q2 | Q3 | Q4 | Q_: (d_emb//n_heads)
Linear(d_emb, d_emb) => | K1 | K2 | K3 | K4 | K_: (d_emb//n_heads)
Linear(d_emb, d_emb) => | V1 | V2 | V3 | V4 | V_: (d_emb//n_heads)
                        ---------------------
scaled_dot_product   => | A1 | A2 | A3 | A4 |
                        ---------------------
Linear(d_emb, d_emb) => |   Output: (emb)   |


With custom head dim: 

                                |  Input: (d_emb)   |
                                ---------------------
Linear(d_emb, d_qkv*n_heads) => |   Q1   |   Q2   |   Q3   |   Q4   | Q_: (d_qkv)
Linear(d_emb, d_qkv*n_heads) => |   K1   |   K2   |   K3   |   K4   | K_: (d_qkv)
Linear(d_emb, d_qkv*n_heads) => |   V1   |   V2   |   V3   |   V4   | V_: (d_qkv)
                                -------------------------------------
scaled_dot_product           => |   A1   |   A2   |   A3   |   A4   | A_: (d_qkv)
                                -------------------------------------
Linear(d_qkv*n_heads, d_emb) => |  Output: (d_emb)  |
"""


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_emb: int,
        n_heads: int,
        d_qkv: int,
        dropout_p: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.d_qkv = d_emb // n_heads if d_qkv == 0 or d_qkv is None else d_qkv
        self.d_emb = d_emb

        self.W_q = nn.Linear(d_emb, self.d_qkv * n_heads)
        self.W_k = nn.Linear(d_emb, self.d_qkv * n_heads)
        self.W_v = nn.Linear(d_emb, self.d_qkv * n_heads)
        self.W_o = nn.Linear(self.d_qkv * n_heads, d_emb)

    # reshape q, k or v to (d_batch, n_heads, d_seq, d_qkv) for input to scaled_dot_prod
    def _reshape_qkv(self, qkv: torch.FloatTensor):
        assert (
            qkv.ndim == 3
        ), "input to MHA must have 3 dimensions: (d_batch, d_seq, d_qkv) where d_qkv may be d_model or an arbirary value"
        d_batch, d_seq = qkv.shape[:2]
        return qkv.view(d_batch, d_seq, self.n_heads, self.d_qkv).transpose(1, 2)

    def forward(
        self, q_in: torch.FloatTensor, k_in: torch.FloatTensor, v_in: torch.FloatTensor
    ):
        d_batch, d_seq = q_in.shape[:2]
        # q,k,v: (d_batch, d_seq, d_emb) -> (d_batch, n_heads, d_seq, d_qkv)
        q = self._reshape_qkv(self.W_q(q_in))
        k = self._reshape_qkv(self.W_k(k_in))
        v = self._reshape_qkv(self.W_v(v_in))

        attn = scaled_dot_prod(q, k, v, self.dropout_p)
        # attn: (d_batch, d_seq, n_heads, d_qkv) -> (d_batch, d_seq, d_emb)
        attn = attn.transpose(2, 1).reshape(d_batch, d_seq, self.n_heads * self.d_qkv)
        return self.W_o(attn)
