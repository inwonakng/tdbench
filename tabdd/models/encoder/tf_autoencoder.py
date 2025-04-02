from typing import List, Tuple
import torch

from .base_encoder import BaseEncoder
from ..modules import MLPModule, TFEncoder, FeaturesEmbedding


class TFAutoEncoder(BaseEncoder):
    name = "TFAutoEncoder"

    def __init__(
        self,
        latent_dim: int,
        n_layers: int,
        n_heads: int,
        dropout_p: float,
        mlp_dim: int,
        layer_norm_eps: float,
        decoder_hidden_dims: Tuple[int] | List[int],
        d_qkv: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.mlp_dim = mlp_dim
        self.layer_norm_eps = layer_norm_eps

        self.embedding = FeaturesEmbedding(self.in_dim + 1, latent_dim)

        self.encoder = TFEncoder(
            emb_dim=self.latent_dim,
            d_qkv=d_qkv,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout_p=dropout_p,
            mlp_dim=mlp_dim,
            layer_norm_eps=layer_norm_eps,
        )

        self.decoder = MLPModule(
            in_dim=latent_dim,
            out_dim=self.in_dim,
            hidden_dims=decoder_hidden_dims,
            dropout_p=0,
            use_batchnorm=False,
        )

    def encode(
        self,
        x: torch.Tensor,
        feature_idx: torch.Tensor,
        feature_mask: torch.Tensor,
        feature_categ_mask: torch.Tensor,
        **kwargs,
    ):
        feature_val = torch.gather(x, 1, feature_idx).float()

        is_onehot = (feature_val == 1).all()
        is_cont_repeated = any(
            (feature_mask == i).sum() > 1 and not categ
            for i, categ in enumerate(feature_categ_mask)
        )
        is_ple = (not is_onehot) and is_cont_repeated

        feature_idx_with_cls = torch.hstack(
            [
                torch.zeros(feature_idx.size(0), 1).to(feature_idx),
                feature_idx + 1,
            ]
        )
        embeddings = self.embedding(feature_idx_with_cls)
        if not is_onehot and not is_ple:
            # need to add 1 for the cls embedding
            feature_val = torch.hstack(
                [
                    torch.ones(feature_val.size(0), 1).to(feature_idx).float(),
                    feature_val,
                ]
            ).unsqueeze(-1)
            embeddings = embeddings * feature_val
        elif is_ple:
            cls_embs = embeddings[:, :1, :]
            to_concat = [cls_embs]
            ori_features = feature_mask[feature_idx][0]
            for f_idx, is_categ in enumerate(feature_categ_mask):
                is_cur_feature = ori_features == f_idx
                cur_f_embs = embeddings[:, 1:, :][:, is_cur_feature, :]
                cur_f_vals = feature_val[:, is_cur_feature]
                if is_categ:
                    to_concat.append(cur_f_embs)
                else:
                    to_concat.append(
                        (cur_f_embs * cur_f_vals.unsqueeze(-1)).sum(1).unsqueeze(1)
                    )
            embeddings = torch.cat(to_concat, dim=1)
        enc_out = self.encoder(embeddings)
        out = enc_out[:, 0]

        return out

    def decode(
        self,
        encoded: torch.Tensor,
        **kwargs,
    ):
        return self.decoder(encoded.to(self.device))

    def forward(
        self,
        x: torch.Tensor,
        feature_idx: torch.Tensor,
        **kwargs,
    ):
        encoded = self.encode(
            x=x,
            feature_idx=feature_idx,
        )
        decoded = self.decode(encoded)
        return decoded
