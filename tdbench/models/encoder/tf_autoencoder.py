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
        decoder_hidden_dims: tuple[int] | list[int],
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
        x: torch.FloatTensor,
        feature_idx: torch.LongTensor,
        **kwargs,
    ):
        feature_idx_with_cls = torch.hstack(
            [
                torch.zeros(feature_idx.size(0), 1).to(feature_idx),
                feature_idx + 1,
            ]
        )
        embeddings = self.embedding(feature_idx_with_cls)
        enc_out = self.encoder(embeddings)
        out = enc_out[:, 0]
        return out

    def decode(
        self,
        encoded: torch.FloatTensor,
        **kwargs,
    ):
        return self.decoder(encoded.to(self.device))

    def forward(self, x: torch.FloatTensor, feature_idx: torch.LongTensor, **kwargs):
        encoded = self.encode(x, feature_idx)
        decoded = self.decode(encoded)
        return decoded
