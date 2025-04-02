import torch

from .base_encoder import BaseEncoder
from ..modules import MLPModule, FeaturesEmbedding

class MLPAutoEncoder(BaseEncoder):
    name = 'MLPAutoEncoder'
    def __init__(
        self, 
        # in_dim: int,
        latent_dim: int, 
        encoder_dims: tuple[int] | list[int],
        decoder_dims: tuple[int] | list[int],
        dropout_p: float,
        embed_dim: int = -1,
        use_embedding: bool = False,
        **kwargs,
    ):
        """Simple linear autoencoder

        Args:
            input_dim (int): Shape of the original data.
            latent_dim (int): Shape of the latent representation of data.
            encoder_dims (list[int]): list of dimensions to use for the encoder.
            decoder_dims (list[int]): list of dimensions to use for the decoder.
        """
        super().__init__(
            # in_dim = in_dim,
            # num_features = num_features,
            **kwargs
        )

        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.dropout_p = dropout_p
        self.embed_dim = embed_dim
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = FeaturesEmbedding(self.in_dim, embed_dim)
            model_in_dim = len(self.feature_categ_mask) * embed_dim
        else:
            model_in_dim = self.in_dim

        self.encoder = MLPModule(
            in_dim = model_in_dim,
            out_dim = latent_dim,
            hidden_dims = encoder_dims,
            embed_dim = embed_dim,
            dropout_p = dropout_p,
        )

        self.decoder = MLPModule(
            in_dim=latent_dim,
            out_dim = self.in_dim,
            hidden_dims = decoder_dims,
            dropout_p = 0,
            use_batchnorm=False,
        )

        self.model_params = [self.encoder.parameters(), self.decoder.parameters()]
        self.save_hyperparameters()

    def encode(
        self,
        x: torch.FloatTensor,
        feature_idx: torch.LongTensor,
        **kwargs
    ):
        if self.use_embedding:
            feature_val = x.gather(dim = 1, index=feature_idx)
            x = self.embedding(feature_idx, feature_val).flatten(1)
        else:
            x = x.float()
        return self.encoder(x)

    def decode(
        self,
        x: torch.FloatTensor,
    ):
        return self.decoder(x)

    def forward(
        self,
        x: torch.FloatTensor,
        **kwargs
    ):
        return self.decode(self.encode(x, **kwargs))
