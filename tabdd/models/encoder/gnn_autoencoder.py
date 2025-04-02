import torch

from .base_encoder import BaseEncoder
from ..modules import GNNEncoder, MLPModule

class GNNAutoEncoder(BaseEncoder):
    name = 'GNNAutoEncoder'
    def __init__(
        self,
        graph_layer: str,
        graph_aggr: str,
        n_graph_layers: int,
        latent_dim: int,
        edge_direction: str,
        drop_edge_p: float,
        decoder_hidden_dims: tuple[int] | list[int],
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self.graph_layer = graph_layer
        self.graph_aggr = graph_aggr
        self.n_graph_layers = n_graph_layers
        self.latent_dim = latent_dim
        self.edge_direction = edge_direction
        self.drop_edge_p = drop_edge_p
        self.decoder_hidden_dims = decoder_hidden_dims

        self.encoder = GNNEncoder(
            in_dim = self.in_dim,
            emb_dim = latent_dim,
            n_layer = n_graph_layers,
            drop_edge_p = drop_edge_p,
            edge_direction = edge_direction,
            graph_layer= graph_layer,
            graph_aggr = graph_aggr,
        )
        self.decoder = MLPModule(
            in_dim = latent_dim,
            out_dim = self.in_dim,
            hidden_dims = decoder_hidden_dims,
            dropout_p=0,
            use_batchnorm=False,
        )

    def encode(
        self,
        x: torch.FloatTensor,
        **kwargs,
    ):
        return self.encoder(x)

    def decode(
        self,
        encoded: torch.FloatTensor,
        **kwargs,
    ):
        return self.decoder(encoded.to(self.device))

    def forward(
        self, 
        x: torch.FloatTensor, 
        **kwargs
    ):
        encoded = self.encode(x)        
        decoded = self.decode(encoded)

        return decoded
    
