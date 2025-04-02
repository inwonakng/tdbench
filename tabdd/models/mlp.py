import torch

from .base_model import BaseModel
from .modules import FeaturesEmbedding, MLPModule

class MLP(BaseModel):
    name = 'MLP'
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        dropout_p: float,
        # num_features: int = -1,
        embed_dim: int = -1,
        use_embedding: bool = False,
        use_batchnorm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout_p
        self.embed_dim = embed_dim
        self.use_embedding = use_embedding
        self.use_batchnorm = use_batchnorm

        if use_embedding:
            self.embedding = FeaturesEmbedding(in_dim, embed_dim)
            model_in_dim = self.num_features * embed_dim
        else:
            model_in_dim = in_dim

        self.model = MLPModule(
            in_dim = model_in_dim,
            out_dim = out_dim,
            hidden_dims = hidden_dims,
            dropout_p = dropout_p,
            use_batchnorm = use_batchnorm
        )

    def forward(
        self, 
        x: torch.FloatTensor | torch.LongTensor, 
        feature_idx: torch.LongTensor = None, 
        **kwargs
    ):
        if self.use_embedding:
            feature_val = x.gather(dim = 1, index=feature_idx)
            x = self.embedding(feature_idx, feature_val).flatten(1)
        else: 
            x = x.float()
        return self.model(x)
