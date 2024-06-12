import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .modules import FeaturesEmbedding, FeaturesLinear, FactorizationMachine, MLPModule
# from .mlp import MLP

class DeepFM(BaseModel):
    name = 'DeepFM'
    task = 'predict'
    def __init__(
        self, 
        in_dim: int,
        out_dim: int, 
        num_features: int, 
        embed_dim: int,
        hidden_dims:list[int],
        dropout_p: float = 0.0,
        use_val: bool = True,
        device: torch.device = 'cpu',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.use_val = use_val
        self.linear = FeaturesLinear(in_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(in_dim, embed_dim)
        self.mlp = MLPModule(
            in_dim = num_features * embed_dim,
            out_dim = out_dim,
            hidden_dims = hidden_dims,
            dropout_p=dropout_p,
        )

    def forward(
        self, x:torch.FloatTensor, 
        feature_idx:torch.LongTensor, 
        **kwargs
    ) -> torch.FloatTensor:
        """
        :param feature_idx: Long tensor of size ``(batch_size, num_fields)``
        """

        if self.use_val:
            feature_val = x.gather(dim = 1,index=feature_idx)
        else:
            feature_val = torch.ones_like(feature_idx)
        embed = self.embedding(feature_idx, feature_val)
        combined = self.linear(feature_idx, feature_val) + self.fm(embed) + self.mlp(embed.flatten(1))
        return F.softmax(combined.squeeze(1), dim =1)