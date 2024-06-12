from torch import nn

class MLPModule(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        dropout_p: float,
        use_batchnorm: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        layer_shapes = [in_dim] + list(hidden_dims)
        self.model = nn.Sequential(
            *([
                nn.Sequential(
                    nn.Linear(_in,_out),
                    nn.BatchNorm1d(_out),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=dropout_p),
                ) if use_batchnorm else
                nn.Sequential(
                    nn.Linear(_in,_out),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=dropout_p),
                )
                for _in, _out in zip(layer_shapes[:-1], layer_shapes[1:])
            ] + [
                nn.Linear(layer_shapes[-1], out_dim)
            ])
        )

    def forward(self,x, **kwargs):
        return self.model(x)
