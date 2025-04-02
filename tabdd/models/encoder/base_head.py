from typing import Literal

from torch import nn

from ..utils import (
    get_criterion,
)


class BaseHead(nn.Module):
    def __init__(
        self,
        name: str,
        task: Literal["reconstruct", "classify"],
        criterion: str,
        metrics: list[str],
        module: nn.Module,
    ) -> None:
        super().__init__()
        self.name = name
        self.module = module
        self.criterion = get_criterion(criterion)
        self.metrics = metrics

    def forward(self, x, **kwargs):
        return self.module(x,**kwargs)

    def eval():
        ...
