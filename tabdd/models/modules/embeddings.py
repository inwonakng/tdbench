import torch
from torch import nn

class FeaturesLinear(nn.Module):
    def __init__(self, in_dim: int, output_dim: int =1):
        super().__init__()
        self.fc = nn.Embedding(in_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, idx: torch.LongTensor, val:torch.FloatTensor) -> torch.FloatTensor:
        """_summary_

        Args:
            idx (torch.LongTensor): Long tensor of shape ``(batch_size, num_fields)``
            val (torch.FloatTensor): Float tensor of shape ``(batch_size, num_fields)``

        Returns:
            torch.FloatTensor: Float tensor of shape ``(batch_size, 1)``
        """
        return (self.fc(idx) * val.unsqueeze(2)).sum(dim=1) + self.bias

class FeaturesEmbedding(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(in_dim, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(
        self, 
        idx: torch.LongTensor, 
        val: torch.FloatTensor|None = None,
    ) -> torch.FloatTensor:
        """_summary_

        Args:
            idx (torch.LongTensor): Long tensor of shape ``(batch_size, in_dim)``
            val (torch.FloatTensor): Float tensor of shape ``(batch_size, in_dim)``

        Returns:
            torch.FloatTensor: Float tensor of shape ``(batch_size, in_dim, embed_dim)``
        """
        if val is None:
            return self.embedding(idx)
        else:
            return self.embedding(idx) * val.unsqueeze(2)

class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum: bool =True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.Tensor[float]): Float tensor of shape ``(batch_size, in_dim, embed_dim)``

        Returns:
            torch.FloatTensor: Float tensor of shape ``(batch_size, 1)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    
