from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import Sequential,GCNConv, GATConv, SAGEConv

def build_edges(
    x: torch.LongTensor,
    col_to_row: bool = True,
    row_to_col: bool = False,
) -> torch.LongTensor:
    """Builds the edges between samples and one-hot features.
    Each edge is directed, but the returned edge tensor contains edges for both directions between two nodes.

    Args:
        x (torch.LongTensor): A one-hot encoded matric where each row represents a sample.

    Returns:
        torch.LongTensor: A 2d tensor in `(2,num_edges)` that represents src and dst. 
    """

    if not col_to_row and not row_to_col:
        raise Exception('There will be no edges! Are you sure?')

    n_row, n_col = x.shape
    # num_features = train_data.feature_idx.size(1)
    row_idx, col_idx = x.nonzero().transpose(0, 1)
    # offset row_idx by number of features so node ids don't overlap
    row_idx += n_col

    if col_to_row and row_to_col:
        edge_index = torch.stack([
            torch.cat([row_idx, col_idx]), 
            torch.cat([col_idx, row_idx])
        ])
    elif col_to_row:
        edge_index = torch.stack([
            col_idx, 
            row_idx,
        ])
    elif row_to_col:
        edge_index = torch.stack([
            row_idx,
            col_idx, 
        ])
    return edge_index

def drop_edge(
    edge_index: torch.LongTensor,
    dropout_prob: float,
):
    drop_mask = (torch.FloatTensor(edge_index.size(1), 1).uniform_() > dropout_prob).view(-1)
    edge_index_new = edge_index[:, drop_mask]
    return edge_index_new

GNN_LAYER_MAPPING = {
    'GCN': GCNConv,
    'GAT': GATConv,
    'SAGE': SAGEConv,
}

class GNNEncoder(nn.Module):
    def __init__(
        self,
        graph_layer: str,
        graph_aggr: str,
        in_dim: int,
        emb_dim: int,
        n_layer: int,
        drop_edge_p: float,
        edge_direction: Literal['unidirectional', 'bidirectional', 'multipass_*'],
    ):
        super().__init__()

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.n_layer = n_layer
        self.drop_edge_p = drop_edge_p
        self.edge_direction = edge_direction
        self.graph_layer = graph_layer
        self.embedding = nn.Embedding(in_dim, emb_dim)

        self.model = Sequential('x, edge_index', [
            (GNN_LAYER_MAPPING[graph_layer](
                in_channels = emb_dim,
                out_channels = emb_dim,
                aggr = graph_aggr,
            ), 'x, edge_index -> x')
            for _ in range(n_layer)
        ])

    def forward(
        self,
        x: torch.LongTensor,
        # edge_index, 
        **kwargs
    ):
        n_row, n_feature = x.shape
        if n_feature != self.in_dim: 
            raise Exception('dimensions do not match!')
        # build edges from data
        if self.edge_direction == 'unidirectional':
            edge_index = build_edges(x, col_to_row=True, row_to_col=False)
        elif self.edge_direction == 'bidirectional':
            edge_index = build_edges(x, col_to_row=True, row_to_col=True)
        elif 'multipass' in self.edge_direction:
            edge_index_col_row = build_edges(x, col_to_row=True, row_to_col=False)
            edge_index_row_col = build_edges(x, col_to_row=False, row_to_col=True)

        all_feature_idxs = torch.arange(self.in_dim).to(x)
        feature_embs = self.embedding(all_feature_idxs)
        node_emb = torch.vstack([
            feature_embs,
            torch.zeros(n_row, self.emb_dim).to(x)
        ])

        if self.edge_direction in ['unidirectional', 'bidirectional']:
            edge_index = drop_edge(edge_index, self.drop_edge_p).to(x)
            out = self.model(node_emb, edge_index)

        elif 'multipass' in self.edge_direction:
            edge_index_col_row = drop_edge(edge_index_col_row, self.drop_edge_p).to(x)
            edge_index_row_col = drop_edge(edge_index_row_col, self.drop_edge_p).to(x)

            num_pass = int(self.edge_direction.split('_')[1])

            for _ in range(num_pass):
                # first propagate from col to row
                node_emb = self.model(node_emb, edge_index_col_row)
                # then from the rows back into col
                node_emb = self.model(node_emb, edge_index_row_col)
            # finally, propagate `updated` col embs into row again.
            out = self.model(node_emb, edge_index_col_row)
        else:
            raise NotImplementedError

        return out[n_feature:]
