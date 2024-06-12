import torch
from torch.nn import functional as F
import numpy as np

def cross_entropy(
    input: torch.FloatTensor, 
    target: torch.LongTensor, 
    **kwargs
) -> torch.Tensor:
    return F.cross_entropy(input, target)

def tabular_recon(
    input: torch.FloatTensor, 
    target: torch.LongTensor, 
    feature_mask: torch.LongTensor,
    feature_categ_mask: torch.BoolTensor,
    balanced: bool = False,
    **kwargs
):
    n_features = feature_mask.max() + 1
    losses = []
    for i,is_cat in zip(range(n_features), feature_categ_mask):
        mask = (feature_mask == i)
        if is_cat:
            loss = F.cross_entropy(input[:, mask], target[:, mask].argmax(1)) 
            if balanced:
                loss /= np.log(mask.sum())
                # print(mask.sum())
            losses.append(loss)
        else:
            losses.append(F.mse_loss(input[:, mask], target[:,mask].float()))
    return sum(losses) / len(losses)

def balanced_tabular_recon(
    input: torch.FloatTensor, 
    target: torch.LongTensor, 
    feature_mask: torch.LongTensor,
    feature_categ_mask: torch.BoolTensor,
):
    return tabular_recon(
        input, 
        target, 
        feature_mask, 
        feature_categ_mask, 
        balanced=True
    )

def simple_grad_match(
    input: torch.FloatTensor,
    target: torch.FloatTensor,
    feature_mask: torch.LongTensor,
    feature_categ_mask: torch.LongTensor,
    tester: torch.nn.Module,
    y_true: torch.LongTensor,
    **kwargs
):
    '''
    input: `(batch_size, num_rows, num_features)`
    target: `(batch_size, num_rows, num_features)`
    feature_mask: `(num_features)`
    feature_categ_mask: `(num_original_features)`
    tester: `(batch_size, num_rows, num_features)`
    '''
    ...
    
CRITERION_MAPPING = {
    '': None,
    'cross_entropy': cross_entropy,
    'tabular_recon': tabular_recon,
    'balanced_tabular_recon': balanced_tabular_recon,
    # 'mixed_recon': mixed_recon,
}

def get_criterion(criterion_name: str) -> callable:
    return CRITERION_MAPPING[criterion_name]
