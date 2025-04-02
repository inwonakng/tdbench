from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    ReduceLROnPlateau,
    ExponentialLR, 
    ChainedScheduler,
    StepLR,
    ConstantLR,
    CosineAnnealingLR
)
def get_scheduler(
    optimizer: Optimizer,
    sch_name: str,
) -> LRScheduler | ReduceLROnPlateau | None:
    sch = None
    if sch_name == 'ReduceLROnPlateau':
        sch = ReduceLROnPlateau(optimizer)
    elif sch_name == 'CosineAnnealingLR':
        sch = CosineAnnealingLR(optimizer, T_max=20)
    elif sch_name == 'ExponentialLR':
        sch = ExponentialLR(optimizer, gamma=0.1)
    elif sch_name == 'StepLR':
        sch = StepLR(optimizer, step_size=20)
    elif sch_name == 'ConstantLR':
        sch = ConstantLR(optimizer)
    else:
        raise NotImplementedError
    return sch

def get_schedulers(
    optimizer: Optimizer, 
    sch_names: str, 
    **sch_params
) -> list[LRScheduler | ReduceLROnPlateau]:
    schedulers = [get_scheduler(optimizer, sch_name) for sch_name in sch_names]
    return schedulers

def step_scheduler(scheduler, metric = None):
    if isinstance(scheduler, ReduceLROnPlateau): 
        scheduler.step(metric)
    else:
        scheduler.step()
