from torch.optim import SGD, Adam, Optimizer

def get_optimizer(
    model_params:dict, 
    opt_name:str, 
    **opt_params
) -> Optimizer | None:
    opt = None
    if opt_name == 'Adam':
        opt = Adam(model_params, **opt_params)
    elif opt_name == 'SGD':
        opt = SGD(model_params, **opt_params)
    else:
        raise NotImplementedError
    return opt
