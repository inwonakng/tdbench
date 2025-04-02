from sklearn.preprocessing import StandardScaler, MaxAbsScaler
    
def build_scaler(scale_mode) -> StandardScaler | MaxAbsScaler:
    if scale_mode == 'standard':
        return StandardScaler()
    elif scale_mode == 'maxabs':
        return MaxAbsScaler()
    else:
        raise NotImplementedError
