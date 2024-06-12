def pretty_encoder_name(encoder_name):
    if encoder_name.lower() == 'original':
        return 'Original'
    elif encoder_name.lower() == 'mlpautoencoder':
        return 'MLP'
    elif encoder_name.lower() == 'gnnautoencoder':
        return 'GNN'
    elif encoder_name.lower() == 'multiheadmlpautoencoder':
        return 'MLP-MultiHead'
    elif encoder_name.lower() == 'multiheadgnnautoencoder':
        return 'GNN-MultiHead'
    else:
        raise NotImplementedError(f'Encoder name {encoder_name} is unknown!')