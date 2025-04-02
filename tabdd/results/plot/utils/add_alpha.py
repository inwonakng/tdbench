def add_alpha(rgb, alpha):
    return rgb.replace('rgb(','rgba(').replace(')',f', {alpha})')