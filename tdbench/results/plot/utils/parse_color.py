def parse_color(color): 
    if 'rgb' in color: return color
    else: return f'rgb{tuple(int(color[i+1:i+3], 16) for i in (0, 2, 4))}'

