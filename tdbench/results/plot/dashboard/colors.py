from plotly import colors
from .config import (
    DATA_MODES,
    AUTOENCODERS,
    MULTIHEAD_AUTOENCODERS,
)
from .utils import parse_color, add_alpha

unique_colors = (
    [parse_color(c) for c in colors.qualitative.Plotly[:7]]
    + [parse_color(c) for c in colors.qualitative.Prism[:5]]
    + [parse_color(c) for c in colors.qualitative.Bold[:8]]
    + [parse_color(c) for c in colors.qualitative.Vivid[:5]]
    + [parse_color(c) for c in colors.qualitative.Set2[:5]]
    + [parse_color(colors.qualitative.Dark24[i*2]) for i in range(len(colors.qualitative.Dark24)//2)]
    + [parse_color(c) for c in colors.qualitative.Set1[:5]]
    + [parse_color(c) for c in colors.qualitative.T10]
    + [parse_color(c) for c in colors.qualitative.Antique]
)

distill_results_colormap = {
    dm: {
        'main': unique_colors[i%len(unique_colors)],
        'fill': add_alpha(unique_colors[i%len(unique_colors)], 0.2)
    }
    for i,dm in enumerate(DATA_MODES)
}

encoder_stats_colormap = {
    **{
        en: {
            'score': parse_color(colors.qualitative.Set2[i]),
            'encoder': parse_color(colors.qualitative.Plotly[i*2+2]),
            'decoder': parse_color(colors.qualitative.Plotly[i*2+3]),
        }
        for i,en in enumerate(AUTOENCODERS)    
    },
    **{
        en: {
            'recon_score': parse_color(colors.qualitative.Set2[i]),
            'predict_score': parse_color(colors.qualitative.Set2[i+2]),
            'encoder': parse_color(colors.qualitative.Pastel[i*3]),
            'decoder': parse_color(colors.qualitative.Pastel[i*3+1]),
            'classifier': parse_color(colors.qualitative.Pastel[i*3+2]),
        }
        for i,en in enumerate(MULTIHEAD_AUTOENCODERS)    
    }
}

dataset_stats_colormap = {
    'labels': [parse_color(c) for c in colors.qualitative.Plotly],
    'features': {
        'continuous': parse_color(colors.qualitative.Plotly[-1]),
        'categorical': parse_color(colors.qualitative.Plotly[-2]),
    }
}
