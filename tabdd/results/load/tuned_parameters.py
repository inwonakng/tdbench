import pandas as pd

from tabdd.config.pipeline import get_pipeline_configs
from tabdd.tune.classifier import TuneClassifierRun

def load_tuned_parameters(
    dataset_name: str,
    classifier_name: str,
    min_distill_size: int = 10,
    max_distill_size: int = 100,
) -> tuple[pd.DataFrame | None, bool]:

    configs = get_pipeline_configs(dataset_name, classifier_name, 'all')
    raw_params = []
    all_done = True
    for config in configs:
        config.tune_hyperopt = True
        run = TuneClassifierRun(config)
        if not run.is_complete: 
            all_done = False
            continue
        raw_params += [
            {
                'Data Mode': 'Original',
                'N': min_distill_size,
                'Param': param,
                'Value': value,
                'Distill Group': 'Baseline',
            }
            for param, value in run.load_best_params().items()
        ]
        
    if len(raw_params) == 0: return None, False

    raw_params = pd.DataFrame(raw_params)
    distilled = raw_params[raw_params['Data Mode'].str.contains('KMeans|KIP|Sample|Agglo')]
    baselines = raw_params[~raw_params['Data Mode'].str.contains('KMeans|KIP|Sample|Agglo')]
    baselines =  pd.concat([baselines.assign(**{'N': n}) for n in raw_params['N'].unique()])

    params = pd.concat([
        baselines,
        distilled
    ])

    params = params[
        (params['N'] >= min_distill_size) &
        (params['N'] <= max_distill_size)
    ]

    return params, all_done
