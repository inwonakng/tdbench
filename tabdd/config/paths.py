import os
from pathlib import Path

CUR_DIR = Path(__file__).parent
ROOT_DIR = CUR_DIR.parent.parent

LOGGER_DIR = ROOT_DIR / "logs"
LOGGER_DIR.mkdir(exist_ok=True)

RAY_TMP_DIR = os.getenv("RAY_TMP_DIR")
RAY_RESULT_DIR = os.getenv("RAY_RESULT_DIR")

DATA_REPO_DIR = Path(os.getenv('DATA_REPO_DIR'))
DATA_DIR = (DATA_REPO_DIR / 'data').resolve()

DASHBOARD_ASSETS_DIR = (ROOT_DIR / 'dashboard_assets').resolve()
# PLOTS_DIR = 
PLOTS_CACHE_DIR = (CUR_DIR / '../results/plot/.cache').resolve()
RESULTS_CACHE_DIR = (CUR_DIR / '../results/.cache').resolve()

RUN_CONFIG_DIR = str((CUR_DIR / "../../config").resolve())
FIGURES_DIR = ROOT_DIR / "figures"
