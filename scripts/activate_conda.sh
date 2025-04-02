#!/bin/bash

# conda env should be specified in the tabdd/.env
source .env
# run hook and activate
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate $CONDA_ENV
