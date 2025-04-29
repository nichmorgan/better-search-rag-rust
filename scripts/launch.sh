#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cores=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=bsrr
#SBATCH --output=.logs/%j_bsrr.out
#SBATCH --error=.logs/%j_bsrr.err

SCRIPT_DIR="./scripts"
OLLAMA_RUN="$SCRIPT_DIR/ollama_run.sh"
APP_RUN="$SCRIPT_DIR/app_run.sh"

chmod +x $OLLAMA_RUN $APP_RUN

$OLLAMA_RUN
$APP_RUN

