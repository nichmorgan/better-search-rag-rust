#!/bin/bash

#SBATCH --mem=300GB
#SBATCH --cores=16
#SBATCH --ntasks=6
#SBATCH --output=.logs/%j_slurm.out
#SBATCH --error=.logs/%j_slurm.err
#SBATCH --time=00:10:00

module purge

OUT_FILE="./target/release/better-search-rag-rust"
MODEL_DIR=".volumes/models/nomic_embed_text_onnx/"
MODEL_NAME="nomic-ai/nomic-embed-text-v1.5"


# Check if force parameter is provided
if [ "$1" = "force" ]; then
    echo "Force flag detected. Removing model directory if it exists..."
    rm -rf "$MODEL_DIR"
fi

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory doesn't exist. Running export command..."

    module load anaconda3 
    conda env update -f environment.yaml -p .venv

    conda activate .venv/
    optimum-cli export onnx --model $MODEL_NAME $MODEL_DIR --trust-remote-code
    conda deactivate
else
    echo "Model directory already exists. Skipping export command."
fi

module load openmpi

cargo build --release
chmod +x $OUT_FILE

$OUT_FILE
