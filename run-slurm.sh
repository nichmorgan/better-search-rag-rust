#!/bin/bash

#SBATCH --mem=300GB
#SBATCH --cores=12
#SBATCH --ntasks=6
#SBATCH --output=.logs/%j_slurm.out
#SBATCH --error=.logs/%j_slurm.err
#SBATCH --time=01:00:00

module load openmpi llvm ollama

OUT_FILE="./target/release/better-search-rag-rust"
export OLLAMA_MODELS=".volumes/ollama/"

cargo build --release
chmod +x $OUT_FILE
$OUT_FILE


