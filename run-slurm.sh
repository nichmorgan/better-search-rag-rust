#!/bin/bash

#SBATCH --mem=300GB
#SBATCH --cores=16
#SBATCH --ntasks=6
#SBATCH --output=.logs/%j_slurm.out
#SBATCH --error=.logs/%j_slurm.err
#SBATCH --time=00:10:00


OUT_FILE="./target/release/better-search-rag-rust"
export OLLAMA_MODELS=".volumes/ollama/"

module load openmpi llvm ollama

cargo build --release
chmod +x $OUT_FILE

ollama serve &
$OUT_FILE


