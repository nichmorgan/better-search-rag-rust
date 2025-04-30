#!/bin/bash

# Get project root
PROJECT_ROOT="/scratch/mcn97/projects/better-search-rag-rust"
cd "$PROJECT_ROOT"

# Source MPI configuration
source $PROJECT_ROOT/scripts/setup_mpi.sh

echo "Running app with $(mpirun --version | head -n 1)"
mpirun -np 2 $PROJECT_ROOT/target/release/better-search-rag-rust