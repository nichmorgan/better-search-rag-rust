#!/bin/bash

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Source MPI configuration
source ./scripts/setup_mpi.sh

echo "Running app with $(mpirun --version | head -n 1)"
mpirun -np 2 ./target/release/better-search-rag-rust