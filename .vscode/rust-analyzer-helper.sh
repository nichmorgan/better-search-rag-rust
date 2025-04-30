#!/bin/bash

# Set explicit paths to cargo and rustc
export CARGO_HOME="/scratch/mcn97/.cargo"
export RUSTUP_HOME="/scratch/mcn97/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"
PROJECT_ROOT="/scratch/mcn97/projects/better-search-rag-rust"

# Then source your MPI setup

if [[ -f "$PROJECT_ROOT/scripts/setup_mpi.sh" ]]; then
    source "$PROJECT_ROOT/scripts/setup_mpi.sh"
else
    echo "Error: MPI setup script not found"
    exit 1
fi

# Debug information
echo "Using cargo at: $(which cargo)"
echo "Using rustc at: $(which rustc)"

# Run the actual command
"$@"