#!/bin/bash

# Source MPI setup with error checking
MPI_SETUP="./scripts/setup_mpi.sh"
if [[ -f "$MPI_SETUP" ]]; then
    source "$MPI_SETUP"
else
    echo "Error: MPI setup script not found at $MPI_SETUP"
    exit 1
fi

# Run the actual command
"$@"