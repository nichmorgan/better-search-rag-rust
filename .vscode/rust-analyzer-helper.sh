#!/bin/bash

# Source MPI setup
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/setup_mpi.sh"

# Run the actual command
"$@"