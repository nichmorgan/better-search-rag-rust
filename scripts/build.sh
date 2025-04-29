#!/bin/bash
# scripts/build.sh

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Source the MPI setup
source ./scripts/setup_mpi.sh

# Build with cargo
echo "Building with configured MPI environment..."
cargo build --release

if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed. Debugging info:"
    echo "Current compilers in PATH:"
    ls -la $VENV_DIR/bin/*-linux-gnu-*
    echo "MPI compiler wrapper settings:"
    mpicc --showme:compile
    mpicc --showme:link
fi