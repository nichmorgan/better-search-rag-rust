#!/bin/bash

# Get absolute path to the .venv directory
PROJECT_ROOT="/scratch/mcn97/projects/better-search-rag-rust"
VENV_DIR="$PROJECT_ROOT/.venv"
echo "Using OpenMPI from: $VENV_DIR"

# Add Conda bin to PATH first
export PATH="$VENV_DIR/bin:$PATH"

# Set compiler environment variables - using the conda-linux-gnu compiler that exists
export CC="$VENV_DIR/bin/x86_64-conda-linux-gnu-cc"
export CXX="$VENV_DIR/bin/x86_64-conda-linux-gnu-c++"

# Set OpenMPI environment variables
export OMPI_CC="$CC"
export OMPI_CXX="$CXX"

# Ensure mpicc and mpicxx use the right compilers
export OMPI_MPICC="$CC"
export OMPI_MPICXX="$CXX"

# Set environment variables for building
export RUSTFLAGS="-L $VENV_DIR/lib -C link-arg=-Wl,-rpath,$VENV_DIR/lib"
export LIBRARY_PATH="$VENV_DIR/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_DIR/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$VENV_DIR/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$VENV_DIR/include:$CPLUS_INCLUDE_PATH"
export PKG_CONFIG_PATH="$VENV_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"

# MPI-specific environment variables
export MPI_HOME="$VENV_DIR"
export OMPI_DIR="$VENV_DIR"

# Verify MPI and compiler configuration
echo "===== MPI Configuration ====="
echo "CC: $CC"
echo "CXX: $CXX"
echo "OMPI_CC: $OMPI_CC"
echo "MPI_HOME: $MPI_HOME"
which mpicc
mpicc --version
echo "============================="