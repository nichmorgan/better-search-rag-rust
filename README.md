# better-search-rag-rust

A parallel implementation of top-k retrieval for code embeddings using MPI, demonstrating significant speedup while maintaining high accuracy. This project implements a custom vector storage solution and parallel distance calculation algorithms for efficient similarity-based code search.

## Overview

This system converts source code snippets into vector embeddings, stores them using a custom persistent storage solution in Rust, and implements parallel distance calculations with MPI to retrieve the most similar snippets. The implementation avoids existing vector databases to enable direct computation of distances between embeddings using a block distribution approach.

## System Requirements

- Linux-based operating system (tested on Ubuntu 24.04)
- Rust (2024 edition)
- OpenMPI 4.1+
- Python 3.12+ (for model export)
- At least 8GB RAM per process (depending on model size)

## Dependencies Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Install OpenMPI

```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev
```

### 3. Install Python and Conda

```bash
sudo apt install -y python3-pip
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
```

### 4. Create directories

```bash
mkdir -p .volumes/models/nomic_embed_text_onnx
mkdir -p .volumes/vstore
mkdir -p .logs
mkdir -p .repos
```

### 5. Install Python dependencies

```bash
conda env create -f environment.yaml -p .venv
conda activate .venv/
```

### 6. Clone JabRef Repository

The project uses the JabRef codebase for testing. Clone it with:

```bash
git clone https://github.com/JabRef/jabref.git .repos/jabref
```

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/nichmorgan/better-search-rag-rust.git
cd better-search-rag-rust
```

2. Build the project:
```bash
# Make sure you are NOT in a conda environment
conda deactivate  # Deactivate conda if it's active

cargo build --release
chmod +x ./target/release/better-search-rag-rust
```

3. Export the embedding model to ONNX format:
```bash
conda activate .venv/
optimum-cli export onnx --model nomic-ai/nomic-embed-text-v1.5 .volumes/models/nomic_embed_text_onnx --trust-remote-code
conda deactivate  # Important: deactivate conda after exporting the model
```

> **IMPORTANT**: Make sure to deactivate conda before building and running the project, as conda uses its own version of MPI by default which may conflict with the system MPI.

## Running the Project

The project can be run using MPI with varying numbers of processes. There are several ways to run it:

### Option 1: Using the Makefile

The simplest way to run the project is using the provided Makefile:

```bash
# Make sure conda is deactivated
conda deactivate  # If conda is still active

# Run with default number of processes (N=6)
make run

# Run with a specific number of processes
N=4 make run

# Skip the processing phase
SKIP_PROCESS=true N=8 make run
```

### Option 2: Using MPI directly

```bash
# Make sure conda is deactivated
conda deactivate  # If conda is still active

# Run with 4 processes
mpiexec -n 4 ./target/release/better-search-rag-rust
```

### Option 3: Using the SLURM script

If you're using a SLURM-based cluster like Monsoon:

```bash
# Run with 6 processes
N=6 ./run-slurm.sh
```

### Environment Variables

- `SKIP_PROCESS`: Set to "true" to skip the embedding generation phase (useful for running just the similarity search on pre-generated embeddings)
- `N`: Set the number of MPI processes when using the run script

Example:
```bash
SKIP_PROCESS=true N=8 ./run-slurm.sh
```

## Project Structure

- `src/`: Contains all Rust source code
  - `llm/`: Embedding model implementation
  - `mpi_helpers/`: MPI parallelization utilities
  - `vectorstore/`: Custom vector storage
  - `metrics.rs`: Accuracy measurement
  - `main.rs`: Entry point
- `.volumes/`: Storage for models and vector data
- `.logs/`: Output logs
- `.repos/jabref/`: JabRef source code used for testing

## Benchmark Setup

For performance evaluation, the system was tested on:
- AWS EC2 c6i.24xlarge (96 vCPU, 192 GiB RAM)
- JabRef codebase with ~2,305 Java files
- Process counts from 1 to 24

## Performance Considerations

- Each MPI process loads a separate instance of the embedding model, which requires significant memory
- Running with too many processes may cause out-of-memory errors due to total machine memory limitations (OOM was observed at N=32 on the AWS EC2 c6i.24xlarge with 192GB RAM)
- The maximum number of processes is limited by your machine's available memory. As a rule of thumb, ensure you have at least 8GB RAM per process
- For optimal performance, adjust the chunk size in `main.rs` based on your hardware capabilities
- The project uses block distribution for load balancing, which works best when files have similar sizes

## Troubleshooting

- If you encounter memory issues, try reducing the number of processes or increasing the available memory
- Make sure the JabRef repository is properly cloned to `.repos/jabref`
- Check the `.logs` directory for error messages if the program fails
- If the model fails to load, verify that the ONNX export was successful
- If you encounter MPI errors or issues with the OpenMPI version, ensure that conda is deactivated when building and running the project
- If you see errors related to "mpicc not found" or MPI incompatibilities, verify that you're using the system MPI installation and not conda's MPI

## Citation

If you use this code in your research, please cite:

```
@inproceedings{nicholson2025accelerating,
  title={Accelerating Top-k RAG with Parallel Embedding Retrieval},
  author={Nicholson, Morgan C.},
  booktitle={CS552 High Performance Computing Final Project},
  year={2025},
  organization={Northern Arizona University}
}
```

## License

MIT