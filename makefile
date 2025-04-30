N = 6
OUT_FILE = ./target/release/better-search-rag-rust

run-ollama:
	OLLAMA_MODELS=".volumes/ollama/"" ollama serve & >> .logs/ollama

setup:
	cargo build --release && \
	chmod +x ${OUT_FILE} && \
	$(MAKE) run-ollama

run:
	mpiexec -n $N target/release/better-search-rag-rust 

run-slurm:
	srun --mem 300GB --ntasks $N --cores 32 && \
	module load openmpi llvm ollama && \
	$(MAKE) setup && \
	${OUT_FILE}