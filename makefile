N ?= 6
SKIP_PROCESS ?= false
OUT_FILE = ./target/release/better-search-rag-rust

run-ollama:
	OLLAMA_MODELS=".volumes/ollama/"" ollama serve

setup:
	cargo build --release && \
	chmod +x ${OUT_FILE} && \
	$(MAKE) run-ollama

run:
	SKIP_PROCESS=$(SKIP_PROCESS) mpiexec -n $(N) target/release/better-search-rag-rust 

run-slurm:
	SKIP_PROCESS=$(SKIP_PROCESS) srun --mem 300GB --ntasks $(N) --cores 32 && \
	module load openmpi llvm ollama && \
	$(MAKE) setup && \
	${OUT_FILE}
