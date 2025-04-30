N = 6

run-dev:
	cargo build && \
	mpiexec -n $N target/debug/better-search-rag-rust 

run-prod:
	cargo build && \
	mpiexec -n $N target/debug/better-search-rag-rust 