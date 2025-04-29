# General purpose Makefile with Ollama SLURM support

LOG_DIR = .logs
JOBS_DIR = ./scripts

# Default target
.PHONY: all
all: ollama-start

# ---- Ollama SLURM targets ----
LAUNCH_SCRIPT = $(JOBS_DIR)/launch.sh

CARGO_HOME=/scratch/mcn97/.cargo
RUSTUP_HOME=/scratch/mcn97/.rustup
CARGO=srun --mem=1GB --cores=16 $(CARGO_HOME)/bin/cargo

.PHONY: setup
setup:
	mkdir -p $(LOG_DIR)
	chmod +x $(LAUNCH_SCRIPT) $(OLLAMA_RUN_SCRIPT) $(OLLAMA_CHECK_SCRIPT)

.PHONY: stop start
stop:
	@echo "Finding and cancelling bsrr jobs..."
	@JOB_IDS=$$(squeue -u $$USER -o "%.18i %.40j" | grep "bsrr" | awk '{print $$1}'); \
	if [ -n "$$JOB_IDS" ]; then \
		echo "Found jobs: $$JOB_IDS"; \
		for JOB_ID in $$JOB_IDS; do \
			echo "Cancelling job $$JOB_ID"; \
			scancel $$JOB_ID; \
		done; \
		echo "All bsrr jobs cancelled."; \
	else \
		echo "No running bsrr jobs found."; \
	fi

start: setup stop
	sbatch $(LAUNCH_SCRIPT)

build:
	$(CARGO) clean
	module load gcc openmpi && $(CARGO) build --release

# Clean up
.PHONY: clean
clean:
	rm -rf $(LOG_DIR)/