#!/bin/bash

module load openmpi gcc

echo "Running app"
cargo build --release


