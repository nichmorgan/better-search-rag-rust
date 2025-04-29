#!/bin/bash

# Set Ollama model directory
export OLLAMA_MODELS=".volumes/ollama/models"

# Set Ollama Host
export OLLAMA_HOST=0.0.0.0

# # Force CPU mode since the GPU has compatibility issues (compute capability 3.7)
# export CUDA_VISIBLE_DEVICES="-1"

# Verbose
export OLLAMA_DEBUG=0
# The maximum number of models that can be loaded concurrently
export OLLAMA_MAX_LOADED_MODELS=2
# The maximum number of parallel requests each model will process
export OLLAMA_NUM_PARALLEL=4 
# The maximum number of requests that can be queued
export OLLAMA_MAX_QUEUE=512
# Optional: Enable Flash Attention to reduce memory usage
export OLLAMA_FLASH_ATTENTION=1

# Create the models directory if it doesn't exist
mkdir -p $OLLAMA_MODELS

# Load ollama module
module purge
module load cuda
module load ollama

# Display the configuration
echo "Ollama Configuration:"
echo "--------------------"
echo "Models directory: $OLLAMA_MODELS"
echo "Max loaded models: $OLLAMA_MAX_LOADED_MODELS"
echo "Max parallel requests per model: $OLLAMA_NUM_PARALLEL"
echo "Max queued requests: $OLLAMA_MAX_QUEUE"
echo "Flash Attention: Enabled"
echo "--------------------"

# Run the Ollama server
ollama serve &

# Function to check if Ollama API is ready
function is_ollama_ready() {
    ollama -v >/dev/null
    return $?
}

# Wait until Ollama is ready
echo "⏳ Waiting for Ollama server to become available..."

until is_ollama_ready; do
    echo "🔄 Ollama not ready yet. Retrying in 5 seconds..."
    sleep 5
done

echo "✅ Ollama is now running!"

ollama pull nomic-embed-text