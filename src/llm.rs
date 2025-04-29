// src/ollama.rs

use anyhow::{Context, Result};
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::{IntoUrlSealed, Ollama};
use std::sync::Arc;
use tokio::sync::Mutex;

// Client for Ollama API interactions
#[derive(Clone, Default)]
pub struct LlmService {
    client: Ollama,
    embedding_model: String,
}

// Shared state for the client
pub type SharedLlmService = Arc<Mutex<LlmService>>;

impl LlmService {
    pub fn new(url: String, embedding_model: String) -> Self {
        Self {
            client: Ollama::from_url(url.into_url().expect("Not a valid url")),
            embedding_model,
        }
    }

    pub fn default() -> Self {
        Self::new(
            String::from("http://localhost:11434"),
            String::from("nomic-embed-text"),
        )
    }

    pub fn shared(&self) -> SharedLlmService {
        Arc::new(Mutex::new(self.clone()))
    }

    // Get embedding for text
    pub async fn get_embeddings(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let request = GenerateEmbeddingsRequest::new(self.embedding_model.clone(), texts.into());
        let response = self
            .client
            .generate_embeddings(request)
            .await
            .context("Failed to generate embeddings")?;

        Ok(response.embeddings)
    }
}
