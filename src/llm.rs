// src/ollama.rs

use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::{IntoUrlSealed, Ollama};

pub const DIMENSION: usize = 512;
pub const MODEL: &str = "nomic-embed-text";

// Client for Ollama API interactions
#[derive(Default)]
pub struct LlmService {
    client: Ollama,
    embedding_model: String,
}

// Shared state for the client
impl LlmService {
    pub fn new(url: String, embedding_model: String) -> Self {
        Self {
            client: Ollama::from_url(url.into_url().expect("Not a valid url")),
            embedding_model,
        }
    }

    pub fn default() -> Self {
        Self::new(String::from("http://localhost:11434"), String::from(MODEL))
    }

    pub async fn check_models(&self) {
        if self
            .client
            .show_model_info(self.embedding_model.clone())
            .await
            .is_err()
        {
            println!("Model {} not found, downloading...", &self.embedding_model);
            self.client
                .pull_model(self.embedding_model.clone(), false)
                .await
                .expect("Fail to pull model");
        }
        println!("Models all set");
    }

    pub async fn get_embeddings(
        &self,
        texts: &Vec<String>,
    ) -> Result<Vec<Vec<f32>>, std::fmt::Error> {
        let request =
            GenerateEmbeddingsRequest::new(self.embedding_model.clone(), texts.to_vec().into());
        let response = self
            .client
            .generate_embeddings(request)
            .await
            .expect("Failed to generate embeddings");

        Ok(response.embeddings)
    }
}
