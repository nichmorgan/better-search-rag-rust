use ort::Error;

pub mod hf;

pub trait LlmService {
    type Error;

    fn default() -> Result<Self, Error>
    where
        Self: Sized;

    fn get_embeddings(&self, texts: &Vec<String>) -> Result<Vec<Vec<f32>>, Error>;
    
}
