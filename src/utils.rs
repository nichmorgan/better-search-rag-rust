#[cfg(test)]
pub mod tests {
    use rand::Rng;
    use tempfile::{tempdir, TempDir};

    use crate::{llm::DIMENSION, vectorstore::polars::PolarsVectorstore};

    pub fn get_vstore_dir() -> TempDir {
        tempdir().unwrap()
    }

    pub fn generate_mock_embeddings() -> Vec<f32> {
        let mut rng = rand::rng();
        let mut result = Vec::with_capacity(DIMENSION);

        for _ in 0..DIMENSION {
            result.push(rng.random_range(-1.0..1.0));
        }

        result
    }

    pub fn generate_many_mock_embeddings(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|_| generate_mock_embeddings()).collect()
    }

    // Helper function to create a test vectorstore with sample data
    pub fn sample_vstore(vstore: &mut PolarsVectorstore, n: usize) -> Vec<Vec<f32>> {
        let rand_2d: Vec<Vec<f32>> = generate_many_mock_embeddings(n);

        vstore.append_many(&rand_2d).unwrap();
        rand_2d
    }
}