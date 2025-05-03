pub mod arrow;
pub mod polars;

use ndarray::{Array1, Array2};

/// Trait defining operations for storing and retrieving vectors
pub trait VectorStorage {
    /// Error type returned by operations
    type Error: std::error::Error;

    /// Creates a new storage file optimized for parallel access
    fn create_or_load_storage(&self, reset: bool) -> Result<(), Self::Error>;

    /// Writes a slice of vectors at a specific position
    fn write_slice(&self, vectors: &Array2<f32>, start_idx: usize) -> Result<(), Self::Error>;

    /// Reads a slice of vectors
    fn read_slice(&self, start_idx: usize, count: usize) -> Result<Array2<f32>, Self::Error>;

    /// Appends a single vector, returns the index
    fn append_vector(&self, vector: &Array1<f32>) -> Result<(), Self::Error>;

    fn append_vectors(&self, new_vectors: &Array2<f32>) -> Result<(), Self::Error>;

    /// Gets a specific vector by index
    fn get_vector(&self, index: usize) -> Result<Array1<f32>, Self::Error>;

    fn get_count(&self) -> Result<usize, Self::Error>;
}

// Calculate consine similarity between two vectors
// Returns a value between -1 and 1, where 1 indicates identical vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    // Handle zero vectors to avoid NaN
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

// Calculate cosine distance between rwo vectors
// Returns a value between 0 and 2, where 0 indicates identical vectors
pub fn cosine_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    1.0 - cosine_similarity(a, b)
}
