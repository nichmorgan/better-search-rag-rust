// pub mod arrow;
pub mod polars;

type Serie = Vec<f32>;
type DataFrame = Vec<Serie>;

/// Trait defining operations for storing and retrieving vectors
pub trait VectorStorage {
    /// Error type returned by operations
    type Error: std::error::Error;

    /// Creates a new storage file optimized for parallel access
    fn create_or_load_storage(&self, reset: bool) -> Result<(), Self::Error>;

    /// Writes a slice of vectors at a specific position
    fn write_slice(&self, vectors: &DataFrame, start_idx: usize) -> Result<(), Self::Error>;

    /// Reads a slice of vectors
    fn read_slice(&self, start_idx: usize, count: usize) -> Result<DataFrame, Self::Error>;

    /// Appends a single vector, returns the index
    fn append_vector(&self, vector: &Serie) -> Result<(), Self::Error>;

    fn append_vectors(&self, new_vectors: &DataFrame) -> Result<(), Self::Error>;

    /// Gets a specific vector by index
    fn get_vector(&self, index: usize) -> Result<Serie, Self::Error>;

    fn get_count(&self) -> Result<usize, Self::Error>;
}

pub fn cosine_distance(a: &Serie, b: &Serie) -> f32 {
    // Check if vectors are of equal length
    if a.len() != b.len() || a.is_empty() {
        return 1.0; // Maximum distance for invalid inputs
    }

    // Calculate dot product
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(a_i, b_i)| a_i * b_i).sum();

    // Calculate magnitude of vector a
    let magnitude_a: f32 = a.iter().map(|a_i| a_i * a_i).sum::<f32>().sqrt();

    // Calculate magnitude of vector b
    let magnitude_b: f32 = b.iter().map(|b_i| b_i * b_i).sum::<f32>().sqrt();

    // Handle zero magnitude edge case
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 1.0; // Maximum distance if either vector has zero magnitude
    }

    // Cosine distance = 1 - cosine similarity
    1.0 - (dot_product / (magnitude_a * magnitude_b))
}
