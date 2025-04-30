pub mod arrow;

use ndarray::{Array1, Array2};
use std::path::Path;

/// Trait defining operations for storing and retrieving vectors
pub trait VectorStorage {
    /// Error type returned by operations
    type Error: std::error::Error;

    /// Creates a new storage file optimized for parallel access
    fn create_storage<P: AsRef<Path>>(
        path: P,
        dimension: usize,
        chunk_size: usize,
        reset: bool,
    ) -> Result<(), Self::Error>;

    /// Writes a slice of vectors at a specific position
    fn write_slice<P: AsRef<Path>>(
        path: P,
        vectors: &Array2<f32>,
        start_idx: usize,
    ) -> Result<(), Self::Error>;

    /// Reads a slice of vectors
    fn read_slice<P: AsRef<Path>>(
        path: P,
        start_idx: usize,
        count: usize,
    ) -> Result<Array2<f32>, Self::Error>;

    /// Appends a single vector, returns the index
    fn append_vector<P: AsRef<Path>>(path: P, vector: &Array1<f32>) -> Result<usize, Self::Error>;

    /// Gets a specific vector by index
    fn get_vector<P: AsRef<Path>>(path: P, index: usize) -> Result<Array1<f32>, Self::Error>;
}
