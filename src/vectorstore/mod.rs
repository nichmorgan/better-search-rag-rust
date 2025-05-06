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

