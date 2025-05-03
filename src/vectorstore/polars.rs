// src/vectorstore/polars.rs

use ndarray::{Array1, Array2};
use polars::prelude::*;
use tokio::io::AsyncWriteExt;
use std::{error::Error, fmt, fs::create_dir_all, path::Path};

use super::VectorStorage;

#[derive(Debug)]
pub enum PolarsStorageError {
    IoError(std::io::Error),
    PolarsError(PolarsError),
    NotFound,
}

impl fmt::Display for PolarsStorageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PolarsStorageError::IoError(e) => write!(f, "I/O error: {}", e),
            PolarsStorageError::PolarsError(e) => write!(f, "Polars error: {}", e),
            PolarsStorageError::NotFound => write!(f, "Resource not found"),
        }
    }
}

impl Error for PolarsStorageError {}

impl From<std::io::Error> for PolarsStorageError {
    fn from(error: std::io::Error) -> Self {
        PolarsStorageError::IoError(error)
    }
}

impl From<PolarsError> for PolarsStorageError {
    fn from(error: PolarsError) -> Self {
        PolarsStorageError::PolarsError(error)
    }
}

pub struct PolarsVectorStorage<P: AsRef<Path>> {
    path: P,
    dimension: usize,
    chunk_size: usize,
}

impl<P: AsRef<Path>> PolarsVectorStorage<P> {
    pub fn new(path: P, dimension: usize, chunk_size: usize) -> Self {
        Self { path, dimension, chunk_size }
    }

    // Helper method to convert embeddings to a list Series
    fn embeddings_to_series(&self, vectors: &Array2<f32>) -> Series {
        // First convert 2D array to Vec of Vecs
        let vector_data: Vec<Vec<f32>> = vectors
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        
        // Convert to Series using ListChunked
        let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new("vector".into(), vector_data.len(), vector_data.len() * self.dimension, DataType::Float32);
        
        for vec in vector_data {
            builder.append_slice(&vec);
        }
        
        builder.finish().into_series()
    }
    
    // Helper to create DataFrame with embeddings
    fn create_df(&self, vectors: &Array2<f32>, start_idx: usize) -> Result<DataFrame, PolarsStorageError> {
        let num_vectors = vectors.nrows();
        
        // Create ID column
        let ids: Vec<u32> = (start_idx..start_idx + num_vectors)
            .map(|i| i as u32)
            .collect();
        
        let id_series = UInt32Chunked::new("id".into(), &ids).into_series();
        let vector_series = self.embeddings_to_series(vectors);
        
        let df = DataFrame::new(vec![id_series.into(), vector_series.into()])?;
        Ok(df)
    }
    
    // Helper to read DataFrame from file
    fn read_df(&self) -> Result<DataFrame, PolarsStorageError> {
        let path_ref = self.path.as_ref();
        
        if !path_ref.exists() {
            return Err(PolarsStorageError::NotFound);
        }
        
        let df = ParquetReader::new(std::fs::File::open(path_ref)?)
            .finish()?;
            
        Ok(df)
    }
    
    // Helper to extract embeddings from DataFrame
    fn extract_embeddings(&self, df: &DataFrame, start_idx: usize, count: usize) -> Result<Array2<f32>, PolarsStorageError> {
        let id_col = df.column("id")?.u32()?;
        
        // Create the logical mask for filtering
        let mask = id_col.clone().into_series().is_between(
            Expr::Literal(LiteralValue::UInt32(start_idx as u32)),
            Expr::Literal(LiteralValue::UInt32((start_idx + count - 1) as u32)),
            true
        );
        
        let filtered = df.filter(&mask)?;
            
        if filtered.height() == 0 {
            return Err(PolarsStorageError::NotFound);
        }
        
        let vectors = filtered.column("vector")?;
        
        // Convert list Series to Array2
        let mut result = Array2::zeros((filtered.height(), self.dimension));
        
        let list_column = vectors.list()?;
        for (i, row_opt) in list_column.iter().enumerate() {
            if let Some(row) = row_opt {
                let row_values = row.to_vec();
                if row_values.len() == self.dimension {
                    for (j, val) in row_values.into_iter().enumerate() {
                        if let AnyValue::Float32(v) = val {
                            result[[i, j]] = v;
                        } else if let AnyValue::Float64(v) = val {
                            result[[i, j]] = v as f32;
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
}

impl<P: AsRef<Path>> VectorStorage for PolarsVectorStorage<P> {
    type Error = PolarsStorageError;

    fn create_or_load_storage(&self, reset: bool) -> Result<(), Self::Error> {
        let path_ref = self.path.as_ref();
        
        // Check if file exists and handle reset flag
        if path_ref.exists() {
            if reset {
                std::fs::remove_file(path_ref)?;
            } else {
                return Ok(());
            }
        }
        
        // Ensure parent directory exists
        if let Some(parent) = path_ref.parent() {
            if !parent.exists() {
                create_dir_all(parent)?;
            }
        }
        
        // Create empty DataFrame with proper schema
        let empty_ids: Vec<u32> = Vec::new();
        let id_series = UInt32Chunked::new("id".into(), empty_ids).into_series();
        let empty_list: Vec<Vec<f32>> = vec![];
        
        let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new("vector".into(), 0, 0, DataType::Float32);
        let vector_series = builder.finish().into_series();
        
        let mut df = DataFrame::new(vec![id_series.into(), vector_series.into])?;
        
        // Write empty DataFrame to Parquet
        ParquetWriter::new(std::fs::File::create(path_ref)?)
            .with_compression(ParquetCompression::Snappy)
            .finish(&mut df)?;
            
        Ok(())
    }

    fn write_slice(&self, vectors: &Array2<f32>, start_idx: usize) -> Result<(), Self::Error> {
        let path_ref = self.path.as_ref();
        
        // Create file if it doesn't exist
        if !path_ref.exists() {
            self.create_or_load_storage(false)?;
        }
        
        // Create DataFrame from vectors
        let mut df = self.create_df(vectors, start_idx)?;
        
        // Write to Parquet file
        ParquetWriter::new(std::fs::File::create(path_ref)?)
            .with_compression(ParquetCompression::Snappy)
            .finish(&mut df)?;
            
        Ok(())
    }

    fn read_slice(&self, start_idx: usize, count: usize) -> Result<Array2<f32>, Self::Error> {
        let df = self.read_df()?;
        self.extract_embeddings(&df, start_idx, count)
    }

    fn append_vector(&self, vector: &Array1<f32>) -> Result<(), Self::Error> {
        // Convert Array1 to Array2 for append_vectors
        let dim = vector.len();
        let mut array2 = Array2::zeros((1, dim));
        for i in 0..dim {
            array2[[0, i]] = vector[i];
        }
        
        self.append_vectors(&array2)
    }

    fn append_vectors(&self, new_vectors: &Array2<f32>) -> Result<(), Self::Error> {
        let path_ref = self.path.as_ref();
        
        // If file doesn't exist, just write directly
        if !path_ref.exists() {
            return self.write_slice(new_vectors, 0);
        }
        
        // Read existing DataFrame
        let existing_df = self.read_df()?;
        let current_count = existing_df.height();
        
        // Create DataFrame from new vectors
        let new_df = self.create_df(new_vectors, current_count)?;
        
        // Concatenate DataFrames
        let mut combined_df = existing_df.vstack(&new_df)?;
        
        // Write back to file
        ParquetWriter::new(std::fs::File::create(path_ref)?)
            .with_compression(ParquetCompression::Snappy)
            .finish(&mut combined_df)?;
            
        Ok(())
    }

    fn get_vector(&self, index: usize) -> Result<Array1<f32>, Self::Error> {
        let df = self.read_df()?;
        
        // Create mask for the specific ID
        let id_filter = df.column("id")?.u32()?.equal(index as u32);
        let filtered = df.filter(&id_filter)?;
        
        if filtered.height() == 0 {
            return Err(PolarsStorageError::NotFound);
        }
        
        let vector_col = filtered.column("vector")?.list()?;
        
        if let Some(vector_data) = vector_col.get(0) {
            let values = vector_data.to_vec();
            let mut result = Array1::zeros(self.dimension);
            
            for (i, val) in values.into_iter().enumerate() {
                if i >= self.dimension {
                    break;
                }
                
                match val {
                    AnyValue::Float32(v) => result[i] = v,
                    AnyValue::Float64(v) => result[i] = v as f32,
                    _ => return Err(PolarsStorageError::PolarsError(PolarsError::ComputeError(
                        "Invalid vector data type".into()
                    ))),
                }
            }
            
            Ok(result)
        } else {
            Err(PolarsStorageError::NotFound)
        }
    }

    fn get_count(&self) -> Result<usize, Self::Error> {
        if !self.path.as_ref().exists() {
            return Ok(0);
        }
        
        let df = self.read_df()?;
        Ok(df.height())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use tempfile::tempdir;

    fn create_test_vectors(count: usize, dim: usize) -> Array2<f32> {
        let mut data = Vec::with_capacity(count * dim);
        for i in 0..count {
            for j in 0..dim {
                data.push((i * dim + j) as f32 / 10.0);
            }
        }
        Array::from_shape_vec((count, dim), data).unwrap()
    }

    #[test]
    fn test_create_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let vstore = PolarsVectorStorage::new(&path, 128, 1000);

        let result = vstore.create_or_load_storage(false);
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_write_and_read_slice() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128;
        let vstore = PolarsVectorStorage::new(&path, dim, 1000);

        // Create test vectors
        let vectors = create_test_vectors(10, dim);

        // Write vectors
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Read vectors back
        let read_result = vstore.read_slice(0, 10);
        assert!(read_result.is_ok());

        let read_vectors = read_result.unwrap();
        assert_eq!(read_vectors.shape(), vectors.shape());

        // Verify data correctness
        for i in 0..vectors.nrows() {
            for j in 0..vectors.ncols() {
                assert!((vectors[[i, j]] - read_vectors[[i, j]]).abs() < 1e-5);
            }
        }
    }
}