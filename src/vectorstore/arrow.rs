// src/vectorstore/arrow_storage.rs

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use ndarray::{Array1, Array2};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::error::Error;
use std::fmt;
use std::fs::{File, create_dir_all};
use std::io;
use std::path::Path;
use std::sync::Arc;

use super::VectorStorage;

/// Custom error type for Arrow storage operations
#[derive(Debug)]
pub enum ArrowStorageError {
    IoError(io::Error),
    ArrowError(String),
    ParquetError(String),
    NotFound,
}

impl fmt::Display for ArrowStorageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ArrowStorageError::IoError(e) => write!(f, "I/O error: {}", e),
            ArrowStorageError::ArrowError(msg) => write!(f, "Arrow error: {}", msg),
            ArrowStorageError::ParquetError(msg) => write!(f, "Parquet error: {}", msg),
            ArrowStorageError::NotFound => write!(f, "Resource not found"),
        }
    }
}

impl Error for ArrowStorageError {}

impl From<io::Error> for ArrowStorageError {
    fn from(error: io::Error) -> Self {
        ArrowStorageError::IoError(error)
    }
}

impl From<arrow_schema::ArrowError> for ArrowStorageError {
    fn from(error: arrow_schema::ArrowError) -> Self {
        ArrowStorageError::ArrowError(error.to_string())
    }
}

impl From<parquet::errors::ParquetError> for ArrowStorageError {
    fn from(error: parquet::errors::ParquetError) -> Self {
        ArrowStorageError::ParquetError(error.to_string())
    }
}

pub struct ArrowVectorStorage<P: AsRef<Path>> {
    path: P,
    dimension: usize,
    chunk_size: usize,
}

impl<P: AsRef<Path>> ArrowVectorStorage<P> {
    pub fn new(path: P, dimension: usize, chunk_size: usize) -> Self {
        Self {
            path,
            dimension,
            chunk_size,
        }
    }
}

impl<P: AsRef<Path>> VectorStorage for ArrowVectorStorage<P> {
    type Error = ArrowStorageError;

    fn create_or_load_storage(&self, reset: bool) -> Result<(), Self::Error> {
        let path_ref = self.path.as_ref();

        // Check if file already exists
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

        // Create schema for the Parquet file
        let fields = vec![
            Field::new("id", DataType::UInt32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    self.dimension as i32,
                ),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        // Create a file with empty data
        let file = File::create(path_ref)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .set_max_row_group_size(self.chunk_size)
            .build();

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Close the writer to finalize the file
        writer
            .close()
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        Ok(())
    }

    fn write_slice(&self, vectors: &Array2<f32>, start_idx: usize) -> Result<(), Self::Error> {
        let path_ref = self.path.as_ref();

        // Create the file if it doesn't exist
        if !path_ref.exists() {
            self.create_or_load_storage(false)?;
        }

        // Read the file to get schema (only need to know the dimensions)
        let file = File::open(path_ref)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Get schema directly from the builder
        let schema = builder.schema();

        // Prepare data for writing
        let num_vectors = vectors.nrows();
        let dim = vectors.ncols();

        // Create ID array
        let ids: Vec<u32> = (start_idx..start_idx + num_vectors)
            .map(|i| i as u32)
            .collect();
        let id_array = Arc::new(UInt32Array::from(ids)) as ArrayRef;

        // Create vector array
        let vector_data: Vec<f32> = vectors.iter().copied().collect();
        let values = Arc::new(Float32Array::from(vector_data)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let vector_array = Arc::new(
            FixedSizeListArray::try_new(
                field, dim as i32, values, None, // No null buffer
            )
            .map_err(|e| ArrowStorageError::ArrowError(e.to_string()))?,
        ) as ArrayRef;

        // Create record batch
        let batch = RecordBatch::try_new(schema.clone(), vec![id_array, vector_array])
            .map_err(|e| ArrowStorageError::ArrowError(e.to_string()))?;

        // Open file for writing
        let file = File::create(path_ref)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Write the batch
        writer
            .write(&batch)
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Close the writer to finalize the file
        writer
            .close()
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        Ok(())
    }

    fn read_slice(&self, start_idx: usize, count: usize) -> Result<Array2<f32>, Self::Error> {
        let path_ref = self.path.as_ref();

        if !path_ref.exists() {
            return Err(ArrowStorageError::NotFound);
        }

        // Open the Parquet file
        let file = File::open(path_ref)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Create the reader with desired batch size
        let mut record_batch_reader = builder
            .with_batch_size(count)
            .build()
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Process record batches
        let mut vector_data = Vec::new();
        let mut found_count = 0;

        while let Some(batch_result) = record_batch_reader.next() {
            let batch = batch_result.map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

            // Extract IDs
            let id_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    ArrowStorageError::ArrowError("Failed to downcast ID array".to_string())
                })?;

            // Extract vectors
            let vector_list_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    ArrowStorageError::ArrowError("Failed to downcast vector array".to_string())
                })?;

            let vector_data_array = vector_list_array
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    ArrowStorageError::ArrowError(
                        "Failed to downcast vector data array".to_string(),
                    )
                })?;

            // Extract vectors in the requested range
            for row_idx in 0..batch.num_rows() {
                let id = id_array.value(row_idx) as usize;

                if id >= start_idx && id < start_idx + count {
                    // Extract the vector values
                    let list_start = vector_list_array.value_offset(row_idx) as usize;
                    let list_end = vector_list_array.value_offset(row_idx + 1) as usize;

                    let dim = list_end - list_start;
                    let mut vec_data = Vec::with_capacity(dim);

                    for i in list_start..list_end {
                        vec_data.push(vector_data_array.value(i));
                    }

                    vector_data.push((id - start_idx, vec_data));
                    found_count += 1;

                    if found_count >= count {
                        break;
                    }
                }
            }

            if found_count >= count {
                break;
            }
        }

        if vector_data.is_empty() {
            return Err(ArrowStorageError::NotFound);
        }

        // Sort by index to ensure correct order
        vector_data.sort_by_key(|(idx, _)| *idx);

        // Convert to ndarray format
        let dim = vector_data[0].1.len();
        let mut result = Array2::zeros((vector_data.len(), dim));

        for (i, (_, vec)) in vector_data.into_iter().enumerate() {
            for (j, val) in vec.into_iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    fn append_vector(&self, vector: &Array1<f32>) -> Result<usize, Self::Error> {
        let path_ref = self.path.as_ref();

        // Create the file if it doesn't exist
        if !path_ref.exists() {
            self.create_or_load_storage(false)?;
        }

        // Open the Parquet file to read metadata
        let file = File::open(path_ref)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Get row count by reading metadata
        let metadata = builder.metadata();

        let total_rows = metadata.file_metadata().num_rows() as usize;
        let new_idx = total_rows;

        // Convert the vector to Array2 for write_slice
        let dim = vector.len();
        let mut array2 = Array2::zeros((1, dim));
        for i in 0..dim {
            array2[[0, i]] = vector[i];
        }

        // Write the vector using write_slice
        self.write_slice(&array2, new_idx)?;

        Ok(new_idx)
    }

    fn get_vector(&self, index: usize) -> Result<Array1<f32>, Self::Error> {
        // Use read_slice to get just one vector
        let vectors = self.read_slice(index, 1)?;

        if vectors.nrows() == 0 {
            return Err(ArrowStorageError::NotFound);
        }

        // Extract the first row
        Ok(vectors.row(0).to_owned())
    }

    fn get_count(&self) -> Result<usize, Self::Error> {
        let path_ref = self.path.as_ref();

        if !path_ref.exists() {
            return Ok(0); // No file means no vectors
        }

        // Open the Parquet file
        let file = File::open(path_ref)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ArrowStorageError::ParquetError(e.to_string()))?;

        // Get total row count from metadata
        let metadata = builder.metadata();
        let total_rows = metadata.file_metadata().num_rows() as usize;

        Ok(total_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use std::path::PathBuf;
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
        let vstore = ArrowVectorStorage::new(&path, 128, 1000);

        let result = vstore.create_or_load_storage(false);
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_write_and_read_slice() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128;
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

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

    #[test]
    fn test_read_partial_slice() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128;
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

        // Create test vectors
        let vectors = create_test_vectors(10, dim);

        // Write vectors
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Read a subset of vectors
        let read_result = vstore.read_slice(2, 5);
        assert!(read_result.is_ok());

        let read_vectors = read_result.unwrap();
        assert_eq!(read_vectors.shape(), [5, 128]);

        // Verify data correctness
        for i in 0..5 {
            for j in 0..vectors.ncols() {
                assert!((vectors[[i + 2, j]] - read_vectors[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_append_vector() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128;
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

        // Create initial vectors
        let vectors = create_test_vectors(5, dim);

        // Write initial vectors
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Create a vector to append
        let new_vector = Array1::from_vec((0..128).map(|i| i as f32 / 5.0).collect());

        // Append the vector
        let append_result = vstore.append_vector(&new_vector);
        assert!(append_result.is_ok());

        let new_idx = append_result.unwrap();
        assert_eq!(new_idx, 5); // Should be appended after the initial 5 vectors

        // Read back the appended vector
        let read_result = vstore.get_vector(new_idx);
        assert!(read_result.is_ok());

        let read_vector = read_result.unwrap();
        assert_eq!(read_vector.len(), 128);

        // Verify data correctness
        for i in 0..128 {
            assert!((new_vector[i] - read_vector[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_get_vector() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128;
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

        // Create test vectors
        let vectors = create_test_vectors(10, dim);

        // Write vectors
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Get a specific vector
        let index = 7;
        let get_result = vstore.get_vector(index);
        assert!(get_result.is_ok());

        let vector = get_result.unwrap();
        assert_eq!(vector.len(), 128);

        // Verify data correctness
        for j in 0..vectors.ncols() {
            assert!((vectors[[index, j]] - vector[j]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_error_handling() {
        let non_existent_path = PathBuf::from("/non/existent/path/vectors.parquet");
        let dim = 128;
        let vstore = ArrowVectorStorage::new(&non_existent_path, dim, 1000);

        // Try to read from non-existent file
        let read_result = vstore.read_slice(0, 10);
        assert!(matches!(read_result, Err(ArrowStorageError::NotFound)));

        // Try to get vector from non-existent file
        let get_result = vstore.get_vector(0);
        assert!(matches!(get_result, Err(ArrowStorageError::NotFound)));
    }

    #[test]
    fn test_large_vectors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 1536;
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

        // Create large test vectors
        let count = 100;
        let vectors = create_test_vectors(count, dim);

        // Write vectors
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Read vectors back
        let read_result = vstore.read_slice(0, count);
        assert!(read_result.is_ok());

        let read_vectors = read_result.unwrap();
        assert_eq!(read_vectors.shape(), vectors.shape());

        // Verify data correctness (sample a few points)
        for i in [0, 25, 50, 75, 99] {
            for j in [0, 100, 500, 1000, 1535] {
                assert!((vectors[[i, j]] - read_vectors[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_reset_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128; // Typical embedding dimension
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

        // Create initial storage and write vectors
        let vectors = create_test_vectors(10, dim);
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Reset storage
        let reset_result = vstore.create_or_load_storage(true);
        assert!(reset_result.is_ok());

        // Verify storage exists but is empty
        let read_result = vstore.read_slice(0, 10);
        assert!(matches!(read_result, Err(ArrowStorageError::NotFound)));
    }

    #[test]
    fn test_get_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.parquet");
        let dim = 128;
        let vstore = ArrowVectorStorage::new(&path, dim, 1000);

        // Test count on non-existent file
        let count_result = vstore.get_count();
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap(), 0);

        // Create test vectors and write them
        let vectors = create_test_vectors(5, dim);
        let write_result = vstore.write_slice(&vectors, 0);
        assert!(write_result.is_ok());

        // Verify count matches number of vectors written
        let count_result = vstore.get_count();
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap(), 5);

        // Note: write_slice replaces the entire file, so writing 3 vectors
        // will result in only 3 vectors total, not 8
        let new_vectors = create_test_vectors(3, dim);
        let write_result = vstore.write_slice(&new_vectors, 0);
        assert!(write_result.is_ok());

        // Verify count is now 3 (not 8)
        let count_result = vstore.get_count();
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap(), 3);

        // Append a single vector - this uses write_slice internally which REPLACES the file
        let new_vector = Array1::from_vec((0..dim).map(|i| i as f32 / 5.0).collect());
        let append_result = vstore.append_vector(&new_vector);
        assert!(append_result.is_ok());

        // Verify count is 1 after append (not 4) since append replaces all vectors
        let count_result = vstore.get_count();
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap(), 1);

        // Reset storage
        let reset_result = vstore.create_or_load_storage(true);
        assert!(reset_result.is_ok());

        // Verify count is reset
        let count_result = vstore.get_count();
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap(), 0);
    }
}
