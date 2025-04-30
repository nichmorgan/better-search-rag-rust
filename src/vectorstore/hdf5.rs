use std::path::Path;
use ndarray::{Array1, Array2};
use hdf5::{File, Result};

use super::VectorStorage;

/// Implementation of VectorStorage using HDF5
pub struct Hdf5VectorStorage;

impl VectorStorage for Hdf5VectorStorage {
    type Error = hdf5::Error;
    
    fn create_storage<P: AsRef<Path>>(
        path: P, 
        num_vectors: usize, 
        dimension: usize,
        chunk_size: usize,
        reset: bool
    ) -> Result<()> {
        let path_ref = path.as_ref();
        
        // Check if file already exists
        if path_ref.exists() {
            if reset {
                // Delete the file if reset is true
                std::fs::remove_file(path_ref).map_err(|e| 
                    hdf5::Error::Internal(format!("Failed to remove existing file: {}", e)))?;
            } else {
                // If file exists and reset is false, return without doing anything
                return Ok(());
            }
        }
    
        let file = File::create(path)?;
        let group = file.create_group("vectors")?;
        
        // Create extensible dataset with chunking for parallel access
        let _dataset = group.new_dataset::<f32>()
            .chunk((chunk_size, dimension))
            .shape((num_vectors, dimension))
            .deflate(3)
            .create("data")?;
        
        // Store metadata
        let attr = group.new_attr::<usize>().create("num_vectors")?;
        attr.write(&[num_vectors])?;
        
        let attr = group.new_attr::<usize>().create("dimension")?;
        attr.write(&[dimension])?;
        
        Ok(())
    }
    
    fn write_slice<P: AsRef<Path>>(
        path: P,
        vectors: &Array2<f32>, 
        start_idx: usize
    ) -> Result<()> {
        let file = File::open_rw(path)?;
        let group = file.group("vectors")?;
        let dataset = group.dataset("data")?;
        
        // Write the slice at the specified position
        dataset.write_slice(vectors, (start_idx, 0))?;
        
        Ok(())
    }
    
    fn read_slice<P: AsRef<Path>>(
        path: P,
        start_idx: usize, 
        count: usize
    ) -> Result<Array2<f32>> {
        let file = File::open(path)?;
        let group = file.group("vectors")?;
        let dataset = group.dataset("data")?;
        
        // Get the vector dimension
        let shape = dataset.shape();
        let dim = shape[1];
        
        // Calculate actual count (handle edge case)
        let available = shape[0] - start_idx;
        let actual_count = std::cmp::min(count, available);
        
        // Read the slice - this is optimized for chunked access
        let slice = dataset.slice(s![start_idx:start_idx+actual_count, 0:dim]).read::<Array2<f32>>()?;

        
        Ok(slice)
    }
    
    fn append_vector<P: AsRef<Path>>(
        path: P,
        vector: &Array1<f32>
    ) -> Result<usize> {
        let file = File::open_rw(path)?;
        let group = file.group("vectors")?;
        let dataset = group.dataset("data")?;
        let attr = group.attr("num_vectors")?;
        
        // Atomic read of current count
        let mut num_vectors = vec![0usize; 1];
        let num_vectors: Vec<usize> = attr.read()?;
        let current_idx = num_vectors[0];
        
        // Resize dataset if needed
        let shape = dataset.shape();
        if current_idx >= shape[0] {
            dataset.resize((shape[0] + 1000, shape[1]))?; // Grow by chunks
        }
        
        // Write the vector
        let vector_2d = vector.clone().into_shape((1, vector.len())).unwrap();
        dataset.write_slice(&vector_2d, (current_idx, 0))?;
        
        // Update count atomically
        num_vectors[0] = current_idx + 1;
        attr.write(&num_vectors)?;
        
        Ok(current_idx)
    }
    
    fn get_vector<P: AsRef<Path>>(
        path: P,
        index: usize
    ) -> Result<Array1<f32>> {
        let file = File::open(path)?;
        let group = file.group("vectors")?;
        let dataset = group.dataset("data")?;
        
        let shape = dataset.shape();
        if index >= shape[0] {
            return Err(hdf5::Error::Internal("Index out of bounds".to_string()));
        }
        
        let dim = shape[1];
        let slice = dataset.slice(s![index:index+1, 0:dim]).read::<Array2<f32>>()?;
        
        // Convert to 1D array
        let vector = slice.into_shape((dim,)).unwrap();
        
        Ok(vector)
    }
    
    fn compute_similarities<P: AsRef<Path>>(
        path: P,
        query: &Array1<f32>,
        start_idx: usize,
        count: usize
    ) -> Result<Vec<(usize, f32)>> {
        // Read the slice for local computation
        let vectors = Self::read_slice(path, start_idx, count)?;
        let dim = vectors.shape()[1];
        
        // Precompute query vector norm
        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        let query_norm = query_norm_sq.sqrt();
        
        // Calculate similarities
        let mut similarities = Vec::with_capacity(count);
        for i in 0..vectors.shape()[0] {
            let row = vectors.slice(ndarray::s![i, ..]);
            
            // Compute dot product
            let mut dot_product = 0.0;
            let mut vec_norm_sq = 0.0;
            
            for j in 0..dim {
                dot_product += query[j] * row[j];
                vec_norm_sq += row[j] * row[j];
            }
            
            let vec_norm = vec_norm_sq.sqrt();
            let similarity = if query_norm * vec_norm > 1e-6 {
                dot_product / (query_norm * vec_norm)
            } else {
                0.0
            };
            
            similarities.push((start_idx + i, similarity));
        }
        
        Ok(similarities)
    }
}