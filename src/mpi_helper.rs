use crate::{
    source,
    vectorstore::{VectorStorage, cosine_distance, polars::PolarsVectorStorage},
};

use std::{ops::Mul, path::Path};

use mpi::traits::*;
use ndarray::{Array1, Array2};

pub const ROOT: i32 = 0;

pub fn is_root(rank: i32) -> bool {
    ROOT == rank
}

pub struct RankInterval {
    pub start_index: usize,
    pub end_index: usize,
}

impl RankInterval {
    pub fn get_count(&self) -> usize {
        self.end_index - self.start_index
    }
}

pub struct RankSlice<T> {
    pub interval: RankInterval,
    pub slice: Vec<T>,
}

pub fn interval_by_rank(rank: i32, size: i32, count: usize) -> RankInterval {
    let per_rank = if size as usize > count {
        1
    } else {
        count.div_ceil(size as usize)
    };

    let start_index = per_rank.mul(rank as usize);
    let end_index = if rank == size - 1 {
        count
    } else {
        std::cmp::min(start_index + per_rank, count)
    };

    RankInterval {
        start_index,
        end_index,
    }
}

pub fn slice_by_rank<T: Clone>(rank: i32, size: i32, arr: &Vec<T>) -> RankSlice<T> {
    let interval = interval_by_rank(rank, size, arr.len());
    let slice = arr[interval.start_index..interval.end_index].to_vec();

    RankSlice { interval, slice }
}

pub fn read_files<C: Communicator>(
    dir: &str,
    extensions: &[&str],
    world: &C,
    rank: i32,
    size: i32,
) -> Vec<String> {
    let files = source::find_files_by_extensions(dir, &extensions);
    let mut rank_contents = Vec::new();

    if is_root(rank) {
        println!(
            "Found {} files filtered by extensions: {:?}",
            files.len(),
            extensions
        );
    }
    world.barrier();

    if rank as usize >= files.len() {
        println!("[Rank {}] No files to process.", rank);
    } else {
        let slice_data = slice_by_rank(rank, size, &files);
        rank_contents.extend(slice_data.slice.iter().map(source::read_file).flatten());
    }

    rank_contents
}

pub fn get_local_vstore_path(vstore_dir: &Path, rank: i32) -> String {
    vstore_dir.join(format!("rank_{}.parquet", rank)).to_str().unwrap().to_string()
}

pub fn get_global_vstore_path(vstore_dir: &Path) -> String {
    vstore_dir.join("global.parquet").to_str().unwrap().to_string()
}

// Function to create and write to a process-specific vector store
pub fn process_store_vectors(
    embeddings: &Vec<Vec<f32>>,
    vstore_dir: &Path,
    rank: i32,
    dimension: usize,
    chunk_size: usize,
) -> Result<usize, String> {
    if embeddings.is_empty() {
        return Ok(0);
    }
    
    // Create the storage for this rank
    let vstore_path = get_local_vstore_path(vstore_dir, rank);
    let vstore = PolarsVectorStorage::new(&vstore_path, dimension, chunk_size);
    match vstore.create_or_load_storage(true) {
        Ok(_) => println!("[Rank {}] Created process storage file", rank),
        Err(e) => return Err(format!("Error creating storage: {:?}", e)),
    }

    // Convert all embeddings to a single Array2
    let mut data = Array2::zeros((embeddings.len(), dimension));
    for (i, vec) in embeddings.iter().enumerate() {
        for (j, val) in vec.iter().enumerate() {
            data[[i, j]] = *val;
        }
    }

    // Store all vectors at once
    match vstore.write_slice(&data, 0) {
        Ok(_) => {
            println!(
                "[Rank {}] saved {} vectors in 0 seconds",
                rank,
                embeddings.len()
            );
            Ok(embeddings.len())
        },
        Err(e) => Err(format!("Error saving vectors: {:?}", e)),
    }
}

// Function to merge all process vector stores into a single one
pub fn merge_vector_stores(
    size: i32,
    vstore_dir: &Path,
    dimension: usize,
    chunk_size: usize,
) -> Result<usize, String> {
    // Create the global storage file
    let global_path = get_global_vstore_path(vstore_dir);
    let global_vstore = PolarsVectorStorage::new(&global_path, dimension, chunk_size);
    match global_vstore.create_or_load_storage(true) {
        Ok(_) => println!("Created final storage file"),
        Err(e) => return Err(format!("Error creating final storage: {:?}", e)),
    }

    let mut total_vectors = 0;
    
    // Process each rank's data
    for r in 0..size {
        let rank_path = get_local_vstore_path(vstore_dir, r);
        
        // Skip if this rank's file doesn't exist
        if !Path::new(&rank_path).exists() {
            println!("Rank {} file not found, skipping", r);
            continue;
        }
        
        let rank_vstore = PolarsVectorStorage::new(&rank_path, dimension, chunk_size);
        let rank_count = match rank_vstore.get_count() {
            Ok(count) => count,
            Err(_) => {
                println!("Error reading count from rank {} file, skipping", r);
                continue;
            }
        };
        
        if rank_count == 0 {
            println!("Rank {} has no vectors", r);
            continue;
        }
        
        // Read all vectors from this rank
        match rank_vstore.read_slice(0, rank_count) {
            Ok(vectors) => {
                // Append these vectors to the global store
                match global_vstore.append_vectors(&vectors) {
                    Ok(_) => {
                        println!("Merged {} vectors from rank {}", vectors.nrows(), r);
                        total_vectors += vectors.nrows();
                    },
                    Err(e) => println!("Error appending vectors from rank {}: {:?}", r, e),
                }
            },
            Err(e) => {
                println!("Error reading vectors from rank {}: {:?}", r, e);
                continue;
            }
        }
    }
    
    println!("Merged {} total vectors", total_vectors);
    Ok(total_vectors)
}

fn similarity_search<C: Communicator, V: VectorStorage>(
    vstore_dir: &Path,
    dimension: usize,
    chunk_size: usize,
    rank: i32,
    size: i32,
    top_k: usize,
) -> Vec<f32> {
    let vstore = PolarsVectorStorage::new(get_global_vstore_path(vstore_dir), dimension, chunk_size);
    let vstore_count = vstore.get_count().expect("Fail to get count");
    let target_vector: Array1<f32> = vstore.get_vector(0).expect("Fail to get first vector");
    let rank_interval = interval_by_rank(rank, size, vstore_count);
    let rank_vectors: Array2<f32> = vstore
        .read_slice(rank_interval.start_index, rank_interval.get_count())
        .expect("Fail to read slice");
    let mut distances: Vec<f32> = rank_vectors
        .rows()
        .into_iter()
        .map(|v| cosine_distance(&v.to_owned(), &target_vector))
        .collect();
    distances.sort_by(|a, b| b.partial_cmp(&a).unwrap());
    distances.iter().take(top_k).cloned().collect()
}

pub fn mpi_finish(rank: i32) {
    println!("[Rank {}] Finished", rank);
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;
    use crate::vectorstore::arrow::ArrowVectorStorage;

    // Helper function to create test vectors with distinguishable patterns
    fn create_test_vectors(count: usize, dim: usize, base_value: f32) -> Vec<Vec<f32>> {
        let mut vectors = Vec::with_capacity(count);
        for i in 0..count {
            let mut vector = Vec::with_capacity(dim);
            for j in 0..dim {
                vector.push(base_value + (i as f32 * 0.1) + (j as f32 * 0.01));
            }
            vectors.push(vector);
        }
        vectors
    }

    #[test]
    fn test_interval_by_rank() {
        // Equal distribution case
        let interval = interval_by_rank(1, 4, 100);
        assert_eq!(interval.start_index, 25);
        assert_eq!(interval.end_index, 50);
        
        // Last rank with remainder case
        let interval = interval_by_rank(3, 4, 90);
        assert_eq!(interval.start_index, 69);
        assert_eq!(interval.end_index, 90);
        
        // More ranks than items case
        let interval = interval_by_rank(2, 10, 5);
        assert_eq!(interval.start_index, 2);
        assert_eq!(interval.end_index, 3);
    }

    #[test]
    fn test_slice_by_rank() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        // Middle rank
        let slice_data = slice_by_rank(1, 3, &data);
        assert_eq!(slice_data.interval.start_index, 4);
        assert_eq!(slice_data.interval.end_index, 8);
        assert_eq!(slice_data.slice, vec![5, 6, 7, 8]);
        
        // Last rank
        let slice_data = slice_by_rank(2, 3, &data);
        assert_eq!(slice_data.interval.start_index, 8);
        assert_eq!(slice_data.interval.end_index, 10);
        assert_eq!(slice_data.slice, vec![9, 10]);
    }

    #[test]
    fn test_cosine_distance() {
        // Identical vectors should have distance 0
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let v2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let distance = cosine_distance(&v1, &v2);
        assert!((distance - 0.0).abs() < 1e-5);
        
        // Orthogonal vectors should have distance 1
        let v1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let distance = cosine_distance(&v1, &v2);
        assert!((distance - 1.0).abs() < 1e-5);
        
        // Opposite vectors should have distance 2
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let v2 = Array1::from_vec(vec![-1.0, -2.0, -3.0, -4.0]);
        let distance = cosine_distance(&v1, &v2);
        assert!((distance - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_process_store_vectors() {
        let dir = tempdir().unwrap();
        let vstore_dir = dir.path();
        let rank = 1;
        let dimension = 4;
        let chunk_size = 100;
        
        // Create test vectors
        let vectors = create_test_vectors(5, dimension, 1.0);
        
        // Store vectors
        let result = process_store_vectors(&vectors, vstore_dir, rank, dimension, chunk_size);
        assert!(result.is_ok());
        
        // Verify the file was created
        let vstore_path = get_local_vstore_path(vstore_dir, rank);
        assert!(Path::new(&vstore_path).exists());
        
        // Create a fresh storage instance to read back the data
        let vstore = ArrowVectorStorage::new(vstore_path, dimension, chunk_size);
        let count = vstore.get_count().unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_merge_vector_stores() {
        let dir = tempdir().unwrap();
        let vstore_dir = dir.path();
        let dimension = 4;
        let chunk_size = 100;
        let size = 3; // Number of ranks
        
        // Prepare embeddings for each rank
        let vectors_rank0 = create_test_vectors(3, dimension, 1.0);
        let vectors_rank1 = create_test_vectors(4, dimension, 2.0);
        let vectors_rank2 = create_test_vectors(2, dimension, 3.0);
        
        // Store vectors separately for each rank
        process_store_vectors(&vectors_rank0, vstore_dir, 0, dimension, chunk_size).unwrap();
        process_store_vectors(&vectors_rank1, vstore_dir, 1, dimension, chunk_size).unwrap();
        process_store_vectors(&vectors_rank2, vstore_dir, 2, dimension, chunk_size).unwrap();
        
        // Merge the vector stores
        let result = merge_vector_stores(size, vstore_dir, dimension, chunk_size);
        assert!(result.is_ok());
        
        // Verify the global file was created
        let global_path = get_global_vstore_path(vstore_dir);
        assert!(Path::new(&global_path).exists());
        
        // Create a fresh storage instance to read back the merged data
        let global_vstore = ArrowVectorStorage::new(global_path, dimension, chunk_size);
        let count = global_vstore.get_count().unwrap();
        
        // Should have merged all vectors (3+4+2=9)
        assert_eq!(count, 9);
        
        // Verify vectors from each rank are present by checking a sample
        let v0 = global_vstore.get_vector(0).unwrap();
        assert!((v0[0] - 1.0).abs() < 1e-5); // First vector from rank 0
        
        let v3 = global_vstore.get_vector(3).unwrap();
        assert!((v3[0] - 2.0).abs() < 1e-5); // First vector from rank 1
        
        let v7 = global_vstore.get_vector(7).unwrap();
        assert!((v7[0] - 3.0).abs() < 1e-5); // First vector from rank 2
    }

    #[test]
    fn test_merge_empty_vector_stores() {
        let dir = tempdir().unwrap();
        let vstore_dir = dir.path();
        let dimension = 4;
        let chunk_size = 100;
        let size = 3; // Number of ranks
        
        // Create empty stores for each rank
        for rank in 0..size {
            let vstore_path = get_local_vstore_path(vstore_dir, rank);
            let vstore = ArrowVectorStorage::new(&vstore_path, dimension, chunk_size);
            vstore.create_or_load_storage(true).unwrap();
        }
        
        // Merge the empty vector stores
        let result = merge_vector_stores(size, vstore_dir, dimension, chunk_size);
        assert!(result.is_ok());
        
        // Verify the global file exists but is empty
        let global_path = get_global_vstore_path(vstore_dir);
        assert!(Path::new(&global_path).exists());
        
        let global_vstore = ArrowVectorStorage::new(global_path, dimension, chunk_size);
        let count = global_vstore.get_count().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_merge_with_missing_ranks() {
        let dir = tempdir().unwrap();
        let vstore_dir = dir.path();
        let dimension = 4;
        let chunk_size = 100;
        let size = 3; // Number of ranks
        
        // Only create stores for ranks 0 and 2 (skip rank 1)
        let vectors_rank0 = create_test_vectors(3, dimension, 1.0);
        let vectors_rank2 = create_test_vectors(2, dimension, 3.0);
        
        process_store_vectors(&vectors_rank0, vstore_dir, 0, dimension, chunk_size).unwrap();
        process_store_vectors(&vectors_rank2, vstore_dir, 2, dimension, chunk_size).unwrap();
        
        // Merge the vector stores
        let result = merge_vector_stores(size, vstore_dir, dimension, chunk_size);
        assert!(result.is_ok());
        
        // Verify the global file was created
        let global_path = get_global_vstore_path(vstore_dir);
        assert!(Path::new(&global_path).exists());
        
        // Create a fresh storage instance to read back the merged data
        let global_vstore = ArrowVectorStorage::new(global_path, dimension, chunk_size);
        let count = global_vstore.get_count().unwrap();
        
        // Should have merged vectors from ranks 0 and 2 (3+2=5)
        assert_eq!(count, 5);
    }
    
    #[test]
    fn test_merge_large_vectors() {
        let dir = tempdir().unwrap();
        let vstore_dir = dir.path();
        let dimension = 128; // Typical embedding dimension
        let chunk_size = 100;
        let size = 2; // Number of ranks
        
        // Create large test vectors for each rank
        let vectors_rank0 = create_test_vectors(50, dimension, 1.0);
        let vectors_rank1 = create_test_vectors(50, dimension, 2.0);
        
        // Store vectors for each rank
        process_store_vectors(&vectors_rank0, vstore_dir, 0, dimension, chunk_size).unwrap();
        process_store_vectors(&vectors_rank1, vstore_dir, 1, dimension, chunk_size).unwrap();
        
        // Merge the vector stores
        let result = merge_vector_stores(size, vstore_dir, dimension, chunk_size);
        assert!(result.is_ok());
        
        // Create a fresh storage instance to read back the merged data
        let global_path = get_global_vstore_path(vstore_dir);
        let global_vstore = ArrowVectorStorage::new(global_path, dimension, chunk_size);
        let count = global_vstore.get_count().unwrap();
        
        // Should have merged all vectors (50+50=100)
        assert_eq!(count, 100);
    }
}