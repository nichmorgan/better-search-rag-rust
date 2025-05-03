use crate::{
    source,
    vectorstore::{VectorStorage, cosine_distance, arrow::ArrowVectorStorage},
};

use std::{ops::Mul, path::Path, time::Instant};

use mpi::traits::*;
use ndarray::{Array1, Array2};

pub const ROOT: i32 = 0;

pub fn is_root(rank: i32) -> bool {
    ROOT == rank
}

pub struct RankInterval {
    pub per_rank: usize,
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
        per_rank,
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
    vstore_dir.join(format!("rank_{}", rank)).to_str().unwrap().to_string()
}

pub fn get_global_vstore_path(vstore_dir: &Path) -> String {
    vstore_dir.join("global").to_str().unwrap().to_string()
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
    let vstore = ArrowVectorStorage::new(get_local_vstore_path(vstore_dir, rank), dimension, chunk_size);
    match vstore.create_or_load_storage(true) {
        Ok(_) => println!("[Rank {}] Created process storage file", rank),
        Err(e) => return Err(format!("Error creating storage: {:?}", e)),
    }

    let start = Instant::now();
    let mut successful_vectors = 0;

    for (i, vector) in embeddings.iter().enumerate() {
        match vstore.append_vector(&Array1::from_vec(vector.clone())) {
            Ok(_idx) => {
                successful_vectors += 1;
                if i % 10 == 0 || i == embeddings.len() - 1 {
                    println!(
                        "[Rank {}] Saved vector {}/{}",
                        rank,
                        i + 1,
                        embeddings.len()
                    );
                }
            }
            Err(e) => {
                println!(
                    "[Rank {}] Error saving vector {}/{}: {:?}",
                    rank,
                    i + 1,
                    embeddings.len(),
                    e
                );
            }
        }
    }

    let elapsed = Instant::now() - start;
    println!(
        "[Rank {}] saved {} vectors in {} seconds",
        rank,
        successful_vectors,
        elapsed.as_secs()
    );

    Ok(successful_vectors)
}

// Function to merge all process vector stores into a single one
pub fn merge_vector_stores(
    size: i32,
    vstore_dir: &Path,
    dimension: usize,
    chunk_size: usize,
) -> Result<usize, String> {
    // Create the final storage file
    let global_vstore = ArrowVectorStorage::new(get_global_vstore_path(vstore_dir), dimension, chunk_size);
    match global_vstore.create_or_load_storage(true) {
        Ok(_) => println!("Created final storage file"),
        Err(e) => return Err(format!("Error creating final storage: {:?}", e)),
    }

    let start = Instant::now();
    // Merge all process files into the final storage
    let mut total_vectors = 0;
    for r in 0..size {
        let process_vstore = ArrowVectorStorage::new(get_local_vstore_path(vstore_dir, r), dimension, chunk_size);
        
        // Read all vectors from this process file
        let mut count = 0;
        let batch_size = 100; // Process in batches to avoid memory issues
        let mut start_idx = 0;

        loop {
            match process_vstore.read_slice(start_idx, batch_size) {
                Ok(vectors) => {
                    if vectors.nrows() == 0 {
                        break;
                    }

                    // Write these vectors to the main storage
                    match global_vstore.write_slice(&vectors, total_vectors) {
                        Ok(_) => {}
                        Err(e) => println!("Error writing batch from local vstore: {:?}", e),
                    }

                    count += vectors.nrows();
                    total_vectors += vectors.nrows();
                    start_idx += batch_size;

                    if vectors.nrows() < batch_size {
                        break;
                    }
                }
                Err(e) => {
                    let error_msg = format!("{:?}", e);
                    if error_msg.contains("NotFound") {
                        println!("No more vectors in rank {} file", r);
                    } else {
                        println!("Error reading from rank {} file: {:?}", r, e);
                    }
                    break;
                }
            }

            println!("Merged {} vectors from local vstore", count);
        }
    }

    let elapsed = Instant::now() - start;
    println!(
        "Merged {} total vectors in {} seconds",
        total_vectors,
        elapsed.as_secs()
    );

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
    let vstore = ArrowVectorStorage::new(get_global_vstore_path(vstore_dir), dimension, chunk_size);
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
