use crate::{source, vectorstore};

use std::{fs, ops::Mul, path::Path, time::Instant};

use mpi::traits::*;
use ndarray::Array1;
use vectorstore::{VectorStorage, arrow::ArrowVectorStorage};

pub const ROOT: i32 = 0;

pub fn is_root(rank: i32) -> bool {
    ROOT == rank
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
        let files_per_rank = if size as usize > files.len() {
            1
        } else {
            files.len().div_ceil(size as usize)
        };
        let start_index = files_per_rank.mul(rank as usize);
        let end_index = if rank == size - 1 {
            files.len()
        } else {
            std::cmp::min(start_index + files_per_rank, files.len())
        };

        let rank_files = &files[start_index..end_index];
        rank_contents.extend(rank_files.iter().map(source::read_file).flatten());
    }

    rank_contents
}

// Function to create and write to a process-specific vector store
pub fn process_store_vectors(
    embeddings: &Vec<Vec<f32>>,
    path: &str,
    rank: i32,
    dimension: usize,
    chunk_size: usize,
) -> Result<usize, String> {
    if embeddings.is_empty() {
        return Ok(0);
    }

    let process_path = format!("{}_rank_{}", path, rank);
    let vstore = ArrowVectorStorage::new(&process_path, dimension, chunk_size);

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
pub fn merge_vector_stores<C: Communicator>(
    world: &C,
    base_path: &str,
    dimension: usize,
    chunk_size: usize,
) -> Result<usize, String> {
    // Create the final storage file
    let vstore = ArrowVectorStorage::new(&base_path, dimension, chunk_size);

    match vstore.create_or_load_storage(true) {
        Ok(_) => println!("Created final storage file"),
        Err(e) => return Err(format!("Error creating final storage: {:?}", e)),
    }

    let start = Instant::now();
    let size = world.size();

    // Merge all process files into the final storage
    let mut total_vectors = 0;
    for r in 0..size {
        let process_file = format!("{}_rank_{}", base_path, r);
        let process_vstore = ArrowVectorStorage::new(&process_file, dimension, chunk_size);

        if Path::new(&process_file).exists() {
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
                        match vstore.write_slice(&vectors, total_vectors) {
                            Ok(_) => {}
                            Err(e) => println!("Error writing batch from rank {}: {:?}", r, e),
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
            }

            println!("Merged {} vectors from rank {}", count, r);

            // Clean up the process file
            let _ = fs::remove_file(&process_file);
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

// fn similarity_search<C: Communicator, V: VectorStorage>(
//     world: &C,
//     rank: i32,
//     size: i32,
//     vstore: &V,
// ) {

// }

pub fn mpi_finish(rank: i32) {
    println!("[Rank {}] Finished", rank);
    std::process::exit(0);
}