use crate::{
    llm::LlmService,
    metrics::cosine_distance,
    source,
    vectorstore::polars::{PolarsVectorstore, SliceArgs},
};
use std::{ops::Mul, path::Path};

use mpi::traits::*;
use polars::error::PolarsError;

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

pub fn read_files_chunked<C: Communicator>(
    dir: &str,
    extensions: &[&str],
    world: &C,
    rank: i32,
    size: i32,
    chunk_size: usize, // Number of files to process in one chunk
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
        return rank_contents;
    }

    let slice_data = slice_by_rank(rank, size, &files);
    let total_files = slice_data.slice.len();
    let total_chunks = (total_files + chunk_size - 1) / chunk_size;

    println!(
        "[Rank {}] Processing {} files in {} chunks (chunk size: {})",
        rank, total_files, total_chunks, chunk_size
    );

    // Process files in chunks
    for chunk_index in 0..total_chunks {
        let chunk_start = chunk_index * chunk_size;
        let chunk_end = std::cmp::min(chunk_start + chunk_size, total_files);
        let chunk = &slice_data.slice[chunk_start..chunk_end];

        println!(
            "[Rank {}] Processing chunk {}/{} ({} files)",
            rank,
            chunk_index + 1,
            total_chunks,
            chunk.len()
        );

        // Process this chunk
        let chunk_contents: Vec<String> = chunk
            .iter()
            .filter_map(|path| source::read_file(path))
            .collect();

        rank_contents.extend(chunk_contents);
    }

    println!(
        "[Rank {}] Completed processing {} files",
        rank,
        rank_contents.len()
    );
    rank_contents
}

pub fn get_local_vstore(vstore_dir: &Path, rank: i32, empty: bool) -> PolarsVectorstore {
    PolarsVectorstore::new(
        vstore_dir
            .join(format!("rank_{}.parquet", rank))
            .to_str()
            .unwrap(),
        empty,
    )
}

// In src/mpi_helper.rs
pub fn get_global_vstore(dir: &Path, empty: bool) -> PolarsVectorstore {
    // Create a consistent file path by always using "global.parquet"
    let file_path = dir.join("global.parquet").to_str().unwrap().to_string();
    PolarsVectorstore::new(&file_path, empty)
}

pub fn process_store_vectors(
    vstore: &mut PolarsVectorstore,
    embeddings: &Vec<Vec<f32>>,
    rank: i32,
) -> Result<usize, String> {
    if embeddings.is_empty() {
        return Ok(0);
    }

    match vstore.append_many(&embeddings) {
        Ok(_) => {
            println!(
                "[Rank {}] saved {} vectors in 0 seconds",
                rank,
                embeddings.len()
            );
            Ok(embeddings.len())
        }
        Err(e) => Err(format!("Error saving vectors: {:?}", e)),
    }
}

pub fn process_files_embeddings_chunked<C: Communicator, E>(
    dir: &str,
    extensions: &[&str],
    world: &C,
    rank: i32,
    size: i32,
    llm_service: &impl LlmService<Error = E>,
    vstore: &mut PolarsVectorstore,
    chunk_size: usize,
) -> Result<usize, String>
where
    E: std::fmt::Debug,
{
    let files = source::find_files_by_extensions(dir, &extensions);

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
        return Ok(0);
    }

    let slice_data = slice_by_rank(rank, size, &files);
    let total_files = slice_data.slice.len();
    let total_chunks = (total_files + chunk_size - 1) / chunk_size;

    println!(
        "[Rank {}] Processing {} files in {} chunks (chunk size: {})",
        rank, total_files, total_chunks, chunk_size
    );

    let mut total_processed = 0;

    // Process files in chunks
    for chunk_index in 0..total_chunks {
        let chunk_start = chunk_index * chunk_size;
        let chunk_end = std::cmp::min(chunk_start + chunk_size, total_files);
        let chunk_files = &slice_data.slice[chunk_start..chunk_end];

        println!(
            "[Rank {}] Processing chunk {}/{} ({} files)",
            rank,
            chunk_index + 1,
            total_chunks,
            chunk_files.len()
        );

        // Read files in this chunk
        let read_start = std::time::Instant::now();
        let chunk_contents: Vec<String> = chunk_files
            .iter()
            .filter_map(|path| source::read_file(path))
            .collect();

        println!(
            "[Rank {}] Read {} files in {:?}",
            rank,
            chunk_contents.len(),
            read_start.elapsed()
        );

        if chunk_contents.is_empty() {
            println!(
                "[Rank {}] No valid content in chunk {}, skipping",
                rank,
                chunk_index + 1
            );
            continue;
        }

        // Generate embeddings for this chunk
        let embed_start = std::time::Instant::now();
        let chunk_embeddings = match llm_service.get_embeddings(&chunk_contents) {
            Ok(emb) => emb,
            Err(e) => {
                println!("[Rank {}] Error generating embeddings: {:?}", rank, e);
                continue;
            }
        };

        println!(
            "[Rank {}] Generated {} embeddings in {:?}",
            rank,
            chunk_embeddings.len(),
            embed_start.elapsed()
        );

        // Store embeddings for this chunk
        let store_start = std::time::Instant::now();
        match process_store_vectors(vstore, &chunk_embeddings, rank) {
            Ok(count) => {
                println!(
                    "[Rank {}] Stored {} embeddings in {:?}",
                    rank,
                    count,
                    store_start.elapsed()
                );
                total_processed += count;
            }
            Err(e) => {
                println!("[Rank {}] Error storing embeddings: {}", rank, e);
            }
        }

        // Explicitly release memory
        drop(chunk_contents);
        drop(chunk_embeddings);

        println!(
            "[Rank {}] Completed chunk {}/{} - Total processed: {}",
            rank,
            chunk_index + 1,
            total_chunks,
            total_processed
        );
    }

    let persist_start = std::time::Instant::now();
    if let Err(e) = vstore.persist() {
        println!("[Rank {}] Error persisting embeddings: {:?}", rank, e);
    } else {
        println!(
            "[Rank {}] Persisted embeddings in {:?}",
            rank,
            persist_start.elapsed()
        );
    }

    println!(
        "[Rank {}] Completed all chunks. Total embeddings: {}",
        rank, total_processed
    );
    Ok(total_processed)
}

pub fn merge_vector_stores(size: i32, vstore_dir: &Path) -> Result<PolarsVectorstore, String> {
    let mut global_vstore = get_global_vstore(vstore_dir, true);
    let mut total_vectors = 0;

    // Process each rank's data
    for r in 0..size {
        let rank_vstore = get_local_vstore(vstore_dir, r, false);

        // Read all vectors from this rank
        match rank_vstore.get_many(None) {
            Ok(vectors) => {
                if vectors.is_empty() {
                    println!("[Rank {}] Skipping empty vector from get_many", r);
                    continue;
                }
                // Append these vectors to the global store
                match global_vstore.append_many(&vectors) {
                    Ok(_) => {
                        println!("Merged {} vectors from rank {}", vectors.len(), r);
                        total_vectors += vectors.len();
                    }
                    Err(e) => println!(
                        "[Rank {}] Error appending vectors {:?}: {:?}",
                        r, vectors, e
                    ),
                }
            }
            Err(e) => {
                println!("[Rank {}] Error reading vectors: {:?}", r, e);
                continue;
            }
        }
    }

    println!("Merged {} total vectors", total_vectors);
    Ok(global_vstore)
}

pub fn similarity_search(
    vstore_dir: &Path,
    rank: i32,
    size: i32,
    top_k: usize,
) -> Result<Vec<(usize, f32)>, PolarsError> {
    let vstore = get_global_vstore(vstore_dir, false);
    let vstore_count = vstore.get_count()?;
    let target_vector = vstore.get(0)?;
    
    // Determine the portion of vectors this rank should process
    let rank_interval = interval_by_rank(rank, size, vstore_count);
    
    // Retrieve vectors for this rank
    let rank_vectors = vstore.get_many(Some(SliceArgs {
        offset: rank_interval.start_index as i32,
        length: rank_interval.get_count(),
    }))?;
    
    // Calculate distances between target vector and each vector in this rank's portion
    let mut distances: Vec<(usize, f32)> = rank_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // Convert local index to global index
            let global_idx = rank_interval.start_index + i;
            (global_idx, cosine_distance(v, &target_vector))
        })
        .collect();
    
    // Sort by distance (smallest first) and keep only top_k
    distances.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    if distances.len() > top_k {
        distances.truncate(top_k);
    }
    
    Ok(distances)
}

pub fn mpi_finish(rank: i32) {
    println!("[Rank {}] Finished", rank);
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::*;

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
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![1.0, 2.0, 3.0, 4.0];
        let distance = cosine_distance(&v1, &v2);
        assert!((distance - 0.0).abs() < 1e-5);

        // Orthogonal vectors should have distance 1
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let distance = cosine_distance(&v1, &v2);
        assert!((distance - 1.0).abs() < 1e-5);

        // Opposite vectors should have distance 2
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![-1.0, -2.0, -3.0, -4.0];
        let distance = cosine_distance(&v1, &v2);
        assert!((distance - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_process_store_vectors() {
        let dir = get_vstore_dir();
        let vstore_dir = dir.path();
        let rank = 1;
        let n = 5;

        // Create test vectors
        let vectors = generate_many_mock_embeddings(n);
        let mut vstore = get_local_vstore(vstore_dir, rank, true);

        // Store vectors
        let result = process_store_vectors(&mut vstore, &vectors, rank);
        assert!(result.is_ok());

        // Create a fresh storage instance to read back the data
        let count = vstore.get_count().unwrap();
        assert_eq!(count, n);
        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_merge_vector_stores() {
        let dir = get_vstore_dir();
        let vstore_dir = dir.path();
        let size = 3; // Number of ranks
        let n = 3;

        for rank in (0..size).into_iter() {
            let vectors_rank = generate_many_mock_embeddings(n);
            let mut local_vstore = get_local_vstore(vstore_dir, rank, true);
            process_store_vectors(&mut local_vstore, &vectors_rank, rank).unwrap();
            local_vstore.persist().unwrap();
        }

        // Merge the vector stores
        let global_vstore = merge_vector_stores(size, vstore_dir).unwrap();
        let count = global_vstore.get_count().unwrap();

        // Should have merged all vectors (3+4+2=9)
        assert_eq!(count, n * size as usize);
        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_merge_empty_vector_stores() {
        let dir = get_vstore_dir();
        let vstore_dir = dir.path();
        let size = 3; // Number of ranks

        // Merge the empty vector stores
        let global_vstore = merge_vector_stores(size, vstore_dir).unwrap();
        let count = global_vstore.get_count().unwrap();
        assert_eq!(count, 0);
        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_merge_with_missing_ranks() {
        let dir = get_vstore_dir();
        let vstore_dir = dir.path();
        let size = 3; // Number of ranks
        let n = 3;

        // Only create stores for ranks 0 and 2 (skip rank 1)
        for rank in [0, 2] {
            let vector_rank = generate_many_mock_embeddings(n);
            let mut local_vstore = get_local_vstore(vstore_dir, rank, true);
            process_store_vectors(&mut local_vstore, &vector_rank, rank).unwrap();
            local_vstore.persist().unwrap();
        }

        // Merge the vector stores
        let global_vstore = merge_vector_stores(size, vstore_dir).unwrap();
        let count: usize = global_vstore.get_count().unwrap();

        assert_eq!(count, n * 2);
        dir.close().unwrap_or_default();
    }

    #[test]
    fn test_merge_large_vectors() {
        let dir = get_vstore_dir();
        let vstore_dir = dir.path();
        let size = 2; // Number of ranks
        let n = 50;

        for rank in (0..size).into_iter() {
            let vector_rank = generate_many_mock_embeddings(n);
            let mut local_vstore = get_local_vstore(vstore_dir, rank, true);
            process_store_vectors(&mut local_vstore, &vector_rank, rank).unwrap();
            local_vstore.persist().unwrap();
        }

        // Merge the vector stores
        let global_vstore = merge_vector_stores(size, vstore_dir).unwrap();
        let count = global_vstore.get_count().unwrap();

        // Should have merged all vectors (50+50=100)
        assert_eq!(count, n * 2);
        dir.close().unwrap_or_default();
    }
}
