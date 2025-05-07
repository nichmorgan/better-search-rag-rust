use std::path::Path;

use mpi::traits::Communicator;
use mpi::collective::CommunicatorCollectives; // Import the trait for barrier

use crate::{
    llm::LlmService,
    mpi_helpers::{
        is_root,
        load_balance::slice_by_rank,
        vectorstore::{get_global_vstore, get_local_vstore},
    },
    source,
    vectorstore::polars::PolarsVectorstore,
};

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
