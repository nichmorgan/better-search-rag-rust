mod llm;
mod mpi_helper;
mod source;
mod vectorstore;

use std::{fs, time::Instant};

use mpi::traits::*;

use mpi_helper::*;

#[tokio::main]
async fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    println!("Process {} of {} initialized", rank, size);

    let extensions = ["py"];
    let dir = ".repos/jabref";
    let vstore_path = ".volumes/vstore";
    let chunk_size = 512;

    // Create base directories if they don't exist
    if rank == 0 {
        fs::create_dir_all(".volumes").unwrap_or_default();
    }

    let llm_service = llm::LlmService::default();
    if rank == 0 {
        llm_service.check_models().await;
    }
    world.barrier();

    // Step 1: Read files distributed across processes
    let start = Instant::now();
    let rank_contents = read_files(dir, &extensions, &world, rank, size);
    let elapsed = Instant::now() - start;
    println!(
        "[Rank {}] read {} files in {} seconds",
        rank,
        rank_contents.len(),
        elapsed.as_secs()
    );

    // Step 2: Generate embeddings in parallel
    let start = Instant::now();
    let embeddings = if !rank_contents.is_empty() {
        llm_service
            .get_embeddings(&rank_contents)
            .await
            .expect("Failed to embed")
    } else {
        Vec::new()
    };
    let elapsed = Instant::now() - start;
    println!(
        "[Rank {}] embedded {} files in {} seconds",
        rank,
        embeddings.len(),
        elapsed.as_secs()
    );

    // Step 3: Get embedding dimensions and coordinate across processes
    let mut dim = if !embeddings.is_empty() {
        embeddings[0].len()
    } else {
        0
    };

    println!("[Rank {}] Local dimension: {}", rank, dim);

    // FIX: Use a root broadcast approach for gathering dimensions
    world.process_at_rank(0).broadcast_into(&mut dim);
    println!("[Rank {}] Received dimension: {}", rank, dim);

    if dim == 0 {
        println!(
            "[Rank {}] No valid embeddings found across processes.",
            rank
        );
        world.barrier();
        std::process::exit(0);
    }

    // Step 4: Each process stores its vectors
    if !embeddings.is_empty() {
        let result = process_store_vectors(&embeddings, vstore_path, rank, dim, chunk_size);
        if let Err(e) = result {
            println!("[Rank {}] Error in vector storage: {}", rank, e);
        }
    } else {
        println!("[Rank {}] No embeddings to store", rank);
    }

    // Wait for all processes to finish writing their files
    world.barrier();

    // Step 5: Process 0 merges all storage files
    if rank == 0 {
        let result = merge_vector_stores(&world, vstore_path, dim, chunk_size);
        if let Err(e) = result {
            println!("Error merging vector stores: {}", e);
        }
    }

    // After all MPI operations are done
    world.barrier();
    if rank == 0 {
        println!("MPI operations completed successfully");
    }

    // Clean exit to avoid finalization issues
    std::process::exit(0);
}

