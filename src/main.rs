mod llm;
mod mpi_helper;
mod source;
mod vectorstore;
mod utils;

use std::{fs, path::Path, time::Instant};

use mpi::traits::*;

use mpi_helper::*;

#[tokio::main]
async fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let root_process = world.process_at_rank(ROOT);

    println!("Process {} of {} initialized", rank, size);

    let extensions = ["py"];
    let dir = ".repos/jabref";
    let vstore_dir = Path::new(".volumes/vstore");

    // Create base directories if they don't exist
    if is_root(rank) {
        fs::create_dir_all(".volumes").unwrap_or_default();
    }

    let llm_service = llm::LlmService::default();
    let mut local_vstore = get_local_vstore(vstore_dir, rank, true);

    if is_root(rank) {
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
    root_process.broadcast_into(&mut dim);
    println!("[Rank {}] Received dimension: {}", rank, dim);

    if dim == 0 {
        println!(
            "[Rank {}] No valid embeddings found across processes.",
            rank
        );
        world.barrier();
        mpi_finish(rank);
    }

    // Step 4: Each process stores its vectors
    if !embeddings.is_empty() {
        let result = process_store_vectors(&mut local_vstore, &embeddings, rank);
        if let Err(e) = result {
            println!("[Rank {}] Error in vector storage: {}", rank, e);
        }
        local_vstore
            .persist()
            .expect(&format!("[Rank {}] Fail to persist local vstore", rank));
    } else {
        println!("[Rank {}] No embeddings to store", rank);
    }

    // Wait for all processes to finish writing their files
    world.barrier();

    // Step 5: Process 0 merges all storage files
    if is_root(rank) {
        let mut global_vstore = merge_vector_stores(size, vstore_dir).expect("Fail to merging vector stores");
        global_vstore.persist().expect("Fail to persist global vstore");
    }

    // Step 6: Metrics
    let _target_vector_index = 0;

    // After all MPI operations are done
    world.barrier();
    if is_root(rank) {
        println!("MPI operations completed successfully");
    }

    // Clean exit to avoid finalization issues
    mpi_finish(rank);
}
