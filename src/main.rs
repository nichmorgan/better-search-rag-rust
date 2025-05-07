mod llm;
mod metrics;
mod mpi_helpers;
mod source;
mod utils;
mod vectorstore;

use std::{env, fs, path::Path};

use llm::LlmService;
use mpi::traits::*;

use mpi_helpers::{
    metrics::{calculate_accuracy_metrics, parallel_top_k_similarity_search, print_top_k_results},
    tasks::{merge_vector_stores, process_files_embeddings_chunked},
    vectorstore::get_local_vstore,
    *,
};

fn generate_msg(rank: i32, message: &str) -> String {
    format!("[Rank {}] {}", rank, message)
}

#[tokio::main]
async fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    println!("Process {} of {} initialized", rank, size);

    let extensions = ["java"];
    let dir = ".repos/jabref";
    let chunk_size = 32;
    let vstore_dir = Path::new(".volumes/vstore");
    let skip_process: bool = env::var("SKIP_PROCESS")
        .unwrap_or_default()
        .parse()
        .unwrap_or(false);

    // Create base directories if they don't exist
    if is_root(rank) {
        fs::create_dir_all(".volumes").unwrap_or_default();
    }

    let llm_service =
        llm::hf::HfService::default().expect(&generate_msg(rank, "Fail to load llm service"));
    let mut local_vstore = get_local_vstore(vstore_dir, rank, true);

    if !skip_process {
        process_files_embeddings_chunked(
            dir,
            &extensions,
            &world,
            rank,
            size,
            &llm_service,
            &mut local_vstore,
            chunk_size,
        )
        .unwrap_or_else(|e| {
            println!("[Rank {}] Processing error: {}", rank, e);
            0
        });
        // Wait for all processes to finish writing their files
        world.barrier();

        // Step 5: Process 0 merges all storage files
        if is_root(rank) {
            merge_vector_stores(size, vstore_dir)
                .expect(&generate_msg(rank, "Fail to merging vector stores"))
                .persist()
                .expect(&generate_msg(rank, "Fail to persist global vstore"));
        }
    }
    world.barrier();

    // Step 6: Metrics calculation - Top-k similarity search
    let top_k = 50;
    let query_idx = 0; // Using the first vector as the query

    // Perform parallel top-k similarity search
    let mut target_vector = vec![0.0; 768]; // Placeholder for the target vector
    if is_root(rank) {
        println!("[Rank {}] Generating target vector", rank);
        target_vector = llm_service.get_embeddings(vec!["Hello, world!".to_string()].as_ref()).unwrap()[0].clone();
    }
    println!("[Rank {}] Broadcasting target vector", rank);
    world.process_at_rank(ROOT).broadcast_into(&mut target_vector);
    println!("[Rank {}] Target vector broadcasted", rank);
    println!("[Rank {}] Starting top-k similarity search", rank);

    // Perform the top-k similarity search
    let global_top_k = parallel_top_k_similarity_search(&world, rank, size, vstore_dir, top_k, &target_vector);

    // Root process handles the results and metrics calculation
    if is_root(rank) {
        if let Some(top_k_results) = global_top_k {
            // Print results
            print_top_k_results(&top_k_results);

            // Calculate accuracy metrics
            let (mrr, recall, overlap) =
                calculate_accuracy_metrics(&top_k_results, query_idx, top_k);

            println!("Accuracy Metrics:");
            println!("  Mean Reciprocal Rank (MRR): {:.4}", mrr);
            println!("  Recall@{}: {:.4}", top_k, recall);
            println!("  Top-k Overlap: {:.4}", overlap);

            // Additional measurements could be added here:
            // - Execution time comparisons
            // - Speedup calculations
            // - Efficiency metrics
        } else {
            println!("Error: Failed to compute global top-k results");
        }
    }

    // After all MPI operations are done
    world.barrier();
    if is_root(rank) {
        println!("MPI operations completed successfully");
    }

    if is_root(rank) {
        println!("MPI operations completed successfully");
    }

    // Clean exit to avoid finalization issues
    mpi_finish(rank);
}
