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
    benchmark::{BenchmarkManager, BenchmarkTimer, time_operation},
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
    
    // Initialize benchmark manager
    let mut benchmark_manager = BenchmarkManager::new(rank);
    let overall_timer = BenchmarkTimer::start("total_execution");

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

    // Time LLM service loading
    let llm_service = time_operation(&mut benchmark_manager, "llm_service_loading", None, || {
        llm::hf::HfService::default().expect(&generate_msg(rank, "Fail to load llm service"))
    });
    
    let mut local_vstore = get_local_vstore(vstore_dir, rank, !skip_process);

    // Time the embedding generation and processing
    if !skip_process {
        let embedding_timer = BenchmarkTimer::start("embedding_generation");
        
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
        
        // Record embedding generation timing with item count
        benchmark_manager.record(
            embedding_timer.stop().clone()
        );
        
        // Wait for all processes to finish writing their files
        world.barrier();

        // Step 5: Process 0 merges all storage files
        if is_root(rank) {
            let merge_timer = BenchmarkTimer::start("vector_store_merge");
            
            let mut merged_store = merge_vector_stores(size, vstore_dir)
                .expect(&generate_msg(rank, "Fail to merging vector stores"));
                
            let vector_count = merged_store.get_count().unwrap_or(0);
            
            merged_store.persist()
                .expect(&generate_msg(rank, "Fail to persist global vstore"));
                
            benchmark_manager.record(
                BenchmarkTimer {
                    name: merge_timer.name,
                    start: merge_timer.start,
                    items: Some(vector_count),
                }.stop()
            );
        }
    }
    world.barrier();

    // Step 6: Metrics calculation - Top-k similarity search
    let top_k = 50;
    let query_idx = 0; // Using the first vector as the query

    // Time the similarity search
    let search_timer = BenchmarkTimer::start("similarity_search");
    
    // Perform parallel top-k similarity search
    let mut target_vector = vec![0.0; 768]; // Placeholder for the target vector
    if is_root(rank) {
        println!("[Rank {}] Generating target vector", rank);
        target_vector = local_vstore.get(0).unwrap();
    }
    println!("[Rank {}] Broadcasting target vector", rank);
    world.process_at_rank(ROOT).broadcast_into(&mut target_vector);
    println!("[Rank {}] Target vector broadcasted", rank);
    println!("[Rank {}] Starting top-k similarity search", rank);

    // Perform the top-k similarity search
    let global_top_k = parallel_top_k_similarity_search(&world, rank, size, vstore_dir, top_k, &target_vector);
    
    // Record similarity search timing
    benchmark_manager.record(search_timer.stop());

    // Time the metrics calculation
    let metrics_timer = BenchmarkTimer::start("metrics_calculation");
    
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
    
    // Record metrics calculation timing
    benchmark_manager.record(metrics_timer.stop());
    
    // Record total execution time
    benchmark_manager.record(overall_timer.stop());

    // After all MPI operations are done, generate and print benchmark report
    world.barrier();
    
    if is_root(rank) {
        // Generate and print the benchmark report
        println!("\n\n");
        println!("{}", benchmark_manager.generate_report(&world, None));
    }

    world.barrier();
    println!("[Rank {}] Process completed, terminating", rank);
}