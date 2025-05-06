mod llm;
mod metrics;
mod mpi_helper;
mod source;
mod utils;
mod vectorstore;

use std::{fs, path::Path, time::Instant};

use llm::LlmService;
use mpi::{collective::SystemOperation, traits::*};

use mpi_helper::*;

fn generate_msg(rank: i32, message: &str) -> String {
    format!("[Rank {}] {}", rank, message)
}

#[tokio::main]
async fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let root_process = world.process_at_rank(ROOT);

    println!("Process {} of {} initialized", rank, size);

    let extensions = ["java"];
    let dir = ".repos/jabref";
    let vstore_dir = Path::new(".volumes/vstore");

    // Create base directories if they don't exist
    if is_root(rank) {
        fs::create_dir_all(".volumes").unwrap_or_default();
    }

    let llm_service =
        llm::hf::HfService::default().expect(&generate_msg(rank, "Fail to load llm service"));
    let mut local_vstore = get_local_vstore(vstore_dir, rank, true);

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
            .expect(&generate_msg(rank, "Fail to persist local vstore"));
    } else {
        println!("[Rank {}] No embeddings to store", rank);
    }

    // Wait for all processes to finish writing their files
    world.barrier();

    // Step 5: Process 0 merges all storage files
    if is_root(rank) {
        merge_vector_stores(size, vstore_dir)
            .expect(&generate_msg(rank, "Fail to merging vector stores"))
            .persist()
            .expect(&generate_msg(rank, "Fail to persist global vstore"));
    }

    // Step 6: Metrics
    let top_k = 50;
    let (local_distance_indexes, local_distances): (Vec<_>, Vec<_>) =
        similarity_search(vstore_dir, rank, size, top_k)
            .expect(&generate_msg(rank, "Fail to calculate similarity search"))
            .iter()
            .cloned()
            .unzip();
    println!("[Rank {}] distances: {:?}", rank, local_distances);

    let mut total_distances: usize = 0;
    world.all_reduce_into(
        &local_distances.len(),
        &mut total_distances,
        SystemOperation::sum(),
    );

    let mut global_distance_indexes = vec![0usize; total_distances];
    let mut global_distances = vec![0f32; total_distances];
    world.all_gather_into(&local_distance_indexes, &mut global_distance_indexes[..]);
    world.all_gather_into(&local_distances, &mut global_distances[..]);

    if is_root(rank) {
        let mut global_distances: Vec<(&usize, &f32)> = global_distance_indexes
            .iter()
            .zip(global_distances.iter())
            .collect();
        global_distances.sort_by(|(_, a), (_, b)| b.partial_cmp(&a).unwrap());

        let mut global_index_topk: Vec<&usize> = vec![];
        let mut global_topk: Vec<(&usize, &f32)> = vec![];
        for (idx, dist) in global_distances.iter().cloned() {
            if global_index_topk.len() == top_k {
                break;
            }
            if global_index_topk.contains(&idx) {
                continue;
            }
            global_index_topk.push(idx);
            global_topk.push((idx, dist));
        }

        println!("global_topk: {:?}", global_topk);
    }

    // After all MPI operations are done
    world.barrier();
    if is_root(rank) {
        println!("MPI operations completed successfully");
    }

    // Clean exit to avoid finalization issues
    mpi_finish(rank);
}
