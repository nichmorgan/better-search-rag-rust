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
    let chunk_size = 32;
    let vstore_dir = Path::new(".volumes/vstore");

    // Create base directories if they don't exist
    if is_root(rank) {
        fs::create_dir_all(".volumes").unwrap_or_default();
    }

    let llm_service =
        llm::hf::HfService::default().expect(&generate_msg(rank, "Fail to load llm service"));
    let mut local_vstore = get_local_vstore(vstore_dir, rank, true);
    let processed_count = process_files_embeddings_chunked(
        dir,
        &extensions,
        &world,
        rank,
        size,
        &llm_service,
        &mut local_vstore,
        chunk_size,
    ).unwrap_or_else(|e| {
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
