mod llm;
mod source;
mod vectorstore;

use std::{ops::Mul, time::Instant};

use mpi::traits::*;
use ndarray::Array1;
use source::read_file;
use vectorstore::{arrow::ArrowVectorStorage, VectorStorage};

fn read_files<C: Communicator>(dir: &str, extensions: &[&str], world: &C, rank: i32, size: i32) -> Vec<String> {
    let files = source::find_files_by_extensions(dir, &extensions);
    let mut rank_contents = Vec::new();

    if rank == 0 {
        println!(
            "Found {} files filtered by extensions: {:?}",
            files.len(),
            extensions
        );
    }
    world.barrier();

    if rank as usize > files.len() - 1 {
        println!("[Rank {}] No files to me.", rank);
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
            start_index + files_per_rank
        };

        let rank_files = if rank as usize > files.len() {
            &[]
        } else {
            &files[start_index..end_index]
        };
        rank_contents.extend(rank_files.iter().map(read_file).flatten());
    }

    rank_contents
}

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
    let vstore_dim = 512;
    let chunk_size = 512;
    let reset = true;

    let llm_service = llm::LlmService::default();
    ArrowVectorStorage::create_storage(vstore_path, vstore_dim, chunk_size, reset).expect("Fail to create vstore");

    if rank == 0 {
        llm_service.check_models().await;
    }
    world.barrier();

    
    let start = Instant::now();
    let rank_contents = read_files(dir, &extensions, &world, rank, size);
    let elapsed = Instant::now() - start;
    println!(
        "[Rank {}] readed {} files in {} seconds",
        rank,
        rank_contents.len(),
        elapsed.as_secs()
    );

    let start = Instant::now();
    let embeddings = llm_service.get_embeddings(&rank_contents).await.expect("Fail to embed");
    let elapsed = Instant::now() - start;
    println!(
        "[Rank {}] embed {} files in {} seconds",
        rank,
        embeddings.len(),
        elapsed.as_secs()
    );

    let start = Instant::now();
    embeddings.iter().for_each(|vector| {
        ArrowVectorStorage::append_vector(vstore_path, &Array1::from_vec(vector.to_vec())).expect("Fail to append vector");
    });
    let elapsed = Instant::now() - start;
    println!(
        "[Rank {}] saved {} vectors in {} seconds",
        rank,
        embeddings.len(),
        elapsed.as_secs()
    );

    // After all MPI operations are done but before finalization
    world.barrier();
    if rank == 0 {
        println!("MPI operations completed successfully");
    }
    // Force exit to avoid finalization issues
    std::process::exit(0);
}
