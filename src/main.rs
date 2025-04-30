mod llm;
mod source;
mod vectorstore;

use std::{ops::Mul, time::Instant};

use mpi::traits::*;
use source::read_file;

#[tokio::main]
async fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    println!("Process {} of {} initialized", rank, size);

    let llm_service = llm::LlmService::default();
    if rank == 0 {
        llm_service.check_models().await;
    }
    world.barrier();

    let extensions = ["py"];
    let files = source::find_files_by_extensions(".repos/jabref", &extensions);

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
        let start = Instant::now();
        let contents: Vec<String> = rank_files.iter().map(read_file).flatten().collect();
        let elapsed = Instant::now() - start;
        println!(
            "[Rank {}] readed {}/{} files in {} seconds",
            rank,
            contents.len(),
            rank_files.len(),
            elapsed.as_secs()
        );

        let start = Instant::now();
        let embeddings = llm_service
            .get_embeddings(&contents)
            .await
            .expect("Fail to embed");
        let elapsed = Instant::now() - start;
        println!(
            "[Rank {}] embed {} files in {} seconds",
            rank,
            embeddings.len(),
            elapsed.as_secs()
        );
    }

    // After all MPI operations are done but before finalization
    world.barrier();
    if rank == 0 {
        println!("MPI operations completed successfully");
    }
    // Force exit to avoid finalization issues
    std::process::exit(0);
}
