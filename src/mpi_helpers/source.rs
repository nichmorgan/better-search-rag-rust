use crate::{
    mpi_helpers::{is_root, load_balance::slice_by_rank},
    source,
};

use mpi::traits::*;

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
