use crate::{
    llm::LlmService,
    source,
    vectorstore::polars::{PolarsVectorstore, SliceArgs},
};
use std::{ops::Mul, path::Path};

use mpi::traits::*;
use polars::error::PolarsError;

use super::is_root;

pub struct RankInterval {
    pub start_index: usize,
    pub end_index: usize,
}

impl RankInterval {
    pub fn get_count(&self) -> usize {
        self.end_index - self.start_index
    }
}

pub struct RankSlice<T> {
    pub interval: RankInterval,
    pub slice: Vec<T>,
}

pub fn interval_by_rank(rank: i32, size: i32, count: usize) -> RankInterval {
    let per_rank = if size as usize > count {
        1
    } else {
        count.div_ceil(size as usize)
    };

    let start_index = per_rank.mul(rank as usize);
    let end_index = if rank == size - 1 {
        count
    } else {
        std::cmp::min(start_index + per_rank, count)
    };

    RankInterval {
        start_index,
        end_index,
    }
}

pub fn slice_by_rank<T: Clone>(rank: i32, size: i32, arr: &Vec<T>) -> RankSlice<T> {
    let interval = interval_by_rank(rank, size, arr.len());
    let slice = arr[interval.start_index..interval.end_index].to_vec();

    RankSlice { interval, slice }
}

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
