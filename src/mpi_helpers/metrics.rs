// mpi_helpers/metrics.rs

use mpi::collective::CommunicatorCollectives; // Import the trait for barrier
use mpi::traits::*;
use std::collections::HashSet;
use std::path::Path;

use crate::metrics::cosine_distance;
use crate::mpi_helpers::load_balance::interval_by_rank;
use crate::mpi_helpers::vectorstore::get_global_vstore;
use crate::mpi_helpers::{ROOT, is_root};
use crate::vectorstore::polars::SliceArgs;
use polars::error::PolarsError;

// Improved version of similarity_search that returns top-k local results
pub fn compute_local_top_k(
    vstore_dir: &Path,
    rank: i32,
    size: i32,
    top_k: usize,
    target_vector: &Vec<f32>,
) -> Result<Vec<(usize, f32)>, PolarsError> {
    let vstore = get_global_vstore(vstore_dir, false);
    let vstore_count = vstore.get_count()?;

    // Determine the portion of vectors this rank should process
    let rank_interval = interval_by_rank(rank, size, vstore_count);

    // Retrieve vectors for this rank
    let rank_vectors = vstore.get_many(Some(SliceArgs {
        offset: rank_interval.start_index as i32,
        length: rank_interval.get_count(),
    }))?;

    // Calculate distances between target vector and each vector in this rank's portion
    let mut distances: Vec<(usize, f32)> = rank_vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            // Convert local index to global index
            let global_idx = rank_interval.start_index + i;
            (global_idx, cosine_distance(v, &target_vector))
        })
        .collect();

    // Sort by distance (smallest first) and keep only top_k
    distances.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    if distances.len() > top_k {
        distances.truncate(top_k);
    }

    Ok(distances)
}

// Gather all local top-k results to root process
pub fn gather_top_k_results<C: Communicator>(
    world: &C,
    rank: i32,
    local_top_k: &Vec<(usize, f32)>,
) -> (Vec<usize>, Vec<f32>) {
    // Prepare local data
    let local_count = local_top_k.len();
    let local_indices: Vec<usize> = local_top_k.iter().map(|(idx, _)| *idx).collect();
    let local_distances: Vec<f32> = local_top_k.iter().map(|(_, dist)| *dist).collect();

    // Get counts from all processes
    let mut all_counts = vec![0; world.size() as usize];
    world.all_gather_into(&local_count, &mut all_counts);

    // Calculate total elements to be received
    let total_count: usize = all_counts.iter().sum();

    // Prepare buffers for gathering data
    let mut global_indices = if is_root(rank) {
        vec![0; total_count]
    } else {
        vec![]
    };
    let mut global_distances = if is_root(rank) {
        vec![0.0; total_count]
    } else {
        vec![]
    };

    // Create displacements for various buffer sizes
    let mut displacements = vec![0; world.size() as usize];
    for i in 1..world.size() as usize {
        displacements[i] = displacements[i - 1] + all_counts[i - 1];
    }

    // Gather data from all processes to root
    // Fixed approach using gatherv
    if is_root(rank) {
        for i in 0..world.size() {
            if i == rank {
                // Root process's own data - copy directly
                for j in 0..local_count {
                    global_indices[displacements[i as usize] + j] = local_indices[j];
                    global_distances[displacements[i as usize] + j] = local_distances[j];
                }
            } else {
                // Receive data from other processes
                let source = world.process_at_rank(i);
                let count = all_counts[i as usize];

                if count > 0 {
                    let idx_start = displacements[i as usize];

                    // Receive indices
                    let indices = source.receive_vec::<usize>().0;
                    for (j, &idx) in indices.iter().enumerate() {
                        if j < count {
                            global_indices[idx_start + j] = idx;
                        }
                    }

                    // Receive distances
                    let distances = source.receive_vec::<f32>().0;
                    for (j, &dist) in distances.iter().enumerate() {
                        if j < count {
                            global_distances[idx_start + j] = dist;
                        }
                    }
                }
            }
        }
    } else {
        // Non-root processes send their data
        let root = world.process_at_rank(ROOT);
        root.send(&local_indices[..]);
        root.send(&local_distances[..]);
    }

    // Ensure all communications are complete
    world.barrier();

    (global_indices, global_distances)
}

// Compute global top-k from gathered results
pub fn compute_global_top_k(
    global_indices: Vec<usize>,
    global_distances: Vec<f32>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    // Create pairs of (index, distance) for sorting
    let mut global_results: Vec<(usize, f32)> = global_indices
        .into_iter()
        .zip(global_distances.into_iter())
        .collect();

    // Sort by distance (smallest first)
    global_results.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

    // Truncate to get exactly top_k unique results
    let mut unique_results = Vec::with_capacity(top_k);
    let mut seen_indices = HashSet::new();

    for (idx, dist) in global_results {
        if !seen_indices.contains(&idx) {
            seen_indices.insert(idx);
            unique_results.push((idx, dist));

            if unique_results.len() >= top_k {
                break;
            }
        }
    }

    unique_results
}

// Top-k search coordinating function
pub fn parallel_top_k_similarity_search<C: Communicator>(
    world: &C,
    rank: i32,
    size: i32,
    vstore_dir: &Path,
    top_k: usize,
    target_vector: &Vec<f32>,
) -> Option<Vec<(usize, f32)>> {
    println!("[Rank {}] Calculating top-{} distances", rank, top_k);

    // Compute local top-k
    let local_top_k = match compute_local_top_k(vstore_dir, rank, size, top_k, target_vector) {
        Ok(results) => results,
        Err(err) => {
            println!("[Rank {}] Error in similarity search: {:?}", rank, err);
            vec![]
        }
    };

    // Gather results to root
    let (global_indices, global_distances) = gather_top_k_results(world, rank, &local_top_k);

    // Critical: Ensure all processes wait here with a barrier
    world.barrier();

    // Only root computes global top-k
    if is_root(rank) {
        let global_top_k = compute_global_top_k(global_indices, global_distances, top_k);
        Some(global_top_k)
    } else {
        None
    }
}

// Print top-k results
pub fn print_top_k_results(top_k_results: &Vec<(usize, f32)>) {
    println!("Global top-{} results:", top_k_results.len());
    for (i, (idx, dist)) in top_k_results.iter().enumerate() {
        println!("  {}. Index: {}, Distance: {}", i + 1, idx, dist);
    }
}

// Calculate accuracy metrics using global top-k results
pub fn calculate_accuracy_metrics(
    top_k_results: &Vec<(usize, f32)>,
    query_idx: usize,
    top_k: usize,
) -> (f32, f32, f32) {
    // Find position of query in results (if present)
    let mut position = 0;
    for (i, (idx, _)) in top_k_results.iter().enumerate() {
        if *idx == query_idx {
            position = i + 1;
            break;
        }
    }

    // Calculate MRR (Mean Reciprocal Rank)
    let mrr = if position > 0 {
        1.0 / position as f32
    } else {
        0.0
    };

    // Calculate Recall@k
    let recall = if position > 0 && position <= top_k {
        1.0
    } else {
        0.0
    };

    // For this single-query case, Top-k Overlap is binary (1.0 if found, 0.0 if not)
    let overlap = if position > 0 { 1.0 } else { 0.0 };

    (mrr, recall, overlap)
}
