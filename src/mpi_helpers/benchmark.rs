// src/mpi_helpers/benchmark.rs
use std::collections::HashMap;
use std::time::{Duration, Instant};

use mpi::collective::CommunicatorCollectives;
use mpi::traits::*;

use super::is_root;

/// Represents a benchmark timing for a specific operation
#[derive(Clone, Debug)]
pub struct BenchmarkTiming {
    /// Operation name
    pub name: String,
    /// Duration of the operation
    pub duration: Duration,
    /// Number of items processed (if applicable)
    pub items_processed: Option<usize>,
}

impl BenchmarkTiming {
    /// Create a new benchmark timing with the given name, duration, and optional items processed
    pub fn new(name: &str, duration: Duration, items_processed: Option<usize>) -> Self {
        Self {
            name: name.to_string(),
            duration,
            items_processed,
        }
    }

    /// Get the duration in seconds as a float
    pub fn seconds(&self) -> f64 {
        self.duration.as_secs_f64()
    }

    /// Get the throughput (items per second) if items_processed is available
    pub fn throughput(&self) -> Option<f64> {
        self.items_processed
            .map(|items| items as f64 / self.seconds())
    }
}

/// A timer for measuring operation durations
pub struct BenchmarkTimer {
    /// Operation name
    pub name: String,
    /// Start time
    pub start: Instant,
    /// Items to be processed (if applicable)
    pub items: Option<usize>,
}

impl BenchmarkTimer {
    /// Start a new timer for the given operation
    pub fn start(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            items: None,
        }
    }

    /// Start a new timer with the number of items to be processed
    pub fn start_with_items(name: &str, items: usize) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            items: Some(items),
        }
    }

    /// Stop the timer and return a BenchmarkTiming
    pub fn stop(self) -> BenchmarkTiming {
        let duration = self.start.elapsed();
        BenchmarkTiming::new(&self.name, duration, self.items)
    }
}

/// Manages benchmark timings for a process
pub struct BenchmarkManager {
    /// MPI rank of this process
    rank: i32,
    /// Map of timings by operation name
    timings: HashMap<String, BenchmarkTiming>,
}

impl BenchmarkManager {
    /// Create a new benchmark manager for the given rank
    pub fn new(rank: i32) -> Self {
        Self {
            rank,
            timings: HashMap::new(),
        }
    }

    /// Record a timing
    pub fn record(&mut self, timing: BenchmarkTiming) {
        self.timings.insert(timing.name.clone(), timing);
    }

    /// Get a timing by name
    pub fn get(&self, name: &str) -> Option<&BenchmarkTiming> {
        self.timings.get(name)
    }

    /// Get all timings
    pub fn all_timings(&self) -> Vec<BenchmarkTiming> {
        self.timings.values().cloned().collect()
    }

    /// Convert benchmark data to MPI-compatible format
    fn prepare_timing_data(&self) -> (Vec<Vec<u8>>, Vec<f64>, Vec<i32>) {
        let timings = self.all_timings();
        let mut names: Vec<Vec<u8>> = Vec::with_capacity(timings.len());
        let mut seconds: Vec<f64> = Vec::with_capacity(timings.len());
        let mut item_counts: Vec<i32> = Vec::with_capacity(timings.len());

        for timing in timings {
            names.push(timing.clone().name.bytes().map(|b| b).collect::<Vec<u8>>());
            seconds.push(timing.clone().seconds());
            item_counts.push(match &timing.items_processed {
                Some(count) => count.clone() as i32,
                None => -1, // Use -1 to indicate no items count
            });
        }

        (names, seconds, item_counts)
    }

    /// Gather all timing data from all processes to the root process
    pub fn gather_timings<C: Communicator>(&self, world: &C) -> HashMap<i32, Vec<BenchmarkTiming>> {
        println!("[Rank {}] Entering gather_timings", self.rank);
        let mut results = HashMap::new();

        // Add local timings to results
        if is_root(self.rank) {
            println!("[Rank {}] Root adding local timings", self.rank);
            results.insert(self.rank, self.all_timings());
        }

        // Number of timing entries on this process
        let local_count = self.timings.len() as i32;
        println!("[Rank {}] Local timing count: {}", self.rank, local_count);

        // Root needs to know how many entries to expect from each process
        println!("[Rank {}] Before all_gather_into for counts", self.rank);
        let mut all_counts = vec![0; world.size() as usize];
        world.all_gather_into(&local_count, &mut all_counts);
        println!("[Rank {}] After all_gather_into for counts", self.rank);

        if is_root(self.rank) {
            println!("[Rank {}] Root receiving data from other ranks", self.rank);
            // Root process gathers timing data from all processes
            for src_rank in 1..world.size() {
                let count = all_counts[src_rank as usize];
                println!(
                    "[Rank {}] Expecting {} timings from rank {}",
                    self.rank, count, src_rank
                );

                if count > 0 {
                    let process = world.process_at_rank(src_rank);
                    println!("[Rank {}] Receiving data from rank {}", self.rank, src_rank);

                    // Receive names, durations, and item counts separately
                    println!(
                        "[Rank {}] Receiving count from rank {}",
                        self.rank, src_rank
                    );
                    let (rank_count, _) = process.receive::<i32>();
                    println!(
                        "[Rank {}] Received count {} from rank {}",
                        self.rank, rank_count, src_rank
                    );

                    println!(
                        "[Rank {}] Receiving names from rank {}",
                        self.rank, src_rank
                    );
                    let names = (0..rank_count)
                        .map(|i| {
                            println!(
                                "[Rank {}] Receiving name {} of {} from rank {}",
                                self.rank,
                                i + 1,
                                rank_count,
                                src_rank
                            );
                            String::from_utf8(process.receive_vec::<u8>().0)
                                .unwrap_or_else(|_| "Invalid UTF-8".to_string())
                        })
                        .collect::<Vec<_>>();
                    println!(
                        "[Rank {}] Received all names from rank {}",
                        self.rank, src_rank
                    );

                    println!(
                        "[Rank {}] Receiving seconds from rank {}",
                        self.rank, src_rank
                    );
                    let (seconds, _) = process.receive_vec::<f64>();
                    println!(
                        "[Rank {}] Received seconds from rank {}",
                        self.rank, src_rank
                    );

                    println!(
                        "[Rank {}] Receiving item counts from rank {}",
                        self.rank, src_rank
                    );
                    let (item_counts, _) = process.receive_vec::<i32>();
                    println!(
                        "[Rank {}] Received item counts from rank {}",
                        self.rank, src_rank
                    );

                    // Reconstruct BenchmarkTiming objects
                    let timings: Vec<BenchmarkTiming> = names
                        .iter()
                        .zip(seconds.iter())
                        .zip(item_counts.iter())
                        .map(|((name, &sec), &items)| {
                            BenchmarkTiming::new(
                                name,
                                Duration::from_secs_f64(sec),
                                if items >= 0 {
                                    Some(items as usize)
                                } else {
                                    None
                                },
                            )
                        })
                        .collect();

                    results.insert(src_rank, timings);
                    println!(
                        "[Rank {}] Completed processing data from rank {}",
                        self.rank, src_rank
                    );
                }
            }
            println!(
                "[Rank {}] Root completed receiving data from all ranks",
                self.rank
            );
        } else {
            // Non-root processes send their data if they have any
            if local_count > 0 {
                println!("[Rank {}] Non-root sending data to root", self.rank);
                let root = world.process_at_rank(0);
                let (names, seconds, item_counts) = self.prepare_timing_data();

                // Send each vector separately
                println!("[Rank {}] Sending local_count: {}", self.rank, local_count);
                root.send(&local_count);

                println!("[Rank {}] Sending {} names", self.rank, names.len());
                names.iter().for_each(|name| {
                    root.send(name);
                });

                println!(
                    "[Rank {}] Sending {} seconds values",
                    self.rank,
                    seconds.len()
                );
                root.send(&seconds[..]);

                println!(
                    "[Rank {}] Sending {} item counts",
                    self.rank,
                    item_counts.len()
                );
                root.send(&item_counts[..]);

                println!("[Rank {}] Completed sending all data to root", self.rank);
            } else {
                println!("[Rank {}] No timing data to send", self.rank);
            }
        }

        // Add a barrier to ensure all ranks complete this function
        println!(
            "[Rank {}] Before final barrier in gather_timings",
            self.rank
        );
        world.barrier();
        println!("[Rank {}] After final barrier in gather_timings", self.rank);

        println!("[Rank {}] Exiting gather_timings", self.rank);
        results
    }

    /// Generate a performance report from all timings
    pub fn generate_report<C: Communicator>(
        &self,
        world: &C,
        sequential_times: Option<HashMap<String, f64>>,
    ) -> String {
        println!("[Rank {}] Entering generate_report", self.rank);

        // Log the start of timing data gathering
        println!("[Rank {}] Starting to gather benchmark data", self.rank);
        let all_timings = self.gather_timings(world);
        println!("[Rank {}] Completed gathering benchmark data", self.rank);

        let mut report = String::new();

        // Only root process generates the actual report text
        if is_root(self.rank) {
            println!("[Rank {}] Root is generating report text", self.rank);
            report.push_str("==== PARALLEL PERFORMANCE REPORT ====\n\n");

            // Collect all unique operation names
            let mut operation_names = std::collections::HashSet::new();
            for (_, rank_timings) in &all_timings {
                for timing in rank_timings {
                    operation_names.insert(timing.name.clone());
                }
            }

            // Sort operation names for consistent output
            let mut sorted_names: Vec<String> = operation_names.into_iter().collect();
            sorted_names.sort();

            println!(
                "[Rank {}] Processing {} operations",
                self.rank,
                sorted_names.len()
            );

            // Generate report for each operation
            for (idx, op_name) in sorted_names.iter().enumerate() {
                println!(
                    "[Rank {}] Processing operation {}/{}: {}",
                    self.rank,
                    idx + 1,
                    sorted_names.len(),
                    op_name
                );

                report.push_str(&format!("Operation: {}\n", op_name));

                // Collect timings for this operation across all ranks
                let mut op_timings = Vec::new();
                for (&rank, rank_timings) in &all_timings {
                    for timing in rank_timings {
                        if timing.name == *op_name {
                            op_timings.push((rank, timing));
                        }
                    }
                }

                if op_timings.is_empty() {
                    report.push_str("  No timing data available\n\n");
                    continue;
                }

                // Calculate statistics
                let total_time: f64 = op_timings.iter().map(|(_, t)| t.seconds()).sum();
                let avg_time: f64 = total_time / op_timings.len() as f64;
                let min_time: f64 = op_timings
                    .iter()
                    .map(|(_, t)| t.seconds())
                    .fold(f64::INFINITY, |a, b| a.min(b));
                let max_time: f64 = op_timings
                    .iter()
                    .map(|(_, t)| t.seconds())
                    .fold(0.0, |a, b| a.max(b));

                // Find the rank with min and max time
                let min_rank = op_timings
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        a.seconds()
                            .partial_cmp(&b.seconds())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(r, _)| *r)
                    .unwrap_or(0);

                let max_rank = op_timings
                    .iter()
                    .max_by(|(_, a), (_, b)| {
                        a.seconds()
                            .partial_cmp(&b.seconds())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(r, _)| *r)
                    .unwrap_or(0);

                // Calculate speedup if sequential time is provided
                let speedup = if let Some(ref seq_times) = sequential_times {
                    if let Some(seq_time) = seq_times.get(op_name) {
                        Some(*seq_time / min_time)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Report summary statistics
                report.push_str(&format!(
                    "  Min time: {:.4} sec (Rank {})\n",
                    min_time, min_rank
                ));
                report.push_str(&format!(
                    "  Max time: {:.4} sec (Rank {})\n",
                    max_time, max_rank
                ));
                report.push_str(&format!("  Avg time: {:.4} sec\n", avg_time));
                if let Some(speedup_val) = speedup {
                    report.push_str(&format!("  Speedup: {:.2}x\n", speedup_val));
                }

                // Add more report generation logic...
                // (rest of the reporting code remains the same)
            }

            println!("[Rank {}] Completed generating report text", self.rank);
        } else {
            println!("[Rank {}] Non-root process in generate_report", self.rank);
        }

        // This synchronization is important to ensure all processes complete this function
        println!(
            "[Rank {}] Calling barrier before exiting generate_report",
            self.rank
        );
        world.barrier();
        println!("[Rank {}] Passed barrier in generate_report", self.rank);

        // All ranks return something, but only root has meaningful content
        println!("[Rank {}] Exiting generate_report", self.rank);
        report
    }
}

/// Convenience function to update the benchmark manager with a timed operation
pub fn time_operation<F, T>(
    manager: &mut BenchmarkManager,
    name: &str,
    items: Option<usize>,
    operation: F,
) -> T
where
    F: FnOnce() -> T,
{
    // Create timer with appropriate constructor based on whether items is provided
    let timer = if let Some(count) = items {
        BenchmarkTimer::start_with_items(name, count)
    } else {
        BenchmarkTimer::start(name)
    };

    // Execute the operation
    let result = operation();

    // Stop the timer and record the timing
    let timing = timer.stop();
    manager.record(timing);

    result
}

/// Convenience function to create a formatted time string from seconds
pub fn format_time(seconds: f64) -> String {
    if seconds < 0.001 {
        format!("{:.2} μs", seconds * 1_000_000.0)
    } else if seconds < 1.0 {
        format!("{:.2} ms", seconds * 1_000.0)
    } else if seconds < 60.0 {
        format!("{:.2} sec", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let secs = seconds - (minutes * 60.0);
        format!("{:.0}m {:.2}s", minutes, secs)
    } else {
        let hours = (seconds / 3600.0).floor();
        let minutes = ((seconds - (hours * 3600.0)) / 60.0).floor();
        let secs = seconds - (hours * 3600.0) - (minutes * 60.0);
        format!("{:.0}h {:.0}m {:.2}s", hours, minutes, secs)
    }
}
