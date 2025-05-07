pub mod load_balance;
pub mod metrics;
pub mod source;
pub mod tasks;
pub mod vectorstore;

pub const ROOT: i32 = 0;

pub fn is_root(rank: i32) -> bool {
    ROOT == rank
}

pub fn mpi_finish(rank: i32) {
    println!("[Rank {}] Finished", rank);
    std::process::exit(0);
}
