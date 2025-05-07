pub mod benchmark;
pub mod load_balance;
pub mod metrics;
pub mod source;
pub mod tasks;
pub mod vectorstore;

pub const ROOT: i32 = 0;

pub fn is_root(rank: i32) -> bool {
    ROOT == rank
}
