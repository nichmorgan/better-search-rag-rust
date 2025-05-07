use std::path::Path;

use crate::vectorstore::polars::PolarsVectorstore;

pub fn get_local_vstore(vstore_dir: &Path, rank: i32, empty: bool) -> PolarsVectorstore {
    PolarsVectorstore::new(
        vstore_dir
            .join(format!("rank_{}.parquet", rank))
            .to_str()
            .unwrap(),
        empty,
    )
}

// In src/mpi_helper.rs
pub fn get_global_vstore(dir: &Path, empty: bool) -> PolarsVectorstore {
    // Create a consistent file path by always using "global.parquet"
    let file_path = dir.join("global.parquet").to_str().unwrap().to_string();
    PolarsVectorstore::new(&file_path, empty)
}
