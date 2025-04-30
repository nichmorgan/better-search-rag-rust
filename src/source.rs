use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

pub fn find_files_by_extensions(dir: &str, extensions: &[&str]) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map_or(false, |ext| extensions.contains(&ext))
        })
        .map(|entry| entry.path().to_path_buf())
        .collect()
}

pub fn read_file<P: AsRef<Path>>(path: P) -> Option<String> {
    // Try to open the file, return None if it fails
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return None,
    };

    // Check file size (10MB = 10 * 1024 * 1024 bytes)
    let metadata = match file.metadata() {
        Ok(m) => m,
        Err(_) => return None,
    };

    let file_size = metadata.len();
    if file_size > 10 * 1024 * 1024 {
        return None; // Skip files larger than 10MB
    }

    // Preallocate buffer with exact file size for efficiency
    let mut buffer = String::with_capacity(file_size as usize);

    // Read the file into the string
    match file.read_to_string(&mut buffer) {
        Ok(_) => Some(buffer),
        Err(_) => None,
    }
}
