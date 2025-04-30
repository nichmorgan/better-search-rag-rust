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


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_find_files_by_extensions() {
        // Create a temporary directory structure
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let base_path = temp_dir.path();
        
        // Create subdirectories
        let sub_dir1 = base_path.join("subdir1");
        let sub_dir2 = base_path.join("subdir2");
        fs::create_dir(&sub_dir1).expect("Failed to create subdir1");
        fs::create_dir(&sub_dir2).expect("Failed to create subdir2");
        
        // Create test files with different extensions
        let files = [
            (base_path.join("test1.txt"), "test1 content"),
            (base_path.join("test2.rs"), "test2 content"),
            (sub_dir1.join("test3.txt"), "test3 content"),
            (sub_dir1.join("test4.md"), "test4 content"),
            (sub_dir2.join("test5.rs"), "test5 content"),
            (sub_dir2.join("test6"), "test6 content"),  // No extension
        ];
        
        for (path, content) in &files {
            let mut file = File::create(path).expect("Failed to create test file");
            file.write_all(content.as_bytes()).expect("Failed to write to file");
        }
        
        // Test finding .txt files
        let txt_files = find_files_by_extensions(base_path.to_str().unwrap(), &["txt"]);
        assert_eq!(txt_files.len(), 2);
        assert!(txt_files.iter().any(|p| p.file_name().unwrap() == "test1.txt"));
        assert!(txt_files.iter().any(|p| p.file_name().unwrap() == "test3.txt"));
        
        // Test finding .rs files
        let rs_files = find_files_by_extensions(base_path.to_str().unwrap(), &["rs"]);
        assert_eq!(rs_files.len(), 2);
        assert!(rs_files.iter().any(|p| p.file_name().unwrap() == "test2.rs"));
        assert!(rs_files.iter().any(|p| p.file_name().unwrap() == "test5.rs"));
        
        // Test finding multiple extensions
        let multi_ext_files = find_files_by_extensions(base_path.to_str().unwrap(), &["txt", "md"]);
        assert_eq!(multi_ext_files.len(), 3);
        assert!(multi_ext_files.iter().any(|p| p.file_name().unwrap() == "test1.txt"));
        assert!(multi_ext_files.iter().any(|p| p.file_name().unwrap() == "test3.txt"));
        assert!(multi_ext_files.iter().any(|p| p.file_name().unwrap() == "test4.md"));
        
        // Test finding no matching extensions
        let no_match_files = find_files_by_extensions(base_path.to_str().unwrap(), &["pdf"]);
        assert_eq!(no_match_files.len(), 0);
        
        // Test with non-existent directory
        let non_existent = find_files_by_extensions("non_existent_dir", &["txt"]);
        assert_eq!(non_existent.len(), 0);
    }

    #[test]
    fn test_read_file() {
        // Create a temporary directory
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test_read.txt");
        
        // Create and write to test file
        let content = "This is test content for read_file";
        {
            let mut file = File::create(&file_path).expect("Failed to create test file");
            file.write_all(content.as_bytes()).expect("Failed to write to file");
        }
        
        // Test reading an existing file
        let read_content = read_file(&file_path);
        assert_eq!(read_content, Some(content.to_string()));
        
        // Test reading a non-existent file
        let non_existent_file = temp_dir.path().join("non_existent.txt");
        let non_existent_content = read_file(&non_existent_file);
        assert_eq!(non_existent_content, None);
    }

    #[test]
    fn test_read_file_size_limit() {
        // Create a temporary directory
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("large_file.txt");
        
        // Create a file larger than 10MB
        let large_content = "X".repeat(11 * 1024 * 1024); // 11MB
        {
            let mut file = File::create(&file_path).expect("Failed to create large test file");
            file.write_all(large_content.as_bytes()).expect("Failed to write to large file");
        }
        
        // Test reading a file larger than the size limit
        let read_large_content = read_file(&file_path);
        assert_eq!(read_large_content, None);
    }

    #[test]
    fn test_read_file_with_unicode() {
        // Create a temporary directory
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("unicode.txt");
        
        // Create and write unicode content to test file
        let unicode_content = "Hello, 世界! Привет, мир! 👋";
        {
            let mut file = File::create(&file_path).expect("Failed to create unicode test file");
            file.write_all(unicode_content.as_bytes()).expect("Failed to write unicode to file");
        }
        
        // Test reading unicode content
        let read_content = read_file(&file_path);
        assert_eq!(read_content, Some(unicode_content.to_string()));
    }

    #[test]
    fn test_find_files_empty_directory() {
        // Create an empty temporary directory
        let temp_dir = tempdir().expect("Failed to create temp directory");
        
        // Test finding files in an empty directory
        let files = find_files_by_extensions(temp_dir.path().to_str().unwrap(), &["txt"]);
        assert_eq!(files.len(), 0);
    }

    #[test]
    fn test_find_files_empty_extensions() {
        // Create a temporary directory
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");
        
        // Create a test file
        {
            let mut file = File::create(&file_path).expect("Failed to create test file");
            file.write_all(b"test content").expect("Failed to write to file");
        }
        
        // Test finding files with empty extensions list
        let files = find_files_by_extensions(temp_dir.path().to_str().unwrap(), &[]);
        assert_eq!(files.len(), 0);
    }
}
