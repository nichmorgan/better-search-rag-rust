use std::env;
use std::path::PathBuf;

fn main() {
    // Get the current directory
    let current_dir = env::current_dir().unwrap();
    println!("Current directory: {}", current_dir.display());
    
    // Create absolute path to .venv
    let venv_dir = current_dir.join(".venv");
    println!("Venv directory: {}", venv_dir.display());
    
    // Check if the directory exists
    if !venv_dir.exists() {
        println!("cargo:warning=.venv directory not found at {}", venv_dir.display());
    } else {
        println!("cargo:rustc-link-search=native={}", venv_dir.join("lib").display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", venv_dir.join("lib").display());
    }
    
    // Force rebuild on environment changes
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=OMPI_CC");
    println!("cargo:rerun-if-env-changed=OMPI_CXX");
    
    // Always rerun if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}