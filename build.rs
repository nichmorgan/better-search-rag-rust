use std::env;
use std::path::PathBuf;

fn main() {
    let current_dir = env::current_dir().unwrap();
    let venv_dir = current_dir.join(".venv");
    
    // Tell cargo to look for libraries in .venv/lib
    println!("cargo:rustc-link-search=native={}", venv_dir.join("lib").display());
    
    // Tell cargo to tell the linker to add .venv/lib to rpath
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", venv_dir.join("lib").display());

    // Force rebuild if any of these environment variables change
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=OMPI_CC");
    println!("cargo:rerun-if-env-changed=OMPI_CXX");
    
    // Always rerun if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}