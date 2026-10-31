//! Integration test: confirms the OS correctly provides a virtual
//! memory mapping for a known file.

use std::path::Path;

// Integration tests live outside `src/`, so we import the crate by name.
use gpu_grep::scanner::mmap::MappedFile;

#[test]
fn test_mmap_loads_cargo_toml() {
    let path = Path::new("Cargo.toml");
    let mapped = MappedFile::open(path).expect("Failed to mmap Cargo.toml");

    let size = mapped.len();
    println!("Mapped Cargo.toml: {} bytes", size);

    // Cargo.toml is non-empty
    assert!(size > 0, "Mapped file should have non-zero size");
    assert!(!mapped.is_empty());

    // Sanity check: first bytes should be "[package]"
    let content = std::str::from_utf8(mapped.as_bytes()).expect("Cargo.toml should be valid UTF-8");
    assert!(
        content.starts_with("[package]"),
        "Expected Cargo.toml to start with [package], got: {:?}",
        &content[..20]
    );
}

#[test]
fn test_mmap_nonexistent_file_returns_error() {
    let path = Path::new("this_file_does_not_exist.txt");
    let result = MappedFile::open(path);
    assert!(
        result.is_err(),
        "Opening a nonexistent file should return an error"
    );
}
