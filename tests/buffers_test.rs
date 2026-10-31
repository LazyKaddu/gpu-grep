//! Integration tests for Buffer Management & Compute Dispatch (src/gpu/buffers.rs).

use gpu_grep::gpu::buffers::search;
use gpu_grep::gpu::engine::GpuContext;
use gpu_grep::scanner::mmap::MappedFile;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[test]
fn test_buffers_search_cargo_toml() {
    let ctx = GpuContext::init(false);
    let path = Path::new("Cargo.toml");
    let mapped = MappedFile::open(path).expect("Failed to mmap Cargo.toml");

    // Search for "gpu-grep" (which is the package name in Cargo.toml)
    let pattern = b"gpu-grep";
    let matches = search(&ctx, &mapped, pattern, false);

    assert!(
        !matches.is_empty(),
        "Expected at least 1 match for 'gpu-grep' in Cargo.toml"
    );

    // Verify each returned offset actually contains the search pattern
    let bytes = mapped.as_bytes();
    for &offset in &matches {
        assert_eq!(
            &bytes[offset..offset + pattern.len()],
            pattern,
            "Byte offset {} should match pattern",
            offset
        );
    }
}

#[test]
fn test_buffers_search_no_match() {
    let ctx = GpuContext::init(false);
    let path = Path::new("Cargo.toml");
    let mapped = MappedFile::open(path).expect("Failed to mmap Cargo.toml");

    let pattern = b"nonexistent_pattern_string_xyz_999";
    let matches = search(&ctx, &mapped, pattern, false);

    assert!(
        matches.is_empty(),
        "Expected 0 matches for nonexistent pattern"
    );
}

#[test]
fn test_buffers_empty_pattern_returns_empty() {
    let ctx = GpuContext::init(false);
    let path = Path::new("Cargo.toml");
    let mapped = MappedFile::open(path).expect("Failed to mmap Cargo.toml");

    let matches = search(&ctx, &mapped, b"", false);
    assert!(matches.is_empty(), "Empty pattern should yield 0 matches");
}

#[test]
fn test_buffers_synthetic_multi_match() {
    let ctx = GpuContext::init(false);

    // Create a temporary file with known content and repeating patterns
    let temp_path = std::env::temp_dir().join("gpu_grep_test_multi.txt");
    {
        let mut file = File::create(&temp_path).expect("Failed to create temp file");
        let content = b"alpha beta gamma MATCH1 delta epsilon MATCH2 zeta eta MATCH3 theta";
        file.write_all(content).expect("Failed to write to temp file");
    }

    let mapped = MappedFile::open(&temp_path).expect("Failed to mmap temp file");
    let pattern = b"MATCH";
    let matches = search(&ctx, &mapped, pattern, false);

    let bytes = mapped.as_bytes();
    assert_eq!(matches.len(), 3, "Expected exactly 3 matches for 'MATCH'");
    for &offset in &matches {
        assert_eq!(&bytes[offset..offset + pattern.len()], pattern);
    }

    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_buffers_multi_chunk_boundary_overlap() {
    use gpu_grep::gpu::buffers::GpuScanner;

    let ctx = GpuContext::init(false);

    // Create a 200-byte file where the pattern "STRADDLE" starts at byte 60.
    // With chunk size 64, Chunk 0 covers 0..64 and Chunk 1 covers 64..128.
    // The pattern spans across bytes 60..68 (4 bytes in Chunk 0, 4 bytes in Chunk 1).
    let mut data = vec![b'x'; 200];
    let pattern = b"STRADDLE";
    data[60..68].copy_from_slice(pattern);

    let temp_path = std::env::temp_dir().join("gpu_grep_test_boundary.txt");
    {
        let mut file = File::create(&temp_path).expect("Failed to create temp file");
        file.write_all(&data).expect("Failed to write to temp file");
    }

    let mapped = MappedFile::open(&temp_path).expect("Failed to mmap temp file");

    // Force chunk size to 64 bytes
    let mut scanner = GpuScanner::with_chunk_size(&ctx, pattern, mapped.len(), 64, true);
    let matches = scanner.search_mmap(&ctx, &mapped, pattern, true);

    println!("Multi-chunk matches: {:?}", matches);
    assert_eq!(
        matches,
        vec![60],
        "Sliding window overlap should capture the match at boundary 60"
    );

    let _ = std::fs::remove_file(&temp_path);
}

