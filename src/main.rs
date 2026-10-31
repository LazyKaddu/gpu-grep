mod cli;
mod gpu;
mod scanner;

use clap::Parser;
use cli::Cli;
use gpu::engine::GpuContext;
use scanner::mmap::MappedFile;

fn main() {
    let args = Cli::parse();

    if args.verbose {
        println!("[verbose] Pattern : \"{}\"", args.pattern);
        println!("[verbose] File    : {:?}", args.file);
        println!("[verbose] Case-insensitive: {}", args.ignore_case);
    }

    // Stage 2: Zero-copy file mapping
    let mapped = MappedFile::open(&args.file).unwrap_or_else(|e| {
        eprintln!("error: failed to open {:?}: {}", args.file, e);
        std::process::exit(1);
    });

    if args.verbose {
        println!("[verbose] Mapped {} bytes", mapped.len());
    }

    // Stage 3: Initialize GPU
    let gpu = GpuContext::init(args.verbose);

    // Stage 4 & 5: Buffer Management & Compute Dispatch
    let matches = gpu::buffers::search(&gpu, &mapped, args.pattern.as_bytes(), args.verbose);

    println!(
        "Found {} match(es) for \"{}\" in {:?} ({} bytes)",
        matches.len(),
        args.pattern,
        args.file,
        mapped.len()
    );
}
