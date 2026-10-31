mod cli;
mod scanner;

use clap::Parser;
use cli::Cli;
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

    println!(
        "Searching for \"{}\" in {:?} ({} bytes)...",
        args.pattern,
        args.file,
        mapped.len()
    );
}
