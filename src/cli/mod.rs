//! Command-line interface for gpu-grep.
//!
//! Uses the `clap` derive API to parse all user-facing arguments.
//! The [`Cli`] struct is the single source of truth consumed by `main.rs`.

use clap::Parser;
use std::path::PathBuf;

/// A zero-copy, GPU-accelerated text search CLI for massive datasets.
#[derive(Parser, Debug)]
#[command(name = "gpu-grep", version, about)]
pub struct Cli {
    /// The string pattern to search for
    pub pattern: String,

    /// The path to the file to search
    pub file: PathBuf,

    /// Perform a case-insensitive search (Coming soon)
    #[arg(short, long)]
    pub ignore_case: bool,

    /// Print detailed GPU timing and chunking metrics
    #[arg(short, long)]
    pub verbose: bool,
}
