mod cli;

use clap::Parser;
use cli::Cli;

fn main() {
    let args = Cli::parse();

    if args.verbose {
        println!("[verbose] Pattern : \"{}\"", args.pattern);
        println!("[verbose] File    : {:?}", args.file);
        println!("[verbose] Case-insensitive: {}", args.ignore_case);
    }

    println!("Searching for \"{}\" in {:?}...", args.pattern, args.file);
}
