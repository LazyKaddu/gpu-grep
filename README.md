# gpu-grep

[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg)](https://www.rust-lang.org)
[![wgpu](https://img.shields.io/badge/WebGPU-wgpu-blue.svg)](https://wgpu.rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A zero-copy, GPU-accelerated text search CLI for massive datasets.**

`gpu-grep` bypasses the CPU bottleneck of traditional text search. By combining Rust's memory-mapped files (`mmap`) with native WebGPU (`wgpu`) compute shaders, it streams gigabytes of log files directly across the PCIe bus to the GPU. Millions of GPU threads evaluate string matches in parallel, completely offloading the search workload from the system's CPU.

This tool is designed specifically for massive, monolithic datasets (10GB+ server logs, CSV dumps, genomic data) where traditional CPU-bound tools like `grep` or `ripgrep` become heavily IO and CPU bottlenecked.

---

## Features

*   **Zero-Copy CPU Pipeline:** Uses `memmap2` to map files directly into virtual memory. The CPU never eagerly copies file data into application RAM.
*   **Hardware Agnostic GPU Compute:** Powered by `wgpu`, the compute shader runs natively on Apple Metal (macOS), DirectX 12 (Windows), and Vulkan (Linux) without requiring proprietary APIs like CUDA.
*   **Double-Buffered Streaming:** Masks PCIe transfer latency by utilizing a ping-pong buffer system. The next 256MB chunk uploads while the GPU processes the current chunk.
*   **Lock-Free GPU Aggregation:** GPU threads use atomic counters to safely record match indices directly into a mapped output buffer, avoiding race conditions.

---

## Architecture Overview

`gpu-grep` is built on a 5-stage streaming pipeline:

1.  **The CLI Orchestrator (CPU):** Parses arguments using `clap` and initializes the headless `wgpu` device (preferring discrete GPUs for optimal memory bandwidth).
2.  **The Zero-Copy Loader (OS/Kernel):** Maps the target file into virtual memory via page faults. 
3.  **The Streaming Engine (PCIe Bus):** Slices the memory map into 256MB chunks and performs DMA-transfers directly into WebGPU Storage Buffers.
4.  **The Compute Kernel (GPU/WGSL):** Hundreds of thousands of WGSL threads execute concurrently. Each thread unpacks 4-byte (`u32`) aligned memory blocks, evaluates the search string, and atomically writes the byte offset of any matches to an Output Buffer.
5.  **The Aggregator & Formatter (CPU):** The CPU maps the Output Buffer back to system RAM, uses the raw byte indices to locate the surrounding `\n` characters in the `mmap` slice, and prints the reconstructed line to `stdout`.

---

## Installation

### Prerequisites
*   [Rust toolchain](https://rustup.rs/) (v1.80 or higher)
*   A GPU with Vulkan, Metal, or DirectX 12 support.

### Build from Source

Clone the repository and build using Cargo. It is highly recommended to build in release mode, as debug mode disables critical LLVM optimizations required for fast memory handling and mapping.

```bash
git clone https://github.com/yourusername/gpu-grep.git
cd gpu-grep
cargo build --release
```

The compiled binary will be located at `target/release/gpu-grep`.

To install it globally on your system, run:
```bash
cargo install --path .
```

---

## Usage

The CLI interface is designed to closely mimic standard Unix search tools.

```bash
# Search a massive log file for a specific error code
gpu-grep "ERR_TIMEOUT_509" /var/log/nginx/access.log

# Get detailed performance timings for the GPU pipeline
gpu-grep "FATAL" ./massive_database_dump.csv --verbose
```

### Command Line Arguments

```text
Usage: gpu-grep [OPTIONS] <PATTERN> <FILE>

Arguments:
  <PATTERN>  The string pattern to search for
  <FILE>     The path to the file to search

Options:
  -i, --ignore-case  Perform a case-insensitive search (Coming soon)
  -v, --verbose      Print detailed GPU timing and chunking metrics
  -h, --help         Print help
  -V, --version      Print version
```

---

## Project Structure

```text
gpu-grep/
├── Cargo.toml
├── shaders/
│   └── compute.wgsl       # WGSL WebGPU compute kernel logic
└── src/
    ├── main.rs            # Pipeline orchestrator
    ├── cli/
    │   └── mod.rs         # clap argument parsing
    ├── gpu/
    │   ├── engine.rs      # wgpu Device & Queue initialization
    │   └── buffers.rs     # Ping-pong Storage and Staging buffers
    └── scanner/
        ├── mmap.rs        # memmap2 zero-copy file loading
        └── format.rs      # Match index to text line reconstruction
```

---

## Known Limitations and Trade-offs

*   **PCIe Latency Cost:** Transferring data to the GPU requires overhead. For small files (< 500MB), CPU-only tools like `ripgrep` will likely out-perform `gpu-grep`. This architecture is explicitly optimized for multi-gigabyte files where parallel processing eclipses the PCIe transfer cost.
*   **Pattern Matching Limitation:** Currently optimized for exact string matching. Advanced Regex support requires complex state-machine compilation on the GPU and is not yet implemented.
*   **WGSL Alignment Constraints:** WebGPU does not support `u8` storage buffers natively. File data must be packed into `u32` blocks, requiring bitwise extraction inside the compute shader.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.