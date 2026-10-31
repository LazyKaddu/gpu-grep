//! Buffer management, memory streaming, and compute pass dispatch.
//!
//! Bridges the CPU memory-mapped file with the GPU compute shader by:
//! 1. Allocating Storage buffers (data chunks, search pattern, match results)
//!    and a Staging buffer for host readback.
//! 2. Streaming file data over PCIe in chunks via direct zero-copy DMA transfers.
//! 3. Overlapping chunk boundaries by `(pattern_len - 1)` bytes to guarantee
//!    no search pattern spanning across chunks is missed.
//! 4. Reading back match offsets to CPU system memory.

use crate::gpu::engine::GpuContext;
use crate::scanner::mmap::MappedFile;
use std::time::Instant;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

const SHADER_SRC: &str = include_str!("../../shaders/compute.wgsl");

/// Default maximum target chunk size in bytes (256 MB).
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024 * 1024;

/// Maximum number of match indices recorded per chunk.
pub const MAX_MATCHES_PER_CHUNK: usize = 1_000_000;

/// GPU search engine owning the compute pipeline, bind groups, and allocated buffers.
pub struct GpuScanner {
    pipeline: ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: BindGroupLayout,
    #[allow(dead_code)]
    pattern_buf: Buffer,
    data_buf: Buffer,
    results_buf: Buffer,
    staging_buf: Buffer,
    bind_group: BindGroup,
    chunk_size: usize,
    max_results: usize,
}

impl GpuScanner {
    /// Initializes the compute pipeline and allocates GPU storage buffers.
    ///
    /// Automatically caps chunk size to the hardware's maximum storage buffer
    /// binding limit (e.g. 128 MB on some integrated GPUs, 256 MB or more on discrete GPUs).
    pub fn new(context: &GpuContext, pattern: &[u8], file_len: usize, verbose: bool) -> Self {
        Self::with_chunk_size(context, pattern, file_len, DEFAULT_CHUNK_SIZE, verbose)
    }

    /// Initializes GpuScanner with a custom target chunk size (useful for testing multi-chunk streaming).
    pub fn with_chunk_size(
        context: &GpuContext,
        pattern: &[u8],
        file_len: usize,
        target_chunk_size: usize,
        verbose: bool,
    ) -> Self {
        let alloc_start = Instant::now();

        // 1. Compile WGSL Shader
        let shader = context
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("gpu_grep_compute_shader"),
                source: ShaderSource::Wgsl(SHADER_SRC.into()),
            });

        // 2. Define Bind Group Layout matching shaders/compute.wgsl (@group 0)
        let bind_group_layout = context
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("gpu_grep_bind_group_layout"),
                entries: &[
                    // Binding 0: pattern (read-only storage buffer)
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: data (read-only storage buffer)
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: results (read-write storage buffer)
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // 3. Create Compute Pipeline
        let pipeline_layout = context
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("gpu_grep_pipeline_layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("gpu_grep_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // 4. Create and upload Pattern Buffer (one u32 per character)
        let pattern_u32: Vec<u32> = pattern.iter().map(|&b| b as u32).collect();
        let pattern_bytes: &[u8] = bytemuck::cast_slice(&pattern_u32);
        let pattern_buf_size = (pattern_bytes.len().max(4) as u64).next_multiple_of(4);
        let pattern_buf = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_grep_pattern_buf"),
            size: pattern_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !pattern_bytes.is_empty() {
            context.queue.write_buffer(&pattern_buf, 0, pattern_bytes);
        }

        // 5. Calculate Chunk Buffer Size respecting hardware limits and overlap
        let max_storage = context.device.limits().max_storage_buffer_binding_size as usize;
        let overlap = pattern.len().saturating_sub(1);
        let target_chunk = target_chunk_size.min(max_storage.saturating_sub(overlap));
        let chunk_size = file_len.min(target_chunk).max(64).next_multiple_of(4);
        let data_buf_size = ((chunk_size + overlap) as u64).next_multiple_of(4);

        let data_buf = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_grep_data_buf"),
            size: data_buf_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 6. Create Results and Staging Buffers
        let max_results = MAX_MATCHES_PER_CHUNK;
        let results_size = ((1 + max_results) * 4) as u64; // 1 u32 counter + max_results indices
        let results_buf = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_grep_results_buf"),
            size: results_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buf = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_grep_staging_buf"),
            size: results_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 7. Assemble Bind Group
        let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("gpu_grep_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pattern_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: data_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        if verbose {
            println!(
                "[verbose] Buffers allocated in {:?} (chunk buffer: {} MB, max binding: {} MB)",
                alloc_start.elapsed(),
                chunk_size / (1024 * 1024),
                max_storage / (1024 * 1024)
            );
        }

        Self {
            pipeline,
            bind_group_layout,
            pattern_buf,
            data_buf,
            results_buf,
            staging_buf,
            bind_group,
            chunk_size,
            max_results,
        }
    }

    /// Slices the memory map, streams chunks across PCIe, dispatches compute,
    /// and extracts matching byte indices.
    pub fn search_mmap(
        &mut self,
        context: &GpuContext,
        mapped: &MappedFile,
        pattern: &[u8],
        verbose: bool,
    ) -> Vec<usize> {
        let file_bytes = mapped.as_bytes();
        let file_len = file_bytes.len();

        if pattern.is_empty() || file_len < pattern.len() {
            return Vec::new();
        }

        let start_time = Instant::now();
        let step = self.chunk_size;
        let overlap = pattern.len() - 1;
        let mut all_matches = Vec::new();

        let total_chunks = (file_len + step - 1) / step;
        let mut dma_duration = std::time::Duration::ZERO;
        let mut compute_duration = std::time::Duration::ZERO;

        for chunk_start in (0..file_len).step_by(step) {
            let chunk_end = (chunk_start + step + overlap).min(file_len);
            let chunk = &file_bytes[chunk_start..chunk_end];

            // 1. Direct DMA upload of chunk bytes
            let dma_start = Instant::now();
            let full_len = chunk.len() - (chunk.len() % 4);
            if full_len > 0 {
                context
                    .queue
                    .write_buffer(&self.data_buf, 0, &chunk[..full_len]);
            }
            let rem = chunk.len() % 4;
            if rem > 0 {
                let mut tail = [0u8; 4];
                tail[..rem].copy_from_slice(&chunk[full_len..]);
                context
                    .queue
                    .write_buffer(&self.data_buf, full_len as u64, &tail);
            }

            // Reset atomic match counter to 0 before dispatch
            context.queue.write_buffer(&self.results_buf, 0, &[0u8; 4]);
            dma_duration += dma_start.elapsed();

            // 2. Encode and submit compute pass
            let compute_start = Instant::now();
            let mut encoder = context
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("gpu_grep_compute_encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("gpu_grep_compute_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                let workgroups = ((chunk.len() as u32) + 255) / 256;
                pass.dispatch_workgroups(workgroups.max(1), 1, 1);
            }

            // Copy results from storage buffer to staging buffer
            encoder.copy_buffer_to_buffer(
                &self.results_buf,
                0,
                &self.staging_buf,
                0,
                self.staging_buf.size(),
            );

            let submission_index = context.queue.submit(std::iter::once(encoder.finish()));

            // 3. Read back results to CPU
            let slice = self.staging_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });

            context
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index),
                    timeout: None,
                })
                .unwrap();

            rx.recv().unwrap().unwrap();
            compute_duration += compute_start.elapsed();

            // 4. Extract raw match indices
            let mapped_range = slice.get_mapped_range().unwrap();
            let result_words: &[u32] = bytemuck::cast_slice(&mapped_range);
            let count = (result_words[0] as usize).min(self.max_results);

            let is_last_chunk = chunk_end == file_len;
            for &local_idx in &result_words[1..=count] {
                let local_idx = local_idx as usize;
                // Avoid duplicates: only accept matches starting within this chunk's
                // primary non-overlapping range, or all matches if this is the final chunk.
                if local_idx < step || is_last_chunk {
                    let global_idx = chunk_start + local_idx;
                    if global_idx + pattern.len() <= file_len {
                        all_matches.push(global_idx);
                    }
                }
            }

            drop(mapped_range);
            self.staging_buf.unmap();
        }

        all_matches.sort_unstable();
        all_matches.dedup();

        if verbose {
            println!("[verbose] Chunks processed : {}", total_chunks);
            println!("[verbose] PCIe DMA upload  : {:?}", dma_duration);
            println!("[verbose] GPU compute time : {:?}", compute_duration);
            println!("[verbose] Total search time: {:?}", start_time.elapsed());
            println!("[verbose] Matches found    : {}", all_matches.len());
        }

        all_matches
    }
}

/// Convenience function to execute a complete GPU search across a memory-mapped file.
pub fn search(
    context: &GpuContext,
    mapped: &MappedFile,
    pattern: &[u8],
    verbose: bool,
) -> Vec<usize> {
    let mut scanner = GpuScanner::new(context, pattern, mapped.len(), verbose);
    scanner.search_mmap(context, mapped, pattern, verbose)
}
