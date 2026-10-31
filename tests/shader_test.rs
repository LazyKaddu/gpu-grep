//! Integration test: verifies that shaders/compute.wgsl compiles and executes
//! correctly on the GPU, finding exact byte matches via bitwise extraction.

use gpu_grep::gpu::engine::GpuContext;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

const SHADER_SRC: &str = include_str!("../shaders/compute.wgsl");

#[test]
fn test_shader_compiles_successfully() {
    let ctx = GpuContext::init(false);
    let _shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("compute_shader"),
        source: ShaderSource::Wgsl(SHADER_SRC.into()),
    });
}

#[test]
fn test_shader_finds_exact_matches() {
    let ctx = GpuContext::init(true);

    // 1. Create Shader Module
    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("compute_shader"),
        source: ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    // 2. Define Bind Group Layout matching the WGSL decorators
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
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

    // 3. Create Pipeline Layout and Compute Pipeline
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("compute_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // 4. Prepare Test Data
    // Target text contains "gpu-grep" at byte 0 and byte 40.
    let text = b"gpu-grep accelerates text search! grep with gpu-grep is fast.";
    let pattern_str = b"gpu-grep";

    // Pattern: one u32 per character
    let pattern_u32: Vec<u32> = pattern_str.iter().map(|&b| b as u32).collect();

    // Data: pack bytes into u32 words (4 bytes per u32, little-endian)
    let mut data_u32: Vec<u32> = Vec::new();
    for chunk in text.chunks(4) {
        let mut word = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            word |= (b as u32) << (i * 8);
        }
        data_u32.push(word);
    }

    // Results buffer: 1 u32 for count + capacity for up to 64 match indices
    let max_matches = 64usize;
    let results_initial = vec![0u32; 1 + max_matches];

    // 5. Create GPU Buffers
    let pattern_bytes: &[u8] = bytemuck::cast_slice(&pattern_u32);
    let pattern_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("pattern_buffer"),
        size: pattern_bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&pattern_buf, 0, pattern_bytes);

    let data_bytes: &[u8] = bytemuck::cast_slice(&data_u32);
    let data_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("data_buffer"),
        size: data_bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&data_buf, 0, data_bytes);

    let results_raw_bytes: &[u8] = bytemuck::cast_slice(&results_initial);
    let results_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("results_buffer"),
        size: results_raw_bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&results_buf, 0, results_raw_bytes);

    // Staging buffer to read results back to CPU
    let staging_buf = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("staging_buffer"),
        size: results_raw_bytes.len() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 6. Create Bind Group
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("compute_bind_group"),
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

    // 7. Dispatch Compute Shader
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("compute_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // Each workgroup has 256 threads; each thread handles 1 byte
        let workgroups = (text.len() as u32 + 255) / 256;
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }

    // Copy results to staging buffer
    encoder.copy_buffer_to_buffer(
        &results_buf,
        0,
        &staging_buf,
        0,
        results_raw_bytes.len() as u64,
    );

    let submission_index = ctx.queue.submit(std::iter::once(encoder.finish()));

    // 8. Read back results to CPU
    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).expect("Failed to send map result");
    });

    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
        .unwrap();
    rx.recv().unwrap().unwrap();

    let mapped = slice.get_mapped_range().unwrap();
    let result_u32s: &[u32] = bytemuck::cast_slice(&mapped);

    let match_count = result_u32s[0] as usize;
    let match_indices = &result_u32s[1..=match_count];

    println!("Detected match count: {}", match_count);
    for &idx in match_indices {
        let snippet = &text[idx as usize..(idx as usize + pattern_str.len()).min(text.len())];
        println!(
            "  Match at byte offset: {} -> {:?}",
            idx,
            std::str::from_utf8(snippet).unwrap_or("<invalid utf8>")
        );
    }

    assert_eq!(match_count, 2, "Expected exactly 2 matches");
    let mut sorted_indices = match_indices.to_vec();
    sorted_indices.sort();
    assert_eq!(sorted_indices, vec![0, 44], "Matches should be at offsets 0 and 44");

    drop(mapped);
    staging_buf.unmap();
}
