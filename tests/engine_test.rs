//! Integration test: confirms the GPU hardware boots up and the
//! WebGPU device and queue can be initialized and accept commands.

use gpu_grep::gpu::engine::GpuContext;

#[test]
fn test_gpu_context_initialization() {
    let ctx = GpuContext::init(true);

    let limits = ctx.device.limits();
    println!("GPU max_buffer_size: {} bytes", limits.max_buffer_size);
    println!(
        "GPU max_storage_buffer_binding_size: {} bytes",
        limits.max_storage_buffer_binding_size
    );
    println!(
        "GPU max_compute_invocations_per_workgroup: {}",
        limits.max_compute_invocations_per_workgroup
    );

    // Verify critical WebGPU compute capabilities
    assert!(
        limits.max_buffer_size > 0,
        "Device must support non-zero buffer allocations"
    );
    assert!(
        limits.max_storage_buffer_binding_size > 0,
        "Device must support storage buffers for grep compute"
    );
    assert!(
        limits.max_compute_invocations_per_workgroup >= 64,
        "Device must support at least 64 compute invocations per workgroup"
    );
}

#[test]
fn test_gpu_buffer_allocation_and_queue_submission() {
    let ctx = GpuContext::init(false);

    // Test creating a storage buffer on the device
    let buffer_size: u64 = 256;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_storage_buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    assert_eq!(buffer.size(), buffer_size);

    // Test encoding a command and submitting to the queue
    let encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_command_encoder"),
        });

    let command_buffer = encoder.finish();
    ctx.queue.submit(std::iter::once(command_buffer));
}
