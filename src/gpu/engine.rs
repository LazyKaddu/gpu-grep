//! Headless WebGPU device and queue initialization.
//!
//! Requests a high-performance adapter (discrete GPU when available)
//! and creates the compute device. All initialization is async
//! internally but exposed as a blocking call via `pollster`.

use wgpu::{
    Device, DeviceDescriptor, Instance, InstanceDescriptor, MemoryHints, PowerPreference, Queue,
    RequestAdapterOptions,
};

/// A live GPU context ready to accept compute work.
///
/// Holds the [`Device`] (for creating pipelines and buffers)
/// and the [`Queue`] (for submitting command encoders).
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    /// Initializes the GPU: Instance → Adapter → Device + Queue.
    ///
    /// Forces `HighPerformance` power preference to grab the discrete
    /// GPU when available (rather than an integrated/low-power chip).
    ///
    /// # Panics
    /// Panics with a user-friendly message if:
    /// - No compatible GPU adapter is found on the system.
    /// - The device request fails (e.g. required features unsupported).
    pub fn init(verbose: bool) -> Self {
        pollster::block_on(Self::init_async(verbose))
    }

    /// Async implementation of GPU initialization.
    async fn init_async(verbose: bool) -> Self {
        // Step 1: Create the wgpu instance with default backends
        // (Vulkan on Linux, Metal on macOS, DX12 on Windows)
        let instance = Instance::new(InstanceDescriptor::new_without_display_handle());

        // Step 2: Request a high-performance adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .unwrap_or_else(|e| {
                eprintln!("gpu-grep: fatal: no compatible GPU found on this system.");
                eprintln!("  Ensure you have a Vulkan, Metal, or DirectX 12 capable GPU.");
                eprintln!("  Error: {e}");
                std::process::exit(1);
            });

        if verbose {
            let info = adapter.get_info();
            eprintln!("[verbose] GPU adapter : {}", info.name);
            eprintln!("[verbose] Backend     : {:?}", info.backend);
            eprintln!("[verbose] Driver      : {}", info.driver);
        }

        // Step 3: Request the device and queue
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("gpu-grep compute device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .unwrap_or_else(|e| {
                eprintln!("gpu-grep: fatal: failed to create GPU device.");
                eprintln!("  Error: {e}");
                std::process::exit(1);
            });

        if verbose {
            eprintln!("[verbose] GPU device ready");
        }

        Self { device, queue }
    }
}
