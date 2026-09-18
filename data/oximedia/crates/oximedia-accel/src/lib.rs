//! Hardware acceleration abstraction layer for `OxiMedia`.
//!
//! `oximedia-accel` provides GPU-accelerated computation for video processing
//! operations. It includes a Pure-Rust CPU fallback (always available), an
//! optional `wgpu`-backed WebGPU backend (`webgpu` feature), and an optional
//! Vulkan compute backend (`vulkan-backend` feature) for systems that want
//! real GPU dispatch via `vulkano`.
//!
//! # Pure Rust default / opt-in Vulkan backend
//!
//! The **default build of this crate compiles zero C/C++ code**. Vulkan
//! support is gated behind the non-default `vulkan-backend` Cargo feature
//! because `vulkano-shaders` compiles GLSL shaders to SPIR-V via a proc-macro
//! that pulls in `shaderc-sys`, which builds the `shaderc`/`glslang` C++
//! toolchain from source (see `README.md` for the native build prerequisites
//! this entails). Without `vulkan-backend`:
//!
//! - [`AccelContext::new`] and `AccelContext::with_device_selector`-style
//!   construction never attempt to load Vulkan; they select the Pure-Rust
//!   [`cpu_fallback::CpuAccel`] backend (or `webgpu`, when that feature is
//!   enabled and available) directly.
//! - Every Vulkan-only module (`device`, `buffer`, `vulkan`, `kernels`,
//!   `descriptor_pool`) is entirely absent from the compiled crate.
//! - A runtime request that specifically requires the Vulkan backend (e.g.
//!   `oximedia_accel::compute_backend::VulkanComputeBackend` reporting
//!   availability) returns a clear [`AccelError`], never a panic.
//!
//! Enable real Vulkan dispatch with:
//!
//! ```toml
//! [dependencies]
//! oximedia-accel = { version = "*", features = ["vulkan-backend"] }
//! ```
//!
//! # Features
//!
//! - **Device Management**: Automatic GPU device enumeration and selection (`vulkan-backend`)
//! - **Buffer Management**: Efficient GPU memory allocation and transfer (`vulkan-backend`)
//! - **Compute Kernels**: Image scaling, color conversion, motion estimation
//! - **CPU Fallback**: Automatic fallback to CPU implementations (always available)
//! - **Safe Vulkan**: Uses vulkano for safe Vulkan API access (opt-in via `vulkan-backend`)
//!
//! # Backend Architecture
//!
//! The acceleration layer is designed around the [`HardwareAccel`] trait,
//! which provides a unified interface for both GPU and CPU implementations.
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         HardwareAccel Trait             │
//! └─────────────────────────────────────────┘
//!          │                      │
//!          ▼                      ▼
//!   ┌────────────┐         ┌────────────┐
//!   │ VulkanAccel│         │ CpuFallback│
//!   │(opt-in via │         │  (Pure     │
//!   │vulkan-     │         │   Rust,    │
//!   │backend)    │         │  default)  │
//!   └────────────┘         └────────────┘
//! ```
//!
//! # GPU Data-Flow Architecture
//!
//! The way data travels between the host (CPU) and the GPU depends on the
//! memory architecture of the selected device.
//!
//! ## Discrete GPU path (PCIe staging required)
//!
//! Used when `device::DevicePreference::Discrete` is selected (the default,
//! requires the `vulkan-backend` feature) and the
//! GPU has its own dedicated VRAM connected via PCIe.  Two DMA transfers are
//! required — one to upload and one to download — through a host-visible
//! *staging buffer*:
//!
//! ```text
//! Host (CPU) buffer
//!       │  (host-write)
//!       ▼
//! Staging buffer (host-visible | device-local)
//!       │  (DMA transfer over PCIe)
//!       ▼
//! GPU VRAM (device-local)
//!       │  (compute shader)
//!       ▼
//! Staging buffer (host-visible | device-local)
//!       │  (DMA transfer over PCIe)
//!       ▼
//! Host (CPU) buffer  (host-read)
//! ```
//!
//! ## Integrated GPU path (UMA, direct map)
//!
//! Used when `device::DevicePreference::Integrated` is selected (requires the
//! `vulkan-backend` feature) and the GPU
//! shares system RAM (Unified Memory Architecture).  No staging copy is needed
//! because the same physical allocation is both host-visible and device-local:
//!
//! ```text
//! Unified memory (host-visible | device-local)
//!       │  (zero-copy compute shader)
//!       ▼
//! Unified memory  (result in same allocation)
//! ```
//!
//! The discrete path maximises GPU throughput at the cost of PCIe bandwidth;
//! the integrated path eliminates that overhead but shares memory bandwidth
//! with the CPU.  `AccelContext::new()` picks the discrete path by default;
//! pass a custom `device::DeviceSelector` to `AccelContext::with_device_selector`
//! to choose otherwise (both require the `vulkan-backend` feature).
//!
//! # Example
//!
//! ```no_run
//! use oximedia_accel::{AccelContext, HardwareAccel, ScaleFilter};
//! use oximedia_core::types::PixelFormat;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create acceleration context (automatically selects GPU or CPU)
//! let accel = AccelContext::new()?;
//!
//! // Perform image scaling
//! let input = vec![0u8; 1920 * 1080 * 3];
//! let output = accel.scale_image(
//!     &input,
//!     1920, 1080,
//!     1280, 720,
//!     PixelFormat::Rgb24,
//!     ScaleFilter::Bilinear,
//! )?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

pub mod accel_profile;
pub mod accel_stats;
pub mod async_compute;
pub mod bilateral_gpu;
#[cfg(feature = "vulkan-backend")]
pub mod buffer;
pub mod buffer_ring;
pub mod cache;
pub mod compute_backend;
pub mod cpu_fallback;
pub mod cpu_simd;
#[cfg(feature = "vulkan-backend")]
pub mod device;
pub mod device_caps;
pub mod dispatch;
pub mod error;
pub mod fence_timeline;
pub mod gpu_ops;
#[cfg(feature = "vulkan-backend")]
pub mod kernels;
pub mod memory_access;
pub mod memory_arena;
pub mod memory_bandwidth;
pub mod metal_backend;
pub mod multi_gpu;
pub mod ops;
pub mod pipeline_accel;
pub mod pipeline_cache;
pub mod pool;
pub mod prefetch;
pub mod shaders;
mod stress_tests;
pub mod subgroup;
pub mod task_graph;
pub mod task_scheduler;
pub mod temporal_nr;
pub mod traits;
pub mod vectorize;
#[cfg(feature = "vulkan-backend")]
pub mod vulkan;
pub mod webgpu_backend;
pub mod work_item;
pub mod workgroup;

// Re-export commonly used items
pub use error::{AccelError, AccelResult};
pub use traits::{HardwareAccel, ScaleFilter};

// Re-export new feature types
pub use accel_stats::{AccelProfiler, ProfileEntry};
pub use memory_arena::{MemoryPressureMonitor, MemoryPressurePolicy, PressureLevel};
pub use ops::convolution::{ConvolutionConfig, ConvolutionFilter, EdgeMode};
pub use ops::deinterlace::{DeinterlaceConfig, DeinterlaceMethod, FieldOrder};

// Re-export colour / HDR operations
pub use ops::color::{
    hlg_to_sdr_tonemap, pq_to_sdr_tonemap, rgb_to_yuv420, yuv420_to_rgb, YuvRange, YuvStandard,
};
pub use ops::{alpha_blend, alpha_blend_rgba};

// Re-export GPU compute operations
pub use ops::affine_gpu::{apply_affine, AffineTransform};
pub use ops::dct_gpu::{
    forward_dct_8x8, forward_dct_batch, inverse_dct_8x8, inverse_dct_batch, DctBlock,
};
pub use ops::histogram_gpu::{compute_histogram, GpuHistogram, HISTOGRAM_BINS};

// Re-export WebGPU backend
pub use webgpu_backend::{WebGpuAccelBackend, WebGpuAdapterInfo, WebGpuBackendState};

// Re-export bilateral spatial NR (CPU always, GPU when `webgpu` feature on)
#[cfg(feature = "webgpu")]
pub use bilateral_gpu::BilateralGpu;
pub use bilateral_gpu::{bilateral_cpu, BilateralParams, MAX_BILATERAL_RADIUS};

// Re-export temporal NR
#[cfg(feature = "webgpu")]
pub use temporal_nr::TemporalNr;
pub use temporal_nr::{temporal_nr_cpu, TemporalNrParams};

// Re-export workgroup auto-tuning
pub use workgroup::{compute_optimal_workgroup, OpType};

// Re-export descriptor pool
#[cfg(feature = "vulkan-backend")]
pub mod descriptor_pool;

#[cfg(feature = "vulkan-backend")]
use device::DeviceSelector;
use std::sync::Arc;
#[cfg(feature = "vulkan-backend")]
use vulkan::VulkanAccel;

/// Main acceleration context that automatically selects GPU or CPU backend.
///
/// This is the primary entry point for hardware-accelerated operations.
/// When the `vulkan-backend` feature is enabled it will attempt to
/// initialize Vulkan compute first, falling back to CPU if GPU acceleration
/// is unavailable. Without that feature (the default, Pure-Rust build) it
/// always uses the CPU fallback backend directly — no Vulkan code is even
/// compiled into the crate.
pub struct AccelContext {
    backend: AccelBackend,
}

enum AccelBackend {
    #[cfg(feature = "vulkan-backend")]
    Vulkan(Arc<VulkanAccel>),
    Cpu(Arc<cpu_fallback::CpuAccel>),
}

impl AccelContext {
    /// Creates a new acceleration context.
    ///
    /// With the `vulkan-backend` feature enabled, attempts to initialize GPU
    /// acceleration first, falling back to CPU if Vulkan is unavailable or
    /// device selection fails. Without that feature (Pure-Rust default
    /// build), always selects the CPU fallback backend.
    ///
    /// # Errors
    ///
    /// Returns an error only if both GPU and CPU initialization fail,
    /// which should be extremely rare.
    pub fn new() -> AccelResult<Self> {
        #[cfg(feature = "vulkan-backend")]
        {
            Self::with_device_selector(&DeviceSelector::default())
        }
        #[cfg(not(feature = "vulkan-backend"))]
        {
            Ok(Self::cpu_only())
        }
    }

    /// Creates a new acceleration context with a custom Vulkan device
    /// selector.
    ///
    /// Only available when the `vulkan-backend` feature is enabled, since
    /// [`device::DeviceSelector`] is a Vulkan-specific type.
    ///
    /// # Errors
    ///
    /// Returns an error if both GPU and CPU initialization fail.
    #[cfg(feature = "vulkan-backend")]
    pub fn with_device_selector(selector: &DeviceSelector) -> AccelResult<Self> {
        match VulkanAccel::new(selector) {
            Ok(vulkan) => {
                tracing::info!("Hardware acceleration: Vulkan GPU");
                Ok(Self {
                    backend: AccelBackend::Vulkan(Arc::new(vulkan)),
                })
            }
            Err(e) => {
                tracing::warn!("Vulkan initialization failed: {}, using CPU fallback", e);
                Ok(Self {
                    backend: AccelBackend::Cpu(Arc::new(cpu_fallback::CpuAccel::new())),
                })
            }
        }
    }

    /// Forces CPU-only acceleration (no GPU).
    ///
    /// Useful for testing or when GPU acceleration is explicitly unwanted.
    #[must_use]
    pub fn cpu_only() -> Self {
        tracing::info!("Hardware acceleration: CPU only (forced)");
        Self {
            backend: AccelBackend::Cpu(Arc::new(cpu_fallback::CpuAccel::new())),
        }
    }

    /// Returns `true` if using GPU acceleration.
    #[must_use]
    pub fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "vulkan-backend")]
        {
            matches!(self.backend, AccelBackend::Vulkan(_))
        }
        #[cfg(not(feature = "vulkan-backend"))]
        {
            false
        }
    }

    /// Returns the name of the current backend.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        match &self.backend {
            #[cfg(feature = "vulkan-backend")]
            AccelBackend::Vulkan(v) => v.device_name(),
            AccelBackend::Cpu(_) => "CPU",
        }
    }
}

impl HardwareAccel for AccelContext {
    fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: oximedia_core::PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        match &self.backend {
            #[cfg(feature = "vulkan-backend")]
            AccelBackend::Vulkan(v) => v.scale_image(
                input, src_width, src_height, dst_width, dst_height, format, filter,
            ),
            AccelBackend::Cpu(c) => c.scale_image(
                input, src_width, src_height, dst_width, dst_height, format, filter,
            ),
        }
    }

    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: oximedia_core::PixelFormat,
        dst_format: oximedia_core::PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        match &self.backend {
            #[cfg(feature = "vulkan-backend")]
            AccelBackend::Vulkan(v) => {
                v.convert_color(input, width, height, src_format, dst_format)
            }
            AccelBackend::Cpu(c) => c.convert_color(input, width, height, src_format, dst_format),
        }
    }

    fn motion_estimation(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>> {
        match &self.backend {
            #[cfg(feature = "vulkan-backend")]
            AccelBackend::Vulkan(v) => {
                v.motion_estimation(reference, current, width, height, block_size)
            }
            AccelBackend::Cpu(c) => {
                c.motion_estimation(reference, current, width, height, block_size)
            }
        }
    }
}
