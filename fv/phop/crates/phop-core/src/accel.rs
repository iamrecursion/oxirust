//! Accelerator selection for the tensorized EML forward pass.
//!
//! phop has four forward-eval backends: NVIDIA CUDA (`gpu-cuda`, exact `f64`, via oxicuda), native
//! Apple Metal (`gpu-metal`, `f32`, via oxicuda-metal, macOS only), a portable
//! WebGPU/Metal/Vulkan/DX12 path (`gpu-wgpu`, `f32`, via `crate::wgpu_forward`), and the
//! always-available CPU path (`f64`). [`gpu_backend`] picks the best one present at runtime, in the
//! order **CUDA → Metal → wgpu → CPU** — CUDA first for its `f64` precision and existing tuned
//! kernels, native Metal next on Apple hardware, wgpu for portability elsewhere, CPU as the
//! universal fallback.

/// A compute backend for the EML forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA via oxicuda (`f64`). Requires the `gpu-cuda` feature **and** a device at runtime.
    Cuda,
    /// Native Apple Metal via oxicuda-metal (`f32`). Requires the `gpu-metal` feature **and** a
    /// Metal device at runtime (macOS only).
    Metal,
    /// Portable WebGPU/Metal/Vulkan/DX12 via wgpu (`f32`). Requires the `gpu-wgpu` feature **and** an
    /// adapter at runtime.
    Wgpu,
    /// CPU fallback — always available, exact `f64`.
    Cpu,
}

/// Select the best available forward-eval backend at runtime: **CUDA → Metal → wgpu → CPU**.
///
/// Each accelerator branch only exists when its feature is compiled in, and is taken only if the
/// corresponding device/adapter is actually present (so a build with both GPU features still degrades
/// cleanly to CPU on a machine with neither).
#[must_use]
pub fn gpu_backend() -> GpuBackend {
    #[cfg(feature = "gpu-cuda")]
    {
        if crate::gpu::cuda_available() {
            return GpuBackend::Cuda;
        }
    }
    #[cfg(feature = "gpu-metal")]
    {
        if crate::metal::metal_available() {
            return GpuBackend::Metal;
        }
    }
    #[cfg(feature = "gpu-wgpu")]
    {
        if crate::wgpu_forward::wgpu_available() {
            return GpuBackend::Wgpu;
        }
    }
    GpuBackend::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_cpu_without_gpu_features() {
        // With no GPU feature compiled in, the only possible choice is CPU. (We deliberately do NOT
        // call `gpu_backend()` when a GPU feature is enabled: the wgpu adapter probe can crash on
        // broken software drivers, and the meaningful invariant — CPU when no accelerator exists — is
        // exactly this branch.)
        #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal", feature = "gpu-wgpu")))]
        assert_eq!(gpu_backend(), GpuBackend::Cpu);
    }

    #[test]
    fn backend_variants_are_distinct() {
        assert_ne!(GpuBackend::Cuda, GpuBackend::Wgpu);
        assert_ne!(GpuBackend::Wgpu, GpuBackend::Cpu);
        assert_ne!(GpuBackend::Cuda, GpuBackend::Metal);
        assert_ne!(GpuBackend::Metal, GpuBackend::Wgpu);
        assert_ne!(GpuBackend::Metal, GpuBackend::Cpu);
    }
}
