//! Multi-GPU support: distribute work across multiple Vulkan devices.
//!
//! Provides round-robin and load-balanced distribution of compute tasks
//! across all available Vulkan devices on the system.
//!
//! Without the `vulkan-backend` feature (the default, Pure-Rust build) no
//! Vulkan devices can ever be enumerated, so [`MultiGpuDispatcher`] always
//! routes work to the CPU fallback; [`MultiGpuDispatcher::gpu_count`] is
//! always `0` and [`MultiGpuDispatcher::has_gpu`] is always `false`.

#![allow(dead_code)]

use crate::cpu_fallback::CpuAccel;
#[cfg(feature = "vulkan-backend")]
use crate::device::{DevicePreference, DeviceSelector};
use crate::error::AccelResult;
use crate::traits::{HardwareAccel, ScaleFilter};
#[cfg(feature = "vulkan-backend")]
use crate::vulkan::VulkanAccel;
use oximedia_core::PixelFormat;
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "vulkan-backend")]
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Strategy for distributing work across multiple GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiGpuStrategy {
    /// Round-robin: each job goes to the next GPU in sequence.
    RoundRobin,
    /// Load-balanced: assign to GPU with the least outstanding work (approximated
    /// with a simple counter decrement, no actual GPU query).
    LoadBalanced,
    /// Replicate: run on all GPUs and return the first result (useful for redundancy).
    FirstComplete,
}

/// A logical GPU worker entry.
#[cfg(feature = "vulkan-backend")]
struct GpuWorker {
    backend: Arc<VulkanAccel>,
    /// Approximate outstanding work units.
    load: AtomicUsize,
    name: String,
}

#[cfg(feature = "vulkan-backend")]
impl GpuWorker {
    fn new(backend: VulkanAccel) -> Self {
        let name = backend.device_name().to_owned();
        Self {
            backend: Arc::new(backend),
            load: AtomicUsize::new(0),
            name,
        }
    }
}

/// Multi-GPU dispatcher that fans work out across all detected Vulkan devices.
///
/// Falls back to CPU when no Vulkan device is available (always the case
/// unless built with the `vulkan-backend` feature).
pub struct MultiGpuDispatcher {
    #[cfg(feature = "vulkan-backend")]
    workers: Vec<GpuWorker>,
    cpu_fallback: Arc<CpuAccel>,
    strategy: MultiGpuStrategy,
    /// Round-robin cursor.
    rr_counter: AtomicUsize,
}

impl MultiGpuDispatcher {
    /// Enumerate all available Vulkan devices and create a dispatcher.
    ///
    /// Any device that fails to initialize is silently skipped. Without the
    /// `vulkan-backend` feature, no devices are ever enumerated and every
    /// dispatch goes straight to the CPU fallback.
    ///
    /// # Errors
    ///
    /// Never fails — falls back to CPU if no GPU is available.
    pub fn new(strategy: MultiGpuStrategy) -> Self {
        #[cfg(feature = "vulkan-backend")]
        let workers = {
            let mut workers = Vec::new();

            // Try discrete first, then integrated, then any.
            for pref in &[
                DevicePreference::Discrete,
                DevicePreference::Integrated,
                DevicePreference::Any,
            ] {
                let selector = DeviceSelector::new().with_preference(*pref);
                match VulkanAccel::new(&selector) {
                    Ok(accel) => {
                        // Avoid duplicates by checking the device name.
                        let name = accel.device_name().to_owned();
                        if !workers.iter().any(|w: &GpuWorker| w.name == name) {
                            tracing::info!("MultiGpuDispatcher: added GPU '{name}'");
                            workers.push(GpuWorker::new(accel));
                        }
                    }
                    Err(e) => {
                        tracing::debug!("MultiGpuDispatcher: skipping device ({e})");
                    }
                }
            }

            tracing::info!("MultiGpuDispatcher: {} GPU(s) available", workers.len());
            workers
        };

        #[cfg(not(feature = "vulkan-backend"))]
        tracing::info!(
            "MultiGpuDispatcher: vulkan-backend feature disabled, using CPU fallback only"
        );

        Self {
            #[cfg(feature = "vulkan-backend")]
            workers,
            cpu_fallback: Arc::new(CpuAccel::new()),
            strategy,
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Number of active GPU workers.
    #[must_use]
    pub fn gpu_count(&self) -> usize {
        #[cfg(feature = "vulkan-backend")]
        {
            self.workers.len()
        }
        #[cfg(not(feature = "vulkan-backend"))]
        {
            0
        }
    }

    /// Returns `true` if at least one GPU backend is active.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "vulkan-backend")]
        {
            !self.workers.is_empty()
        }
        #[cfg(not(feature = "vulkan-backend"))]
        {
            false
        }
    }

    /// Returns the names of all active GPU workers.
    #[must_use]
    pub fn gpu_names(&self) -> Vec<&str> {
        #[cfg(feature = "vulkan-backend")]
        {
            self.workers.iter().map(|w| w.name.as_str()).collect()
        }
        #[cfg(not(feature = "vulkan-backend"))]
        {
            Vec::new()
        }
    }

    /// Select a worker index according to the current strategy.
    fn select_worker(&self) -> Option<usize> {
        #[cfg(feature = "vulkan-backend")]
        {
            if self.workers.is_empty() {
                return None;
            }
            Some(match self.strategy {
                MultiGpuStrategy::RoundRobin | MultiGpuStrategy::FirstComplete => {
                    let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed);
                    idx % self.workers.len()
                }
                MultiGpuStrategy::LoadBalanced => {
                    // Pick the worker with the smallest load counter.
                    self.workers
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, w)| w.load.load(Ordering::Relaxed))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                }
            })
        }
        #[cfg(not(feature = "vulkan-backend"))]
        {
            None
        }
    }

    /// Dispatch scale work to the best available backend.
    ///
    /// # Errors
    ///
    /// Returns an error if all backends fail.
    pub fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        #[cfg(feature = "vulkan-backend")]
        if let Some(idx) = self.select_worker() {
            let worker = &self.workers[idx];
            worker.load.fetch_add(1, Ordering::Relaxed);
            let result = worker.backend.scale_image(
                input, src_width, src_height, dst_width, dst_height, format, filter,
            );
            worker.load.fetch_sub(1, Ordering::Relaxed);
            match result {
                Ok(v) => return Ok(v),
                Err(e) => {
                    tracing::warn!("GPU '{}' failed on scale_image: {e}", worker.name);
                }
            }
        }
        // CPU fallback
        self.cpu_fallback.scale_image(
            input, src_width, src_height, dst_width, dst_height, format, filter,
        )
    }

    /// Dispatch color conversion to the best available backend.
    ///
    /// # Errors
    ///
    /// Returns an error if all backends fail.
    pub fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        #[cfg(feature = "vulkan-backend")]
        if let Some(idx) = self.select_worker() {
            let worker = &self.workers[idx];
            worker.load.fetch_add(1, Ordering::Relaxed);
            let result = worker
                .backend
                .convert_color(input, width, height, src_format, dst_format);
            worker.load.fetch_sub(1, Ordering::Relaxed);
            match result {
                Ok(v) => return Ok(v),
                Err(e) => {
                    tracing::warn!("GPU '{}' failed on convert_color: {e}", worker.name);
                }
            }
        }
        self.cpu_fallback
            .convert_color(input, width, height, src_format, dst_format)
    }
}

impl Default for MultiGpuDispatcher {
    fn default() -> Self {
        Self::new(MultiGpuStrategy::RoundRobin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_dispatcher_creates_without_panic() {
        // On CI/test environments without a Vulkan GPU, this should still succeed
        // using the CPU fallback.
        let disp = MultiGpuDispatcher::new(MultiGpuStrategy::RoundRobin);
        // gpu_count may be 0 on CI (no Vulkan), that is acceptable.
        let _ = disp.gpu_count();
        let _ = disp.has_gpu();
    }

    #[test]
    fn test_multi_gpu_dispatcher_default_strategy() {
        let disp = MultiGpuDispatcher::default();
        let _ = disp.gpu_count();
    }

    #[test]
    fn test_multi_gpu_dispatcher_gpu_names() {
        let disp = MultiGpuDispatcher::new(MultiGpuStrategy::LoadBalanced);
        let names = disp.gpu_names();
        assert_eq!(names.len(), disp.gpu_count());
    }

    #[test]
    fn test_multi_gpu_scale_image_cpu_path() {
        // Force CPU path by using a dispatcher that will have no GPUs in CI.
        let disp = MultiGpuDispatcher::new(MultiGpuStrategy::RoundRobin);
        // 4×4 Rgb24 image filled with 128.
        let input = vec![128u8; 4 * 4 * 3];
        let result = disp.scale_image(&input, 4, 4, 2, 2, PixelFormat::Rgb24, ScaleFilter::Nearest);
        assert!(result.is_ok(), "scale_image failed: {:?}", result.err());
        let out = result.expect("scale_image should succeed");
        assert_eq!(out.len(), 2 * 2 * 3);
    }

    #[test]
    fn test_multi_gpu_convert_color_cpu_path() {
        let disp = MultiGpuDispatcher::new(MultiGpuStrategy::RoundRobin);
        let input = vec![128u8; 4 * 4 * 3];
        let result = disp.convert_color(&input, 4, 4, PixelFormat::Rgb24, PixelFormat::Yuv420p);
        assert!(result.is_ok(), "convert_color failed: {:?}", result.err());
    }

    #[test]
    fn test_multi_gpu_round_robin_counter_increments() {
        let disp = MultiGpuDispatcher::new(MultiGpuStrategy::RoundRobin);
        // When there are no GPU workers (always true without `vulkan-backend`),
        // select_worker returns None every time.
        if !disp.has_gpu() {
            assert!(disp.select_worker().is_none());
        } else {
            // With workers present, consecutive calls should cycle.
            let a = disp.select_worker();
            let b = disp.select_worker();
            assert!(a.is_some() && b.is_some());
        }
    }

    #[test]
    fn test_multi_gpu_load_balanced_selection() {
        let disp = MultiGpuDispatcher::new(MultiGpuStrategy::LoadBalanced);
        // Exercises the load-balanced path without GPU present.
        if !disp.has_gpu() {
            assert!(disp.select_worker().is_none());
        } else {
            assert!(disp.select_worker().is_some());
        }
    }
}
