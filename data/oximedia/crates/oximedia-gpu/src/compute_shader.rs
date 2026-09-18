//! GPU compute shader simulator.
//!
//! Simulates GPU-style work-group execution using Rayon thread-pool parallelism.
//! Each [`ShaderKernel`] receives a [`ThreadGroupContext`] per element that
//! mirrors the `gl_GlobalInvocationID` / `gl_LocalInvocationID` / `gl_WorkGroupID`
//! semantics of GLSL/HLSL compute shaders.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::compute_shader::{ComputeShaderSimulator, ThreadGroupContext};
//!
//! let sim = ComputeShaderSimulator::new(64);
//! let kernel = sim.create_kernel("double", |ctx: &ThreadGroupContext, v: &mut u32| {
//!     *v = *v * 2;
//! });
//!
//! let mut data = vec![1u32, 2, 3, 4];
//! let work_groups = (data.len() + sim.default_group_size() - 1) / sim.default_group_size();
//! kernel.execute(&mut data, work_groups);
//! assert_eq!(data, [2, 4, 6, 8]);
//! ```

use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors returned by compute shader operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ShaderError {
    /// The requested group size is zero or otherwise invalid.
    #[error("Invalid group size: {0}")]
    InvalidGroupSize(String),
    /// The data slice passed to the kernel is empty.
    #[error("Data slice is empty")]
    EmptyData,
    /// A kernel closure panicked during execution.
    #[error("Kernel panicked: {0}")]
    KernelPanic(String),
}

// ─── ThreadGroupContext ───────────────────────────────────────────────────────

/// Execution context passed to each invocation of a kernel closure.
///
/// Mirrors the built-in variables available in GLSL/HLSL/WGSL compute shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadGroupContext {
    /// Index of the work group this thread belongs to (`gl_WorkGroupID`).
    pub group_id: usize,
    /// Thread index within the work group (`gl_LocalInvocationID`).
    pub local_id: usize,
    /// Number of threads per work group.
    pub group_size: usize,
    /// Flat global index across all work groups (`gl_GlobalInvocationID`).
    ///
    /// Equals `group_id * group_size + local_id`.
    pub global_id: usize,
}

impl ThreadGroupContext {
    /// Construct a context from its constituent indices.
    #[must_use]
    pub fn new(group_id: usize, local_id: usize, group_size: usize) -> Self {
        Self {
            group_id,
            local_id,
            group_size,
            global_id: group_id * group_size + local_id,
        }
    }
}

// ─── ShaderKernel ─────────────────────────────────────────────────────────────

/// Type-erased kernel function: closure that receives context + mutable element.
type KernelFn<T> = Arc<dyn Fn(&ThreadGroupContext, &mut T) + Send + Sync>;

/// A named, parameterised GPU kernel ready for parallel dispatch.
pub struct ShaderKernel<T: Send + Sync> {
    kernel_fn: KernelFn<T>,
    group_size: usize,
    name: String,
}

impl<T: Send + Sync> ShaderKernel<T> {
    /// Create a new kernel with an explicit group size.
    ///
    /// # Panics
    ///
    /// Does not panic; `group_size = 0` is normalised to 1 at runtime.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        group_size: usize,
        f: impl Fn(&ThreadGroupContext, &mut T) + Send + Sync + 'static,
    ) -> Self {
        Self {
            kernel_fn: Arc::new(f),
            group_size: group_size.max(1),
            name: name.into(),
        }
    }

    /// Execute the kernel over `data` using `work_groups` groups in parallel.
    ///
    /// Elements at indices `>= work_groups * group_size` are silently ignored,
    /// matching GPU semantics where excess invocations are masked out.
    pub fn execute(&self, data: &mut [T], work_groups: usize) {
        if data.is_empty() || work_groups == 0 {
            return;
        }
        let gs = self.group_size;
        let kfn = Arc::clone(&self.kernel_fn);

        data.par_iter_mut().enumerate().for_each(|(i, elem)| {
            let group_id = i / gs;
            let local_id = i % gs;
            // Only process elements within the declared work-group count.
            if group_id < work_groups {
                let ctx = ThreadGroupContext::new(group_id, local_id, gs);
                kfn(&ctx, elem);
            }
        });
    }

    /// The number of threads per work group.
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// The human-readable label for this kernel.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ─── DispatchConfig ──────────────────────────────────────────────────────────

/// Configuration bundle for a single kernel dispatch.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Number of work groups to dispatch.
    pub work_groups: usize,
    /// Threads per work group (overrides kernel default when > 0).
    pub group_size: usize,
    /// Human-readable label used for profiling/logging.
    pub label: String,
}

impl DispatchConfig {
    /// Convenience constructor.
    #[must_use]
    pub fn new(work_groups: usize, group_size: usize, label: impl Into<String>) -> Self {
        Self {
            work_groups,
            group_size,
            label: label.into(),
        }
    }
}

// ─── ComputeShaderSimulator ───────────────────────────────────────────────────

/// High-level entry point for simulated GPU compute.
///
/// Manages a default group size and provides factory methods for creating
/// typed [`ShaderKernel`] instances.
#[derive(Debug, Clone)]
pub struct ComputeShaderSimulator {
    default_group_size: usize,
}

impl ComputeShaderSimulator {
    /// Create a simulator with the given default work-group size.
    ///
    /// Sizes of 0 are normalised to 64 (a common GPU default).
    #[must_use]
    pub fn new(default_group_size: usize) -> Self {
        Self {
            default_group_size: if default_group_size == 0 {
                64
            } else {
                default_group_size
            },
        }
    }

    /// The default number of threads per work group.
    #[must_use]
    pub fn default_group_size(&self) -> usize {
        self.default_group_size
    }

    /// Create a kernel using the simulator's default group size.
    #[must_use]
    pub fn create_kernel<T: Send + Sync + 'static>(
        &self,
        name: impl Into<String>,
        f: impl Fn(&ThreadGroupContext, &mut T) + Send + Sync + 'static,
    ) -> ShaderKernel<T> {
        ShaderKernel::new(name, self.default_group_size, f)
    }

    /// Create a kernel with a custom group size, ignoring the simulator default.
    #[must_use]
    pub fn create_kernel_with_group_size<T: Send + Sync + 'static>(
        &self,
        name: impl Into<String>,
        group_size: usize,
        f: impl Fn(&ThreadGroupContext, &mut T) + Send + Sync + 'static,
    ) -> ShaderKernel<T> {
        ShaderKernel::new(name, group_size, f)
    }

    /// Dispatch `kernel` over `data` using `work_groups` work groups.
    pub fn dispatch<T: Send + Sync>(
        &self,
        kernel: &ShaderKernel<T>,
        data: &mut [T],
        work_groups: usize,
    ) {
        kernel.execute(data, work_groups);
    }

    /// Dispatch `kernel` and wait for all threads to complete.
    ///
    /// Rayon's `par_iter_mut` already joins all threads before returning, so
    /// this is semantically equivalent to [`dispatch`].  The method exists to
    /// model GPU barriers explicitly in calling code.
    ///
    /// [`dispatch`]: ComputeShaderSimulator::dispatch
    pub fn dispatch_with_barrier<T: Send + Sync + Clone>(
        &self,
        kernel: &ShaderKernel<T>,
        data: &mut [T],
        work_groups: usize,
    ) {
        // Rayon join semantics: all parallel work completes before the call
        // returns — no additional synchronisation primitive needed.
        kernel.execute(data, work_groups);
        // Conceptual barrier point; Rayon's fork-join ensures this.
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn work_groups_for(len: usize, group_size: usize) -> usize {
        (len + group_size - 1) / group_size
    }

    // ── ThreadGroupContext ────────────────────────────────────────────────────

    #[test]
    fn test_thread_group_context_global_id() {
        let ctx = ThreadGroupContext::new(3, 5, 8);
        assert_eq!(ctx.group_id, 3);
        assert_eq!(ctx.local_id, 5);
        assert_eq!(ctx.group_size, 8);
        assert_eq!(ctx.global_id, 3 * 8 + 5);
    }

    #[test]
    fn test_thread_group_context_zero_group() {
        let ctx = ThreadGroupContext::new(0, 0, 64);
        assert_eq!(ctx.global_id, 0);
    }

    // ── ShaderKernel ─────────────────────────────────────────────────────────

    #[test]
    fn test_shader_kernel_name_and_group_size() {
        let k = ShaderKernel::new(
            "test_kernel",
            32,
            |_ctx: &ThreadGroupContext, _v: &mut u32| {},
        );
        assert_eq!(k.name(), "test_kernel");
        assert_eq!(k.group_size(), 32);
    }

    #[test]
    fn test_shader_kernel_group_size_zero_normalised() {
        let k = ShaderKernel::new("k", 0, |_ctx: &ThreadGroupContext, _v: &mut u32| {});
        assert_eq!(k.group_size(), 1);
    }

    #[test]
    fn test_execute_multiply_by_two() {
        let k = ShaderKernel::new("double", 4, |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v *= 2;
        });
        let mut data = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let wg = work_groups_for(data.len(), 4);
        k.execute(&mut data, wg);
        assert_eq!(data, [2, 4, 6, 8, 10, 12, 14, 16]);
    }

    #[test]
    fn test_execute_fill_with_global_id() {
        let k = ShaderKernel::new("fill_id", 8, |ctx: &ThreadGroupContext, v: &mut usize| {
            *v = ctx.global_id;
        });
        let mut data = vec![0usize; 16];
        let wg = work_groups_for(data.len(), 8);
        k.execute(&mut data, wg);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i, "element {i} should equal its global_id");
        }
    }

    #[test]
    fn test_execute_work_groups_larger_than_needed() {
        // Extra work groups simply have no data elements to process.
        let k = ShaderKernel::new("k", 4, |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v += 10;
        });
        let mut data = vec![0u32; 6]; // 6 elements, group_size=4 → 2 groups
        k.execute(&mut data, 100); // 100 work groups requested – fine
        assert!(data.iter().all(|&v| v == 10));
    }

    #[test]
    fn test_execute_single_work_group() {
        let k = ShaderKernel::new("k", 8, |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v = 42;
        });
        let mut data = vec![0u32; 8];
        k.execute(&mut data, 1);
        assert!(data.iter().all(|&v| v == 42));
    }

    #[test]
    fn test_execute_empty_data_no_panic() {
        let k = ShaderKernel::new("k", 8, |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v = 1;
        });
        let mut data: Vec<u32> = vec![];
        // Must not panic
        k.execute(&mut data, 4);
        assert!(data.is_empty());
    }

    #[test]
    fn test_execute_f32_scale() {
        let factor = 2.5_f32;
        let k = ShaderKernel::new(
            "scale_f32",
            4,
            move |_ctx: &ThreadGroupContext, v: &mut f32| {
                *v *= factor;
            },
        );
        let mut data = vec![1.0_f32, 2.0, 3.0, 4.0];
        k.execute(&mut data, 1);
        for (i, &v) in data.iter().enumerate() {
            let expected = (i as f32 + 1.0) * factor;
            assert!(
                (v - expected).abs() < 1e-5,
                "element {i}: got {v}, expected {expected}"
            );
        }
    }

    // ── ComputeShaderSimulator ────────────────────────────────────────────────

    #[test]
    fn test_simulator_default_group_size() {
        let sim = ComputeShaderSimulator::new(64);
        assert_eq!(sim.default_group_size(), 64);
    }

    #[test]
    fn test_simulator_zero_group_size_normalised() {
        let sim = ComputeShaderSimulator::new(0);
        assert_eq!(sim.default_group_size(), 64);
    }

    #[test]
    fn test_simulator_create_kernel_and_dispatch() {
        let sim = ComputeShaderSimulator::new(4);
        let kernel = sim.create_kernel("incr", |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v += 1;
        });
        let mut data = vec![0u32; 8];
        let wg = work_groups_for(data.len(), sim.default_group_size());
        sim.dispatch(&kernel, &mut data, wg);
        assert!(data.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_simulator_create_kernel_with_group_size() {
        let sim = ComputeShaderSimulator::new(64);
        let kernel = sim.create_kernel_with_group_size(
            "k16",
            16,
            |_ctx: &ThreadGroupContext, v: &mut u32| {
                *v = 99;
            },
        );
        assert_eq!(kernel.group_size(), 16);
        let mut data = vec![0u32; 32];
        let wg = work_groups_for(data.len(), 16);
        kernel.execute(&mut data, wg);
        assert!(data.iter().all(|&v| v == 99));
    }

    #[test]
    fn test_dispatch_with_barrier() {
        let sim = ComputeShaderSimulator::new(8);
        let k = sim.create_kernel("b_k", |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v = 7;
        });
        let mut data = vec![0u32; 8];
        sim.dispatch_with_barrier(&k, &mut data, 1);
        assert!(data.iter().all(|&v| v == 7));
    }

    #[test]
    fn test_multiple_kernels_on_same_data() {
        let sim = ComputeShaderSimulator::new(4);
        let k1 = sim.create_kernel("add1", |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v += 1;
        });
        let k2 = sim.create_kernel("mul3", |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v *= 3;
        });
        let mut data = vec![0u32; 4];
        let wg = 1;
        sim.dispatch(&k1, &mut data, wg);
        sim.dispatch(&k2, &mut data, wg);
        // Each element: (0+1)*3 = 3
        assert!(data.iter().all(|&v| v == 3));
    }

    #[test]
    fn test_large_data_set() {
        let sim = ComputeShaderSimulator::new(64);
        let k = sim.create_kernel("large", |_ctx: &ThreadGroupContext, v: &mut u32| {
            *v += 1;
        });
        let mut data = vec![0u32; 10_000];
        let wg = work_groups_for(data.len(), sim.default_group_size());
        sim.dispatch(&k, &mut data, wg);
        assert!(data.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_kernel_captures_closure_state_with_atomic() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let sim = ComputeShaderSimulator::new(8);
        let k = sim.create_kernel(
            "counter_k",
            move |_ctx: &ThreadGroupContext, _v: &mut u32| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            },
        );
        let mut data = vec![0u32; 16];
        let wg = work_groups_for(data.len(), sim.default_group_size());
        k.execute(&mut data, wg);
        assert_eq!(counter.load(Ordering::Relaxed), 16);
    }
}
