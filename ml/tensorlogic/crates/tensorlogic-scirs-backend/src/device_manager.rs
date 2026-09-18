//! Operation-level device selection and management.
//!
//! This module provides a pluggable device-selection framework that decides
//! *per operation* whether to execute on CPU or GPU, based on tensor shape,
//! operation kind, and hardware availability.
//!
//! ## Architecture
//!
//! ```text
//! DeviceManager          — owns a Box<dyn DeviceSelector>
//!   └─ DeviceSelector    — trait: select(op, shape) → Device
//!        └─ HeuristicSelector — GPU iff available ∧ large ∧ gpu-friendly op
//! ```
//!
//! ## Quick start
//!
//! ```rust
//! use tensorlogic_scirs_backend::device_manager::{
//!     DeviceConfig, DeviceManager, OpDescriptor, OpKind,
//! };
//!
//! let config = DeviceConfig::default().with_gpu_available(true).with_gpu_threshold(1_048_576);
//! let mgr = DeviceManager::with_heuristic(config);
//!
//! let op = OpDescriptor { kind: OpKind::MatMul };
//! let large_shape = [1024_usize, 1024];
//! let device = mgr.select(&op, &large_shape);
//! // → Gpu(0) when GPU is available and shape product ≥ threshold
//! ```

use crate::device::{Device, DeviceType};

// ──────────────────────────────────────────────
// OpKind
// ──────────────────────────────────────────────

/// Describes the kind of compute operation, used by the device selector
/// heuristic to decide whether GPU execution is beneficial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// Dense matrix multiplication / tensor contraction.
    MatMul,

    /// Element-wise operations (add, relu, sigmoid, …).
    Elementwise,

    /// Reduction operations (sum, max, mean, …).
    Reduce,

    /// Any other operation type.
    Other,
}

impl OpKind {
    /// Returns `true` when this kind of operation is well-suited for GPU
    /// execution (high arithmetic intensity, large memory bandwidth demands).
    ///
    /// Currently `MatMul` and `Elementwise` are considered GPU-friendly.
    pub fn is_gpu_friendly(self) -> bool {
        matches!(self, OpKind::MatMul | OpKind::Elementwise)
    }
}

// ──────────────────────────────────────────────
// OpDescriptor
// ──────────────────────────────────────────────

/// Descriptor passed to the [`DeviceSelector`] for each scheduled operation.
///
/// Callers can extend this with additional fields in future without breaking
/// implementations that only inspect `kind`.
#[derive(Debug, Clone)]
pub struct OpDescriptor {
    /// The high-level kind of the operation.
    pub kind: OpKind,
}

// ──────────────────────────────────────────────
// DeviceSelector trait
// ──────────────────────────────────────────────

/// Trait for selecting a compute [`Device`] for a given operation.
///
/// Implementors decide, given an [`OpDescriptor`] and the tensor shape,
/// which device should execute the operation.  The returned device must
/// be valid for the current system; callers are free to treat an invalid
/// device as an error.
///
/// # Thread safety
///
/// Implementations must be `Send + Sync` so that [`DeviceManager`] can be
/// shared across threads.
pub trait DeviceSelector: Send + Sync {
    /// Select the best device for an operation described by `op` acting on a
    /// tensor with the given `shape`.
    fn select(&self, op: &OpDescriptor, shape: &[usize]) -> Device;
}

// ──────────────────────────────────────────────
// DeviceConfig
// ──────────────────────────────────────────────

/// Configuration for the built-in [`HeuristicSelector`].
///
/// Use the builder methods to customise thresholds and forced-device overrides.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_scirs_backend::device_manager::DeviceConfig;
///
/// // Enable GPU when tensors have ≥ 4 M elements
/// let cfg = DeviceConfig::default()
///     .with_gpu_available(true)
///     .with_gpu_threshold(4_194_304);
/// ```
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Minimum number of tensor elements required to consider GPU execution.
    gpu_threshold_elems: usize,

    /// Whether a GPU device is actually available on this machine.
    gpu_available: bool,

    /// Index of the GPU to target (used only when a GPU is selected).
    gpu_index: u32,

    /// When `Some`, always return this device regardless of other settings.
    forced: Option<ForcedDevice>,
}

/// Internal forced-device discriminant to avoid storing a full `Device` clone
/// (which is not `Copy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForcedDevice {
    Cpu,
    Gpu(u32),
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            gpu_threshold_elems: 1_048_576, // 1 M elements
            gpu_available: false,
            gpu_index: 0,
            forced: None,
        }
    }
}

impl DeviceConfig {
    /// Set the element count threshold above which GPU execution is considered.
    ///
    /// Tensors with fewer than `n` elements will always run on CPU regardless
    /// of GPU availability.
    pub fn with_gpu_threshold(mut self, n: usize) -> Self {
        self.gpu_threshold_elems = n;
        self
    }

    /// Declare whether a GPU is available on the current system.
    pub fn with_gpu_available(mut self, avail: bool) -> Self {
        self.gpu_available = avail;
        self
    }

    /// Force all operations to run on CPU, overriding every other setting.
    pub fn force_cpu(mut self) -> Self {
        self.forced = Some(ForcedDevice::Cpu);
        self
    }

    /// Force all operations to run on the GPU with the given device index,
    /// overriding every other setting.
    pub fn force_gpu(mut self, idx: u32) -> Self {
        self.forced = Some(ForcedDevice::Gpu(idx));
        self
    }

    /// Set the GPU device index used when GPU execution is selected.
    pub fn with_gpu_index(mut self, idx: u32) -> Self {
        self.gpu_index = idx;
        self
    }
}

// ──────────────────────────────────────────────
// HeuristicSelector
// ──────────────────────────────────────────────

/// A heuristic [`DeviceSelector`] that routes ops to GPU when three conditions
/// are simultaneously satisfied:
///
/// 1. `config.gpu_available` is `true`.
/// 2. The tensor element count (`shape.iter().product()`) is ≥
///    `config.gpu_threshold_elems`.
/// 3. `op.kind.is_gpu_friendly()` returns `true`.
///
/// Any `force_cpu` / `force_gpu` override in [`DeviceConfig`] takes
/// precedence over all three conditions.
pub struct HeuristicSelector {
    config: DeviceConfig,
}

impl HeuristicSelector {
    /// Create a new `HeuristicSelector` from the given configuration.
    pub fn new(config: DeviceConfig) -> Self {
        Self { config }
    }
}

impl DeviceSelector for HeuristicSelector {
    fn select(&self, op: &OpDescriptor, shape: &[usize]) -> Device {
        // Forced override wins unconditionally.
        if let Some(forced) = self.config.forced {
            return match forced {
                ForcedDevice::Cpu => Device::cpu(),
                ForcedDevice::Gpu(idx) => Device {
                    device_type: DeviceType::Cuda,
                    index: idx as usize,
                },
            };
        }

        let n_elems: usize = shape.iter().product();

        if self.config.gpu_available
            && n_elems >= self.config.gpu_threshold_elems
            && op.kind.is_gpu_friendly()
        {
            Device {
                device_type: DeviceType::Cuda,
                index: self.config.gpu_index as usize,
            }
        } else {
            Device::cpu()
        }
    }
}

// ──────────────────────────────────────────────
// DeviceManager
// ──────────────────────────────────────────────

/// Operation-level device manager wrapping a pluggable [`DeviceSelector`].
///
/// `DeviceManager` is the public entry point for the device-selection framework.
/// Callers construct one with a selector of their choice (or use
/// [`DeviceManager::with_heuristic`] for the built-in heuristic), then call
/// [`DeviceManager::select`] once per scheduled operation.
///
/// # Examples
///
/// ```rust
/// use tensorlogic_scirs_backend::device_manager::{
///     DeviceConfig, DeviceManager, OpDescriptor, OpKind,
/// };
///
/// let mgr = DeviceManager::with_heuristic(DeviceConfig::default());
/// let op  = OpDescriptor { kind: OpKind::MatMul };
/// let dev = mgr.select(&op, &[32, 32]);
/// assert!(dev.is_cpu()); // GPU not available by default
/// ```
pub struct DeviceManager {
    selector: Box<dyn DeviceSelector>,
}

impl DeviceManager {
    /// Create a `DeviceManager` backed by any [`DeviceSelector`] implementation.
    pub fn new(selector: impl DeviceSelector + 'static) -> Self {
        Self {
            selector: Box::new(selector),
        }
    }

    /// Create a `DeviceManager` backed by the built-in [`HeuristicSelector`]
    /// configured with `config`.
    pub fn with_heuristic(config: DeviceConfig) -> Self {
        Self::new(HeuristicSelector::new(config))
    }

    /// Select the compute device for an operation described by `op` acting on
    /// a tensor with the given `shape`.
    pub fn select(&self, op: &OpDescriptor, shape: &[usize]) -> Device {
        self.selector.select(op, shape)
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a DeviceConfig with GPU available at default threshold (1 M).
    fn gpu_config() -> DeviceConfig {
        DeviceConfig::default().with_gpu_available(true)
    }

    // Helper: build a tiny shape (10 elements total).
    fn tiny_shape() -> [usize; 2] {
        [2, 5] // 10 elements
    }

    // Helper: build a large shape (2 M elements).
    fn large_shape() -> [usize; 2] {
        [1024, 2048] // 2 097 152 elements > 1 M threshold
    }

    // ── OpKind ──────────────────────────────────────────────────────────────

    #[test]
    fn test_op_kind_gpu_friendly() {
        assert!(OpKind::MatMul.is_gpu_friendly());
        assert!(OpKind::Elementwise.is_gpu_friendly());
        assert!(!OpKind::Reduce.is_gpu_friendly());
        assert!(!OpKind::Other.is_gpu_friendly());
    }

    // ── Heuristic: tiny tensor → CPU even when GPU available ─────────────

    #[test]
    fn test_tiny_tensor_routes_to_cpu() {
        let mgr = DeviceManager::with_heuristic(gpu_config());
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let dev = mgr.select(&op, &tiny_shape());
        assert!(dev.is_cpu(), "tiny tensor should use CPU");
    }

    // ── Heuristic: large + gpu_available + MatMul → Gpu ─────────────────

    #[test]
    fn test_large_matmul_routes_to_gpu_when_available() {
        let mgr = DeviceManager::with_heuristic(gpu_config());
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let dev = mgr.select(&op, &large_shape());
        assert!(
            dev.is_gpu(),
            "large MatMul with GPU available should use GPU"
        );
    }

    // ── Heuristic: large + gpu_available=false → CPU ─────────────────────

    #[test]
    fn test_large_tensor_cpu_when_gpu_unavailable() {
        let cfg = DeviceConfig::default().with_gpu_available(false);
        let mgr = DeviceManager::with_heuristic(cfg);
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let dev = mgr.select(&op, &large_shape());
        assert!(dev.is_cpu(), "no GPU available → must stay on CPU");
    }

    // ── Heuristic: large + gpu_available + Other kind → CPU ──────────────

    #[test]
    fn test_large_non_gpu_friendly_op_routes_to_cpu() {
        let mgr = DeviceManager::with_heuristic(gpu_config());

        for kind in [OpKind::Reduce, OpKind::Other] {
            let op = OpDescriptor { kind };
            let dev = mgr.select(&op, &large_shape());
            assert!(
                dev.is_cpu(),
                "{kind:?} is not GPU-friendly and should run on CPU"
            );
        }
    }

    // ── force_cpu overrides GPU-eligible combination ──────────────────────

    #[test]
    fn test_force_cpu_overrides_gpu_eligible() {
        let cfg = gpu_config().force_cpu();
        let mgr = DeviceManager::with_heuristic(cfg);
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let dev = mgr.select(&op, &large_shape());
        assert!(dev.is_cpu(), "force_cpu must override GPU eligibility");
    }

    // ── force_gpu overrides CPU-only config ──────────────────────────────

    #[test]
    fn test_force_gpu_overrides_cpu_config() {
        // GPU is not "available" and tensor is tiny, but force_gpu wins.
        let cfg = DeviceConfig::default()
            .with_gpu_available(false)
            .force_gpu(0);
        let mgr = DeviceManager::with_heuristic(cfg);
        let op = OpDescriptor {
            kind: OpKind::Other,
        };
        let dev = mgr.select(&op, &tiny_shape());
        assert!(dev.is_gpu(), "force_gpu must override all other conditions");
    }

    // ── Elementwise large tensor + GPU available ─────────────────────────

    #[test]
    fn test_large_elementwise_routes_to_gpu() {
        let mgr = DeviceManager::with_heuristic(gpu_config());
        let op = OpDescriptor {
            kind: OpKind::Elementwise,
        };
        let dev = mgr.select(&op, &large_shape());
        assert!(
            dev.is_gpu(),
            "large Elementwise with GPU available should use GPU"
        );
    }

    // ── Custom selector injection ─────────────────────────────────────────

    #[test]
    fn test_custom_selector_always_cpu() {
        struct AlwaysCpu;
        impl DeviceSelector for AlwaysCpu {
            fn select(&self, _op: &OpDescriptor, _shape: &[usize]) -> Device {
                Device::cpu()
            }
        }

        let mgr = DeviceManager::new(AlwaysCpu);
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let dev = mgr.select(&op, &large_shape());
        assert!(dev.is_cpu(), "custom selector should override heuristic");
    }

    // ── DeviceConfig builder API ──────────────────────────────────────────

    #[test]
    fn test_device_config_builder_threshold() {
        // Tensor with 512 elements, threshold set to 256 → should go to GPU.
        let cfg = DeviceConfig::default()
            .with_gpu_available(true)
            .with_gpu_threshold(256);
        let mgr = DeviceManager::with_heuristic(cfg);
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let shape = [16_usize, 32]; // 512 elements
        let dev = mgr.select(&op, &shape);
        assert!(dev.is_gpu(), "512 elems > 256 threshold should use GPU");
    }

    #[test]
    fn test_device_config_default_no_gpu() {
        // Default config: gpu_available = false.
        let mgr = DeviceManager::with_heuristic(DeviceConfig::default());
        let op = OpDescriptor {
            kind: OpKind::MatMul,
        };
        let dev = mgr.select(&op, &large_shape());
        assert!(dev.is_cpu(), "default config has no GPU available");
    }
}
