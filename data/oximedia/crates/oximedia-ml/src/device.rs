//! Execution device abstraction.
//!
//! [`DeviceType`] is the user-facing handle for selecting which backend
//! an [`OnnxModel`](crate::model::OnnxModel) should run on. Backends
//! are feature-gated at build time; runtime availability is probed via
//! [`DeviceType::is_available`] so callers can ask for
//! [`DeviceType::auto`] without needing `cfg!` plumbing of their own.
//!
//! ## Probe cascade
//!
//! [`DeviceType::auto`] walks the following order, first success wins:
//!
//! 1. **CUDA** — `oxionnx_cuda::CudaContext::try_new()`
//!    (requires the `cuda` feature).
//! 2. **DirectML** — `oxionnx_directml::DirectMLContext::try_new()`
//!    (requires the `directml` feature; always `None` off Windows).
//! 3. **WebGPU** — `oxionnx_gpu::GpuContext::try_new()`
//!    (requires the `webgpu` feature).
//! 4. **CPU** — always available.
//!
//! Every probe is wrapped in `std::panic::catch_unwind`, so a misbehaving
//! foreign driver can never unwind through our call stack. The result of
//! `auto()` is memoised in a `OnceLock`: calling it twice does not re-init
//! the CUDA driver / wgpu adapter.
//!
//! ## Capability introspection
//!
//! [`DeviceCapabilities`] carries a richer description (device name,
//! memory, compute capability, dtype support) and is produced by
//! [`DeviceCapabilities::probe`] for a specific [`DeviceType`], or
//! [`DeviceCapabilities::probe_all`] for every compiled-in backend at
//! once.
//!
//! ## Example
//!
//! ```
//! use oximedia_ml::{DeviceCapabilities, DeviceType};
//!
//! // Pick the strongest available backend (always succeeds — falls
//! // back to CPU if nothing else is compiled in / usable).
//! let device = DeviceType::auto();
//! assert!(device.is_available());
//!
//! // Full capability report for the selected device.
//! let caps = DeviceCapabilities::best_available();
//! assert_eq!(caps.device_type, device);
//! ```

use core::fmt;
#[cfg(any(feature = "cuda", feature = "webgpu", feature = "directml"))]
use std::panic::AssertUnwindSafe;
use std::sync::OnceLock;

/// Execution backend for ML inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum DeviceType {
    /// Pure-Rust CPU execution (always available).
    Cpu,
    /// NVIDIA CUDA via `oxionnx-cuda` (feature `cuda`).
    Cuda,
    /// WebGPU / wgpu compute backend via `oxionnx-gpu` (feature `webgpu`).
    WebGpu,
    /// Microsoft DirectML (feature `directml`, Windows-only at runtime).
    DirectMl,
    /// Apple CoreML. Reserved variant; no `coreml` feature exists yet, so
    /// this device is never reported as available and `auto()` never
    /// selects it. It exists so API consumers can exhaustively match on
    /// `DeviceType` without having to guess whether CoreML will be added
    /// later.
    CoreMl,
}

/// Memoised result of [`DeviceType::auto`].
///
/// The probe cascade has observable side effects (CUDA driver
/// initialisation, wgpu adapter enumeration), and some of those mutate
/// thread-local state. Caching the first result makes subsequent calls
/// cheap and avoids re-running those side effects.
static AUTO_CACHE: OnceLock<DeviceType> = OnceLock::new();

impl DeviceType {
    /// Return the preferred device available in this build, in the order
    /// **CUDA → DirectML → WebGPU → CPU**. Always succeeds because CPU is
    /// unconditionally available.
    ///
    /// The result is memoised for the lifetime of the process.
    #[must_use]
    pub fn auto() -> Self {
        *AUTO_CACHE.get_or_init(|| {
            if Self::Cuda.is_available() {
                return Self::Cuda;
            }
            if Self::DirectMl.is_available() {
                return Self::DirectMl;
            }
            if Self::WebGpu.is_available() {
                return Self::WebGpu;
            }
            Self::Cpu
        })
    }

    /// Report whether this device is usable in the current build /
    /// runtime environment. A device may be compiled in (feature-gated)
    /// yet still unavailable at runtime — e.g. no GPU detected.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda => cuda_available(),
            Self::WebGpu => webgpu_available(),
            Self::DirectMl => directml_available(),
            Self::CoreMl => false,
        }
    }

    /// Short canonical name matching the feature flag / CLI spelling.
    ///
    /// Retained for backward compatibility with existing call sites;
    /// [`DeviceType::display_name`] is preferred for human-facing output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::WebGpu => "webgpu",
            Self::DirectMl => "directml",
            Self::CoreMl => "coreml",
        }
    }

    /// Human-facing label — identical to [`Self::name`] for now, but
    /// conceptually distinct so downstream UIs can swap in a friendlier
    /// string later without affecting programmatic lookups.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::WebGpu => "WebGPU",
            Self::DirectMl => "DirectML",
            Self::CoreMl => "CoreML",
        }
    }

    /// Run the full cascade probe and return the richer
    /// [`DeviceCapabilities`] record for this device.
    #[must_use]
    pub fn probe_caps(self) -> DeviceCapabilities {
        DeviceCapabilities::probe(self)
    }

    /// Return every [`DeviceType`] whose backend is currently usable.
    ///
    /// The returned vector always contains [`DeviceType::Cpu`] and is
    /// ordered by the probe cascade (CPU last).
    #[must_use]
    pub fn list_available() -> Vec<Self> {
        let mut out = Vec::with_capacity(4);
        if Self::Cuda.is_available() {
            out.push(Self::Cuda);
        }
        if Self::DirectMl.is_available() {
            out.push(Self::DirectMl);
        }
        if Self::WebGpu.is_available() {
            out.push(Self::WebGpu);
        }
        // CoreMl is never currently available.
        out.push(Self::Cpu);
        out
    }

    /// Every variant in enum declaration order.
    ///
    /// Used by [`DeviceCapabilities::probe_all`] and the test suite.
    #[must_use]
    pub const fn all_variants() -> [Self; 5] {
        [
            Self::Cpu,
            Self::Cuda,
            Self::WebGpu,
            Self::DirectMl,
            Self::CoreMl,
        ]
    }
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Rich capability description for a single [`DeviceType`].
///
/// Produced by [`DeviceCapabilities::probe`]. Fields are populated
/// best-effort — anything the backend does not expose is left as `None` /
/// `false`, which lets callers use a single code path regardless of how
/// much telemetry the driver provides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DeviceCapabilities {
    /// Which device this record describes.
    pub device_type: DeviceType,
    /// Whether the device is currently available for inference.
    pub is_available: bool,
    /// Human-facing device name (e.g. "CPU (x86_64)", "NVIDIA GPU via CUDA").
    pub device_name: String,
    /// Total device memory in bytes, if known.
    pub memory_total_bytes: Option<u64>,
    /// Free device memory in bytes, if known.
    pub memory_free_bytes: Option<u64>,
    /// Compute capability string (e.g. "8.6" for Ampere). `None` for CPU
    /// / WebGPU / DirectML / CoreML.
    pub compute_capability: Option<String>,
    /// Whether FP16 (half-precision float) is supported.
    pub supports_fp16: bool,
    /// Whether BF16 (bfloat16) is supported.
    pub supports_bf16: bool,
    /// Whether INT8 quantised inference is supported.
    pub supports_int8: bool,
}

impl DeviceCapabilities {
    /// Probe a specific device and describe its capabilities.
    ///
    /// Always returns a record; unavailable devices get `is_available =
    /// false` with the rest populated from static knowledge.
    #[must_use]
    pub fn probe(device: DeviceType) -> Self {
        match device {
            DeviceType::Cpu => Self {
                device_type: DeviceType::Cpu,
                is_available: true,
                device_name: cpu_device_name(),
                memory_total_bytes: None,
                memory_free_bytes: None,
                compute_capability: None,
                supports_fp16: false,
                supports_bf16: false,
                supports_int8: true,
            },
            DeviceType::Cuda => {
                let live = cuda_available();
                Self {
                    device_type: DeviceType::Cuda,
                    is_available: live,
                    device_name: if live {
                        "NVIDIA GPU via CUDA".to_string()
                    } else {
                        "CUDA (unavailable)".to_string()
                    },
                    memory_total_bytes: None,
                    memory_free_bytes: None,
                    compute_capability: None,
                    supports_fp16: live,
                    supports_bf16: live,
                    supports_int8: live,
                }
            }
            DeviceType::WebGpu => {
                let live = webgpu_available();
                Self {
                    device_type: DeviceType::WebGpu,
                    is_available: live,
                    device_name: if live {
                        "GPU via wgpu".to_string()
                    } else {
                        "WebGPU (unavailable)".to_string()
                    },
                    memory_total_bytes: None,
                    memory_free_bytes: None,
                    compute_capability: None,
                    supports_fp16: false,
                    supports_bf16: false,
                    supports_int8: false,
                }
            }
            DeviceType::DirectMl => {
                let live = directml_available();
                Self {
                    device_type: DeviceType::DirectMl,
                    is_available: live,
                    device_name: if live {
                        "GPU via DirectML".to_string()
                    } else {
                        "DirectML (unavailable)".to_string()
                    },
                    memory_total_bytes: None,
                    memory_free_bytes: None,
                    compute_capability: None,
                    supports_fp16: live,
                    supports_bf16: false,
                    supports_int8: live,
                }
            }
            DeviceType::CoreMl => Self {
                device_type: DeviceType::CoreMl,
                is_available: false,
                device_name: "CoreML (not yet supported)".to_string(),
                memory_total_bytes: None,
                memory_free_bytes: None,
                compute_capability: None,
                supports_fp16: false,
                supports_bf16: false,
                supports_int8: false,
            },
        }
    }

    /// Probe every [`DeviceType`] variant and return a record per device.
    ///
    /// The resulting vector has exactly [`DeviceType::all_variants`] entries
    /// and is ordered to match that array.
    #[must_use]
    pub fn probe_all() -> Vec<Self> {
        DeviceType::all_variants()
            .iter()
            .copied()
            .map(Self::probe)
            .collect()
    }

    /// Return capabilities for the best currently-available device.
    ///
    /// Equivalent to `DeviceCapabilities::probe(DeviceType::auto())`, but
    /// exposed as a named constructor for callers that only need the
    /// capability record.
    #[must_use]
    pub fn best_available() -> Self {
        Self::probe(DeviceType::auto())
    }
}

impl fmt::Display for DeviceCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}]",
            self.device_name,
            if self.is_available {
                "available"
            } else {
                "unavailable"
            }
        )
    }
}

// ---------------------------------------------------------------------------
// Probe primitives
// ---------------------------------------------------------------------------

/// Run a probe closure under `catch_unwind` so a foreign panic cannot
/// unwind through our caller. A panicking probe is treated as
/// "unavailable".
///
/// Only compiled when at least one foreign probe is enabled, otherwise
/// every `*_available()` function is a constant `false` and this helper
/// would be dead code.
#[cfg(any(feature = "cuda", feature = "webgpu", feature = "directml"))]
fn safe_probe<F: FnOnce() -> bool>(probe: F) -> bool {
    std::panic::catch_unwind(AssertUnwindSafe(probe)).unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    safe_probe(|| oxionnx::cuda::CudaContext::try_new().is_some())
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

#[cfg(feature = "webgpu")]
fn webgpu_available() -> bool {
    safe_probe(|| oxionnx::gpu::GpuContext::try_new().is_some())
}

#[cfg(not(feature = "webgpu"))]
fn webgpu_available() -> bool {
    false
}

#[cfg(feature = "directml")]
fn directml_available() -> bool {
    safe_probe(|| oxionnx::directml::DirectMLContext::try_new().is_some())
}

#[cfg(not(feature = "directml"))]
fn directml_available() -> bool {
    false
}

/// Best-effort CPU description string — architecture and pointer width.
fn cpu_device_name() -> String {
    format!(
        "CPU ({}-{})",
        std::env::consts::ARCH,
        core::mem::size_of::<usize>() * 8
    )
}

// ---------------------------------------------------------------------------
// Tests (pure unit tests — heavier synthetic tests live under tests/)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_available() {
        assert!(DeviceType::Cpu.is_available());
        assert_eq!(DeviceType::Cpu.name(), "cpu");
    }

    #[test]
    fn auto_returns_available_device() {
        let device = DeviceType::auto();
        assert!(device.is_available());
    }

    #[test]
    fn default_is_cpu() {
        assert_eq!(DeviceType::default(), DeviceType::Cpu);
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(format!("{}", DeviceType::Cpu), "cpu");
        assert_eq!(format!("{}", DeviceType::Cuda), "cuda");
        assert_eq!(format!("{}", DeviceType::WebGpu), "webgpu");
        assert_eq!(format!("{}", DeviceType::DirectMl), "directml");
        assert_eq!(format!("{}", DeviceType::CoreMl), "coreml");
    }

    #[test]
    fn display_names_are_stable() {
        assert_eq!(DeviceType::Cpu.display_name(), "CPU");
        assert_eq!(DeviceType::Cuda.display_name(), "CUDA");
        assert_eq!(DeviceType::WebGpu.display_name(), "WebGPU");
        assert_eq!(DeviceType::DirectMl.display_name(), "DirectML");
        assert_eq!(DeviceType::CoreMl.display_name(), "CoreML");
    }

    #[test]
    fn coreml_never_available() {
        assert!(!DeviceType::CoreMl.is_available());
    }

    #[test]
    fn all_variants_has_five_entries() {
        assert_eq!(DeviceType::all_variants().len(), 5);
    }

    #[test]
    fn capabilities_cpu_is_available() {
        let caps = DeviceCapabilities::probe(DeviceType::Cpu);
        assert!(caps.is_available);
        assert!(caps.supports_int8);
        assert_eq!(caps.device_type, DeviceType::Cpu);
    }
}
