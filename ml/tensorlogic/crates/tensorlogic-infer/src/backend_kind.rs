//! Backend selection and capability negotiation for TensorLogic inference engines.
//!
//! `BackendKind` is the canonical discriminant for which compute backend is active.
//! - `Scirs` — pure-Rust CPU backend, always available.
//! - `OxiCuda` — NVIDIA GPU backend via `tensorlogic-oxicuda-backend`; requires the
//!   `gpu` feature on that crate.
//! - All remaining variants are Round-5 stubs, planned for Round 6.

use std::env;
use std::str::FromStr;

use thiserror::Error;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced when resolving or validating a [`BackendKind`].
#[derive(Debug, Error)]
pub enum BackendKindError {
    /// The named backend exists in the enum but has not been implemented yet.
    #[error("backend '{name}' is not yet implemented (planned for Round 6)")]
    Unimplemented { name: &'static str },

    /// The string passed to [`BackendKind::from_str`] did not match any known backend.
    #[error("unknown backend name: {0}")]
    UnknownName(String),
}

// ─── Enum ────────────────────────────────────────────────────────────────────

/// Discriminant for the active compute backend.
///
/// Only [`BackendKind::Scirs`] and [`BackendKind::OxiCuda`] are fully supported in
/// Round 5. All other variants are doc-hidden stubs that will be enabled in Round 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// CPU backend powered by SciRS2-Core. Always available.
    Scirs,

    /// NVIDIA GPU backend. Requires `feature = "gpu"` on `tensorlogic-oxicuda-backend`.
    OxiCuda,

    #[doc(hidden)]
    /// Apple Metal GPU backend. Stub — not yet implemented.
    Metal,

    #[doc(hidden)]
    /// Vulkan compute backend. Stub — not yet implemented.
    Vulkan,

    #[doc(hidden)]
    /// AMD ROCm backend. Stub — not yet implemented.
    Rocm,

    #[doc(hidden)]
    /// WebGPU backend. Stub — not yet implemented.
    Webgpu,

    #[doc(hidden)]
    /// Intel Level Zero backend. Stub — not yet implemented.
    Levelzero,
}

// ─── Core methods ─────────────────────────────────────────────────────────────

impl BackendKind {
    /// Returns `OxiCuda` when the environment variable `TENSORLOGIC_BACKEND` is set
    /// to `"oxicuda"`, otherwise returns `Scirs`.
    ///
    /// This is the recommended entry-point for runtime backend selection.
    pub fn default_backend() -> Self {
        match env::var("TENSORLOGIC_BACKEND")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "oxicuda" => Self::OxiCuda,
            _ => Self::Scirs,
        }
    }

    /// Reads `TENSORLOGIC_BACKEND` and maps it to a `BackendKind`.
    ///
    /// Recognised values (case-insensitive):
    /// - `"oxicuda"`, `"gpu"`, `"cuda"` → [`BackendKind::OxiCuda`]
    /// - anything else (including unset) → [`BackendKind::Scirs`]
    ///
    /// Non-NVIDIA variants (`metal`, `vulkan`, `rocm`, `webgpu`, `levelzero`) are
    /// parsed correctly by [`BackendKind::from_str`] but always fail
    /// [`BackendKind::validate`] with [`BackendKindError::Unimplemented`].
    pub fn from_env() -> Self {
        env::var("TENSORLOGIC_BACKEND")
            .ok()
            .and_then(|s| s.parse::<Self>().ok())
            .unwrap_or(Self::Scirs)
    }

    /// Returns `true` when this backend uses a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            Self::OxiCuda
                | Self::Metal
                | Self::Vulkan
                | Self::Rocm
                | Self::Webgpu
                | Self::Levelzero
        )
    }

    /// Returns `true` when this backend supports automatic differentiation.
    ///
    /// In Round 5 only `Scirs` and `OxiCuda` implement autodiff; all other
    /// variants are stubs.
    pub fn supports_autodiff(&self) -> bool {
        matches!(self, Self::Scirs | Self::OxiCuda)
    }

    /// Returns a human-readable, stable identifier for the backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scirs => "scirs",
            Self::OxiCuda => "oxicuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Rocm => "rocm",
            Self::Webgpu => "webgpu",
            Self::Levelzero => "levelzero",
        }
    }

    /// Returns all currently enumerated backends.
    ///
    /// `Scirs` is always fully supported. The remaining backends are listed for
    /// completeness but are not yet implemented (Round 6).
    pub fn available_backends() -> Vec<Self> {
        vec![
            Self::Scirs,     // fully supported
            Self::OxiCuda,   // fully supported when GPU feature enabled
            Self::Metal,     // stub
            Self::Vulkan,    // stub
            Self::Rocm,      // stub
            Self::Webgpu,    // stub
            Self::Levelzero, // stub
        ]
    }

    /// Validates that this backend is fully implemented and can be activated.
    ///
    /// Returns `Ok(())` for `Scirs` and `OxiCuda`; returns
    /// [`BackendKindError::Unimplemented`] for all Round-5 stubs.
    pub fn validate(&self) -> Result<(), BackendKindError> {
        match self {
            Self::Scirs | Self::OxiCuda => Ok(()),
            other => Err(BackendKindError::Unimplemented {
                name: other.as_str(),
            }),
        }
    }
}

// ─── FromStr impl ────────────────────────────────────────────────────────────

impl FromStr for BackendKind {
    type Err = BackendKindError;

    /// Parses a backend name string into a `BackendKind`.
    ///
    /// Accepted aliases (all case-insensitive):
    ///
    /// | Input | Result |
    /// |---|---|
    /// | `"scirs"`, `"cpu"` | `Scirs` |
    /// | `"oxicuda"`, `"gpu"`, `"cuda"` | `OxiCuda` |
    /// | `"metal"` | `Metal` |
    /// | `"vulkan"` | `Vulkan` |
    /// | `"rocm"`, `"hip"` | `Rocm` |
    /// | `"webgpu"` | `Webgpu` |
    /// | `"levelzero"`, `"level_zero"` | `Levelzero` |
    /// | anything else | `Err(BackendKindError::UnknownName)` |
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "scirs" | "cpu" => Ok(Self::Scirs),
            "oxicuda" | "gpu" | "cuda" => Ok(Self::OxiCuda),
            "metal" => Ok(Self::Metal),
            "vulkan" => Ok(Self::Vulkan),
            "rocm" | "hip" => Ok(Self::Rocm),
            "webgpu" => Ok(Self::Webgpu),
            "levelzero" | "level_zero" => Ok(Self::Levelzero),
            other => Err(BackendKindError::UnknownName(other.to_string())),
        }
    }
}
