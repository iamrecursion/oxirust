// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Error types for oxiphysics-gpu

use thiserror::Error;

use crate::cell_list::CellListError;
use crate::compute::{CudaInitError, WgpuInitError};
use crate::gpu_nn::NeuralNetError;
use crate::pipeline::PipelineError;
use crate::shader_registry::RegistryError;

/// Main error type for the gpu module.
///
/// This is the error every fallible entry point of the crate reports or can be
/// converted into: the module-specific errors ([`WgpuInitError`],
/// [`CudaInitError`], [`RegistryError`], [`PipelineError`], [`CellListError`],
/// [`NeuralNetError`]) convert into it with `?` through `From`, and the former
/// `GpuError` (backend initialisation, shader dispatch and read-back failures) is
/// the same type, see [`GpuError`].
///
/// The enum is `#[non_exhaustive]`: new failure kinds can be added without a
/// breaking change, so downstream matches need a wildcard arm.
///
/// ```compile_fail,E0004
/// use oxiphysics_gpu::Error;
///
/// fn code(error: &Error) -> u8 {
///     match error {
///         Error::General(_) => 0,
///     }
/// }
/// ```
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum Error {
    /// Generic error with a free-form message.
    #[error("{0}")]
    General(String),

    /// The wgpu backend failed (no adapter, device request, validation,
    /// read-back, ...).
    #[error("wgpu backend error: {0}")]
    Wgpu(#[from] WgpuInitError),

    /// The CUDA backend failed or is not available.
    #[error("CUDA backend error: {0}")]
    Cuda(#[from] CudaInitError),

    /// The shader registry rejected a lookup or an instantiation.
    #[error("shader registry error: {0}")]
    Registry(#[from] RegistryError),

    /// The simulation pipeline rejected a step.
    #[error("pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// A cell-list update was refused.
    #[error("cell list error: {0}")]
    CellList(#[from] CellListError),

    /// A neural-network construction or backpropagation routine failed.
    #[error("neural network error: {0}")]
    NeuralNet(#[from] NeuralNetError),

    /// Two buffers, grids or arrays that must agree in size do not.
    #[error("dimension mismatch for {what}: expected {expected}, got {got}")]
    DimensionMismatch {
        /// What was being checked (e.g. `"solid mask"`).
        what: &'static str,
        /// Required size.
        expected: usize,
        /// Size that was supplied.
        got: usize,
    },

    /// A configuration value is out of range, non-finite or inconsistent with
    /// another value; the message names the offending field.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// GPU backend initialisation failed (no adapter, device or pipeline).
    #[error("GPU backend init failed: {0}")]
    BackendInit(String),

    /// A shader dispatch or pipeline operation failed.
    #[error("shader dispatch error: {0}")]
    ShaderDispatch(String),

    /// Reading a buffer back from the GPU failed.
    #[error("GPU buffer read-back failed: {0}")]
    ReadBack(String),

    /// A handle or index that names no valid GPU resource was used; carries the
    /// offending value.
    ///
    /// Same condition as [`Error::InvalidBufferHandle`]; this spelling is the
    /// former `GpuError::InvalidHandle` and is kept so that its callers keep
    /// compiling.  Prefer [`Error::InvalidBufferHandle`] in new code.
    #[error("invalid GPU buffer handle: {0}")]
    InvalidHandle(usize),

    /// A GPU buffer allocation failed.
    #[error(
        "buffer allocation failed: requested {requested_bytes} bytes (available {available_bytes})"
    )]
    BufferAllocationFailed {
        /// Bytes requested.
        requested_bytes: usize,
        /// Bytes actually available.
        available_bytes: usize,
    },

    /// An invalid buffer handle was used.
    #[error("invalid buffer handle: {0}")]
    InvalidBufferHandle(usize),

    /// A shader compilation error (mock).
    #[error("shader compilation error in '{shader}': {message}")]
    ShaderCompilationError {
        /// Name of the offending shader.
        shader: String,
        /// Compiler message.
        message: String,
    },

    /// A dispatch exceeded the hardware work-group limit.
    #[error("dispatch size {dispatch_size} exceeds hardware limit {limit}")]
    DispatchLimitExceeded {
        /// Requested dispatch size (number of work-groups).
        dispatch_size: usize,
        /// Hardware maximum.
        limit: usize,
    },

    /// Out-of-bounds grid access.
    #[error("grid index ({i}, {j}, {k}) out of bounds for grid ({nx}, {ny}, {nz})")]
    GridIndexOutOfBounds {
        /// Requested x index.
        i: usize,
        /// Requested y index.
        j: usize,
        /// Requested z index.
        k: usize,
        /// Grid x dimension.
        nx: usize,
        /// Grid y dimension.
        ny: usize,
        /// Grid z dimension.
        nz: usize,
    },

    /// A kernel argument count mismatch.
    #[error("kernel '{kernel}' expects {expected} arguments but got {got}")]
    KernelArgCountMismatch {
        /// Kernel name.
        kernel: String,
        /// Expected number of arguments.
        expected: usize,
        /// Provided number of arguments.
        got: usize,
    },

    /// An unsupported backend feature was requested.
    #[error("unsupported feature: {feature}")]
    UnsupportedFeature {
        /// Description of the unsupported feature.
        feature: String,
    },
}

/// A pipeline-stage error: carries the stage name plus the underlying cause.
#[derive(Debug, Error)]
#[error("pipeline stage '{stage}' failed: {source}")]
pub struct PipelineStageError {
    /// Name of the pipeline stage (e.g. `"vertex_fetch"`, `"sph_density"`).
    pub stage: String,
    /// Root cause.
    pub source: Box<Error>,
}

/// Severity level for a GPU error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ErrorSeverity {
    /// Informational — execution can continue.
    Info,
    /// Warning — partial results may be degraded.
    Warning,
    /// Fatal — must abort current dispatch.
    Fatal,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Info => write!(f, "INFO"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// An error annotated with severity and an optional kernel name.
#[derive(Debug)]
pub struct AnnotatedError {
    /// The underlying error.
    pub error: Error,
    /// Severity classification.
    pub severity: ErrorSeverity,
    /// Optional kernel that triggered the error.
    pub kernel: Option<String>,
}

impl std::fmt::Display for AnnotatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref k) = self.kernel {
            write!(f, "[{}] kernel '{}': {}", self.severity, k, self.error)
        } else {
            write!(f, "[{}] {}", self.severity, self.error)
        }
    }
}

impl AnnotatedError {
    /// Wrap an error as fatal with an optional kernel label.
    pub fn fatal(error: Error, kernel: Option<&str>) -> Self {
        Self {
            error,
            severity: ErrorSeverity::Fatal,
            kernel: kernel.map(str::to_string),
        }
    }

    /// Wrap an error as a warning.
    pub fn warning(error: Error, kernel: Option<&str>) -> Self {
        Self {
            error,
            severity: ErrorSeverity::Warning,
            kernel: kernel.map(str::to_string),
        }
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// GPU operation error (used by `LbmGpuSolver`, `BvhGpuTraverser`, etc.).
///
/// This is [`enum@Error`]: the failures it used to enumerate are the variants
/// [`Error::BackendInit`], [`Error::ShaderDispatch`], [`Error::ReadBack`] and
/// [`Error::InvalidHandle`], with unchanged messages, so there is a single error
/// type and `?` needs no conversion between the solver entry points and the rest
/// of the crate.  The alias remains for source compatibility.
pub type GpuError = Error;

impl Error {
    /// Construct a [`Error::General`] from any `Display`-able value.
    pub fn general(msg: impl std::fmt::Display) -> Self {
        Error::General(msg.to_string())
    }

    /// True when this is a recoverable allocation error.
    pub fn is_allocation_error(&self) -> bool {
        matches!(self, Error::BufferAllocationFailed { .. })
    }

    /// True when this is a shader compilation error.
    pub fn is_shader_error(&self) -> bool {
        matches!(self, Error::ShaderCompilationError { .. })
    }

    /// True when this is an out-of-bounds grid error.
    pub fn is_grid_error(&self) -> bool {
        matches!(self, Error::GridIndexOutOfBounds { .. })
    }

    /// True when this is a kernel argument mismatch error.
    pub fn is_arg_mismatch(&self) -> bool {
        matches!(self, Error::KernelArgCountMismatch { .. })
    }

    /// True when this is an unsupported feature error.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Error::UnsupportedFeature { .. })
    }

    /// True when this is an invalid-configuration error.
    pub fn is_invalid_config(&self) -> bool {
        matches!(self, Error::InvalidConfig(_))
    }

    /// True when this is a size-mismatch error.
    pub fn is_dimension_mismatch(&self) -> bool {
        matches!(self, Error::DimensionMismatch { .. })
    }

    /// True when a compute backend (wgpu, CUDA, or a backend that failed to
    /// initialise) reported the error.
    pub fn is_backend_error(&self) -> bool {
        matches!(
            self,
            Error::Wgpu(_) | Error::Cuda(_) | Error::BackendInit(_)
        )
    }

    /// Construct an [`Error::InvalidConfig`] from a message naming the field.
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Error::InvalidConfig(message.into())
    }

    /// Construct an [`Error::DimensionMismatch`].
    pub fn dimension_mismatch(what: &'static str, expected: usize, got: usize) -> Self {
        Error::DimensionMismatch {
            what,
            expected,
            got,
        }
    }

    /// Wrap this error in a [`PipelineStageError`].
    pub fn in_stage(self, stage: impl Into<String>) -> PipelineStageError {
        PipelineStageError {
            stage: stage.into(),
            source: Box::new(self),
        }
    }

    /// Annotate with fatal severity.
    pub fn fatal(self, kernel: Option<&str>) -> AnnotatedError {
        AnnotatedError::fatal(self, kernel)
    }

    /// Annotate with warning severity.
    pub fn warning(self, kernel: Option<&str>) -> AnnotatedError {
        AnnotatedError::warning(self, kernel)
    }

    /// Convert into a `Result::Err`.
    pub fn into_err<T>(self) -> Result<T> {
        Err(self)
    }
}

// ── Convenience constructors ─────────────────────────────────────────────────

/// Build a [`Error::BufferAllocationFailed`] error.
pub fn alloc_err(requested_bytes: usize, available_bytes: usize) -> Error {
    Error::BufferAllocationFailed {
        requested_bytes,
        available_bytes,
    }
}

/// Build a [`Error::KernelArgCountMismatch`] error.
pub fn arg_mismatch_err(kernel: impl Into<String>, expected: usize, got: usize) -> Error {
    Error::KernelArgCountMismatch {
        kernel: kernel.into(),
        expected,
        got,
    }
}

/// Build a [`Error::GridIndexOutOfBounds`] error.
pub fn grid_oob_err(i: usize, j: usize, k: usize, nx: usize, ny: usize, nz: usize) -> Error {
    Error::GridIndexOutOfBounds {
        i,
        j,
        k,
        nx,
        ny,
        nz,
    }
}

/// Build a [`Error::DispatchLimitExceeded`] error.
pub fn dispatch_limit_err(dispatch_size: usize, limit: usize) -> Error {
    Error::DispatchLimitExceeded {
        dispatch_size,
        limit,
    }
}

/// Build a [`Error::ShaderCompilationError`].
pub fn shader_err(shader: impl Into<String>, message: impl Into<String>) -> Error {
    Error::ShaderCompilationError {
        shader: shader.into(),
        message: message.into(),
    }
}

/// Build an [`Error::UnsupportedFeature`].
pub fn unsupported_err(feature: impl Into<String>) -> Error {
    Error::UnsupportedFeature {
        feature: feature.into(),
    }
}

/// Build an [`Error::DimensionMismatch`].
pub fn dimension_mismatch_err(what: &'static str, expected: usize, got: usize) -> Error {
    Error::dimension_mismatch(what, expected, got)
}

/// Build an [`Error::InvalidConfig`].
pub fn invalid_config_err(message: impl Into<String>) -> Error {
    Error::invalid_config(message)
}

/// Check that `got == expected`; return [`Error::DimensionMismatch`] naming
/// `what` otherwise.
pub fn check_len(what: &'static str, expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::dimension_mismatch(what, expected, got))
    }
}

// ── Error collection ─────────────────────────────────────────────────────────

/// Collect multiple errors from a batch dispatch.  Returns `Ok(())` if the
/// vec is empty, or `Err` containing the first error otherwise.
pub fn collect_errors(errors: Vec<Error>) -> Result<()> {
    errors.into_iter().next().map_or(Ok(()), Err)
}

/// Check a boolean condition; return `Err(Error::General(msg))` if false.
pub fn check(condition: bool, msg: impl std::fmt::Display) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::general(msg))
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_general_error_message() {
        let e = Error::general("something went wrong");
        assert_eq!(e.to_string(), "something went wrong");
    }

    #[test]
    fn test_buffer_allocation_failed_message() {
        let e = Error::BufferAllocationFailed {
            requested_bytes: 1024,
            available_bytes: 512,
        };
        let msg = e.to_string();
        assert!(msg.contains("1024"), "should mention requested bytes");
        assert!(msg.contains("512"), "should mention available bytes");
        assert!(e.is_allocation_error());
    }

    #[test]
    fn test_invalid_buffer_handle() {
        let e = Error::InvalidBufferHandle(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_shader_compilation_error() {
        let e = Error::ShaderCompilationError {
            shader: "sph_density".to_string(),
            message: "undefined symbol".to_string(),
        };
        let msg = e.to_string();
        assert!(msg.contains("sph_density"));
        assert!(msg.contains("undefined symbol"));
        assert!(e.is_shader_error());
    }

    #[test]
    fn test_dispatch_limit_exceeded() {
        let e = Error::DispatchLimitExceeded {
            dispatch_size: 100_000,
            limit: 65535,
        };
        let msg = e.to_string();
        assert!(msg.contains("100000"));
        assert!(msg.contains("65535"));
    }

    #[test]
    fn test_grid_index_out_of_bounds() {
        let e = Error::GridIndexOutOfBounds {
            i: 10,
            j: 5,
            k: 3,
            nx: 8,
            ny: 8,
            nz: 8,
        };
        let msg = e.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains('8'.to_string().as_str()));
    }

    #[test]
    fn test_is_not_shader_error() {
        let e = Error::general("not a shader error");
        assert!(!e.is_shader_error());
    }

    #[test]
    fn test_unsupported_feature() {
        let e = Error::UnsupportedFeature {
            feature: "ray_tracing".to_string(),
        };
        assert!(e.to_string().contains("ray_tracing"));
    }

    // ── New error variant / helper tests ─────────────────────────────────

    #[test]
    fn test_is_grid_error() {
        let e = grid_oob_err(1, 2, 3, 4, 5, 6);
        assert!(e.is_grid_error());
        assert!(!e.is_allocation_error());
    }

    #[test]
    fn test_is_arg_mismatch() {
        let e = arg_mismatch_err("test_kernel", 3, 2);
        assert!(e.is_arg_mismatch());
        assert!(!e.is_shader_error());
    }

    #[test]
    fn test_is_unsupported() {
        let e = unsupported_err("ray_tracing");
        assert!(e.is_unsupported());
    }

    #[test]
    fn test_in_stage_wraps_error() {
        let e = Error::general("boom");
        let wrapped = e.in_stage("sph_density");
        assert!(wrapped.to_string().contains("sph_density"));
        assert!(wrapped.to_string().contains("boom"));
    }

    #[test]
    fn test_alloc_err_convenience() {
        let e = alloc_err(512, 256);
        assert!(e.is_allocation_error());
        assert!(e.to_string().contains("512"));
    }

    #[test]
    fn test_dispatch_limit_err_convenience() {
        let e = dispatch_limit_err(99999, 65535);
        assert!(e.to_string().contains("99999"));
    }

    #[test]
    fn test_shader_err_convenience() {
        let e = shader_err("my_shader", "syntax error");
        assert!(e.is_shader_error());
        assert!(e.to_string().contains("syntax error"));
    }

    #[test]
    fn test_into_err() {
        let result: Result<i32> = Error::general("nope").into_err();
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_errors_empty() {
        assert!(collect_errors(vec![]).is_ok());
    }

    #[test]
    fn test_collect_errors_nonempty() {
        let errs = vec![Error::general("first"), Error::general("second")];
        let result = collect_errors(errs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("first"));
    }

    #[test]
    fn test_check_passes() {
        assert!(check(true, "should not fail").is_ok());
    }

    #[test]
    fn test_check_fails() {
        let r = check(false, "condition violated");
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("condition violated"));
    }

    #[test]
    fn test_annotated_error_fatal_display() {
        let e = Error::general("crash");
        let ann = e.fatal(Some("sph_kernel"));
        let s = ann.to_string();
        assert!(s.contains("FATAL"));
        assert!(s.contains("sph_kernel"));
        assert!(s.contains("crash"));
    }

    #[test]
    fn test_annotated_error_warning_no_kernel() {
        let e = Error::general("degraded");
        let ann = e.warning(None);
        let s = ann.to_string();
        assert!(s.contains("WARNING"));
        assert!(s.contains("degraded"));
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Info < ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning < ErrorSeverity::Fatal);
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(ErrorSeverity::Info.to_string(), "INFO");
        assert_eq!(ErrorSeverity::Warning.to_string(), "WARNING");
        assert_eq!(ErrorSeverity::Fatal.to_string(), "FATAL");
    }

    // ── Unified error type: From conversions, folded GpuError ───────────────

    fn init_wgpu() -> std::result::Result<(), WgpuInitError> {
        Err(WgpuInitError::NoAdapter)
    }

    fn init_cuda() -> std::result::Result<(), CudaInitError> {
        Err(CudaInitError::NoDevice)
    }

    fn lookup_shader() -> std::result::Result<(), RegistryError> {
        Err(RegistryError::UnknownShader("sph_density".to_string()))
    }

    fn through_question_mark() -> Result<()> {
        init_wgpu()?;
        Ok(())
    }

    #[test]
    fn test_wgpu_error_converts_with_question_mark() {
        let e = through_question_mark().unwrap_err();
        assert!(matches!(e, Error::Wgpu(WgpuInitError::NoAdapter)), "{e:?}");
        assert_eq!(
            e.to_string(),
            "wgpu backend error: No compatible GPU adapter found"
        );
        assert!(e.is_backend_error());
    }

    #[test]
    fn test_wgpu_error_is_the_source_of_the_wrapper() {
        use std::error::Error as _;
        let e = Error::from(WgpuInitError::PoisonedLock);
        let source = e.source().expect("a wrapped backend error is the source");
        assert_eq!(source.to_string(), "Internal mutex was poisoned");
    }

    #[test]
    fn test_cuda_error_converts() {
        let e: Error = init_cuda().unwrap_err().into();
        assert!(matches!(e, Error::Cuda(CudaInitError::NoDevice)), "{e:?}");
        assert_eq!(
            e.to_string(),
            "CUDA backend error: no CUDA-capable device found"
        );
        assert!(e.is_backend_error());
    }

    #[test]
    fn test_registry_error_converts() {
        let e: Error = lookup_shader().unwrap_err().into();
        assert!(matches!(e, Error::Registry(_)), "{e:?}");
        assert_eq!(
            e.to_string(),
            "shader registry error: unknown shader: sph_density"
        );
        assert!(!e.is_backend_error());
    }

    #[test]
    fn test_module_errors_convert_into_the_crate_error() {
        let pipeline: Error = PipelineError::InvalidTimeStep { dt: -1.0 }.into();
        assert!(matches!(pipeline, Error::Pipeline(_)), "{pipeline:?}");
        assert!(
            pipeline.to_string().starts_with("pipeline error: "),
            "{pipeline}"
        );

        let cell_list: Error = CellListError::LengthMismatch {
            expected: 4,
            got: 3,
        }
        .into();
        assert!(matches!(cell_list, Error::CellList(_)), "{cell_list:?}");
        assert!(
            cell_list.to_string().starts_with("cell list error: "),
            "{cell_list}"
        );

        let network: Error = NeuralNetError::InvalidLayer("zero width".to_string()).into();
        assert!(matches!(network, Error::NeuralNet(_)), "{network:?}");
        assert!(
            network.to_string().starts_with("neural network error: "),
            "{network}"
        );
    }

    #[test]
    fn test_crate_error_still_converts_into_wgpu_error() {
        let wgpu: WgpuInitError = dimension_mismatch_err("solid mask", 12, 7).into();
        assert_eq!(
            wgpu,
            WgpuInitError::Validation(
                "dimension mismatch for solid mask: expected 12, got 7".to_string()
            )
        );
    }

    #[test]
    fn test_wgpu_error_round_trips_through_the_crate_error() {
        for original in [
            WgpuInitError::NoAdapter,
            WgpuInitError::InvalidHandle(5),
            WgpuInitError::Validation("bad binding".to_string()),
        ] {
            let wrapped: Error = original.clone().into();
            let back: WgpuInitError = wrapped.into();
            assert_eq!(back, original);
        }
    }

    #[test]
    fn test_dimension_mismatch_message_and_predicate() {
        let e = Error::dimension_mismatch("solid mask", 12, 7);
        assert_eq!(
            e.to_string(),
            "dimension mismatch for solid mask: expected 12, got 7"
        );
        assert!(e.is_dimension_mismatch());
        assert!(!e.is_invalid_config());
    }

    #[test]
    fn test_check_len() {
        assert!(check_len("positions", 4, 4).is_ok());
        let e = check_len("positions", 4, 3).unwrap_err();
        assert!(matches!(
            e,
            Error::DimensionMismatch {
                what: "positions",
                expected: 4,
                got: 3
            }
        ));
    }

    #[test]
    fn test_invalid_config_message_and_predicate() {
        let e = invalid_config_err("omega must lie in (0, 2)");
        assert_eq!(
            e.to_string(),
            "invalid configuration: omega must lie in (0, 2)"
        );
        assert!(e.is_invalid_config());
        assert!(!e.is_dimension_mismatch());
    }

    #[test]
    fn test_gpu_error_is_the_crate_error() {
        let backend: GpuError = GpuError::BackendInit("no adapter".to_string());
        assert_eq!(backend.to_string(), "GPU backend init failed: no adapter");
        assert!(matches!(backend, Error::BackendInit(_)));
        assert!(backend.is_backend_error());

        let dispatch = GpuError::ShaderDispatch("bad bind group".to_string());
        assert_eq!(
            dispatch.to_string(),
            "shader dispatch error: bad bind group"
        );

        let read_back = GpuError::ReadBack("map failed".to_string());
        assert_eq!(
            read_back.to_string(),
            "GPU buffer read-back failed: map failed"
        );

        let handle = GpuError::InvalidHandle(7);
        assert_eq!(handle.to_string(), "invalid GPU buffer handle: 7");

        // The folded type needs no conversion to travel through `crate::Result`.
        let result: Result<()> = Err(GpuError::ReadBack("x".to_string()));
        assert!(matches!(result, Err(Error::ReadBack(_))));
    }

    #[test]
    fn test_error_is_clone_send_sync_and_boxable() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<Error>();
        let original = Error::general("kept");
        let copy = original.clone();
        assert_eq!(copy.to_string(), original.to_string());
        let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(copy);
        assert_eq!(boxed.to_string(), "kept");
    }

    #[test]
    fn test_pipeline_stage_error_wraps_backend_error() {
        let wrapped = Error::from(WgpuInitError::NoAdapter).in_stage("sph_density");
        let text = wrapped.to_string();
        assert!(text.contains("sph_density"), "{text}");
        assert!(text.contains("No compatible GPU adapter found"), "{text}");
    }
}
