//! Re-export surface tests for the flagship `tensorlogic` crate.
//!
//! Each sub-crate is re-exported at `tensorlogic::<name>` (either always or
//! feature-gated). These tests pick one real type from each sub-crate and
//! reference it through the flagship path. If a sub-crate is ever dropped
//! from the re-export list in `src/lib.rs`, the corresponding test stops
//! compiling — an immediate, loud failure.

/// Mandatory core crates are always re-exported.
#[test]
fn mandatory_core_modules_resolve() {
    // ir
    let _ir_graph: Option<tensorlogic::ir::EinsumGraph> = None;
    let _ir_term: Option<tensorlogic::ir::Term> = None;
    let _ir_expr: Option<tensorlogic::ir::TLExpr> = None;

    // infer — reference the trait as a bound rather than a trait object,
    // since `TlExecutor` uses associated types and is not necessarily
    // dyn-compatible in every Rust edition.
    fn _needs_exec<E: tensorlogic::infer::TlExecutor>() {}
    fn _needs_autodiff<A: tensorlogic::infer::TlAutodiff>() {}

    // adapters
    let _adapter_err: Option<tensorlogic::adapters::AdapterError> = None;

    // compiler
    let _compile_cfg: Option<tensorlogic::compiler::CompilationConfig> = None;
    let _compile_fn = tensorlogic::compiler::compile_to_einsum;
}

/// `scirs_backend` is mandatory; verify its flagship path resolves to a real type.
#[test]
fn scirs_backend_module_resolves() {
    let _exec: Option<tensorlogic::scirs_backend::Scirs2Exec> = None;
    let _err: Option<tensorlogic::scirs_backend::TlBackendError> = None;
}

/// `train` is optional and gated behind the `train` feature flag.
#[cfg(feature = "train")]
#[test]
fn train_module_resolves() {
    let _err: Option<tensorlogic::train::TrainError> = None;
}

/// `oxirs-bridge` is optional and gated behind the `oxirs` feature flag.
#[cfg(feature = "oxirs")]
#[test]
fn oxirs_bridge_module_resolves() {
    let _err: Option<tensorlogic::oxirs_bridge::BridgeError> = None;
}

/// `quantrs-hooks` is optional and gated behind the `quantrs` feature flag.
#[cfg(feature = "quantrs")]
#[test]
fn quantrs_hooks_module_resolves() {
    let _err: Option<tensorlogic::quantrs_hooks::PgmError> = None;
}

/// `sklears-kernels` is optional and gated behind the `sklears` feature flag.
#[cfg(feature = "sklears")]
#[test]
fn sklears_kernels_module_resolves() {
    let _err: Option<tensorlogic::sklears_kernels::KernelError> = None;
}

/// `trustformers` is optional and gated behind the `trustformers` feature flag.
#[cfg(feature = "trustformers")]
#[test]
fn trustformers_module_resolves() {
    let _err: Option<tensorlogic::trustformers::TrustformerError> = None;
}
