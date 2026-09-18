//! Smoke tests for newly-wired orphan modules in oximedia-pipeline.

#[test]
fn backpressure_module_accessible() {
    // The backpressure module should be reachable via the crate root.
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::backpressure));
}

#[test]
fn composition_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::composition));
}

#[test]
fn diff_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::diff));
}

#[test]
fn dot_export_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::dot_export));
}

#[test]
fn dynamic_reconfig_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::dynamic_reconfig));
}

#[test]
fn format_negotiation_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::format_negotiation));
}

#[test]
fn hardware_validator_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::hardware_validator));
}

#[test]
fn memory_pool_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::memory_pool));
}

#[test]
fn optimizer_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::optimizer));
}

#[test]
fn pipeline_debugger_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::pipeline_debugger));
}

#[test]
fn pipeline_profiler_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::pipeline_profiler));
}

#[test]
fn preset_library_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::preset_library));
}

#[test]
fn replay_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::replay));
}

#[test]
fn simd_scheduler_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::simd_scheduler));
}

#[test]
fn topo_tests_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::topo_tests));
}

#[test]
fn zero_copy_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_pipeline::zero_copy));
}
