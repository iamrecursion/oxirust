// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Naga front-end validation for the crate's WGSL kernels.
//!
//! Unlike the substring checks in `shaders::functions::validate_wgsl_structure`,
//! these tests run the real Pure-Rust naga WGSL parser and validator over each
//! shader source. A shader that fails to parse or type-check (e.g. the legacy
//! `pub(super)` leakage that is invalid WGSL, or a malformed expression) makes
//! the corresponding test fail. No GPU adapter is required — this is a pure
//! compile-time validation of the WGSL itself.

/// Parse + validate one WGSL source through naga and return the module; panics
/// with the shader name and the naga diagnostic on failure.
fn parse_validated(name: &str, src: &str) -> naga::Module {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("WGSL parse failed for `{name}`:\n{}", e.emit_to_string(src)),
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    if let Err(e) = validator.validate(&module) {
        panic!("WGSL validation failed for `{name}`: {e:?}");
    }
    module
}

/// Parse + validate one WGSL source through naga; panics with the shader name
/// and the naga diagnostic on failure.
fn validate_wgsl(name: &str, src: &str) {
    parse_validated(name, src);
}

// ── SPH pipeline kernels (the dispatched deliverable) ─────────────────────────

#[test]
fn sph_cell_list_wgsl_compiles() {
    validate_wgsl(
        "SPH_CELL_LIST_WGSL",
        oxiphysics_gpu::kernels_wgsl::SPH_CELL_LIST_WGSL,
    );
}

#[test]
fn sph_cell_histogram_wgsl_compiles() {
    validate_wgsl(
        "SPH_CELL_HISTOGRAM_WGSL",
        oxiphysics_gpu::kernels_wgsl::SPH_CELL_HISTOGRAM_WGSL,
    );
}

#[test]
fn sph_density_grid_wgsl_compiles() {
    validate_wgsl(
        "SPH_DENSITY_GRID_WGSL",
        oxiphysics_gpu::kernels_wgsl::SPH_DENSITY_GRID_WGSL,
    );
}

#[test]
fn sph_force_grid_wgsl_compiles() {
    validate_wgsl(
        "SPH_FORCE_GRID_WGSL",
        oxiphysics_gpu::kernels_wgsl::SPH_FORCE_GRID_WGSL,
    );
}

#[test]
fn sph_integrate_wgsl_compiles() {
    validate_wgsl(
        "SPH_INTEGRATE_WGSL",
        oxiphysics_gpu::kernels_wgsl::SPH_INTEGRATE_WGSL,
    );
}

#[test]
fn sph_boundary_wgsl_compiles() {
    validate_wgsl(
        "SPH_BOUNDARY_WGSL",
        oxiphysics_gpu::kernels_wgsl::SPH_BOUNDARY_WGSL,
    );
}

// ── Reduction-primitive kernels (already dispatched elsewhere) ────────────────

#[test]
fn reduction_kernels_compile() {
    use oxiphysics_gpu::kernels_wgsl::*;
    validate_wgsl("SCAN_WGSL", SCAN_WGSL);
    validate_wgsl("SCAN_ADD_WGSL", SCAN_ADD_WGSL);
    validate_wgsl("REDUCE_WGSL", REDUCE_WGSL);
    validate_wgsl("COMPACT_WGSL", COMPACT_WGSL);
    validate_wgsl("HISTOGRAM_WGSL", HISTOGRAM_WGSL);
    validate_wgsl("RADIX_HISTOGRAM_WGSL", RADIX_HISTOGRAM_WGSL);
    validate_wgsl("RADIX_SCATTER_WGSL", RADIX_SCATTER_WGSL);
    validate_wgsl("RADIX_SCATTER_PAIRS_WGSL", RADIX_SCATTER_PAIRS_WGSL);
}

// ── shader_registry templates (the 4 promoted + the rest) ─────────────────────

/// All ten built-in `shaders::functions` WGSL templates must now be valid WGSL.
///
/// This is the regression guard for the `pub(super)` leakage that previously
/// made `CELL_LIST_WGSL`, `SPH_FORCE_WGSL`, `INTEGRATE_WGSL`, and
/// `BOUNDARY_ENFORCE_WGSL` (plus six siblings) parse-fail under any real WGSL
/// compiler while passing the old substring-only validator.
#[test]
fn shader_registry_templates_compile() {
    use oxiphysics_gpu::shaders::{
        BOUNDARY_ENFORCE_WGSL, BROADPHASE_SORT_SHADER, CELL_LIST_WGSL, INTEGRATE_WGSL,
        LBM_BGK_D2Q9_WGSL, LBM_STREAMING_SHADER, RIGID_INTEGRATE_SHADER, SDF_COMPUTE_WGSL,
        SPH_DENSITY_WGSL, SPH_FORCE_WGSL,
    };
    validate_wgsl("SPH_DENSITY_WGSL", SPH_DENSITY_WGSL);
    validate_wgsl("INTEGRATE_WGSL", INTEGRATE_WGSL);
    validate_wgsl("LBM_BGK_D2Q9_WGSL", LBM_BGK_D2Q9_WGSL);
    validate_wgsl("CELL_LIST_WGSL", CELL_LIST_WGSL);
    validate_wgsl("SDF_COMPUTE_WGSL", SDF_COMPUTE_WGSL);
    validate_wgsl("LBM_STREAMING_SHADER", LBM_STREAMING_SHADER);
    validate_wgsl("RIGID_INTEGRATE_SHADER", RIGID_INTEGRATE_SHADER);
    validate_wgsl("BROADPHASE_SORT_SHADER", BROADPHASE_SORT_SHADER);
    validate_wgsl("SPH_FORCE_WGSL", SPH_FORCE_WGSL);
    validate_wgsl("BOUNDARY_ENFORCE_WGSL", BOUNDARY_ENFORCE_WGSL);
}

// ── Every shipped kernel: validity, entry points, folded 1-D dispatches ───────

/// Every compute shader embedded in a Rust constant: its label, its source and
/// the compute entry points it must expose.  The shader files of
/// `src/kernels_wgsl` and `src/shaders` are covered by a directory scan below.
fn embedded_kernels() -> Vec<(&'static str, &'static str, &'static [&'static str])> {
    use oxiphysics_gpu::compute::wgpu_backend as backend;
    use oxiphysics_gpu::{kernels_wgsl as kernels, shaders};
    vec![
        ("SCAN_WGSL", kernels::SCAN_WGSL, &["scan_block"]),
        (
            "SCAN_ADD_WGSL",
            kernels::SCAN_ADD_WGSL,
            &["add_block_offsets"],
        ),
        (
            "REDUCE_WGSL",
            kernels::REDUCE_WGSL,
            &["reduce_sum_block", "reduce_max_block"],
        ),
        ("COMPACT_WGSL", kernels::COMPACT_WGSL, &["scatter_kept"]),
        (
            "HISTOGRAM_WGSL",
            kernels::HISTOGRAM_WGSL,
            &["histogram_main"],
        ),
        (
            "RADIX_HISTOGRAM_WGSL",
            kernels::RADIX_HISTOGRAM_WGSL,
            &["radix_histogram"],
        ),
        (
            "RADIX_SCATTER_WGSL",
            kernels::RADIX_SCATTER_WGSL,
            &["radix_scatter"],
        ),
        (
            "RADIX_SCATTER_PAIRS_WGSL",
            kernels::RADIX_SCATTER_PAIRS_WGSL,
            &["radix_scatter_pairs"],
        ),
        (
            "SPH_CELL_LIST_WGSL",
            kernels::SPH_CELL_LIST_WGSL,
            &["sph_cell_list"],
        ),
        (
            "SPH_CELL_HISTOGRAM_WGSL",
            kernels::SPH_CELL_HISTOGRAM_WGSL,
            &["sph_cell_histogram"],
        ),
        (
            "SPH_DENSITY_GRID_WGSL",
            kernels::SPH_DENSITY_GRID_WGSL,
            &["sph_density"],
        ),
        (
            "SPH_FORCE_GRID_WGSL",
            kernels::SPH_FORCE_GRID_WGSL,
            &["sph_force"],
        ),
        (
            "SPH_INTEGRATE_WGSL",
            kernels::SPH_INTEGRATE_WGSL,
            &["sph_integrate"],
        ),
        (
            "SPH_BOUNDARY_WGSL",
            kernels::SPH_BOUNDARY_WGSL,
            &["sph_boundary"],
        ),
        (
            "WGSL_PARALLEL_SCAN",
            backend::WGSL_PARALLEL_SCAN,
            &["exclusive_scan"],
        ),
        (
            "WGSL_SPH_DENSITY",
            backend::WGSL_SPH_DENSITY,
            &["sph_density"],
        ),
        (
            "WGSL_BVH_TRAVERSAL",
            backend::WGSL_BVH_TRAVERSAL,
            &["bvh_traverse"],
        ),
        ("SPH_DENSITY_WGSL", shaders::SPH_DENSITY_WGSL, &["main"]),
        ("INTEGRATE_WGSL", shaders::INTEGRATE_WGSL, &["main"]),
        ("LBM_BGK_D2Q9_WGSL", shaders::LBM_BGK_D2Q9_WGSL, &["main"]),
        ("CELL_LIST_WGSL", shaders::CELL_LIST_WGSL, &["main"]),
        ("SDF_COMPUTE_WGSL", shaders::SDF_COMPUTE_WGSL, &["main"]),
        (
            "LBM_STREAMING_SHADER",
            shaders::LBM_STREAMING_SHADER,
            &["main"],
        ),
        (
            "RIGID_INTEGRATE_SHADER",
            shaders::RIGID_INTEGRATE_SHADER,
            &["main"],
        ),
        (
            "BROADPHASE_SORT_SHADER",
            shaders::BROADPHASE_SORT_SHADER,
            &["main"],
        ),
        ("SPH_FORCE_WGSL", shaders::SPH_FORCE_WGSL, &["main"]),
        (
            "BOUNDARY_ENFORCE_WGSL",
            shaders::BOUNDARY_ENFORCE_WGSL,
            &["main"],
        ),
    ]
}

/// True when `entry` reads `num_workgroups.x`, which a 1-D kernel needs to
/// linearise a dispatch that was folded into the y dimension.
fn reads_num_workgroups_x(entry: &naga::EntryPoint) -> bool {
    let function = &entry.function;
    function
        .arguments
        .iter()
        .enumerate()
        .any(|(position, argument)| {
            matches!(
                argument.binding,
                Some(naga::Binding::BuiltIn(naga::BuiltIn::NumWorkGroups))
            ) && function
                .expressions
                .iter()
                .any(|(_, expression)| match expression {
                    naga::Expression::AccessIndex { base, index: 0 } => matches!(
                        function.expressions[*base],
                        naga::Expression::FunctionArgument(arg) if arg as usize == position
                    ),
                    _ => false,
                })
        })
}

/// Assert that every 1-D compute entry point of `module` (a workgroup size of
/// the form `(n, 1, 1)`) linearises its indices with `num_workgroups`, and
/// return how many entry points were checked.
///
/// A 1-D kernel dispatched over more than 65535 workgroups is folded into the
/// y dimension (`dispatch_count_for`); a kernel that indexes with `gid.x` alone
/// would then process only the first row of the grid and silently skip the rest.
fn assert_one_dimensional_kernels_fold(name: &str, module: &naga::Module) -> usize {
    let mut checked = 0;
    for entry in &module.entry_points {
        let one_dimensional = entry.workgroup_size[1] == 1 && entry.workgroup_size[2] == 1;
        if entry.stage != naga::ShaderStage::Compute || !one_dimensional {
            continue;
        }
        assert!(
            reads_num_workgroups_x(entry),
            "`{name}` entry point `{}` is a 1-D kernel that never reads `num_workgroups.x`: \
             a dispatch folded into y would be linearised wrongly",
            entry.name
        );
        checked += 1;
    }
    checked
}

/// Names of the compute entry points of `module`, sorted.
fn compute_entry_points(module: &naga::Module) -> Vec<String> {
    let mut names: Vec<String> = module
        .entry_points
        .iter()
        .filter(|entry| entry.stage == naga::ShaderStage::Compute)
        .map(|entry| entry.name.clone())
        .collect();
    names.sort();
    names
}

/// Sources and names of the `.wgsl` files shipped in `src/kernels_wgsl` and
/// `src/shaders`.
fn wgsl_files() -> Vec<(String, String)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    for directory in ["kernels_wgsl", "shaders"] {
        let entries = std::fs::read_dir(root.join(directory)).expect("kernel directory");
        for entry in entries {
            let path = entry.expect("directory entry").path();
            if path.extension().is_some_and(|ext| ext == "wgsl") {
                let name = format!(
                    "{directory}/{}",
                    path.file_name().expect("file name").to_string_lossy()
                );
                files.push((name, std::fs::read_to_string(&path).expect("kernel source")));
            }
        }
    }
    files.sort();
    files
}

/// Every WGSL constant of the crate parses, validates and exposes exactly the
/// compute entry points the host code dispatches.  The `wgpu_backend` demo
/// kernels used to fail here: `pass` and `shared` are reserved WGSL words.
#[test]
fn embedded_kernels_compile_and_expose_their_entry_points() {
    for (name, src, expected) in embedded_kernels() {
        let module = parse_validated(name, src);
        let mut expected: Vec<String> = expected.iter().map(|entry| entry.to_string()).collect();
        expected.sort();
        assert_eq!(compute_entry_points(&module), expected, "`{name}`");
    }
}

/// Every shipped 1-D kernel of the Rust constants reads `num_workgroups.x`.
#[test]
fn embedded_one_dimensional_kernels_linearise_with_num_workgroups() {
    let mut checked = 0;
    for (name, src, _) in embedded_kernels() {
        checked += assert_one_dimensional_kernels_fold(name, &parse_validated(name, src));
    }
    assert!(checked >= 28, "only {checked} 1-D kernels were checked");
}

/// Every `.wgsl` file (including the runtime LBM and BVH shaders) validates, and
/// its 1-D kernels read `num_workgroups.x`.  A shader file added later is
/// picked up by the directory scan.
#[test]
fn wgsl_files_compile_and_their_one_dimensional_kernels_linearise_with_num_workgroups() {
    let files = wgsl_files();
    assert!(files.len() >= 16, "found only {} shader files", files.len());
    let mut checked = 0;
    for (name, src) in &files {
        checked += assert_one_dimensional_kernels_fold(name, &parse_validated(name, src));
    }
    assert!(checked >= 16, "only {checked} 1-D kernels were checked");
}

/// The 3-D LBM kernel is exempt from folding: its workgroups are indexed per
/// axis, not linearised.
#[test]
fn lbm_kernel_is_three_dimensional_and_exempt_from_folding() {
    let files = wgsl_files();
    let (name, src) = files
        .iter()
        .find(|(name, _)| name.ends_with("lbm_bgk_d3q19.wgsl"))
        .expect("the D3Q19 kernel is shipped");
    let module = parse_validated(name, src);
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "main")
        .expect("main entry point");
    assert_eq!(entry.workgroup_size, [4, 4, 4]);
    assert_eq!(assert_one_dimensional_kernels_fold(name, &module), 0);
}

/// The guard itself must reject a 1-D kernel that indexes with `gid.x` alone.
#[test]
fn guard_rejects_an_unfolded_one_dimensional_kernel() {
    let unfolded = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
        @compute @workgroup_size(64)\n\
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) { data[gid.x] = 1u; }\n";
    let unused_builtin = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
        @compute @workgroup_size(64)\n\
        fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
                @builtin(num_workgroups) nwg: vec3<u32>) { data[gid.x] = 1u; }\n";
    let folded = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
        @compute @workgroup_size(64)\n\
        fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
                @builtin(num_workgroups) nwg: vec3<u32>) {\n\
            data[gid.x + gid.y * nwg.x * 64u] = 1u;\n\
        }\n";
    for (label, src) in [("unfolded", unfolded), ("unused builtin", unused_builtin)] {
        let module = parse_validated(label, src);
        let entry = &module.entry_points[0];
        assert!(!reads_num_workgroups_x(entry), "{label} must be rejected");
    }
    let module = parse_validated("folded", folded);
    assert!(reads_num_workgroups_x(&module.entry_points[0]));
    assert_eq!(assert_one_dimensional_kernels_fold("folded", &module), 1);
}
