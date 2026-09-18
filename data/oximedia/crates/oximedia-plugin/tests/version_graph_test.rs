//! Integration tests for the version resolver with complex dependency graphs
//! (10+ interdependent plugins).
//!
//! These tests model realistic plugin ecosystems where plugins have
//! transitive dependencies, conflicting constraints, and rich version ranges.

use oximedia_plugin::version_resolver::{
    DependencyResolver, PluginDependency, SemVer, VersionConstraint,
};
// ── Helpers ───────────────────────────────────────────────────────────────────

fn v(major: u32, minor: u32, patch: u32) -> SemVer {
    SemVer::new(major, minor, patch)
}

fn dep(id: &str, c: VersionConstraint) -> PluginDependency {
    PluginDependency::new(id, c)
}

fn resolver_with(entries: &[(&str, &[SemVer])]) -> DependencyResolver {
    let mut r = DependencyResolver::new();
    for (id, versions) in entries {
        r.register(*id, versions.to_vec());
    }
    r
}

// ── 10-plugin linear chain ─────────────────────────────────────────────────

/// p0 → p1 → p2 → … → p9 (linear dependency chain).
/// Each plugin is available at exactly one version.
#[test]
fn test_linear_chain_10_plugins() {
    let v100 = vec![v(1, 0, 0)];
    let plugins: Vec<(&str, &[SemVer])> = vec![
        ("p0", v100.as_slice()),
        ("p1", v100.as_slice()),
        ("p2", v100.as_slice()),
        ("p3", v100.as_slice()),
        ("p4", v100.as_slice()),
        ("p5", v100.as_slice()),
        ("p6", v100.as_slice()),
        ("p7", v100.as_slice()),
        ("p8", v100.as_slice()),
        ("p9", v100.as_slice()),
    ];
    let r = resolver_with(&plugins);

    // Root requires p0-p9, each at >=1.0.0.
    let deps: Vec<PluginDependency> = (0..10)
        .map(|i| dep(&format!("p{i}"), VersionConstraint::AtLeast(v(1, 0, 0))))
        .collect();

    let result = r.resolve(&deps).expect("resolve linear chain");
    assert_eq!(result.len(), 10);
    for i in 0..10u32 {
        assert_eq!(result[&format!("p{i}")], v(1, 0, 0));
    }
}

// ── 10 plugins, picks highest satisfying ──────────────────────────────────

#[test]
fn test_highest_version_selected_across_10_plugins() {
    let versions_a: Vec<SemVer> = (1..=5).map(|n| v(1, n, 0)).collect();
    let versions_b: Vec<SemVer> = (1..=3).map(|n| v(2, n, 0)).collect();

    let mut r = DependencyResolver::new();
    for i in 0..5u32 {
        // plugins 0-4 have versions 1.1.0–1.5.0
        r.register(format!("plugin-a{i}"), versions_a.clone());
        // plugins 5-9 have versions 2.1.0–2.3.0
        r.register(format!("plugin-b{i}"), versions_b.clone());
    }

    let deps: Vec<PluginDependency> = (0..5u32)
        .map(|i| {
            dep(
                &format!("plugin-a{i}"),
                VersionConstraint::Compatible(v(1, 0, 0)),
            )
        })
        .chain((0..5u32).map(|i| {
            dep(
                &format!("plugin-b{i}"),
                VersionConstraint::Compatible(v(2, 0, 0)),
            )
        }))
        .collect();

    let result = r.resolve(&deps).expect("resolve 10 plugins");

    for i in 0..5u32 {
        assert_eq!(result[&format!("plugin-a{i}")], v(1, 5, 0));
        assert_eq!(result[&format!("plugin-b{i}")], v(2, 3, 0));
    }
}

// ── Diamond dependency ─────────────────────────────────────────────────────

/// A requires B and C; both B and C require D.
/// D has two available versions; constraints from B and C must be intersected.
#[test]
fn test_diamond_dependency_constraint_merge() {
    let mut r = DependencyResolver::new();
    r.register("D", vec![v(1, 0, 0), v(1, 5, 0), v(2, 0, 0)]);
    r.register("B", vec![v(1, 0, 0)]);
    r.register("C", vec![v(1, 0, 0)]);

    // B needs D >=1.0.0 (satisfied by all)
    // C needs D <=1.5.0 (rules out 2.0.0)
    // Combined: 1.0.0 <= D <= 1.5.0 → picks 1.5.0
    let deps = vec![
        dep("B", VersionConstraint::AtLeast(v(1, 0, 0))),
        dep("C", VersionConstraint::AtLeast(v(1, 0, 0))),
        dep("D", VersionConstraint::AtLeast(v(1, 0, 0))),
        dep("D", VersionConstraint::AtMost(v(1, 5, 0))),
    ];

    let result = r.resolve(&deps).expect("diamond resolve");
    assert_eq!(result["D"], v(1, 5, 0));
}

// ── Conflict in 10-plugin graph ────────────────────────────────────────────

#[test]
fn test_conflict_in_large_graph() {
    let mut r = DependencyResolver::new();
    for i in 0..9u32 {
        r.register(format!("q{i}"), vec![v(1, 0, 0)]);
    }
    // q9 only has version 1.0.0 but we demand >=2.0.0.
    r.register("q9", vec![v(1, 0, 0)]);

    let mut deps: Vec<PluginDependency> = (0..9u32)
        .map(|i| dep(&format!("q{i}"), VersionConstraint::AtLeast(v(1, 0, 0))))
        .collect();
    // Conflicting requirement for q9.
    deps.push(dep("q9", VersionConstraint::AtLeast(v(2, 0, 0))));

    let err = r.resolve(&deps).expect_err("should conflict");
    assert!(err.to_string().contains("q9"));
}

// ── Not-found in 10-plugin graph ──────────────────────────────────────────

#[test]
fn test_not_found_in_large_graph() {
    let mut r = DependencyResolver::new();
    for i in 0..9u32 {
        r.register(format!("r{i}"), vec![v(1, 0, 0)]);
    }
    // r9 not registered.

    let mut deps: Vec<PluginDependency> = (0..9u32)
        .map(|i| dep(&format!("r{i}"), VersionConstraint::AtLeast(v(1, 0, 0))))
        .collect();
    deps.push(dep("r9", VersionConstraint::AtLeast(v(1, 0, 0))));

    let err = r.resolve(&deps).expect_err("should be not found");
    assert!(err.to_string().contains("r9"));
}

// ── Same dep with two constraints → range intersection ────────────────────

#[test]
fn test_two_constraints_on_same_plugin_intersect() {
    let mut r = DependencyResolver::new();
    r.register("core", vec![v(1, 0, 0), v(1, 5, 0), v(1, 9, 0), v(2, 0, 0)]);

    let deps = vec![
        dep("core", VersionConstraint::AtLeast(v(1, 5, 0))),
        dep("core", VersionConstraint::AtMost(v(1, 9, 0))),
    ];

    let result = r.resolve(&deps).expect("intersect");
    assert_eq!(result["core"], v(1, 9, 0)); // highest in [1.5, 1.9]
}

// ── All range: wildcard-like constraints ──────────────────────────────────

#[test]
fn test_at_least_zero_selects_highest() {
    let mut r = DependencyResolver::new();
    let versions: Vec<SemVer> = (1..=10).map(|n| v(1, n, 0)).collect();
    r.register("lib", versions);

    let deps = vec![dep("lib", VersionConstraint::AtLeast(v(0, 0, 0)))];
    let result = r.resolve(&deps).expect("all range");
    assert_eq!(result["lib"], v(1, 10, 0));
}

// ── Exact constraint prevents upgrade ─────────────────────────────────────

#[test]
fn test_exact_constraint_prevents_upgrade() {
    let mut r = DependencyResolver::new();
    r.register("codec", vec![v(1, 0, 0), v(1, 5, 0), v(2, 0, 0)]);

    let deps = vec![dep("codec", VersionConstraint::Exact(v(1, 0, 0)))];
    let result = r.resolve(&deps).expect("exact");
    assert_eq!(result["codec"], v(1, 0, 0));
}

// ── Compatible constraint respects major boundary ─────────────────────────

#[test]
fn test_compatible_constraint_major_boundary() {
    let mut r = DependencyResolver::new();
    r.register("enc", vec![v(1, 0, 0), v(1, 9, 0), v(2, 0, 0)]);

    let deps = vec![dep("enc", VersionConstraint::Compatible(v(1, 0, 0)))];
    let result = r.resolve(&deps).expect("compatible");
    // 2.0.0 is outside major=1 boundary.
    assert_eq!(result["enc"], v(1, 9, 0));
}

// ── Concurrent constraints across many plugins ────────────────────────────

/// Registers 12 plugins, each with 5 available versions, and resolves all.
#[test]
fn test_12_plugins_resolved_correctly() {
    let mut r = DependencyResolver::new();
    let all_names: Vec<String> = (0..12).map(|i| format!("mega-plugin-{i}")).collect();

    for name in &all_names {
        let versions: Vec<SemVer> = (1..=5).map(|n| v(1, n, 0)).collect();
        r.register(name.clone(), versions);
    }

    let deps: Vec<PluginDependency> = all_names
        .iter()
        .map(|name| dep(name, VersionConstraint::AtLeast(v(1, 3, 0))))
        .collect();

    let result = r.resolve(&deps).expect("12 plugin resolve");
    assert_eq!(result.len(), 12);

    for name in &all_names {
        // Highest satisfying version for >=1.3.0 among {1.1, 1.2, 1.3, 1.4, 1.5}
        assert_eq!(result[name], v(1, 5, 0), "wrong version for {name}");
    }
}

// ── Range constraint ─────────────────────────────────────────────────────

#[test]
fn test_range_constraint_in_complex_graph() {
    let mut r = DependencyResolver::new();
    r.register("base", vec![v(1, 0, 0), v(1, 5, 0), v(2, 0, 0), v(3, 0, 0)]);
    r.register("ext-a", vec![v(1, 0, 0)]);
    r.register("ext-b", vec![v(1, 0, 0)]);

    let deps = vec![
        dep(
            "base",
            VersionConstraint::Range {
                min: v(1, 0, 0),
                max: v(2, 0, 0),
            },
        ),
        dep("ext-a", VersionConstraint::AtLeast(v(1, 0, 0))),
        dep("ext-b", VersionConstraint::AtLeast(v(1, 0, 0))),
    ];

    let result = r.resolve(&deps).expect("range complex");
    assert_eq!(result["base"], v(2, 0, 0)); // highest in [1.0, 2.0]
    assert_eq!(result["ext-a"], v(1, 0, 0));
    assert_eq!(result["ext-b"], v(1, 0, 0));
}

// ── Two conflicting Exact constraints for same plugin ─────────────────────

#[test]
fn test_two_exact_constraints_conflict() {
    let mut r = DependencyResolver::new();
    r.register("plug", vec![v(1, 0, 0), v(2, 0, 0)]);

    let deps = vec![
        dep("plug", VersionConstraint::Exact(v(1, 0, 0))),
        dep("plug", VersionConstraint::Exact(v(2, 0, 0))),
    ];

    // The intersection of Exact(1.0.0) and Exact(2.0.0) is empty.
    let err = r.resolve(&deps).expect_err("should conflict");
    assert!(err.to_string().contains("plug"));
}

// ── Version set with pre-release versions ─────────────────────────────────

#[test]
fn test_pre_release_versions_in_set() {
    let mut r = DependencyResolver::new();
    r.register(
        "lib",
        vec![SemVer::with_pre(1, 0, 0, "alpha"), v(1, 0, 0), v(1, 1, 0)],
    );

    let deps = vec![dep("lib", VersionConstraint::AtLeast(v(1, 0, 0)))];
    let result = r.resolve(&deps).expect("pre-release");
    // Highest satisfying: 1.1.0 (numeric comparison ignores pre).
    assert_eq!(result["lib"], v(1, 1, 0));
}
