//! Compile-only feature-matrix harness.
//!
//! Proves that every Cargo feature flag declared in this crate's `Cargo.toml`
//! builds **independently** — `cargo check --no-default-features --features
//! <flag>` must succeed for each one in isolation — so a downstream consumer
//! who picks exactly one feature never hits a missing re-export or an
//! accidental cross-feature dependency that only showed up because some
//! *other* feature happened to be enabled in CI.
//!
//! `full` is deliberately excluded from the sweep below: it is already
//! exercised end-to-end (build + run) by the `prelude_smoke` module in
//! `tests/integration.rs`, and re-checking it here would just double the
//! (already very large) cost of this harness for zero additional coverage.
//!
//! Because that cost is so large (~40 min of wall-clock time for 50+ nested
//! `cargo check` invocations), the sole test here is marked `#[ignore]` and is
//! therefore skipped by the default `cargo test` / `cargo nextest run`. Run it
//! explicitly with `cargo nextest run -p oximedia --run-ignored all` (or
//! `cargo test -p oximedia --test feature_matrix -- --ignored`), typically in
//! CI or before a release.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

/// Meta-feature that is validated elsewhere (see module docs) and therefore
/// skipped by this per-feature sweep.
const SKIP_FEATURES: &[&str] = &["full"];

/// Reads the `[features]` table out of this crate's own `Cargo.toml` and
/// returns every declared feature name except the ones in [`SKIP_FEATURES`].
fn discover_features_to_check() -> BTreeSet<String> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let contents = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "Reading manifest {} must succeed: {e}",
            manifest_path.display()
        )
    });
    let document: toml::Table = contents.parse().unwrap_or_else(|e| {
        panic!(
            "Parsing manifest {} as TOML must succeed: {e}",
            manifest_path.display()
        )
    });

    let features_table = document
        .get("features")
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| {
            panic!(
                "{} must contain a [features] table",
                manifest_path.display()
            )
        });

    features_table
        .keys()
        .filter(|name| !SKIP_FEATURES.contains(&name.as_str()))
        .cloned()
        .collect()
}

/// Runs `cargo check --no-default-features --features <feature>` against
/// this crate's manifest and returns the captured output.
fn run_cargo_check_for_feature(
    manifest_path: &std::path::Path,
    feature: &str,
) -> std::process::Output {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());

    Command::new(&cargo)
        .arg("check")
        .arg("--no-default-features")
        .arg("--features")
        .arg(feature)
        .arg("--lib")
        .arg("--manifest-path")
        .arg(manifest_path)
        .output()
        .unwrap_or_else(|e| {
            panic!("Spawning `cargo check --features {feature}` must succeed to launch: {e}")
        })
}

/// Every feature flag in `Cargo.toml` must build independently — proving
/// that the facade's `#[cfg(feature = "…")]` gates are self-contained and
/// that no per-crate feature silently relies on a sibling feature being
/// enabled at the same time (aside from documented implied activations,
/// e.g. `normalize` -> `metering`, which Cargo resolves automatically).
#[test]
#[ignore = "compile-only feature-matrix sweep: spawns `cargo check` once per feature (50+ nested invocations, ~40 min wall-clock); excluded from the default suite. Run explicitly with `cargo nextest run -p oximedia --run-ignored all` or `cargo test -p oximedia --test feature_matrix -- --ignored`."]
fn test_every_feature_builds_independently() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let features = discover_features_to_check();

    assert!(
        features.len() > 50,
        "Expected to discover a large number of per-crate feature flags in \
         Cargo.toml, only found {}: {features:?}",
        features.len()
    );

    let mut failures: Vec<(String, String)> = Vec::new();

    for feature in &features {
        let output = run_cargo_check_for_feature(&manifest_path, feature);
        if !output.status.success() {
            failures.push((
                feature.clone(),
                String::from_utf8_lossy(&output.stderr).into_owned(),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} features failed to build independently:\n{}",
        failures.len(),
        features.len(),
        failures
            .iter()
            .map(|(feature, stderr)| format!("--- {feature} ---\n{stderr}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
