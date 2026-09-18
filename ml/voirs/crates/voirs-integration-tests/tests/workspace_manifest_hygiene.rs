//! Workspace manifest hygiene checks.
//!
//! Regression guard for GitHub issue #5 (cool-japan/voirs), "Issues with
//! crates and doesn't build.", which reported two fresh-clone build failures:
//!
//!   1. `oxiarc-deflate` was declared in `[workspace.dependencies]` as
//!      `{ version = "...", path = "../oxiarc/oxiarc-deflate" }`. That path
//!      escapes the repository, so every clone without a sibling `oxiarc`
//!      checkout failed to even load the workspace:
//!      `failed to read .../oxiarc/oxiarc-deflate/Cargo.toml`.
//!   2. `procfs` is a Linux-only crate. It must stay inside a Linux-specific
//!      `[target.'cfg(target_os = "linux")'.dependencies]` table so that
//!      enabling the feature that uses it can never break a macOS or Windows
//!      build (the code path itself is guarded by
//!      `voirs-ffi/src/c_api/utils.rs::estimate_current_memory_usage`).
//!
//! Both defects are manifest-level, so they are caught here by parsing every
//! `Cargo.toml` in the repository rather than by compiling anything: a broken
//! manifest cannot be detected by a build that never starts.
//!
//! The same walk also enforces the COOLJAPAN compression policy (all
//! compression goes through OxiARC): no banned compression crate may be
//! declared directly, and `procfs` must be declared without its default
//! features, which would otherwise pull `flate2` into every Linux build.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Component, Path, PathBuf};

use toml::{Table, Value};

/// Dependency table names that may appear both at the top level of a manifest
/// and inside a `[target.'<spec>']` table.
const DEPENDENCY_KINDS: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

/// Crates that only build on Linux (they bind `/proc` or Linux-only system
/// libraries) and therefore must never appear in a platform-independent
/// dependency table.
const LINUX_ONLY_CRATES: &[&str] = &[
    "alsa",
    "alsa-sys",
    "inotify",
    "libpulse-binding",
    "libpulse-sys",
    "procfs",
    "udev",
];

/// Compression crates banned by COOLJAPAN policy; the `oxiarc-*` crates are
/// the replacements. They may still appear transitively (see the wrapper
/// lists in `deny.toml`), but never as a direct declaration in this
/// repository.
const BANNED_COMPRESSION_CRATES: &[&str] = &[
    "brotli",
    "bzip2",
    "bzip2-sys",
    "flate2",
    "lz4",
    "lz4-sys",
    "lz4_flex",
    "miniz_oxide",
    "snap",
    "tar",
    "zip",
    "zstd",
    "zstd-sys",
];

/// Directories that are never searched for manifests (build output and
/// third-party trees that this repository does not own).
const SKIPPED_DIRECTORIES: &[&str] = &["target", "node_modules"];

/// Where a dependency entry was declared, which decides how strictly it is
/// checked: only real package dependency tables are subject to the
/// platform-gating rule.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TableKind {
    /// `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]` and
    /// their `[target.'<spec>'...]` variants -- these actually pull crates
    /// into a package's build graph.
    Package,
    /// `[workspace.dependencies]` -- inert definitions that only take effect
    /// where a member references them with `workspace = true`.
    WorkspaceDefinition,
    /// `[patch.<registry>]` / `[replace]` -- source redirections.
    Redirect,
}

/// One dependency table found in a manifest, with enough context to report a
/// precise location back to the developer.
struct DependencyTable<'a> {
    /// Human readable table header, e.g. ``[target.'cfg(target_os = "linux")'.dependencies]``.
    label: String,
    kind: TableKind,
    /// The `cfg(...)` expression or target triple, when the table is target
    /// specific.
    target: Option<String>,
    entries: &'a Table,
}

/// Locate the repository root by walking up from this crate towards the first
/// manifest that declares a `[workspace]`.
fn repository_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join("Cargo.toml");
        if !candidate.is_file() {
            continue;
        }
        if parse_manifest(&candidate)?.contains_key("workspace") {
            return Ok(fs::canonicalize(ancestor)?);
        }
    }

    Err(format!(
        "no workspace manifest found above {}",
        manifest_dir.display()
    )
    .into())
}

/// Parse a manifest into a TOML table, annotating I/O and syntax errors with
/// the offending file.
fn parse_manifest(path: &Path) -> Result<Table, Box<dyn Error>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let manifest = text
        .parse::<Table>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(manifest)
}

/// Collect every `Cargo.toml` below `root`, skipping build output and hidden
/// directories.
fn collect_manifests(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut manifests = Vec::new();
    let mut pending = vec![root.to_path_buf()];

    while let Some(directory) = pending.pop() {
        for entry in
            fs::read_dir(&directory).map_err(|e| format!("{}: {e}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();

            if entry.file_type()?.is_dir() {
                if name.starts_with('.') || SKIPPED_DIRECTORIES.contains(&name.as_ref()) {
                    continue;
                }
                pending.push(path);
            } else if name == "Cargo.toml" {
                manifests.push(path);
            }
        }
    }

    manifests.sort();
    Ok(manifests)
}

/// Enumerate every dependency table of a manifest, including target specific
/// ones, `[workspace.dependencies]`, `[patch.*]` and `[replace]`.
fn dependency_tables(manifest: &Table) -> Vec<DependencyTable<'_>> {
    let mut tables = Vec::new();

    for kind in DEPENDENCY_KINDS {
        if let Some(entries) = manifest.get(*kind).and_then(Value::as_table) {
            tables.push(DependencyTable {
                label: format!("[{kind}]"),
                kind: TableKind::Package,
                target: None,
                entries,
            });
        }
    }

    if let Some(entries) = manifest
        .get("workspace")
        .and_then(Value::as_table)
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(Value::as_table)
    {
        tables.push(DependencyTable {
            label: "[workspace.dependencies]".to_string(),
            kind: TableKind::WorkspaceDefinition,
            target: None,
            entries,
        });
    }

    if let Some(targets) = manifest.get("target").and_then(Value::as_table) {
        for (spec, target) in targets {
            let Some(target) = target.as_table() else {
                continue;
            };
            for kind in DEPENDENCY_KINDS {
                if let Some(entries) = target.get(*kind).and_then(Value::as_table) {
                    tables.push(DependencyTable {
                        label: format!("[target.'{spec}'.{kind}]"),
                        kind: TableKind::Package,
                        target: Some(spec.clone()),
                        entries,
                    });
                }
            }
        }
    }

    if let Some(registries) = manifest.get("patch").and_then(Value::as_table) {
        for (registry, entries) in registries {
            if let Some(entries) = entries.as_table() {
                tables.push(DependencyTable {
                    label: format!("[patch.{registry}]"),
                    kind: TableKind::Redirect,
                    target: None,
                    entries,
                });
            }
        }
    }

    if let Some(entries) = manifest.get("replace").and_then(Value::as_table) {
        tables.push(DependencyTable {
            label: "[replace]".to_string(),
            kind: TableKind::Redirect,
            target: None,
            entries,
        });
    }

    tables
}

/// Resolve `.` and `..` components without touching the filesystem, so that a
/// path escaping the repository is still detectable when its target does not
/// exist (which is exactly the fresh-clone situation of issue #5).
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }

    normalized
}

/// The crate actually pulled in by a dependency entry, honoring renames such
/// as `metal-rs = { package = "metal" }`.
fn declared_package_name<'a>(key: &'a str, spec: &'a Value) -> &'a str {
    spec.as_table()
        .and_then(|spec| spec.get("package"))
        .and_then(Value::as_str)
        .unwrap_or(key)
}

/// Whether a `[target.'<spec>']` key selects Linux (and nothing else).
///
/// Accepts explicit Linux/Android triples as well as `cfg(...)` expressions
/// whose every `target_os` test names Linux or Android. Anything negated
/// (`not(...)`) or mentioning another OS is rejected, because such a table is
/// also active off Linux.
fn is_linux_only_target(spec: &str) -> bool {
    let normalized: String = spec.chars().filter(|c| !c.is_whitespace()).collect();

    if !normalized.starts_with("cfg(") {
        // Explicit target triple, e.g. `x86_64-unknown-linux-gnu`.
        return normalized.contains("-linux") || normalized.contains("-android");
    }

    if normalized.contains("not(") {
        return false;
    }

    let mut names_an_os = false;
    for fragment in normalized.split("target_os=\"").skip(1) {
        let os = fragment.split('"').next().unwrap_or_default();
        if os != "linux" && os != "android" {
            return false;
        }
        names_an_os = true;
    }

    names_an_os
}

/// Path of `path` relative to the repository root, for readable diagnostics.
fn display_relative(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(relative) => relative.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

/// Issue #5, first failure: a `path` dependency pointing outside the
/// repository makes the workspace unloadable for everyone who does not happen
/// to have the referenced sibling checkout.
#[test]
fn test_issue_5_path_dependencies_stay_inside_repository() -> Result<(), Box<dyn Error>> {
    let root = repository_root()?;
    let manifests = collect_manifests(&root)?;
    assert!(
        !manifests.is_empty(),
        "no Cargo.toml found under {}",
        root.display()
    );

    let mut failures = Vec::new();

    for manifest_path in &manifests {
        let manifest = parse_manifest(manifest_path)?;
        let manifest_dir = manifest_path
            .parent()
            .ok_or_else(|| format!("{} has no parent directory", manifest_path.display()))?;
        let location = display_relative(&root, manifest_path);

        for table in dependency_tables(&manifest) {
            for (name, spec) in table.entries {
                let Some(path) = spec
                    .as_table()
                    .and_then(|spec| spec.get("path"))
                    .and_then(Value::as_str)
                else {
                    continue;
                };

                let resolved = normalize_lexically(&manifest_dir.join(path));
                if !resolved.starts_with(&root) {
                    failures.push(format!(
                        "{location}: {} `{name}` has path = \"{path}\", which resolves to {} \
                         outside the repository",
                        table.label,
                        resolved.display()
                    ));
                } else if !resolved.join("Cargo.toml").is_file() {
                    failures.push(format!(
                        "{location}: {} `{name}` has path = \"{path}\", but {} contains no \
                         Cargo.toml",
                        table.label,
                        display_relative(&root, &resolved)
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "path dependencies must resolve inside the repository so that a fresh clone builds \
         (GitHub issue #5):\n  {}",
        failures.join("\n  ")
    );

    Ok(())
}

/// Issue #5, first failure (continued): the workspace member list itself must
/// resolve inside the repository, otherwise loading the workspace fails before
/// any dependency is even looked at.
#[test]
fn test_issue_5_workspace_members_resolve_inside_repository() -> Result<(), Box<dyn Error>> {
    let root = repository_root()?;
    let manifest = parse_manifest(&root.join("Cargo.toml"))?;
    let workspace = manifest
        .get("workspace")
        .and_then(Value::as_table)
        .ok_or("the repository root manifest declares no [workspace]")?;

    let mut failures = Vec::new();
    let mut checked = 0_usize;

    for list in ["members", "default-members"] {
        let Some(members) = workspace.get(list).and_then(Value::as_array) else {
            continue;
        };

        for member in members {
            let Some(member) = member.as_str() else {
                continue;
            };
            if member.contains('*') {
                // Glob patterns are expanded by Cargo; nothing to resolve here.
                continue;
            }

            checked += 1;
            let resolved = normalize_lexically(&root.join(member));
            if !resolved.starts_with(&root) {
                failures.push(format!(
                    "[workspace] {list} entry `{member}` escapes the repository"
                ));
            } else if !resolved.join("Cargo.toml").is_file() {
                failures.push(format!(
                    "[workspace] {list} entry `{member}` has no Cargo.toml"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "every workspace member must exist inside the repository (GitHub issue #5):\n  {}",
        failures.join("\n  ")
    );
    assert!(checked > 0, "the workspace declares no members to check");

    Ok(())
}

/// Issue #5, second failure: `procfs` (and the other Linux-only crates used
/// for platform integration) must be declared under a Linux target table, so
/// that no feature combination can drag them into a macOS or Windows build.
#[test]
fn test_issue_5_linux_only_dependencies_are_target_gated() -> Result<(), Box<dyn Error>> {
    let root = repository_root()?;
    let manifests = collect_manifests(&root)?;

    let mut failures = Vec::new();
    let mut gated = BTreeSet::new();

    for manifest_path in &manifests {
        let manifest = parse_manifest(manifest_path)?;
        let location = display_relative(&root, manifest_path);

        for table in dependency_tables(&manifest) {
            if table.kind != TableKind::Package {
                continue;
            }

            for (name, spec) in table.entries {
                let package = declared_package_name(name, spec);
                if !LINUX_ONLY_CRATES.contains(&package) {
                    continue;
                }

                match table.target.as_deref() {
                    Some(target) if is_linux_only_target(target) => {
                        gated.insert(package.to_string());
                    }
                    _ => failures.push(format!(
                        "{location}: Linux-only crate `{package}` is declared in {}; move it to \
                         [target.'cfg(target_os = \"linux\")'.{}]",
                        table.label,
                        if table.label.contains("dev-dependencies") {
                            "dev-dependencies"
                        } else if table.label.contains("build-dependencies") {
                            "build-dependencies"
                        } else {
                            "dependencies"
                        }
                    )),
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "Linux-only crates must be declared under a Linux target table (GitHub issue #5):\n  {}",
        failures.join("\n  ")
    );
    assert!(
        gated.contains("procfs"),
        "expected `procfs` to still be declared under a Linux target table; if it was removed on \
         purpose, drop it from LINUX_ONLY_CRATES in this test as well"
    );

    Ok(())
}

/// The helpers above encode the two rules of this file; check them against
/// hand-written cases so a false "everything is fine" cannot go unnoticed.
#[test]
fn test_issue_5_hygiene_helpers_reject_broken_manifests() -> Result<(), Box<dyn Error>> {
    // The exact shape that broke the fresh clone in issue #5.
    let broken = r#"
        [workspace.dependencies]
        oxiarc-deflate = { version = "0.2.5", path = "../oxiarc/oxiarc-deflate" }

        [dependencies]
        procfs = { version = "0.18", optional = true }
    "#
    .parse::<Table>()?;

    let root = Path::new("/repo");
    let manifest_dir = Path::new("/repo");
    let tables = dependency_tables(&broken);

    let escaping = tables
        .iter()
        .flat_map(|table| table.entries.iter())
        .filter_map(|(_, spec)| {
            spec.as_table()
                .and_then(|spec| spec.get("path"))
                .and_then(Value::as_str)
        })
        .filter(|path| !normalize_lexically(&manifest_dir.join(path)).starts_with(root))
        .count();
    assert_eq!(escaping, 1, "`../oxiarc/...` must be detected as escaping");

    let ungated = tables
        .iter()
        .filter(|table| table.kind == TableKind::Package)
        .flat_map(|table| {
            table
                .entries
                .iter()
                .map(move |(name, spec)| (table.target.clone(), declared_package_name(name, spec)))
        })
        .filter(|(target, package)| {
            LINUX_ONLY_CRATES.contains(package)
                && !target.as_deref().is_some_and(is_linux_only_target)
        })
        .count();
    assert_eq!(ungated, 1, "un-gated `procfs` must be detected");

    // Target predicates: Linux-only ones are accepted, everything else is not.
    assert!(is_linux_only_target("cfg(target_os = \"linux\")"));
    assert!(is_linux_only_target(
        "cfg(any(target_os = \"linux\", target_os = \"android\"))"
    ));
    assert!(is_linux_only_target("x86_64-unknown-linux-gnu"));
    assert!(!is_linux_only_target("cfg(unix)"));
    assert!(!is_linux_only_target("cfg(not(target_os = \"linux\"))"));
    assert!(!is_linux_only_target(
        "cfg(any(target_os = \"linux\", target_os = \"macos\"))"
    ));
    assert!(!is_linux_only_target("x86_64-apple-darwin"));

    // Lexical normalization keeps in-repository paths inside the repository.
    assert!(normalize_lexically(Path::new("/repo/crates/../patches/cudarc")).starts_with(root));
    assert!(!normalize_lexically(Path::new("/repo/../oxiarc")).starts_with(root));

    Ok(())
}

/// Names of banned compression crates declared directly in `manifest`, as
/// `(table label, declared key, crate)` triples. Every table kind counts:
/// even an inert `[workspace.dependencies]` entry is an invitation to use it.
fn banned_compression_declarations(manifest: &Table) -> Vec<(String, String, String)> {
    let mut found = Vec::new();

    for table in dependency_tables(manifest) {
        for (name, spec) in table.entries {
            let package = declared_package_name(name, spec);
            if BANNED_COMPRESSION_CRATES.contains(&package) {
                found.push((table.label.clone(), name.clone(), package.to_string()));
            }
        }
    }

    found
}

/// Declarations of `procfs` that keep its default features, which include
/// `flate2` (for `/proc/config.gz`).
fn procfs_declarations_with_default_features(manifest: &Table) -> Vec<String> {
    let mut found = Vec::new();

    for table in dependency_tables(manifest) {
        for (name, spec) in table.entries {
            if declared_package_name(name, spec) != "procfs" {
                continue;
            }
            let default_features = spec
                .as_table()
                .and_then(|spec| spec.get("default-features"))
                .and_then(Value::as_bool)
                .unwrap_or(true);
            if default_features {
                found.push(table.label.clone());
            }
        }
    }

    found
}

/// COOLJAPAN compression policy: no workspace manifest may declare `flate2`,
/// `zstd`, `bzip2`, `lz4`, `tar`, `snap`, `brotli`, `miniz_oxide` or `zip`
/// directly -- compression goes through `oxiarc-*`.
#[test]
fn test_no_banned_compression_crates_declared_directly() -> Result<(), Box<dyn Error>> {
    // The detector itself must see through the usual declaration shapes.
    let banned = r#"
        [workspace.dependencies]
        flate2 = "1"

        [dependencies]
        deflate = { package = "miniz_oxide", version = "0.8" }

        [target.'cfg(unix)'.dev-dependencies]
        zstd = { version = "0.13" }

        [dependencies.oxiarc-deflate]
        version = "0.4"
    "#
    .parse::<Table>()?;
    let detected: BTreeSet<String> = banned_compression_declarations(&banned)
        .into_iter()
        .map(|(_, _, package)| package)
        .collect();
    let expected: BTreeSet<String> = ["flate2", "miniz_oxide", "zstd"]
        .iter()
        .map(|name| name.to_string())
        .collect();
    assert_eq!(detected, expected);

    let root = repository_root()?;
    let mut failures = Vec::new();

    for manifest_path in collect_manifests(&root)? {
        let manifest = parse_manifest(&manifest_path)?;
        let location = display_relative(&root, &manifest_path);
        for (label, name, package) in banned_compression_declarations(&manifest) {
            failures.push(format!(
                "{location}: {label} declares `{name}` (crate `{package}`); use the oxiarc-* \
                 equivalent instead"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "banned compression crates must not be declared directly (COOLJAPAN policy):\n  {}",
        failures.join("\n  ")
    );

    Ok(())
}

/// `procfs`'s default features pull `flate2` (and `chrono`) into every Linux
/// build, although `voirs-ffi` only calls `Process::myself()?.stat()`.
#[test]
fn test_procfs_declared_without_default_features() -> Result<(), Box<dyn Error>> {
    let with_defaults = r#"
        [target.'cfg(target_os = "linux")'.dependencies]
        procfs = { version = "0.18", optional = true }
    "#
    .parse::<Table>()?;
    assert_eq!(
        procfs_declarations_with_default_features(&with_defaults).len(),
        1
    );

    let without_defaults = r#"
        [target.'cfg(target_os = "linux")'.dependencies]
        procfs = { version = "0.18", optional = true, default-features = false }
    "#
    .parse::<Table>()?;
    assert!(procfs_declarations_with_default_features(&without_defaults).is_empty());

    let root = repository_root()?;
    let mut failures = Vec::new();

    for manifest_path in collect_manifests(&root)? {
        let manifest = parse_manifest(&manifest_path)?;
        let location = display_relative(&root, &manifest_path);
        for label in procfs_declarations_with_default_features(&manifest) {
            failures.push(format!(
                "{location}: {label} declares `procfs` without `default-features = false`, \
                 which pulls flate2 into the Linux build"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "procfs must not enable its default features (COOLJAPAN compression policy):\n  {}",
        failures.join("\n  ")
    );

    Ok(())
}

/// Every feature entry a manifest's `default` feature activates, following
/// the crate's own feature definitions transitively. Entries that point
/// elsewhere (`dep:x`, `x/feature`, `x?/feature`) are recorded verbatim.
fn default_feature_closure(manifest: &Table) -> BTreeSet<String> {
    let features = manifest.get("features").and_then(Value::as_table);
    let mut closure = BTreeSet::new();
    let mut pending = vec![String::from("default")];

    while let Some(feature) = pending.pop() {
        let Some(entries) = features
            .and_then(|features| features.get(&feature))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for entry in entries.iter().filter_map(Value::as_str) {
            if closure.insert(entry.to_string()) {
                pending.push(entry.to_string());
            }
        }
    }

    closure
}

/// Opt-in codecs that must stay out of default builds, as `(package, feature)`:
/// parquet's flate2-backed GZIP codec and voirs-singing's MusicXML support
/// (the `musicxml` crate depends on `miniz_oxide`).
const OPT_IN_ONLY_FEATURES: &[(&str, &str)] = &[
    ("parquet", "flate2"),
    ("parquet", "flate2-zlib-rs"),
    ("parquet", "flate2-rust_backend"),
    ("parquet", "flate2-rust_backened"),
    ("parquet", "snap"),
    ("parquet", "lz4"),
    ("parquet", "brotli"),
    ("parquet", "zstd"),
    ("voirs-dataset", "parquet-gzip"),
    ("voirs-dataset", "parquet-snappy"),
    ("voirs-dataset", "parquet-lz4"),
    ("voirs-singing", "musicxml-support"),
];

/// Violations of [`OPT_IN_ONLY_FEATURES`] in one manifest: activated by its
/// default feature set, or baked into a dependency declaration (which would
/// turn the feature on for every build of the declaring crate).
fn default_enabled_opt_in_features(manifest: &Table) -> Vec<String> {
    let package = manifest
        .get("package")
        .and_then(Value::as_table)
        .and_then(|package| package.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut found = Vec::new();

    for entry in default_feature_closure(manifest) {
        let violates = OPT_IN_ONLY_FEATURES.iter().any(|(owner, feature)| {
            let direct = *owner == package && entry == *feature;
            let forwarded =
                entry == format!("{owner}/{feature}") || entry == format!("{owner}?/{feature}");
            direct || forwarded
        });
        if violates {
            found.push(format!("default features enable `{entry}`"));
        }
    }

    for table in dependency_tables(manifest) {
        for (name, spec) in table.entries {
            let owner = declared_package_name(name, spec);

            // parquet's own default feature set includes the GZIP codec, so
            // every non-inheriting declaration must switch defaults off.
            if owner == "parquet" {
                let spec_table = spec.as_table();
                let inherits = spec_table
                    .and_then(|spec| spec.get("workspace"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let defaults_off = spec_table
                    .and_then(|spec| spec.get("default-features"))
                    .and_then(Value::as_bool)
                    == Some(false);
                if !inherits && !defaults_off {
                    found.push(format!(
                        "{} `{name}` keeps parquet's default features (which include GZIP)",
                        table.label
                    ));
                }
            }

            let Some(requested) = spec
                .as_table()
                .and_then(|spec| spec.get("features"))
                .and_then(Value::as_array)
            else {
                continue;
            };
            for feature in requested.iter().filter_map(Value::as_str) {
                if OPT_IN_ONLY_FEATURES.contains(&(owner, feature)) {
                    found.push(format!(
                        "{} `{name}` requests feature `{feature}`",
                        table.label
                    ));
                }
            }
        }
    }

    found
}

/// COOLJAPAN compression policy, feature level: parquet's GZIP codec (flate2)
/// and MusicXML support (miniz_oxide) are opt-in features and must never be
/// part of any crate's default feature set or of a dependency declaration.
#[test]
fn test_default_features_exclude_opt_in_codecs() -> Result<(), Box<dyn Error>> {
    // The detector must catch the shapes that previously enabled them.
    let regressed = r#"
        [package]
        name = "voirs-cli"

        [dependencies]
        parquet = { version = "59", default-features = false, features = ["arrow", "flate2-zlib-rs"] }
        arrow-parquet = { package = "parquet", version = "59" }

        [features]
        default = ["singing"]
        singing = ["dep:voirs-singing", "voirs-singing/musicxml-support"]
    "#
    .parse::<Table>()?;
    // flate2-zlib-rs requested, parquet defaults kept, musicxml-support by default.
    assert_eq!(default_enabled_opt_in_features(&regressed).len(), 3);

    let opt_in_only = r#"
        [package]
        name = "voirs-dataset"

        [features]
        default = []
        parquet-gzip = ["parquet/flate2-zlib-rs"]
    "#
    .parse::<Table>()?;
    assert!(default_enabled_opt_in_features(&opt_in_only).is_empty());

    let root = repository_root()?;
    let mut failures = Vec::new();

    for manifest_path in collect_manifests(&root)? {
        let manifest = parse_manifest(&manifest_path)?;
        let location = display_relative(&root, &manifest_path);
        for violation in default_enabled_opt_in_features(&manifest) {
            failures.push(format!("{location}: {violation}"));
        }
    }

    assert!(
        failures.is_empty(),
        "parquet GZIP (flate2) and MusicXML (miniz_oxide) must stay opt-in \
         (voirs-dataset `parquet-gzip`, voirs-singing `musicxml-support`):\n  {}",
        failures.join("\n  ")
    );

    Ok(())
}

/// Crates replaced by Pure-Rust / OxiARC-friendly equivalents, which must not
/// come back as direct declarations:
///
/// - `candle-*` -> `oxicandle-*` (declared as `candle-core = { package =
///   "oxicandle-core" }`; upstream candle-core's `tokenizers` uses the C
///   `onig` backend),
/// - `tract-*` -> `oxionnx-proto` (tract-nnef pulled flate2 + tar),
/// - `backtrace` -> `std::backtrace` (pulled miniz_oxide),
/// - `async-compression` -> `voirs_sdk::http_compression` (brotli + flate2).
const REPLACED_CRATES: &[&str] = &[
    "async-compression",
    "backtrace",
    "candle-core",
    "candle-nn",
    "candle-transformers",
    "tract-core",
    "tract-nnef",
    "tract-onnx",
];

/// Direct declarations of [`REPLACED_CRATES`] (by effective package name) and
/// requests for tower-http's brotli/flate2-based (de)compression features.
fn replaced_declarations(manifest: &Table) -> Vec<String> {
    let mut found = Vec::new();

    for table in dependency_tables(manifest) {
        for (name, spec) in table.entries {
            let package = declared_package_name(name, spec);
            // `workspace = true` inherits the package (and any rename) from
            // `[workspace.dependencies]`, which is checked on its own.
            let inherits = spec
                .as_table()
                .and_then(|spec| spec.get("workspace"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !inherits && REPLACED_CRATES.contains(&package) {
                found.push(format!(
                    "{} declares `{name}` (crate `{package}`)",
                    table.label
                ));
            }
            if package == "tower-http" {
                let features = spec
                    .as_table()
                    .and_then(|spec| spec.get("features"))
                    .and_then(Value::as_array);
                for feature in features.into_iter().flatten().filter_map(Value::as_str) {
                    if feature.starts_with("compression-") || feature.starts_with("decompression-")
                    {
                        found.push(format!(
                            "{} requests tower-http feature `{feature}` (use voirs_sdk::http_compression)",
                            table.label
                        ));
                    }
                }
            }
        }
    }

    if let Some(features) = manifest.get("features").and_then(Value::as_table) {
        for entries in features.values().filter_map(Value::as_array) {
            for entry in entries.iter().filter_map(Value::as_str) {
                if let Some(feature) = entry.strip_prefix("tower-http/") {
                    if feature.starts_with("compression-") || feature.starts_with("decompression-")
                    {
                        found.push(format!("[features] forwards `{entry}`"));
                    }
                }
            }
        }
    }

    found
}

/// The replacements made for the COOLJAPAN default-build policy stay in place.
#[test]
fn test_replaced_crates_are_not_declared() -> Result<(), Box<dyn Error>> {
    let regressed = r#"
        [dependencies]
        candle-core = "0.11.0"
        oxi = { package = "oxicandle-core", version = "0.11.0" }
        backtrace = "0.3"
        tower-http = { version = "0.7", features = ["cors", "compression-gzip"] }

        [features]
        http = ["tower-http/compression-br"]
    "#
    .parse::<Table>()?;
    assert_eq!(replaced_declarations(&regressed).len(), 4);

    let renamed = r#"
        [dependencies]
        candle-core = { package = "oxicandle-core", version = "0.11.0" }
        candle-nn.workspace = true
        tower-http = { version = "0.7", features = ["cors", "trace"] }
    "#
    .parse::<Table>()?;
    assert!(replaced_declarations(&renamed).is_empty());

    let root = repository_root()?;
    let mut failures = Vec::new();
    for manifest_path in collect_manifests(&root)? {
        let manifest = parse_manifest(&manifest_path)?;
        let location = display_relative(&root, &manifest_path);
        for violation in replaced_declarations(&manifest) {
            failures.push(format!("{location}: {violation}"));
        }
    }

    assert!(
        failures.is_empty(),
        "replaced crates / features must not be declared again:\n  {}",
        failures.join("\n  ")
    );
    Ok(())
}
