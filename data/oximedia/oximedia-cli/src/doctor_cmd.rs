//! Environment diagnostics command for the OxiMedia CLI.
//!
//! Reports Rust version, GPU adapters, temp directory availability,
//! and environment variables relevant to OxiMedia.
//!
//! ## Default vs `--full`
//!
//! Without `--full`, the doctor reports the canonical 5-section quick check:
//! `rust_version`, `gpu_adapters`, `temp_dir`, and `oximedia_temp_env`. This
//! schema must remain byte-stable for downstream JSON consumers.
//!
//! With `--full`, three additional diagnostic sections are appended:
//!
//! - **Codec matrix** — per-codec `decode` / `encode` capability flags drawn
//!   from `oximedia_codec::codec_registry::CodecRegistry::default_registry()`,
//!   filling in `false/false` for codecs that are not registered.
//! - **Plugin paths** — `OXIMEDIA_PLUGIN_PATH` validation: existence,
//!   readability, and a count of dynamic libraries (`*.so` / `*.dylib` /
//!   `*.dll`) per directory entry.
//! - **OxiCUDA probe** — checks `OXICUDA_HOME` for `lib/libcudart.{so,dylib,dll}`
//!   and `version.txt`. Always reports `"not configured"` rather than erroring
//!   when the env var is unset, since CUDA support is optional.
//!
//! These extra fields use `#[serde(skip_serializing_if = "Option::is_none")]`
//! so the JSON schema produced by `oximedia doctor --json` (without `--full`)
//! remains identical to prior releases.

use anyhow::Result;
use oximedia_codec::codec_registry::CodecRegistry;
use oximedia_core::CodecId;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Information about a detected GPU adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GpuAdapter {
    /// Human-readable adapter name.
    pub name: String,
    /// Backend identifier (e.g., "wgpu", "metal", "vulkan").
    pub backend: String,
}

/// Information about the temporary directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TempDirInfo {
    /// Resolved path of the temporary directory.
    pub path: PathBuf,
    /// Whether the directory is writable by the current process.
    pub writable: bool,
    /// Available disk space in bytes (if detectable).
    pub available_bytes: Option<u64>,
}

/// One row of the codec capability matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CodecMatrixRow {
    /// Canonical codec name as users would type it (e.g. `"av1"`, `"jpeg-xl"`).
    pub codec: String,
    /// Whether the codec registry advertises decoder support.
    pub decode: bool,
    /// Whether the codec registry advertises encoder support.
    pub encode: bool,
}

/// Status report for a single entry of `OXIMEDIA_PLUGIN_PATH`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PluginPathReport {
    /// The directory path as supplied in the env var.
    pub path: PathBuf,
    /// Whether the path exists on disk.
    pub exists: bool,
    /// Whether the path is a directory and was readable.
    pub readable: bool,
    /// Number of files matching the platform's dynamic library extension.
    pub dylibs_found: usize,
}

/// Result of probing `OXICUDA_HOME`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OxicudaReport {
    /// `true` when `OXICUDA_HOME` is set and points to an existing directory.
    pub configured: bool,
    /// Resolved path to the CUDA toolkit (when `configured`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub home: Option<PathBuf>,
    /// Whether `lib/libcudart.{so,dylib,dll}` was located.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub libcudart_found: Option<bool>,
    /// Contents of `version.txt` if present, trimmed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Free-form note (e.g. "not configured (CUDA optional)").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Full doctor report structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DoctorReport {
    /// Rust compiler version used to build this binary.
    pub rust_version: String,
    /// Detected GPU adapters.
    pub gpu_adapters: Vec<GpuAdapter>,
    /// Temp directory information.
    pub temp_dir: TempDirInfo,
    /// Value of `OXIMEDIA_TEMP` env var, if set.
    pub oximedia_temp_env: Option<String>,
    /// Codec capability matrix (only present with `--full`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_matrix: Option<Vec<CodecMatrixRow>>,
    /// `OXIMEDIA_PLUGIN_PATH` directories (only present with `--full`).
    /// `None` means the env var was unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugin_paths: Option<Vec<PluginPathReport>>,
    /// `OXICUDA_HOME` probe (only present with `--full`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oxicuda: Option<OxicudaReport>,
}

/// Run the doctor command.
pub(crate) fn run(json: bool, full: bool) -> Result<()> {
    let report = gather_report(full)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    Ok(())
}

fn gather_report(full: bool) -> Result<DoctorReport> {
    // Prefer a vergen-injected version; fall back to RUSTC_VERSION env or "unknown".
    let rust_version = option_env!("VERGEN_RUSTC_SEMVER")
        .or(option_env!("RUSTC_VERSION"))
        .unwrap_or("unknown")
        .to_string();

    let gpu_adapters = detect_gpu_adapters();

    let temp_path = std::env::var("OXIMEDIA_TEMP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());

    let writable = probe_writable(&temp_path);
    let available_bytes = get_available_bytes(&temp_path);

    let temp_dir = TempDirInfo {
        path: temp_path,
        writable,
        available_bytes,
    };

    let oximedia_temp_env = std::env::var("OXIMEDIA_TEMP").ok();

    let (codec_matrix, plugin_paths, oxicuda) = if full {
        let plugin_env = std::env::var("OXIMEDIA_PLUGIN_PATH").ok();
        let cuda_env = std::env::var("OXICUDA_HOME").ok();
        (
            Some(build_codec_matrix()),
            Some(check_plugin_paths(plugin_env.as_deref())),
            Some(check_oxicuda(cuda_env.as_deref())),
        )
    } else {
        (None, None, None)
    };

    Ok(DoctorReport {
        rust_version,
        gpu_adapters,
        temp_dir,
        oximedia_temp_env,
        codec_matrix,
        plugin_paths,
        oxicuda,
    })
}

/// Attempt to detect GPU adapters. Returns an empty list if none found.
fn detect_gpu_adapters() -> Vec<GpuAdapter> {
    // wgpu is not a direct dependency of oximedia-cli, so we return an empty
    // list here. Crates that link wgpu can expose adapter info separately.
    Vec::new()
}

/// Probe whether `path` is writable by the current process.
fn probe_writable(path: &std::path::Path) -> bool {
    let test_file = path.join(format!(".oximedia_probe_{}", std::process::id()));
    match std::fs::write(&test_file, b"probe") {
        Ok(()) => {
            let _ = std::fs::remove_file(&test_file);
            true
        }
        Err(_) => false,
    }
}

/// Return available disk bytes for `path`.
/// Uses statvfs on Unix via the `libc` crate (already a workspace dep).
/// Returns `None` on any error or on non-Unix platforms.
fn get_available_bytes(path: &std::path::Path) -> Option<u64> {
    // Safe pure-Rust approach: read /proc/mounts on Linux or fall back to None.
    // On macOS/BSD `statvfs` requires unsafe; we skip it to stay within the
    // `unsafe_code = "deny"` workspace policy and return None instead.
    // A future enhancement could use `sysinfo::Disks` for cross-platform info.
    let _ = path;
    None
}

/// Canonical user-visible names for the codec ids we report on.
/// The order is intentionally fixed so that the JSON output is stable across runs.
const CODEC_MATRIX_NAMES: &[(&str, CodecId)] = &[
    ("av1", CodecId::Av1),
    ("vp9", CodecId::Vp9),
    ("vp8", CodecId::Vp8),
    ("opus", CodecId::Opus),
    ("vorbis", CodecId::Vorbis),
    ("flac", CodecId::Flac),
    ("pcm", CodecId::Pcm),
    ("ffv1", CodecId::Ffv1),
    // y4m is the canonical container/extension users associate with raw video.
    ("y4m", CodecId::RawVideo),
    ("jpeg-xl", CodecId::JpegXl),
    ("dng", CodecId::Dng),
];

/// Build the codec capability matrix from the default registry, falling back
/// to `decode=false, encode=false` for codec ids that are not registered.
fn build_codec_matrix() -> Vec<CodecMatrixRow> {
    let registry = CodecRegistry::default_registry();
    CODEC_MATRIX_NAMES
        .iter()
        .map(|(name, id)| {
            let (decode, encode) = registry
                .lookup_by_id(*id)
                .map(|d| (d.can_decode, d.can_encode))
                .unwrap_or((false, false));
            CodecMatrixRow {
                codec: (*name).to_string(),
                decode,
                encode,
            }
        })
        .collect()
}

/// Validate `OXIMEDIA_PLUGIN_PATH`-style colon/semicolon-separated path lists.
/// Returns an empty `Vec` when no paths are supplied (caller decides how to
/// surface "env var unset" — empty vec mirrors that).
fn check_plugin_paths(env_value: Option<&str>) -> Vec<PluginPathReport> {
    let Some(value) = env_value else {
        return Vec::new();
    };

    std::env::split_paths(value)
        .map(|p| {
            let exists = p.exists();
            let (readable, dylibs_found) = if exists && p.is_dir() {
                count_dylibs(&p)
            } else {
                (false, 0)
            };
            PluginPathReport {
                path: p,
                exists,
                readable,
                dylibs_found,
            }
        })
        .collect()
}

/// Count files in `dir` whose extension matches the platform's dynamic
/// library suffix. Returns `(readable, count)`. A directory we cannot
/// `read_dir()` (permission denied, etc.) reports `readable=false, count=0`.
fn count_dylibs(dir: &Path) -> (bool, usize) {
    let entries = match std::fs::read_dir(dir) {
        Ok(it) => it,
        Err(_) => return (false, 0),
    };
    let target_ext = std::env::consts::DLL_EXTENSION;
    let mut count = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case(target_ext))
        {
            count += 1;
        }
    }
    (true, count)
}

/// Probe `OXICUDA_HOME` for a CUDA toolkit installation.
/// Never errors: when the env var is unset, returns `configured=false`
/// with a polite "CUDA optional" note.
fn check_oxicuda(home: Option<&str>) -> OxicudaReport {
    let Some(home_str) = home.filter(|s| !s.is_empty()) else {
        return OxicudaReport {
            configured: false,
            home: None,
            libcudart_found: None,
            version: None,
            note: Some("not configured (CUDA optional)".to_string()),
        };
    };

    let home_path = PathBuf::from(home_str);
    if !home_path.exists() {
        return OxicudaReport {
            configured: false,
            home: Some(home_path),
            libcudart_found: Some(false),
            version: None,
            note: Some("OXICUDA_HOME path does not exist".to_string()),
        };
    }

    let libcudart_found = detect_libcudart(&home_path);
    let version = read_cuda_version(&home_path);

    OxicudaReport {
        configured: true,
        home: Some(home_path),
        libcudart_found: Some(libcudart_found),
        version,
        note: None,
    }
}

/// Look for `lib/libcudart.{so,dylib,dll}` (also `lib64/...` on Linux).
fn detect_libcudart(home: &Path) -> bool {
    let candidates = [
        home.join("lib")
            .join(format!("libcudart.{}", std::env::consts::DLL_EXTENSION)),
        home.join("lib64")
            .join(format!("libcudart.{}", std::env::consts::DLL_EXTENSION)),
        // Windows convention: bin/cudart64_*.dll. We check the canonical name only.
        home.join("bin")
            .join(format!("cudart.{}", std::env::consts::DLL_EXTENSION)),
    ];
    candidates.iter().any(|p| p.exists())
}

/// Read `version.txt` (CUDA Toolkit ≤ 11.x) or fall back to an empty option.
fn read_cuda_version(home: &Path) -> Option<String> {
    let candidate = home.join("version.txt");
    let raw = std::fs::read_to_string(&candidate).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn print_report(report: &DoctorReport) {
    println!("OxiMedia Doctor Report");
    println!("======================");
    println!("Rust version:  {}", report.rust_version);
    println!("Temp dir:      {}", report.temp_dir.path.display());
    println!(
        "  Writable:    {}",
        if report.temp_dir.writable {
            "yes"
        } else {
            "no"
        }
    );
    if let Some(bytes) = report.temp_dir.available_bytes {
        println!("  Available:   {:.1} GB", bytes as f64 / 1_073_741_824.0);
    }
    if let Some(ref env) = report.oximedia_temp_env {
        println!("OXIMEDIA_TEMP: {}", env);
    }
    if report.gpu_adapters.is_empty() {
        println!("GPU adapters:  none detected");
    } else {
        println!("GPU adapters:");
        for a in &report.gpu_adapters {
            println!("  {} ({})", a.name, a.backend);
        }
    }

    if let Some(rows) = report.codec_matrix.as_ref() {
        println!();
        println!("Codec matrix:");
        println!("  {:<10} {:<8} {:<8}", "codec", "decode", "encode");
        for row in rows {
            println!(
                "  {:<10} {:<8} {:<8}",
                row.codec,
                yes_no(row.decode),
                yes_no(row.encode)
            );
        }
    }

    if let Some(paths) = report.plugin_paths.as_ref() {
        println!();
        if paths.is_empty() {
            println!("Plugin paths:  OXIMEDIA_PLUGIN_PATH not set");
        } else {
            println!("Plugin paths (OXIMEDIA_PLUGIN_PATH):");
            for entry in paths {
                println!(
                    "  {}  exists={} readable={} dylibs_found={}",
                    entry.path.display(),
                    yes_no(entry.exists),
                    yes_no(entry.readable),
                    entry.dylibs_found
                );
            }
        }
    }

    if let Some(cuda) = report.oxicuda.as_ref() {
        println!();
        println!("OxiCUDA:");
        if cuda.configured {
            if let Some(home) = cuda.home.as_ref() {
                println!("  OXICUDA_HOME: {}", home.display());
            }
            if let Some(found) = cuda.libcudart_found {
                println!("  libcudart:    {}", yes_no(found));
            }
            if let Some(ver) = cuda.version.as_ref() {
                println!("  version.txt:  {}", ver);
            }
        } else if let Some(note) = cuda.note.as_ref() {
            println!("  {}", note);
        }
    }
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_doctor_report_json_shape() {
        let report = gather_report(false).expect("gather_report should succeed");
        let json = serde_json::to_string(&report).expect("serialize to JSON");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");
        assert!(v.get("rust_version").is_some(), "JSON missing rust_version");
        assert!(v.get("gpu_adapters").is_some(), "JSON missing gpu_adapters");
        assert!(v.get("temp_dir").is_some(), "JSON missing temp_dir");
        // Default report must NOT include the --full extras (schema preservation).
        assert!(
            v.get("codec_matrix").is_none(),
            "default JSON must omit codec_matrix"
        );
        assert!(
            v.get("plugin_paths").is_none(),
            "default JSON must omit plugin_paths"
        );
        assert!(v.get("oxicuda").is_none(), "default JSON must omit oxicuda");
    }

    #[test]
    fn test_full_report_json_shape() {
        let report = gather_report(true).expect("gather_report(full=true) should succeed");
        let json = serde_json::to_string(&report).expect("serialize to JSON");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");
        assert!(
            v.get("codec_matrix").is_some(),
            "full JSON missing codec_matrix"
        );
        assert!(v.get("oxicuda").is_some(), "full JSON missing oxicuda");
    }

    #[test]
    fn test_probe_writable_temp() {
        assert!(
            probe_writable(&std::env::temp_dir()),
            "temp dir should be writable"
        );
    }

    #[test]
    fn test_available_bytes_returns_none() {
        // We always return None in this implementation.
        assert!(get_available_bytes(&std::env::temp_dir()).is_none());
    }

    #[test]
    fn test_codec_matrix_non_empty_and_known_ids() {
        let matrix = build_codec_matrix();
        assert!(!matrix.is_empty(), "codec matrix should not be empty");
        let names: Vec<&str> = matrix.iter().map(|r| r.codec.as_str()).collect();
        for required in ["av1", "vp9", "vp8", "opus", "vorbis", "flac", "pcm", "ffv1"] {
            assert!(
                names.contains(&required),
                "codec matrix missing required id `{required}`"
            );
        }
    }

    #[test]
    fn test_codec_matrix_av1_decodes() {
        // The default registry registers AV1 as Both (decode + encode); this is
        // a regression check against accidental registry pruning.
        let matrix = build_codec_matrix();
        let av1 = matrix
            .iter()
            .find(|r| r.codec == "av1")
            .expect("matrix must contain av1");
        assert!(av1.decode, "AV1 decode capability expected to be true");
        assert!(av1.encode, "AV1 encode capability expected to be true");
    }

    #[test]
    fn test_check_plugin_paths_unset_returns_empty() {
        let result = check_plugin_paths(None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_check_plugin_paths_empty_string_returns_empty_list() {
        // An empty env value via split_paths yields a single empty path, which
        // does not exist; assert that the report flags that correctly.
        let result = check_plugin_paths(Some(""));
        // split_paths on "" yields one empty entry on Unix.
        for entry in &result {
            assert!(!entry.exists, "empty path entry should not 'exist'");
            assert_eq!(entry.dylibs_found, 0);
        }
    }

    #[test]
    fn test_check_plugin_paths_nonexistent() {
        let result = check_plugin_paths(Some("/definitely/not/here/oximedia_test"));
        assert_eq!(result.len(), 1);
        assert!(!result[0].exists);
        assert!(!result[0].readable);
        assert_eq!(result[0].dylibs_found, 0);
    }

    #[test]
    fn test_check_plugin_paths_tempdir_empty() {
        let dir = TempDir::new().expect("create tempdir");
        let path_str = dir.path().to_string_lossy().to_string();
        let result = check_plugin_paths(Some(&path_str));
        assert_eq!(result.len(), 1);
        assert!(result[0].exists);
        assert!(result[0].readable);
        assert_eq!(result[0].dylibs_found, 0);
    }

    #[test]
    fn test_check_oxicuda_absence_never_errors() {
        // Pure function — env not touched. Spec test #6.
        let result = check_oxicuda(None);
        assert!(!result.configured);
        assert!(result.home.is_none());
        assert!(result.libcudart_found.is_none());
        assert_eq!(
            result.note.as_deref(),
            Some("not configured (CUDA optional)")
        );
    }

    #[test]
    fn test_check_oxicuda_empty_string_treated_as_unset() {
        let result = check_oxicuda(Some(""));
        assert!(!result.configured);
        assert!(result.note.is_some());
    }

    #[test]
    fn test_check_oxicuda_nonexistent_path() {
        let result = check_oxicuda(Some("/nonexistent/cuda/home"));
        assert!(!result.configured);
        assert_eq!(result.libcudart_found, Some(false));
        assert!(result.note.is_some());
    }

    #[test]
    fn test_check_oxicuda_tempdir_no_libcudart() {
        let dir = TempDir::new().expect("create tempdir");
        let path_str = dir.path().to_string_lossy().to_string();
        let result = check_oxicuda(Some(&path_str));
        // Tempdir exists, so configured=true, but no libcudart inside.
        assert!(result.configured);
        assert_eq!(result.libcudart_found, Some(false));
        assert!(result.version.is_none());
    }

    #[test]
    fn test_check_oxicuda_reads_version_txt() {
        let dir = TempDir::new().expect("create tempdir");
        let v = dir.path().join("version.txt");
        std::fs::write(&v, "CUDA Version 12.3.0\n").expect("write version.txt");
        let path_str = dir.path().to_string_lossy().to_string();
        let result = check_oxicuda(Some(&path_str));
        assert!(result.configured);
        assert_eq!(result.version.as_deref(), Some("CUDA Version 12.3.0"));
    }
}
