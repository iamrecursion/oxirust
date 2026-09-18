//! System fingerprint module.
//!
//! Captures hardware and software environment information at benchmark time for
//! result reproducibility and cross-machine regression detection.

use serde::{Deserialize, Serialize};

/// A snapshot of the execution environment at benchmark time.
///
/// Used to determine whether two benchmark runs were conducted on sufficiently
/// similar hardware that a direct performance comparison is meaningful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemFingerprint {
    /// Operating system name (e.g. `"linux"`, `"macos"`, `"windows"`).
    pub os: String,
    /// CPU architecture (e.g. `"x86_64"`, `"aarch64"`).
    pub arch: String,
    /// CPU model string read from `/proc/cpuinfo` or equivalent (best-effort).
    pub cpu_model: String,
    /// Number of logical CPU cores available to the process.
    pub cpu_cores: u32,
    /// Total system RAM in megabytes (best-effort; may be 0 if unavailable).
    pub ram_mb: u64,
    /// Rust compiler version string (from `VERGEN_RUSTC_SEMVER` env or compile
    /// time constant).
    pub rust_version: String,
    /// ISO-8601 date string when the binary was built (`YYYY-MM-DD`).
    pub build_date: String,
}

impl SystemFingerprint {
    /// Capture a new fingerprint of the current execution environment.
    ///
    /// Fields that cannot be determined are populated with `"unknown"` or `0`.
    #[must_use]
    pub fn capture() -> Self {
        Self {
            os: Self::detect_os(),
            arch: Self::detect_arch(),
            cpu_model: Self::detect_cpu_model(),
            cpu_cores: Self::detect_cpu_cores(),
            ram_mb: Self::detect_ram_mb(),
            rust_version: Self::detect_rust_version(),
            build_date: Self::detect_build_date(),
        }
    }

    /// Serialise the fingerprint to a compact JSON string.
    ///
    /// # Panics
    ///
    /// This function does not panic — `serde_json` only fails on maps with
    /// non-string keys, which `SystemFingerprint` does not have.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialise a fingerprint from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` when the input is not valid JSON or is
    /// missing required fields.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Determine whether `other` is close enough to `self` for a regression
    /// comparison to be valid.
    ///
    /// The rule is: OS, CPU architecture, and CPU model must all match
    /// case-insensitively.  RAM and core count may differ (e.g. when some
    /// cores are offline) and software versions are intentionally ignored.
    #[must_use]
    pub fn matches_for_regression(&self, other: &Self) -> bool {
        self.os.to_lowercase() == other.os.to_lowercase()
            && self.arch.to_lowercase() == other.arch.to_lowercase()
            && Self::cpu_models_match(&self.cpu_model, &other.cpu_model)
    }

    // ------------------------------------------------------------------
    // Detection helpers
    // ------------------------------------------------------------------

    fn detect_os() -> String {
        std::env::consts::OS.to_string()
    }

    fn detect_arch() -> String {
        std::env::consts::ARCH.to_string()
    }

    fn detect_cpu_cores() -> u32 {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1) as u32
    }

    fn detect_cpu_model() -> String {
        // Linux: parse /proc/cpuinfo
        #[cfg(target_os = "linux")]
        {
            if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in info.lines() {
                    if line.starts_with("model name") {
                        if let Some(pos) = line.find(':') {
                            return line[pos + 1..].trim().to_string();
                        }
                    }
                }
            }
        }
        // macOS: use sysctl
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(out) = Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
            {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return s;
                }
            }
        }
        // Windows: read from registry via wmic
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            if let Ok(out) = Command::new("wmic")
                .args(["cpu", "get", "name", "/value"])
                .output()
            {
                let s = String::from_utf8_lossy(&out.stdout);
                for line in s.lines() {
                    if line.starts_with("Name=") {
                        return line[5..].trim().to_string();
                    }
                }
            }
        }
        "unknown".to_string()
    }

    fn detect_ram_mb() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
                for line in info.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(kb_str) = parts.get(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb / 1024;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(out) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if let Ok(bytes) = s.parse::<u64>() {
                    return bytes / (1024 * 1024);
                }
            }
        }
        0
    }

    fn detect_rust_version() -> String {
        // At compile time, `CARGO_PKG_RUST_VERSION` is the MSRV from Cargo.toml
        // (not the actual compiler version).  We prefer the env var set by
        // `vergen` if present, otherwise fall back to the MSRV constant.
        option_env!("VERGEN_RUSTC_SEMVER")
            .or(option_env!("RUSTC_VERSION"))
            .unwrap_or(env!("CARGO_PKG_RUST_VERSION"))
            .to_string()
    }

    fn detect_build_date() -> String {
        // `vergen` injects `VERGEN_BUILD_DATE`; fall back to compile-time epoch.
        if let Some(date) = option_env!("VERGEN_BUILD_DATE") {
            return date.to_string();
        }
        // Construct a date from __DATE__ equivalent via env vars injected by
        // build.rs, or return a generic placeholder.
        option_env!("BUILD_DATE").unwrap_or("unknown").to_string()
    }

    /// Fuzzy CPU model comparison: normalise whitespace and compare ignoring
    /// minor vendor suffixes / stepping information.
    fn cpu_models_match(a: &str, b: &str) -> bool {
        if a == "unknown" || b == "unknown" {
            // If we couldn't detect either model, assume they match.
            return true;
        }
        // Normalise: lowercase, collapse whitespace, strip trailing `@ N.NNGHz`.
        let norm = |s: &str| -> String {
            let s = s.to_lowercase();
            let s = if let Some(idx) = s.find(" @") {
                &s[..idx]
            } else {
                &s
            };
            s.split_whitespace().collect::<Vec<_>>().join(" ")
        };
        norm(a) == norm(b)
    }
}

impl std::fmt::Display for SystemFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{os}/{arch} | {cpu} ({cores} cores, {ram} MB RAM) | rust {rv} built {bd}",
            os = self.os,
            arch = self.arch,
            cpu = self.cpu_model,
            cores = self.cpu_cores,
            ram = self.ram_mb,
            rv = self.rust_version,
            bd = self.build_date,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- capture() ----

    #[test]
    fn test_capture_os_nonempty() {
        let fp = SystemFingerprint::capture();
        assert!(!fp.os.is_empty(), "OS string should not be empty");
    }

    #[test]
    fn test_capture_arch_nonempty() {
        let fp = SystemFingerprint::capture();
        assert!(!fp.arch.is_empty(), "arch string should not be empty");
    }

    #[test]
    fn test_capture_cpu_cores_positive() {
        let fp = SystemFingerprint::capture();
        assert!(fp.cpu_cores >= 1, "cpu_cores should be at least 1");
    }

    #[test]
    fn test_capture_rust_version_nonempty() {
        let fp = SystemFingerprint::capture();
        assert!(
            !fp.rust_version.is_empty(),
            "rust_version should not be empty"
        );
    }

    // ---- to_json / from_json round-trip ----

    #[test]
    fn test_json_round_trip() {
        let fp = SystemFingerprint::capture();
        let json = fp.to_json();
        let restored = SystemFingerprint::from_json(&json).expect("deserialisation failed");
        assert_eq!(fp.os, restored.os);
        assert_eq!(fp.arch, restored.arch);
        assert_eq!(fp.cpu_cores, restored.cpu_cores);
        assert_eq!(fp.rust_version, restored.rust_version);
    }

    #[test]
    fn test_json_contains_expected_keys() {
        let fp = SystemFingerprint::capture();
        let json = fp.to_json();
        assert!(json.contains("\"os\""));
        assert!(json.contains("\"arch\""));
        assert!(json.contains("\"cpu_cores\""));
        assert!(json.contains("\"ram_mb\""));
    }

    // ---- matches_for_regression ----

    #[test]
    fn test_matches_for_regression_identical() {
        let fp = SystemFingerprint::capture();
        assert!(
            fp.matches_for_regression(&fp),
            "fingerprint should match itself"
        );
    }

    #[test]
    fn test_matches_for_regression_different_os() {
        let fp = SystemFingerprint::capture();
        let mut other = fp.clone();
        other.os = "other-os".to_string();
        assert!(
            !fp.matches_for_regression(&other),
            "different OS should not match"
        );
    }

    #[test]
    fn test_matches_for_regression_different_arch() {
        let fp = SystemFingerprint::capture();
        let mut other = fp.clone();
        other.arch = "riscv64".to_string();
        assert!(
            !fp.matches_for_regression(&other),
            "different arch should not match"
        );
    }

    #[test]
    fn test_matches_for_regression_different_rust_version_ok() {
        let fp = SystemFingerprint::capture();
        let mut other = fp.clone();
        other.rust_version = "1.99.0".to_string();
        // Rust version difference is intentionally ignored.
        assert!(
            fp.matches_for_regression(&other),
            "different rust version should still match"
        );
    }

    // ---- cpu_models_match helper ----

    #[test]
    fn test_cpu_model_match_normalised() {
        assert!(SystemFingerprint::cpu_models_match(
            "Intel(R) Core(TM) i9-13900K @ 3.00GHz",
            "Intel(R) Core(TM) i9-13900K @ 3.00GHz"
        ));
    }

    #[test]
    fn test_cpu_model_match_strips_freq() {
        assert!(SystemFingerprint::cpu_models_match(
            "Intel(R) Core(TM) i9-13900K @ 3.00GHz",
            "Intel(R) Core(TM) i9-13900K @ 5.40GHz"
        ));
    }

    #[test]
    fn test_cpu_model_mismatch() {
        assert!(!SystemFingerprint::cpu_models_match(
            "AMD Ryzen 9 7950X",
            "Intel(R) Core(TM) i9-13900K"
        ));
    }

    // ---- Display ----

    #[test]
    fn test_display_contains_os() {
        let fp = SystemFingerprint::capture();
        let s = fp.to_string();
        assert!(s.contains(&fp.os), "Display output should contain OS: {s}");
    }
}
