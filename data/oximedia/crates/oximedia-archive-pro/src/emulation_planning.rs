//! Emulation planning for long-term format accessibility.
//!
//! This module provides tools to plan emulation strategies for digital preservation:
//! - **EmulationFeasibility** - Native / Emulated / Degraded / NotFeasible
//! - **SoftwareRegistry** - Find emulators by format
//! - **EmulationPlanner** - Plan the best emulation strategy
//! - **VirtualMachineSpec** - Recommend VM environment for emulation

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// EmulatedFormat
// ─────────────────────────────────────────────────────────────

/// Description of a format that requires emulation for playback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmulatedFormat {
    /// Format name (e.g., "QuickTime MPEG-4")
    pub format_name: String,
    /// Name of the emulator or compatibility layer required
    pub emulator: String,
    /// Emulator version string
    pub version: String,
    /// Additional notes on emulation quality or caveats
    pub notes: String,
}

impl EmulatedFormat {
    /// Create a new emulated format descriptor.
    #[must_use]
    pub fn new(
        format_name: impl Into<String>,
        emulator: impl Into<String>,
        version: impl Into<String>,
        notes: impl Into<String>,
    ) -> Self {
        Self {
            format_name: format_name.into(),
            emulator: emulator.into(),
            version: version.into(),
            notes: notes.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// EmulationFeasibility
// ─────────────────────────────────────────────────────────────

/// Indicates how well a format can be accessed via emulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmulationFeasibility {
    /// Format is natively supported on current hardware/software
    Native,
    /// Fully functional via an emulator
    Emulated,
    /// Functional but with reduced quality (audio/video artefacts, speed issues)
    Degraded,
    /// Cannot be emulated with known tools
    NotFeasible,
}

impl EmulationFeasibility {
    /// Returns the expected playback quality percentage (0–100).
    #[must_use]
    pub fn quality_pct(&self) -> f32 {
        match self {
            Self::Native => 100.0,
            Self::Emulated => 90.0,
            Self::Degraded => 60.0,
            Self::NotFeasible => 0.0,
        }
    }

    /// Returns the variant name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Native => "Native",
            Self::Emulated => "Emulated",
            Self::Degraded => "Degraded",
            Self::NotFeasible => "NotFeasible",
        }
    }

    /// Returns true if the format is usable (quality > 0%).
    #[must_use]
    pub fn is_usable(&self) -> bool {
        *self != Self::NotFeasible
    }
}

// ─────────────────────────────────────────────────────────────
// SoftwareEntry
// ─────────────────────────────────────────────────────────────

/// An entry describing an emulator or compatibility layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareEntry {
    /// Software name
    pub name: String,
    /// Software version
    pub version: String,
    /// List of format extensions / names this software can handle
    pub supported_formats: Vec<String>,
    /// Whether this is open-source software
    pub open_source: bool,
    /// Year of the most recent release
    pub last_updated_year: u32,
}

impl SoftwareEntry {
    /// Returns true if this software supports the given format (case-insensitive).
    #[must_use]
    pub fn supports_format(&self, format: &str) -> bool {
        let lower = format.to_lowercase();
        self.supported_formats
            .iter()
            .any(|f| f.to_lowercase() == lower)
    }

    /// Returns true if the software is considered actively maintained (updated within 3 years).
    ///
    /// Uses 2026 as the reference year.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.last_updated_year >= 2023
    }
}

// ─────────────────────────────────────────────────────────────
// SoftwareRegistry
// ─────────────────────────────────────────────────────────────

/// Registry of emulators and compatibility layers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SoftwareRegistry {
    /// All registered software entries
    pub entries: Vec<SoftwareEntry>,
}

impl SoftwareRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a registry pre-populated with common media emulators.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.entries.push(SoftwareEntry {
            name: "FFmpeg".to_string(),
            version: "6.1".to_string(),
            supported_formats: vec![
                "mp4".into(),
                "avi".into(),
                "mov".into(),
                "mkv".into(),
                "flv".into(),
                "wmv".into(),
                "rm".into(),
                "webm".into(),
                "ogg".into(),
                "flac".into(),
                "wav".into(),
            ],
            open_source: true,
            last_updated_year: 2025,
        });
        registry.entries.push(SoftwareEntry {
            name: "VLC".to_string(),
            version: "3.0".to_string(),
            supported_formats: vec![
                "mp4".into(),
                "avi".into(),
                "mkv".into(),
                "flv".into(),
                "rm".into(),
                "3gp".into(),
                "asf".into(),
                "wmv".into(),
            ],
            open_source: true,
            last_updated_year: 2024,
        });
        registry.entries.push(SoftwareEntry {
            name: "DOSBox".to_string(),
            version: "0.74".to_string(),
            supported_formats: vec!["fli".into(), "flc".into(), "avi".into()],
            open_source: true,
            last_updated_year: 2022,
        });
        registry
    }

    /// Register a new software entry.
    pub fn register(&mut self, entry: SoftwareEntry) {
        self.entries.push(entry);
    }

    /// Find the first software entry that can handle the given format.
    #[must_use]
    pub fn find_emulator(&self, format: &str) -> Option<&SoftwareEntry> {
        self.entries.iter().find(|e| e.supports_format(format))
    }

    /// Find all software entries that can handle the given format.
    #[must_use]
    pub fn find_all_emulators(&self, format: &str) -> Vec<&SoftwareEntry> {
        self.entries
            .iter()
            .filter(|e| e.supports_format(format))
            .collect()
    }

    /// Returns the number of registered entries.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

// ─────────────────────────────────────────────────────────────
// EmulationPlanner
// ─────────────────────────────────────────────────────────────

/// Plans the best emulation strategy for a given format.
pub struct EmulationPlanner;

impl EmulationPlanner {
    /// Determine the emulation feasibility for a given asset format.
    ///
    /// * `asset_format` – file extension or format name (case-insensitive)
    /// * `registry` – software registry to consult
    #[must_use]
    pub fn plan(asset_format: &str, registry: &SoftwareRegistry) -> EmulationFeasibility {
        let lower = asset_format.to_lowercase();

        // Native preservation formats – always natively supported
        let native = ["mkv", "ffv1", "flac", "wav", "png", "tiff", "jp2"];
        if native.contains(&lower.as_str()) {
            return EmulationFeasibility::Native;
        }

        // Find an emulator in the registry
        match registry.find_emulator(&lower) {
            Some(entry) if entry.is_active() => EmulationFeasibility::Emulated,
            Some(_) => EmulationFeasibility::Degraded, // stale software
            None => {
                // Heuristic: known-problematic legacy formats
                let legacy = ["3gp", "rm", "asf", "wmv"];
                if legacy.contains(&lower.as_str()) {
                    EmulationFeasibility::Degraded
                } else {
                    EmulationFeasibility::NotFeasible
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// VirtualMachineSpec
// ─────────────────────────────────────────────────────────────

/// Hardware/software specification for an emulation VM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualMachineSpec {
    /// Operating system name and version (e.g., "Ubuntu 22.04 LTS")
    pub os: String,
    /// CPU architecture (e.g., "x86_64")
    pub cpu_arch: String,
    /// Recommended memory in gigabytes
    pub memory_gb: u32,
    /// Recommended storage in gigabytes
    pub storage_gb: u32,
}

impl VirtualMachineSpec {
    /// Returns true if this VM has sufficient resources (≥ 8 GB RAM, ≥ 100 GB storage).
    #[must_use]
    pub fn is_adequate(&self) -> bool {
        self.memory_gb >= 8 && self.storage_gb >= 100
    }
}

/// Recommends a VM configuration based on the format to emulate.
pub struct EmulationEnvironment;

impl EmulationEnvironment {
    /// Recommend a `VirtualMachineSpec` for emulating the given format.
    ///
    /// Uses rule-based logic: video formats get more memory/storage than image/audio.
    #[must_use]
    pub fn recommend_vm(format: &str) -> VirtualMachineSpec {
        let lower = format.to_lowercase();
        let is_video = [
            "mp4", "avi", "mov", "mkv", "flv", "wmv", "rm", "webm", "3gp",
        ]
        .contains(&lower.as_str());
        let is_legacy = ["rm", "flv", "asf", "wmv", "3gp"].contains(&lower.as_str());

        if is_legacy {
            VirtualMachineSpec {
                os: "Ubuntu 20.04 LTS".to_string(),
                cpu_arch: "x86_64".to_string(),
                memory_gb: 8,
                storage_gb: 200,
            }
        } else if is_video {
            VirtualMachineSpec {
                os: "Ubuntu 22.04 LTS".to_string(),
                cpu_arch: "x86_64".to_string(),
                memory_gb: 16,
                storage_gb: 500,
            }
        } else {
            // Image, audio, or document
            VirtualMachineSpec {
                os: "Ubuntu 22.04 LTS".to_string(),
                cpu_arch: "x86_64".to_string(),
                memory_gb: 4,
                storage_gb: 100,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EmulationFeasibility ──────────────────────────────────

    #[test]
    fn test_quality_pct_native() {
        assert!((EmulationFeasibility::Native.quality_pct() - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_pct_not_feasible() {
        assert_eq!(EmulationFeasibility::NotFeasible.quality_pct(), 0.0);
    }

    #[test]
    fn test_is_usable() {
        assert!(EmulationFeasibility::Emulated.is_usable());
        assert!(!EmulationFeasibility::NotFeasible.is_usable());
    }

    #[test]
    fn test_feasibility_names() {
        assert_eq!(EmulationFeasibility::Degraded.name(), "Degraded");
        assert_eq!(EmulationFeasibility::Native.name(), "Native");
    }

    // ── SoftwareEntry ─────────────────────────────────────────

    #[test]
    fn test_supports_format_case_insensitive() {
        let entry = SoftwareEntry {
            name: "FFmpeg".into(),
            version: "6.0".into(),
            supported_formats: vec!["MKV".into(), "mp4".into()],
            open_source: true,
            last_updated_year: 2024,
        };
        assert!(entry.supports_format("mkv"));
        assert!(entry.supports_format("MP4"));
        assert!(!entry.supports_format("rm"));
    }

    #[test]
    fn test_is_active_recent() {
        let entry = SoftwareEntry {
            name: "X".into(),
            version: "1.0".into(),
            supported_formats: vec![],
            open_source: false,
            last_updated_year: 2025,
        };
        assert!(entry.is_active());
    }

    #[test]
    fn test_is_active_old() {
        let entry = SoftwareEntry {
            name: "X".into(),
            version: "1.0".into(),
            supported_formats: vec![],
            open_source: false,
            last_updated_year: 2015,
        };
        assert!(!entry.is_active());
    }

    // ── SoftwareRegistry ──────────────────────────────────────

    #[test]
    fn test_registry_find_emulator() {
        let registry = SoftwareRegistry::with_defaults();
        let entry = registry.find_emulator("mp4");
        assert!(entry.is_some());
    }

    #[test]
    fn test_registry_find_unknown_format() {
        let registry = SoftwareRegistry::with_defaults();
        let entry = registry.find_emulator("unknownxyz");
        assert!(entry.is_none());
    }

    #[test]
    fn test_registry_count() {
        let registry = SoftwareRegistry::with_defaults();
        assert!(registry.count() >= 3);
    }

    #[test]
    fn test_registry_register() {
        let mut registry = SoftwareRegistry::new();
        registry.register(SoftwareEntry {
            name: "Custom".into(),
            version: "1.0".into(),
            supported_formats: vec!["xyz".into()],
            open_source: true,
            last_updated_year: 2025,
        });
        assert_eq!(registry.count(), 1);
        assert!(registry.find_emulator("xyz").is_some());
    }

    // ── EmulationPlanner ──────────────────────────────────────

    #[test]
    fn test_plan_native_format() {
        let registry = SoftwareRegistry::with_defaults();
        let f = EmulationPlanner::plan("mkv", &registry);
        assert_eq!(f, EmulationFeasibility::Native);
    }

    #[test]
    fn test_plan_known_emulated_format() {
        let registry = SoftwareRegistry::with_defaults();
        let f = EmulationPlanner::plan("mp4", &registry);
        assert_eq!(f, EmulationFeasibility::Emulated);
    }

    #[test]
    fn test_plan_unknown_format() {
        let registry = SoftwareRegistry::new(); // empty
        let f = EmulationPlanner::plan("bizarreformat", &registry);
        assert_eq!(f, EmulationFeasibility::NotFeasible);
    }

    // ── VirtualMachineSpec ────────────────────────────────────

    #[test]
    fn test_recommend_vm_video() {
        let spec = EmulationEnvironment::recommend_vm("mp4");
        assert!(spec.memory_gb >= 8);
        assert!(spec.is_adequate());
    }

    #[test]
    fn test_recommend_vm_legacy() {
        let spec = EmulationEnvironment::recommend_vm("rm");
        assert_eq!(spec.os, "Ubuntu 20.04 LTS");
    }

    #[test]
    fn test_recommend_vm_image() {
        let spec = EmulationEnvironment::recommend_vm("png");
        assert!(spec.storage_gb >= 100);
    }

    #[test]
    fn test_vm_spec_is_adequate() {
        let adequate = VirtualMachineSpec {
            os: "Linux".into(),
            cpu_arch: "x86_64".into(),
            memory_gb: 16,
            storage_gb: 500,
        };
        assert!(adequate.is_adequate());

        let insufficient = VirtualMachineSpec {
            os: "Linux".into(),
            cpu_arch: "x86".into(),
            memory_gb: 2,
            storage_gb: 50,
        };
        assert!(!insufficient.is_adequate());
    }
}
