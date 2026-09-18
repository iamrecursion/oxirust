//! LTO (Linear Tape-Open) tape archive presets and inventory management.
//!
//! This module provides preset configurations and inventory tools for LTO tape
//! generations 6 through 10, supporting professional media archive workflows.

/// LTO tape generation identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum LtoGeneration {
    /// LTO-6 (native 2.5 TB, up to 160 MB/s).
    Lto6,
    /// LTO-7 (native 6.0 TB, up to 300 MB/s).
    Lto7,
    /// LTO-8 (native 12.0 TB, up to 360 MB/s).
    Lto8,
    /// LTO-9 (native 18.0 TB, up to 400 MB/s).
    Lto9,
    /// LTO-10 (native 36.0 TB, up to 780 MB/s, estimated).
    Lto10,
}

impl LtoGeneration {
    /// Native (uncompressed) capacity in terabytes.
    #[must_use]
    pub fn capacity_tb(&self) -> f64 {
        match self {
            Self::Lto6 => 2.5,
            Self::Lto7 => 6.0,
            Self::Lto8 => 12.0,
            Self::Lto9 => 18.0,
            Self::Lto10 => 36.0,
        }
    }

    /// Native sustained transfer rate in MB/s.
    #[must_use]
    pub fn transfer_rate_mbps(&self) -> f32 {
        match self {
            Self::Lto6 => 160.0,
            Self::Lto7 => 300.0,
            Self::Lto8 => 360.0,
            Self::Lto9 => 400.0,
            Self::Lto10 => 780.0,
        }
    }

    /// Human-readable generation label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Lto6 => "LTO-6",
            Self::Lto7 => "LTO-7",
            Self::Lto8 => "LTO-8",
            Self::Lto9 => "LTO-9",
            Self::Lto10 => "LTO-10",
        }
    }
}

/// Status of an individual LTO cartridge.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CartridgeStatus {
    /// Cartridge is available and ready for use.
    Available,
    /// Cartridge is currently being written.
    Writing,
    /// Cartridge is full (no writable capacity remaining).
    Full,
    /// Cartridge has been retired from active use.
    Retired,
    /// Cartridge has encountered an error.
    Error,
}

impl CartridgeStatus {
    /// Returns `true` if the cartridge can be used for writing.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Available | Self::Writing)
    }
}

/// An individual LTO tape cartridge.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LtoCartridge {
    /// Tape barcode label.
    pub barcode: String,
    /// LTO generation of this cartridge.
    pub generation: LtoGeneration,
    /// Used capacity in gigabytes.
    pub used_gb: f64,
    /// Current status of the cartridge.
    pub status: CartridgeStatus,
}

impl LtoCartridge {
    /// Remaining usable capacity in gigabytes.
    #[must_use]
    pub fn remaining_gb(&self) -> f64 {
        let capacity_gb = self.generation.capacity_tb() * 1024.0;
        (capacity_gb - self.used_gb).max(0.0)
    }

    /// Returns `true` if the cartridge has remaining writable capacity and is usable.
    #[must_use]
    pub fn has_space(&self) -> bool {
        self.status.is_usable() && self.remaining_gb() > 0.0
    }
}

/// LTO write preset configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LtoPreset {
    /// Target LTO generation.
    pub generation: LtoGeneration,
    /// Block size in kilobytes (common values: 512, 1024, 2048).
    pub block_size_kb: u32,
    /// Enable hardware compression.
    pub compression: bool,
    /// Perform a read-back verify pass after writing.
    pub verify_after_write: bool,
}

impl LtoPreset {
    /// Create the recommended preset for the given LTO generation.
    #[must_use]
    pub fn recommended(gen: LtoGeneration) -> Self {
        let (block_size_kb, compression, verify) = match gen {
            LtoGeneration::Lto6 => (1024, true, true),
            LtoGeneration::Lto7 => (1024, true, true),
            LtoGeneration::Lto8 => (2048, false, true), // already-compressed media
            LtoGeneration::Lto9 => (2048, false, true),
            LtoGeneration::Lto10 => (4096, false, true),
        };
        Self {
            generation: gen,
            block_size_kb,
            compression,
            verify_after_write: verify,
        }
    }
}

/// Inventory of LTO cartridges in a tape library.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct LtoLibraryInventory {
    /// All cartridges registered in this inventory.
    pub cartridges: Vec<LtoCartridge>,
}

impl LtoLibraryInventory {
    /// Create an empty inventory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total available (writable) capacity across all usable cartridges, in gigabytes.
    #[must_use]
    pub fn available_capacity_gb(&self) -> f64 {
        self.cartridges
            .iter()
            .filter(|c| c.has_space())
            .map(|c| c.remaining_gb())
            .sum()
    }

    /// Return references to all cartridges with the specified status.
    #[must_use]
    pub fn cartridges_by_status(&self, status: &CartridgeStatus) -> Vec<&LtoCartridge> {
        self.cartridges
            .iter()
            .filter(|c| &c.status == status)
            .collect()
    }

    /// Add a cartridge to the inventory.
    pub fn add_cartridge(&mut self, cartridge: LtoCartridge) {
        self.cartridges.push(cartridge);
    }

    /// Total number of cartridges in the inventory.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.cartridges.len()
    }

    /// Cartridges that are both usable and have remaining space.
    #[must_use]
    pub fn available_cartridges(&self) -> Vec<&LtoCartridge> {
        self.cartridges.iter().filter(|c| c.has_space()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lto6_capacity() {
        assert!((LtoGeneration::Lto6.capacity_tb() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lto9_capacity() {
        assert!((LtoGeneration::Lto9.capacity_tb() - 18.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lto10_transfer_rate() {
        assert!((LtoGeneration::Lto10.transfer_rate_mbps() - 780.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lto8_label() {
        assert_eq!(LtoGeneration::Lto8.label(), "LTO-8");
    }

    #[test]
    fn test_cartridge_status_usable() {
        assert!(CartridgeStatus::Available.is_usable());
        assert!(CartridgeStatus::Writing.is_usable());
        assert!(!CartridgeStatus::Full.is_usable());
        assert!(!CartridgeStatus::Retired.is_usable());
        assert!(!CartridgeStatus::Error.is_usable());
    }

    #[test]
    fn test_cartridge_remaining_gb() {
        let cartridge = LtoCartridge {
            barcode: "LTO001L8".to_string(),
            generation: LtoGeneration::Lto8,
            used_gb: 2048.0,
            status: CartridgeStatus::Available,
        };
        // LTO-8 = 12 TB = 12288 GB; used 2048 GB => 10240 GB remaining
        let remaining = cartridge.remaining_gb();
        assert!((remaining - 10240.0).abs() < 0.01);
    }

    #[test]
    fn test_cartridge_has_space() {
        let cartridge = LtoCartridge {
            barcode: "LTO002L9".to_string(),
            generation: LtoGeneration::Lto9,
            used_gb: 0.0,
            status: CartridgeStatus::Available,
        };
        assert!(cartridge.has_space());
    }

    #[test]
    fn test_cartridge_full_no_space() {
        let capacity_gb = LtoGeneration::Lto6.capacity_tb() * 1024.0;
        let cartridge = LtoCartridge {
            barcode: "LTO003L6".to_string(),
            generation: LtoGeneration::Lto6,
            used_gb: capacity_gb,
            status: CartridgeStatus::Full,
        };
        assert!(!cartridge.has_space());
    }

    #[test]
    fn test_preset_recommended_lto9() {
        let preset = LtoPreset::recommended(LtoGeneration::Lto9);
        assert_eq!(preset.block_size_kb, 2048);
        assert!(preset.verify_after_write);
    }

    #[test]
    fn test_preset_recommended_lto10_large_blocks() {
        let preset = LtoPreset::recommended(LtoGeneration::Lto10);
        assert_eq!(preset.block_size_kb, 4096);
    }

    #[test]
    fn test_inventory_available_capacity() {
        let mut inv = LtoLibraryInventory::new();
        inv.add_cartridge(LtoCartridge {
            barcode: "A001L9".to_string(),
            generation: LtoGeneration::Lto9,
            used_gb: 0.0,
            status: CartridgeStatus::Available,
        });
        inv.add_cartridge(LtoCartridge {
            barcode: "A002L9".to_string(),
            generation: LtoGeneration::Lto9,
            used_gb: 0.0,
            status: CartridgeStatus::Full,
        });
        // Only the Available cartridge contributes capacity.
        let cap = inv.available_capacity_gb();
        assert!(cap > 0.0);
        // LTO-9 = 18 TB = 18432 GB
        assert!((cap - 18432.0).abs() < 0.01);
    }

    #[test]
    fn test_inventory_cartridges_by_status() {
        let mut inv = LtoLibraryInventory::new();
        inv.add_cartridge(LtoCartridge {
            barcode: "E001L8".to_string(),
            generation: LtoGeneration::Lto8,
            used_gb: 0.0,
            status: CartridgeStatus::Error,
        });
        inv.add_cartridge(LtoCartridge {
            barcode: "A001L8".to_string(),
            generation: LtoGeneration::Lto8,
            used_gb: 0.0,
            status: CartridgeStatus::Available,
        });
        let errors = inv.cartridges_by_status(&CartridgeStatus::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].barcode, "E001L8");
    }
}
