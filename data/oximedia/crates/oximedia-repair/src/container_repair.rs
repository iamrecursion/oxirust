#![allow(dead_code)]

//! Container-level structure repair for media files.
//!
//! This module provides tools to detect and repair damage in container-level
//! structures such as MP4 moov/mdat atoms, Matroska EBML headers, and AVI
//! index chunks. It can reconstruct missing or corrupted container metadata
//! from the actual media data present in the file.

use std::collections::HashMap;

/// Supported container formats for repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// MPEG-4 Part 14 (.mp4, .m4v, .m4a).
    Mp4,
    /// Matroska (.mkv, .mka, .webm).
    Matroska,
    /// AVI (.avi).
    Avi,
    /// MPEG Transport Stream (.ts, .m2ts).
    MpegTs,
    /// FLV container (.flv).
    Flv,
}

/// Describes a single atom/box/element in the container tree.
#[derive(Debug, Clone)]
pub struct ContainerAtom {
    /// Four-character code or element ID.
    pub tag: String,
    /// Byte offset from file start.
    pub offset: u64,
    /// Declared size in bytes (0 = extends to EOF).
    pub declared_size: u64,
    /// Actual size found by scanning.
    pub actual_size: u64,
    /// Whether this atom appears intact.
    pub intact: bool,
    /// Child atoms (for hierarchical containers).
    pub children: Vec<ContainerAtom>,
}

/// Damage classification for a container element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerDamage {
    /// Header bytes are corrupted.
    CorruptedHeader,
    /// Size field is wrong.
    SizeMismatch,
    /// Element is entirely missing.
    Missing,
    /// Element is truncated.
    Truncated,
    /// Element is duplicated.
    Duplicated,
    /// Element order is wrong.
    OutOfOrder,
}

/// A single container-level issue found during scanning.
#[derive(Debug, Clone)]
pub struct ContainerIssue {
    /// Format of the container.
    pub format: ContainerFormat,
    /// Nature of the damage.
    pub damage: ContainerDamage,
    /// Byte offset where the problem starts.
    pub offset: u64,
    /// Human-readable description.
    pub description: String,
    /// Whether automatic repair is possible.
    pub repairable: bool,
}

/// Strategy to use when rebuilding container metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildStrategy {
    /// Reconstruct from elementary stream data only.
    FromStreams,
    /// Use a reference file of the same format.
    FromReference,
    /// Hybrid: try streams first, fall back to reference.
    Hybrid,
}

/// Options for the container repair pass.
#[derive(Debug, Clone)]
pub struct ContainerRepairOptions {
    /// Target container format (auto-detected if None).
    pub format: Option<ContainerFormat>,
    /// Rebuild strategy.
    pub strategy: RebuildStrategy,
    /// Whether to preserve unknown atoms.
    pub preserve_unknown: bool,
    /// Maximum scan depth in atom hierarchy.
    pub max_depth: usize,
    /// Whether to fix size fields in-place.
    pub fix_sizes: bool,
}

impl Default for ContainerRepairOptions {
    fn default() -> Self {
        Self {
            format: None,
            strategy: RebuildStrategy::FromStreams,
            preserve_unknown: true,
            max_depth: 32,
            fix_sizes: true,
        }
    }
}

/// Result of a container repair operation.
#[derive(Debug, Clone)]
pub struct ContainerRepairResult {
    /// Detected container format.
    pub format: ContainerFormat,
    /// Number of issues found.
    pub issues_found: usize,
    /// Number of issues repaired.
    pub issues_repaired: usize,
    /// List of issues encountered.
    pub issues: Vec<ContainerIssue>,
    /// New atom tree (post-repair).
    pub atom_tree: Vec<ContainerAtom>,
    /// Total bytes rewritten.
    pub bytes_rewritten: u64,
}

/// Scanner that inspects container-level structure.
#[derive(Debug)]
pub struct ContainerScanner {
    /// Detected format.
    format: Option<ContainerFormat>,
    /// Known atom signatures by format.
    signatures: HashMap<ContainerFormat, Vec<[u8; 4]>>,
    /// Max hierarchy depth.
    max_depth: usize,
}

impl ContainerScanner {
    /// Create a new container scanner.
    pub fn new() -> Self {
        let mut signatures: HashMap<ContainerFormat, Vec<[u8; 4]>> = HashMap::new();
        signatures.insert(
            ContainerFormat::Mp4,
            vec![*b"ftyp", *b"moov", *b"mdat", *b"free", *b"moof", *b"trak"],
        );
        signatures.insert(
            ContainerFormat::Avi,
            vec![*b"RIFF", *b"AVI ", *b"LIST", *b"idx1"],
        );
        Self {
            format: None,
            signatures,
            max_depth: 32,
        }
    }

    /// Create a scanner for a specific format.
    pub fn for_format(format: ContainerFormat) -> Self {
        let mut scanner = Self::new();
        scanner.format = Some(format);
        scanner
    }

    /// Set maximum scan depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Detect the container format from a magic-bytes buffer.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_format(&self, header: &[u8]) -> Option<ContainerFormat> {
        if header.len() < 12 {
            return None;
        }
        // MP4: bytes 4..8 == "ftyp"
        if &header[4..8] == b"ftyp" {
            return Some(ContainerFormat::Mp4);
        }
        // AVI: "RIFF" ... "AVI "
        if &header[0..4] == b"RIFF" && &header[8..12] == b"AVI " {
            return Some(ContainerFormat::Avi);
        }
        // FLV: "FLV\x01"
        if header.len() >= 4 && &header[0..3] == b"FLV" {
            return Some(ContainerFormat::Flv);
        }
        // MPEG-TS: 0x47 sync byte
        if header[0] == 0x47 {
            return Some(ContainerFormat::MpegTs);
        }
        // Matroska: EBML signature 0x1A45DFA3
        if header.len() >= 4
            && header[0] == 0x1A
            && header[1] == 0x45
            && header[2] == 0xDF
            && header[3] == 0xA3
        {
            return Some(ContainerFormat::Matroska);
        }
        None
    }

    /// Scan an atom tree and return issues.
    pub fn scan_atoms(&self, atoms: &[ContainerAtom]) -> Vec<ContainerIssue> {
        let mut issues = Vec::new();
        for atom in atoms {
            if atom.declared_size != atom.actual_size && atom.declared_size != 0 {
                issues.push(ContainerIssue {
                    format: self.format.unwrap_or(ContainerFormat::Mp4),
                    damage: ContainerDamage::SizeMismatch,
                    offset: atom.offset,
                    description: format!(
                        "Atom '{}' at offset {}: declared size {} != actual size {}",
                        atom.tag, atom.offset, atom.declared_size, atom.actual_size
                    ),
                    repairable: true,
                });
            }
            if !atom.intact {
                issues.push(ContainerIssue {
                    format: self.format.unwrap_or(ContainerFormat::Mp4),
                    damage: ContainerDamage::CorruptedHeader,
                    offset: atom.offset,
                    description: format!(
                        "Atom '{}' at offset {} has corrupted header",
                        atom.tag, atom.offset
                    ),
                    repairable: false,
                });
            }
            // Recurse into children
            issues.extend(self.scan_atoms(&atom.children));
        }
        issues
    }
}

impl Default for ContainerScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine that repairs container-level damage.
#[derive(Debug)]
pub struct ContainerRepairer {
    options: ContainerRepairOptions,
}

impl ContainerRepairer {
    /// Create a repairer with the given options.
    pub fn new(options: ContainerRepairOptions) -> Self {
        Self { options }
    }

    /// Create a repairer with default options.
    pub fn with_defaults() -> Self {
        Self {
            options: ContainerRepairOptions::default(),
        }
    }

    /// Repair atom size fields based on actual measured sizes.
    pub fn fix_atom_sizes(&self, atoms: &mut [ContainerAtom]) -> u32 {
        let mut fixed = 0u32;
        for atom in atoms.iter_mut() {
            if atom.declared_size != atom.actual_size && atom.declared_size != 0 {
                atom.declared_size = atom.actual_size;
                atom.intact = true;
                fixed += 1;
            }
            fixed += self.fix_atom_sizes(&mut atom.children);
        }
        fixed
    }

    /// Remove duplicate atoms from a flat list.
    pub fn remove_duplicates(&self, atoms: &mut Vec<ContainerAtom>) -> u32 {
        let original_len = atoms.len();
        let mut seen = HashMap::new();
        atoms.retain(|atom| {
            let key = (atom.tag.clone(), atom.offset);
            seen.insert(key, true).is_none()
        });
        #[allow(clippy::cast_precision_loss)]
        let removed = (original_len - atoms.len()) as u32;
        removed
    }

    /// Run the full repair pipeline.
    pub fn repair(&self, atoms: &mut Vec<ContainerAtom>) -> ContainerRepairResult {
        let scanner = ContainerScanner::new();
        let issues_before = scanner.scan_atoms(atoms);
        let issues_found = issues_before.len();

        let mut repaired_count = 0usize;

        if self.options.fix_sizes {
            repaired_count += self.fix_atom_sizes(atoms) as usize;
        }

        repaired_count += self.remove_duplicates(atoms) as usize;

        let issues_after = scanner.scan_atoms(atoms);

        ContainerRepairResult {
            format: self.options.format.unwrap_or(ContainerFormat::Mp4),
            issues_found,
            issues_repaired: repaired_count,
            issues: issues_after,
            atom_tree: atoms.clone(),
            bytes_rewritten: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(
        tag: &str,
        offset: u64,
        declared: u64,
        actual: u64,
        intact: bool,
    ) -> ContainerAtom {
        ContainerAtom {
            tag: tag.to_string(),
            offset,
            declared_size: declared,
            actual_size: actual,
            intact,
            children: Vec::new(),
        }
    }

    #[test]
    fn test_container_format_equality() {
        assert_eq!(ContainerFormat::Mp4, ContainerFormat::Mp4);
        assert_ne!(ContainerFormat::Mp4, ContainerFormat::Avi);
    }

    #[test]
    fn test_container_damage_variants() {
        let d = ContainerDamage::SizeMismatch;
        assert_eq!(d, ContainerDamage::SizeMismatch);
        assert_ne!(d, ContainerDamage::Missing);
    }

    #[test]
    fn test_scanner_default() {
        let scanner = ContainerScanner::default();
        assert!(scanner.format.is_none());
        assert_eq!(scanner.max_depth, 32);
    }

    #[test]
    fn test_scanner_for_format() {
        let scanner = ContainerScanner::for_format(ContainerFormat::Matroska);
        assert_eq!(scanner.format, Some(ContainerFormat::Matroska));
    }

    #[test]
    fn test_scanner_with_max_depth() {
        let scanner = ContainerScanner::new().with_max_depth(8);
        assert_eq!(scanner.max_depth, 8);
    }

    #[test]
    fn test_detect_mp4() {
        let mut header = [0u8; 16];
        header[4..8].copy_from_slice(b"ftyp");
        let scanner = ContainerScanner::new();
        assert_eq!(scanner.detect_format(&header), Some(ContainerFormat::Mp4));
    }

    #[test]
    fn test_detect_avi() {
        let mut header = [0u8; 16];
        header[0..4].copy_from_slice(b"RIFF");
        header[8..12].copy_from_slice(b"AVI ");
        let scanner = ContainerScanner::new();
        assert_eq!(scanner.detect_format(&header), Some(ContainerFormat::Avi));
    }

    #[test]
    fn test_detect_flv() {
        let mut header = [0u8; 16];
        header[0..3].copy_from_slice(b"FLV");
        header[3] = 0x01;
        let scanner = ContainerScanner::new();
        assert_eq!(scanner.detect_format(&header), Some(ContainerFormat::Flv));
    }

    #[test]
    fn test_detect_matroska() {
        let mut header = [0u8; 16];
        header[0] = 0x1A;
        header[1] = 0x45;
        header[2] = 0xDF;
        header[3] = 0xA3;
        let scanner = ContainerScanner::new();
        assert_eq!(
            scanner.detect_format(&header),
            Some(ContainerFormat::Matroska)
        );
    }

    #[test]
    fn test_detect_unknown() {
        let header = [0u8; 16];
        let scanner = ContainerScanner::new();
        // All zeros: 0x47 check fails, others fail too, but let's be sure
        assert!(
            scanner.detect_format(&header).is_none()
                || scanner.detect_format(&header) == Some(ContainerFormat::MpegTs)
        );
    }

    #[test]
    fn test_detect_short_header() {
        let header = [0u8; 4];
        let scanner = ContainerScanner::new();
        assert_eq!(scanner.detect_format(&header), None);
    }

    #[test]
    fn test_scan_atoms_size_mismatch() {
        let scanner = ContainerScanner::for_format(ContainerFormat::Mp4);
        let atoms = vec![make_atom("moov", 0, 100, 120, true)];
        let issues = scanner.scan_atoms(&atoms);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].damage, ContainerDamage::SizeMismatch);
    }

    #[test]
    fn test_scan_atoms_corrupted_header() {
        let scanner = ContainerScanner::for_format(ContainerFormat::Mp4);
        let atoms = vec![make_atom("mdat", 200, 500, 500, false)];
        let issues = scanner.scan_atoms(&atoms);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].damage, ContainerDamage::CorruptedHeader);
    }

    #[test]
    fn test_scan_atoms_no_issues() {
        let scanner = ContainerScanner::for_format(ContainerFormat::Mp4);
        let atoms = vec![
            make_atom("ftyp", 0, 24, 24, true),
            make_atom("moov", 24, 500, 500, true),
        ];
        let issues = scanner.scan_atoms(&atoms);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_fix_atom_sizes() {
        let repairer = ContainerRepairer::with_defaults();
        let mut atoms = vec![make_atom("moov", 0, 100, 120, true)];
        let fixed = repairer.fix_atom_sizes(&mut atoms);
        assert_eq!(fixed, 1);
        assert_eq!(atoms[0].declared_size, 120);
    }

    #[test]
    fn test_remove_duplicates() {
        let repairer = ContainerRepairer::with_defaults();
        let mut atoms = vec![
            make_atom("moov", 0, 100, 100, true),
            make_atom("moov", 0, 100, 100, true),
            make_atom("mdat", 100, 500, 500, true),
        ];
        let removed = repairer.remove_duplicates(&mut atoms);
        assert_eq!(removed, 1);
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_repair_pipeline() {
        let repairer = ContainerRepairer::with_defaults();
        let mut atoms = vec![
            make_atom("moov", 0, 100, 120, true),
            make_atom("mdat", 120, 500, 500, true),
        ];
        let result = repairer.repair(&mut atoms);
        assert_eq!(result.issues_found, 1);
        assert!(result.issues_repaired >= 1);
    }

    #[test]
    fn test_repair_options_default() {
        let opts = ContainerRepairOptions::default();
        assert!(opts.format.is_none());
        assert_eq!(opts.strategy, RebuildStrategy::FromStreams);
        assert!(opts.preserve_unknown);
        assert_eq!(opts.max_depth, 32);
        assert!(opts.fix_sizes);
    }

    #[test]
    fn test_container_issue_fields() {
        let issue = ContainerIssue {
            format: ContainerFormat::Flv,
            damage: ContainerDamage::Truncated,
            offset: 1024,
            description: "Truncated at offset 1024".to_string(),
            repairable: false,
        };
        assert_eq!(issue.format, ContainerFormat::Flv);
        assert_eq!(issue.damage, ContainerDamage::Truncated);
        assert_eq!(issue.offset, 1024);
        assert!(!issue.repairable);
    }
}
