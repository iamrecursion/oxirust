//! `BagIt` package format implementation
//!
//! `BagIt` is a hierarchical file system packaging format for storage and transfer
//! of digital content. See: <https://tools.ietf.org/html/rfc8493>

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator, ChecksumVerifier, FileChecksum};
use crate::{Error, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// `BagIt` version
const BAGIT_VERSION: &str = "1.0";
const BAGIT_ENCODING: &str = "UTF-8";

/// `BagIt` package
#[derive(Debug, Clone)]
pub struct BagItPackage {
    /// Root directory of the bag
    pub root: PathBuf,
    /// Bag metadata
    pub metadata: HashMap<String, String>,
    /// Checksum algorithm used
    pub algorithm: ChecksumAlgorithm,
}

impl BagItPackage {
    /// Load an existing `BagIt` package
    ///
    /// # Errors
    ///
    /// Returns an error if the bag is invalid or cannot be read
    pub fn load(root: &Path) -> Result<Self> {
        let bagit_txt = root.join("bagit.txt");
        if !bagit_txt.exists() {
            return Err(Error::InvalidBag("Missing bagit.txt".to_string()));
        }

        let metadata = Self::read_metadata(root)?;
        let algorithm = Self::detect_algorithm(root)?;

        Ok(Self {
            root: root.to_path_buf(),
            metadata,
            algorithm,
        })
    }

    /// Read bag metadata from bag-info.txt
    fn read_metadata(root: &Path) -> Result<HashMap<String, String>> {
        let bag_info = root.join("bag-info.txt");
        let mut metadata = HashMap::new();

        if bag_info.exists() {
            let file = File::open(bag_info)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if let Some((key, value)) = line.split_once(':') {
                    metadata.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }

        Ok(metadata)
    }

    /// Detect the checksum algorithm used
    fn detect_algorithm(root: &Path) -> Result<ChecksumAlgorithm> {
        // Try different manifest filename formats
        for (algo, names) in &[
            (
                ChecksumAlgorithm::Sha256,
                vec!["manifest-sha-256.txt", "manifest-sha256.txt"],
            ),
            (
                ChecksumAlgorithm::Sha512,
                vec!["manifest-sha-512.txt", "manifest-sha512.txt"],
            ),
            (ChecksumAlgorithm::Md5, vec!["manifest-md5.txt"]),
        ] {
            for name in names {
                if root.join(name).exists() {
                    return Ok(*algo);
                }
            }
        }
        Err(Error::InvalidBag("No manifest file found".to_string()))
    }

    /// Get the data directory
    #[must_use]
    pub fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }

    /// List all files in the bag
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest cannot be read
    pub fn list_files(&self) -> Result<Vec<PathBuf>> {
        let manifest = self.read_manifest()?;
        Ok(manifest.into_keys().collect())
    }

    /// Read the manifest file
    fn read_manifest(&self) -> Result<HashMap<PathBuf, String>> {
        let manifest_name = format!("manifest-{}.txt", self.algorithm.name().to_lowercase());
        let manifest_path = self.root.join(manifest_name);

        let mut manifest = HashMap::new();
        let file = File::open(manifest_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Some((hash, path)) = line.split_once(char::is_whitespace) {
                manifest.insert(PathBuf::from(path.trim()), hash.trim().to_string());
            }
        }

        Ok(manifest)
    }
}

/// `BagIt` package builder
pub struct BagItBuilder {
    root: PathBuf,
    algorithm: ChecksumAlgorithm,
    metadata: HashMap<String, String>,
    files: Vec<(PathBuf, PathBuf)>, // (source, destination in bag)
}

impl BagItBuilder {
    /// Create a new `BagIt` builder
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            algorithm: ChecksumAlgorithm::Sha256,
            metadata: HashMap::new(),
            files: Vec::new(),
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add bag metadata
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a file to the bag
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_file(mut self, source: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }

        let filename = source
            .file_name()
            .ok_or_else(|| Error::InvalidBag("Invalid filename".to_string()))?;
        let dest = PathBuf::from("data").join(filename);
        self.files.push((source.to_path_buf(), dest));
        Ok(self)
    }

    /// Add a file with a custom destination path in the bag
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_file_as(mut self, source: &Path, dest: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }

        let dest_in_data = PathBuf::from("data").join(dest);
        self.files.push((source.to_path_buf(), dest_in_data));
        Ok(self)
    }

    /// Build the `BagIt` package
    ///
    /// # Errors
    ///
    /// Returns an error if the bag cannot be created
    pub fn build(self) -> Result<BagItPackage> {
        // Create bag directory structure
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(self.root.join("data"))?;

        // Write bagit.txt
        let bagit_txt = self.root.join("bagit.txt");
        let mut file = File::create(bagit_txt)?;
        writeln!(file, "BagIt-Version: {BAGIT_VERSION}")?;
        writeln!(file, "Tag-File-Character-Encoding: {BAGIT_ENCODING}")?;

        // Copy files first, then generate checksums in parallel using rayon
        let mut dest_paths: Vec<(PathBuf, PathBuf)> = Vec::new(); // (dest_in_bag, dest_relative)
        for (source, dest) in &self.files {
            let target = self.root.join(dest);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &target)?;
            dest_paths.push((target, dest.clone()));
        }

        let algorithm = self.algorithm;
        let generator = ChecksumGenerator::new().with_algorithms(vec![algorithm]);

        // Parallel checksum generation via rayon par_iter
        let checksums: Vec<(PathBuf, FileChecksum)> = dest_paths
            .par_iter()
            .map(|(target, dest)| generator.generate_file(target).map(|cs| (dest.clone(), cs)))
            .collect::<Result<Vec<_>>>()?;

        // Write manifest
        let manifest_name = format!("manifest-{}.txt", self.algorithm.name().to_lowercase());
        let manifest_path = self.root.join(manifest_name);
        let mut manifest_file = File::create(manifest_path)?;

        for (path, checksum) in &checksums {
            if let Some(hash) = checksum.checksums.get(&self.algorithm) {
                writeln!(manifest_file, "{}  {}", hash, path.display())?;
            }
        }

        // Write bag-info.txt
        let mut metadata = self.metadata.clone();
        metadata
            .entry("Bagging-Date".to_string())
            .or_insert_with(|| chrono::Utc::now().format("%Y-%m-%d").to_string());
        metadata
            .entry("Bag-Software-Agent".to_string())
            .or_insert_with(|| "oximedia-archive-pro".to_string());
        metadata
            .entry("Payload-Oxum".to_string())
            .or_insert_with(|| {
                let total_size: u64 = checksums.iter().map(|(_, cs)| cs.size).sum();
                format!("{}.{}", total_size, checksums.len())
            });

        let bag_info = self.root.join("bag-info.txt");
        let mut info_file = File::create(bag_info)?;
        for (key, value) in &metadata {
            writeln!(info_file, "{key}: {value}")?;
        }

        Ok(BagItPackage {
            root: self.root,
            metadata,
            algorithm: self.algorithm,
        })
    }
}

/// `BagIt` package validator
pub struct BagItValidator;

impl BagItValidator {
    /// Validate a `BagIt` package
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails
    pub fn validate(bag: &BagItPackage) -> Result<()> {
        // Check for required files
        if !bag.root.join("bagit.txt").exists() {
            return Err(Error::InvalidBag("Missing bagit.txt".to_string()));
        }

        // Read and verify manifest
        let manifest = bag.read_manifest()?;
        let verifier = ChecksumVerifier::with_algorithms(vec![bag.algorithm]);

        for (path, expected_hash) in manifest {
            let full_path = bag.root.join(&path);
            if !full_path.exists() {
                return Err(Error::InvalidBag(format!(
                    "Missing file in manifest: {}",
                    path.display()
                )));
            }

            let checksum = FileChecksum {
                path: full_path.clone(),
                size: fs::metadata(&full_path)?.len(),
                checksums: [(bag.algorithm, expected_hash)].into_iter().collect(),
                timestamp: chrono::Utc::now(),
            };

            let report = verifier.verify_file(&checksum)?;
            if !report.is_success() {
                return Err(Error::InvalidBag(format!(
                    "Checksum mismatch for {}",
                    path.display()
                )));
            }
        }

        Ok(())
    }
}

// ─── BagIt v1.0 Compliance Verification ──────────────────────────────────────

/// Severity level of a compliance issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComplianceLevel {
    /// Fatal structural error – the bag is not spec-conformant.
    Error,
    /// Non-fatal advisory issue.
    Warning,
    /// Informational note.
    Info,
}

/// A single compliance finding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComplianceFinding {
    /// Severity of the finding.
    pub level: ComplianceLevel,
    /// Human-readable description.
    pub message: String,
}

impl ComplianceFinding {
    fn error(msg: impl Into<String>) -> Self {
        Self {
            level: ComplianceLevel::Error,
            message: msg.into(),
        }
    }

    fn warning(msg: impl Into<String>) -> Self {
        Self {
            level: ComplianceLevel::Warning,
            message: msg.into(),
        }
    }

    fn info(msg: impl Into<String>) -> Self {
        Self {
            level: ComplianceLevel::Info,
            message: msg.into(),
        }
    }
}

/// Report produced by `verify_bagit_v1_compliance`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BagitComplianceReport {
    /// Whether the bag meets all mandatory BagIt v1.0 requirements (no Error findings).
    pub is_compliant: bool,
    /// Individual findings (errors, warnings, info notes).
    pub findings: Vec<ComplianceFinding>,
    /// Detected BagIt-Version string (e.g. `"1.0"`).
    pub detected_version: Option<String>,
    /// Detected Tag-File-Character-Encoding string.
    pub detected_encoding: Option<String>,
    /// Tag-manifest filenames found (e.g. `["tagmanifest-sha256.txt"]`).
    pub tag_manifests: Vec<String>,
    /// Whether a `fetch.txt` was present.
    pub has_fetch_txt: bool,
}

impl BagitComplianceReport {
    /// Returns `true` when there are no `Error`-level findings.
    #[must_use]
    pub fn is_fully_compliant(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.level == ComplianceLevel::Error)
    }
}

/// Verify that the bag rooted at `bag_path` conforms to RFC 8493 (BagIt v1.0).
///
/// Checks performed:
/// 1. `bagit.txt` exists and declares `BagIt-Version: 1.0`.
/// 2. `Tag-File-Character-Encoding` is declared in `bagit.txt`.
/// 3. At least one payload manifest (`manifest-*.txt`) is present.
/// 4. At least one tag-manifest (`tagmanifest-*.txt`) is present (SHOULD requirement).
/// 5. If `fetch.txt` is present, its existence is noted.
/// 6. The `data/` payload directory exists.
///
/// # Errors
///
/// Returns an error only if `bag_path` is not accessible (I/O error).
/// Compliance failures are reported as findings inside the returned
/// `BagitComplianceReport` rather than as `Err` variants.
pub fn verify_bagit_v1_compliance(bag_path: &Path) -> Result<BagitComplianceReport> {
    let mut findings: Vec<ComplianceFinding> = Vec::new();
    let mut detected_version: Option<String> = None;
    let mut detected_encoding: Option<String> = None;
    let mut tag_manifests: Vec<String> = Vec::new();

    // ── 1. bagit.txt presence ────────────────────────────────────────────────
    let bagit_txt = bag_path.join("bagit.txt");
    if !bagit_txt.exists() {
        findings.push(ComplianceFinding::error(
            "bagit.txt is missing — required by RFC 8493 §2.1.1",
        ));
    } else {
        // Parse key/value pairs from bagit.txt
        let file = File::open(&bagit_txt)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some((key, value)) = trimmed.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                match key {
                    "BagIt-Version" => {
                        detected_version = Some(value.to_string());
                        if value != "1.0" {
                            findings.push(ComplianceFinding::warning(format!(
                                "BagIt-Version is '{value}'; expected '1.0' for v1.0 compliance"
                            )));
                        } else {
                            findings
                                .push(ComplianceFinding::info("BagIt-Version: 1.0 — confirmed"));
                        }
                    }
                    "Tag-File-Character-Encoding" => {
                        detected_encoding = Some(value.to_string());
                        if !value.eq_ignore_ascii_case("utf-8")
                            && !value.eq_ignore_ascii_case("utf8")
                        {
                            findings.push(ComplianceFinding::warning(format!(
                                "Tag-File-Character-Encoding is '{value}'; UTF-8 is strongly recommended"
                            )));
                        } else {
                            findings.push(ComplianceFinding::info(
                                "Tag-File-Character-Encoding: UTF-8 — confirmed",
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        if detected_version.is_none() {
            findings.push(ComplianceFinding::error(
                "bagit.txt lacks 'BagIt-Version' label — required by RFC 8493 §2.1.1",
            ));
        }
        if detected_encoding.is_none() {
            findings.push(ComplianceFinding::error(
                "bagit.txt lacks 'Tag-File-Character-Encoding' label — required by RFC 8493 §2.1.1",
            ));
        }
    }

    // ── 2. Payload manifest check ────────────────────────────────────────────
    let payload_manifest_names = [
        "manifest-md5.txt",
        "manifest-sha1.txt",
        "manifest-sha256.txt",
        "manifest-sha-256.txt",
        "manifest-sha512.txt",
        "manifest-sha-512.txt",
    ];
    let has_payload_manifest = payload_manifest_names
        .iter()
        .any(|name| bag_path.join(name).exists());
    if !has_payload_manifest {
        findings.push(ComplianceFinding::error(
            "No payload manifest (manifest-<algorithm>.txt) found — required by RFC 8493 §2.1.3",
        ));
    } else {
        findings.push(ComplianceFinding::info(
            "Payload manifest present — RFC 8493 §2.1.3 satisfied",
        ));
    }

    // ── 3. Tag-manifest check ────────────────────────────────────────────────
    let tag_manifest_names = [
        "tagmanifest-md5.txt",
        "tagmanifest-sha1.txt",
        "tagmanifest-sha256.txt",
        "tagmanifest-sha-256.txt",
        "tagmanifest-sha512.txt",
        "tagmanifest-sha-512.txt",
    ];
    for name in &tag_manifest_names {
        if bag_path.join(name).exists() {
            tag_manifests.push((*name).to_string());
        }
    }
    if tag_manifests.is_empty() {
        findings.push(ComplianceFinding::warning(
            "No tag-manifest (tagmanifest-<algorithm>.txt) found — SHOULD be present per RFC 8493 §2.2.1",
        ));
    } else {
        findings.push(ComplianceFinding::info(format!(
            "Tag-manifest(s) found: {}",
            tag_manifests.join(", ")
        )));
    }

    // ── 4. fetch.txt presence ────────────────────────────────────────────────
    let has_fetch_txt = bag_path.join("fetch.txt").exists();
    if has_fetch_txt {
        findings.push(ComplianceFinding::info(
            "fetch.txt is present — holey bag; fetching required before validation",
        ));
    }

    // ── 5. data/ directory ───────────────────────────────────────────────────
    if !bag_path.join("data").is_dir() {
        findings.push(ComplianceFinding::error(
            "data/ payload directory is missing — required by RFC 8493 §2.1.2",
        ));
    }

    let is_compliant = !findings.iter().any(|f| f.level == ComplianceLevel::Error);

    Ok(BagitComplianceReport {
        is_compliant,
        findings,
        detected_version,
        detected_encoding,
        tag_manifests,
        has_fetch_txt,
    })
}

// ─── Incremental BagIt Update ─────────────────────────────────────────────────

/// Result of an incremental bag update operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IncrementalUpdateResult {
    /// Files added to the bag.
    pub added: Vec<PathBuf>,
    /// Files removed from the bag.
    pub removed: Vec<PathBuf>,
    /// New manifest checksum count.
    pub total_files: usize,
    /// Timestamp of the update.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Incrementally update an existing `BagIt` bag by adding or removing files
/// without re-computing checksums for unchanged files.
///
/// # How it works
/// 1. Load the existing manifest into memory.
/// 2. Remove each path in `files_to_remove` from the manifest and delete the
///    physical file from `bag.data_dir()`.
/// 3. Copy each `(source, dest_in_data)` pair in `files_to_add` into the bag,
///    compute their checksums, and insert them into the manifest.
/// 4. Rewrite the manifest and bag-info.txt (updating Payload-Oxum).
///
/// # Errors
///
/// Returns an error on any I/O failure or checksum computation failure.
pub fn incremental_update(
    bag: &BagItPackage,
    files_to_add: &[(&std::path::Path, &std::path::Path)], // (source, dest_relative_in_data)
    files_to_remove: &[&std::path::Path],                  // dest paths relative to bag root
) -> Result<IncrementalUpdateResult> {
    // Load existing manifest
    let mut manifest: HashMap<PathBuf, String> = bag.read_manifest()?;

    let mut added = Vec::new();
    let mut removed = Vec::new();

    // ── Remove files ─────────────────────────────────────────────────────────
    for rel_path in files_to_remove {
        let full_path = bag.root.join(rel_path);
        if full_path.exists() {
            fs::remove_file(&full_path)?;
        }
        manifest.remove(*rel_path);
        removed.push((*rel_path).to_path_buf());
    }

    // ── Add files ─────────────────────────────────────────────────────────────
    let generator = ChecksumGenerator::new().with_algorithms(vec![bag.algorithm]);
    for (source, dest_rel) in files_to_add {
        let dest_in_data = PathBuf::from("data").join(dest_rel);
        let full_dest = bag.root.join(&dest_in_data);
        if let Some(parent) = full_dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source, &full_dest)?;

        // Compute checksum for the newly added file
        let cs = generator.generate_file(&full_dest)?;
        let hash = cs
            .checksums
            .get(&bag.algorithm)
            .cloned()
            .ok_or_else(|| Error::Metadata("checksum missing after generation".to_string()))?;
        manifest.insert(dest_in_data.clone(), hash);
        added.push(dest_in_data);
    }

    // ── Rewrite manifest ──────────────────────────────────────────────────────
    let manifest_name = format!("manifest-{}.txt", bag.algorithm.name().to_lowercase());
    let manifest_path = bag.root.join(manifest_name);
    let mut manifest_file = File::create(manifest_path)?;
    let total_files = manifest.len();
    let mut total_bytes: u64 = 0;
    for (path, hash) in &manifest {
        let full = bag.root.join(path);
        if full.exists() {
            total_bytes += fs::metadata(&full).map(|m| m.len()).unwrap_or(0);
        }
        writeln!(manifest_file, "{hash}  {}", path.display())?;
    }

    // ── Update bag-info.txt (Payload-Oxum) ────────────────────────────────────
    let bag_info_path = bag.root.join("bag-info.txt");
    let mut meta = bag.metadata.clone();
    meta.insert(
        "Payload-Oxum".to_string(),
        format!("{}.{}", total_bytes, total_files),
    );
    meta.insert(
        "Bag-Software-Agent".to_string(),
        "oximedia-archive-pro (incremental update)".to_string(),
    );
    let mut info_file = File::create(bag_info_path)?;
    for (key, value) in &meta {
        writeln!(info_file, "{key}: {value}")?;
    }

    Ok(IncrementalUpdateResult {
        added,
        removed,
        total_files,
        timestamp: chrono::Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_create_bagit_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("test-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Test content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let bag = BagItBuilder::new(bag_dir.clone())
            .with_metadata("Contact-Name", "Test User")
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        assert!(bag.root.join("bagit.txt").exists());
        assert!(bag.root.join("bag-info.txt").exists());
        assert!(bag.root.join("data").exists());
    }

    #[test]
    fn test_validate_bagit_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("test-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Validation test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let bag = BagItBuilder::new(bag_dir.clone())
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let result = BagItValidator::validate(&bag);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_bagit_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("test-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Load test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        BagItBuilder::new(bag_dir.clone())
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let loaded = BagItPackage::load(&bag_dir).expect("operation should succeed");
        assert_eq!(loaded.algorithm, ChecksumAlgorithm::Sha256);
    }

    #[test]
    fn test_bagit_v1_compliance_valid_bag() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("compliant-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Compliance test content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        // Build a standard bag (BagIt-Version: 1.0, UTF-8 declared)
        BagItBuilder::new(bag_dir.clone())
            .with_metadata("Contact-Name", "Compliance Tester")
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let report =
            verify_bagit_v1_compliance(&bag_dir).expect("compliance check should not I/O-fail");
        // A freshly built bag has bagit.txt with v1.0, payload manifest, and data/
        // (no tag-manifest, so we get one Warning, but no Errors)
        assert!(
            report.is_compliant,
            "bag should be compliant; findings: {:?}",
            report.findings
        );
        assert_eq!(report.detected_version.as_deref(), Some("1.0"));
        assert_eq!(report.detected_encoding.as_deref(), Some("UTF-8"));
        assert!(!report.has_fetch_txt);
    }

    #[test]
    fn test_bagit_v1_compliance_missing_bagit_txt() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("broken-bag");
        fs::create_dir_all(&bag_dir).expect("operation should succeed");
        fs::create_dir_all(bag_dir.join("data")).expect("operation should succeed");
        // No bagit.txt, no manifest

        let report = verify_bagit_v1_compliance(&bag_dir).expect("should return report");
        assert!(
            !report.is_compliant,
            "bag without bagit.txt must not be compliant"
        );
        let has_error = report
            .findings
            .iter()
            .any(|f| f.level == ComplianceLevel::Error && f.message.contains("bagit.txt"));
        assert!(has_error, "should report bagit.txt error");
    }

    #[test]
    fn test_bagit_v1_compliance_with_tagmanifest() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("tagged-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Tagged bag content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        BagItBuilder::new(bag_dir.clone())
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        // Manually write a tagmanifest
        let mut tman =
            File::create(bag_dir.join("tagmanifest-sha256.txt")).expect("operation should succeed");
        writeln!(tman, "deadbeef  bagit.txt").expect("write should succeed");

        let report = verify_bagit_v1_compliance(&bag_dir).expect("should return report");
        assert_eq!(report.tag_manifests, vec!["tagmanifest-sha256.txt"]);
        // No Warning about missing tagmanifest
        let has_tag_warning = report
            .findings
            .iter()
            .any(|f| f.level == ComplianceLevel::Warning && f.message.contains("tag-manifest"));
        assert!(
            !has_tag_warning,
            "should not warn about tag-manifest when present"
        );
    }

    #[test]
    fn test_bagit_v1_compliance_fetch_txt_noted() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("holey-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Holey bag")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        BagItBuilder::new(bag_dir.clone())
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        // Write fetch.txt
        fs::write(
            bag_dir.join("fetch.txt"),
            "https://example.com/file.mkv 1024 data/file.mkv\n",
        )
        .expect("write should succeed");

        let report = verify_bagit_v1_compliance(&bag_dir).expect("should return report");
        assert!(report.has_fetch_txt, "fetch.txt should be detected");
        let fetch_info = report
            .findings
            .iter()
            .any(|f| f.level == ComplianceLevel::Info && f.message.contains("fetch.txt"));
        assert!(fetch_info, "fetch.txt info finding expected");
    }

    #[test]
    fn test_parallel_manifest_generation() {
        // Build a bag with multiple files and verify all checksums are correct
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("parallel-bag");

        let mut builder = BagItBuilder::new(bag_dir.clone());
        // Create 5 temp files
        let mut temp_files = Vec::new();
        for i in 0..5_usize {
            let mut f = NamedTempFile::new().expect("operation should succeed");
            f.write_all(format!("content of file {i}").as_bytes())
                .expect("write should succeed");
            f.flush().expect("flush should succeed");
            builder = builder.add_file(f.path()).expect("add_file should succeed");
            temp_files.push(f);
        }

        let bag = builder.build().expect("build should succeed");
        // Validate that checksums are all correct
        BagItValidator::validate(&bag).expect("validation should pass after parallel manifest gen");
    }

    // ── Incremental update tests ───────────────────────────────────────────────

    #[test]
    fn test_incremental_add_file() {
        let temp_dir = TempDir::new().expect("temp dir");
        let bag_dir = temp_dir.path().join("inc-bag");

        let mut orig = NamedTempFile::new().expect("temp file");
        orig.write_all(b"original content").expect("write");
        orig.flush().expect("flush");

        let bag = BagItBuilder::new(bag_dir.clone())
            .add_file(orig.path())
            .expect("add_file")
            .build()
            .expect("build");

        // Add a new file incrementally
        let mut new_file = NamedTempFile::new().expect("new temp file");
        new_file
            .write_all(b"new incremental content")
            .expect("write");
        new_file.flush().expect("flush");

        let result = incremental_update(
            &bag,
            &[(new_file.path(), std::path::Path::new("added.txt"))],
            &[],
        )
        .expect("incremental_update should succeed");

        assert_eq!(result.added.len(), 1);
        assert_eq!(result.removed.len(), 0);
        assert_eq!(result.total_files, 2);
        assert!(bag_dir.join("data/added.txt").exists());
    }

    #[test]
    fn test_incremental_remove_file() {
        let temp_dir = TempDir::new().expect("temp dir");
        let bag_dir = temp_dir.path().join("rm-bag");

        let mut f1 = NamedTempFile::new().expect("temp file");
        f1.write_all(b"keep this").expect("write");
        f1.flush().expect("flush");
        let f1_name = f1.path().file_name().expect("filename").to_owned();

        let mut f2 = NamedTempFile::new().expect("temp file");
        f2.write_all(b"remove this").expect("write");
        f2.flush().expect("flush");
        let f2_name = f2.path().file_name().expect("filename").to_owned();

        let bag = BagItBuilder::new(bag_dir.clone())
            .add_file(f1.path())
            .expect("add f1")
            .add_file(f2.path())
            .expect("add f2")
            .build()
            .expect("build");

        // Remove f2 from the bag
        let remove_path = PathBuf::from("data").join(&f2_name);
        let result = incremental_update(&bag, &[], &[remove_path.as_path()])
            .expect("incremental_update should succeed");

        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.total_files, 1);
        assert!(!bag_dir.join("data").join(&f2_name).exists());
        assert!(bag_dir.join("data").join(&f1_name).exists());
    }

    #[test]
    fn test_incremental_add_and_remove() {
        let temp_dir = TempDir::new().expect("temp dir");
        let bag_dir = temp_dir.path().join("mixed-bag");

        let mut existing = NamedTempFile::new().expect("temp file");
        existing.write_all(b"existing").expect("write");
        existing.flush().expect("flush");
        let existing_name = existing.path().file_name().expect("name").to_owned();

        let bag = BagItBuilder::new(bag_dir.clone())
            .add_file(existing.path())
            .expect("add")
            .build()
            .expect("build");

        let mut new_f = NamedTempFile::new().expect("new tmp");
        new_f.write_all(b"brand new").expect("write");
        new_f.flush().expect("flush");

        let remove_path = PathBuf::from("data").join(&existing_name);
        let result = incremental_update(
            &bag,
            &[(new_f.path(), std::path::Path::new("fresh.txt"))],
            &[remove_path.as_path()],
        )
        .expect("incremental_update");

        assert_eq!(result.added.len(), 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.total_files, 1);
        assert!(bag_dir.join("data/fresh.txt").exists());
        assert!(!bag_dir.join("data").join(&existing_name).exists());
    }

    #[test]
    fn test_bagit_conformance_suite_simulation() {
        // Simulates Library of Congress BagIt conformance check:
        // - bagit.txt with BagIt-Version: 1.0 and encoding
        // - payload manifest
        // - data/ directory with at least one file
        // - bag-info.txt
        let temp_dir = TempDir::new().expect("temp dir");
        let bag_dir = temp_dir.path().join("loc-conformance");

        let mut content = NamedTempFile::new().expect("tmp");
        content
            .write_all(b"LoC BagIt conformance test data")
            .expect("write");
        content.flush().expect("flush");

        let bag = BagItBuilder::new(bag_dir.clone())
            .with_metadata("Organization-Address", "Library of Congress, Washington DC")
            .with_metadata("Contact-Email", "ndsa@loc.gov")
            .with_metadata("Source-Organization", "OxiMedia Test Suite")
            .add_file(content.path())
            .expect("add_file")
            .build()
            .expect("build");

        // RFC 8493 §2.1.1 bagit.txt must exist
        assert!(bag.root.join("bagit.txt").exists(), "bagit.txt must exist");
        // data/ directory
        assert!(bag.root.join("data").is_dir(), "data/ must be a directory");
        // payload manifest
        let manifest = bag.root.join("manifest-sha-256.txt");
        let manifest_alt = bag.root.join("manifest-sha256.txt");
        assert!(
            manifest.exists() || manifest_alt.exists(),
            "payload manifest must exist"
        );
        // bag-info.txt
        assert!(
            bag.root.join("bag-info.txt").exists(),
            "bag-info.txt must exist"
        );

        // Full compliance check
        let report = verify_bagit_v1_compliance(&bag_dir).expect("compliance check should succeed");
        assert!(
            report.is_compliant,
            "conformance suite bag must be compliant"
        );
        assert_eq!(
            report.detected_version.as_deref(),
            Some("1.0"),
            "BagIt-Version must be 1.0"
        );

        // Validation (checksum check)
        BagItValidator::validate(&bag).expect("BagIt validation must pass");
    }
}
