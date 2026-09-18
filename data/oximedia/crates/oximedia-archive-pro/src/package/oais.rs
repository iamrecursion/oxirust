//! OAIS (Open Archival Information System) package format
//!
//! OAIS defines three package types:
//! - SIP (Submission Information Package): For submission to archive
//! - AIP (Archival Information Package): For long-term preservation
//! - DIP (Dissemination Information Package): For access/distribution

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// OAIS package types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OaisPackageType {
    /// Submission Information Package
    Sip,
    /// Archival Information Package
    Aip,
    /// Dissemination Information Package
    Dip,
}

impl OaisPackageType {
    /// Returns the package type name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sip => "SIP",
            Self::Aip => "AIP",
            Self::Dip => "DIP",
        }
    }

    /// Returns a description of the package type
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Sip => "Submission Information Package - for archive submission",
            Self::Aip => "Archival Information Package - for long-term preservation",
            Self::Dip => "Dissemination Information Package - for access and distribution",
        }
    }
}

/// OAIS package structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaisPackage {
    /// Package root directory
    pub root: PathBuf,
    /// Package type
    pub package_type: OaisPackageType,
    /// Package identifier
    pub id: String,
    /// Creation timestamp
    pub created: chrono::DateTime<chrono::Utc>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl OaisPackage {
    /// Load an existing OAIS package
    ///
    /// # Errors
    ///
    /// Returns an error if the package is invalid
    pub fn load(root: &Path) -> Result<Self> {
        let manifest_path = root.join("OAIS-MANIFEST.json");
        if !manifest_path.exists() {
            return Err(Error::InvalidOais("Missing OAIS-MANIFEST.json".to_string()));
        }

        let file = File::open(manifest_path)?;
        let package: Self = serde_json::from_reader(file)
            .map_err(|e| Error::InvalidOais(format!("Invalid manifest: {e}")))?;

        Ok(package)
    }

    /// Get the content directory
    #[must_use]
    pub fn content_dir(&self) -> PathBuf {
        self.root.join("content")
    }

    /// Get the metadata directory
    #[must_use]
    pub fn metadata_dir(&self) -> PathBuf {
        self.root.join("metadata")
    }

    /// Get the submission documentation directory (SIP only)
    #[must_use]
    pub fn submission_dir(&self) -> PathBuf {
        self.root.join("submission")
    }

    /// Get the preservation metadata directory (AIP only)
    #[must_use]
    pub fn preservation_dir(&self) -> PathBuf {
        self.root.join("preservation")
    }
}

/// OAIS package builder
pub struct OaisBuilder {
    root: PathBuf,
    package_type: OaisPackageType,
    id: String,
    algorithm: ChecksumAlgorithm,
    metadata: HashMap<String, String>,
    content_files: Vec<(PathBuf, PathBuf)>,
    metadata_files: Vec<(PathBuf, PathBuf)>,
}

impl OaisBuilder {
    /// Create a new OAIS builder
    #[must_use]
    pub fn new(root: PathBuf, package_type: OaisPackageType, id: String) -> Self {
        Self {
            root,
            package_type,
            id,
            algorithm: ChecksumAlgorithm::Sha256,
            metadata: HashMap::new(),
            content_files: Vec::new(),
            metadata_files: Vec::new(),
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add package metadata
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a content file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_content_file(mut self, source: &Path, dest: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }
        self.content_files
            .push((source.to_path_buf(), dest.to_path_buf()));
        Ok(self)
    }

    /// Add a metadata file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_metadata_file(mut self, source: &Path, dest: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }
        self.metadata_files
            .push((source.to_path_buf(), dest.to_path_buf()));
        Ok(self)
    }

    /// Build the OAIS package
    ///
    /// # Errors
    ///
    /// Returns an error if the package cannot be created
    pub fn build(self) -> Result<OaisPackage> {
        // Create directory structure
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(self.root.join("content"))?;
        fs::create_dir_all(self.root.join("metadata"))?;

        match self.package_type {
            OaisPackageType::Sip => {
                fs::create_dir_all(self.root.join("submission"))?;
            }
            OaisPackageType::Aip => {
                fs::create_dir_all(self.root.join("preservation"))?;
            }
            OaisPackageType::Dip => {
                // DIP may have additional access-specific directories
            }
        }

        // Copy content files
        let generator = ChecksumGenerator::new().with_algorithms(vec![self.algorithm]);
        let mut content_checksums = HashMap::new();

        for (source, dest) in &self.content_files {
            let target = self.root.join("content").join(dest);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &target)?;

            let checksum = generator.generate_file(&target)?;
            if let Some(hash) = checksum.checksums.get(&self.algorithm) {
                content_checksums.insert(dest.display().to_string(), hash.clone());
            }
        }

        // Copy metadata files
        for (source, dest) in &self.metadata_files {
            let target = self.root.join("metadata").join(dest);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &target)?;
        }

        // Write checksums file
        let checksums_path = self.root.join("CHECKSUMS.txt");
        let mut checksums_file = File::create(checksums_path)?;
        for (path, hash) in &content_checksums {
            writeln!(checksums_file, "{hash}  content/{path}")?;
        }

        // Create package manifest
        let package = OaisPackage {
            root: self.root.clone(),
            package_type: self.package_type,
            id: self.id,
            created: chrono::Utc::now(),
            metadata: self.metadata,
        };

        let manifest_path = self.root.join("OAIS-MANIFEST.json");
        let manifest_file = File::create(manifest_path)?;
        serde_json::to_writer_pretty(manifest_file, &package)
            .map_err(|e| Error::InvalidOais(format!("Failed to write manifest: {e}")))?;

        Ok(package)
    }
}

/// Configuration for DIP generation from an AIP.
#[derive(Debug, Clone)]
pub struct DipGenerationConfig {
    /// Whether to include preservation metadata in the DIP.
    pub include_preservation_metadata: bool,
    /// Whether to include original submission metadata.
    pub include_submission_metadata: bool,
    /// Optional format filter: only include content files matching these extensions.
    pub format_filter: Option<Vec<String>>,
    /// Optional maximum number of content files to include.
    pub max_files: Option<usize>,
    /// Custom metadata to add to the DIP.
    pub extra_metadata: HashMap<String, String>,
}

impl Default for DipGenerationConfig {
    fn default() -> Self {
        Self {
            include_preservation_metadata: true,
            include_submission_metadata: false,
            format_filter: None,
            max_files: None,
            extra_metadata: HashMap::new(),
        }
    }
}

/// Generator that creates DIP (Dissemination Information Package) from an AIP.
///
/// The DIP is the OAIS package type designed for access and distribution.
/// It typically contains a subset of the AIP content, stripped of
/// preservation-specific data, and packaged for end-user consumption.
pub struct DipGenerator;

impl DipGenerator {
    /// Generate a DIP from an existing AIP.
    ///
    /// This reads the AIP's content and metadata, then produces a new DIP package
    /// at the specified output path. The DIP contains the content files from the AIP,
    /// optionally filtered by the configuration, along with a CHECKSUMS manifest and
    /// an OAIS-MANIFEST.json describing the DIP.
    ///
    /// # Arguments
    ///
    /// * `aip` - Reference to the source AIP package (must have `package_type == Aip`)
    /// * `output_root` - Directory path where the DIP will be created
    /// * `dip_id` - Unique identifier for the new DIP
    /// * `config` - Configuration controlling what gets included
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source package is not an AIP
    /// - Content files cannot be read or copied
    /// - The output directory cannot be created
    pub fn generate(
        aip: &OaisPackage,
        output_root: &Path,
        dip_id: &str,
        config: &DipGenerationConfig,
    ) -> Result<OaisPackage> {
        // Validate source is an AIP
        if aip.package_type != OaisPackageType::Aip {
            return Err(Error::InvalidOais(format!(
                "Source package must be AIP, got {}",
                aip.package_type.name()
            )));
        }

        // Create DIP directory structure
        fs::create_dir_all(output_root)?;
        let dip_content_dir = output_root.join("content");
        let dip_metadata_dir = output_root.join("metadata");
        fs::create_dir_all(&dip_content_dir)?;
        fs::create_dir_all(&dip_metadata_dir)?;

        // Enumerate AIP content files
        let aip_content_dir = aip.content_dir();
        let mut content_files: Vec<PathBuf> = Vec::new();

        if aip_content_dir.exists() {
            for entry in walkdir::WalkDir::new(&aip_content_dir)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.into_path();

                // Apply format filter
                if let Some(ref exts) = config.format_filter {
                    let file_ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if !exts.iter().any(|e| e.eq_ignore_ascii_case(file_ext)) {
                        continue;
                    }
                }

                content_files.push(path);
            }
        }

        // Apply max_files limit
        if let Some(max) = config.max_files {
            content_files.truncate(max);
        }

        // Copy content files and generate checksums
        let generator = ChecksumGenerator::new().with_algorithms(vec![ChecksumAlgorithm::Sha256]);
        let mut content_checksums: HashMap<String, String> = HashMap::new();

        for source_path in &content_files {
            let relative = source_path
                .strip_prefix(&aip_content_dir)
                .unwrap_or(source_path);
            let target = dip_content_dir.join(relative);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source_path, &target)?;

            let checksum = generator.generate_file(&target)?;
            if let Some(hash) = checksum.checksums.get(&ChecksumAlgorithm::Sha256) {
                content_checksums.insert(relative.display().to_string(), hash.clone());
            }
        }

        // Copy metadata files if configured
        let aip_metadata_dir = aip.metadata_dir();
        if config.include_preservation_metadata && aip_metadata_dir.exists() {
            for entry in walkdir::WalkDir::new(&aip_metadata_dir)
                .follow_links(false)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
            {
                let source = entry.into_path();
                let relative = source.strip_prefix(&aip_metadata_dir).unwrap_or(&source);
                let target = dip_metadata_dir.join(relative);
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&source, &target)?;
            }
        }

        // Copy submission metadata if available and configured
        if config.include_submission_metadata {
            let aip_submission_dir = aip.submission_dir();
            if aip_submission_dir.exists() {
                let dip_submission_dir = output_root.join("submission");
                fs::create_dir_all(&dip_submission_dir)?;
                for entry in walkdir::WalkDir::new(&aip_submission_dir)
                    .follow_links(false)
                    .into_iter()
                    .filter_map(std::result::Result::ok)
                    .filter(|e| e.file_type().is_file())
                {
                    let source = entry.into_path();
                    let relative = source.strip_prefix(&aip_submission_dir).unwrap_or(&source);
                    let target = dip_submission_dir.join(relative);
                    if let Some(parent) = target.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::copy(&source, &target)?;
                }
            }
        }

        // Write CHECKSUMS.txt
        let checksums_path = output_root.join("CHECKSUMS.txt");
        let mut checksums_file = File::create(checksums_path)?;
        for (path, hash) in &content_checksums {
            writeln!(checksums_file, "{hash}  content/{path}")?;
        }

        // Build DIP metadata
        let mut dip_metadata = aip.metadata.clone();
        dip_metadata.insert("Source-AIP-ID".to_string(), aip.id.clone());
        dip_metadata.insert(
            "DIP-Generation-Date".to_string(),
            chrono::Utc::now().to_rfc3339(),
        );
        dip_metadata.insert(
            "Content-File-Count".to_string(),
            content_files.len().to_string(),
        );
        for (k, v) in &config.extra_metadata {
            dip_metadata.insert(k.clone(), v.clone());
        }

        let dip_package = OaisPackage {
            root: output_root.to_path_buf(),
            package_type: OaisPackageType::Dip,
            id: dip_id.to_string(),
            created: chrono::Utc::now(),
            metadata: dip_metadata,
        };

        // Write OAIS-MANIFEST.json
        let manifest_path = output_root.join("OAIS-MANIFEST.json");
        let manifest_file = File::create(manifest_path)?;
        serde_json::to_writer_pretty(manifest_file, &dip_package)
            .map_err(|e| Error::InvalidOais(format!("Failed to write DIP manifest: {e}")))?;

        Ok(dip_package)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_oais_package_types() {
        assert_eq!(OaisPackageType::Sip.name(), "SIP");
        assert_eq!(OaisPackageType::Aip.name(), "AIP");
        assert_eq!(OaisPackageType::Dip.name(), "DIP");
    }

    #[test]
    fn test_create_sip_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("test-sip");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"SIP content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let package =
            OaisBuilder::new(pkg_dir.clone(), OaisPackageType::Sip, "SIP-001".to_string())
                .with_metadata("Creator", "Test System")
                .add_content_file(test_file.path(), Path::new("video.mkv"))
                .expect("operation should succeed")
                .build()
                .expect("operation should succeed");

        assert_eq!(package.package_type, OaisPackageType::Sip);
        assert_eq!(package.id, "SIP-001");
        assert!(package.root.join("content").exists());
        assert!(package.root.join("metadata").exists());
        assert!(package.root.join("submission").exists());
        assert!(package.root.join("OAIS-MANIFEST.json").exists());
    }

    #[test]
    fn test_create_aip_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("test-aip");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"AIP content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let package =
            OaisBuilder::new(pkg_dir.clone(), OaisPackageType::Aip, "AIP-001".to_string())
                .add_content_file(test_file.path(), Path::new("preservation.mkv"))
                .expect("operation should succeed")
                .build()
                .expect("operation should succeed");

        assert_eq!(package.package_type, OaisPackageType::Aip);
        assert!(package.root.join("preservation").exists());
    }

    #[test]
    fn test_load_oais_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("test-load");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Load test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        OaisBuilder::new(pkg_dir.clone(), OaisPackageType::Dip, "DIP-001".to_string())
            .add_content_file(test_file.path(), Path::new("access.mp4"))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let loaded = OaisPackage::load(&pkg_dir).expect("operation should succeed");
        assert_eq!(loaded.package_type, OaisPackageType::Dip);
        assert_eq!(loaded.id, "DIP-001");
    }

    // ── DIP generation tests ────────────────────────────────────

    fn create_test_aip(temp_dir: &TempDir) -> OaisPackage {
        let pkg_dir = temp_dir.path().join("source-aip");

        let mut video_file = NamedTempFile::new().expect("operation should succeed");
        video_file
            .write_all(b"video content bytes here")
            .expect("operation should succeed");
        video_file.flush().expect("operation should succeed");

        let mut audio_file = NamedTempFile::new().expect("operation should succeed");
        audio_file
            .write_all(b"audio content bytes here")
            .expect("operation should succeed");
        audio_file.flush().expect("operation should succeed");

        OaisBuilder::new(pkg_dir, OaisPackageType::Aip, "AIP-TEST-001".to_string())
            .with_metadata("Creator", "Test System")
            .with_metadata("Description", "Test AIP for DIP generation")
            .add_content_file(video_file.path(), Path::new("video.mkv"))
            .expect("operation should succeed")
            .add_content_file(audio_file.path(), Path::new("audio.flac"))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed")
    }

    #[test]
    fn test_dip_generation_basic() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let aip = create_test_aip(&temp_dir);
        let dip_dir = temp_dir.path().join("output-dip");

        let config = DipGenerationConfig::default();
        let dip = DipGenerator::generate(&aip, &dip_dir, "DIP-001", &config)
            .expect("operation should succeed");

        assert_eq!(dip.package_type, OaisPackageType::Dip);
        assert_eq!(dip.id, "DIP-001");
        assert!(dip.root.join("content").exists());
        assert!(dip.root.join("metadata").exists());
        assert!(dip.root.join("OAIS-MANIFEST.json").exists());
        assert!(dip.root.join("CHECKSUMS.txt").exists());
    }

    #[test]
    fn test_dip_metadata_references_source_aip() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let aip = create_test_aip(&temp_dir);
        let dip_dir = temp_dir.path().join("dip-meta-test");

        let config = DipGenerationConfig::default();
        let dip = DipGenerator::generate(&aip, &dip_dir, "DIP-002", &config)
            .expect("operation should succeed");

        assert_eq!(
            dip.metadata.get("Source-AIP-ID").map(String::as_str),
            Some("AIP-TEST-001")
        );
        assert!(dip.metadata.contains_key("DIP-Generation-Date"));
        assert!(dip.metadata.contains_key("Content-File-Count"));
    }

    #[test]
    fn test_dip_generation_with_format_filter() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let aip = create_test_aip(&temp_dir);
        let dip_dir = temp_dir.path().join("dip-filter-test");

        let config = DipGenerationConfig {
            format_filter: Some(vec!["mkv".to_string()]),
            ..Default::default()
        };
        let dip = DipGenerator::generate(&aip, &dip_dir, "DIP-FILTER", &config)
            .expect("operation should succeed");

        // Only .mkv files should be present
        assert!(dip.root.join("content/video.mkv").exists());
        assert!(!dip.root.join("content/audio.flac").exists());
    }

    #[test]
    fn test_dip_generation_with_max_files() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let aip = create_test_aip(&temp_dir);
        let dip_dir = temp_dir.path().join("dip-max-test");

        let config = DipGenerationConfig {
            max_files: Some(1),
            ..Default::default()
        };
        let dip = DipGenerator::generate(&aip, &dip_dir, "DIP-MAX", &config)
            .expect("operation should succeed");

        assert_eq!(
            dip.metadata.get("Content-File-Count").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn test_dip_generation_with_extra_metadata() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let aip = create_test_aip(&temp_dir);
        let dip_dir = temp_dir.path().join("dip-extra-meta");

        let mut extra = HashMap::new();
        extra.insert("Access-Level".to_string(), "Public".to_string());
        extra.insert(
            "Requested-By".to_string(),
            "researcher@example.org".to_string(),
        );
        let config = DipGenerationConfig {
            extra_metadata: extra,
            ..Default::default()
        };
        let dip = DipGenerator::generate(&aip, &dip_dir, "DIP-EXTRA", &config)
            .expect("operation should succeed");

        assert_eq!(
            dip.metadata.get("Access-Level").map(String::as_str),
            Some("Public")
        );
        assert_eq!(
            dip.metadata.get("Requested-By").map(String::as_str),
            Some("researcher@example.org")
        );
    }

    #[test]
    fn test_dip_generation_rejects_non_aip() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("sip-source");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"SIP content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let sip = OaisBuilder::new(pkg_dir, OaisPackageType::Sip, "SIP-001".to_string())
            .add_content_file(test_file.path(), Path::new("file.mkv"))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let dip_dir = temp_dir.path().join("dip-from-sip");
        let config = DipGenerationConfig::default();
        let result = DipGenerator::generate(&sip, &dip_dir, "DIP-ERR", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_dip_can_be_loaded_back() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let aip = create_test_aip(&temp_dir);
        let dip_dir = temp_dir.path().join("dip-roundtrip");

        let config = DipGenerationConfig::default();
        DipGenerator::generate(&aip, &dip_dir, "DIP-RT", &config)
            .expect("operation should succeed");

        let loaded = OaisPackage::load(&dip_dir).expect("operation should succeed");
        assert_eq!(loaded.package_type, OaisPackageType::Dip);
        assert_eq!(loaded.id, "DIP-RT");
    }
}
