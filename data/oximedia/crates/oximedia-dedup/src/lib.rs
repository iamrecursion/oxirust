//! Media deduplication and duplicate detection for `OxiMedia`.
//!
//! `oximedia-dedup` provides comprehensive duplicate detection and media deduplication
//! for the `OxiMedia` multimedia framework. This includes:
//!
//! - **Cryptographic hashing**: BLAKE3-based exact duplicate detection
//! - **Visual similarity**: Perceptual hashing, SSIM, histogram, and feature matching
//! - **Audio fingerprinting**: Audio fingerprint comparison and waveform similarity
//! - **Metadata matching**: Fuzzy metadata comparison for near-duplicates
//! - **Storage optimization**: Fast SQLite-based indexing for large libraries
//! - **Reporting**: Comprehensive duplicate reports with similarity scoring
//!
//! # Modules
//!
//! - [`hash`]: Cryptographic and content-based hashing
//! - [`visual`]: Visual similarity detection
//! - [`audio`]: Audio fingerprint comparison
//! - [`metadata`]: Metadata-based deduplication
//! - `database`: SQLite-based indexing and lookup
//! - [`report`]: Duplicate detection reports
//!
//! # Example
//!
//! The full indexing pipeline below uses `DuplicateDetector`, which is only
//! available with the `sqlite` feature enabled (it backs the persistent index
//! with SQLite). The example is therefore marked `ignore` in the default build.
//!
//! ```ignore
//! use oximedia_dedup::{DuplicateDetector, DetectionStrategy, DedupConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = DedupConfig::default();
//! let mut detector = DuplicateDetector::new(config).await?;
//!
//! // Add files to the index
//! detector.add_file("/path/to/video1.mp4").await?;
//! detector.add_file("/path/to/video2.mp4").await?;
//!
//! // Find duplicates
//! let duplicates = detector.find_duplicates(DetectionStrategy::All).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Strategy Selection Guide
//!
//! | Strategy | Speed | Precision | Use case |
//! |---|---|---|---|
//! | `ExactHash` | Very fast | Perfect (no false positives) | Bit-for-bit identical files — best first pass for any library |
//! | `Fast` | Fast | High | Quick scan: hash + perceptual + metadata; good default for large libraries |
//! | `PerceptualHash` | Fast | Good | Visually identical images/frames that were re-encoded or lightly cropped |
//! | `Histogram` | Fast | Moderate | Color-similar frames regardless of spatial layout |
//! | `AudioFingerprint` | Moderate | High | Same audio in different codecs or with minor edits |
//! | `Metadata` | Fast | Low–moderate | Likely duplicates with same duration/resolution (combine with visual pass) |
//! | `Ssim` | Slow | Very high | Near-identical video frames where pHash gives too many false positives |
//! | `FeatureMatch` | Slow | High | Cropped, rotated, or partially occluded duplicates |
//! | `VisualAll` | Slow | Very high | Combined visual pipeline: pHash + Histogram + SSIM + FeatureMatch |
//! | `All` | Very slow | Maximum | Full pipeline — all methods; use only for final authoritative scan |
//!
//! **Recommended workflow:**
//! 1. Run `ExactHash` first to catch perfect duplicates cheaply.
//! 2. Run `Fast` for a broad near-duplicate sweep.
//! 3. Run `VisualAll` or `All` for a precision clean-up pass on the remainder.
//!
//! ## Detection Method Trade-offs
//!
//! | Method | Accuracy | CPU cost | Memory | False-positive risk | Notes |
//! |---|---|---|---|---|---|
//! | BLAKE3 hash | 100% | Very low | O(1) | None | Misses re-encoded or edited copies |
//! | dHash (8×8) | High | Very low | O(1) | Low | Robust to resize; sensitive to crops |
//! | pHash (DCT) | High | Low | O(1) | Low–medium | Better than dHash for brightness shifts |
//! | wHash (wavelet) | High | Low | O(1) | Low | Most robust to combined transforms |
//! | SSIM | Very high | High | O(WH) | Very low | Pixel-accurate; slow for large images |
//! | Histogram | Moderate | Low | O(256) | Medium | Colour match only; ignores structure |
//! | FeatureMatch | High | Very high | O(N×D) | Low | Works on crops/rotations; expensive |
//! | AudioFingerprint | High | Moderate | O(T) | Low | Spectral-peak based; codec-agnostic |
//! | Metadata | Low–moderate | Very low | O(1) | High | Use only as a pre-filter |
//!
//! **Bloom-filter pre-screening** (`DedupConfig::bloom_prescreen = true`) reduces
//! the number of pairwise comparisons by rejecting definitely-unique items before
//! the expensive perceptual-hash phase.  Recommended for libraries with > 10 K files.
//!
//! **LSH acceleration** (`DedupConfig::use_lsh = true`, default) replaces O(n²)
//! pairwise perceptual-hash comparison with sub-quadratic approximate nearest-
//! neighbour lookup via `BitLshIndex`.  Adjust `lsh_num_tables` (more tables →
//! better recall, more memory) and `lsh_bits_per_table` (fewer bits → more
//! candidates → better recall at higher CPU cost).

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

pub mod audio;
pub mod audio_fingerprint;
pub mod bloom_filter;
pub mod bloom_prescreen;
pub mod chromagram;
pub mod cluster;
pub mod content_id;
pub mod content_signature;
pub mod cross_format;
#[cfg(feature = "sqlite")]
pub mod database;
pub mod dedup_cache;
pub mod dedup_index;
pub mod dedup_policy;
pub mod dedup_queue;
pub mod dedup_report;
pub mod dedup_report_detailed;
pub mod dedup_report_ext;
pub mod dedup_stats;
pub mod exact_match;
pub mod frame_hash;
pub mod fuzzy_match;
pub mod hash;
pub mod hash_store;
pub mod hierarchical;
pub mod incremental;
pub mod lsh_index;
pub mod merge_strategy;
pub mod metadata;
pub mod minhash;
pub mod near_duplicate;
pub mod near_duplicate_cluster;
pub mod network_dedup;
pub mod parallel_indexer;
pub mod perceptual_hash;
pub mod persistent_cache;
pub mod phash;
pub mod progress;
pub mod quality;
pub mod report;
pub mod rolling_hash;
pub mod segment_dedup;
pub mod signature_store;
pub mod similarity_index;
pub mod space_savings;
pub mod stream_dedup;
pub mod video_dedup;
pub mod video_dedup_pipeline;
pub mod video_segment_dedup;
pub mod visual;

#[cfg(feature = "sqlite")]
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

#[cfg(feature = "sqlite")]
pub use database::DedupDatabase;
pub use merge_strategy::{AppliedAction, MergeExecutor, MergeReport};
pub use report::{DuplicateGroup, DuplicateReport, SimilarityScore};

// ---------------------------------------------------------------------------
// Internal helpers used by the stub implementations
// ---------------------------------------------------------------------------

/// Decode a lowercase hex string into a byte vector.
///
/// # Errors
///
/// Returns `DedupError::Hash` if the string contains non-hex characters or
/// has an odd number of characters.
#[cfg(feature = "sqlite")]
fn decode_hex_bytes(hex: &str) -> DedupResult<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(DedupError::Hash(format!(
            "odd-length hex string: len={}",
            hex.len()
        )));
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| DedupError::Hash(format!("invalid hex byte at {i}: {e}")))
        })
        .collect()
}

/// Compute the cosine similarity between two f64 slices.
///
/// Returns a value in [−1, 1] or 0.0 when either vector is zero-magnitude.
#[cfg(feature = "sqlite")]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < f64::EPSILON || mag_b < f64::EPSILON {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Generic pairwise grouping helper for perceptual hash comparison.
///
/// Takes a slice of `(path, hash)` pairs, a maximum Hamming distance
/// threshold, a distance function, a similarity function (0.0‒1.0), and a
/// method label.  Returns non-overlapping duplicate groups.
#[cfg(feature = "sqlite")]
fn group_by_pairwise_similarity<H, FDist, FSim>(
    items: &[(String, H)],
    max_distance: u32,
    dist_fn: FDist,
    sim_fn: FSim,
    method: &str,
) -> DedupResult<Vec<DuplicateGroup>>
where
    FDist: Fn(&H, &H) -> u32,
    FSim: Fn(&H, &H) -> f64,
{
    let mut groups: Vec<DuplicateGroup> = Vec::new();
    let mut assigned = vec![false; items.len()];

    for i in 0..items.len() {
        if assigned[i] {
            continue;
        }
        let mut group_files = vec![items[i].0.clone()];
        let mut best_score = 0.0f64;

        for j in (i + 1)..items.len() {
            if assigned[j] {
                continue;
            }
            let dist = dist_fn(&items[i].1, &items[j].1);
            if dist <= max_distance {
                let sim = sim_fn(&items[i].1, &items[j].1);
                group_files.push(items[j].0.clone());
                assigned[j] = true;
                if sim > best_score {
                    best_score = sim;
                }
            }
        }

        if group_files.len() > 1 {
            assigned[i] = true;
            groups.push(DuplicateGroup {
                files: group_files,
                scores: vec![SimilarityScore {
                    method: method.to_string(),
                    score: best_score,
                    metadata: Vec::new(),
                }],
            });
        }
    }

    Ok(groups)
}

/// Deduplication error type.
#[derive(Error, Debug)]
pub enum DedupError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Database error
    #[cfg(feature = "sqlite")]
    #[error("Database error: {0}")]
    Database(#[from] oxisql_core::OxiSqlError),

    /// Database error (non-sqlite variant)
    #[cfg(not(feature = "sqlite"))]
    #[error("Database error: {0}")]
    Database(String),

    /// Miscellaneous error not covered by a more specific variant.
    #[error("{0}")]
    Other(String),

    /// Hashing error
    #[error("Hashing error: {0}")]
    Hash(String),

    /// Visual processing error
    #[error("Visual processing error: {0}")]
    Visual(String),

    /// Audio processing error
    #[error("Audio processing error: {0}")]
    Audio(String),

    /// Metadata processing error
    #[error("Metadata processing error: {0}")]
    Metadata(String),

    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Core library error
    #[error("OxiMedia core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

/// Deduplication result type.
pub type DedupResult<T> = Result<T, DedupError>;

/// Detection strategy for finding duplicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionStrategy {
    /// Exact duplicates only (cryptographic hash)
    ExactHash,

    /// Visual similarity using perceptual hashing
    PerceptualHash,

    /// Visual similarity using SSIM
    Ssim,

    /// Visual similarity using histogram comparison
    Histogram,

    /// Visual similarity using feature matching
    FeatureMatch,

    /// Audio fingerprint comparison
    AudioFingerprint,

    /// Metadata-based matching
    Metadata,

    /// All detection methods
    All,

    /// Combination of visual methods
    VisualAll,

    /// Combination of fast methods (hash + perceptual + metadata)
    Fast,
}

impl DetectionStrategy {
    /// Check if strategy includes exact hashing.
    #[must_use]
    pub fn includes_hash(self) -> bool {
        matches!(self, Self::ExactHash | Self::All | Self::Fast)
    }

    /// Check if strategy includes perceptual hashing.
    #[must_use]
    pub fn includes_perceptual(self) -> bool {
        matches!(
            self,
            Self::PerceptualHash | Self::All | Self::VisualAll | Self::Fast
        )
    }

    /// Check if strategy includes SSIM.
    #[must_use]
    pub fn includes_ssim(self) -> bool {
        matches!(self, Self::Ssim | Self::All | Self::VisualAll)
    }

    /// Check if strategy includes histogram.
    #[must_use]
    pub fn includes_histogram(self) -> bool {
        matches!(self, Self::Histogram | Self::All | Self::VisualAll)
    }

    /// Check if strategy includes feature matching.
    #[must_use]
    pub fn includes_feature_match(self) -> bool {
        matches!(self, Self::FeatureMatch | Self::All | Self::VisualAll)
    }

    /// Check if strategy includes audio fingerprinting.
    #[must_use]
    pub fn includes_audio(self) -> bool {
        matches!(self, Self::AudioFingerprint | Self::All)
    }

    /// Check if strategy includes metadata.
    #[must_use]
    pub fn includes_metadata(self) -> bool {
        matches!(self, Self::Metadata | Self::All | Self::Fast)
    }
}

/// Configuration for deduplication.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Database path
    pub database_path: PathBuf,

    /// Perceptual hash similarity threshold (0.0-1.0)
    pub perceptual_threshold: f64,

    /// SSIM similarity threshold (0.0-1.0)
    pub ssim_threshold: f64,

    /// Histogram similarity threshold (0.0-1.0)
    pub histogram_threshold: f64,

    /// Feature match threshold (minimum number of matches)
    pub feature_match_threshold: usize,

    /// Audio fingerprint similarity threshold (0.0-1.0)
    pub audio_threshold: f64,

    /// Metadata similarity threshold (0.0-1.0)
    pub metadata_threshold: f64,

    /// Enable parallel processing
    pub parallel: bool,

    /// Number of frames to sample for video analysis
    pub sample_frames: usize,

    /// Chunk size for content-based chunking (bytes)
    pub chunk_size: usize,

    /// Thumbnail resolution for SSIM duplicate detection.
    ///
    /// Specifies both width and height of the grayscale thumbnail used for
    /// SSIM comparison.  Must be >= 4.  Default is 8 (i.e. 8x8 = 64 pixels).
    /// Higher values give more accurate SSIM at the cost of storage and CPU.
    pub thumbnail_resolution: usize,

    /// Enable bloom filter pre-screening before expensive perceptual comparisons.
    ///
    /// When enabled, a bloom filter is used to quickly reject items whose
    /// content hash is already known to be unique, avoiding expensive
    /// pairwise perceptual hash comparisons.
    pub bloom_prescreen: bool,

    /// Expected capacity for the bloom filter pre-screener.
    pub bloom_capacity: usize,

    /// False positive rate for the bloom filter pre-screener.
    pub bloom_fpr: f32,

    /// Use LSH acceleration for perceptual hash deduplication.
    ///
    /// When enabled, `find_perceptual_duplicates()` uses a `BitLshIndex`
    /// instead of O(n^2) pairwise comparison.  This provides sub-quadratic
    /// performance for large libraries at the cost of slightly reduced recall.
    pub use_lsh: bool,

    /// Number of LSH hash tables (more = better recall, more memory).
    pub lsh_num_tables: usize,

    /// Bits sampled per LSH table (fewer = more candidates = better recall).
    pub lsh_bits_per_table: usize,

    /// Deterministic seed for LSH projections.
    pub lsh_seed: u64,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("oximedia_dedup.db"),
            perceptual_threshold: 0.95,
            ssim_threshold: 0.90,
            histogram_threshold: 0.85,
            feature_match_threshold: 50,
            audio_threshold: 0.90,
            metadata_threshold: 0.80,
            parallel: true,
            sample_frames: 10,
            chunk_size: 4096,
            thumbnail_resolution: 8,
            bloom_prescreen: false,
            bloom_capacity: 10_000,
            bloom_fpr: 0.01,
            use_lsh: true,
            lsh_num_tables: 8,
            lsh_bits_per_table: 8,
            lsh_seed: 42,
        }
    }
}

/// Main duplicate detector.
#[cfg(feature = "sqlite")]
pub struct DuplicateDetector {
    config: DedupConfig,
    database: DedupDatabase,
    /// Optional Bloom filter for fast-path duplicate pre-screening.
    ///
    /// Populated when `DedupConfig::bloom_prescreen` is `true`.  Stores
    /// raw BLAKE3 hash bytes of every indexed file so that definitely-unique
    /// files can be rejected without expensive pairwise comparisons.
    bloom: Option<bloom_filter::BloomFilter>,
}

#[cfg(feature = "sqlite")]
impl DuplicateDetector {
    /// Create a new duplicate detector.
    ///
    /// When `config.bloom_prescreen` is `true`, a `BloomFilter` is
    /// created using `config.bloom_capacity` and `config.bloom_fpr`.
    /// Every file indexed via `add_file` or `par_index_files` will
    /// automatically populate the filter so it can be used for fast-path
    /// rejection in subsequent duplicate-detection passes.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub async fn new(config: DedupConfig) -> DedupResult<Self> {
        let database = DedupDatabase::open(&config.database_path).await?;
        let bloom = if config.bloom_prescreen {
            Some(bloom_filter::BloomFilter::new(
                config.bloom_capacity,
                config.bloom_fpr,
            ))
        } else {
            None
        };
        Ok(Self {
            config,
            database,
            bloom,
        })
    }

    /// Add a file to the deduplication index.
    ///
    /// If bloom pre-screening is enabled, the file's BLAKE3 hash bytes are
    /// also inserted into the in-memory Bloom filter so that future
    /// `might_be_duplicate` calls can provide fast-path rejection.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or processed.
    pub async fn add_file(&mut self, path: impl AsRef<Path>) -> DedupResult<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(DedupError::FileNotFound(path.to_path_buf()));
        }

        // Compute hash
        let file_hash = hash::compute_file_hash(path)?;

        // Populate bloom filter (fast-path pre-screener) if enabled.
        if let Some(ref mut bloom) = self.bloom {
            bloom.insert(file_hash.as_bytes());
        }

        // Store in database
        self.database.insert_file(path, &file_hash.to_hex()).await?;

        Ok(())
    }

    /// Add multiple files sequentially.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be processed.
    pub async fn add_files(&mut self, paths: &[impl AsRef<Path>]) -> DedupResult<Vec<String>> {
        let mut errors = Vec::new();

        for path in paths {
            if let Err(e) = self.add_file(path).await {
                errors.push(format!("{}: {}", path.as_ref().display(), e));
            }
        }

        Ok(errors)
    }

    /// Add multiple files to the index using parallel hash computation.
    ///
    /// This method computes file hashes (BLAKE3) in parallel using rayon, then
    /// merges the results into the database sequentially.  The parallelism
    /// benefit is greatest for large libraries where hash I/O and computation
    /// dominate.  Database inserts are performed sequentially afterwards
    /// because they require exclusive `&mut self` access.
    ///
    /// Errors from individual files are collected and returned rather than
    /// aborting the entire batch.
    ///
    /// # Errors
    ///
    /// Returns the list of per-file error strings.  An empty `Vec` means all
    /// files were indexed successfully.
    pub async fn par_index_files<P>(&mut self, paths: &[P]) -> DedupResult<Vec<String>>
    where
        P: AsRef<Path> + Sync,
    {
        use rayon::prelude::*;

        // Phase 1: compute hashes in parallel (CPU-intensive, embarrassingly parallel).
        let hash_results: Vec<(PathBuf, DedupResult<hash::FileHash>)> = paths
            .par_iter()
            .map(|p| {
                let path = p.as_ref().to_path_buf();
                if !path.exists() {
                    return (path.clone(), Err(DedupError::FileNotFound(path)));
                }
                let result = hash::compute_file_hash(&path);
                (path, result)
            })
            .collect();

        // Phase 2: merge into DB sequentially (requires exclusive &mut self).
        let mut errors = Vec::new();
        for (path, result) in hash_results {
            match result {
                Ok(file_hash) => {
                    if let Err(e) = self.database.insert_file(&path, &file_hash.to_hex()).await {
                        errors.push(format!("{}: {}", path.display(), e));
                    }
                }
                Err(e) => {
                    errors.push(format!("{}: {}", path.display(), e));
                }
            }
        }

        Ok(errors)
    }

    /// Find duplicates using the specified strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if duplicate detection fails.
    pub async fn find_duplicates(
        &self,
        strategy: DetectionStrategy,
    ) -> DedupResult<DuplicateReport> {
        self.find_duplicates_with_progress(strategy, &progress::NullReporter)
            .await
    }

    /// Find duplicates with progress reporting.
    ///
    /// Like `find_duplicates` but emits progress events through the
    /// supplied [`ProgressReporter`](progress::ProgressReporter).  This is
    /// the primary integration point for large-library deduplication where
    /// the caller wants to display a progress bar or support cancellation.
    ///
    /// # Errors
    ///
    /// Returns an error if duplicate detection fails.
    pub async fn find_duplicates_with_progress(
        &self,
        strategy: DetectionStrategy,
        reporter: &dyn progress::ProgressReporter,
    ) -> DedupResult<DuplicateReport> {
        use progress::{ProgressEvent, ProgressTracker};

        let run_start = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut report = DuplicateReport::new();

        // Count phases for total progress.
        let phase_count = [
            strategy.includes_hash(),
            strategy.includes_perceptual(),
            strategy.includes_ssim(),
            strategy.includes_histogram(),
            strategy.includes_feature_match(),
            strategy.includes_audio(),
            strategy.includes_metadata(),
        ]
        .iter()
        .filter(|&&b| b)
        .count();

        let mut completed_phases = 0usize;

        // Exact hash duplicates
        if strategy.includes_hash() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "exact_hash", 0);
            let hash_dups = self.find_hash_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = hash_dups.len();
            report.add_groups(hash_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // Perceptual hash duplicates
        if strategy.includes_perceptual() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "perceptual_hash", 0);
            let perceptual_dups = self.find_perceptual_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = perceptual_dups.len();
            report.add_groups(perceptual_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // SSIM duplicates
        if strategy.includes_ssim() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "ssim", 0);
            let ssim_dups = self.find_ssim_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = ssim_dups.len();
            report.add_groups(ssim_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // Histogram duplicates
        if strategy.includes_histogram() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "histogram", 0);
            let histogram_dups = self.find_histogram_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = histogram_dups.len();
            report.add_groups(histogram_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // Feature match duplicates
        if strategy.includes_feature_match() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "feature_match", 0);
            let feature_dups = self.find_feature_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = feature_dups.len();
            report.add_groups(feature_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // Audio fingerprint duplicates
        if strategy.includes_audio() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "audio_fingerprint", 0);
            let audio_dups = self.find_audio_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = audio_dups.len();
            report.add_groups(audio_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // Metadata duplicates
        if strategy.includes_metadata() {
            if reporter.is_cancelled() {
                return Ok(report);
            }
            let mut tracker = ProgressTracker::new(reporter, "metadata", 0);
            let metadata_dups = self.find_metadata_duplicates().await?;
            tracker.tick_batch(1);
            let groups_found = metadata_dups.len();
            report.add_groups(metadata_dups);
            tracker.complete(groups_found);
            completed_phases += 1;
        }

        // Emit run completed event.
        let run_end = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        reporter.on_event(&ProgressEvent::RunCompleted {
            total_groups: report.groups.len(),
            total_elapsed_ms: run_end.saturating_sub(run_start),
        });

        let _ = (phase_count, completed_phases); // used for bookkeeping

        Ok(report)
    }

    /// Find exact duplicates by cryptographic hash.
    async fn find_hash_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        let duplicates = self.database.find_duplicate_hashes().await?;
        let mut groups = Vec::new();

        for (hash, files) in duplicates {
            if files.len() > 1 {
                groups.push(DuplicateGroup {
                    files,
                    scores: vec![SimilarityScore {
                        method: "exact_hash".to_string(),
                        score: 1.0,
                        metadata: vec![("hash".to_string(), hash)],
                    }],
                });
            }
        }

        Ok(groups)
    }

    /// Find perceptual hash duplicates.
    ///
    /// When `config.use_lsh` is enabled (the default), uses a
    /// [`BitLshIndex`](lsh_index::BitLshIndex) for sub-quadratic performance.
    /// Otherwise falls back to O(n^2) pairwise comparison.
    ///
    /// Loads perceptual hashes stored in the `fingerprints` table under the key
    /// `"phash"`.  Pairs with a Hamming distance below the threshold derived
    /// from `config.perceptual_threshold` are grouped together.
    async fn find_perceptual_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        // Threshold: perceptual_threshold is 0.0-1.0 similarity.
        // Hamming distance over 64 bits → distance ≤ (1 - threshold) * 64.
        let max_hamming = ((1.0 - self.config.perceptual_threshold) * 64.0) as u32;

        // Fetch all stored perceptual hash fingerprints.
        let stored = self.database.get_all_fingerprints_by_type("phash").await?;

        // Build a list of (path, PerceptualHash) from stored hex strings.
        let mut hashes: Vec<(String, visual::PerceptualHash)> = Vec::new();
        for (path, hex) in stored {
            if let Ok(value) = u64::from_str_radix(&hex, 16) {
                hashes.push((path, visual::PerceptualHash::new(value, 64)));
            }
        }

        // If no stored hashes, nothing to compare.
        if hashes.len() < 2 {
            return Ok(Vec::new());
        }

        // Bloom filter pre-screening: discard definitely-unique perceptual hashes
        // before any expensive pairwise or LSH comparison.
        //
        // Strategy: quantise each 64-bit pHash down to its top 16 bits and run
        // the items through the shared `prescreen_perceptual_hashes` helper.
        // Items whose quantised hash has never been seen before are provably
        // unique (no false negatives in a Bloom filter) and are dropped from
        // the candidate set.  Remaining items are forwarded to the LSH/pairwise
        // pass as before.
        let hashes: Vec<(String, visual::PerceptualHash)> = if self.config.bloom_prescreen {
            let raw: Vec<u64> = hashes.iter().map(|(_, ph)| ph.hash()).collect();
            let prescreen = bloom_filter::prescreen_perceptual_hashes(
                &raw,
                16, // quantize_bits: top 16 bits capture coarse visual similarity
                self.config.bloom_capacity,
                self.config.bloom_fpr,
            );
            prescreen
                .candidates
                .iter()
                .filter_map(|&idx| hashes.get(idx).cloned())
                .collect()
        } else {
            hashes
        };

        // After bloom pre-screening, re-check candidate count.
        if hashes.len() < 2 {
            return Ok(Vec::new());
        }

        if self.config.use_lsh {
            self.find_perceptual_duplicates_lsh(&hashes, max_hamming)
        } else {
            group_by_pairwise_similarity(
                &hashes,
                max_hamming,
                |h1, h2| h1.hamming_distance(h2),
                |h1, h2| h1.similarity(h2),
                "perceptual_hash",
            )
        }
    }

    /// LSH-accelerated perceptual hash duplicate detection.
    ///
    /// Replaces the O(n^2) pairwise comparison with sub-quadratic LSH
    /// candidate generation followed by exact Hamming distance verification.
    fn find_perceptual_duplicates_lsh(
        &self,
        hashes: &[(String, visual::PerceptualHash)],
        max_hamming: u32,
    ) -> DedupResult<Vec<DuplicateGroup>> {
        // Build id <-> path mapping.
        let id_hashes: Vec<(u64, u64)> = hashes
            .iter()
            .enumerate()
            .map(|(i, (_, ph))| (i as u64, ph.hash()))
            .collect();

        // Run LSH dedup pass.
        let lsh_result = lsh_index::lsh_dedup_pass(
            &id_hashes,
            max_hamming,
            self.config.lsh_num_tables,
            self.config.lsh_bits_per_table,
            self.config.lsh_seed,
        );

        // Group by transitive closure.
        let all_ids: Vec<u64> = (0..hashes.len() as u64).collect();
        let groups = lsh_index::group_by_lsh_pairs(&lsh_result.pairs, &all_ids);

        // Convert back to DuplicateGroup with paths.
        let mut result = Vec::new();
        for group_ids in &groups {
            let files: Vec<String> = group_ids
                .iter()
                .filter_map(|&id| hashes.get(id as usize).map(|(p, _)| p.clone()))
                .collect();

            if files.len() < 2 {
                continue;
            }

            // Find best pairwise similarity within the group for scoring.
            let mut best_sim = 0.0f64;
            for i in 0..group_ids.len() {
                for j in (i + 1)..group_ids.len() {
                    let ia = group_ids[i] as usize;
                    let ib = group_ids[j] as usize;
                    if let (Some((_, ha)), Some((_, hb))) = (hashes.get(ia), hashes.get(ib)) {
                        let sim = ha.similarity(hb);
                        if sim > best_sim {
                            best_sim = sim;
                        }
                    }
                }
            }

            result.push(DuplicateGroup {
                files,
                scores: vec![SimilarityScore {
                    method: "perceptual_hash_lsh".to_string(),
                    score: best_sim,
                    metadata: vec![
                        (
                            "lsh_candidates".to_string(),
                            lsh_result.candidates_checked.to_string(),
                        ),
                        (
                            "comparison_ratio".to_string(),
                            format!("{:.4}", lsh_result.comparison_ratio()),
                        ),
                    ],
                }],
            });
        }

        Ok(result)
    }

    /// Find SSIM duplicates.
    ///
    /// Retrieves stored thumbnail pixel data (type `"thumbnail"`) from the
    /// fingerprints table, reconstructs grayscale `Image` objects, and
    /// computes the Structural Similarity Index (SSIM) between every unique
    /// pair.  Pairs with SSIM above `config.ssim_threshold` are grouped.
    ///
    /// Thumbnail resolution is controlled by `config.thumbnail_resolution`.
    async fn find_ssim_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        let threshold = self.config.ssim_threshold;
        let res = self.config.thumbnail_resolution.max(4);
        let expected_bytes = res * res;

        // Thumbnail images are stored hex-encoded in the fingerprints table.
        let stored = self
            .database
            .get_all_fingerprints_by_type("thumbnail")
            .await?;

        // Decode hex → bytes → Image (configurable resolution, grayscale).
        let mut images: Vec<(String, visual::Image)> = Vec::new();
        for (path, hex) in stored {
            let bytes = decode_hex_bytes(&hex)?;
            // Accept thumbnails matching the configured resolution.
            if bytes.len() == expected_bytes {
                if let Ok(img) = visual::Image::from_data(res, res, 1, bytes) {
                    images.push((path, img));
                }
            }
        }

        if images.len() < 2 {
            return Ok(Vec::new());
        }

        let ssim_params = visual::SsimParams::default();
        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut assigned = vec![false; images.len()];

        for i in 0..images.len() {
            if assigned[i] {
                continue;
            }
            let mut group_files = vec![images[i].0.clone()];
            let mut best_score = 0.0f64;

            for j in (i + 1)..images.len() {
                if assigned[j] {
                    continue;
                }
                let ssim = visual::compute_ssim(&images[i].1, &images[j].1, &ssim_params);
                if ssim >= threshold {
                    group_files.push(images[j].0.clone());
                    assigned[j] = true;
                    if ssim > best_score {
                        best_score = ssim;
                    }
                }
            }

            if group_files.len() > 1 {
                assigned[i] = true;
                groups.push(DuplicateGroup {
                    files: group_files,
                    scores: vec![SimilarityScore {
                        method: "ssim".to_string(),
                        score: best_score,
                        metadata: Vec::new(),
                    }],
                });
            }
        }

        Ok(groups)
    }

    /// Find histogram duplicates.
    ///
    /// Loads stored colour histogram fingerprints (type `"histogram"`) from
    /// the database.  The data is a JSON-encoded flat array of `u32` bin
    /// counts (three channels × 256 bins = 768 values).  Histogram
    /// correlation is computed between every pair; pairs above
    /// `config.histogram_threshold` are grouped.
    async fn find_histogram_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        let threshold = self.config.histogram_threshold;

        let stored = self
            .database
            .get_all_fingerprints_by_type("histogram")
            .await?;

        // Decode stored JSON histogram data → Vec<Vec<u32>>.
        let mut histograms: Vec<(String, Vec<Vec<u32>>)> = Vec::new();
        for (path, json_str) in stored {
            if let Ok(flat) = serde_json::from_str::<Vec<u32>>(&json_str) {
                // Each channel has 256 bins; infer channel count.
                if flat.len() % 256 == 0 && !flat.is_empty() {
                    let channels = flat.len() / 256;
                    let hist: Vec<Vec<u32>> = (0..channels)
                        .map(|c| flat[c * 256..(c + 1) * 256].to_vec())
                        .collect();
                    histograms.push((path, hist));
                }
            }
        }

        if histograms.len() < 2 {
            return Ok(Vec::new());
        }

        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut assigned = vec![false; histograms.len()];

        for i in 0..histograms.len() {
            if assigned[i] {
                continue;
            }
            let mut group_files = vec![histograms[i].0.clone()];
            let mut best_score = 0.0f64;

            for j in (i + 1)..histograms.len() {
                if assigned[j] {
                    continue;
                }
                let corr = visual::compare_histograms(&histograms[i].1, &histograms[j].1);
                if corr >= threshold {
                    group_files.push(histograms[j].0.clone());
                    assigned[j] = true;
                    if corr > best_score {
                        best_score = corr;
                    }
                }
            }

            if group_files.len() > 1 {
                assigned[i] = true;
                groups.push(DuplicateGroup {
                    files: group_files,
                    scores: vec![SimilarityScore {
                        method: "histogram".to_string(),
                        score: best_score,
                        metadata: Vec::new(),
                    }],
                });
            }
        }

        Ok(groups)
    }

    /// Find feature match duplicates.
    ///
    /// Loads stored feature-vector fingerprints (type `"feature_vector"`) from
    /// the database.  Each feature vector is a JSON-encoded `Vec<f64>`.
    /// Cosine similarity is computed between every pair; pairs whose cosine
    /// similarity exceeds `config.perceptual_threshold` (reused as a generic
    /// visual similarity threshold) are grouped.
    async fn find_feature_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        let threshold = self.config.perceptual_threshold;

        let stored = self
            .database
            .get_all_fingerprints_by_type("feature_vector")
            .await?;

        // Decode JSON feature vectors.
        let mut vectors: Vec<(String, Vec<f64>)> = Vec::new();
        for (path, json_str) in stored {
            if let Ok(vec) = serde_json::from_str::<Vec<f64>>(&json_str) {
                if !vec.is_empty() {
                    vectors.push((path, vec));
                }
            }
        }

        if vectors.len() < 2 {
            return Ok(Vec::new());
        }

        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut assigned = vec![false; vectors.len()];

        for i in 0..vectors.len() {
            if assigned[i] {
                continue;
            }
            let mut group_files = vec![vectors[i].0.clone()];
            let mut best_score = 0.0f64;

            for j in (i + 1)..vectors.len() {
                if assigned[j] {
                    continue;
                }
                let sim = cosine_similarity(&vectors[i].1, &vectors[j].1);
                if sim >= threshold {
                    group_files.push(vectors[j].0.clone());
                    assigned[j] = true;
                    if sim > best_score {
                        best_score = sim;
                    }
                }
            }

            if group_files.len() > 1 {
                assigned[i] = true;
                groups.push(DuplicateGroup {
                    files: group_files,
                    scores: vec![SimilarityScore {
                        method: "feature_vector".to_string(),
                        score: best_score,
                        metadata: Vec::new(),
                    }],
                });
            }
        }

        Ok(groups)
    }

    /// Find audio fingerprint duplicates.
    ///
    /// Loads stored audio fingerprint data (type `"audio_fingerprint"`) from
    /// the database.  Each fingerprint is stored as a hex string of bytes.
    /// Pairs whose bit-level Hamming distance is within the threshold derived
    /// from `config.audio_threshold` are grouped together.
    async fn find_audio_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        let threshold = self.config.audio_threshold;

        let stored = self
            .database
            .get_all_fingerprints_by_type("audio_fingerprint")
            .await?;

        // Decode hex fingerprints → AudioFingerprint.
        let mut fingerprints: Vec<(String, audio::AudioFingerprint)> = Vec::new();
        for (path, hex) in stored {
            let bytes = decode_hex_bytes(&hex)?;
            if !bytes.is_empty() {
                fingerprints.push((path, audio::AudioFingerprint::new(bytes, 11025, 0.0)));
            }
        }

        if fingerprints.len() < 2 {
            return Ok(Vec::new());
        }

        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut assigned = vec![false; fingerprints.len()];

        for i in 0..fingerprints.len() {
            if assigned[i] {
                continue;
            }
            let mut group_files = vec![fingerprints[i].0.clone()];
            let mut best_score = 0.0f64;

            for j in (i + 1)..fingerprints.len() {
                if assigned[j] {
                    continue;
                }
                let sim = fingerprints[i].1.similarity(&fingerprints[j].1);
                if sim >= threshold {
                    group_files.push(fingerprints[j].0.clone());
                    assigned[j] = true;
                    if sim > best_score {
                        best_score = sim;
                    }
                }
            }

            if group_files.len() > 1 {
                assigned[i] = true;
                groups.push(DuplicateGroup {
                    files: group_files,
                    scores: vec![SimilarityScore {
                        method: "audio_fingerprint".to_string(),
                        score: best_score,
                        metadata: Vec::new(),
                    }],
                });
            }
        }

        Ok(groups)
    }

    /// Find metadata duplicates.
    ///
    /// Fetches all files with their stored metadata from the database and
    /// compares every unique pair using `metadata::compare_metadata`.  The
    /// key signals for a "near-duplicate" are:
    ///
    /// - Duration within ±1 second of each other.
    /// - Same video resolution (or both without resolution data).
    /// - Same video and audio codec.
    ///
    /// The overall weighted metadata similarity must exceed
    /// `config.metadata_threshold`.
    async fn find_metadata_duplicates(&self) -> DedupResult<Vec<DuplicateGroup>> {
        use metadata::{compare_metadata, MediaMetadata};
        use std::path::PathBuf;

        let threshold = self.config.metadata_threshold;

        let rows = self.database.get_all_files_with_metadata().await?;

        if rows.len() < 2 {
            return Ok(Vec::new());
        }

        // Reconstruct MediaMetadata objects from the DB rows.
        let media_meta: Vec<MediaMetadata> = rows
            .iter()
            .map(
                |(path, duration, width, height, video_codec, audio_codec, container)| {
                    let fs_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    let mut m = MediaMetadata::new(PathBuf::from(path), fs_size);
                    m.duration = *duration;
                    m.width = width.map(|v| v as u32);
                    m.height = height.map(|v| v as u32);
                    m.video_codec = video_codec.clone();
                    m.audio_codec = audio_codec.clone();
                    m.container = container.clone();
                    m
                },
            )
            .collect();

        let paths: Vec<String> = rows.iter().map(|(p, ..)| p.clone()).collect();

        let mut groups: Vec<DuplicateGroup> = Vec::new();
        let mut assigned = vec![false; media_meta.len()];

        for i in 0..media_meta.len() {
            if assigned[i] {
                continue;
            }
            let mut group_files = vec![paths[i].clone()];
            let mut best_score = 0.0f64;
            let mut best_duration_diff: Option<f64> = None;

            for j in (i + 1)..media_meta.len() {
                if assigned[j] {
                    continue;
                }

                // Fast pre-filter: duration must match within ±1 second
                // when both files have duration information stored.
                let duration_ok = match (media_meta[i].duration, media_meta[j].duration) {
                    (Some(d1), Some(d2)) => (d1 - d2).abs() <= 1.0,
                    _ => true, // No duration data → don't discard
                };
                if !duration_ok {
                    continue;
                }

                let sim = compare_metadata(&media_meta[i], &media_meta[j]);
                let score = sim.overall_score();
                if score >= threshold {
                    group_files.push(paths[j].clone());
                    assigned[j] = true;
                    if score > best_score {
                        best_score = score;
                        best_duration_diff = match (media_meta[i].duration, media_meta[j].duration)
                        {
                            (Some(d1), Some(d2)) => Some((d1 - d2).abs()),
                            _ => None,
                        };
                    }
                }
            }

            if group_files.len() > 1 {
                assigned[i] = true;
                let mut score_entry = SimilarityScore {
                    method: "metadata".to_string(),
                    score: best_score,
                    metadata: Vec::new(),
                };
                if let Some(diff) = best_duration_diff {
                    score_entry
                        .metadata
                        .push(("duration_diff_secs".to_string(), format!("{diff:.3}")));
                }
                groups.push(DuplicateGroup {
                    files: group_files,
                    scores: vec![score_entry],
                });
            }
        }

        Ok(groups)
    }

    /// Get database statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if database query fails.
    pub async fn get_stats(&self) -> DedupResult<DedupStats> {
        let total_files = self.database.count_files().await?;
        let total_hashes = self.database.count_unique_hashes().await?;

        Ok(DedupStats {
            total_files,
            total_hashes,
            duplicate_files: total_files.saturating_sub(total_hashes),
        })
    }

    /// Close the database.
    pub async fn close(self) -> DedupResult<()> {
        self.database.close().await?;
        Ok(())
    }

    /// Fast-path bloom filter check: does this hash *possibly* exist in the index?
    ///
    /// Returns `true` if the Bloom filter reports the hash *might* be a
    /// duplicate (i.e., the same bytes were inserted previously).  Returns
    /// `false` only if the hash is **definitely** not present — meaning the
    /// file is provably unique and expensive pairwise comparisons can be
    /// skipped entirely.
    ///
    /// When bloom pre-screening is disabled (`config.bloom_prescreen == false`)
    /// this always returns `true` so callers always fall through to the full
    /// comparison path.
    #[must_use]
    pub fn might_be_duplicate(&self, hash_bytes: &[u8]) -> bool {
        match &self.bloom {
            Some(bloom) => bloom.contains(hash_bytes),
            None => true,
        }
    }

    /// Reset the in-memory bloom filter without touching the database.
    ///
    /// Useful after a bulk-index session to free the bloom filter's memory,
    /// or to rebuild it from scratch with a different capacity.  The database
    /// index is not affected.
    pub fn reset_bloom(&mut self) {
        if let Some(ref mut bloom) = self.bloom {
            bloom.clear();
        }
    }
}

/// Deduplication statistics.
#[derive(Debug, Clone)]
pub struct DedupStats {
    /// Total number of indexed files
    pub total_files: usize,

    /// Total number of unique hashes
    pub total_hashes: usize,

    /// Number of duplicate files
    pub duplicate_files: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_strategy() {
        assert!(DetectionStrategy::ExactHash.includes_hash());
        assert!(!DetectionStrategy::ExactHash.includes_perceptual());

        assert!(DetectionStrategy::All.includes_hash());
        assert!(DetectionStrategy::All.includes_perceptual());
        assert!(DetectionStrategy::All.includes_audio());

        assert!(DetectionStrategy::Fast.includes_hash());
        assert!(DetectionStrategy::Fast.includes_perceptual());
        assert!(!DetectionStrategy::Fast.includes_ssim());
    }

    #[test]
    fn test_config_default() {
        let config = DedupConfig::default();
        assert_eq!(config.perceptual_threshold, 0.95);
        assert_eq!(config.ssim_threshold, 0.90);
        assert!(config.parallel);
    }

    #[test]
    fn test_config_lsh_defaults() {
        let config = DedupConfig::default();
        assert!(config.use_lsh);
        assert_eq!(config.lsh_num_tables, 8);
        assert_eq!(config.lsh_bits_per_table, 8);
        assert_eq!(config.lsh_seed, 42);
    }

    #[test]
    fn test_config_bloom_defaults() {
        let config = DedupConfig::default();
        // bloom_prescreen is off by default; capacity and fpr are set
        assert!(!config.bloom_prescreen);
        assert_eq!(config.bloom_capacity, 10_000);
        assert!((config.bloom_fpr - 0.01f32).abs() < f32::EPSILON);
    }

    /// Compile-time check: `par_index_files` accepts an empty slice without panicking.
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_par_index_files_empty_slice() {
        use std::path::PathBuf;
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!(
            "oxidedup_test_par_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let config = DedupConfig {
            database_path: db_path.clone(),
            ..DedupConfig::default()
        };
        if let Ok(mut detector) = DuplicateDetector::new(config).await {
            let no_paths: &[PathBuf] = &[];
            let errors = detector
                .par_index_files(no_paths)
                .await
                .expect("par_index_files should succeed on empty input");
            assert!(errors.is_empty(), "No errors expected for empty input");
            let _ = detector.close().await;
        }
        let _ = std::fs::remove_file(&db_path);
    }

    /// par_index_files returns per-file errors for non-existent paths (no panic).
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_par_index_files_nonexistent_paths() {
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!(
            "oxidedup_test_par_ne_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let config = DedupConfig {
            database_path: db_path.clone(),
            ..DedupConfig::default()
        };
        if let Ok(mut detector) = DuplicateDetector::new(config).await {
            let missing = vec![
                PathBuf::from("/nonexistent/path/a.mp4"),
                PathBuf::from("/nonexistent/path/b.mp4"),
            ];
            let errors = detector
                .par_index_files(&missing)
                .await
                .expect("par_index_files should return Ok even when files are missing");
            assert_eq!(errors.len(), 2, "Should have one error per missing file");
            let _ = detector.close().await;
        }
        let _ = std::fs::remove_file(&db_path);
    }

    // ---- Bloom filter wiring tests ----

    /// When bloom_prescreen is false (default), might_be_duplicate always returns true.
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_might_be_duplicate_no_bloom_always_true() {
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!(
            "oxidedup_bloom_noscreen_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let config = DedupConfig {
            database_path: db_path.clone(),
            bloom_prescreen: false,
            ..DedupConfig::default()
        };
        if let Ok(detector) = DuplicateDetector::new(config).await {
            // Without a bloom filter, every hash is a "maybe duplicate"
            assert!(
                detector.might_be_duplicate(b"some_hash_bytes"),
                "Should always return true when bloom is disabled"
            );
            assert!(
                detector.might_be_duplicate(b""),
                "Empty bytes: should return true without bloom"
            );
            let _ = detector.close().await;
        }
        let _ = std::fs::remove_file(&db_path);
    }

    /// When bloom_prescreen is enabled, unknown hashes return false from might_be_duplicate.
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_might_be_duplicate_with_bloom_unknown_hash() {
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!(
            "oxidedup_bloom_unknown_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let config = DedupConfig {
            database_path: db_path.clone(),
            bloom_prescreen: true,
            bloom_capacity: 1000,
            bloom_fpr: 0.01,
            ..DedupConfig::default()
        };
        if let Ok(detector) = DuplicateDetector::new(config).await {
            // A freshly created detector has an empty bloom filter — unknown hashes
            // must return false (definitely not a duplicate)
            assert!(
                !detector.might_be_duplicate(b"never_inserted_hash"),
                "Unknown hash should return false from a fresh bloom filter"
            );
            let _ = detector.close().await;
        }
        let _ = std::fs::remove_file(&db_path);
    }

    /// reset_bloom clears the filter so previously-seen hashes return false.
    #[tokio::test]
    #[cfg(feature = "sqlite")]
    async fn test_reset_bloom_clears_state() {
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!(
            "oxidedup_bloom_reset_{}.db",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos()
        ));
        let config = DedupConfig {
            database_path: db_path.clone(),
            bloom_prescreen: true,
            bloom_capacity: 1000,
            bloom_fpr: 0.01,
            ..DedupConfig::default()
        };
        if let Ok(mut detector) = DuplicateDetector::new(config).await {
            // Manually insert into the bloom filter by inserting known bytes
            if let Some(ref mut bloom) = detector.bloom {
                bloom.insert(b"known_hash");
            }
            // Now it should report a potential duplicate
            assert!(
                detector.might_be_duplicate(b"known_hash"),
                "After insert, bloom should report potential duplicate"
            );
            // After reset, the same hash must not be found
            detector.reset_bloom();
            assert!(
                !detector.might_be_duplicate(b"known_hash"),
                "After reset_bloom, hash should not be found"
            );
            let _ = detector.close().await;
        }
        let _ = std::fs::remove_file(&db_path);
    }
}
