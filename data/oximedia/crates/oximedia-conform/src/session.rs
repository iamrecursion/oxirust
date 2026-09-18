//! Conform session management.

use crate::config::ConformConfig;
use crate::error::{ConformError, ConformResult};
use crate::exporters::report::{AmbiguousMatch, MatchReport};
use crate::exporters::{project::ProjectExporter, sequence::SequenceExporter, Exporter};
use crate::importers::{edl::EdlImporter, TimelineImporter};
use crate::matching::strategies::MatchStrategy;
use crate::media::{catalog::MediaCatalog, scanner::MediaScanner, ScanProgress};
use crate::qc::{checker::QualityChecker, validator::Validator};
use crate::timeline::{Timeline, TimelineClip, Track, TrackKind};
use crate::types::{ClipMatch, ClipReference, OutputFormat};
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Conform session status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    /// Session created but not started.
    Created,
    /// Scanning source media.
    Scanning,
    /// Matching clips.
    Matching,
    /// Conforming.
    Conforming,
    /// Exporting.
    Exporting,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
}

/// A conform session that manages the entire conforming workflow.
pub struct ConformSession {
    /// Unique session ID.
    id: Uuid,
    /// Session name.
    name: String,
    /// EDL or timeline path.
    timeline_path: PathBuf,
    /// Source media paths to scan.
    source_paths: Vec<PathBuf>,
    /// Output path for conformed sequence.
    output_path: PathBuf,
    /// Configuration.
    config: ConformConfig,
    /// Media catalog.
    catalog: MediaCatalog,
    /// Clips to match.
    clips: Vec<ClipReference>,
    /// Match results.
    matches: Arc<RwLock<Vec<ClipMatch>>>,
    /// Missing clips.
    missing: Arc<RwLock<Vec<ClipReference>>>,
    /// Ambiguous matches.
    ambiguous: Arc<RwLock<Vec<AmbiguousMatch>>>,
    /// Session status.
    status: Arc<RwLock<SessionStatus>>,
}

impl ConformSession {
    /// Create a new conform session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be created.
    pub fn new<P: AsRef<Path>>(
        name: String,
        timeline_path: P,
        source_paths: Vec<PathBuf>,
        output_path: PathBuf,
        config: ConformConfig,
    ) -> ConformResult<Self> {
        let id = Uuid::new_v4();
        let catalog = MediaCatalog::in_memory()?;

        Ok(Self {
            id,
            name,
            timeline_path: timeline_path.as_ref().to_path_buf(),
            source_paths,
            output_path,
            config,
            catalog,
            clips: Vec::new(),
            matches: Arc::new(RwLock::new(Vec::new())),
            missing: Arc::new(RwLock::new(Vec::new())),
            ambiguous: Arc::new(RwLock::new(Vec::new())),
            status: Arc::new(RwLock::new(SessionStatus::Created)),
        })
    }

    /// Create a new conform session with a database.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be created.
    pub fn with_database<P: AsRef<Path>>(
        name: String,
        timeline_path: P,
        source_paths: Vec<PathBuf>,
        output_path: PathBuf,
        config: ConformConfig,
        db_path: P,
    ) -> ConformResult<Self> {
        let id = Uuid::new_v4();
        let catalog = MediaCatalog::new(db_path)?;

        Ok(Self {
            id,
            name,
            timeline_path: timeline_path.as_ref().to_path_buf(),
            source_paths,
            output_path,
            config,
            catalog,
            clips: Vec::new(),
            matches: Arc::new(RwLock::new(Vec::new())),
            missing: Arc::new(RwLock::new(Vec::new())),
            ambiguous: Arc::new(RwLock::new(Vec::new())),
            status: Arc::new(RwLock::new(SessionStatus::Created)),
        })
    }

    /// Get the session ID.
    #[must_use]
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the session name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the current status.
    #[must_use]
    pub fn status(&self) -> SessionStatus {
        *self.status.read()
    }

    /// Set the status.
    fn set_status(&self, status: SessionStatus) {
        *self.status.write() = status;
    }

    /// Load clips from the timeline file.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline cannot be loaded.
    pub async fn load_timeline(&mut self) -> ConformResult<()> {
        info!("Loading timeline from {}", self.timeline_path.display());

        // Detect format and import
        let clips = if self.timeline_path.extension().and_then(|e| e.to_str()) == Some("edl") {
            let importer = EdlImporter::new();
            importer.import(&self.timeline_path)?
        } else {
            return Err(ConformError::UnsupportedFormat(
                "Only EDL files are currently supported".to_string(),
            ));
        };

        info!("Loaded {} clips from timeline", clips.len());
        self.clips = clips;
        Ok(())
    }

    /// Scan source media directories.
    ///
    /// # Errors
    ///
    /// Returns an error if scanning fails.
    pub async fn scan_sources(&mut self) -> ConformResult<ScanProgress> {
        self.set_status(SessionStatus::Scanning);
        info!("Scanning {} source paths", self.source_paths.len());

        let mut scanner = MediaScanner::new();
        scanner.generate_checksums(self.config.verify_checksums);

        let progress = Arc::new(ScanProgress::new());
        let mut all_media = Vec::new();

        for source_path in &self.source_paths {
            info!("Scanning {}", source_path.display());
            let media = scanner.scan(source_path, Some(progress.clone()))?;
            all_media.extend(media);
        }

        info!("Found {} media files", all_media.len());
        self.catalog.add_many(all_media)?;

        Ok((*progress).clone())
    }

    /// Match clips to media files.
    ///
    /// # Errors
    ///
    /// Returns an error if matching fails.
    pub async fn match_clips(&mut self) -> ConformResult<MatchReport> {
        self.set_status(SessionStatus::Matching);
        info!("Matching {} clips", self.clips.len());

        let strategy = MatchStrategy::new(self.config.clone());
        let all_media = self.catalog.get_all()?;

        let mut matched = Vec::new();
        let mut missing = Vec::new();
        let mut ambiguous = Vec::new();

        for clip in &self.clips {
            let clip_matches = strategy.match_clip(clip, &all_media);

            if clip_matches.is_empty() {
                warn!("No match found for clip: {}", clip.id);
                missing.push(clip.clone());
            } else if clip_matches.len() == 1 {
                debug!(
                    "Found match for clip {}: score {:.2}",
                    clip.id, clip_matches[0].score
                );
                matched.push(clip_matches[0].clone());
            } else {
                // Multiple matches - check if ambiguous
                if strategy.is_ambiguous(&clip_matches, 0.05) {
                    warn!(
                        "Ambiguous matches for clip {}: {} candidates",
                        clip.id,
                        clip_matches.len()
                    );
                    ambiguous.push(AmbiguousMatch {
                        clip: clip.clone(),
                        candidates: clip_matches,
                    });
                } else {
                    // Use best match
                    debug!(
                        "Using best match for clip {}: score {:.2}",
                        clip.id, clip_matches[0].score
                    );
                    matched.push(clip_matches[0].clone());
                }
            }
        }

        info!(
            "Matching complete: {} matched, {} missing, {} ambiguous",
            matched.len(),
            missing.len(),
            ambiguous.len()
        );

        *self.matches.write() = matched.clone();
        *self.missing.write() = missing.clone();
        *self.ambiguous.write() = ambiguous.clone();

        Ok(MatchReport::new(matched, missing, ambiguous))
    }

    /// Validate matches.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate(&self) -> ConformResult<()> {
        info!("Validating matches");

        let validator = Validator::new(self.config.clone());
        let matches = self.matches.read();
        let report = validator.validate_all(&matches);

        if !report.is_valid {
            if self.config.strict_validation {
                return Err(ConformError::Validation(format!(
                    "Validation failed with {} errors",
                    report.errors.len()
                )));
            }
            warn!(
                "Validation found {} errors and {} warnings",
                report.errors.len(),
                report.warnings.len()
            );
        }

        let checker = QualityChecker::new();
        checker.check_format_consistency(&matches)?;
        checker.check_timecode_continuity(&matches)?;

        info!("Validation complete");
        Ok(())
    }

    /// Perform conforming.
    ///
    /// # Errors
    ///
    /// Returns an error if conforming fails.
    pub async fn conform(&mut self) -> ConformResult<Timeline> {
        self.set_status(SessionStatus::Conforming);
        info!("Conforming sequence");

        let matches = self.matches.read();
        if matches.is_empty() {
            return Err(ConformError::Other("No matches to conform".to_string()));
        }

        // Build timeline
        let fps = matches[0].clip.fps;
        let mut timeline = Timeline::new(self.name.clone(), fps);

        // Create video track
        let mut video_track = Track::new("V1".to_string(), TrackKind::Video);
        for clip_match in matches.iter() {
            if clip_match.clip.track.has_video() {
                let clip = TimelineClip::from_match(clip_match);
                video_track.add_clip(clip);
            }
        }
        video_track.sort_clips();
        timeline.add_video_track(video_track);

        // Create audio track
        let mut audio_track = Track::new("A1".to_string(), TrackKind::Audio);
        for clip_match in matches.iter() {
            if clip_match.clip.track.has_audio() {
                let clip = TimelineClip::from_match(clip_match);
                audio_track.add_clip(clip);
            }
        }
        audio_track.sort_clips();
        timeline.add_audio_track(audio_track);

        info!(
            "Timeline created with {} video tracks and {} audio tracks",
            timeline.video_tracks.len(),
            timeline.audio_tracks.len()
        );

        self.set_status(SessionStatus::Completed);
        Ok(timeline)
    }

    /// Export the conformed sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if export fails.
    pub async fn export(&self, format: OutputFormat) -> ConformResult<()> {
        self.set_status(SessionStatus::Exporting);
        info!("Exporting to {:?}", format);

        let matches = self.matches.read();

        match format {
            OutputFormat::Mp4
            | OutputFormat::Matroska
            | OutputFormat::FrameSequenceDpx
            | OutputFormat::FrameSequenceTiff
            | OutputFormat::FrameSequencePng => {
                let exporter = SequenceExporter::new(matches.clone());
                exporter.export(&self.output_path, format)?;
            }
            OutputFormat::Edl
            | OutputFormat::FcpXml
            | OutputFormat::PremiereXml
            | OutputFormat::Aaf => {
                let exporter = ProjectExporter::new(matches.clone());
                exporter.export(&self.output_path, format)?;
            }
        }

        info!("Export complete");
        Ok(())
    }

    /// Generate a match report.
    #[must_use]
    pub fn get_match_report(&self) -> MatchReport {
        MatchReport::new(
            self.matches.read().clone(),
            self.missing.read().clone(),
            self.ambiguous.read().clone(),
        )
    }

    /// Run the complete conform workflow.
    ///
    /// # Errors
    ///
    /// Returns an error if any step fails.
    pub async fn run(&mut self) -> ConformResult<MatchReport> {
        info!("Starting conform session: {}", self.name);

        // Load timeline
        self.load_timeline().await?;

        // Scan sources
        self.scan_sources().await?;

        // Match clips
        let report = self.match_clips().await?;

        // Validate
        self.validate()?;

        // Conform
        self.conform().await?;

        info!("Conform session complete");
        Ok(report)
    }
}

// ── Watch folder mode ─────────────────────────────────────────────────────────

/// A lightweight watch-folder session that polls a directory at a fixed
/// interval and reports newly detected source files.
///
/// This is intentionally synchronous so it can be driven from a background
/// thread or a polling loop without requiring an async runtime.
pub struct WatchFolderSession {
    /// Directory to poll for new source files.
    pub poll_dir: PathBuf,
    /// Polling interval in seconds.
    pub poll_interval_secs: u64,
    /// Known paths from the previous poll — used to detect new arrivals.
    known_paths: parking_lot::Mutex<std::collections::HashSet<PathBuf>>,
}

impl WatchFolderSession {
    /// Create a new `WatchFolderSession` for `poll_dir` with the given
    /// polling interval.
    ///
    /// The set of "known" paths is initialised to the files present in
    /// `poll_dir` at construction time so that pre-existing files are not
    /// reported as new on the first call to [`check_for_new_sources`].
    ///
    /// [`check_for_new_sources`]: WatchFolderSession::check_for_new_sources
    #[must_use]
    pub fn new(poll_dir: PathBuf, poll_interval_secs: u64) -> Self {
        let initial = collect_source_paths(&poll_dir);
        Self {
            poll_dir,
            poll_interval_secs,
            known_paths: parking_lot::Mutex::new(initial),
        }
    }

    /// Poll `poll_dir` once and return paths of **newly detected** source
    /// files (files present now that were not present on the previous call).
    ///
    /// After this call the internal set of known paths is updated to include
    /// all currently visible files.
    #[must_use]
    pub fn check_for_new_sources(&self) -> Vec<PathBuf> {
        let current = collect_source_paths(&self.poll_dir);

        let mut known = self.known_paths.lock();
        let new_paths: Vec<PathBuf> = current
            .iter()
            .filter(|p| !known.contains(*p))
            .cloned()
            .collect();

        // Update known set with everything we see now.
        *known = current;
        new_paths
    }
}

/// Collect all regular-file paths inside `dir` (non-recursive, depth=1).
fn collect_source_paths(dir: &Path) -> std::collections::HashSet<PathBuf> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let ft = entry.file_type().ok()?;
            if ft.is_file() {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect()
}

use crate::types::TrackType;

impl TrackType {
    #[must_use]
    pub(crate) const fn has_video(&self) -> bool {
        matches!(self, Self::Video | Self::AudioVideo)
    }

    #[must_use]
    pub(crate) const fn has_audio(&self) -> bool {
        matches!(self, Self::Audio | Self::AudioVideo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = ConformSession::new(
            "Test Session".to_string(),
            PathBuf::from("/test/timeline.edl"),
            vec![PathBuf::from("/test/media")],
            PathBuf::from("/test/output"),
            ConformConfig::default(),
        )
        .expect("test expectation failed");

        assert_eq!(session.name(), "Test Session");
        assert_eq!(session.status(), SessionStatus::Created);
    }

    #[test]
    fn test_status_changes() {
        let session = ConformSession::new(
            "Test".to_string(),
            PathBuf::from("/test/timeline.edl"),
            vec![],
            PathBuf::from("/test/output"),
            ConformConfig::default(),
        )
        .expect("test expectation failed");

        assert_eq!(session.status(), SessionStatus::Created);
        session.set_status(SessionStatus::Scanning);
        assert_eq!(session.status(), SessionStatus::Scanning);
    }

    // ── WatchFolderSession tests ──────────────────────────────────────────────

    #[test]
    fn test_watch_folder_no_new_files_initially() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir should be created");
        // Place a file before constructing the session
        std::fs::write(temp_dir.path().join("pre_existing.mov"), b"x")
            .expect("write should succeed");

        let session = WatchFolderSession::new(temp_dir.path().to_path_buf(), 5);
        // On first check, nothing should be new (pre-existing files are known)
        let new = session.check_for_new_sources();
        assert!(
            new.is_empty(),
            "pre-existing files must not be reported as new, got {new:?}"
        );
    }

    #[test]
    fn test_watch_folder_detects_new_file() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir should be created");

        let session = WatchFolderSession::new(temp_dir.path().to_path_buf(), 5);

        // No files yet — first check returns nothing
        assert!(session.check_for_new_sources().is_empty());

        // Add a new file
        let new_file = temp_dir.path().join("shot_001.mov");
        std::fs::write(&new_file, b"data").expect("write should succeed");

        let new = session.check_for_new_sources();
        assert_eq!(new.len(), 1, "new file should be detected");
        assert_eq!(new[0], new_file);
    }

    #[test]
    fn test_watch_folder_does_not_re_report_known_files() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir should be created");
        let session = WatchFolderSession::new(temp_dir.path().to_path_buf(), 5);

        // Introduce a file and consume the notification
        std::fs::write(temp_dir.path().join("clip.mov"), b"data").expect("write should succeed");
        let first = session.check_for_new_sources();
        assert_eq!(first.len(), 1);

        // Second call must not re-report the same file
        let second = session.check_for_new_sources();
        assert!(
            second.is_empty(),
            "already-reported file must not be reported again"
        );
    }

    #[test]
    fn test_watch_folder_detects_multiple_new_files() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir should be created");
        let session = WatchFolderSession::new(temp_dir.path().to_path_buf(), 1);

        // Baseline check
        assert!(session.check_for_new_sources().is_empty());

        // Add three files
        for i in 0..3u8 {
            std::fs::write(temp_dir.path().join(format!("clip_{i}.mov")), b"data")
                .expect("write should succeed");
        }

        let new = session.check_for_new_sources();
        assert_eq!(new.len(), 3, "all three new files should be detected");
    }
}
