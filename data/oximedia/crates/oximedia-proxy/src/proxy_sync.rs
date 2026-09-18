//! Proxy synchronization and database migration for OxiMedia proxy system.
//!
//! Provides proxy-to-original timecode alignment, sync verification,
//! drift detection, and workstation-to-workstation proxy database migration.

/// A timecode value expressed as total frames at a given frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameTimecode {
    /// Total frame count from zero.
    pub frame_number: u64,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
}

impl FrameTimecode {
    /// Create a new frame timecode.
    #[must_use]
    pub fn new(frame_number: u64, fps_num: u32, fps_den: u32) -> Self {
        Self {
            frame_number,
            fps_num,
            fps_den,
        }
    }

    /// Create a timecode from hours, minutes, seconds, frames at 24fps.
    #[must_use]
    pub fn from_hmsf(hours: u64, minutes: u64, seconds: u64, frames: u64, fps: u64) -> Self {
        let total_frames = ((hours * 3600 + minutes * 60 + seconds) * fps) + frames;
        Self {
            frame_number: total_frames,
            fps_num: fps as u32,
            fps_den: 1,
        }
    }

    /// Frame rate as a floating-point value.
    #[must_use]
    pub fn fps_f64(&self) -> f64 {
        self.fps_num as f64 / self.fps_den as f64
    }

    /// Duration in seconds.
    #[must_use]
    pub fn as_seconds(&self) -> f64 {
        self.frame_number as f64 / self.fps_f64()
    }

    /// Compute the difference in frames from another timecode.
    ///
    /// Returns a positive value if `self` is later than `other`.
    #[must_use]
    pub fn frame_diff(&self, other: &Self) -> i64 {
        self.frame_number as i64 - other.frame_number as i64
    }
}

/// A sync point linking a proxy frame to an original frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncPoint {
    /// Timecode in the proxy media.
    pub proxy_tc: FrameTimecode,
    /// Corresponding timecode in the original media.
    pub original_tc: FrameTimecode,
    /// Whether this sync point was verified by checksum comparison.
    pub verified: bool,
}

impl SyncPoint {
    /// Create a new sync point.
    #[must_use]
    pub fn new(proxy_tc: FrameTimecode, original_tc: FrameTimecode) -> Self {
        Self {
            proxy_tc,
            original_tc,
            verified: false,
        }
    }

    /// Mark this sync point as verified.
    #[must_use]
    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Frame offset between proxy and original (in proxy frames).
    #[must_use]
    pub fn frame_offset(&self) -> i64 {
        self.proxy_tc.frame_number as i64 - self.original_tc.frame_number as i64
    }
}

/// Result of a sync verification check.
#[derive(Debug, Clone)]
pub struct SyncVerificationResult {
    /// Clip or session identifier.
    pub clip_id: String,
    /// Whether the sync is valid within tolerance.
    pub in_sync: bool,
    /// Maximum frame drift detected.
    pub max_drift_frames: i64,
    /// Number of sync points checked.
    pub points_checked: usize,
    /// Number of sync points that passed verification.
    pub points_passed: usize,
}

impl SyncVerificationResult {
    /// Fraction of sync points that passed [0.0, 1.0].
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        if self.points_checked == 0 {
            return 1.0;
        }
        self.points_passed as f32 / self.points_checked as f32
    }
}

/// Tolerance settings for sync verification.
#[derive(Debug, Clone, Copy)]
pub struct SyncTolerance {
    /// Maximum allowed frame drift (inclusive).
    pub max_drift_frames: u32,
    /// Minimum fraction of sync points that must pass.
    pub min_pass_rate: f32,
}

impl SyncTolerance {
    /// Create a tolerance with given drift and pass rate.
    #[must_use]
    pub fn new(max_drift_frames: u32, min_pass_rate: f32) -> Self {
        Self {
            max_drift_frames,
            min_pass_rate: min_pass_rate.clamp(0.0, 1.0),
        }
    }

    /// Strict tolerance: zero drift allowed, 100% pass rate required.
    #[must_use]
    pub fn strict() -> Self {
        Self::new(0, 1.0)
    }

    /// Lenient tolerance: up to 2 frames drift, 90% pass rate.
    #[must_use]
    pub fn lenient() -> Self {
        Self::new(2, 0.9)
    }
}

impl Default for SyncTolerance {
    fn default() -> Self {
        Self::new(1, 0.95)
    }
}

/// Verifies proxy-to-original synchronization.
pub struct ProxySyncVerifier {
    /// Sync points to check.
    sync_points: Vec<SyncPoint>,
    /// Tolerance settings.
    tolerance: SyncTolerance,
}

impl ProxySyncVerifier {
    /// Create a new verifier with default tolerance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sync_points: Vec::new(),
            tolerance: SyncTolerance::default(),
        }
    }

    /// Set the tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: SyncTolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Add a sync point.
    pub fn add_sync_point(&mut self, point: SyncPoint) {
        self.sync_points.push(point);
    }

    /// Run verification and return the result for the given clip id.
    #[must_use]
    pub fn verify(&self, clip_id: impl Into<String>) -> SyncVerificationResult {
        let clip_id = clip_id.into();
        if self.sync_points.is_empty() {
            return SyncVerificationResult {
                clip_id,
                in_sync: true,
                max_drift_frames: 0,
                points_checked: 0,
                points_passed: 0,
            };
        }

        let mut max_drift = 0i64;
        let mut passed = 0usize;

        for sp in &self.sync_points {
            let drift = sp.frame_offset().abs();
            if drift > max_drift {
                max_drift = drift;
            }
            if drift <= self.tolerance.max_drift_frames as i64 {
                passed += 1;
            }
        }

        let total = self.sync_points.len();
        let pass_rate = passed as f32 / total as f32;

        SyncVerificationResult {
            clip_id,
            in_sync: pass_rate >= self.tolerance.min_pass_rate,
            max_drift_frames: max_drift,
            points_checked: total,
            points_passed: passed,
        }
    }

    /// Number of sync points registered.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.sync_points.len()
    }
}

impl Default for ProxySyncVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Proxy database migration — cross-workstation export / import / rebase
// ─────────────────────────────────────────────────────────────────────────────

/// A portable entry describing one proxy–source relationship, suitable for
/// serialisation and cross-workstation transfer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MigrationProxyEntry {
    /// Path to the source (original high-resolution) media file.
    /// May be absolute or relative to `ProxyDbExport::root_prefix`.
    pub source_path: std::path::PathBuf,
    /// Path to the proxy file.
    /// May be absolute or relative to `ProxyDbExport::root_prefix`.
    pub proxy_path: std::path::PathBuf,
    /// Optional content-addressable identifier (e.g. fingerprint hash).
    pub media_id: Option<String>,
}

/// A portable snapshot of a proxy database suitable for workstation transfer.
///
/// Serialise with `serde_json` (or any `serde`-compatible format), copy to the
/// target workstation, then call [`ProxyDbExport::import_with_rebase`] to
/// re-root all paths.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProxyDbExport {
    /// All proxy entries captured at export time.
    pub entries: Vec<MigrationProxyEntry>,
    /// Absolute path prefix that all entries were relative to on the source
    /// workstation.  Used by `import_with_rebase` to strip the old prefix.
    pub root_prefix: std::path::PathBuf,
    /// Unix timestamp (seconds since epoch) recording when this export was
    /// created.
    pub created_at: u64,
}

impl ProxyDbExport {
    /// Create a new export snapshot.
    ///
    /// `entries` should contain all proxy entries to transfer.
    /// `root_prefix` is the absolute root directory these paths belong to on
    /// the current workstation (e.g. `/Volumes/Media`).
    #[must_use]
    pub fn new(entries: Vec<MigrationProxyEntry>, root_prefix: std::path::PathBuf) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            entries,
            root_prefix,
            created_at,
        }
    }

    /// Import entries from this export, rebasing every path from
    /// `self.root_prefix` to `new_root`.
    ///
    /// For each entry the method:
    /// 1. Attempts to strip `self.root_prefix` from `source_path` and
    ///    `proxy_path`.
    /// 2. If strippable, prepends `new_root` to produce the relocated path.
    ///    If the path is not relative to `root_prefix` it is kept as-is.
    /// 3. Checks whether the rebased `proxy_path` exists on disk.
    ///
    /// Returns the list of rebased entries and a [`RebaseResult`] summary.
    pub fn import_with_rebase(
        &self,
        new_root: &std::path::Path,
    ) -> (Vec<MigrationProxyEntry>, RebaseResult) {
        let mut rebased_entries: Vec<MigrationProxyEntry> = Vec::with_capacity(self.entries.len());
        let mut entries_relinked: usize = 0;
        let mut entries_missing: Vec<std::path::PathBuf> = Vec::new();

        for entry in &self.entries {
            let new_source = rebase_path(&entry.source_path, &self.root_prefix, new_root);
            let new_proxy = rebase_path(&entry.proxy_path, &self.root_prefix, new_root);

            let exists = new_proxy.exists();
            if exists {
                entries_relinked += 1;
            } else {
                entries_missing.push(new_proxy.clone());
            }

            rebased_entries.push(MigrationProxyEntry {
                source_path: new_source,
                proxy_path: new_proxy,
                media_id: entry.media_id.clone(),
            });
        }

        let result = RebaseResult {
            entries_relinked,
            entries_missing,
        };
        (rebased_entries, result)
    }
}

/// Rebase `path` from `old_root` to `new_root`.
///
/// If `path` starts with `old_root`, the `old_root` prefix is stripped and
/// `new_root` is prepended.  Otherwise the path is returned unchanged.
fn rebase_path(
    path: &std::path::Path,
    old_root: &std::path::Path,
    new_root: &std::path::Path,
) -> std::path::PathBuf {
    match path.strip_prefix(old_root) {
        Ok(relative) => new_root.join(relative),
        Err(_) => path.to_path_buf(),
    }
}

/// Summary produced by [`ProxyDbExport::import_with_rebase`].
#[derive(Debug, Clone)]
pub struct RebaseResult {
    /// Number of entries whose rebased proxy file was found on disk.
    pub entries_relinked: usize,
    /// Rebased proxy paths that do **not** exist on disk.
    pub entries_missing: Vec<std::path::PathBuf>,
}

/// Aligns proxy timecode to original timecode using a known offset.
pub struct TimecodeAligner {
    /// Known frame offset (proxy_frame = original_frame + offset).
    offset_frames: i64,
    /// Frame rate used for alignment.
    fps_num: u32,
    /// Frame rate denominator.
    fps_den: u32,
}

impl TimecodeAligner {
    /// Create an aligner with a known offset.
    #[must_use]
    pub fn new(offset_frames: i64, fps_num: u32, fps_den: u32) -> Self {
        Self {
            offset_frames,
            fps_num,
            fps_den,
        }
    }

    /// Create an aligner with zero offset.
    #[must_use]
    pub fn zero(fps_num: u32, fps_den: u32) -> Self {
        Self::new(0, fps_num, fps_den)
    }

    /// Compute the proxy frame number for a given original frame number.
    #[must_use]
    pub fn original_to_proxy(&self, original_frame: u64) -> u64 {
        (original_frame as i64 + self.offset_frames).max(0) as u64
    }

    /// Compute the original frame number for a given proxy frame number.
    #[must_use]
    pub fn proxy_to_original(&self, proxy_frame: u64) -> u64 {
        (proxy_frame as i64 - self.offset_frames).max(0) as u64
    }

    /// Get the frame offset.
    #[must_use]
    pub fn offset_frames(&self) -> i64 {
        self.offset_frames
    }

    /// Offset expressed in seconds.
    #[must_use]
    pub fn offset_seconds(&self) -> f64 {
        self.offset_frames as f64 / (self.fps_num as f64 / self.fps_den as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_timecode_fps_f64() {
        let tc = FrameTimecode::new(0, 24, 1);
        assert!((tc.fps_f64() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_frame_timecode_from_hmsf() {
        // 1 hour at 24fps = 86400 frames
        let tc = FrameTimecode::from_hmsf(1, 0, 0, 0, 24);
        assert_eq!(tc.frame_number, 86400);
    }

    #[test]
    fn test_frame_timecode_as_seconds() {
        let tc = FrameTimecode::new(240, 24, 1);
        assert!((tc.as_seconds() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_frame_timecode_frame_diff() {
        let tc1 = FrameTimecode::new(100, 24, 1);
        let tc2 = FrameTimecode::new(95, 24, 1);
        assert_eq!(tc1.frame_diff(&tc2), 5);
        assert_eq!(tc2.frame_diff(&tc1), -5);
    }

    #[test]
    fn test_sync_point_frame_offset_zero() {
        let tc = FrameTimecode::new(100, 24, 1);
        let sp = SyncPoint::new(tc, tc);
        assert_eq!(sp.frame_offset(), 0);
    }

    #[test]
    fn test_sync_point_frame_offset_nonzero() {
        let proxy_tc = FrameTimecode::new(105, 24, 1);
        let orig_tc = FrameTimecode::new(100, 24, 1);
        let sp = SyncPoint::new(proxy_tc, orig_tc);
        assert_eq!(sp.frame_offset(), 5);
    }

    #[test]
    fn test_sync_point_verified() {
        let tc = FrameTimecode::new(100, 24, 1);
        let sp = SyncPoint::new(tc, tc).verified();
        assert!(sp.verified);
    }

    #[test]
    fn test_sync_tolerance_default() {
        let t = SyncTolerance::default();
        assert_eq!(t.max_drift_frames, 1);
        assert!((t.min_pass_rate - 0.95).abs() < 1e-5);
    }

    #[test]
    fn test_sync_tolerance_strict() {
        let t = SyncTolerance::strict();
        assert_eq!(t.max_drift_frames, 0);
        assert!((t.min_pass_rate - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_proxy_sync_verifier_empty() {
        let verifier = ProxySyncVerifier::new();
        let result = verifier.verify("clip001");
        assert!(result.in_sync);
        assert_eq!(result.points_checked, 0);
    }

    #[test]
    fn test_proxy_sync_verifier_all_in_sync() {
        let mut verifier = ProxySyncVerifier::new();
        let tc = FrameTimecode::new(100, 24, 1);
        verifier.add_sync_point(SyncPoint::new(tc, tc));
        verifier.add_sync_point(SyncPoint::new(
            FrameTimecode::new(200, 24, 1),
            FrameTimecode::new(200, 24, 1),
        ));
        let result = verifier.verify("clip001");
        assert!(result.in_sync);
        assert_eq!(result.max_drift_frames, 0);
    }

    #[test]
    fn test_proxy_sync_verifier_drift_exceeds_tolerance() {
        let mut verifier = ProxySyncVerifier::new().with_tolerance(SyncTolerance::strict());
        verifier.add_sync_point(SyncPoint::new(
            FrameTimecode::new(105, 24, 1),
            FrameTimecode::new(100, 24, 1),
        ));
        let result = verifier.verify("clip001");
        assert!(!result.in_sync);
        assert_eq!(result.max_drift_frames, 5);
    }

    #[test]
    fn test_timecode_aligner_original_to_proxy() {
        let aligner = TimecodeAligner::new(10, 24, 1);
        assert_eq!(aligner.original_to_proxy(100), 110);
    }

    #[test]
    fn test_timecode_aligner_proxy_to_original() {
        let aligner = TimecodeAligner::new(10, 24, 1);
        assert_eq!(aligner.proxy_to_original(110), 100);
    }

    #[test]
    fn test_timecode_aligner_zero() {
        let aligner = TimecodeAligner::zero(25, 1);
        assert_eq!(aligner.original_to_proxy(500), 500);
        assert_eq!(aligner.proxy_to_original(500), 500);
    }

    #[test]
    fn test_timecode_aligner_offset_seconds() {
        let aligner = TimecodeAligner::new(48, 24, 1); // 2 seconds at 24fps
        assert!((aligner.offset_seconds() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sync_verification_pass_rate() {
        let result = SyncVerificationResult {
            clip_id: "c1".to_string(),
            in_sync: true,
            max_drift_frames: 0,
            points_checked: 10,
            points_passed: 9,
        };
        assert!((result.pass_rate() - 0.9).abs() < 1e-5);
    }

    // ── proxy database migration tests ────────────────────────────────────────

    fn make_entry(src: &str, proxy: &str) -> MigrationProxyEntry {
        MigrationProxyEntry {
            source_path: std::path::PathBuf::from(src),
            proxy_path: std::path::PathBuf::from(proxy),
            media_id: None,
        }
    }

    /// Round-trip: export and then import with the same root; entries preserved.
    #[test]
    fn test_proxy_export_import_round_trip() {
        let root = std::path::PathBuf::from("/media/root");
        let entry = MigrationProxyEntry {
            source_path: root.join("originals/clip001.mov"),
            proxy_path: root.join("proxies/clip001_proxy.mp4"),
            media_id: Some("abc123".to_string()),
        };
        let export = ProxyDbExport::new(vec![entry.clone()], root.clone());

        let (rebased, result) = export.import_with_rebase(&root);

        assert_eq!(rebased.len(), 1);
        assert_eq!(rebased[0].source_path, entry.source_path);
        assert_eq!(rebased[0].proxy_path, entry.proxy_path);
        assert_eq!(rebased[0].media_id, entry.media_id);
        // No real files exist; just verifying the structural round-trip.
        let _ = result; // result.entries_relinked is 0 (no real file on disk)
    }

    /// Rebase rewrites /old/root/* → /new/root/*.
    #[test]
    fn test_proxy_rebase_rewrites_root() {
        let old_root = std::path::PathBuf::from("/old/root");
        let new_root = std::path::PathBuf::from("/new/root");

        let entries = vec![make_entry(
            "/old/root/originals/clip.mov",
            "/old/root/proxies/clip_proxy.mp4",
        )];
        let export = ProxyDbExport::new(entries, old_root.clone());

        let (rebased, _result) = export.import_with_rebase(&new_root);

        assert_eq!(rebased.len(), 1);
        assert_eq!(
            rebased[0].source_path,
            std::path::PathBuf::from("/new/root/originals/clip.mov")
        );
        assert_eq!(
            rebased[0].proxy_path,
            std::path::PathBuf::from("/new/root/proxies/clip_proxy.mp4")
        );
    }

    /// Paths outside root_prefix are kept unchanged.
    #[test]
    fn test_proxy_rebase_non_prefix_path_unchanged() {
        let old_root = std::path::PathBuf::from("/old/root");
        let new_root = std::path::PathBuf::from("/new/root");

        // source_path is NOT under old_root
        let entries = vec![MigrationProxyEntry {
            source_path: std::path::PathBuf::from("/elsewhere/clip.mov"),
            proxy_path: std::path::PathBuf::from("/old/root/proxies/clip_proxy.mp4"),
            media_id: None,
        }];
        let export = ProxyDbExport::new(entries, old_root);
        let (rebased, _) = export.import_with_rebase(&new_root);

        // source_path should be unchanged (not under old_root)
        assert_eq!(
            rebased[0].source_path,
            std::path::PathBuf::from("/elsewhere/clip.mov")
        );
        // proxy_path should be rebased
        assert_eq!(
            rebased[0].proxy_path,
            std::path::PathBuf::from("/new/root/proxies/clip_proxy.mp4")
        );
    }

    /// import_with_rebase reports missing files; uses temp_dir for a real file.
    #[test]
    fn test_proxy_rebase_reports_missing() {
        let tmp = std::env::temp_dir();

        // Create a real proxy file that will be "found".
        let real_proxy_name = format!(
            "proxy_rebase_test_{}.mp4",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        );
        let real_proxy_src_path = tmp.join("proxy_rebase_src").join(&real_proxy_name);
        let real_proxy_dst_path = tmp.join("proxy_rebase_dst").join(&real_proxy_name);

        // We export from "src" root and import into "dst" root.
        let src_root = tmp.join("proxy_rebase_src");
        let dst_root = tmp.join("proxy_rebase_dst");

        std::fs::create_dir_all(&src_root).expect("create src dir");
        std::fs::create_dir_all(&dst_root).expect("create dst dir");

        // Create the real proxy at the *destination* path so one entry can be
        // relinked.
        std::fs::write(&real_proxy_dst_path, b"proxy data").expect("write proxy");

        // One entry that *will* be found (proxy exists at new root)
        let relinked_entry = MigrationProxyEntry {
            source_path: src_root.join("original.mov"),
            proxy_path: real_proxy_src_path.clone(),
            media_id: None,
        };
        // One entry that *will not* be found (no file at new root)
        let missing_entry = MigrationProxyEntry {
            source_path: src_root.join("missing_original.mov"),
            proxy_path: src_root.join("missing_proxy.mp4"),
            media_id: None,
        };

        let export = ProxyDbExport::new(vec![relinked_entry, missing_entry], src_root.clone());
        let (rebased, result) = export.import_with_rebase(&dst_root);

        assert_eq!(rebased.len(), 2);
        assert_eq!(result.entries_relinked, 1);
        assert_eq!(result.entries_missing.len(), 1);
        assert!(result.entries_missing[0]
            .to_string_lossy()
            .contains("missing_proxy.mp4"));

        // Clean up
        let _ = std::fs::remove_file(&real_proxy_dst_path);
        let _ = std::fs::remove_dir(&dst_root);
        let _ = std::fs::remove_dir(&src_root);
    }

    /// Simulates a network interruption during a proxy-database migration:
    /// the first `import_with_rebase` pass only sees *some* of the proxy
    /// files at the destination (as if the transfer was cut off partway
    /// through copying), reports the rest as missing, and then — once the
    /// remaining files "arrive" (simulating the transfer resuming) — a
    /// second `import_with_rebase` pass against the same export/new_root
    /// resolves every previously-missing entry. This is the real resume
    /// mechanism the `proxy_sync` migration API provides: re-running
    /// `import_with_rebase` is idempotent and simply re-checks disk state,
    /// so nothing needs to track "what already succeeded" — a caller can
    /// safely retry after a truncated transfer completes.
    #[test]
    fn test_proxy_sync_resume_after_truncated_transfer() {
        let tmp = std::env::temp_dir();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);

        let src_root = tmp.join(format!("proxy_resume_src_{unique}"));
        let dst_root = tmp.join(format!("proxy_resume_dst_{unique}"));
        std::fs::create_dir_all(&src_root).expect("create src dir");
        std::fs::create_dir_all(&dst_root).expect("create dst dir");

        // Five proxy entries in the export snapshot, as if migrating a whole
        // project's proxy database to a new workstation.
        let entries: Vec<MigrationProxyEntry> = (0..5)
            .map(|i| MigrationProxyEntry {
                source_path: src_root.join(format!("clip{i:02}.mov")),
                proxy_path: src_root.join(format!("clip{i:02}_proxy.mp4")),
                media_id: Some(format!("media-{i:02}")),
            })
            .collect();
        let export = ProxyDbExport::new(entries, src_root.clone());

        // ── "Network interruption": only the first 2 of 5 proxy files made
        // it across before the transfer was cut off. ───────────────────────
        for i in 0..2 {
            let dst_path = dst_root.join(format!("clip{i:02}_proxy.mp4"));
            std::fs::write(&dst_path, b"partial transfer data").expect("write partial proxy");
        }

        let (_rebased_partial, partial_result) = export.import_with_rebase(&dst_root);
        assert_eq!(
            partial_result.entries_relinked, 2,
            "only the 2 files that survived the truncated transfer should be found"
        );
        assert_eq!(
            partial_result.entries_missing.len(),
            3,
            "the 3 files that never arrived must be reported missing"
        );

        // ── "Resume": the remaining 3 proxy files finish copying. ──────────
        for i in 2..5 {
            let dst_path = dst_root.join(format!("clip{i:02}_proxy.mp4"));
            std::fs::write(&dst_path, b"resumed transfer data").expect("write resumed proxy");
        }

        // Re-running import_with_rebase against the identical export/new_root
        // is the resume operation: it is idempotent and now finds everything.
        let (rebased_full, resumed_result) = export.import_with_rebase(&dst_root);
        assert_eq!(
            resumed_result.entries_relinked, 5,
            "after resume, all 5 entries must be relinked"
        );
        assert!(
            resumed_result.entries_missing.is_empty(),
            "after resume, no entries should remain missing, got: {:?}",
            resumed_result.entries_missing
        );
        assert_eq!(rebased_full.len(), 5);
        for (i, entry) in rebased_full.iter().enumerate() {
            assert_eq!(
                entry.proxy_path,
                dst_root.join(format!("clip{i:02}_proxy.mp4"))
            );
            assert!(
                entry.proxy_path.exists(),
                "resumed proxy file must exist on disk: {}",
                entry.proxy_path.display()
            );
        }

        // Clean up.
        for i in 0..5 {
            let _ = std::fs::remove_file(dst_root.join(format!("clip{i:02}_proxy.mp4")));
        }
        let _ = std::fs::remove_dir(&dst_root);
        let _ = std::fs::remove_dir(&src_root);
    }
}
