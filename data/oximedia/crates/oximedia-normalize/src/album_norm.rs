//! Album and compilation loudness normalization.
//!
//! Implements album-mode normalization where all tracks are normalized relative
//! to the loudest track in the album, preserving the intended dynamic relationships
//! between tracks. This is distinct from per-track normalization, which adjusts
//! each track independently to the same absolute target.
//!
//! # Algorithm
//!
//! 1. **Measure**: Measure the integrated loudness of every track (ITU-R BS.1770 or RMS proxy).
//! 2. **Reference selection**: The loudest track determines the reference loudness.
//!    All other tracks are scaled so that loudness relationships are preserved.
//! 3. **Album target**: Optionally constrain the loudest track to an absolute album
//!    target (e.g. −14 LUFS for streaming).
//! 4. **Gain computation**: Each track receives a gain offset = `album_target - reference_lufs`,
//!    which is identical for all tracks, preserving the relative balance.
//! 5. **Per-track limits**: A configurable max gain / max attenuation limit prevents
//!    excessive correction on outlier tracks.
//!
//! # Example
//!
//! ```rust
//! use oximedia_normalize::album_norm::{AlbumNormConfig, AlbumNormSession};
//!
//! let cfg = AlbumNormConfig::streaming();
//! let mut session = AlbumNormSession::new(cfg);
//!
//! session.add_track("track01.wav", -18.5);
//! session.add_track("track02.wav", -20.1);
//! session.add_track("track03.wav", -17.3); // loudest
//!
//! let plan = session.compute_plan().expect("non-empty session");
//! assert_eq!(plan.track_gains.len(), 3);
//! // All tracks receive the same gain offset
//! let gains: Vec<f64> = plan.track_gains.iter().map(|tg| tg.gain_db).collect();
//! let diff = (gains[0] - gains[1]).abs();
//! assert!(diff < 1e-9, "all tracks must receive the same absolute gain offset");
//! ```

/// A single track entry in the album normalization session.
#[derive(Debug, Clone)]
pub struct AlbumTrackEntry {
    /// Track identifier (filename, ID, etc.).
    pub id: String,
    /// Measured integrated loudness in LUFS.
    pub measured_lufs: f64,
}

impl AlbumTrackEntry {
    /// Create a new track entry.
    pub fn new(id: impl Into<String>, measured_lufs: f64) -> Self {
        Self {
            id: id.into(),
            measured_lufs,
        }
    }
}

/// Configuration for album/compilation normalization.
#[derive(Debug, Clone)]
pub struct AlbumNormConfig {
    /// Target loudness for the album (LUFS).
    ///
    /// The loudest track will be normalized to this level; all other tracks
    /// receive the same gain offset to preserve relative loudness.
    pub album_target_lufs: f64,

    /// Maximum gain allowed per track in dB.
    pub max_gain_db: f64,

    /// Maximum attenuation allowed per track in dB (positive value).
    pub max_attenuation_db: f64,

    /// Tolerance: if the album reference is already within this LU range of
    /// the target, normalization is considered unnecessary.
    pub tolerance_lu: f64,

    /// If true, only attenuate — never boost any track.  Useful for limiting
    /// without risk of clipping.
    pub prevent_boost: bool,
}

impl AlbumNormConfig {
    /// Create a config targeting −14 LUFS (streaming standard).
    pub fn streaming() -> Self {
        Self {
            album_target_lufs: -14.0,
            max_gain_db: 20.0,
            max_attenuation_db: 30.0,
            tolerance_lu: 0.5,
            prevent_boost: false,
        }
    }

    /// Create a config targeting −23 LUFS (broadcast / EBU R128).
    pub fn broadcast() -> Self {
        Self {
            album_target_lufs: -23.0,
            max_gain_db: 15.0,
            max_attenuation_db: 30.0,
            tolerance_lu: 1.0,
            prevent_boost: false,
        }
    }

    /// Create a config targeting −16 LUFS (podcast / spoken word).
    pub fn podcast() -> Self {
        Self {
            album_target_lufs: -16.0,
            max_gain_db: 20.0,
            max_attenuation_db: 30.0,
            tolerance_lu: 0.5,
            prevent_boost: false,
        }
    }

    /// Create a custom configuration.
    pub fn custom(album_target_lufs: f64) -> Self {
        Self {
            album_target_lufs,
            max_gain_db: 20.0,
            max_attenuation_db: 30.0,
            tolerance_lu: 0.5,
            prevent_boost: false,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(-96.0..=0.0).contains(&self.album_target_lufs) {
            return Err(format!(
                "album_target_lufs must be in [-96, 0] LUFS, got {}",
                self.album_target_lufs
            ));
        }
        if self.max_gain_db <= 0.0 || self.max_gain_db > 60.0 {
            return Err(format!(
                "max_gain_db must be in (0, 60] dB, got {}",
                self.max_gain_db
            ));
        }
        if self.max_attenuation_db <= 0.0 || self.max_attenuation_db > 60.0 {
            return Err(format!(
                "max_attenuation_db must be in (0, 60] dB, got {}",
                self.max_attenuation_db
            ));
        }
        if self.tolerance_lu < 0.0 || self.tolerance_lu > 10.0 {
            return Err(format!(
                "tolerance_lu must be in [0, 10] LU, got {}",
                self.tolerance_lu
            ));
        }
        Ok(())
    }
}

/// Per-track gain assignment produced by the album normalization plan.
#[derive(Debug, Clone)]
pub struct TrackGainAssignment {
    /// Track identifier.
    pub id: String,
    /// Measured loudness (LUFS) for this track.
    pub measured_lufs: f64,
    /// Gain to apply in dB.
    pub gain_db: f64,
    /// Estimated output loudness (LUFS) after applying the gain.
    pub output_lufs: f64,
    /// Whether the gain was clamped due to max_gain_db / max_attenuation_db.
    pub was_clamped: bool,
}

impl TrackGainAssignment {
    /// True if the track requires a boost (positive gain).
    pub fn is_boost(&self) -> bool {
        self.gain_db > 0.0
    }

    /// True if the track requires attenuation.
    pub fn is_attenuation(&self) -> bool {
        self.gain_db < 0.0
    }
}

/// The computed album normalization plan.
#[derive(Debug, Clone)]
pub struct AlbumNormPlan {
    /// Uniform gain offset applied to all tracks (before per-track clamping).
    pub album_gain_db: f64,
    /// Reference loudness used for computing the album gain.
    /// This is the loudest track in the session.
    pub reference_lufs: f64,
    /// Target loudness for the album.
    pub target_lufs: f64,
    /// Per-track gain assignments.
    pub track_gains: Vec<TrackGainAssignment>,
    /// Whether all tracks are compliant (within tolerance of their expected output).
    pub all_compliant: bool,
}

impl AlbumNormPlan {
    /// Minimum gain across all tracks in dB.
    pub fn min_gain_db(&self) -> f64 {
        self.track_gains
            .iter()
            .map(|t| t.gain_db)
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum gain across all tracks in dB.
    pub fn max_gain_db(&self) -> f64 {
        self.track_gains
            .iter()
            .map(|t| t.gain_db)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Number of tracks with clamped gain.
    pub fn clamped_count(&self) -> usize {
        self.track_gains.iter().filter(|t| t.was_clamped).count()
    }
}

/// Album normalization session.
///
/// Accumulates track measurements, then computes a consistent gain plan
/// that preserves the loudness relationships between tracks.
pub struct AlbumNormSession {
    config: AlbumNormConfig,
    tracks: Vec<AlbumTrackEntry>,
}

impl AlbumNormSession {
    /// Create a new session with the given configuration.
    pub fn new(config: AlbumNormConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
        }
    }

    /// Add a track to the session.
    ///
    /// # Arguments
    /// * `id` – Track identifier (filename, UUID, etc.).
    /// * `measured_lufs` – Integrated loudness measured with ITU-R BS.1770-4.
    pub fn add_track(&mut self, id: impl Into<String>, measured_lufs: f64) {
        self.tracks.push(AlbumTrackEntry::new(id, measured_lufs));
    }

    /// Number of tracks added to this session.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Compute the album normalization plan.
    ///
    /// The plan is computed by:
    /// 1. Finding the loudest track (reference).
    /// 2. Computing `album_gain = target - reference_lufs`.
    /// 3. Applying `album_gain` to every track (preserves relative balance).
    /// 4. Clamping each track's gain individually.
    ///
    /// Returns `None` if no tracks have been added.
    pub fn compute_plan(&self) -> Option<AlbumNormPlan> {
        if self.tracks.is_empty() {
            return None;
        }

        // Find the loudest track
        let reference_lufs = self
            .tracks
            .iter()
            .map(|t| t.measured_lufs)
            .fold(f64::NEG_INFINITY, f64::max);

        let album_gain_db = self.config.album_target_lufs - reference_lufs;

        // Skip normalization if already within tolerance
        let effective_album_gain = if album_gain_db.abs() <= self.config.tolerance_lu {
            0.0
        } else {
            album_gain_db
        };

        let mut all_compliant = true;

        let track_gains: Vec<TrackGainAssignment> = self
            .tracks
            .iter()
            .map(|track| {
                let raw_gain = effective_album_gain;

                // Per-track clamping
                let (clamped_gain, was_clamped) = clamp_gain(
                    raw_gain,
                    self.config.max_gain_db,
                    self.config.max_attenuation_db,
                    self.config.prevent_boost,
                );

                let output_lufs = track.measured_lufs + clamped_gain;
                let expected_output = track.measured_lufs + effective_album_gain;
                let compliant =
                    (output_lufs - expected_output).abs() <= self.config.tolerance_lu + 0.1;
                if !compliant {
                    all_compliant = false;
                }

                TrackGainAssignment {
                    id: track.id.clone(),
                    measured_lufs: track.measured_lufs,
                    gain_db: clamped_gain,
                    output_lufs,
                    was_clamped,
                }
            })
            .collect();

        Some(AlbumNormPlan {
            album_gain_db: effective_album_gain,
            reference_lufs,
            target_lufs: self.config.album_target_lufs,
            track_gains,
            all_compliant,
        })
    }

    /// Reset the session, clearing all tracks.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Get the configuration.
    pub fn config(&self) -> &AlbumNormConfig {
        &self.config
    }

    /// Get a reference to the accumulated track entries.
    pub fn tracks(&self) -> &[AlbumTrackEntry] {
        &self.tracks
    }
}

/// Apply a computed gain plan to a slice of f32 samples in-place.
///
/// Uses batch (SIMD-friendly) gain application for performance.
///
/// # Arguments
/// * `samples` – Interleaved PCM samples.
/// * `gain_db` – Gain from a [`TrackGainAssignment`].
pub fn apply_album_gain(samples: &mut [f32], gain_db: f64) {
    if gain_db.abs() < 1e-9 {
        return;
    }
    let gain_linear = 10.0_f64.powf(gain_db / 20.0) as f32;
    crate::simd_gain::apply_gain_f32_inplace_batch(samples, gain_linear);
}

/// Apply album gain to f64 samples in-place.
///
/// Uses batch (SIMD-friendly) gain application for performance.
pub fn apply_album_gain_f64(samples: &mut [f64], gain_db: f64) {
    if gain_db.abs() < 1e-9 {
        return;
    }
    let gain_linear = 10.0_f64.powf(gain_db / 20.0);
    crate::simd_gain::apply_gain_f64_inplace_batch(samples, gain_linear);
}

/// Process an entire album: compute the plan and apply gain to each track's samples.
///
/// `tracks_samples` is a slice of mutable sample buffers, one per track (in the
/// same order as they were added to the session). Returns the plan.
///
/// Returns `None` if the session is empty.
pub fn process_album(
    session: &AlbumNormSession,
    tracks_samples: &mut [&mut [f32]],
) -> Option<AlbumNormPlan> {
    let plan = session.compute_plan()?;
    for (i, track_samples) in tracks_samples.iter_mut().enumerate() {
        if i < plan.track_gains.len() {
            apply_album_gain(track_samples, plan.track_gains[i].gain_db);
        }
    }
    Some(plan)
}

/// Clamp a gain value to the configured limits.
///
/// Returns `(clamped_gain, was_clamped)`.
fn clamp_gain(
    raw_gain: f64,
    max_gain_db: f64,
    max_attenuation_db: f64,
    prevent_boost: bool,
) -> (f64, bool) {
    if prevent_boost && raw_gain > 0.0 {
        return (0.0, true);
    }
    if raw_gain > max_gain_db {
        return (max_gain_db, true);
    }
    if raw_gain < -max_attenuation_db {
        return (-max_attenuation_db, true);
    }
    (raw_gain, false)
}

/// Statistics derived from an album normalization plan.
#[derive(Debug, Clone)]
pub struct AlbumNormStats {
    /// Number of tracks in the plan.
    pub track_count: usize,
    /// Reference loudness (loudest track) in LUFS.
    pub reference_lufs: f64,
    /// Album gain applied in dB.
    pub album_gain_db: f64,
    /// Loudness range across all tracks before normalization (max − min) in LU.
    pub loudness_range_lu: f64,
    /// Number of tracks where gain was clamped.
    pub clamped_count: usize,
}

impl AlbumNormStats {
    /// Compute statistics from a plan and the original tracks.
    pub fn from_plan(plan: &AlbumNormPlan, tracks: &[AlbumTrackEntry]) -> Self {
        let min_lufs = tracks
            .iter()
            .map(|t| t.measured_lufs)
            .fold(f64::INFINITY, f64::min);
        let max_lufs = tracks
            .iter()
            .map(|t| t.measured_lufs)
            .fold(f64::NEG_INFINITY, f64::max);

        Self {
            track_count: tracks.len(),
            reference_lufs: plan.reference_lufs,
            album_gain_db: plan.album_gain_db,
            loudness_range_lu: max_lufs - min_lufs,
            clamped_count: plan.clamped_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session_with_tracks(cfg: AlbumNormConfig, lufs_values: &[f64]) -> AlbumNormSession {
        let mut session = AlbumNormSession::new(cfg);
        for (i, &lufs) in lufs_values.iter().enumerate() {
            session.add_track(format!("track{:02}", i + 1), lufs);
        }
        session
    }

    // ─── AlbumNormConfig ────────────────────────────────────────────────────

    #[test]
    fn test_config_streaming_target() {
        let cfg = AlbumNormConfig::streaming();
        assert!((cfg.album_target_lufs - (-14.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_broadcast_target() {
        let cfg = AlbumNormConfig::broadcast();
        assert!((cfg.album_target_lufs - (-23.0)).abs() < f64::EPSILON);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_podcast_target() {
        let cfg = AlbumNormConfig::podcast();
        assert!((cfg.album_target_lufs - (-16.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_custom_target() {
        let cfg = AlbumNormConfig::custom(-18.0);
        assert!((cfg.album_target_lufs - (-18.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_validation_invalid_target() {
        let cfg = AlbumNormConfig::custom(5.0); // positive not allowed
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_gain() {
        let mut cfg = AlbumNormConfig::streaming();
        cfg.max_gain_db = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ─── AlbumNormSession ───────────────────────────────────────────────────

    #[test]
    fn test_session_empty_returns_none() {
        let session = AlbumNormSession::new(AlbumNormConfig::streaming());
        assert!(session.compute_plan().is_none());
    }

    #[test]
    fn test_session_track_count() {
        let mut session = AlbumNormSession::new(AlbumNormConfig::streaming());
        session.add_track("t1", -18.0);
        session.add_track("t2", -20.0);
        assert_eq!(session.track_count(), 2);
    }

    #[test]
    fn test_session_reset() {
        let mut session = AlbumNormSession::new(AlbumNormConfig::streaming());
        session.add_track("t1", -18.0);
        session.reset();
        assert_eq!(session.track_count(), 0);
    }

    // ─── Plan computation: uniform gain ────────────────────────────────────

    #[test]
    fn test_plan_uniform_gain_offset() {
        // All tracks should receive exactly the same gain (album gain).
        let cfg = AlbumNormConfig::streaming();
        let session = make_session_with_tracks(cfg, &[-18.5, -20.1, -17.3]);
        let plan = session.compute_plan().expect("non-empty session");

        let gains: Vec<f64> = plan.track_gains.iter().map(|tg| tg.gain_db).collect();
        for &g in &gains {
            assert!(
                (g - gains[0]).abs() < 1e-9,
                "all tracks must receive the same gain offset; got {:?}",
                gains
            );
        }
    }

    #[test]
    fn test_plan_reference_is_loudest_track() {
        let cfg = AlbumNormConfig::streaming();
        let session = make_session_with_tracks(cfg, &[-18.5, -17.3, -20.1]);
        let plan = session.compute_plan().expect("non-empty session");
        assert!((plan.reference_lufs - (-17.3)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_plan_album_gain_formula() {
        let cfg = AlbumNormConfig::streaming(); // target = -14 LUFS
        let session = make_session_with_tracks(cfg, &[-18.0, -20.0]);
        let plan = session.compute_plan().expect("non-empty session");
        // reference = -18.0, target = -14.0, gain = +4 dB
        assert!(
            (plan.album_gain_db - 4.0).abs() < 0.01,
            "expected album_gain = +4 dB, got {}",
            plan.album_gain_db
        );
    }

    #[test]
    fn test_plan_output_lufs_correct() {
        let cfg = AlbumNormConfig::custom(-16.0);
        // 3 tracks: -16.0, -18.0, -22.0
        let session = make_session_with_tracks(cfg, &[-16.0, -18.0, -22.0]);
        let plan = session.compute_plan().expect("non-empty session");
        // reference is -16.0, target is -16.0, so album_gain ≈ 0 (within tolerance)
        for tg in &plan.track_gains {
            let expected = tg.measured_lufs + plan.album_gain_db;
            assert!(
                (tg.output_lufs - expected).abs() < 0.01,
                "output LUFS mismatch for {}: expected {}, got {}",
                tg.id,
                expected,
                tg.output_lufs
            );
        }
    }

    #[test]
    fn test_plan_single_track() {
        let cfg = AlbumNormConfig::streaming(); // target = -14
        let mut session = AlbumNormSession::new(cfg);
        session.add_track("only", -20.0);
        let plan = session.compute_plan().expect("non-empty");
        assert!(
            (plan.album_gain_db - 6.0).abs() < 0.01,
            "expected +6 dB, got {}",
            plan.album_gain_db
        );
        assert_eq!(plan.track_gains.len(), 1);
    }

    #[test]
    fn test_plan_within_tolerance_noop() {
        let cfg = AlbumNormConfig::streaming(); // tolerance = 0.5 LU, target = -14
        let session = make_session_with_tracks(cfg, &[-14.3]); // within ±0.5 LU
        let plan = session.compute_plan().expect("non-empty");
        assert!(
            plan.album_gain_db.abs() < f64::EPSILON,
            "should be a no-op, got {} dB",
            plan.album_gain_db
        );
    }

    #[test]
    fn test_plan_clamped_gain() {
        let mut cfg = AlbumNormConfig::streaming();
        cfg.max_gain_db = 3.0; // cap at +3 dB
        let session = make_session_with_tracks(cfg, &[-20.0]); // needs +6 dB
        let plan = session.compute_plan().expect("non-empty");
        let tg = &plan.track_gains[0];
        assert!(
            (tg.gain_db - 3.0).abs() < 0.01,
            "expected clamped to +3 dB, got {}",
            tg.gain_db
        );
        assert!(tg.was_clamped);
    }

    #[test]
    fn test_plan_prevent_boost() {
        let mut cfg = AlbumNormConfig::streaming();
        cfg.prevent_boost = true;
        let session = make_session_with_tracks(cfg, &[-20.0]); // would need +6 dB
        let plan = session.compute_plan().expect("non-empty");
        let tg = &plan.track_gains[0];
        assert!(
            tg.gain_db <= 0.0,
            "boost prevented: gain should be ≤ 0, got {}",
            tg.gain_db
        );
        assert!(tg.was_clamped);
    }

    // ─── apply_album_gain ───────────────────────────────────────────────────

    #[test]
    fn test_apply_album_gain_zero_noop() {
        let original = vec![0.5_f32; 100];
        let mut samples = original.clone();
        apply_album_gain(&mut samples, 0.0);
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_album_gain_boost() {
        let mut samples = vec![0.5_f32; 100];
        apply_album_gain(&mut samples, 6.0); // +6 dB ≈ ×1.995
        let expected = 0.5 * 10.0_f32.powf(6.0 / 20.0);
        assert!(
            (samples[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            samples[0]
        );
    }

    #[test]
    fn test_apply_album_gain_cut() {
        let mut samples = vec![0.5_f32; 100];
        apply_album_gain(&mut samples, -6.0); // −6 dB ≈ ×0.501
        let expected = 0.5 * 10.0_f32.powf(-6.0 / 20.0);
        assert!((samples[0] - expected).abs() < 1e-5);
    }

    // ─── AlbumNormStats ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_loudness_range() {
        let cfg = AlbumNormConfig::streaming();
        let session = make_session_with_tracks(cfg.clone(), &[-15.0, -22.0, -18.0]);
        let plan = session.compute_plan().expect("non-empty");
        let stats = AlbumNormStats::from_plan(&plan, session.tracks());
        assert!(
            (stats.loudness_range_lu - 7.0).abs() < 0.01,
            "expected LRA = 7 LU, got {}",
            stats.loudness_range_lu
        );
    }

    #[test]
    fn test_stats_track_count() {
        let cfg = AlbumNormConfig::streaming();
        let session = make_session_with_tracks(cfg, &[-16.0, -18.0, -20.0, -22.0]);
        let plan = session.compute_plan().expect("non-empty");
        let stats = AlbumNormStats::from_plan(&plan, session.tracks());
        assert_eq!(stats.track_count, 4);
    }

    // ─── AlbumNormPlan helpers ───────────────────────────────────────────────

    #[test]
    fn test_plan_min_max_gain() {
        let mut cfg = AlbumNormConfig::streaming();
        cfg.max_gain_db = 3.0;
        let session = make_session_with_tracks(cfg, &[-20.0, -19.0]);
        let plan = session.compute_plan().expect("non-empty");
        assert!(
            (plan.max_gain_db() - plan.min_gain_db()).abs() < 1e-9,
            "all gains should be equal (same album gain)"
        );
    }

    // ─── apply_album_gain_f64 ────────────────────────────────────────────────

    #[test]
    fn test_apply_album_gain_f64_zero_noop() {
        let original = vec![0.5_f64; 100];
        let mut samples = original.clone();
        apply_album_gain_f64(&mut samples, 0.0);
        for (a, b) in samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_apply_album_gain_f64_boost() {
        let mut samples = vec![0.5_f64; 100];
        apply_album_gain_f64(&mut samples, 6.0);
        let expected = 0.5 * 10.0_f64.powf(6.0 / 20.0);
        assert!(
            (samples[0] - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            samples[0]
        );
    }

    // ─── process_album ───────────────────────────────────────────────────────

    #[test]
    fn test_process_album() {
        let cfg = AlbumNormConfig::streaming(); // target -14
        let mut session = AlbumNormSession::new(cfg);
        session.add_track("t1", -18.0);
        session.add_track("t2", -20.0);

        let mut track1_samples = vec![0.5_f32; 100];
        let mut track2_samples = vec![0.3_f32; 100];

        let plan = process_album(
            &session,
            &mut [&mut track1_samples[..], &mut track2_samples[..]],
        );

        assert!(plan.is_some());
        let plan = plan.expect("non-empty");

        // Both tracks should have the same gain applied
        let gain = plan.album_gain_db;
        let gain_linear = 10.0_f64.powf(gain / 20.0) as f32;

        assert!(
            (track1_samples[0] - 0.5 * gain_linear).abs() < 1e-5,
            "track1 gain mismatch"
        );
        assert!(
            (track2_samples[0] - 0.3 * gain_linear).abs() < 1e-5,
            "track2 gain mismatch"
        );
    }

    #[test]
    fn test_process_album_empty_session() {
        let cfg = AlbumNormConfig::streaming();
        let session = AlbumNormSession::new(cfg);
        let result = process_album(&session, &mut []);
        assert!(result.is_none());
    }
}
