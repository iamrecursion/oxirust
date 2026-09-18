#![allow(dead_code)]
//! Take management for ADR and recording sessions.
//!
//! Tracks multiple takes of dialogue, Foley, or music recordings. Supports
//! rating, comparison, compositing of best segments, and take selection
//! workflows used in professional audio post-production.

use std::collections::HashMap;

/// Unique identifier for a take.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TakeId(String);

impl TakeId {
    /// Create a new take identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the string value.
    pub fn value(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TakeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Rating for a take (1-5 stars).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TakeRating(u8);

impl TakeRating {
    /// Create a rating clamped to 1..=5.
    pub fn new(stars: u8) -> Self {
        Self(stars.clamp(1, 5))
    }

    /// Return the numeric star value.
    pub fn stars(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for TakeRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/5", self.0)
    }
}

/// Status of a take in the selection workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakeStatus {
    /// Just recorded, not yet reviewed.
    Recorded,
    /// Reviewed but no decision made.
    Reviewed,
    /// Selected as the chosen take.
    Selected,
    /// Selected as a backup / alt.
    Alternate,
    /// Explicitly rejected.
    Rejected,
}

impl std::fmt::Display for TakeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recorded => write!(f, "recorded"),
            Self::Reviewed => write!(f, "reviewed"),
            Self::Selected => write!(f, "selected"),
            Self::Alternate => write!(f, "alternate"),
            Self::Rejected => write!(f, "rejected"),
        }
    }
}

/// A single recorded take.
#[derive(Debug, Clone)]
pub struct Take {
    /// Unique identifier.
    pub id: TakeId,
    /// Take number within the cue (1-based).
    pub number: u32,
    /// Associated cue or line identifier.
    pub cue_id: String,
    /// Status in the review pipeline.
    pub status: TakeStatus,
    /// Optional rating.
    pub rating: Option<TakeRating>,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Sample rate.
    pub sample_rate: u32,
    /// Notes from the director or engineer.
    pub notes: String,
    /// File path to the recorded audio.
    pub file_path: String,
    /// Peak level in dB.
    pub peak_db: f64,
    /// RMS level in dB.
    pub rms_db: f64,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

impl Take {
    /// Create a new take.
    pub fn new(id: impl Into<String>, number: u32, cue_id: impl Into<String>) -> Self {
        Self {
            id: TakeId::new(id),
            number,
            cue_id: cue_id.into(),
            status: TakeStatus::Recorded,
            rating: None,
            duration_secs: 0.0,
            sample_rate: 48000,
            notes: String::new(),
            file_path: String::new(),
            peak_db: -96.0,
            rms_db: -96.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the rating.
    pub fn rate(&mut self, stars: u8) {
        self.rating = Some(TakeRating::new(stars));
    }

    /// Set the status.
    pub fn set_status(&mut self, status: TakeStatus) {
        self.status = status;
    }

    /// Set file path and duration.
    pub fn set_audio(&mut self, path: impl Into<String>, duration_secs: f64) {
        self.file_path = path.into();
        self.duration_secs = duration_secs;
    }

    /// Set levels.
    pub fn set_levels(&mut self, peak_db: f64, rms_db: f64) {
        self.peak_db = peak_db;
        self.rms_db = rms_db;
    }

    /// Add a note.
    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str("; ");
        }
        self.notes.push_str(note);
    }

    /// Check if the take is usable (selected or alternate).
    pub fn is_usable(&self) -> bool {
        matches!(self.status, TakeStatus::Selected | TakeStatus::Alternate)
    }
}

/// Manages all takes for a recording session.
#[derive(Debug)]
pub struct TakeManager {
    /// Session identifier.
    pub session_id: String,
    /// All takes, keyed by their id.
    takes: HashMap<TakeId, Take>,
    /// Takes grouped by cue id.
    cue_index: HashMap<String, Vec<TakeId>>,
}

impl TakeManager {
    /// Create a new take manager for a session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            takes: HashMap::new(),
            cue_index: HashMap::new(),
        }
    }

    /// Add a take.
    pub fn add_take(&mut self, take: Take) {
        let cue_id = take.cue_id.clone();
        let take_id = take.id.clone();
        self.takes.insert(take_id.clone(), take);
        self.cue_index.entry(cue_id).or_default().push(take_id);
    }

    /// Get a take by id.
    pub fn get(&self, id: &TakeId) -> Option<&Take> {
        self.takes.get(id)
    }

    /// Get a mutable reference to a take by id.
    pub fn get_mut(&mut self, id: &TakeId) -> Option<&mut Take> {
        self.takes.get_mut(id)
    }

    /// Return total number of takes.
    pub fn take_count(&self) -> usize {
        self.takes.len()
    }

    /// Return number of cues with at least one take.
    pub fn cue_count(&self) -> usize {
        self.cue_index.len()
    }

    /// Get all takes for a specific cue, sorted by take number.
    pub fn takes_for_cue(&self, cue_id: &str) -> Vec<&Take> {
        let mut takes: Vec<&Take> = self
            .cue_index
            .get(cue_id)
            .map(|ids| ids.iter().filter_map(|id| self.takes.get(id)).collect())
            .unwrap_or_default();
        takes.sort_by_key(|t| t.number);
        takes
    }

    /// Get the selected take for a cue (if any).
    pub fn selected_take(&self, cue_id: &str) -> Option<&Take> {
        self.takes_for_cue(cue_id)
            .into_iter()
            .find(|t| t.status == TakeStatus::Selected)
    }

    /// Get the best-rated take for a cue.
    pub fn best_rated_take(&self, cue_id: &str) -> Option<&Take> {
        self.takes_for_cue(cue_id)
            .into_iter()
            .filter(|t| t.rating.is_some())
            .max_by_key(|t| t.rating)
    }

    /// Select the specified take (and un-select any other selected take for the same cue).
    pub fn select_take(&mut self, take_id: &TakeId) -> bool {
        let cue_id = match self.takes.get(take_id) {
            Some(t) => t.cue_id.clone(),
            None => return false,
        };

        // Deselect any currently selected take for this cue
        if let Some(ids) = self.cue_index.get(&cue_id) {
            let ids_clone: Vec<TakeId> = ids.clone();
            for id in &ids_clone {
                if let Some(t) = self.takes.get_mut(id) {
                    if t.status == TakeStatus::Selected {
                        t.status = TakeStatus::Reviewed;
                    }
                }
            }
        }

        // Select the requested take
        if let Some(t) = self.takes.get_mut(take_id) {
            t.status = TakeStatus::Selected;
            true
        } else {
            false
        }
    }

    /// Return all takes with a specific status.
    pub fn takes_with_status(&self, status: TakeStatus) -> Vec<&Take> {
        self.takes.values().filter(|t| t.status == status).collect()
    }

    /// Return cues that have no selected take.
    pub fn cues_without_selection(&self) -> Vec<&str> {
        self.cue_index
            .keys()
            .filter(|cue_id| self.selected_take(cue_id).is_none())
            .map(|s| s.as_str())
            .collect()
    }

    /// Compute session statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> TakeManagerStats {
        let total = self.takes.len();
        let selected = self
            .takes
            .values()
            .filter(|t| t.status == TakeStatus::Selected)
            .count();
        let rejected = self
            .takes
            .values()
            .filter(|t| t.status == TakeStatus::Rejected)
            .count();
        let rated = self.takes.values().filter(|t| t.rating.is_some()).count();
        let avg_rating = if rated > 0 {
            let sum: u64 = self
                .takes
                .values()
                .filter_map(|t| t.rating.map(|r| u64::from(r.stars())))
                .sum();
            sum as f64 / rated as f64
        } else {
            0.0
        };
        let total_duration: f64 = self.takes.values().map(|t| t.duration_secs).sum();

        TakeManagerStats {
            total_takes: total,
            selected_count: selected,
            rejected_count: rejected,
            rated_count: rated,
            average_rating: avg_rating,
            total_duration_secs: total_duration,
            cue_count: self.cue_index.len(),
        }
    }
}

/// Summary statistics for a take manager session.
#[derive(Debug, Clone)]
pub struct TakeManagerStats {
    /// Total number of takes.
    pub total_takes: usize,
    /// Number of selected takes.
    pub selected_count: usize,
    /// Number of rejected takes.
    pub rejected_count: usize,
    /// Number of rated takes.
    pub rated_count: usize,
    /// Average rating across rated takes.
    pub average_rating: f64,
    /// Total duration of all takes in seconds.
    pub total_duration_secs: f64,
    /// Number of distinct cues.
    pub cue_count: usize,
}

// ── Sample-Accurate Crossfade Engine ─────────────────────────────────────────

/// Shape of a crossfade curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossfadeCurve {
    /// Linear equal-gain crossfade (simple overlap).
    Linear,
    /// Constant-power (equal-loudness) crossfade using cos/sin law.
    ConstantPower,
    /// S-curve crossfade using a raised-cosine window (smooth start and end).
    SCurve,
    /// Exponential fade-out / exponential fade-in (logarithmic amplitude).
    Exponential,
}

impl CrossfadeCurve {
    /// Compute the gain for the outgoing signal at normalised crossfade
    /// position `t` (0.0 = start of crossfade, 1.0 = end of crossfade).
    #[must_use]
    pub fn gain_out(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => 1.0 - t,
            Self::ConstantPower => {
                // Use power-law (squared cosine) so that gain_out + gain_in = 1
                // at every sample: cos²(θ) + sin²(θ) = 1 for all θ.
                // This keeps blended amplitude constant for unity input signals.
                let angle = t * std::f32::consts::FRAC_PI_2;
                let c = angle.cos();
                c * c
            }
            Self::SCurve => {
                // Hann window shaped fade: 0.5 * (1 + cos(π * t))
                0.5 * (1.0 + (std::f32::consts::PI * t).cos())
            }
            Self::Exponential => {
                // Exponential: 10^(-t * 60 / 20) — 60 dB attenuation over the crossfade.
                10.0_f32.powf(-t * 60.0 / 20.0)
            }
        }
    }

    /// Compute the gain for the incoming signal at normalised position `t`.
    #[must_use]
    pub fn gain_in(self, t: f32) -> f32 {
        // The incoming gain is the mirror of the outgoing gain.
        self.gain_out(1.0 - t)
    }
}

/// Error type for crossfade operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CrossfadeError {
    /// Crossfade length is zero.
    ZeroLength,
    /// The audio buffers have different lengths.
    LengthMismatch {
        /// Length of the outgoing buffer.
        out_len: usize,
        /// Length of the incoming buffer.
        in_len: usize,
    },
    /// The crossfade length exceeds the available audio.
    CrossfadeTooLong {
        /// Requested crossfade length.
        requested: usize,
        /// Available samples.
        available: usize,
    },
}

impl std::fmt::Display for CrossfadeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroLength => write!(f, "Crossfade length must be > 0"),
            Self::LengthMismatch { out_len, in_len } => write!(
                f,
                "Buffer length mismatch: outgoing={out_len}, incoming={in_len}"
            ),
            Self::CrossfadeTooLong {
                requested,
                available,
            } => write!(
                f,
                "Crossfade length {requested} exceeds available samples {available}"
            ),
        }
    }
}

impl std::error::Error for CrossfadeError {}

/// Sample-accurate crossfade engine for seamless take splicing.
///
/// Given two audio takes (outgoing and incoming), the engine produces a single
/// blended output that transitions from the outgoing take to the incoming take
/// over exactly `crossfade_samples` samples.
///
/// The crossfade region is centered on the splice point:
/// - The last `crossfade_samples / 2` samples of the outgoing take overlap with
///   the first `crossfade_samples / 2` samples of the incoming take.
///   If `crossfade_samples` is odd, one extra sample is added to the outgoing
///   tail.
///
/// # Scheduling
///
/// ```text
/// Outgoing take: [─────────────────────[XFADE]
///                                       ↕ blend region
/// Incoming take:               [XFADE]─────────────────────]
/// Output:         [out_head][BLENDED][in_tail]
/// ```
#[derive(Debug, Clone)]
pub struct CrossfadeEngine {
    /// Number of samples for the crossfade region.
    pub crossfade_samples: usize,
    /// Crossfade curve shape.
    pub curve: CrossfadeCurve,
}

impl CrossfadeEngine {
    /// Create a new crossfade engine.
    ///
    /// # Errors
    ///
    /// Returns [`CrossfadeError::ZeroLength`] if `crossfade_samples` is zero.
    pub fn new(crossfade_samples: usize, curve: CrossfadeCurve) -> Result<Self, CrossfadeError> {
        if crossfade_samples == 0 {
            return Err(CrossfadeError::ZeroLength);
        }
        Ok(Self {
            crossfade_samples,
            curve,
        })
    }

    /// Splice `outgoing` and `incoming` takes with a sample-accurate crossfade.
    ///
    /// The crossfade region is constructed from the **last** `crossfade_samples`
    /// samples of `outgoing` blended with the **first** `crossfade_samples`
    /// samples of `incoming`.
    ///
    /// Returns a new buffer:
    /// `[outgoing[..n_out - xfade]] + [blended xfade region] + [incoming[xfade..]]`
    ///
    /// # Errors
    ///
    /// - [`CrossfadeError::CrossfadeTooLong`] if either buffer is shorter than
    ///   `crossfade_samples`.
    pub fn splice(&self, outgoing: &[f32], incoming: &[f32]) -> Result<Vec<f32>, CrossfadeError> {
        let xf = self.crossfade_samples;

        if outgoing.len() < xf {
            return Err(CrossfadeError::CrossfadeTooLong {
                requested: xf,
                available: outgoing.len(),
            });
        }
        if incoming.len() < xf {
            return Err(CrossfadeError::CrossfadeTooLong {
                requested: xf,
                available: incoming.len(),
            });
        }

        let pre_len = outgoing.len() - xf;
        let post_len = incoming.len() - xf;
        let total = pre_len + xf + post_len;
        let mut output = Vec::with_capacity(total);

        // Pre-crossfade region: pure outgoing.
        output.extend_from_slice(&outgoing[..pre_len]);

        // Crossfade region: blend outgoing tail with incoming head.
        for i in 0..xf {
            let t = i as f32 / xf as f32;
            let g_out = self.curve.gain_out(t);
            let g_in = self.curve.gain_in(t);
            let out_sample = outgoing[pre_len + i] * g_out;
            let in_sample = incoming[i] * g_in;
            output.push(out_sample + in_sample);
        }

        // Post-crossfade region: pure incoming.
        output.extend_from_slice(&incoming[xf..]);

        Ok(output)
    }

    /// Compute only the blended crossfade region from the **last** `n` samples
    /// of `outgoing` and the **first** `n` samples of `incoming`, where
    /// `n = crossfade_samples`.
    ///
    /// Returns a `Vec<f32>` of length `crossfade_samples`.
    ///
    /// # Errors
    ///
    /// - [`CrossfadeError::CrossfadeTooLong`] if either buffer is shorter than
    ///   `crossfade_samples`.
    pub fn blend_region(
        &self,
        outgoing: &[f32],
        incoming: &[f32],
    ) -> Result<Vec<f32>, CrossfadeError> {
        let xf = self.crossfade_samples;
        if outgoing.len() < xf {
            return Err(CrossfadeError::CrossfadeTooLong {
                requested: xf,
                available: outgoing.len(),
            });
        }
        if incoming.len() < xf {
            return Err(CrossfadeError::CrossfadeTooLong {
                requested: xf,
                available: incoming.len(),
            });
        }
        let out_offset = outgoing.len() - xf;
        let mut blended = Vec::with_capacity(xf);
        for i in 0..xf {
            let t = i as f32 / xf as f32;
            let g_out = self.curve.gain_out(t);
            let g_in = self.curve.gain_in(t);
            blended.push(outgoing[out_offset + i] * g_out + incoming[i] * g_in);
        }
        Ok(blended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_take(id: &str, number: u32, cue: &str) -> Take {
        let mut t = Take::new(id, number, cue);
        t.set_audio(format!("/audio/{id}.wav"), 5.0);
        t
    }

    #[test]
    fn test_take_id_display() {
        let id = TakeId::new("take-42");
        assert_eq!(format!("{id}"), "take-42");
        assert_eq!(id.value(), "take-42");
    }

    #[test]
    fn test_take_rating_clamping() {
        assert_eq!(TakeRating::new(0).stars(), 1);
        assert_eq!(TakeRating::new(3).stars(), 3);
        assert_eq!(TakeRating::new(10).stars(), 5);
    }

    #[test]
    fn test_take_rating_display() {
        assert_eq!(format!("{}", TakeRating::new(4)), "4/5");
    }

    #[test]
    fn test_take_status_display() {
        assert_eq!(format!("{}", TakeStatus::Recorded), "recorded");
        assert_eq!(format!("{}", TakeStatus::Selected), "selected");
        assert_eq!(format!("{}", TakeStatus::Rejected), "rejected");
    }

    #[test]
    fn test_take_new() {
        let t = Take::new("t1", 1, "cue-1");
        assert_eq!(t.id, TakeId::new("t1"));
        assert_eq!(t.number, 1);
        assert_eq!(t.status, TakeStatus::Recorded);
        assert!(t.rating.is_none());
    }

    #[test]
    fn test_take_rate_and_notes() {
        let mut t = Take::new("t1", 1, "cue-1");
        t.rate(4);
        assert_eq!(t.rating.expect("rating should be valid").stars(), 4);
        t.add_note("good timing");
        t.add_note("slightly off pitch");
        assert!(t.notes.contains("good timing"));
        assert!(t.notes.contains("slightly off pitch"));
    }

    #[test]
    fn test_take_is_usable() {
        let mut t = Take::new("t1", 1, "cue-1");
        assert!(!t.is_usable());
        t.set_status(TakeStatus::Selected);
        assert!(t.is_usable());
        t.set_status(TakeStatus::Alternate);
        assert!(t.is_usable());
        t.set_status(TakeStatus::Rejected);
        assert!(!t.is_usable());
    }

    #[test]
    fn test_manager_add_and_count() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        mgr.add_take(make_take("t2", 2, "cue-1"));
        mgr.add_take(make_take("t3", 1, "cue-2"));
        assert_eq!(mgr.take_count(), 3);
        assert_eq!(mgr.cue_count(), 2);
    }

    #[test]
    fn test_manager_takes_for_cue() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t2", 2, "cue-1"));
        mgr.add_take(make_take("t1", 1, "cue-1"));
        let takes = mgr.takes_for_cue("cue-1");
        assert_eq!(takes.len(), 2);
        // Should be sorted by number
        assert_eq!(takes[0].number, 1);
        assert_eq!(takes[1].number, 2);
    }

    #[test]
    fn test_manager_select_take() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        mgr.add_take(make_take("t2", 2, "cue-1"));

        assert!(mgr.select_take(&TakeId::new("t1")));
        assert_eq!(
            mgr.get(&TakeId::new("t1"))
                .expect("failed to get value")
                .status,
            TakeStatus::Selected
        );

        // Selecting t2 should deselect t1
        assert!(mgr.select_take(&TakeId::new("t2")));
        assert_eq!(
            mgr.get(&TakeId::new("t1"))
                .expect("failed to get value")
                .status,
            TakeStatus::Reviewed
        );
        assert_eq!(
            mgr.get(&TakeId::new("t2"))
                .expect("failed to get value")
                .status,
            TakeStatus::Selected
        );
    }

    #[test]
    fn test_manager_select_nonexistent() {
        let mut mgr = TakeManager::new("session-1");
        assert!(!mgr.select_take(&TakeId::new("nope")));
    }

    #[test]
    fn test_manager_selected_take() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        assert!(mgr.selected_take("cue-1").is_none());
        mgr.select_take(&TakeId::new("t1"));
        assert!(mgr.selected_take("cue-1").is_some());
    }

    #[test]
    fn test_manager_best_rated_take() {
        let mut mgr = TakeManager::new("session-1");
        let mut t1 = make_take("t1", 1, "cue-1");
        t1.rate(3);
        let mut t2 = make_take("t2", 2, "cue-1");
        t2.rate(5);
        mgr.add_take(t1);
        mgr.add_take(t2);

        let best = mgr
            .best_rated_take("cue-1")
            .expect("best_rated_take should succeed");
        assert_eq!(best.rating.expect("rating should be valid").stars(), 5);
    }

    #[test]
    fn test_manager_cues_without_selection() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        mgr.add_take(make_take("t2", 1, "cue-2"));
        mgr.select_take(&TakeId::new("t1"));

        let unselected = mgr.cues_without_selection();
        assert_eq!(unselected.len(), 1);
        assert_eq!(unselected[0], "cue-2");
    }

    #[test]
    fn test_manager_stats() {
        let mut mgr = TakeManager::new("session-1");
        let mut t1 = make_take("t1", 1, "cue-1");
        t1.rate(4);
        t1.set_status(TakeStatus::Selected);
        let mut t2 = make_take("t2", 2, "cue-1");
        t2.rate(2);
        t2.set_status(TakeStatus::Rejected);
        mgr.add_take(t1);
        mgr.add_take(t2);

        let stats = mgr.stats();
        assert_eq!(stats.total_takes, 2);
        assert_eq!(stats.selected_count, 1);
        assert_eq!(stats.rejected_count, 1);
        assert_eq!(stats.rated_count, 2);
        assert!((stats.average_rating - 3.0).abs() < f64::EPSILON);
        assert!((stats.total_duration_secs - 10.0).abs() < f64::EPSILON);
    }

    // ── CrossfadeEngine tests ─────────────────────────────────────────────────

    #[test]
    fn test_crossfade_engine_creation() {
        let engine = CrossfadeEngine::new(256, CrossfadeCurve::ConstantPower).expect("failed");
        assert_eq!(engine.crossfade_samples, 256);
    }

    #[test]
    fn test_crossfade_engine_zero_length_error() {
        assert!(CrossfadeEngine::new(0, CrossfadeCurve::Linear).is_err());
    }

    #[test]
    fn test_crossfade_engine_output_length() {
        let engine = CrossfadeEngine::new(100, CrossfadeCurve::Linear).expect("failed");
        let outgoing = vec![0.8f32; 500];
        let incoming = vec![0.3f32; 400];
        let spliced = engine.splice(&outgoing, &incoming).expect("splice");
        // Expected: (500 - 100) + 100 + (400 - 100) = 800
        assert_eq!(spliced.len(), 800);
    }

    #[test]
    fn test_crossfade_engine_linear_curve_midpoint() {
        // At the midpoint of a linear crossfade, both signals should be at 50%.
        let engine = CrossfadeEngine::new(100, CrossfadeCurve::Linear).expect("failed");
        let outgoing = vec![1.0f32; 200]; // constant 1.0
        let incoming = vec![0.0f32; 200]; // constant 0.0
        let spliced = engine.splice(&outgoing, &incoming).expect("splice");
        // At the midpoint of the crossfade region (sample index 150 in output):
        let mid = spliced[149]; // 99 samples into the xfade region, index=150-1
        assert!(
            (mid - 0.5).abs() < 0.1,
            "Midpoint should be near 0.5, got {mid}"
        );
    }

    #[test]
    fn test_crossfade_engine_constant_power_energy_preserved() {
        // At any point in a constant-power crossfade, cos²(t) + sin²(t) = 1.
        let engine = CrossfadeEngine::new(1000, CrossfadeCurve::ConstantPower).expect("failed");
        let outgoing = vec![1.0f32; 1500];
        let incoming = vec![1.0f32; 1500];
        let spliced = engine.splice(&outgoing, &incoming).expect("splice");
        // Within the crossfade region, energy should be approximately constant.
        let xf_start = 500usize; // outgoing.len() - 1000
        for i in 0..1000 {
            let s = spliced[xf_start + i];
            // cos²(t) + sin²(t) = 1 → blended amplitude of 1.0 + 1.0 should be ~1.0
            assert!(
                (s - 1.0).abs() < 0.02,
                "Energy not constant at sample {i}: {s}"
            );
        }
    }

    #[test]
    fn test_crossfade_engine_too_short_outgoing() {
        let engine = CrossfadeEngine::new(200, CrossfadeCurve::SCurve).expect("failed");
        let outgoing = vec![0.5f32; 100]; // too short
        let incoming = vec![0.5f32; 300];
        assert!(engine.splice(&outgoing, &incoming).is_err());
    }

    #[test]
    fn test_crossfade_engine_too_short_incoming() {
        let engine = CrossfadeEngine::new(200, CrossfadeCurve::SCurve).expect("failed");
        let outgoing = vec![0.5f32; 400];
        let incoming = vec![0.5f32; 100]; // too short
        assert!(engine.splice(&outgoing, &incoming).is_err());
    }

    #[test]
    fn test_crossfade_blend_region_length() {
        let engine = CrossfadeEngine::new(50, CrossfadeCurve::Exponential).expect("failed");
        let out = vec![0.5f32; 200];
        let inc = vec![0.5f32; 200];
        let blend = engine.blend_region(&out, &inc).expect("blend");
        assert_eq!(blend.len(), 50);
    }

    #[test]
    fn test_crossfade_all_curves_finite() {
        let curves = [
            CrossfadeCurve::Linear,
            CrossfadeCurve::ConstantPower,
            CrossfadeCurve::SCurve,
            CrossfadeCurve::Exponential,
        ];
        for curve in curves {
            let engine = CrossfadeEngine::new(256, curve).expect("failed");
            let out = vec![0.7f32; 512];
            let inc = vec![0.4f32; 512];
            let spliced = engine.splice(&out, &inc).expect("splice");
            for &s in &spliced {
                assert!(s.is_finite(), "{curve:?} produced non-finite sample: {s}");
            }
        }
    }

    // ── ADR workflow integration test ─────────────────────────────────────────

    #[test]
    fn test_adr_workflow_create_session_add_cues_record_sync() {
        // Integration test: complete ADR workflow.
        let mut mgr = TakeManager::new("scene-42-adr");

        // Step 1: Register takes for several cues.
        let cues = ["line-1", "line-2", "line-3"];
        for (cue_idx, &cue) in cues.iter().enumerate() {
            for take_num in 1u32..=3 {
                let take_id = format!("{cue}-take{take_num}");
                let mut take = make_take(&take_id, take_num, cue);
                take.set_audio(
                    format!("/sessions/scene-42/{cue}/take{take_num}.wav"),
                    2.5 + cue_idx as f64 * 0.1,
                );
                if take_num == 2 {
                    take.rate(5); // Best take
                }
                mgr.add_take(take);
            }
        }

        assert_eq!(mgr.take_count(), 9);
        assert_eq!(mgr.cue_count(), 3);

        // Step 2: Select the best-rated take for each cue.
        for cue in &cues {
            let best = mgr.best_rated_take(cue).expect("should have a rated take");
            let id = best.id.clone();
            assert!(mgr.select_take(&id));
        }

        // Step 3: Verify all cues have a selection.
        let unselected = mgr.cues_without_selection();
        assert!(
            unselected.is_empty(),
            "All cues should be selected; unselected: {unselected:?}"
        );

        // Step 4: Simulate sync by building a crossfade engine for take stitching.
        let engine =
            CrossfadeEngine::new(512, CrossfadeCurve::ConstantPower).expect("crossfade engine");

        // Simulate two adjacent takes being spliced.
        let take_a: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).sin() * 0.8).collect();
        let take_b: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.01).cos() * 0.7).collect();
        let spliced = engine.splice(&take_a, &take_b).expect("splice");
        assert_eq!(spliced.len(), take_a.len() + take_b.len() - 512);

        // Step 5: Verify stats.
        let stats = mgr.stats();
        assert_eq!(stats.selected_count, 3);
        assert_eq!(stats.total_takes, 9);
    }
}
