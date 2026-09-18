//! Restoration planning and scheduling.
//!
//! A `RestorePlan` is an ordered sequence of `RestoreStep`s that describes
//! the complete set of operations required to restore a piece of media.
//! The planner can estimate the total wall-clock time required and validate
//! that the plan is well-formed before execution begins.

#![allow(dead_code)]

use std::time::Duration;

/// A single step in a restoration workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreStep {
    /// Remove DC offset from audio.
    RemoveDcOffset,
    /// Apply click and pop removal to audio.
    ClickRemoval,
    /// Reduce or eliminate tape hiss.
    HissReduction,
    /// Remove hum at the fundamental and harmonics.
    HumRemoval,
    /// Repair clipped audio peaks.
    Declip,
    /// Correct wow and flutter in tape recordings.
    WowFlutterCorrection,
    /// Add synthetic film grain to video.
    GrainSynthesis,
    /// Correct colour fade.
    ColorCorrection,
    /// Remove flickering from film scanning.
    Deflicker,
    /// Detect and handle telecine (3:2 pulldown).
    TelecineDetection,
    /// Deband video (remove banding artefacts).
    Deband,
    /// Up-scale video resolution.
    VideoUpscale,
    /// A custom, user-supplied restoration step.
    Custom {
        /// Short name for the custom step.
        name: String,
        /// Estimated duration for this step.
        estimated_duration: Duration,
    },
}

impl RestoreStep {
    /// Human-readable label for this step.
    pub fn label(&self) -> &str {
        match self {
            Self::RemoveDcOffset => "DC Offset Removal",
            Self::ClickRemoval => "Click/Pop Removal",
            Self::HissReduction => "Hiss Reduction",
            Self::HumRemoval => "Hum Removal",
            Self::Declip => "Declipping",
            Self::WowFlutterCorrection => "Wow/Flutter Correction",
            Self::GrainSynthesis => "Grain Synthesis",
            Self::ColorCorrection => "Colour Correction",
            Self::Deflicker => "Deflicker",
            Self::TelecineDetection => "Telecine Detection",
            Self::Deband => "Debanding",
            Self::VideoUpscale => "Video Upscale",
            Self::Custom { name, .. } => name.as_str(),
        }
    }

    /// Base time estimate for this step per minute of media.
    ///
    /// Returns `None` for steps where the duration is purely dependent on
    /// media length and no simple heuristic exists.
    pub fn base_duration_per_minute(&self) -> Duration {
        match self {
            Self::RemoveDcOffset => Duration::from_millis(200),
            Self::ClickRemoval => Duration::from_secs(3),
            Self::HissReduction => Duration::from_secs(2),
            Self::HumRemoval => Duration::from_millis(1_500),
            Self::Declip => Duration::from_millis(2_500),
            Self::WowFlutterCorrection => Duration::from_secs(4),
            Self::GrainSynthesis => Duration::from_millis(500),
            Self::ColorCorrection => Duration::from_millis(800),
            Self::Deflicker => Duration::from_millis(1_200),
            Self::TelecineDetection => Duration::from_millis(600),
            Self::Deband => Duration::from_millis(700),
            Self::VideoUpscale => Duration::from_secs(60),
            Self::Custom {
                estimated_duration, ..
            } => *estimated_duration,
        }
    }

    /// Returns `true` if this step is audio-only.
    pub fn is_audio_step(&self) -> bool {
        matches!(
            self,
            Self::RemoveDcOffset
                | Self::ClickRemoval
                | Self::HissReduction
                | Self::HumRemoval
                | Self::Declip
                | Self::WowFlutterCorrection
        )
    }

    /// Returns `true` if this step is video-only.
    pub fn is_video_step(&self) -> bool {
        matches!(
            self,
            Self::GrainSynthesis
                | Self::ColorCorrection
                | Self::Deflicker
                | Self::TelecineDetection
                | Self::Deband
                | Self::VideoUpscale
        )
    }
}

/// An ordered restoration plan consisting of multiple steps.
#[derive(Debug, Clone, Default)]
pub struct RestorePlan {
    steps: Vec<RestoreStep>,
    /// Optional name for this plan (e.g. "Vinyl 78 RPM Restoration").
    pub name: String,
}

impl RestorePlan {
    /// Create a new, empty restoration plan.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            name: name.into(),
        }
    }

    /// Append a step to the end of the plan.
    pub fn add_step(&mut self, step: RestoreStep) {
        self.steps.push(step);
    }

    /// Prepend a step at the beginning of the plan.
    pub fn prepend_step(&mut self, step: RestoreStep) {
        self.steps.insert(0, step);
    }

    /// All steps in this plan.
    pub fn steps(&self) -> &[RestoreStep] {
        &self.steps
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the plan contains no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Audio-only steps.
    pub fn audio_steps(&self) -> Vec<&RestoreStep> {
        self.steps.iter().filter(|s| s.is_audio_step()).collect()
    }

    /// Video-only steps.
    pub fn video_steps(&self) -> Vec<&RestoreStep> {
        self.steps.iter().filter(|s| s.is_video_step()).collect()
    }
}

/// Builds and validates restoration plans, and estimates their execution time.
#[derive(Debug, Default)]
pub struct RestorePlanner;

impl RestorePlanner {
    /// Create a new `RestorePlanner`.
    pub fn new() -> Self {
        Self
    }

    /// Estimate the total time required to execute `plan` on media of
    /// duration `media_duration`.
    ///
    /// Returns the summed base time for all steps scaled by the number of
    /// minutes of media.
    pub fn estimate_time(&self, plan: &RestorePlan, media_duration: Duration) -> Duration {
        #[allow(clippy::cast_precision_loss)]
        let minutes = media_duration.as_secs_f64() / 60.0;
        let total_ms: u128 = plan
            .steps()
            .iter()
            .map(|s| {
                let base = s.base_duration_per_minute().as_millis();
                // scale by fractional minutes
                (base as f64 * minutes) as u128
            })
            .sum();
        Duration::from_millis(total_ms as u64)
    }

    /// Validate that the plan is well-formed.
    ///
    /// Returns `Ok(())` if the plan is valid, or an `Err` with a description
    /// of the first problem found.
    pub fn validate(&self, plan: &RestorePlan) -> Result<(), String> {
        if plan.is_empty() {
            return Err("Restoration plan is empty".to_string());
        }
        // Hiss reduction should not come before DC offset removal (best practice).
        let dc_pos = plan
            .steps()
            .iter()
            .position(|s| *s == RestoreStep::RemoveDcOffset);
        let hiss_pos = plan
            .steps()
            .iter()
            .position(|s| *s == RestoreStep::HissReduction);
        if let (Some(dc), Some(hiss)) = (dc_pos, hiss_pos) {
            if hiss < dc {
                return Err("HissReduction should come after RemoveDcOffset".to_string());
            }
        }
        Ok(())
    }

    /// Build a standard vinyl record restoration plan.
    pub fn vinyl_preset(&self) -> RestorePlan {
        let mut plan = RestorePlan::new("Vinyl Restoration");
        plan.add_step(RestoreStep::RemoveDcOffset);
        plan.add_step(RestoreStep::HumRemoval);
        plan.add_step(RestoreStep::ClickRemoval);
        plan.add_step(RestoreStep::HissReduction);
        plan
    }

    /// Build a standard tape restoration plan.
    pub fn tape_preset(&self) -> RestorePlan {
        let mut plan = RestorePlan::new("Tape Restoration");
        plan.add_step(RestoreStep::RemoveDcOffset);
        plan.add_step(RestoreStep::WowFlutterCorrection);
        plan.add_step(RestoreStep::HissReduction);
        plan.add_step(RestoreStep::Declip);
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_plan() {
        let plan = RestorePlan::new("empty");
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_add_step() {
        let mut plan = RestorePlan::new("test");
        plan.add_step(RestoreStep::RemoveDcOffset);
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn test_prepend_step() {
        let mut plan = RestorePlan::new("test");
        plan.add_step(RestoreStep::HissReduction);
        plan.prepend_step(RestoreStep::RemoveDcOffset);
        assert_eq!(plan.steps()[0], RestoreStep::RemoveDcOffset);
    }

    #[test]
    fn test_audio_steps_filter() {
        let mut plan = RestorePlan::new("mixed");
        plan.add_step(RestoreStep::ClickRemoval);
        plan.add_step(RestoreStep::Deflicker);
        plan.add_step(RestoreStep::HumRemoval);
        let audio = plan.audio_steps();
        assert_eq!(audio.len(), 2);
    }

    #[test]
    fn test_video_steps_filter() {
        let mut plan = RestorePlan::new("mixed");
        plan.add_step(RestoreStep::ClickRemoval);
        plan.add_step(RestoreStep::Deflicker);
        plan.add_step(RestoreStep::ColorCorrection);
        let video = plan.video_steps();
        assert_eq!(video.len(), 2);
    }

    #[test]
    fn test_estimate_time_zero_duration() {
        let planner = RestorePlanner::new();
        let mut plan = RestorePlan::new("t");
        plan.add_step(RestoreStep::ClickRemoval);
        let t = planner.estimate_time(&plan, Duration::ZERO);
        assert_eq!(t, Duration::ZERO);
    }

    #[test]
    fn test_estimate_time_one_minute() {
        let planner = RestorePlanner::new();
        let mut plan = RestorePlan::new("t");
        plan.add_step(RestoreStep::HissReduction); // 2000ms per minute
        let t = planner.estimate_time(&plan, Duration::from_secs(60));
        // Allow ±10 ms for floating-point rounding
        let ms = t.as_millis();
        assert!(ms >= 1990 && ms <= 2010, "expected ~2000ms, got {ms}ms");
    }

    #[test]
    fn test_validate_empty_plan() {
        let planner = RestorePlanner::new();
        let plan = RestorePlan::new("empty");
        assert!(planner.validate(&plan).is_err());
    }

    #[test]
    fn test_validate_valid_plan() {
        let planner = RestorePlanner::new();
        let mut plan = RestorePlan::new("valid");
        plan.add_step(RestoreStep::RemoveDcOffset);
        plan.add_step(RestoreStep::HissReduction);
        assert!(planner.validate(&plan).is_ok());
    }

    #[test]
    fn test_validate_bad_order() {
        let planner = RestorePlanner::new();
        let mut plan = RestorePlan::new("bad");
        plan.add_step(RestoreStep::HissReduction);
        plan.add_step(RestoreStep::RemoveDcOffset);
        assert!(planner.validate(&plan).is_err());
    }

    #[test]
    fn test_vinyl_preset_not_empty() {
        let planner = RestorePlanner::new();
        let plan = planner.vinyl_preset();
        assert!(!plan.is_empty());
        assert!(planner.validate(&plan).is_ok());
    }

    #[test]
    fn test_tape_preset_not_empty() {
        let planner = RestorePlanner::new();
        let plan = planner.tape_preset();
        assert!(!plan.is_empty());
        assert!(planner.validate(&plan).is_ok());
    }

    #[test]
    fn test_step_labels_non_empty() {
        let steps = [
            RestoreStep::RemoveDcOffset,
            RestoreStep::ClickRemoval,
            RestoreStep::GrainSynthesis,
            RestoreStep::VideoUpscale,
        ];
        for s in &steps {
            assert!(!s.label().is_empty(), "label for {s:?} is empty");
        }
    }

    #[test]
    fn test_custom_step_label() {
        let step = RestoreStep::Custom {
            name: "my_step".to_string(),
            estimated_duration: Duration::from_secs(5),
        };
        assert_eq!(step.label(), "my_step");
    }

    #[test]
    fn test_step_is_audio_video_exclusive() {
        let audio_step = RestoreStep::ClickRemoval;
        let video_step = RestoreStep::Deflicker;
        assert!(audio_step.is_audio_step());
        assert!(!audio_step.is_video_step());
        assert!(video_step.is_video_step());
        assert!(!video_step.is_audio_step());
    }

    #[test]
    fn test_estimate_two_minute_media() {
        let planner = RestorePlanner::new();
        let plan = planner.vinyl_preset();
        let t1 = planner.estimate_time(&plan, Duration::from_secs(60));
        let t2 = planner.estimate_time(&plan, Duration::from_secs(120));
        // Two minutes should take roughly twice as long
        let ratio = t2.as_millis() as f64 / t1.as_millis() as f64;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "ratio should be ~2.0, got {ratio}"
        );
    }
}
