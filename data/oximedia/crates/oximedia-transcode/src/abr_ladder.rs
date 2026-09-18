//! Adaptive bitrate ladder generation, per-rung settings, and ABR rules.
//!
//! Provides tools for generating HLS/DASH ABR ladders with per-rung codec
//! settings and bandwidth-based selection rules.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single rung in an ABR ladder describing one quality level.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrRungConfig {
    /// Label for this rung (e.g., "1080p", "720p").
    pub label: String,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Target video bitrate in bits per second.
    pub video_bitrate_bps: u64,
    /// Target audio bitrate in bits per second.
    pub audio_bitrate_bps: u64,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Constant Rate Factor (lower = better quality).
    pub crf: Option<u8>,
    /// Codec profile (e.g., "high", "main", "baseline").
    pub profile: Option<String>,
    /// Maximum buffer size in bits.
    pub bufsize_bits: Option<u64>,
}

impl AbrRungConfig {
    /// Creates a new ABR rung configuration.
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        width: u32,
        height: u32,
        video_bitrate_bps: u64,
        audio_bitrate_bps: u64,
    ) -> Self {
        Self {
            label: label.into(),
            width,
            height,
            video_bitrate_bps,
            audio_bitrate_bps,
            fps_num: 30,
            fps_den: 1,
            crf: None,
            profile: None,
            bufsize_bits: None,
        }
    }

    /// Sets the frame rate for this rung.
    #[must_use]
    pub fn with_fps(mut self, num: u32, den: u32) -> Self {
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    /// Sets the CRF value for quality-based encoding.
    #[must_use]
    pub fn with_crf(mut self, crf: u8) -> Self {
        self.crf = Some(crf);
        self
    }

    /// Sets the codec profile.
    #[must_use]
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Sets the buffer size in bits (typically 2x the video bitrate).
    #[must_use]
    pub fn with_bufsize(mut self, bufsize_bits: u64) -> Self {
        self.bufsize_bits = Some(bufsize_bits);
        self
    }

    /// Returns the total bitrate (video + audio) in bits per second.
    #[must_use]
    pub fn total_bitrate_bps(&self) -> u64 {
        self.video_bitrate_bps + self.audio_bitrate_bps
    }

    /// Returns the frame rate as a floating point value.
    #[must_use]
    pub fn fps_f64(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    /// Returns the pixel count for this rung.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Strategy for selecting the appropriate ABR rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LadderSelectionStrategy {
    /// Select the highest quality rung that fits within available bandwidth.
    BandwidthFit,
    /// Select the rung whose resolution best matches the display size.
    ResolutionMatch,
    /// Conservatively stay one rung below the maximum fitting rung.
    Conservative,
    /// Aggressively pick the highest rung within 150% of available bandwidth.
    Aggressive,
}

/// Rule for switching between rungs.
#[derive(Debug, Clone)]
pub struct SwitchRule {
    /// Minimum bandwidth required to switch up (in bits per second).
    pub switch_up_bandwidth_bps: u64,
    /// Bandwidth threshold to switch down (in bits per second).
    pub switch_down_bandwidth_bps: u64,
    /// Minimum consecutive measurements before switching up.
    pub switch_up_samples: u32,
    /// Whether to allow switching up more than one rung at a time.
    pub allow_multi_rung_up: bool,
}

impl SwitchRule {
    /// Creates a new switch rule.
    #[must_use]
    pub fn new(switch_up_bps: u64, switch_down_bps: u64) -> Self {
        Self {
            switch_up_bandwidth_bps: switch_up_bps,
            switch_down_bandwidth_bps: switch_down_bps,
            switch_up_samples: 3,
            allow_multi_rung_up: false,
        }
    }

    /// Sets the number of consecutive measurements required to switch up.
    #[must_use]
    pub fn with_switch_up_samples(mut self, samples: u32) -> Self {
        self.switch_up_samples = samples;
        self
    }
}

/// A complete ABR ladder with multiple quality rungs.
#[derive(Debug, Clone)]
pub struct AbrLadderConfig {
    /// All rungs sorted from lowest to highest quality.
    pub rungs: Vec<AbrRungConfig>,
    /// Selection strategy.
    pub strategy: LadderSelectionStrategy,
    /// Switch rules between rungs.
    pub switch_rules: Vec<SwitchRule>,
    /// Segment duration in seconds.
    pub segment_duration_secs: f64,
    /// Target codec for all rungs.
    pub codec: String,
}

impl AbrLadderConfig {
    /// Creates a new ABR ladder configuration.
    #[must_use]
    pub fn new(codec: impl Into<String>) -> Self {
        Self {
            rungs: Vec::new(),
            strategy: LadderSelectionStrategy::BandwidthFit,
            switch_rules: Vec::new(),
            segment_duration_secs: 6.0,
            codec: codec.into(),
        }
    }

    /// Adds a rung to the ladder.
    #[must_use]
    pub fn add_rung(mut self, rung: AbrRungConfig) -> Self {
        self.rungs.push(rung);
        self.rungs.sort_by_key(|r| r.video_bitrate_bps);
        self
    }

    /// Sets the selection strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: LadderSelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the segment duration.
    #[must_use]
    pub fn with_segment_duration(mut self, secs: f64) -> Self {
        self.segment_duration_secs = secs;
        self
    }

    /// Generates the standard Netflix-style HLS ladder for H.264.
    #[must_use]
    pub fn standard_hls_h264() -> Self {
        Self::new("h264")
            .add_rung(
                AbrRungConfig::new("240p", 426, 240, 400_000, 64_000).with_profile("baseline"),
            )
            .add_rung(AbrRungConfig::new("360p", 640, 360, 800_000, 96_000).with_profile("main"))
            .add_rung(AbrRungConfig::new("480p", 854, 480, 1_400_000, 128_000).with_profile("main"))
            .add_rung(
                AbrRungConfig::new("720p", 1280, 720, 2_800_000, 128_000).with_profile("high"),
            )
            .add_rung(
                AbrRungConfig::new("1080p", 1920, 1080, 5_000_000, 192_000).with_profile("high"),
            )
            .add_rung(
                AbrRungConfig::new("4K", 3840, 2160, 15_000_000, 192_000).with_profile("high"),
            )
    }

    /// Selects the best rung for the given available bandwidth.
    #[must_use]
    pub fn select_rung(&self, available_bps: u64) -> Option<&AbrRungConfig> {
        match self.strategy {
            LadderSelectionStrategy::BandwidthFit => self
                .rungs
                .iter()
                .rfind(|r| r.total_bitrate_bps() <= available_bps),
            LadderSelectionStrategy::Conservative => {
                let fitting: Vec<&AbrRungConfig> = self
                    .rungs
                    .iter()
                    .filter(|r| r.total_bitrate_bps() <= available_bps)
                    .collect();
                if fitting.len() > 1 {
                    fitting.get(fitting.len() - 2).copied()
                } else {
                    fitting.into_iter().last()
                }
            }
            LadderSelectionStrategy::Aggressive => self
                .rungs
                .iter()
                .rfind(|r| r.total_bitrate_bps() <= available_bps * 3 / 2),
            LadderSelectionStrategy::ResolutionMatch => {
                // Default to bandwidth fit if no display size info
                self.rungs
                    .iter()
                    .rfind(|r| r.total_bitrate_bps() <= available_bps)
            }
        }
    }

    /// Returns the number of rungs in the ladder.
    #[must_use]
    pub fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// Returns the lowest quality rung.
    #[must_use]
    pub fn lowest_rung(&self) -> Option<&AbrRungConfig> {
        self.rungs.first()
    }

    /// Returns the highest quality rung.
    #[must_use]
    pub fn highest_rung(&self) -> Option<&AbrRungConfig> {
        self.rungs.last()
    }

    /// Generates switch rules for all adjacent rung pairs.
    pub fn generate_switch_rules(&mut self) {
        self.switch_rules.clear();
        for window in self.rungs.windows(2) {
            let lower = &window[0];
            let upper = &window[1];
            // Switch up when bandwidth exceeds upper rung by 20%
            let switch_up = upper.total_bitrate_bps() * 120 / 100;
            // Switch down when bandwidth drops below lower rung
            let switch_down = lower.total_bitrate_bps();
            self.switch_rules
                .push(SwitchRule::new(switch_up, switch_down));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rung_total_bitrate() {
        let rung = AbrRungConfig::new("720p", 1280, 720, 2_800_000, 128_000);
        assert_eq!(rung.total_bitrate_bps(), 2_928_000);
    }

    #[test]
    fn test_rung_fps_f64() {
        let rung = AbrRungConfig::new("1080p", 1920, 1080, 5_000_000, 192_000).with_fps(60, 1);
        assert!((rung.fps_f64() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rung_pixel_count() {
        let rung = AbrRungConfig::new("1080p", 1920, 1080, 5_000_000, 192_000);
        assert_eq!(rung.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_rung_with_crf() {
        let rung = AbrRungConfig::new("720p", 1280, 720, 2_800_000, 128_000).with_crf(23);
        assert_eq!(rung.crf, Some(23));
    }

    #[test]
    fn test_rung_with_profile() {
        let rung = AbrRungConfig::new("1080p", 1920, 1080, 5_000_000, 192_000).with_profile("high");
        assert_eq!(rung.profile.as_deref(), Some("high"));
    }

    #[test]
    fn test_rung_with_bufsize() {
        let rung = AbrRungConfig::new("480p", 854, 480, 1_400_000, 128_000).with_bufsize(2_800_000);
        assert_eq!(rung.bufsize_bits, Some(2_800_000));
    }

    #[test]
    fn test_ladder_rung_count() {
        let ladder = AbrLadderConfig::standard_hls_h264();
        assert_eq!(ladder.rung_count(), 6);
    }

    #[test]
    fn test_ladder_sorted_by_bitrate() {
        let ladder = AbrLadderConfig::standard_hls_h264();
        let bitrates: Vec<u64> = ladder.rungs.iter().map(|r| r.video_bitrate_bps).collect();
        let mut sorted = bitrates.clone();
        sorted.sort_unstable();
        assert_eq!(bitrates, sorted);
    }

    #[test]
    fn test_select_rung_bandwidth_fit() {
        let ladder = AbrLadderConfig::standard_hls_h264();
        // 3 Mbps should select 720p (2.928 Mbps total)
        let rung = ladder
            .select_rung(3_000_000)
            .expect("should succeed in test");
        assert_eq!(rung.label, "720p");
    }

    #[test]
    fn test_select_rung_conservative() {
        let ladder = AbrLadderConfig::standard_hls_h264()
            .with_strategy(LadderSelectionStrategy::Conservative);
        let rung = ladder
            .select_rung(3_000_000)
            .expect("should succeed in test");
        // Should be one below 720p = 480p
        assert_eq!(rung.label, "480p");
    }

    #[test]
    fn test_select_rung_no_fit() {
        let ladder = AbrLadderConfig::standard_hls_h264();
        // Very low bandwidth - no rung fits
        let rung = ladder.select_rung(100_000);
        assert!(rung.is_none());
    }

    #[test]
    fn test_lowest_highest_rung() {
        let ladder = AbrLadderConfig::standard_hls_h264();
        assert_eq!(
            ladder.lowest_rung().expect("should succeed in test").label,
            "240p"
        );
        assert_eq!(
            ladder.highest_rung().expect("should succeed in test").label,
            "4K"
        );
    }

    #[test]
    fn test_generate_switch_rules() {
        let mut ladder = AbrLadderConfig::standard_hls_h264();
        ladder.generate_switch_rules();
        // N rungs => N-1 switch rules
        assert_eq!(ladder.switch_rules.len(), ladder.rung_count() - 1);
    }

    #[test]
    fn test_switch_rule_new() {
        let rule = SwitchRule::new(5_000_000, 2_000_000).with_switch_up_samples(5);
        assert_eq!(rule.switch_up_bandwidth_bps, 5_000_000);
        assert_eq!(rule.switch_down_bandwidth_bps, 2_000_000);
        assert_eq!(rule.switch_up_samples, 5);
    }

    #[test]
    fn test_ladder_segment_duration() {
        let ladder = AbrLadderConfig::new("vp9").with_segment_duration(4.0);
        assert!((ladder.segment_duration_secs - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ladder_codec() {
        let ladder = AbrLadderConfig::new("av1");
        assert_eq!(ladder.codec, "av1");
    }
}
