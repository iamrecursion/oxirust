//! Adaptive Bitrate (ABR) ladder generation for HLS/DASH streaming.

use serde::{Deserialize, Serialize};

/// A single rung in an ABR ladder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbrRung {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Target video bitrate in bits per second.
    pub video_bitrate: u64,
    /// Target audio bitrate in bits per second.
    pub audio_bitrate: u64,
    /// Frame rate as (numerator, denominator).
    pub frame_rate: (u32, u32),
    /// Codec to use for this rung.
    pub codec: String,
    /// Profile name for this rung (e.g., "720p", "1080p").
    pub profile_name: String,
}

/// Strategy for generating ABR ladder rungs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbrStrategy {
    /// Apple HLS recommendations.
    AppleHls,
    /// `YouTube` recommendations.
    YouTube,
    /// Netflix-style ladder.
    Netflix,
    /// Conservative ladder (fewer rungs).
    Conservative,
    /// Aggressive ladder (more rungs).
    Aggressive,
    /// Custom strategy.
    Custom,
}

/// ABR ladder configuration.
#[derive(Debug, Clone)]
pub struct AbrLadder {
    /// The rungs in the ladder, sorted by bitrate (lowest to highest).
    pub rungs: Vec<AbrRung>,
    /// Strategy used to generate the ladder.
    pub strategy: AbrStrategy,
    /// Maximum resolution to include.
    pub max_resolution: (u32, u32),
    /// Minimum resolution to include.
    pub min_resolution: (u32, u32),
}

impl AbrRung {
    /// Creates a new ABR rung.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        video_bitrate: u64,
        audio_bitrate: u64,
        codec: impl Into<String>,
        profile_name: impl Into<String>,
    ) -> Self {
        Self {
            width,
            height,
            video_bitrate,
            audio_bitrate,
            frame_rate: (30, 1),
            codec: codec.into(),
            profile_name: profile_name.into(),
        }
    }

    /// Sets the frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, num: u32, den: u32) -> Self {
        self.frame_rate = (num, den);
        self
    }

    /// Gets the total bitrate (video + audio).
    #[must_use]
    pub fn total_bitrate(&self) -> u64 {
        self.video_bitrate + self.audio_bitrate
    }

    /// Gets the resolution as a string (e.g., "1920x1080").
    #[must_use]
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Checks if this rung is HD quality or higher (720p+).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.height >= 720
    }

    /// Checks if this rung is Full HD quality or higher (1080p+).
    #[must_use]
    pub fn is_full_hd(&self) -> bool {
        self.height >= 1080
    }

    /// Checks if this rung is 4K quality or higher (2160p+).
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.height >= 2160
    }
}

impl AbrLadder {
    /// Creates a new empty ABR ladder.
    #[must_use]
    pub fn new(strategy: AbrStrategy) -> Self {
        Self {
            rungs: Vec::new(),
            strategy,
            max_resolution: (3840, 2160), // 4K
            min_resolution: (426, 240),   // 240p
        }
    }

    /// Adds a rung to the ladder.
    pub fn add_rung(&mut self, rung: AbrRung) {
        self.rungs.push(rung);
        // Keep sorted by total bitrate
        self.rungs.sort_by_key(AbrRung::total_bitrate);
    }

    /// Sets the maximum resolution.
    #[must_use]
    pub fn with_max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_resolution = (width, height);
        self
    }

    /// Sets the minimum resolution.
    #[must_use]
    pub fn with_min_resolution(mut self, width: u32, height: u32) -> Self {
        self.min_resolution = (width, height);
        self
    }

    /// Generates a standard HLS ladder based on Apple recommendations.
    #[must_use]
    pub fn hls_standard() -> Self {
        let mut ladder = Self::new(AbrStrategy::AppleHls);

        // Apple HLS recommendations
        ladder.add_rung(AbrRung::new(426, 240, 400_000, 64_000, "h264", "240p"));
        ladder.add_rung(AbrRung::new(640, 360, 800_000, 96_000, "h264", "360p"));
        ladder.add_rung(AbrRung::new(854, 480, 1_400_000, 128_000, "h264", "480p"));
        ladder.add_rung(AbrRung::new(1280, 720, 2_800_000, 128_000, "h264", "720p"));
        ladder.add_rung(AbrRung::new(
            1920, 1080, 5_000_000, 192_000, "h264", "1080p",
        ));

        ladder
    }

    /// Generates a YouTube-style ABR ladder.
    #[must_use]
    pub fn youtube_standard() -> Self {
        let mut ladder = Self::new(AbrStrategy::YouTube);

        // YouTube recommendations
        ladder.add_rung(AbrRung::new(426, 240, 300_000, 64_000, "vp9", "240p"));
        ladder.add_rung(AbrRung::new(640, 360, 700_000, 96_000, "vp9", "360p"));
        ladder.add_rung(AbrRung::new(854, 480, 1_000_000, 128_000, "vp9", "480p"));
        ladder.add_rung(AbrRung::new(1280, 720, 2_500_000, 128_000, "vp9", "720p"));
        ladder.add_rung(AbrRung::new(1920, 1080, 4_500_000, 192_000, "vp9", "1080p"));
        ladder.add_rung(AbrRung::new(2560, 1440, 9_000_000, 192_000, "vp9", "1440p"));

        ladder
    }

    /// Generates a conservative ladder (fewer rungs for bandwidth savings).
    #[must_use]
    pub fn conservative() -> Self {
        let mut ladder = Self::new(AbrStrategy::Conservative);

        ladder.add_rung(AbrRung::new(640, 360, 600_000, 96_000, "h264", "360p"));
        ladder.add_rung(AbrRung::new(1280, 720, 2_000_000, 128_000, "h264", "720p"));
        ladder.add_rung(AbrRung::new(
            1920, 1080, 4_000_000, 192_000, "h264", "1080p",
        ));

        ladder
    }

    /// Generates an aggressive ladder (more rungs for quality).
    #[must_use]
    pub fn aggressive() -> Self {
        let mut ladder = Self::new(AbrStrategy::Aggressive);

        ladder.add_rung(AbrRung::new(426, 240, 400_000, 64_000, "h264", "240p"));
        ladder.add_rung(AbrRung::new(640, 360, 800_000, 96_000, "h264", "360p"));
        ladder.add_rung(AbrRung::new(854, 480, 1_400_000, 128_000, "h264", "480p"));
        ladder.add_rung(AbrRung::new(960, 540, 2_000_000, 128_000, "h264", "540p"));
        ladder.add_rung(AbrRung::new(1280, 720, 3_000_000, 128_000, "h264", "720p"));
        ladder.add_rung(AbrRung::new(
            1920, 1080, 5_500_000, 192_000, "h264", "1080p",
        ));
        ladder.add_rung(AbrRung::new(
            2560, 1440, 10_000_000, 192_000, "h264", "1440p",
        ));
        ladder.add_rung(AbrRung::new(
            3840, 2160, 20_000_000, 256_000, "h264", "2160p",
        ));

        ladder
    }

    /// Filters rungs based on source resolution.
    ///
    /// Only includes rungs at or below the source resolution.
    #[must_use]
    pub fn filter_by_source(mut self, source_width: u32, source_height: u32) -> Self {
        self.rungs
            .retain(|rung| rung.width <= source_width && rung.height <= source_height);
        self
    }

    /// Gets the number of rungs in the ladder.
    #[must_use]
    pub fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// Gets a rung by index.
    #[must_use]
    pub fn get_rung(&self, index: usize) -> Option<&AbrRung> {
        self.rungs.get(index)
    }

    /// Gets the highest quality rung.
    #[must_use]
    pub fn highest_quality(&self) -> Option<&AbrRung> {
        self.rungs.last()
    }

    /// Gets the lowest quality rung.
    #[must_use]
    pub fn lowest_quality(&self) -> Option<&AbrRung> {
        self.rungs.first()
    }
}

/// Builder for creating custom ABR ladders.
pub struct AbrLadderBuilder {
    ladder: AbrLadder,
}

impl AbrLadderBuilder {
    /// Creates a new builder with the specified strategy.
    #[must_use]
    pub fn new(strategy: AbrStrategy) -> Self {
        Self {
            ladder: AbrLadder::new(strategy),
        }
    }

    /// Adds a rung to the ladder.
    #[must_use]
    pub fn add_rung(mut self, rung: AbrRung) -> Self {
        self.ladder.add_rung(rung);
        self
    }

    /// Adds a rung with the specified parameters.
    #[must_use]
    pub fn add(
        mut self,
        width: u32,
        height: u32,
        video_bitrate: u64,
        audio_bitrate: u64,
        codec: impl Into<String>,
        profile_name: impl Into<String>,
    ) -> Self {
        let rung = AbrRung::new(
            width,
            height,
            video_bitrate,
            audio_bitrate,
            codec,
            profile_name,
        );
        self.ladder.add_rung(rung);
        self
    }

    /// Sets the maximum resolution.
    #[must_use]
    pub fn max_resolution(mut self, width: u32, height: u32) -> Self {
        self.ladder.max_resolution = (width, height);
        self
    }

    /// Sets the minimum resolution.
    #[must_use]
    pub fn min_resolution(mut self, width: u32, height: u32) -> Self {
        self.ladder.min_resolution = (width, height);
        self
    }

    /// Builds the ABR ladder.
    #[must_use]
    pub fn build(self) -> AbrLadder {
        self.ladder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abr_rung_creation() {
        let rung = AbrRung::new(1920, 1080, 5_000_000, 192_000, "h264", "1080p");

        assert_eq!(rung.width, 1920);
        assert_eq!(rung.height, 1080);
        assert_eq!(rung.video_bitrate, 5_000_000);
        assert_eq!(rung.audio_bitrate, 192_000);
        assert_eq!(rung.total_bitrate(), 5_192_000);
        assert_eq!(rung.codec, "h264");
        assert_eq!(rung.profile_name, "1080p");
    }

    #[test]
    fn test_abr_rung_quality_checks() {
        let rung_240p = AbrRung::new(426, 240, 400_000, 64_000, "h264", "240p");
        assert!(!rung_240p.is_hd());
        assert!(!rung_240p.is_full_hd());
        assert!(!rung_240p.is_4k());

        let rung_720p = AbrRung::new(1280, 720, 2_800_000, 128_000, "h264", "720p");
        assert!(rung_720p.is_hd());
        assert!(!rung_720p.is_full_hd());
        assert!(!rung_720p.is_4k());

        let rung_1080p = AbrRung::new(1920, 1080, 5_000_000, 192_000, "h264", "1080p");
        assert!(rung_1080p.is_hd());
        assert!(rung_1080p.is_full_hd());
        assert!(!rung_1080p.is_4k());

        let rung_4k = AbrRung::new(3840, 2160, 20_000_000, 256_000, "h264", "2160p");
        assert!(rung_4k.is_hd());
        assert!(rung_4k.is_full_hd());
        assert!(rung_4k.is_4k());
    }

    #[test]
    fn test_abr_rung_resolution_string() {
        let rung = AbrRung::new(1920, 1080, 5_000_000, 192_000, "h264", "1080p");
        assert_eq!(rung.resolution_string(), "1920x1080");
    }

    #[test]
    fn test_hls_standard_ladder() {
        let ladder = AbrLadder::hls_standard();
        assert_eq!(ladder.rung_count(), 5);
        assert_eq!(ladder.strategy, AbrStrategy::AppleHls);

        let lowest = ladder.lowest_quality().expect("should succeed in test");
        assert_eq!(lowest.profile_name, "240p");

        let highest = ladder.highest_quality().expect("should succeed in test");
        assert_eq!(highest.profile_name, "1080p");
    }

    #[test]
    fn test_youtube_standard_ladder() {
        let ladder = AbrLadder::youtube_standard();
        assert_eq!(ladder.rung_count(), 6);
        assert_eq!(ladder.strategy, AbrStrategy::YouTube);

        let highest = ladder.highest_quality().expect("should succeed in test");
        assert_eq!(highest.profile_name, "1440p");
    }

    #[test]
    fn test_conservative_ladder() {
        let ladder = AbrLadder::conservative();
        assert_eq!(ladder.rung_count(), 3);
        assert_eq!(ladder.strategy, AbrStrategy::Conservative);
    }

    #[test]
    fn test_aggressive_ladder() {
        let ladder = AbrLadder::aggressive();
        assert_eq!(ladder.rung_count(), 8);
        assert_eq!(ladder.strategy, AbrStrategy::Aggressive);
    }

    #[test]
    fn test_ladder_filtering() {
        let ladder = AbrLadder::hls_standard();
        let filtered = ladder.filter_by_source(1280, 720);

        assert_eq!(filtered.rung_count(), 4); // 240p, 360p, 480p, 720p
        let highest = filtered.highest_quality().expect("should succeed in test");
        assert_eq!(highest.profile_name, "720p");
    }

    #[test]
    fn test_ladder_builder() {
        let ladder = AbrLadderBuilder::new(AbrStrategy::Custom)
            .add(640, 360, 800_000, 96_000, "h264", "360p")
            .add(1280, 720, 2_800_000, 128_000, "h264", "720p")
            .add(1920, 1080, 5_000_000, 192_000, "h264", "1080p")
            .max_resolution(1920, 1080)
            .min_resolution(640, 360)
            .build();

        assert_eq!(ladder.rung_count(), 3);
        assert_eq!(ladder.strategy, AbrStrategy::Custom);
    }

    #[test]
    fn test_ladder_sorting() {
        let mut ladder = AbrLadder::new(AbrStrategy::Custom);

        // Add rungs in reverse order
        ladder.add_rung(AbrRung::new(
            1920, 1080, 5_000_000, 192_000, "h264", "1080p",
        ));
        ladder.add_rung(AbrRung::new(640, 360, 800_000, 96_000, "h264", "360p"));
        ladder.add_rung(AbrRung::new(1280, 720, 2_800_000, 128_000, "h264", "720p"));

        // Should be sorted by bitrate
        assert_eq!(ladder.rungs[0].profile_name, "360p");
        assert_eq!(ladder.rungs[1].profile_name, "720p");
        assert_eq!(ladder.rungs[2].profile_name, "1080p");
    }

    // ── ABR ladder validation — HLS 720p source ───────────────────────────────

    /// A 720p-capped HLS ladder must contain 360p and 720p rungs, and every rung
    /// must be at or below 720p (the simulated source resolution).
    #[test]
    fn test_hls_abr_ladder_720p_has_correct_rungs() {
        let ladder = AbrLadder::hls_standard().filter_by_source(1280, 720);

        let heights: Vec<u32> = ladder.rungs.iter().map(|r| r.height).collect();
        assert!(
            heights.contains(&360),
            "HLS 720p ladder must include 360p rung; got {heights:?}"
        );
        assert!(
            heights.contains(&720),
            "HLS 720p ladder must include 720p rung; got {heights:?}"
        );
        assert!(
            heights.iter().all(|&h| h <= 720),
            "All rungs must be at or below 720p for a 720p source; got {heights:?}"
        );
        // Bitrates must be sorted ascending (lowest first — AbrLadder ordering).
        let bitrates: Vec<u64> = ladder.rungs.iter().map(|r| r.video_bitrate).collect();
        let mut sorted = bitrates.clone();
        sorted.sort_unstable();
        assert_eq!(
            bitrates, sorted,
            "Rungs must be ordered by ascending video_bitrate; got {bitrates:?}"
        );
    }

    /// A DASH-like aggressive ladder must contain a 4K rung (height >= 2160) and
    /// a rung with video bitrate >= 15 Mbps, matching real-world DASH UHD profiles.
    #[test]
    fn test_dash_abr_ladder_4k_has_uhd_rung() {
        // `aggressive()` is the highest-density ladder and includes a 2160p rung.
        let ladder = AbrLadder::aggressive();

        assert!(
            ladder.rungs.iter().any(|r| r.height >= 2160),
            "DASH aggressive ladder must contain a UHD (2160p) rung"
        );
        assert!(
            ladder.rungs.iter().any(|r| r.video_bitrate >= 15_000_000),
            "DASH aggressive ladder must have a rung with video_bitrate >= 15 Mbps for UHD"
        );

        // Sanity: highest-quality rung is UHD.
        let top = ladder
            .highest_quality()
            .expect("aggressive ladder must be non-empty");
        assert!(
            top.is_4k(),
            "Highest rung in aggressive ladder must be 4K; height={}",
            top.height
        );
    }
}
