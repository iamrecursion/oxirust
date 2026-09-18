//! File-format quality control: container structure and codec compatibility.
//!
//! Provides `FormatSpec`, `FormatQcChecker`, and `FormatQcResult` for
//! verifying that a media file matches expected container and codec parameters.

#![allow(dead_code)]

/// A set of format constraints used to validate a media file.
#[derive(Debug, Clone)]
pub struct FormatSpec {
    /// Expected container format (e.g. `"mp4"`, `"mov"`, `"mxf"`).
    pub container: String,
    /// Accepted video codecs (empty = any).
    pub video_codecs: Vec<String>,
    /// Accepted audio codecs (empty = any).
    pub audio_codecs: Vec<String>,
    /// Minimum video bitrate in kbps (0 = no minimum).
    pub min_video_bitrate_kbps: u32,
    /// Maximum video bitrate in kbps (0 = no maximum).
    pub max_video_bitrate_kbps: u32,
    /// Minimum audio sample rate in Hz (0 = no minimum).
    pub min_sample_rate_hz: u32,
    /// Whether the file must be streamable (fast-start / MOOV at front).
    pub require_streamable: bool,
}

impl FormatSpec {
    /// Create a basic spec that only constrains the container.
    #[must_use]
    pub fn for_container(container: impl Into<String>) -> Self {
        Self {
            container: container.into(),
            video_codecs: Vec::new(),
            audio_codecs: Vec::new(),
            min_video_bitrate_kbps: 0,
            max_video_bitrate_kbps: 0,
            min_sample_rate_hz: 0,
            require_streamable: false,
        }
    }

    /// Add an accepted video codec.
    #[must_use]
    pub fn with_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codecs.push(codec.into());
        self
    }

    /// Add an accepted audio codec.
    #[must_use]
    pub fn with_audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.audio_codecs.push(codec.into());
        self
    }

    /// Set bitrate bounds.
    #[must_use]
    pub fn with_bitrate_range(mut self, min_kbps: u32, max_kbps: u32) -> Self {
        self.min_video_bitrate_kbps = min_kbps;
        self.max_video_bitrate_kbps = max_kbps;
        self
    }

    /// Require streamable (fast-start) format.
    #[must_use]
    pub fn require_streamable(mut self) -> Self {
        self.require_streamable = true;
        self
    }
}

/// Snapshot of a media file's detected format properties.
#[derive(Debug, Clone)]
pub struct DetectedFormat {
    /// Actual container format.
    pub container: String,
    /// Detected video codec.
    pub video_codec: String,
    /// Detected audio codec.
    pub audio_codec: String,
    /// Measured video bitrate in kbps.
    pub video_bitrate_kbps: u32,
    /// Measured audio sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Whether the file has a streamable layout.
    pub streamable: bool,
}

/// A single format check result.
#[derive(Debug, Clone)]
pub struct FormatCheckResult {
    /// Check identifier.
    pub check: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Human-readable description of the outcome.
    pub message: String,
}

impl FormatCheckResult {
    fn pass(check: impl Into<String>, msg: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            passed: true,
            message: msg.into(),
        }
    }

    fn fail(check: impl Into<String>, msg: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            passed: false,
            message: msg.into(),
        }
    }
}

/// Aggregated result of running a [`FormatQcChecker`].
#[derive(Debug, Default)]
pub struct FormatQcResult {
    /// All check outcomes.
    pub checks: Vec<FormatCheckResult>,
}

impl FormatQcResult {
    /// Return `true` if every check passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Return failing checks.
    #[must_use]
    pub fn failures(&self) -> Vec<&FormatCheckResult> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }

    /// Pass rate in `[0.0, 1.0]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_rate(&self) -> f64 {
        if self.checks.is_empty() {
            return 1.0;
        }
        let n = self.checks.iter().filter(|c| c.passed).count();
        n as f64 / self.checks.len() as f64
    }
}

/// Performs format QC checks against a [`FormatSpec`].
#[derive(Debug)]
pub struct FormatQcChecker {
    spec: FormatSpec,
}

impl FormatQcChecker {
    /// Create a new checker with the given spec.
    #[must_use]
    pub fn new(spec: FormatSpec) -> Self {
        Self { spec }
    }

    /// Run all format checks against the detected file properties.
    #[must_use]
    pub fn check(&self, detected: &DetectedFormat) -> FormatQcResult {
        let mut result = FormatQcResult::default();

        // Container check
        let container_match = self
            .spec
            .container
            .eq_ignore_ascii_case(&detected.container);
        result.checks.push(if container_match {
            FormatCheckResult::pass(
                "container",
                format!("container '{}' matches", detected.container),
            )
        } else {
            FormatCheckResult::fail(
                "container",
                format!(
                    "expected '{}', found '{}'",
                    self.spec.container, detected.container
                ),
            )
        });

        // Video codec check
        if !self.spec.video_codecs.is_empty() {
            let ok = self
                .spec
                .video_codecs
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&detected.video_codec));
            result.checks.push(if ok {
                FormatCheckResult::pass(
                    "video_codec",
                    format!("'{}' is accepted", detected.video_codec),
                )
            } else {
                FormatCheckResult::fail(
                    "video_codec",
                    format!(
                        "'{}' not in accepted list {:?}",
                        detected.video_codec, self.spec.video_codecs
                    ),
                )
            });
        }

        // Audio codec check
        if !self.spec.audio_codecs.is_empty() {
            let ok = self
                .spec
                .audio_codecs
                .iter()
                .any(|c| c.eq_ignore_ascii_case(&detected.audio_codec));
            result.checks.push(if ok {
                FormatCheckResult::pass(
                    "audio_codec",
                    format!("'{}' is accepted", detected.audio_codec),
                )
            } else {
                FormatCheckResult::fail(
                    "audio_codec",
                    format!(
                        "'{}' not in accepted list {:?}",
                        detected.audio_codec, self.spec.audio_codecs
                    ),
                )
            });
        }

        // Min bitrate check
        if self.spec.min_video_bitrate_kbps > 0 {
            let ok = detected.video_bitrate_kbps >= self.spec.min_video_bitrate_kbps;
            result.checks.push(if ok {
                FormatCheckResult::pass(
                    "min_bitrate",
                    format!(
                        "{} kbps >= {} kbps",
                        detected.video_bitrate_kbps, self.spec.min_video_bitrate_kbps
                    ),
                )
            } else {
                FormatCheckResult::fail(
                    "min_bitrate",
                    format!(
                        "{} kbps below minimum {} kbps",
                        detected.video_bitrate_kbps, self.spec.min_video_bitrate_kbps
                    ),
                )
            });
        }

        // Max bitrate check
        if self.spec.max_video_bitrate_kbps > 0 {
            let ok = detected.video_bitrate_kbps <= self.spec.max_video_bitrate_kbps;
            result.checks.push(if ok {
                FormatCheckResult::pass(
                    "max_bitrate",
                    format!(
                        "{} kbps <= {} kbps",
                        detected.video_bitrate_kbps, self.spec.max_video_bitrate_kbps
                    ),
                )
            } else {
                FormatCheckResult::fail(
                    "max_bitrate",
                    format!(
                        "{} kbps exceeds maximum {} kbps",
                        detected.video_bitrate_kbps, self.spec.max_video_bitrate_kbps
                    ),
                )
            });
        }

        // Sample rate check
        if self.spec.min_sample_rate_hz > 0 {
            let ok = detected.sample_rate_hz >= self.spec.min_sample_rate_hz;
            result.checks.push(if ok {
                FormatCheckResult::pass(
                    "sample_rate",
                    format!(
                        "{} Hz >= {} Hz",
                        detected.sample_rate_hz, self.spec.min_sample_rate_hz
                    ),
                )
            } else {
                FormatCheckResult::fail(
                    "sample_rate",
                    format!(
                        "{} Hz below minimum {} Hz",
                        detected.sample_rate_hz, self.spec.min_sample_rate_hz
                    ),
                )
            });
        }

        // Streamable check
        if self.spec.require_streamable {
            result.checks.push(if detected.streamable {
                FormatCheckResult::pass("streamable", "file is streamable (fast-start)")
            } else {
                FormatCheckResult::fail(
                    "streamable",
                    "file is not streamable (MOOV atom not at start)",
                )
            });
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_mp4() -> DetectedFormat {
        DetectedFormat {
            container: "mp4".into(),
            video_codec: "h264".into(),
            audio_codec: "aac".into(),
            video_bitrate_kbps: 5000,
            sample_rate_hz: 48000,
            streamable: true,
        }
    }

    fn streaming_spec() -> FormatSpec {
        FormatSpec::for_container("mp4")
            .with_video_codec("h264")
            .with_audio_codec("aac")
            .with_bitrate_range(1000, 8000)
            .require_streamable()
    }

    #[test]
    fn test_all_checks_pass() {
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&good_mp4());
        assert!(result.passed());
    }

    #[test]
    fn test_container_mismatch_fails() {
        let mut detected = good_mp4();
        detected.container = "mkv".into();
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        assert!(!result.passed());
        assert!(result.failures().iter().any(|f| f.check == "container"));
    }

    #[test]
    fn test_video_codec_mismatch_fails() {
        let mut detected = good_mp4();
        detected.video_codec = "hevc".into();
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        assert!(result.failures().iter().any(|f| f.check == "video_codec"));
    }

    #[test]
    fn test_audio_codec_mismatch_fails() {
        let mut detected = good_mp4();
        detected.audio_codec = "opus".into();
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        assert!(result.failures().iter().any(|f| f.check == "audio_codec"));
    }

    #[test]
    fn test_bitrate_below_min_fails() {
        let mut detected = good_mp4();
        detected.video_bitrate_kbps = 500;
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        assert!(result.failures().iter().any(|f| f.check == "min_bitrate"));
    }

    #[test]
    fn test_bitrate_above_max_fails() {
        let mut detected = good_mp4();
        detected.video_bitrate_kbps = 9000;
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        assert!(result.failures().iter().any(|f| f.check == "max_bitrate"));
    }

    #[test]
    fn test_not_streamable_fails() {
        let mut detected = good_mp4();
        detected.streamable = false;
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        assert!(result.failures().iter().any(|f| f.check == "streamable"));
    }

    #[test]
    fn test_spec_no_codec_constraints_skips_codec_checks() {
        let spec = FormatSpec::for_container("mp4");
        let checker = FormatQcChecker::new(spec);
        let result = checker.check(&good_mp4());
        // Only container check runs
        assert_eq!(result.checks.len(), 1);
        assert!(result.passed());
    }

    #[test]
    fn test_pass_rate_all_pass() {
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&good_mp4());
        assert!((result.pass_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pass_rate_partial() {
        let mut detected = good_mp4();
        detected.container = "mkv".into();
        let checker = FormatQcChecker::new(streaming_spec());
        let result = checker.check(&detected);
        let rate = result.pass_rate();
        assert!(rate > 0.0 && rate < 1.0);
    }

    #[test]
    fn test_format_spec_builder_chain() {
        let spec = FormatSpec::for_container("mxf")
            .with_video_codec("dnxhd")
            .with_audio_codec("pcm");
        assert_eq!(spec.container, "mxf");
        assert_eq!(spec.video_codecs, vec!["dnxhd"]);
        assert_eq!(spec.audio_codecs, vec!["pcm"]);
    }

    #[test]
    fn test_case_insensitive_container() {
        let spec = FormatSpec::for_container("MP4");
        let checker = FormatQcChecker::new(spec);
        let result = checker.check(&good_mp4()); // detected has "mp4"
        assert!(result.passed());
    }

    #[test]
    fn test_empty_result_passed() {
        let result = FormatQcResult::default();
        assert!(result.passed());
    }

    #[test]
    fn test_empty_result_pass_rate_one() {
        let result = FormatQcResult::default();
        assert!((result.pass_rate() - 1.0).abs() < f64::EPSILON);
    }
}
