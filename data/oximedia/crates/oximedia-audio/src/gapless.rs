//! Gapless playback support with encoder delay and padding handling.
//!
//! When audio is encoded with lossy codecs (MP3, AAC, Opus, Vorbis), the encoder
//! typically adds silence at the beginning (encoder delay / pre-skip) and at the
//! end (padding) of the stream. For gapless playback of consecutive tracks, this
//! silence must be trimmed so that the decoded audio matches the original PCM
//! exactly, sample-for-sample.
//!
//! # Architecture
//!
//! ```text
//!  Encoded stream:  [delay][  audio content  ][padding]
//!  After trimming:         [  audio content  ]
//! ```
//!
//! # Supported codecs
//!
//! | Codec  | Delay source                     | Padding source        |
//! |--------|----------------------------------|-----------------------|
//! | MP3    | LAME `enc_delay` in Xing/LAME    | `enc_padding` field   |
//! | AAC    | `edts`/`elst` box, or iTunSMPB   | iTunSMPB              |
//! | Opus   | `pre_skip` in ID header          | `granule_position`    |
//! | Vorbis | First granule position           | Last granule position |
//! | FLAC   | None (always sample-accurate)    | None                  |
//!
//! # Example
//!
//! ```ignore
//! use oximedia_audio::gapless::{GaplessInfo, GaplessTrimmer};
//!
//! let info = GaplessInfo {
//!     encoder_delay_samples: 576,
//!     padding_samples: 1024,
//!     total_samples: Some(44100 * 60),
//!     ..Default::default()
//! };
//! let mut trimmer = GaplessTrimmer::new(info);
//!
//! // Decode frames and trim
//! let decoded_samples = vec![0.0f32; 4096];
//! let trimmed = trimmer.process(&decoded_samples);
//! ```

#![forbid(unsafe_code)]

/// Gapless playback metadata for a single audio stream.
///
/// Describes the encoder delay and padding that must be trimmed
/// for seamless track-to-track playback.
#[derive(Clone, Debug, Default)]
pub struct GaplessInfo {
    /// Number of silent samples prepended by the encoder (pre-skip / encoder delay).
    pub encoder_delay_samples: u64,
    /// Number of padding samples appended by the encoder at the end.
    pub padding_samples: u64,
    /// Total number of valid PCM samples in the stream (if known).
    ///
    /// When set, the trimmer uses this to stop output exactly at the right
    /// sample, regardless of padding information.
    pub total_samples: Option<u64>,
    /// Codec-specific pre-skip (e.g., Opus pre_skip from the ID header).
    pub codec_preskip: u32,
    /// Sample rate in Hz (used for duration calculations).
    pub sample_rate: u32,
}

impl GaplessInfo {
    /// Compute the total delay to trim from the start (encoder delay + codec pre-skip).
    pub fn total_start_trim(&self) -> u64 {
        self.encoder_delay_samples + u64::from(self.codec_preskip)
    }

    /// Compute the valid sample count (if total_samples is known).
    pub fn valid_samples(&self) -> Option<u64> {
        self.total_samples
    }

    /// Compute the duration in seconds (if total_samples and sample_rate are known).
    pub fn duration_secs(&self) -> Option<f64> {
        if self.sample_rate == 0 {
            return None;
        }
        self.total_samples
            .map(|total| total as f64 / f64::from(self.sample_rate))
    }

    /// Create gapless info for Opus with a pre-skip value.
    ///
    /// # Arguments
    ///
    /// * `pre_skip` - Pre-skip from the Opus identification header
    /// * `total_samples` - Total PCM samples (from granule position), if known
    pub fn opus(pre_skip: u32, total_samples: Option<u64>) -> Self {
        Self {
            encoder_delay_samples: 0,
            padding_samples: 0,
            total_samples,
            codec_preskip: pre_skip,
            sample_rate: 48_000,
        }
    }

    /// Create gapless info for MP3 from LAME header data.
    ///
    /// # Arguments
    ///
    /// * `enc_delay` - Encoder delay from LAME header
    /// * `enc_padding` - Encoder padding from LAME header
    /// * `total_samples` - Total valid samples
    /// * `sample_rate` - Sample rate in Hz
    pub fn mp3(
        enc_delay: u64,
        enc_padding: u64,
        total_samples: Option<u64>,
        sample_rate: u32,
    ) -> Self {
        Self {
            encoder_delay_samples: enc_delay,
            padding_samples: enc_padding,
            total_samples,
            codec_preskip: 0,
            sample_rate,
        }
    }

    /// Create gapless info for Vorbis.
    ///
    /// # Arguments
    ///
    /// * `first_granule` - Granule position of the first audio page
    /// * `total_samples` - Total samples from the last granule position
    /// * `sample_rate` - Sample rate in Hz
    pub fn vorbis(first_granule: u64, total_samples: Option<u64>, sample_rate: u32) -> Self {
        Self {
            encoder_delay_samples: first_granule,
            padding_samples: 0,
            total_samples,
            codec_preskip: 0,
            sample_rate,
        }
    }
}

/// State of the gapless trimmer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrimmerState {
    /// Trimming encoder delay from the beginning.
    TrimmingDelay,
    /// Outputting valid audio content.
    Playing,
    /// All valid samples have been output; stream is finished.
    Finished,
}

/// Gapless trimmer that strips encoder delay and padding from decoded audio.
///
/// Feed decoded PCM buffers through [`process`](GaplessTrimmer::process) and
/// it will return only the valid audio content.
pub struct GaplessTrimmer {
    info: GaplessInfo,
    state: TrimmerState,
    /// Number of samples consumed so far (across all calls to `process`).
    samples_consumed: u64,
    /// Number of valid samples output so far.
    samples_output: u64,
    /// Total start trim (cached).
    start_trim: u64,
    /// Maximum samples to output (cached from total_samples).
    max_output: Option<u64>,
}

impl GaplessTrimmer {
    /// Create a new gapless trimmer.
    ///
    /// # Arguments
    ///
    /// * `info` - Gapless playback info describing delay and padding
    pub fn new(info: GaplessInfo) -> Self {
        let start_trim = info.total_start_trim();
        let max_output = info.valid_samples();
        let state = if start_trim > 0 {
            TrimmerState::TrimmingDelay
        } else {
            TrimmerState::Playing
        };

        Self {
            info,
            state,
            samples_consumed: 0,
            samples_output: 0,
            start_trim,
            max_output,
        }
    }

    /// Process a buffer of decoded samples and return the trimmed slice.
    ///
    /// The returned `Vec<f32>` contains only valid audio content with
    /// encoder delay and padding removed.
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.state == TrimmerState::Finished || samples.is_empty() {
            return Vec::new();
        }

        let buf_len = samples.len() as u64;
        let buf_start = self.samples_consumed;
        let buf_end = buf_start + buf_len;

        self.samples_consumed = buf_end;

        // Determine valid range within this buffer
        let valid_start = if buf_start < self.start_trim {
            // Still in the delay region
            let skip = (self.start_trim - buf_start).min(buf_len);
            skip as usize
        } else {
            0
        };

        if valid_start >= samples.len() {
            // Entire buffer is within encoder delay
            return Vec::new();
        }

        // Determine how many samples we can still output
        let remaining_allowed = match self.max_output {
            Some(max) => {
                if self.samples_output >= max {
                    self.state = TrimmerState::Finished;
                    return Vec::new();
                }
                (max - self.samples_output) as usize
            }
            None => {
                // No total_samples known: trim padding from end if specified
                let _total_decoded = buf_end;
                let end_trim_start = if self.info.padding_samples > 0 {
                    // We don't know exact total, so we cannot trim padding
                    // without knowing when the stream ends. Pass everything.
                    usize::MAX
                } else {
                    usize::MAX
                };
                end_trim_start
            }
        };

        let available = samples.len() - valid_start;
        let to_output = available.min(remaining_allowed);

        if to_output == 0 {
            self.state = TrimmerState::Finished;
            return Vec::new();
        }

        self.state = TrimmerState::Playing;
        self.samples_output += to_output as u64;

        if let Some(max) = self.max_output {
            if self.samples_output >= max {
                self.state = TrimmerState::Finished;
            }
        }

        samples[valid_start..valid_start + to_output].to_vec()
    }

    /// Get the current state of the trimmer.
    pub fn state(&self) -> TrimmerState {
        self.state
    }

    /// Get the number of valid samples output so far.
    pub fn samples_output(&self) -> u64 {
        self.samples_output
    }

    /// Get the number of raw samples consumed so far.
    pub fn samples_consumed(&self) -> u64 {
        self.samples_consumed
    }

    /// Check whether all valid samples have been output.
    pub fn is_finished(&self) -> bool {
        self.state == TrimmerState::Finished
    }

    /// Reset the trimmer for replaying from the beginning.
    pub fn reset(&mut self) {
        self.samples_consumed = 0;
        self.samples_output = 0;
        self.state = if self.start_trim > 0 {
            TrimmerState::TrimmingDelay
        } else {
            TrimmerState::Playing
        };
    }

    /// Get the underlying gapless info.
    pub fn info(&self) -> &GaplessInfo {
        &self.info
    }
}

/// Cross-fade buffer for seamless transitions between tracks.
///
/// When playing back a sequence of tracks gaplessly, a short cross-fade
/// at the boundary eliminates any residual clicks from codec artifacts.
pub struct GaplessCrossfader {
    /// Cross-fade length in samples.
    fade_length: usize,
    /// Fade-out tail from the previous track.
    tail_buffer: Vec<f32>,
    /// Whether we have a tail stored.
    has_tail: bool,
}

impl GaplessCrossfader {
    /// Create a new cross-fader.
    ///
    /// # Arguments
    ///
    /// * `fade_length_samples` - Number of samples for the cross-fade
    pub fn new(fade_length_samples: usize) -> Self {
        Self {
            fade_length: fade_length_samples.max(1),
            tail_buffer: vec![0.0; fade_length_samples.max(1)],
            has_tail: false,
        }
    }

    /// Create a cross-fader from a duration in seconds.
    ///
    /// # Arguments
    ///
    /// * `duration_secs` - Cross-fade duration in seconds
    /// * `sample_rate` - Sample rate in Hz
    pub fn from_duration(duration_secs: f64, sample_rate: u32) -> Self {
        let fade_length = (duration_secs * f64::from(sample_rate)).round() as usize;
        Self::new(fade_length)
    }

    /// Store the tail of a finishing track for cross-fading.
    ///
    /// Call this with the last `fade_length` samples of the outgoing track.
    pub fn store_tail(&mut self, samples: &[f32]) {
        let copy_len = samples.len().min(self.fade_length);
        let src_offset = samples.len().saturating_sub(self.fade_length);

        self.tail_buffer.fill(0.0);
        for (i, &s) in samples[src_offset..src_offset + copy_len]
            .iter()
            .enumerate()
        {
            self.tail_buffer[i] = s;
        }
        self.has_tail = true;
    }

    /// Apply cross-fade to the beginning of the next track.
    ///
    /// Modifies the first `fade_length` samples of the incoming buffer
    /// by blending with the stored tail.
    pub fn apply_crossfade(&mut self, samples: &mut [f32]) {
        if !self.has_tail {
            return;
        }

        let fade_samples = samples.len().min(self.fade_length);

        for i in 0..fade_samples {
            let fade_in = i as f32 / self.fade_length as f32;
            let fade_out = 1.0 - fade_in;

            samples[i] = samples[i] * fade_in + self.tail_buffer[i] * fade_out;
        }

        self.has_tail = false;
    }

    /// Get the cross-fade length in samples.
    pub fn fade_length(&self) -> usize {
        self.fade_length
    }

    /// Reset the cross-fader, discarding any stored tail.
    pub fn reset(&mut self) {
        self.tail_buffer.fill(0.0);
        self.has_tail = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- GaplessInfo tests ---

    #[test]
    fn test_gapless_info_default() {
        let info = GaplessInfo::default();
        assert_eq!(info.encoder_delay_samples, 0);
        assert_eq!(info.padding_samples, 0);
        assert!(info.total_samples.is_none());
    }

    #[test]
    fn test_gapless_info_opus() {
        let info = GaplessInfo::opus(312, Some(48_000 * 10));
        assert_eq!(info.codec_preskip, 312);
        assert_eq!(info.total_start_trim(), 312);
        assert_eq!(info.sample_rate, 48_000);
    }

    #[test]
    fn test_gapless_info_mp3() {
        let info = GaplessInfo::mp3(576, 1024, Some(44_100 * 60), 44_100);
        assert_eq!(info.encoder_delay_samples, 576);
        assert_eq!(info.padding_samples, 1024);
        assert_eq!(info.total_start_trim(), 576);
    }

    #[test]
    fn test_gapless_info_vorbis() {
        let info = GaplessInfo::vorbis(4096, Some(44_100 * 30), 44_100);
        assert_eq!(info.encoder_delay_samples, 4096);
        assert_eq!(info.total_start_trim(), 4096);
    }

    #[test]
    fn test_gapless_info_duration() {
        let info = GaplessInfo {
            total_samples: Some(48_000),
            sample_rate: 48_000,
            ..Default::default()
        };
        let dur = info.duration_secs();
        assert!((dur.unwrap_or(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gapless_info_duration_zero_sr() {
        let info = GaplessInfo {
            total_samples: Some(48_000),
            sample_rate: 0,
            ..Default::default()
        };
        assert!(info.duration_secs().is_none());
    }

    // --- GaplessTrimmer tests ---

    #[test]
    fn test_trimmer_no_trim() {
        let info = GaplessInfo::default();
        let mut trimmer = GaplessTrimmer::new(info);
        let input = vec![1.0_f32; 1000];
        let output = trimmer.process(&input);
        assert_eq!(output.len(), 1000);
        assert_eq!(trimmer.state(), TrimmerState::Playing);
    }

    #[test]
    fn test_trimmer_delay_trim() {
        let info = GaplessInfo {
            encoder_delay_samples: 100,
            total_samples: Some(500),
            ..Default::default()
        };
        let mut trimmer = GaplessTrimmer::new(info);

        // First buffer: 200 samples, first 100 should be trimmed
        let input: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let output = trimmer.process(&input);
        assert_eq!(output.len(), 100, "should trim 100 delay samples");
        assert!(
            (output[0] - 100.0).abs() < 1e-6,
            "first valid sample should be 100"
        );
    }

    #[test]
    fn test_trimmer_total_samples_limit() {
        let info = GaplessInfo {
            encoder_delay_samples: 10,
            total_samples: Some(50),
            ..Default::default()
        };
        let mut trimmer = GaplessTrimmer::new(info);

        // Buffer of 100 samples: skip first 10 (delay), output 50, ignore rest
        let input = vec![1.0_f32; 100];
        let output = trimmer.process(&input);
        assert_eq!(output.len(), 50, "should output exactly 50 valid samples");
        assert!(trimmer.is_finished());
    }

    #[test]
    fn test_trimmer_multi_buffer_trim() {
        let info = GaplessInfo {
            encoder_delay_samples: 150,
            total_samples: Some(300),
            ..Default::default()
        };
        let mut trimmer = GaplessTrimmer::new(info);

        // First buffer: 100 samples, all trimmed (still in delay)
        let buf1 = vec![0.0_f32; 100];
        let out1 = trimmer.process(&buf1);
        assert_eq!(out1.len(), 0);

        // Second buffer: 100 samples, first 50 trimmed, last 50 output
        let buf2: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out2 = trimmer.process(&buf2);
        assert_eq!(out2.len(), 50);
        assert!((out2[0] - 50.0).abs() < 1e-6);

        // Third buffer: 200 samples, output up to total limit (300 - 50 = 250)
        let buf3 = vec![1.0_f32; 200];
        let out3 = trimmer.process(&buf3);
        assert_eq!(out3.len(), 200);

        // Fourth buffer: should be limited (only 50 more allowed)
        let buf4 = vec![1.0_f32; 100];
        let out4 = trimmer.process(&buf4);
        assert_eq!(out4.len(), 50);
        assert!(trimmer.is_finished());
    }

    #[test]
    fn test_trimmer_finished_returns_empty() {
        let info = GaplessInfo {
            total_samples: Some(10),
            ..Default::default()
        };
        let mut trimmer = GaplessTrimmer::new(info);
        let _ = trimmer.process(&vec![1.0; 10]);
        assert!(trimmer.is_finished());

        let extra = trimmer.process(&vec![1.0; 100]);
        assert!(extra.is_empty());
    }

    #[test]
    fn test_trimmer_empty_input() {
        let mut trimmer = GaplessTrimmer::new(GaplessInfo::default());
        let out = trimmer.process(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_trimmer_reset() {
        let info = GaplessInfo {
            encoder_delay_samples: 10,
            total_samples: Some(100),
            ..Default::default()
        };
        let mut trimmer = GaplessTrimmer::new(info);
        let _ = trimmer.process(&vec![1.0; 200]);
        assert!(trimmer.is_finished());

        trimmer.reset();
        assert_eq!(trimmer.state(), TrimmerState::TrimmingDelay);
        assert_eq!(trimmer.samples_consumed(), 0);
        assert_eq!(trimmer.samples_output(), 0);
    }

    #[test]
    fn test_trimmer_samples_consumed() {
        let mut trimmer = GaplessTrimmer::new(GaplessInfo::default());
        let _ = trimmer.process(&vec![0.0; 500]);
        assert_eq!(trimmer.samples_consumed(), 500);
        let _ = trimmer.process(&vec![0.0; 300]);
        assert_eq!(trimmer.samples_consumed(), 800);
    }

    #[test]
    fn test_trimmer_opus_preskip() {
        let info = GaplessInfo::opus(312, Some(48_000));
        let mut trimmer = GaplessTrimmer::new(info);

        let buf = vec![0.5_f32; 1000];
        let out = trimmer.process(&buf);
        // Should trim 312 samples from the start
        assert_eq!(out.len(), 688);
    }

    // --- GaplessCrossfader tests ---

    #[test]
    fn test_crossfader_creation() {
        let cf = GaplessCrossfader::new(100);
        assert_eq!(cf.fade_length(), 100);
    }

    #[test]
    fn test_crossfader_from_duration() {
        let cf = GaplessCrossfader::from_duration(0.01, 48_000);
        assert_eq!(cf.fade_length(), 480);
    }

    #[test]
    fn test_crossfader_no_tail_no_change() {
        let mut cf = GaplessCrossfader::new(100);
        let mut samples = vec![1.0_f32; 200];
        cf.apply_crossfade(&mut samples);
        // No tail stored, so samples should be unchanged
        for s in &samples {
            assert!((*s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_crossfader_blend() {
        let mut cf = GaplessCrossfader::new(4);

        // Store tail: [1.0, 1.0, 1.0, 1.0]
        cf.store_tail(&[1.0, 1.0, 1.0, 1.0]);

        // New track starts with zeros
        let mut new_track = vec![0.0_f32; 8];
        cf.apply_crossfade(&mut new_track);

        // At sample 0: fade_in=0, fade_out=1 => 0*0 + 1*1 = 1.0
        assert!((new_track[0] - 1.0).abs() < 1e-4);
        // At sample 2: fade_in=0.5, fade_out=0.5 => 0*0.5 + 1*0.5 = 0.5
        assert!((new_track[2] - 0.5).abs() < 1e-4);
        // At sample 4 (beyond fade): unchanged (0.0)
        assert!(new_track[4].abs() < 1e-6);
    }

    #[test]
    fn test_crossfader_tail_only_uses_last_samples() {
        let mut cf = GaplessCrossfader::new(2);
        // Store a long buffer; only last 2 samples should be used
        cf.store_tail(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        assert!((cf.tail_buffer[0] - 0.4).abs() < 1e-6);
        assert!((cf.tail_buffer[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfader_reset() {
        let mut cf = GaplessCrossfader::new(10);
        cf.store_tail(&[1.0; 10]);
        assert!(cf.has_tail);
        cf.reset();
        assert!(!cf.has_tail);
        for &s in &cf.tail_buffer {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_crossfader_single_use_tail() {
        let mut cf = GaplessCrossfader::new(4);
        cf.store_tail(&[1.0; 4]);
        assert!(cf.has_tail);

        let mut buf = vec![0.0_f32; 8];
        cf.apply_crossfade(&mut buf);
        // After applying, has_tail should be false
        assert!(!cf.has_tail);

        // Applying again should have no effect
        let mut buf2 = vec![0.0_f32; 8];
        cf.apply_crossfade(&mut buf2);
        for &s in &buf2 {
            assert_eq!(s, 0.0);
        }
    }
}
