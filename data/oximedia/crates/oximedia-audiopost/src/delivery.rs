//! Delivery and export functionality for audio post-production.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Audio format for export
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// WAV (uncompressed)
    Wav,
    /// FLAC (lossless)
    Flac,
    /// MP3 (lossy)
    Mp3,
    /// AAC (lossy)
    Aac,
    /// Opus (lossy)
    Opus,
    /// Broadcast Wave Format
    Bwf,
}

impl AudioFormat {
    /// Get file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav | Self::Bwf => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Aac => "m4a",
            Self::Opus => "opus",
        }
    }

    /// Check if format is lossless
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Wav | Self::Flac | Self::Bwf)
    }
}

/// Sample rate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleRate {
    /// 44.1 kHz
    Hz44100,
    /// 48 kHz
    Hz48000,
    /// 88.2 kHz
    Hz88200,
    /// 96 kHz
    Hz96000,
    /// 176.4 kHz
    Hz176400,
    /// 192 kHz
    Hz192000,
}

impl SampleRate {
    /// Get sample rate value
    #[must_use]
    pub fn value(&self) -> u32 {
        match self {
            Self::Hz44100 => 44100,
            Self::Hz48000 => 48000,
            Self::Hz88200 => 88200,
            Self::Hz96000 => 96000,
            Self::Hz176400 => 176400,
            Self::Hz192000 => 192000,
        }
    }

    /// Create from value
    #[must_use]
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            44100 => Some(Self::Hz44100),
            48000 => Some(Self::Hz48000),
            88200 => Some(Self::Hz88200),
            96000 => Some(Self::Hz96000),
            176400 => Some(Self::Hz176400),
            192000 => Some(Self::Hz192000),
            _ => None,
        }
    }
}

/// Bit depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitDepth {
    /// 16-bit integer
    Bit16,
    /// 24-bit integer
    Bit24,
    /// 32-bit float
    Float32,
}

impl BitDepth {
    /// Get bit depth value
    #[must_use]
    pub fn value(&self) -> u16 {
        match self {
            Self::Bit16 => 16,
            Self::Bit24 => 24,
            Self::Float32 => 32,
        }
    }

    /// Check if floating point
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float32)
    }
}

/// Channel configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelConfig {
    /// Mono
    Mono,
    /// Stereo
    Stereo,
    /// 5.1 surround
    Surround51,
    /// 7.1 surround
    Surround71,
    /// 7.1.4 Atmos
    Atmos714,
}

impl ChannelConfig {
    /// Get channel count
    #[must_use]
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos714 => 12,
        }
    }
}

/// Dithering algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DitheringAlgorithm {
    /// No dithering
    None,
    /// Triangular PDF dithering
    Triangular,
    /// Rectangular PDF dithering
    Rectangular,
    /// Noise shaping
    NoiseShaping,
}

/// Normalization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationType {
    /// No normalization
    None,
    /// Peak normalization
    Peak,
    /// RMS normalization
    Rms,
    /// LUFS normalization
    Lufs,
}

/// Export settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSettings {
    /// Output file path
    pub output_path: PathBuf,
    /// Audio format
    pub format: AudioFormat,
    /// Sample rate
    pub sample_rate: SampleRate,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Channel configuration
    pub channels: ChannelConfig,
    /// Dithering algorithm
    pub dithering: DitheringAlgorithm,
    /// Normalization type
    pub normalization: NormalizationType,
    /// Normalization target
    pub normalization_target: f32,
    /// Embed metadata
    pub embed_metadata: bool,
}

impl ExportSettings {
    /// Create new export settings
    #[must_use]
    pub fn new(output_path: PathBuf) -> Self {
        Self {
            output_path,
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Stereo,
            dithering: DitheringAlgorithm::None,
            normalization: NormalizationType::None,
            normalization_target: -1.0,
            embed_metadata: true,
        }
    }

    /// Validate settings
    ///
    /// # Errors
    ///
    /// Returns an error if settings are invalid
    pub fn validate(&self) -> AudioPostResult<()> {
        // Check format compatibility
        if !self.format.is_lossless() && self.bit_depth == BitDepth::Float32 {
            return Err(AudioPostError::Generic(
                "Lossy formats do not support float32".to_string(),
            ));
        }

        // Check channel compatibility
        if matches!(self.format, AudioFormat::Mp3)
            && !matches!(self.channels, ChannelConfig::Mono | ChannelConfig::Stereo)
        {
            return Err(AudioPostError::Generic(
                "MP3 only supports mono/stereo".to_string(),
            ));
        }

        Ok(())
    }
}

/// Broadcast Wave Format metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BwfMetadata {
    /// Description
    pub description: String,
    /// Originator
    pub originator: String,
    /// Originator reference
    pub originator_reference: String,
    /// Origination date (YYYY-MM-DD)
    pub origination_date: String,
    /// Origination time (HH:MM:SS)
    pub origination_time: String,
    /// Time reference (samples since midnight)
    pub time_reference: u64,
    /// Coding history
    pub coding_history: String,
}

impl BwfMetadata {
    /// Create new BWF metadata
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set current date and time
    pub fn set_current_datetime(&mut self) {
        let now = chrono::Utc::now();
        self.origination_date = now.format("%Y-%m-%d").to_string();
        self.origination_time = now.format("%H:%M:%S").to_string();
    }
}

/// iXML metadata for production
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IxmlMetadata {
    /// Project name
    pub project: String,
    /// Scene name
    pub scene: String,
    /// Take number
    pub take: String,
    /// Tape name
    pub tape: String,
    /// File UID
    pub file_uid: String,
    /// Track list
    pub tracks: Vec<TrackMetadata>,
}

/// Track metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadata {
    /// Track number
    pub number: u32,
    /// Track name
    pub name: String,
    /// Channel index
    pub channel_index: u32,
}

impl TrackMetadata {
    /// Create new track metadata
    #[must_use]
    pub fn new(number: u32, name: &str, channel_index: u32) -> Self {
        Self {
            number,
            name: name.to_string(),
            channel_index,
        }
    }
}

/// Deliverable specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliverableSpec {
    /// Specification name
    pub name: String,
    /// Audio format
    pub format: AudioFormat,
    /// Sample rate
    pub sample_rate: SampleRate,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Channel configuration
    pub channels: ChannelConfig,
    /// Loudness target (LUFS)
    pub loudness_target: Option<f32>,
    /// Maximum true peak (dBTP)
    pub max_true_peak: Option<f32>,
    /// Required metadata
    pub require_metadata: bool,
}

impl DeliverableSpec {
    /// Netflix specification
    #[must_use]
    pub fn netflix() -> Self {
        Self {
            name: "Netflix".to_string(),
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Surround51,
            loudness_target: Some(-27.0),
            max_true_peak: Some(-2.0),
            require_metadata: true,
        }
    }

    /// Theatrical DCP specification
    #[must_use]
    pub fn theatrical_dcp() -> Self {
        Self {
            name: "Theatrical DCP".to_string(),
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Surround51,
            loudness_target: Some(-27.0),
            max_true_peak: Some(-10.0),
            require_metadata: false,
        }
    }

    /// Broadcast specification
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            name: "Broadcast".to_string(),
            format: AudioFormat::Bwf,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Stereo,
            loudness_target: Some(-23.0),
            max_true_peak: Some(-1.0),
            require_metadata: true,
        }
    }

    /// Streaming specification
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            name: "Streaming".to_string(),
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Stereo,
            loudness_target: Some(-14.0),
            max_true_peak: Some(-1.0),
            require_metadata: false,
        }
    }
}

/// Exporter for creating deliverables
#[derive(Debug)]
pub struct Exporter {
    settings: ExportSettings,
}

impl Exporter {
    /// Create a new exporter
    #[must_use]
    pub fn new(settings: ExportSettings) -> Self {
        Self { settings }
    }

    /// Validate export settings
    ///
    /// # Errors
    ///
    /// Returns an error if settings are invalid
    pub fn validate(&self) -> AudioPostResult<()> {
        self.settings.validate()
    }

    /// Export audio
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export(&self, _audio: &[f32]) -> AudioPostResult<()> {
        self.validate()?;
        // Placeholder - real implementation would write audio file
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_extension() {
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
    }

    #[test]
    fn test_audio_format_lossless() {
        assert!(AudioFormat::Wav.is_lossless());
        assert!(AudioFormat::Flac.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
    }

    #[test]
    fn test_sample_rate_value() {
        assert_eq!(SampleRate::Hz48000.value(), 48000);
        assert_eq!(SampleRate::Hz96000.value(), 96000);
    }

    #[test]
    fn test_sample_rate_from_value() {
        assert_eq!(SampleRate::from_value(48000), Some(SampleRate::Hz48000));
        assert_eq!(SampleRate::from_value(12345), None);
    }

    #[test]
    fn test_bit_depth_value() {
        assert_eq!(BitDepth::Bit16.value(), 16);
        assert_eq!(BitDepth::Bit24.value(), 24);
        assert_eq!(BitDepth::Float32.value(), 32);
    }

    #[test]
    fn test_bit_depth_is_float() {
        assert!(!BitDepth::Bit16.is_float());
        assert!(BitDepth::Float32.is_float());
    }

    #[test]
    fn test_channel_config_count() {
        assert_eq!(ChannelConfig::Mono.channel_count(), 1);
        assert_eq!(ChannelConfig::Stereo.channel_count(), 2);
        assert_eq!(ChannelConfig::Surround51.channel_count(), 6);
    }

    #[test]
    fn test_export_settings_new() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        assert_eq!(settings.format, AudioFormat::Wav);
        assert_eq!(settings.sample_rate, SampleRate::Hz48000);
    }

    #[test]
    fn test_export_settings_validate() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_invalid_lossy_float() {
        let mut settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.mp3"));
        settings.format = AudioFormat::Mp3;
        settings.bit_depth = BitDepth::Float32;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_bwf_metadata() {
        let mut metadata = BwfMetadata::new();
        metadata.description = "Test audio".to_string();
        metadata.set_current_datetime();
        assert!(!metadata.origination_date.is_empty());
    }

    #[test]
    fn test_ixml_metadata() {
        let mut metadata = IxmlMetadata::default();
        metadata.project = "Test Project".to_string();
        metadata.tracks.push(TrackMetadata::new(1, "Boom", 0));
        assert_eq!(metadata.tracks.len(), 1);
    }

    #[test]
    fn test_deliverable_spec_netflix() {
        let spec = DeliverableSpec::netflix();
        assert_eq!(spec.name, "Netflix");
        assert_eq!(spec.loudness_target, Some(-27.0));
    }

    #[test]
    fn test_deliverable_spec_broadcast() {
        let spec = DeliverableSpec::broadcast();
        assert_eq!(spec.format, AudioFormat::Bwf);
        assert_eq!(spec.loudness_target, Some(-23.0));
    }

    #[test]
    fn test_exporter_creation() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        let exporter = Exporter::new(settings);
        assert!(exporter.validate().is_ok());
    }

    #[test]
    fn test_exporter_export() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        let exporter = Exporter::new(settings);
        let audio = vec![0.0_f32; 1000];
        assert!(exporter.export(&audio).is_ok());
    }

    #[test]
    fn test_track_metadata() {
        let track = TrackMetadata::new(1, "Dialogue", 0);
        assert_eq!(track.number, 1);
        assert_eq!(track.name, "Dialogue");
    }
}

// ── Sample-rate conversion ────────────────────────────────────────────────────

/// Convert `samples` from `from_rate` Hz to `to_rate` Hz using the Pure-Rust
/// band-limited windowed-sinc polyphase resampler from `oximedia-audio`
/// ([`oximedia_audio::Resampler`], `High` quality: 192-tap kernel with fine
/// phase resolution).
///
/// The resampler compensates its own group delay and the stream tail is
/// drained via `flush()`, so the output is time-aligned with the input and
/// has length ≈ `samples.len() * to_rate / from_rate`.
///
/// When the rates are equal the input slice is returned unchanged (zero-copy
/// clone).  Mono-only: the function treats `samples` as a single channel.
///
/// # Errors
///
/// Returns [`AudioPostError::InvalidSampleRate`] for a zero `from_rate` or
/// `to_rate`, and [`AudioPostError::InvalidBufferSize`] for an empty input.
/// Internal resampler errors are wrapped in [`AudioPostError::Generic`].
#[allow(clippy::cast_precision_loss)]
pub fn convert_sample_rate(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> AudioPostResult<Vec<f32>> {
    if from_rate == 0 {
        return Err(AudioPostError::InvalidSampleRate(from_rate));
    }
    if to_rate == 0 {
        return Err(AudioPostError::InvalidSampleRate(to_rate));
    }
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }

    // Identity case – avoid any allocation beyond the clone.
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout, Resampler, ResamplerQuality};
    use oximedia_core::SampleFormat;

    let mut resampler = Resampler::new(from_rate, to_rate, 1, ResamplerQuality::High)
        .map_err(|e| AudioPostError::Generic(format!("Resampler creation failed: {e}")))?;

    // Wrap the mono input in an F32 interleaved frame (little-endian bytes).
    let mut input_bytes = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        input_bytes.extend_from_slice(&s.to_le_bytes());
    }
    let mut frame = AudioFrame::new(SampleFormat::F32, from_rate, ChannelLayout::Mono);
    frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(input_bytes));

    // Extract mono f32 samples from an F32 interleaved/planar frame.
    fn frame_to_f32(frame: &AudioFrame) -> Vec<f32> {
        let data = match &frame.samples {
            AudioBuffer::Interleaved(data) => data.as_ref(),
            AudioBuffer::Planar(planes) => planes.first().map_or(&[][..], |p| p.as_ref()),
        };
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // One-shot: process the whole signal, then drain the filter tail.
    let main = resampler
        .resample(&frame)
        .map_err(|e| AudioPostError::Generic(format!("Resampling failed: {e}")))?;
    let tail = resampler
        .flush()
        .map_err(|e| AudioPostError::Generic(format!("Resampler flush failed: {e}")))?;

    let mut output = frame_to_f32(&main);
    output.extend(frame_to_f32(&tail));

    Ok(output)
}

// ── Loudness normalisation ─────────────────────────────────────────────────────

/// Configuration for EBU R128 loudness normalisation.
#[derive(Debug, Clone)]
pub struct DeliveryConfig {
    /// Target integrated loudness in LUFS (default −23.0 for broadcast).
    pub target_lufs: f32,
    /// Maximum true-peak level in dBTP (default −1.0).
    pub max_true_peak_dbtp: f32,
    /// Tolerance window around the target (default 0.5 LU).
    pub tolerance_lu: f32,
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            target_lufs: -23.0,
            max_true_peak_dbtp: -1.0,
            tolerance_lu: 0.5,
        }
    }
}

/// Normalise `samples` to the integrated loudness specified in `config`.
///
/// The function:
/// 1. Measures the current integrated LUFS via EBU R128 gated metering.
/// 2. Computes the linear gain required to reach `config.target_lufs`.
/// 3. Applies the gain, then clamps the output so no sample exceeds
///    the linear amplitude corresponding to `config.max_true_peak_dbtp`.
///
/// If the input is already within `config.tolerance_lu` of the target,
/// the audio is returned unchanged.
///
/// # Errors
///
/// Returns [`AudioPostError::InvalidSampleRate`] for a zero sample rate or
/// [`AudioPostError::InvalidBufferSize`] for empty input.
/// Signals that are too short or too quiet for EBU R128 gating are passed
/// through unchanged (not treated as errors).
#[allow(clippy::cast_precision_loss)]
pub fn normalize_loudness(
    samples: &[f32],
    sample_rate: u32,
    config: &DeliveryConfig,
) -> AudioPostResult<Vec<f32>> {
    if sample_rate == 0 {
        return Err(AudioPostError::InvalidSampleRate(sample_rate));
    }
    if samples.is_empty() {
        return Err(AudioPostError::InvalidBufferSize(0));
    }

    // Measure current integrated LUFS.
    let measurement =
        oximedia_audio::loudness_gating::GatedLoudnessMeter::measure(samples, sample_rate, 1);
    let current_lufs = measurement.integrated_lufs as f32;

    // If the signal is too short or silent for reliable gating (LUFS is non-finite
    // or below -70 LUFS), pass through unchanged rather than propagating an error.
    // Loudness normalisation requires at least one 400 ms gating block to be meaningful.
    if !current_lufs.is_finite() || current_lufs < -70.0 {
        return Ok(samples.to_vec());
    }

    // Skip if already within tolerance.
    let delta = (current_lufs - config.target_lufs).abs();
    if delta <= config.tolerance_lu {
        return Ok(samples.to_vec());
    }

    // Compute linear gain from dB difference.
    let gain_db = config.target_lufs - current_lufs;
    let gain = 10.0_f32.powf(gain_db / 20.0);

    // Maximum linear amplitude permitted by true-peak ceiling.
    let peak_ceiling = 10.0_f32.powf(config.max_true_peak_dbtp / 20.0);

    let output: Vec<f32> = samples
        .iter()
        .map(|&s| (s * gain).clamp(-peak_ceiling, peak_ceiling))
        .collect();

    Ok(output)
}

#[cfg(test)]
mod resample_tests {
    use super::*;

    /// Generate `n` samples of a unit-amplitude sine at `freq` Hz / `rate` Hz.
    fn sine(freq: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / f64::from(rate);
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32
            })
            .collect()
    }

    /// Least-squares fit of `a·sin(ωt) + b·cos(ωt)` at a known frequency over
    /// `y[skip..len-skip]`; returns `(amplitude, snr_db)` of the fit.
    #[allow(clippy::cast_precision_loss)]
    fn sine_fit(y: &[f32], freq: f64, rate: u32, skip: usize) -> (f64, f64) {
        assert!(y.len() > 2 * skip + 16, "signal too short for sine fit");
        let seg = &y[skip..y.len() - skip];
        let w = 2.0 * std::f64::consts::PI * freq / f64::from(rate);

        let (mut s_ss, mut s_cc, mut s_sc) = (0.0f64, 0.0f64, 0.0f64);
        let (mut r_s, mut r_c) = (0.0f64, 0.0f64);
        for (i, &yv) in seg.iter().enumerate() {
            let t = (skip + i) as f64;
            let (s, c) = ((w * t).sin(), (w * t).cos());
            let yf = f64::from(yv);
            s_ss += s * s;
            s_cc += c * c;
            s_sc += s * c;
            r_s += yf * s;
            r_c += yf * c;
        }
        let det = s_ss * s_cc - s_sc * s_sc;
        assert!(det.abs() > 1e-9, "degenerate sine fit");
        let a = (r_s * s_cc - r_c * s_sc) / det;
        let b = (r_c * s_ss - r_s * s_sc) / det;

        let (mut sig, mut noise) = (0.0f64, 0.0f64);
        for (i, &yv) in seg.iter().enumerate() {
            let t = (skip + i) as f64;
            let fit = a * (w * t).sin() + b * (w * t).cos();
            sig += fit * fit;
            noise += (f64::from(yv) - fit) * (f64::from(yv) - fit);
        }
        let amplitude = (a * a + b * b).sqrt();
        let snr_db = 10.0 * (sig / noise.max(1e-30)).log10();
        (amplitude, snr_db)
    }

    #[test]
    fn test_convert_sample_rate_identity() {
        let input = sine(1000.0, 48_000, 4800);
        let out = match convert_sample_rate(&input, 48_000, 48_000) {
            Ok(v) => v,
            Err(e) => panic!("identity conversion failed: {e}"),
        };
        assert_eq!(out, input);
    }

    #[test]
    fn test_convert_sample_rate_rejects_invalid_input() {
        assert!(convert_sample_rate(&[0.0; 8], 0, 44_100).is_err());
        assert!(convert_sample_rate(&[0.0; 8], 48_000, 0).is_err());
        assert!(convert_sample_rate(&[], 48_000, 44_100).is_err());
    }

    #[test]
    fn test_convert_sample_rate_1khz_48k_to_44k1_preserves_tone() {
        let input = sine(1000.0, 48_000, 48_000); // 1 second
        let out = match convert_sample_rate(&input, 48_000, 44_100) {
            Ok(v) => v,
            Err(e) => panic!("48k->44.1k conversion failed: {e}"),
        };

        // Length must approximate len * 44100/48000 = 44100 (filter edges may
        // add/remove a handful of samples).
        let expected = 44_100_i64;
        #[allow(clippy::cast_possible_wrap)]
        let got = out.len() as i64;
        assert!(
            (got - expected).abs() <= 256,
            "unexpected output length: got {got}, expected ~{expected}"
        );

        // The 1 kHz tone must survive with unit amplitude and high purity.
        let (amplitude, snr_db) = sine_fit(&out, 1000.0, 44_100, 1000);
        assert!(
            (amplitude - 1.0).abs() < 0.02,
            "amplitude not preserved: {amplitude}"
        );
        assert!(snr_db > 60.0, "tone degraded: SNR {snr_db:.1} dB");
    }

    #[test]
    fn test_convert_sample_rate_round_trip_48k_44k1_48k() {
        let input = sine(1000.0, 48_000, 48_000); // 1 second
        let down = match convert_sample_rate(&input, 48_000, 44_100) {
            Ok(v) => v,
            Err(e) => panic!("48k->44.1k conversion failed: {e}"),
        };
        let up = match convert_sample_rate(&down, 44_100, 48_000) {
            Ok(v) => v,
            Err(e) => panic!("44.1k->48k conversion failed: {e}"),
        };

        // Round-trip length within tolerance of the original.
        #[allow(clippy::cast_possible_wrap)]
        let (got, expected) = (up.len() as i64, input.len() as i64);
        assert!(
            (got - expected).abs() <= 512,
            "round-trip length drifted: got {got}, expected ~{expected}"
        );

        // The resampler is group-delay compensated, so mid-signal samples must
        // match the original sine directly (both passes are time-aligned).
        let (amplitude, snr_db) = sine_fit(&up, 1000.0, 48_000, 1500);
        assert!(
            (amplitude - 1.0).abs() < 0.02,
            "round-trip amplitude drifted: {amplitude}"
        );
        assert!(snr_db > 55.0, "round-trip degraded: SNR {snr_db:.1} dB");

        let n = up.len().min(input.len());
        let mut max_err = 0.0f32;
        for i in 2000..n.saturating_sub(2000) {
            max_err = max_err.max((up[i] - input[i]).abs());
        }
        assert!(
            max_err < 0.02,
            "round-trip not time-aligned within tolerance: max_err {max_err}"
        );
    }
}
