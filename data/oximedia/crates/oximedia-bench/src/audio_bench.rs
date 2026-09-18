//! Audio codec benchmarking module.
//!
//! Provides synthetic benchmarks for audio codecs (Opus, FLAC, etc.) without
//! requiring actual encoded media — all measurements are computed from
//! analytical models calibrated to real-world encoder characteristics.

use serde::{Deserialize, Serialize};

/// Identifier for audio codec configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCodecId {
    /// Opus (IETF RFC 6716) — low-latency lossy codec.
    Opus,
    /// FLAC — Free Lossless Audio Codec.
    Flac,
    /// Vorbis (Xiph.org) — lossy codec.
    Vorbis,
    /// PCM — raw uncompressed audio.
    Pcm,
}

impl AudioCodecId {
    /// Return a human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::Flac => "flac",
            Self::Vorbis => "vorbis",
            Self::Pcm => "pcm",
        }
    }

    /// Whether the codec is lossless.
    #[must_use]
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Flac | Self::Pcm)
    }

    /// Nominal bitrate range in kbps `(min, max)` for common configurations.
    #[must_use]
    pub fn nominal_bitrate_range_kbps(self) -> (f64, f64) {
        match self {
            Self::Opus => (6.0, 510.0),
            Self::Flac => (400.0, 1400.0),
            Self::Vorbis => (32.0, 500.0),
            Self::Pcm => (768.0, 4608.0), // 16-bit 48 kHz mono … 32-bit 96 kHz 6ch
        }
    }
}

impl std::fmt::Display for AudioCodecId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration for an audio benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBenchmark {
    /// Audio codec to benchmark.
    pub codec: AudioCodecId,

    /// Target sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,

    /// Number of audio channels.
    pub channels: u8,

    /// Duration of synthetic audio material to process, in seconds.
    pub duration_secs: f32,

    /// Target bitrate in kbps (used for lossy codecs; ignored for PCM).
    pub target_bitrate_kbps: Option<f64>,

    /// Frame (packet) size in milliseconds for Opus/Vorbis.
    pub frame_size_ms: Option<f32>,
}

impl AudioBenchmark {
    /// Create a new audio benchmark with the given codec and basic parameters.
    #[must_use]
    pub fn new(codec: AudioCodecId, sample_rate: u32, channels: u8, duration_secs: f32) -> Self {
        Self {
            codec,
            sample_rate,
            channels,
            duration_secs,
            target_bitrate_kbps: None,
            frame_size_ms: None,
        }
    }

    /// Set the target bitrate in kbps.
    #[must_use]
    pub fn with_target_bitrate(mut self, kbps: f64) -> Self {
        self.target_bitrate_kbps = Some(kbps);
        self
    }

    /// Set the frame size in milliseconds.
    #[must_use]
    pub fn with_frame_size_ms(mut self, ms: f32) -> Self {
        self.frame_size_ms = Some(ms);
        self
    }

    /// Return the total number of PCM samples in the benchmark stream.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        (self.sample_rate as f64 * self.duration_secs as f64) as u64 * u64::from(self.channels)
    }

    /// Run the benchmark and return a result.
    ///
    /// This method uses an analytical performance model derived from published
    /// codec benchmarks rather than invoking a real encoder, making it
    /// deterministic and dependency-free.
    #[must_use]
    pub fn run(&self) -> AudioBenchmarkResult {
        let model = CodecPerfModel::for_codec(self.codec, self.sample_rate, self.channels);

        let effective_bitrate = self
            .target_bitrate_kbps
            .unwrap_or_else(|| model.default_bitrate_kbps());

        let encode_speed = model.encode_speed_ratio(effective_bitrate, self.frame_size_ms);
        let decode_speed = model.decode_speed_ratio(effective_bitrate);
        let latency = model.latency_ms(self.frame_size_ms);

        AudioBenchmarkResult {
            codec: self.codec,
            sample_rate: self.sample_rate,
            channels: self.channels,
            duration_secs: self.duration_secs,
            encode_speed_ratio: encode_speed,
            decode_speed_ratio: decode_speed,
            bitrate_kbps: effective_bitrate,
            latency_ms: latency,
        }
    }
}

/// Result produced by an [`AudioBenchmark`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBenchmarkResult {
    /// Codec that was benchmarked.
    pub codec: AudioCodecId,

    /// Sample rate used.
    pub sample_rate: u32,

    /// Channel count.
    pub channels: u8,

    /// Duration of processed audio in seconds.
    pub duration_secs: f32,

    /// Encode speed as a multiple of real-time.
    ///
    /// `2.0` means the encoder processed 2 seconds of audio per second of
    /// wall-clock time.
    pub encode_speed_ratio: f64,

    /// Decode speed as a multiple of real-time.
    pub decode_speed_ratio: f64,

    /// Effective bitrate in kbps.
    pub bitrate_kbps: f64,

    /// One-way latency introduced by a single codec frame/packet (ms).
    pub latency_ms: f64,
}

impl AudioBenchmarkResult {
    /// Whether the encoding is "real-time capable" (speed ratio ≥ 1.0).
    #[must_use]
    pub fn is_realtime_capable(&self) -> bool {
        self.encode_speed_ratio >= 1.0
    }

    /// Compression ratio versus raw 16-bit PCM.
    #[must_use]
    pub fn compression_ratio_vs_pcm(&self) -> f64 {
        let raw_kbps = self.sample_rate as f64 * 16.0 * self.channels as f64 / 1000.0;
        if self.bitrate_kbps <= 0.0 {
            return 1.0;
        }
        raw_kbps / self.bitrate_kbps
    }
}

// ---------------------------------------------------------------------------
// Internal performance model
// ---------------------------------------------------------------------------

/// Analytical performance model for audio codecs.
///
/// Values are calibrated to representative published benchmarks.
struct CodecPerfModel {
    codec: AudioCodecId,
    sample_rate: u32,
    channels: u8,
}

impl CodecPerfModel {
    fn for_codec(codec: AudioCodecId, sample_rate: u32, channels: u8) -> Self {
        Self {
            codec,
            sample_rate,
            channels,
        }
    }

    /// Default bitrate when no target is specified.
    fn default_bitrate_kbps(&self) -> f64 {
        match self.codec {
            AudioCodecId::Opus => 64.0 * self.channels as f64,
            AudioCodecId::Flac => {
                // FLAC at 16-bit: ~50 % of raw
                self.sample_rate as f64 * 16.0 * self.channels as f64 / 1000.0 * 0.5
            }
            AudioCodecId::Vorbis => 128.0 * self.channels as f64,
            AudioCodecId::Pcm => self.sample_rate as f64 * 16.0 * self.channels as f64 / 1000.0,
        }
    }

    /// Encode speed ratio (×real-time).
    fn encode_speed_ratio(&self, bitrate_kbps: f64, frame_ms: Option<f32>) -> f64 {
        let base = match self.codec {
            // Opus encode speed is roughly proportional to: base_speed / sqrt(bitrate)
            // because higher bitrate → more complex search.  Calibrated to ~50× at 64 kbps.
            AudioCodecId::Opus => {
                let frame = frame_ms.unwrap_or(20.0);
                // Smaller frames = more overhead.
                let frame_factor = (20.0_f64 / frame as f64).sqrt().clamp(0.5, 2.0);
                50.0 / (bitrate_kbps / 64.0).sqrt() * frame_factor
            }
            // FLAC is typically ~20–30× real-time encode.
            AudioCodecId::Flac => 25.0,
            // Vorbis is ~40× at 128 kbps.
            AudioCodecId::Vorbis => 40.0 / (bitrate_kbps / 128.0).sqrt(),
            // PCM is pure memcpy — extremely fast.
            AudioCodecId::Pcm => 5000.0,
        };

        // Scale down for high channel counts (minor overhead).
        let channel_factor = 1.0 / (self.channels as f64).sqrt();
        (base * channel_factor).max(0.1)
    }

    /// Decode speed ratio (×real-time).
    fn decode_speed_ratio(&self, bitrate_kbps: f64) -> f64 {
        match self.codec {
            // Opus decode is much faster than encode (~200× real-time).
            AudioCodecId::Opus => 200.0 / (bitrate_kbps / 64.0).powf(0.25),
            // FLAC decode is ~80× real-time.
            AudioCodecId::Flac => 80.0,
            // Vorbis decode is ~100× real-time.
            AudioCodecId::Vorbis => 100.0,
            // PCM decode is essentially a memcpy.
            AudioCodecId::Pcm => 5000.0,
        }
    }

    /// One-way codec latency in milliseconds.
    fn latency_ms(&self, frame_ms: Option<f32>) -> f64 {
        match self.codec {
            AudioCodecId::Opus => frame_ms.unwrap_or(20.0) as f64,
            AudioCodecId::Flac => 0.0, // streaming, virtually zero algorithmic latency
            AudioCodecId::Vorbis => frame_ms.unwrap_or(64.0) as f64,
            AudioCodecId::Pcm => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Builder for constructing a battery of audio benchmarks.
#[derive(Default)]
pub struct AudioBenchmarkBattery {
    configs: Vec<AudioBenchmark>,
}

impl AudioBenchmarkBattery {
    /// Create an empty battery.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a benchmark to the battery.
    #[must_use]
    pub fn add(mut self, bench: AudioBenchmark) -> Self {
        self.configs.push(bench);
        self
    }

    /// Run all benchmarks and return results.
    #[must_use]
    pub fn run_all(&self) -> Vec<AudioBenchmarkResult> {
        self.configs.iter().map(|b| b.run()).collect()
    }

    /// Run all benchmarks in parallel using rayon.
    #[must_use]
    pub fn run_all_parallel(&self) -> Vec<AudioBenchmarkResult> {
        use rayon::prelude::*;
        self.configs.par_iter().map(|b| b.run()).collect()
    }

    /// Number of configurations in this battery.
    #[must_use]
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether the battery is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AudioCodecId ----

    #[test]
    fn test_codec_id_name() {
        assert_eq!(AudioCodecId::Opus.name(), "opus");
        assert_eq!(AudioCodecId::Flac.name(), "flac");
        assert_eq!(AudioCodecId::Vorbis.name(), "vorbis");
        assert_eq!(AudioCodecId::Pcm.name(), "pcm");
    }

    #[test]
    fn test_codec_is_lossless() {
        assert!(AudioCodecId::Flac.is_lossless());
        assert!(AudioCodecId::Pcm.is_lossless());
        assert!(!AudioCodecId::Opus.is_lossless());
        assert!(!AudioCodecId::Vorbis.is_lossless());
    }

    #[test]
    fn test_codec_bitrate_range_valid() {
        for codec in [
            AudioCodecId::Opus,
            AudioCodecId::Flac,
            AudioCodecId::Vorbis,
            AudioCodecId::Pcm,
        ] {
            let (lo, hi) = codec.nominal_bitrate_range_kbps();
            assert!(lo > 0.0, "{codec} lower bound should be positive");
            assert!(hi > lo, "{codec} upper bound should exceed lower bound");
        }
    }

    // ---- AudioBenchmark construction ----

    #[test]
    fn test_audio_benchmark_new() {
        let b = AudioBenchmark::new(AudioCodecId::Opus, 48000, 2, 10.0);
        assert_eq!(b.codec, AudioCodecId::Opus);
        assert_eq!(b.sample_rate, 48000);
        assert_eq!(b.channels, 2);
        assert!((b.duration_secs - 10.0).abs() < 1e-6);
        assert!(b.target_bitrate_kbps.is_none());
    }

    #[test]
    fn test_total_samples() {
        let b = AudioBenchmark::new(AudioCodecId::Opus, 48000, 2, 1.0);
        assert_eq!(b.total_samples(), 96000);
    }

    // ---- Run results ----

    #[test]
    fn test_opus_benchmark_run() {
        let result = AudioBenchmark::new(AudioCodecId::Opus, 48000, 2, 5.0)
            .with_target_bitrate(64.0)
            .run();
        assert_eq!(result.codec, AudioCodecId::Opus);
        assert!(
            result.encode_speed_ratio > 0.0,
            "encode speed must be positive"
        );
        assert!(
            result.decode_speed_ratio > 0.0,
            "decode speed must be positive"
        );
        assert!((result.bitrate_kbps - 64.0).abs() < 1e-9);
        assert!(result.latency_ms > 0.0);
    }

    #[test]
    fn test_flac_benchmark_run() {
        let result = AudioBenchmark::new(AudioCodecId::Flac, 44100, 2, 10.0).run();
        assert_eq!(result.codec, AudioCodecId::Flac);
        assert!(
            result.encode_speed_ratio > 1.0,
            "FLAC encode should be > 1× real-time"
        );
        assert!(result.decode_speed_ratio > 1.0);
        assert_eq!(result.latency_ms, 0.0);
    }

    #[test]
    fn test_pcm_passthrough_is_fast() {
        let result = AudioBenchmark::new(AudioCodecId::Pcm, 48000, 1, 1.0).run();
        assert!(
            result.encode_speed_ratio > 100.0,
            "PCM should be extremely fast"
        );
        assert_eq!(result.latency_ms, 0.0);
    }

    #[test]
    fn test_opus_higher_bitrate_lower_encode_speed() {
        let low = AudioBenchmark::new(AudioCodecId::Opus, 48000, 1, 1.0)
            .with_target_bitrate(32.0)
            .run();
        let high = AudioBenchmark::new(AudioCodecId::Opus, 48000, 1, 1.0)
            .with_target_bitrate(256.0)
            .run();
        assert!(
            low.encode_speed_ratio > high.encode_speed_ratio,
            "higher Opus bitrate should be slower to encode"
        );
    }

    // ---- Compression ratio ----

    #[test]
    fn test_compression_ratio_opus() {
        let result = AudioBenchmark::new(AudioCodecId::Opus, 48000, 1, 5.0)
            .with_target_bitrate(64.0)
            .run();
        let ratio = result.compression_ratio_vs_pcm();
        // 48000 * 16 / 1000 = 768 kbps raw; 768 / 64 = 12 ×
        assert!(
            ratio > 5.0,
            "Opus at 64 kbps should yield > 5× compression vs PCM"
        );
    }

    // ---- Battery ----

    #[test]
    fn test_battery_run_all() {
        let battery = AudioBenchmarkBattery::new()
            .add(AudioBenchmark::new(AudioCodecId::Opus, 48000, 2, 5.0).with_target_bitrate(64.0))
            .add(AudioBenchmark::new(AudioCodecId::Flac, 44100, 2, 5.0))
            .add(
                AudioBenchmark::new(AudioCodecId::Vorbis, 48000, 2, 5.0).with_target_bitrate(128.0),
            );
        assert_eq!(battery.len(), 3);
        let results = battery.run_all();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_battery_parallel_matches_sequential() {
        let battery = AudioBenchmarkBattery::new()
            .add(AudioBenchmark::new(AudioCodecId::Opus, 48000, 2, 1.0).with_target_bitrate(64.0))
            .add(AudioBenchmark::new(AudioCodecId::Flac, 44100, 2, 1.0));

        let seq = battery.run_all();
        let par = battery.run_all_parallel();
        assert_eq!(seq.len(), par.len());

        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.codec, p.codec);
            assert!((s.encode_speed_ratio - p.encode_speed_ratio).abs() < 1e-9);
        }
    }
}
