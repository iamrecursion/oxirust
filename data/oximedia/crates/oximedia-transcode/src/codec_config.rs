//! Codec-specific configuration and optimization.

use crate::{QualityMode, QualityPreset, RateControlMode, TuneMode};
use serde::{Deserialize, Serialize};

/// Codec-specific configuration.
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Codec name.
    pub codec: String,
    /// Preset (speed/quality tradeoff).
    pub preset: QualityPreset,
    /// Tune for specific content.
    pub tune: Option<TuneMode>,
    /// Profile.
    pub profile: Option<String>,
    /// Level.
    pub level: Option<String>,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Additional options.
    pub options: Vec<(String, String)>,
}

impl CodecConfig {
    /// Creates a new codec configuration.
    #[must_use]
    pub fn new(codec: impl Into<String>) -> Self {
        Self {
            codec: codec.into(),
            preset: QualityPreset::Medium,
            tune: None,
            profile: None,
            level: None,
            rate_control: RateControlMode::Crf(23),
            options: Vec::new(),
        }
    }

    /// Sets the preset.
    #[must_use]
    pub fn preset(mut self, preset: QualityPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Sets the tune mode.
    #[must_use]
    pub fn tune(mut self, tune: TuneMode) -> Self {
        self.tune = Some(tune);
        self
    }

    /// Sets the profile.
    #[must_use]
    pub fn profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Sets the level.
    #[must_use]
    pub fn level(mut self, level: impl Into<String>) -> Self {
        self.level = Some(level.into());
        self
    }

    /// Sets the rate control mode.
    #[must_use]
    pub fn rate_control(mut self, mode: RateControlMode) -> Self {
        self.rate_control = mode;
        self
    }

    /// Adds a custom option.
    #[must_use]
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.push((key.into(), value.into()));
        self
    }
}

/// H.264/AVC specific configuration.
#[derive(Debug, Clone)]
pub struct H264Config {
    base: CodecConfig,
}

impl H264Config {
    /// Creates a new H.264 configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: CodecConfig::new("h264"),
        }
    }

    /// Sets the profile (baseline, main, high, high10, high422, high444).
    #[must_use]
    pub fn profile(mut self, profile: H264Profile) -> Self {
        self.base.profile = Some(profile.as_str().to_string());
        self
    }

    /// Sets the level (e.g., "3.0", "4.0", "5.1").
    #[must_use]
    pub fn level(mut self, level: impl Into<String>) -> Self {
        self.base.level = Some(level.into());
        self
    }

    /// Enables cabac entropy coding.
    #[must_use]
    pub fn cabac(mut self, enable: bool) -> Self {
        self.base.options.push((
            "cabac".to_string(),
            if enable { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Sets the number of reference frames.
    #[must_use]
    pub fn refs(mut self, refs: u8) -> Self {
        self.base
            .options
            .push(("refs".to_string(), refs.to_string()));
        self
    }

    /// Sets the number of B-frames.
    #[must_use]
    pub fn bframes(mut self, bframes: u8) -> Self {
        self.base
            .options
            .push(("bframes".to_string(), bframes.to_string()));
        self
    }

    /// Enables 8x8 DCT.
    #[must_use]
    pub fn dct8x8(mut self, enable: bool) -> Self {
        self.base.options.push((
            "8x8dct".to_string(),
            if enable { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Sets the deblocking filter parameters.
    #[must_use]
    pub fn deblock(mut self, alpha: i8, beta: i8) -> Self {
        self.base
            .options
            .push(("deblock".to_string(), format!("{alpha}:{beta}")));
        self
    }

    /// Converts to base codec config.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        self.base
    }
}

impl Default for H264Config {
    fn default() -> Self {
        Self::new()
    }
}

/// H.264 profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264Profile {
    /// Baseline profile.
    Baseline,
    /// Main profile.
    Main,
    /// High profile.
    High,
    /// High 10 profile (10-bit).
    High10,
    /// High 4:2:2 profile.
    High422,
    /// High 4:4:4 profile.
    High444,
}

impl H264Profile {
    #[must_use]
    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Main => "main",
            Self::High => "high",
            Self::High10 => "high10",
            Self::High422 => "high422",
            Self::High444 => "high444",
        }
    }
}

/// VP9 specific configuration.
#[derive(Debug, Clone)]
pub struct Vp9Config {
    base: CodecConfig,
}

impl Vp9Config {
    /// Creates a new VP9 configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: CodecConfig::new("vp9"),
        }
    }

    /// Creates a VP9 configuration in CRF (constant quality) mode.
    ///
    /// CRF range for VP9 is 0–63; lower values produce higher quality.
    /// Typical values: 31 (good quality), 33 (balanced), 41 (lower quality).
    #[must_use]
    pub fn crf(crf_value: u8) -> Self {
        let mut cfg = Self::new();
        cfg.base.rate_control = RateControlMode::Crf(crf_value);
        cfg
    }

    /// Sets the CPU used (0-8, lower = slower/better).
    #[must_use]
    pub fn cpu_used(mut self, cpu_used: u8) -> Self {
        self.base
            .options
            .push(("cpu-used".to_string(), cpu_used.to_string()));
        self
    }

    /// Sets the tile columns for parallel encoding (0–6, value is log2 of column count).
    #[must_use]
    pub fn tile_columns(mut self, columns: u8) -> Self {
        self.base
            .options
            .push(("tile-columns".to_string(), columns.to_string()));
        self
    }

    /// Sets the tile columns for parallel encoding via builder pattern (0–6).
    #[must_use]
    pub fn with_tile_columns(mut self, cols: u8) -> Self {
        self.base
            .options
            .push(("tile-columns".to_string(), cols.to_string()));
        self
    }

    /// Sets the tile rows (for parallel encoding).
    #[must_use]
    pub fn tile_rows(mut self, rows: u8) -> Self {
        self.base
            .options
            .push(("tile-rows".to_string(), rows.to_string()));
        self
    }

    /// Sets the frame parallel encoding.
    #[must_use]
    pub fn frame_parallel(mut self, enable: bool) -> Self {
        self.base.options.push((
            "frame-parallel".to_string(),
            if enable { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Enables or disables frame-parallel decoding hint via builder pattern.
    #[must_use]
    pub fn with_frame_parallel(mut self, enabled: bool) -> Self {
        self.base.options.push((
            "frame-parallel".to_string(),
            if enabled { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Sets the auto alt reference frames.
    #[must_use]
    pub fn auto_alt_ref(mut self, frames: u8) -> Self {
        self.base
            .options
            .push(("auto-alt-ref".to_string(), frames.to_string()));
        self
    }

    /// Sets the lag in frames (0–25).
    ///
    /// Larger values allow better quality at the cost of encoding latency.
    #[must_use]
    pub fn lag_in_frames(mut self, lag: u32) -> Self {
        self.base
            .options
            .push(("lag-in-frames".to_string(), lag.to_string()));
        self
    }

    /// Sets lag in frames via builder pattern (0–25).
    #[must_use]
    pub fn with_lag_in_frames(mut self, frames: u8) -> Self {
        self.base
            .options
            .push(("lag-in-frames".to_string(), frames.to_string()));
        self
    }

    /// Enables row-based multi-threading.
    #[must_use]
    pub fn row_mt(mut self, enable: bool) -> Self {
        self.base.options.push((
            "row-mt".to_string(),
            if enable { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Enables or disables row-based multi-threading via builder pattern.
    #[must_use]
    pub fn with_row_mt(mut self, enabled: bool) -> Self {
        self.base.options.push((
            "row-mt".to_string(),
            if enabled { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Screen content encoding preset.
    ///
    /// Optimised for screen recordings: fast cpu-used, tile columns for parallelism,
    /// and row-mt for throughput. Uses CRF 33 as a balanced starting point.
    #[must_use]
    pub fn screen_content() -> Self {
        let mut cfg = Self::crf(33);
        cfg.base
            .options
            .push(("cpu-used".to_string(), "5".to_string()));
        cfg.base
            .options
            .push(("tile-columns".to_string(), "2".to_string()));
        cfg.base
            .options
            .push(("row-mt".to_string(), "1".to_string()));
        cfg.base
            .options
            .push(("lag-in-frames".to_string(), "0".to_string()));
        cfg
    }

    /// Converts to base codec config.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        self.base
    }
}

impl Default for Vp9Config {
    fn default() -> Self {
        Self::new()
    }
}

/// AV1 specific configuration.
#[derive(Debug, Clone)]
pub struct Av1Config {
    base: CodecConfig,
}

impl Av1Config {
    /// Creates a new AV1 configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: CodecConfig::new("av1"),
        }
    }

    /// Sets the CPU used (0-8, lower = slower/better).
    #[must_use]
    pub fn cpu_used(mut self, cpu_used: u8) -> Self {
        self.base
            .options
            .push(("cpu-used".to_string(), cpu_used.to_string()));
        self
    }

    /// Sets the tile columns.
    #[must_use]
    pub fn tiles(mut self, columns: u8, rows: u8) -> Self {
        self.base
            .options
            .push(("tiles".to_string(), format!("{columns}x{rows}")));
        self
    }

    /// Enables row-based multi-threading.
    #[must_use]
    pub fn row_mt(mut self, enable: bool) -> Self {
        self.base.options.push((
            "row-mt".to_string(),
            if enable { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Sets the usage mode (good, realtime).
    #[must_use]
    pub fn usage(mut self, usage: Av1Usage) -> Self {
        self.base
            .options
            .push(("usage".to_string(), usage.as_str().to_string()));
        self
    }

    /// Enables film grain synthesis.
    #[must_use]
    pub fn enable_film_grain(mut self, enable: bool) -> Self {
        self.base.options.push((
            "enable-film-grain".to_string(),
            if enable { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Converts to base codec config.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        self.base
    }
}

impl Default for Av1Config {
    fn default() -> Self {
        Self::new()
    }
}

/// AV1 usage modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1Usage {
    /// Good quality mode.
    Good,
    /// Real-time mode.
    Realtime,
}

impl Av1Usage {
    #[must_use]
    fn as_str(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Realtime => "realtime",
        }
    }
}

/// Opus audio codec configuration.
#[derive(Debug, Clone)]
pub struct OpusConfig {
    base: CodecConfig,
}

impl OpusConfig {
    /// Creates a new Opus configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: CodecConfig::new("opus"),
        }
    }

    /// Voice/VOIP optimised preset.
    ///
    /// Uses VOIP application mode, maximum complexity, VBR, and forward error
    /// correction — suitable for speech transmission over lossy networks.
    #[must_use]
    pub fn voice() -> Self {
        Self::new()
            .application(OpusApplication::Voip)
            .complexity(10)
            .vbr(true)
            .with_fec(true)
    }

    /// Music streaming preset.
    ///
    /// Uses Audio application mode, maximum complexity, VBR, and a 20 ms frame
    /// duration — optimal balance for music with transparent quality.
    #[must_use]
    pub fn music() -> Self {
        Self::new()
            .application(OpusApplication::Audio)
            .complexity(10)
            .vbr(true)
            .frame_duration(20.0)
    }

    /// Full-band (20 Hz–20 kHz) audio preset.
    ///
    /// Forces full-band mode with maximum complexity and VBR.
    #[must_use]
    pub fn fullband() -> Self {
        let mut cfg = Self::new()
            .application(OpusApplication::Audio)
            .complexity(10)
            .vbr(true);
        cfg.base
            .options
            .push(("cutoff".to_string(), "20000".to_string()));
        cfg
    }

    /// Sets the application type.
    #[must_use]
    pub fn application(mut self, app: OpusApplication) -> Self {
        self.base
            .options
            .push(("application".to_string(), app.as_str().to_string()));
        self
    }

    /// Sets the complexity (0-10).
    #[must_use]
    pub fn complexity(mut self, complexity: u8) -> Self {
        self.base
            .options
            .push(("complexity".to_string(), complexity.to_string()));
        self
    }

    /// Sets the frame duration in milliseconds.
    #[must_use]
    pub fn frame_duration(mut self, duration_ms: f32) -> Self {
        self.base
            .options
            .push(("frame_duration".to_string(), duration_ms.to_string()));
        self
    }

    /// Enables variable bitrate.
    #[must_use]
    pub fn vbr(mut self, enable: bool) -> Self {
        self.base.options.push((
            "vbr".to_string(),
            if enable { "on" } else { "off" }.to_string(),
        ));
        self
    }

    /// Enables or disables variable bitrate via builder pattern.
    #[must_use]
    pub fn with_vbr(mut self, enabled: bool) -> Self {
        self.base.options.push((
            "vbr".to_string(),
            if enabled { "on" } else { "off" }.to_string(),
        ));
        self
    }

    /// Enables or disables constrained VBR mode.
    ///
    /// Constrained VBR limits bitrate peaks while still allowing variation,
    /// giving better quality than strict CBR with bounded bitrate.
    #[must_use]
    pub fn with_constrained_vbr(mut self, enabled: bool) -> Self {
        self.base.options.push((
            "cvbr".to_string(),
            if enabled { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Enables or disables Discontinuous Transmission (DTX).
    ///
    /// DTX reduces bitrate during silence by sending comfort noise packets,
    /// useful for VOIP applications where silence is frequent.
    #[must_use]
    pub fn with_dtx(mut self, enabled: bool) -> Self {
        self.base.options.push((
            "dtx".to_string(),
            if enabled { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Enables or disables in-band Forward Error Correction (FEC).
    ///
    /// FEC adds redundant audio data that allows partial recovery from packet
    /// loss in VoIP scenarios. Increases bitrate slightly.
    #[must_use]
    pub fn with_fec(mut self, enabled: bool) -> Self {
        self.base.options.push((
            "inband_fec".to_string(),
            if enabled { "1" } else { "0" }.to_string(),
        ));
        self
    }

    /// Sets the expected packet loss percentage (0–100).
    ///
    /// This hint guides the encoder in tuning FEC strength and bitrate
    /// distribution. Requires FEC to be enabled for full effect.
    #[must_use]
    pub fn with_packet_loss_perc(mut self, pct: u8) -> Self {
        self.base
            .options
            .push(("packet_loss_perc".to_string(), pct.to_string()));
        self
    }

    /// Converts to base codec config.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        self.base
    }
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Opus application types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpusApplication {
    /// Voice over IP.
    Voip,
    /// Audio streaming.
    Audio,
    /// Low delay.
    LowDelay,
}

impl OpusApplication {
    #[must_use]
    fn as_str(self) -> &'static str {
        match self {
            Self::Voip => "voip",
            Self::Audio => "audio",
            Self::LowDelay => "lowdelay",
        }
    }
}

/// FFV1 lossless codec levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ffv1Level {
    /// FFV1 Level 1 (older, limited features).
    Level1,
    /// FFV1 Level 3 (modern, slice-based, multithreaded).
    Level3,
}

impl Ffv1Level {
    /// Returns the integer level value.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Level1 => 1,
            Self::Level3 => 3,
        }
    }
}

/// FFV1 entropy coder selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ffv1Coder {
    /// Golomb-Rice entropy coding (faster, less compression).
    GolombRice,
    /// Range (ANS) entropy coding (better compression, slightly slower).
    Range,
}

impl Ffv1Coder {
    /// Returns the integer coder value used by the encoder.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        match self {
            Self::GolombRice => 0,
            Self::Range => 1,
        }
    }
}

/// FFV1 lossless video codec configuration.
#[derive(Debug, Clone)]
pub struct Ffv1Config {
    /// FFV1 level (1 or 3).
    pub level: Ffv1Level,
    /// Entropy coder selection.
    pub coder: Ffv1Coder,
    /// Number of slices for multithreaded encoding (1, 4, 9, 16, 24).
    pub slice_count: u8,
    /// Context model complexity (0=simple, 1=complex/better compression).
    pub context_model: u8,
    /// Enable per-slice CRC checksums for error detection.
    pub checksum: bool,
}

impl Ffv1Config {
    /// Creates a new FFV1 configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            level: Ffv1Level::Level3,
            coder: Ffv1Coder::Range,
            slice_count: 4,
            context_model: 0,
            checksum: true,
        }
    }

    /// Best-compression archival preset.
    ///
    /// Uses Level 3, Range coder, 16 slices, complex context model, and checksums.
    /// Ideal for long-term preservation where file size and integrity matter most.
    #[must_use]
    pub fn lossless_archive() -> Self {
        Self {
            level: Ffv1Level::Level3,
            coder: Ffv1Coder::Range,
            slice_count: 16,
            context_model: 1,
            checksum: true,
        }
    }

    /// Fastest lossless encoding preset.
    ///
    /// Uses Level 1, Golomb-Rice coder, 4 slices, simple context model, no checksums.
    /// Ideal for fast ingest or intermediate encoding.
    #[must_use]
    pub fn lossless_fast() -> Self {
        Self {
            level: Ffv1Level::Level1,
            coder: Ffv1Coder::GolombRice,
            slice_count: 4,
            context_model: 0,
            checksum: false,
        }
    }

    /// Sets the number of encoding slices.
    ///
    /// Valid values: 1, 4, 9, 16, 24. More slices enable better multithreading.
    #[must_use]
    pub fn with_slices(mut self, count: u8) -> Self {
        self.slice_count = count;
        self
    }

    /// Builds a `CodecConfig` from this FFV1 configuration.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        let mut cfg = CodecConfig::new("ffv1");
        cfg.options
            .push(("level".to_string(), self.level.as_u8().to_string()));
        cfg.options
            .push(("coder".to_string(), self.coder.as_u8().to_string()));
        cfg.options
            .push(("slices".to_string(), self.slice_count.to_string()));
        cfg.options
            .push(("context".to_string(), self.context_model.to_string()));
        cfg.options.push((
            "slicecrc".to_string(),
            if self.checksum { "1" } else { "0" }.to_string(),
        ));
        cfg.rate_control = RateControlMode::Crf(0); // lossless
        cfg
    }
}

impl Default for Ffv1Config {
    fn default() -> Self {
        Self::new()
    }
}

/// FLAC lossless audio codec configuration.
#[derive(Debug, Clone)]
pub struct FlacConfig {
    /// Compression level (0=fastest, 8=best compression).
    pub compression_level: u8,
    /// Block size in samples (256–65535, default 4096).
    pub block_size: u32,
    /// Verify decoded output matches input (slower, but ensures correctness).
    pub verify: bool,
}

impl FlacConfig {
    /// Creates a new FLAC configuration with balanced defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            compression_level: 5,
            block_size: 4096,
            verify: false,
        }
    }

    /// Archival preset — maximum compression with verification.
    #[must_use]
    pub fn archival() -> Self {
        Self {
            compression_level: 8,
            block_size: 4096,
            verify: true,
        }
    }

    /// Streaming preset — balanced speed and compression.
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            compression_level: 4,
            block_size: 4096,
            verify: false,
        }
    }

    /// Fast preset — fastest encoding, least compression.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            compression_level: 0,
            block_size: 4096,
            verify: false,
        }
    }

    /// Builds a `CodecConfig` from this FLAC configuration.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        let mut cfg = CodecConfig::new("flac");
        cfg.options.push((
            "compression_level".to_string(),
            self.compression_level.to_string(),
        ));
        cfg.options
            .push(("blocksize".to_string(), self.block_size.to_string()));
        cfg.options.push((
            "lpc_coeff_precision".to_string(),
            "15".to_string(), // maximum precision for archival quality
        ));
        if self.verify {
            cfg.options.push(("verify".to_string(), "1".to_string()));
        }
        cfg.rate_control = RateControlMode::Crf(0); // lossless
        cfg
    }
}

impl Default for FlacConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// JPEG-XL still image encoding configuration.
///
/// Supports both lossless and lossy encoding with advanced options like
/// progressive decoding, effort level, and photon noise ISO simulation.
#[derive(Debug, Clone)]
pub struct JxlConfig {
    /// Quality level for lossy encoding (1.0 = visually lossless, 100.0 = worst).
    ///
    /// Set to `None` for lossless mode.
    pub quality: Option<f32>,
    /// Encoding effort (1 = fastest, 10 = best compression).
    pub effort: JxlEffort,
    /// Enable progressive decoding (DC first, then AC passes).
    pub progressive: bool,
    /// Photon noise ISO equivalent for denoising during encode.
    ///
    /// Setting this enables noise modelling: the encoder treats pixel
    /// noise at the specified ISO level as irrelevant, yielding smaller
    /// files for high-ISO photographs.
    pub photon_noise_iso: Option<u32>,
    /// Number of extra channels (e.g. alpha, depth).
    pub extra_channels: u8,
    /// Enable modular mode (better for lossless, graphics, low-complexity images).
    pub modular: bool,
    /// Color space: "rgb", "xyb" (perceptual, default for lossy), "gray".
    pub color_space: JxlColorSpace,
    /// Bit depth per channel (8, 10, 12, 16, 32).
    pub bit_depth: u8,
}

/// JPEG-XL encoding effort (speed/compression tradeoff).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JxlEffort {
    /// Lightning fast (effort 1).
    Lightning,
    /// Thunder (effort 2).
    Thunder,
    /// Falcon (effort 3).
    Falcon,
    /// Cheetah (effort 4).
    Cheetah,
    /// Hare (effort 5).
    Hare,
    /// Wombat (effort 6).
    Wombat,
    /// Squirrel (effort 7, default).
    Squirrel,
    /// Kitten (effort 8).
    Kitten,
    /// Tortoise (effort 9).
    Tortoise,
    /// Glacier (effort 10, maximum compression).
    Glacier,
}

impl JxlEffort {
    /// Returns the numeric effort level (1-10).
    #[must_use]
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Lightning => 1,
            Self::Thunder => 2,
            Self::Falcon => 3,
            Self::Cheetah => 4,
            Self::Hare => 5,
            Self::Wombat => 6,
            Self::Squirrel => 7,
            Self::Kitten => 8,
            Self::Tortoise => 9,
            Self::Glacier => 10,
        }
    }

    /// Returns the effort name as a string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lightning => "lightning",
            Self::Thunder => "thunder",
            Self::Falcon => "falcon",
            Self::Cheetah => "cheetah",
            Self::Hare => "hare",
            Self::Wombat => "wombat",
            Self::Squirrel => "squirrel",
            Self::Kitten => "kitten",
            Self::Tortoise => "tortoise",
            Self::Glacier => "glacier",
        }
    }
}

/// JPEG-XL color space modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JxlColorSpace {
    /// RGB color space.
    Rgb,
    /// XYB perceptual color space (default for lossy).
    Xyb,
    /// Grayscale.
    Gray,
}

impl JxlColorSpace {
    /// Returns the color space as a string identifier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Xyb => "xyb",
            Self::Gray => "gray",
        }
    }
}

impl JxlConfig {
    /// Creates a new JPEG-XL configuration with balanced defaults (lossy, quality 75).
    #[must_use]
    pub fn new() -> Self {
        Self {
            quality: Some(75.0),
            effort: JxlEffort::Squirrel,
            progressive: false,
            photon_noise_iso: None,
            extra_channels: 0,
            modular: false,
            color_space: JxlColorSpace::Xyb,
            bit_depth: 8,
        }
    }

    /// Lossless encoding preset.
    ///
    /// Uses modular mode with RGB color space for mathematically lossless
    /// compression. Best for graphics, screenshots, and archival.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            quality: None,
            effort: JxlEffort::Tortoise,
            progressive: false,
            photon_noise_iso: None,
            extra_channels: 0,
            modular: true,
            color_space: JxlColorSpace::Rgb,
            bit_depth: 8,
        }
    }

    /// Web delivery preset.
    ///
    /// Lossy with progressive decoding for fast web rendering.
    /// Quality 80 gives excellent visual quality at small file sizes.
    #[must_use]
    pub fn web() -> Self {
        Self {
            quality: Some(80.0),
            effort: JxlEffort::Squirrel,
            progressive: true,
            photon_noise_iso: None,
            extra_channels: 0,
            modular: false,
            color_space: JxlColorSpace::Xyb,
            bit_depth: 8,
        }
    }

    /// Photography preset.
    ///
    /// Visually lossless encoding with photon noise modelling at ISO 400.
    /// Ideal for camera RAW conversions and photo archives.
    #[must_use]
    pub fn photography() -> Self {
        Self {
            quality: Some(90.0),
            effort: JxlEffort::Kitten,
            progressive: true,
            photon_noise_iso: Some(400),
            extra_channels: 0,
            modular: false,
            color_space: JxlColorSpace::Xyb,
            bit_depth: 16,
        }
    }

    /// Sets the quality level (1.0 = visually lossless, 100.0 = worst).
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Sets the encoding effort.
    #[must_use]
    pub fn with_effort(mut self, effort: JxlEffort) -> Self {
        self.effort = effort;
        self
    }

    /// Enables or disables progressive decoding.
    #[must_use]
    pub fn with_progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }

    /// Sets the photon noise ISO for noise modelling.
    #[must_use]
    pub fn with_photon_noise(mut self, iso: u32) -> Self {
        self.photon_noise_iso = Some(iso);
        self
    }

    /// Sets the bit depth per channel.
    #[must_use]
    pub fn with_bit_depth(mut self, depth: u8) -> Self {
        self.bit_depth = depth;
        self
    }

    /// Enables modular mode (better for lossless/graphics).
    #[must_use]
    pub fn with_modular(mut self, modular: bool) -> Self {
        self.modular = modular;
        self
    }

    /// Returns `true` if this is a lossless configuration.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.quality.is_none()
    }

    /// Builds a `CodecConfig` from this JPEG-XL configuration.
    #[must_use]
    pub fn build(self) -> CodecConfig {
        let mut cfg = CodecConfig::new("jxl");

        if let Some(q) = self.quality {
            cfg.options.push(("quality".to_string(), format!("{q:.1}")));
        } else {
            cfg.options.push(("lossless".to_string(), "1".to_string()));
        }

        cfg.options
            .push(("effort".to_string(), self.effort.as_u8().to_string()));

        if self.progressive {
            cfg.options
                .push(("progressive".to_string(), "1".to_string()));
        }

        if let Some(iso) = self.photon_noise_iso {
            cfg.options
                .push(("photon_noise_iso".to_string(), iso.to_string()));
        }

        if self.modular {
            cfg.options.push(("modular".to_string(), "1".to_string()));
        }

        cfg.options.push((
            "color_space".to_string(),
            self.color_space.as_str().to_string(),
        ));

        cfg.options
            .push(("bit_depth".to_string(), self.bit_depth.to_string()));

        // Lossless uses CRF 0; lossy uses quality-based CRF approximation
        cfg.rate_control = if self.is_lossless() {
            RateControlMode::Crf(0)
        } else {
            // Map quality 1-100 to CRF-like value
            let q = self.quality.unwrap_or(75.0);
            let crf = ((100.0 - q) * 0.63) as u8;
            RateControlMode::Crf(crf)
        };

        cfg
    }
}

impl Default for JxlConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates codec configuration from quality mode.
#[must_use]
pub fn codec_config_from_quality(codec: &str, quality: QualityMode) -> CodecConfig {
    let preset = quality.to_preset();
    let crf = quality.to_crf();

    match codec {
        "h264" => H264Config::new()
            .profile(H264Profile::High)
            .refs(3)
            .bframes(3)
            .build()
            .preset(preset)
            .rate_control(RateControlMode::Crf(crf)),
        "vp9" => Vp9Config::new()
            .cpu_used(preset.cpu_used())
            .row_mt(true)
            .build()
            .preset(preset)
            .rate_control(RateControlMode::Crf(crf)),
        "av1" => Av1Config::new()
            .cpu_used(preset.cpu_used())
            .row_mt(true)
            .usage(Av1Usage::Good)
            .build()
            .preset(preset)
            .rate_control(RateControlMode::Crf(crf)),
        "opus" => OpusConfig::new()
            .application(OpusApplication::Audio)
            .complexity(10)
            .vbr(true)
            .build()
            .preset(preset),
        _ => CodecConfig::new(codec)
            .preset(preset)
            .rate_control(RateControlMode::Crf(crf)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_config_new() {
        let config = CodecConfig::new("h264");
        assert_eq!(config.codec, "h264");
        assert_eq!(config.preset, QualityPreset::Medium);
    }

    #[test]
    fn test_h264_config() {
        let config = H264Config::new()
            .profile(H264Profile::High)
            .level("4.0")
            .refs(3)
            .bframes(3)
            .cabac(true)
            .dct8x8(true)
            .build();

        assert_eq!(config.codec, "h264");
        assert_eq!(config.profile, Some("high".to_string()));
        assert_eq!(config.level, Some("4.0".to_string()));
        assert!(config.options.len() > 0);
    }

    #[test]
    fn test_vp9_config() {
        let config = Vp9Config::new()
            .cpu_used(4)
            .tile_columns(2)
            .tile_rows(1)
            .row_mt(true)
            .build();

        assert_eq!(config.codec, "vp9");
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "cpu-used" && v == "4"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "row-mt" && v == "1"));
    }

    #[test]
    fn test_av1_config() {
        let config = Av1Config::new()
            .cpu_used(6)
            .tiles(4, 2)
            .usage(Av1Usage::Good)
            .row_mt(true)
            .build();

        assert_eq!(config.codec, "av1");
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "cpu-used" && v == "6"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "tiles" && v == "4x2"));
    }

    #[test]
    fn test_opus_config() {
        let config = OpusConfig::new()
            .application(OpusApplication::Audio)
            .complexity(10)
            .vbr(true)
            .build();

        assert_eq!(config.codec, "opus");
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "application" && v == "audio"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "complexity" && v == "10"));
    }

    #[test]
    fn test_codec_config_from_quality() {
        let config = codec_config_from_quality("h264", QualityMode::High);
        assert_eq!(config.codec, "h264");
        assert_eq!(config.preset, QualityPreset::Slow);
        assert_eq!(config.rate_control, RateControlMode::Crf(20));
    }

    #[test]
    fn test_h264_profiles() {
        assert_eq!(H264Profile::Baseline.as_str(), "baseline");
        assert_eq!(H264Profile::Main.as_str(), "main");
        assert_eq!(H264Profile::High.as_str(), "high");
    }

    #[test]
    fn test_av1_usage() {
        assert_eq!(Av1Usage::Good.as_str(), "good");
        assert_eq!(Av1Usage::Realtime.as_str(), "realtime");
    }

    #[test]
    fn test_opus_application() {
        assert_eq!(OpusApplication::Voip.as_str(), "voip");
        assert_eq!(OpusApplication::Audio.as_str(), "audio");
        assert_eq!(OpusApplication::LowDelay.as_str(), "lowdelay");
    }

    // ── VP9 CRF and new builder methods ──────────────────────────────────

    #[test]
    fn test_vp9_crf_mode() {
        let config = Vp9Config::crf(33).build();
        assert_eq!(config.codec, "vp9");
        assert!(
            matches!(config.rate_control, RateControlMode::Crf(33)),
            "VP9 CRF mode should use Crf(33)"
        );
    }

    #[test]
    fn test_vp9_crf_range_boundary() {
        // VP9 CRF is valid 0-63; test extremes
        let lo = Vp9Config::crf(0).build();
        let hi = Vp9Config::crf(63).build();
        assert!(matches!(lo.rate_control, RateControlMode::Crf(0)));
        assert!(matches!(hi.rate_control, RateControlMode::Crf(63)));
    }

    #[test]
    fn test_vp9_with_tile_columns() {
        let config = Vp9Config::new().with_tile_columns(3).build();
        assert!(
            config
                .options
                .iter()
                .any(|(k, v)| k == "tile-columns" && v == "3"),
            "with_tile_columns should set tile-columns option"
        );
    }

    #[test]
    fn test_vp9_with_frame_parallel() {
        let config_on = Vp9Config::new().with_frame_parallel(true).build();
        let config_off = Vp9Config::new().with_frame_parallel(false).build();
        assert!(config_on
            .options
            .iter()
            .any(|(k, v)| k == "frame-parallel" && v == "1"));
        assert!(config_off
            .options
            .iter()
            .any(|(k, v)| k == "frame-parallel" && v == "0"));
    }

    #[test]
    fn test_vp9_with_lag_in_frames() {
        let config = Vp9Config::new().with_lag_in_frames(25).build();
        assert!(
            config
                .options
                .iter()
                .any(|(k, v)| k == "lag-in-frames" && v == "25"),
            "with_lag_in_frames should set lag-in-frames option"
        );
    }

    #[test]
    fn test_vp9_with_row_mt() {
        let config_on = Vp9Config::new().with_row_mt(true).build();
        let config_off = Vp9Config::new().with_row_mt(false).build();
        assert!(config_on
            .options
            .iter()
            .any(|(k, v)| k == "row-mt" && v == "1"));
        assert!(config_off
            .options
            .iter()
            .any(|(k, v)| k == "row-mt" && v == "0"));
    }

    #[test]
    fn test_vp9_screen_content() {
        let config = Vp9Config::screen_content().build();
        assert_eq!(config.codec, "vp9");
        // Screen content uses CRF mode
        assert!(matches!(config.rate_control, RateControlMode::Crf(_)));
        // Should have cpu-used set for speed
        assert!(config.options.iter().any(|(k, _)| k == "cpu-used"));
        // Row-MT should be enabled for throughput
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "row-mt" && v == "1"));
    }

    // ── Ffv1Config tests ─────────────────────────────────────────────────

    #[test]
    fn test_ffv1_config_new_defaults() {
        let cfg = Ffv1Config::new();
        assert!(matches!(cfg.level, Ffv1Level::Level3));
        assert!(matches!(cfg.coder, Ffv1Coder::Range));
        assert_eq!(cfg.slice_count, 4);
        assert_eq!(cfg.context_model, 0);
        assert!(cfg.checksum);
    }

    #[test]
    fn test_ffv1_lossless_archive() {
        let cfg = Ffv1Config::lossless_archive();
        assert!(matches!(cfg.level, Ffv1Level::Level3));
        assert!(matches!(cfg.coder, Ffv1Coder::Range));
        assert_eq!(cfg.slice_count, 16);
        assert_eq!(cfg.context_model, 1);
        assert!(cfg.checksum);
    }

    #[test]
    fn test_ffv1_lossless_fast() {
        let cfg = Ffv1Config::lossless_fast();
        assert!(matches!(cfg.level, Ffv1Level::Level1));
        assert!(matches!(cfg.coder, Ffv1Coder::GolombRice));
        assert!(!cfg.checksum);
    }

    #[test]
    fn test_ffv1_with_slices() {
        let cfg = Ffv1Config::new().with_slices(9);
        assert_eq!(cfg.slice_count, 9);
    }

    #[test]
    fn test_ffv1_build() {
        let config = Ffv1Config::new().build();
        assert_eq!(config.codec, "ffv1");
        assert!(config.options.iter().any(|(k, v)| k == "level" && v == "3"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "slices" && v == "4"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "slicecrc" && v == "1"));
    }

    #[test]
    fn test_ffv1_level_values() {
        assert_eq!(Ffv1Level::Level1.as_u8(), 1);
        assert_eq!(Ffv1Level::Level3.as_u8(), 3);
    }

    #[test]
    fn test_ffv1_coder_values() {
        assert_eq!(Ffv1Coder::GolombRice.as_u8(), 0);
        assert_eq!(Ffv1Coder::Range.as_u8(), 1);
    }

    // ── OpusConfig advanced methods ───────────────────────────────────────

    #[test]
    fn test_opus_voice_preset() {
        let config = OpusConfig::voice().build();
        assert_eq!(config.codec, "opus");
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "application" && v == "voip"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "inband_fec" && v == "1"));
    }

    #[test]
    fn test_opus_music_preset() {
        let config = OpusConfig::music().build();
        assert_eq!(config.codec, "opus");
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "application" && v == "audio"));
        assert!(config.options.iter().any(|(k, v)| k == "vbr" && v == "on"));
    }

    #[test]
    fn test_opus_fullband_preset() {
        let config = OpusConfig::fullband().build();
        assert_eq!(config.codec, "opus");
        assert!(config.options.iter().any(|(k, _)| k == "cutoff"));
    }

    #[test]
    fn test_opus_with_vbr() {
        let on = OpusConfig::new().with_vbr(true).build();
        let off = OpusConfig::new().with_vbr(false).build();
        assert!(on.options.iter().any(|(k, v)| k == "vbr" && v == "on"));
        assert!(off.options.iter().any(|(k, v)| k == "vbr" && v == "off"));
    }

    #[test]
    fn test_opus_with_constrained_vbr() {
        let on = OpusConfig::new().with_constrained_vbr(true).build();
        let off = OpusConfig::new().with_constrained_vbr(false).build();
        assert!(on.options.iter().any(|(k, v)| k == "cvbr" && v == "1"));
        assert!(off.options.iter().any(|(k, v)| k == "cvbr" && v == "0"));
    }

    #[test]
    fn test_opus_with_dtx() {
        let on = OpusConfig::new().with_dtx(true).build();
        let off = OpusConfig::new().with_dtx(false).build();
        assert!(on.options.iter().any(|(k, v)| k == "dtx" && v == "1"));
        assert!(off.options.iter().any(|(k, v)| k == "dtx" && v == "0"));
    }

    #[test]
    fn test_opus_with_fec() {
        let config = OpusConfig::new().with_fec(true).build();
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "inband_fec" && v == "1"));
        let config_off = OpusConfig::new().with_fec(false).build();
        assert!(config_off
            .options
            .iter()
            .any(|(k, v)| k == "inband_fec" && v == "0"));
    }

    #[test]
    fn test_opus_with_packet_loss_perc() {
        let config = OpusConfig::new().with_packet_loss_perc(10).build();
        assert!(
            config
                .options
                .iter()
                .any(|(k, v)| k == "packet_loss_perc" && v == "10"),
            "packet_loss_perc option should be set"
        );
    }

    // ── FlacConfig tests ─────────────────────────────────────────────────

    #[test]
    fn test_flac_new_defaults() {
        let cfg = FlacConfig::new();
        assert_eq!(cfg.compression_level, 5);
        assert_eq!(cfg.block_size, 4096);
        assert!(!cfg.verify);
    }

    #[test]
    fn test_flac_archival() {
        let cfg = FlacConfig::archival();
        assert_eq!(cfg.compression_level, 8);
        assert!(cfg.verify);
    }

    #[test]
    fn test_flac_streaming() {
        let cfg = FlacConfig::streaming();
        assert_eq!(cfg.compression_level, 4);
        assert!(!cfg.verify);
    }

    #[test]
    fn test_flac_fast() {
        let cfg = FlacConfig::fast();
        assert_eq!(cfg.compression_level, 0);
    }

    #[test]
    fn test_flac_build() {
        let config = FlacConfig::new().build();
        assert_eq!(config.codec, "flac");
        assert!(
            config.options.iter().any(|(k, _)| k == "compression_level"),
            "FLAC config should include compression_level"
        );
    }

    #[test]
    fn test_flac_archival_build_sets_verify() {
        let config = FlacConfig::archival().build();
        assert_eq!(config.codec, "flac");
        assert!(
            config
                .options
                .iter()
                .any(|(k, v)| k == "verify" && v == "1"),
            "Archival FLAC should enable verify"
        );
    }

    // ── JxlConfig tests ─────────────────────────────────────────────────

    #[test]
    fn test_jxl_new_defaults() {
        let cfg = JxlConfig::new();
        assert_eq!(cfg.quality, Some(75.0));
        assert_eq!(cfg.effort, JxlEffort::Squirrel);
        assert!(!cfg.progressive);
        assert!(cfg.photon_noise_iso.is_none());
        assert!(!cfg.modular);
        assert_eq!(cfg.color_space, JxlColorSpace::Xyb);
        assert_eq!(cfg.bit_depth, 8);
        assert!(!cfg.is_lossless());
    }

    #[test]
    fn test_jxl_lossless() {
        let cfg = JxlConfig::lossless();
        assert!(cfg.is_lossless());
        assert!(cfg.quality.is_none());
        assert!(cfg.modular);
        assert_eq!(cfg.color_space, JxlColorSpace::Rgb);
        assert_eq!(cfg.effort, JxlEffort::Tortoise);
    }

    #[test]
    fn test_jxl_web() {
        let cfg = JxlConfig::web();
        assert_eq!(cfg.quality, Some(80.0));
        assert!(cfg.progressive);
        assert!(!cfg.is_lossless());
    }

    #[test]
    fn test_jxl_photography() {
        let cfg = JxlConfig::photography();
        assert_eq!(cfg.quality, Some(90.0));
        assert_eq!(cfg.photon_noise_iso, Some(400));
        assert_eq!(cfg.bit_depth, 16);
        assert!(cfg.progressive);
    }

    #[test]
    fn test_jxl_with_quality() {
        let cfg = JxlConfig::new().with_quality(50.0);
        assert_eq!(cfg.quality, Some(50.0));
    }

    #[test]
    fn test_jxl_with_effort() {
        let cfg = JxlConfig::new().with_effort(JxlEffort::Glacier);
        assert_eq!(cfg.effort, JxlEffort::Glacier);
    }

    #[test]
    fn test_jxl_with_progressive() {
        let cfg = JxlConfig::new().with_progressive(true);
        assert!(cfg.progressive);
    }

    #[test]
    fn test_jxl_with_photon_noise() {
        let cfg = JxlConfig::new().with_photon_noise(800);
        assert_eq!(cfg.photon_noise_iso, Some(800));
    }

    #[test]
    fn test_jxl_with_bit_depth() {
        let cfg = JxlConfig::new().with_bit_depth(16);
        assert_eq!(cfg.bit_depth, 16);
    }

    #[test]
    fn test_jxl_with_modular() {
        let cfg = JxlConfig::new().with_modular(true);
        assert!(cfg.modular);
    }

    #[test]
    fn test_jxl_build_lossy() {
        let config = JxlConfig::new().build();
        assert_eq!(config.codec, "jxl");
        assert!(config.options.iter().any(|(k, _)| k == "quality"));
        assert!(config.options.iter().any(|(k, _)| k == "effort"));
        assert!(config.options.iter().any(|(k, _)| k == "color_space"));
        assert!(config.options.iter().any(|(k, _)| k == "bit_depth"));
        // Lossy should not set lossless flag
        assert!(!config
            .options
            .iter()
            .any(|(k, v)| k == "lossless" && v == "1"));
    }

    #[test]
    fn test_jxl_build_lossless() {
        let config = JxlConfig::lossless().build();
        assert_eq!(config.codec, "jxl");
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "lossless" && v == "1"));
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "modular" && v == "1"));
        assert_eq!(config.rate_control, RateControlMode::Crf(0));
    }

    #[test]
    fn test_jxl_build_progressive() {
        let config = JxlConfig::web().build();
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "progressive" && v == "1"));
    }

    #[test]
    fn test_jxl_build_photon_noise() {
        let config = JxlConfig::photography().build();
        assert!(config
            .options
            .iter()
            .any(|(k, v)| k == "photon_noise_iso" && v == "400"));
    }

    #[test]
    fn test_jxl_effort_values() {
        assert_eq!(JxlEffort::Lightning.as_u8(), 1);
        assert_eq!(JxlEffort::Thunder.as_u8(), 2);
        assert_eq!(JxlEffort::Falcon.as_u8(), 3);
        assert_eq!(JxlEffort::Cheetah.as_u8(), 4);
        assert_eq!(JxlEffort::Hare.as_u8(), 5);
        assert_eq!(JxlEffort::Wombat.as_u8(), 6);
        assert_eq!(JxlEffort::Squirrel.as_u8(), 7);
        assert_eq!(JxlEffort::Kitten.as_u8(), 8);
        assert_eq!(JxlEffort::Tortoise.as_u8(), 9);
        assert_eq!(JxlEffort::Glacier.as_u8(), 10);
    }

    #[test]
    fn test_jxl_effort_names() {
        assert_eq!(JxlEffort::Lightning.as_str(), "lightning");
        assert_eq!(JxlEffort::Squirrel.as_str(), "squirrel");
        assert_eq!(JxlEffort::Glacier.as_str(), "glacier");
    }

    #[test]
    fn test_jxl_color_space_names() {
        assert_eq!(JxlColorSpace::Rgb.as_str(), "rgb");
        assert_eq!(JxlColorSpace::Xyb.as_str(), "xyb");
        assert_eq!(JxlColorSpace::Gray.as_str(), "gray");
    }

    #[test]
    fn test_jxl_default_is_new() {
        let cfg = JxlConfig::default();
        assert_eq!(cfg.quality, Some(75.0));
        assert_eq!(cfg.effort, JxlEffort::Squirrel);
    }
}
