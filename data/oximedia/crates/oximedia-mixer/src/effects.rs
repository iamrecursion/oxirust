//! Audio effect processing for mixer channels.
//!
//! Provides comprehensive effect categories with professional-grade processing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique effect identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EffectId(pub Uuid);

impl std::fmt::Display for EffectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Effect category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectCategory {
    /// Dynamics processing (compressor, limiter, gate, expander).
    Dynamics,
    /// Equalization (parametric, graphic, shelving).
    Eq,
    /// Time-based effects (reverb, delay, echo).
    TimeBased,
    /// Modulation effects (chorus, flanger, phaser).
    Modulation,
    /// Distortion effects (saturation, overdrive, clipper).
    Distortion,
    /// Filter effects (low-pass, high-pass, band-pass).
    Filter,
    /// Utility effects (gain, phase, delay compensation).
    Utility,
}

/// Effect type with parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effect {
    /// Compressor.
    Compressor(CompressorParams),
    /// Limiter.
    Limiter(LimiterParams),
    /// Gate/Expander.
    Gate(GateParams),
    /// Expander.
    Expander(ExpanderParams),
    /// De-esser.
    DeEsser(DeEsserParams),
    /// Parametric EQ.
    ParametricEq(ParametricEqParams),
    /// Graphic EQ.
    GraphicEq(GraphicEqParams),
    /// High-pass filter.
    HighPass(FilterParams),
    /// Low-pass filter.
    LowPass(FilterParams),
    /// Band-pass filter.
    BandPass(FilterParams),
    /// Notch filter.
    Notch(FilterParams),
    /// Low shelf.
    LowShelf(ShelfParams),
    /// High shelf.
    HighShelf(ShelfParams),
    /// Reverb.
    Reverb(ReverbParams),
    /// Delay.
    Delay(DelayParams),
    /// Echo.
    Echo(EchoParams),
    /// Chorus.
    Chorus(ChorusParams),
    /// Flanger.
    Flanger(FlangerParams),
    /// Phaser.
    Phaser(PhaserParams),
    /// Vibrato.
    Vibrato(VibratoParams),
    /// Tremolo.
    Tremolo(TremoloParams),
    /// Ring modulator.
    RingModulator(RingModulatorParams),
    /// Saturation.
    Saturation(SaturationParams),
    /// Overdrive.
    Overdrive(OverdriveParams),
    /// Distortion.
    Distortion(DistortionParams),
    /// Bit crusher.
    BitCrusher(BitCrusherParams),
    /// Wave shaper.
    WaveShaper(WaveShaperParams),
}

impl Effect {
    /// Get effect category.
    #[must_use]
    pub fn category(&self) -> EffectCategory {
        match self {
            Self::Compressor(_)
            | Self::Limiter(_)
            | Self::Gate(_)
            | Self::Expander(_)
            | Self::DeEsser(_) => EffectCategory::Dynamics,
            Self::ParametricEq(_) | Self::GraphicEq(_) | Self::LowShelf(_) | Self::HighShelf(_) => {
                EffectCategory::Eq
            }
            Self::HighPass(_) | Self::LowPass(_) | Self::BandPass(_) | Self::Notch(_) => {
                EffectCategory::Filter
            }
            Self::Reverb(_) | Self::Delay(_) | Self::Echo(_) => EffectCategory::TimeBased,
            Self::Chorus(_)
            | Self::Flanger(_)
            | Self::Phaser(_)
            | Self::Vibrato(_)
            | Self::Tremolo(_)
            | Self::RingModulator(_) => EffectCategory::Modulation,
            Self::Saturation(_)
            | Self::Overdrive(_)
            | Self::Distortion(_)
            | Self::BitCrusher(_)
            | Self::WaveShaper(_) => EffectCategory::Distortion,
        }
    }

    /// Get effect name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Compressor(_) => "Compressor",
            Self::Limiter(_) => "Limiter",
            Self::Gate(_) => "Gate",
            Self::Expander(_) => "Expander",
            Self::DeEsser(_) => "De-esser",
            Self::ParametricEq(_) => "Parametric EQ",
            Self::GraphicEq(_) => "Graphic EQ",
            Self::HighPass(_) => "High-pass Filter",
            Self::LowPass(_) => "Low-pass Filter",
            Self::BandPass(_) => "Band-pass Filter",
            Self::Notch(_) => "Notch Filter",
            Self::LowShelf(_) => "Low Shelf",
            Self::HighShelf(_) => "High Shelf",
            Self::Reverb(_) => "Reverb",
            Self::Delay(_) => "Delay",
            Self::Echo(_) => "Echo",
            Self::Chorus(_) => "Chorus",
            Self::Flanger(_) => "Flanger",
            Self::Phaser(_) => "Phaser",
            Self::Vibrato(_) => "Vibrato",
            Self::Tremolo(_) => "Tremolo",
            Self::RingModulator(_) => "Ring Modulator",
            Self::Saturation(_) => "Saturation",
            Self::Overdrive(_) => "Overdrive",
            Self::Distortion(_) => "Distortion",
            Self::BitCrusher(_) => "Bit Crusher",
            Self::WaveShaper(_) => "Wave Shaper",
        }
    }
}

/// Effect slot in channel insert chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSlot {
    /// Effect ID.
    pub id: EffectId,
    /// Effect instance.
    pub effect: Effect,
    /// Effect is bypassed.
    pub bypassed: bool,
    /// Wet/dry mix (0.0 = dry, 1.0 = wet).
    pub mix: f32,
    /// Preset name.
    pub preset: Option<String>,
}

impl EffectSlot {
    /// Create a new effect slot.
    #[must_use]
    pub fn new(effect: Effect) -> Self {
        Self {
            id: EffectId(Uuid::new_v4()),
            effect,
            bypassed: false,
            mix: 1.0,
            preset: None,
        }
    }
}

// ============================================================================
// Dynamics Effects
// ============================================================================

/// Compressor parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorParams {
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Ratio (1.0 = no compression, 20.0 = limiting).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f32,
    /// Makeup gain in dB.
    pub makeup_gain_db: f32,
    /// Auto makeup gain.
    pub auto_makeup: bool,
    /// Side-chain input.
    pub sidechain: Option<SideChainConfig>,
    /// RMS or peak detection.
    pub detection_mode: DetectionMode,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 3.0,
            makeup_gain_db: 0.0,
            auto_makeup: true,
            sidechain: None,
            detection_mode: DetectionMode::Rms,
        }
    }
}

/// Limiter parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimiterParams {
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Ceiling in dB.
    pub ceiling_db: f32,
    /// Look-ahead time in milliseconds.
    pub lookahead_ms: f32,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            threshold_db: -1.0,
            release_ms: 50.0,
            ceiling_db: -0.1,
            lookahead_ms: 5.0,
        }
    }
}

/// Gate parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateParams {
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Ratio (1.0 = no gating, inf = full gate).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Hold time in milliseconds.
    pub hold_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Range in dB (max attenuation).
    pub range_db: f32,
    /// Side-chain input.
    pub sidechain: Option<SideChainConfig>,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            ratio: 10.0,
            attack_ms: 1.0,
            hold_ms: 10.0,
            release_ms: 100.0,
            range_db: -80.0,
            sidechain: None,
        }
    }
}

/// Expander parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpanderParams {
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Ratio (1.0 = no expansion, 0.5 = 1:2 expansion).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Knee width in dB.
    pub knee_db: f32,
}

impl Default for ExpanderParams {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            ratio: 2.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 3.0,
        }
    }
}

/// De-esser parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeEsserParams {
    /// Frequency in Hz.
    pub frequency_hz: f32,
    /// Bandwidth in octaves.
    pub bandwidth_octaves: f32,
    /// Threshold in dB.
    pub threshold_db: f32,
    /// Ratio.
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
}

impl Default for DeEsserParams {
    fn default() -> Self {
        Self {
            frequency_hz: 6000.0,
            bandwidth_octaves: 1.0,
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 10.0,
        }
    }
}

/// Detection mode for dynamics processors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMode {
    /// RMS (average) level detection.
    Rms,
    /// Peak level detection.
    Peak,
}

/// Side-chain configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideChainConfig {
    /// Side-chain input channel ID.
    pub source: crate::ChannelId,
    /// Side-chain filter enabled.
    pub filter_enabled: bool,
    /// Side-chain filter frequency in Hz.
    pub filter_freq: f32,
    /// Side-chain filter Q.
    pub filter_q: f32,
}

// ============================================================================
// EQ Effects
// ============================================================================

/// Parametric EQ parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricEqParams {
    /// EQ bands.
    pub bands: Vec<EqBand>,
}

impl Default for ParametricEqParams {
    fn default() -> Self {
        Self {
            bands: vec![
                EqBand::new(EqBandType::HighPass, 80.0, 0.0, 0.707),
                EqBand::new(EqBandType::LowShelf, 100.0, 0.0, 0.707),
                EqBand::new(EqBandType::Bell, 1000.0, 0.0, 1.0),
                EqBand::new(EqBandType::Bell, 3000.0, 0.0, 1.0),
                EqBand::new(EqBandType::HighShelf, 10000.0, 0.0, 0.707),
                EqBand::new(EqBandType::LowPass, 16000.0, 0.0, 0.707),
            ],
        }
    }
}

/// EQ band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqBand {
    /// Band type.
    pub band_type: EqBandType,
    /// Frequency in Hz.
    pub frequency_hz: f32,
    /// Gain in dB (for bell, shelf).
    pub gain_db: f32,
    /// Q (bandwidth).
    pub q: f32,
    /// Band enabled.
    pub enabled: bool,
}

impl EqBand {
    /// Create a new EQ band.
    #[must_use]
    pub fn new(band_type: EqBandType, frequency_hz: f32, gain_db: f32, q: f32) -> Self {
        Self {
            band_type,
            frequency_hz,
            gain_db,
            q,
            enabled: true,
        }
    }
}

/// EQ band type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EqBandType {
    /// Bell/peaking filter.
    Bell,
    /// Low shelf.
    LowShelf,
    /// High shelf.
    HighShelf,
    /// High-pass filter.
    HighPass,
    /// Low-pass filter.
    LowPass,
    /// Notch filter.
    Notch,
}

/// Graphic EQ parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicEqParams {
    /// Band gains in dB.
    pub bands: HashMap<u32, f32>,
}

impl Default for GraphicEqParams {
    fn default() -> Self {
        // 31-band graphic EQ (ISO standard frequencies)
        let frequencies = [
            20, 25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
        ];

        let mut bands = HashMap::new();
        for freq in &frequencies {
            bands.insert(*freq, 0.0);
        }

        Self { bands }
    }
}

/// Filter parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    /// Cutoff frequency in Hz.
    pub frequency_hz: f32,
    /// Q (resonance).
    pub q: f32,
    /// Filter slope in dB/octave.
    pub slope_db_per_oct: u32,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            frequency_hz: 1000.0,
            q: 0.707,
            slope_db_per_oct: 12,
        }
    }
}

/// Shelf filter parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShelfParams {
    /// Frequency in Hz.
    pub frequency_hz: f32,
    /// Gain in dB.
    pub gain_db: f32,
    /// Q (slope).
    pub q: f32,
}

impl Default for ShelfParams {
    fn default() -> Self {
        Self {
            frequency_hz: 1000.0,
            gain_db: 0.0,
            q: 0.707,
        }
    }
}

// ============================================================================
// Time-based Effects
// ============================================================================

/// Reverb parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbParams {
    /// Room size (0.0 = small, 1.0 = large).
    pub room_size: f32,
    /// Damping (0.0 = bright, 1.0 = dark).
    pub damping: f32,
    /// Pre-delay in milliseconds.
    pub predelay_ms: f32,
    /// Width (stereo spread).
    pub width: f32,
    /// Wet level (0.0 = dry, 1.0 = wet).
    pub wet: f32,
    /// Dry level.
    pub dry: f32,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            predelay_ms: 0.0,
            width: 1.0,
            wet: 0.3,
            dry: 0.7,
        }
    }
}

/// Delay parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayParams {
    /// Delay time in milliseconds.
    pub time_ms: f32,
    /// Feedback (0.0 = no repeats, 1.0 = infinite).
    pub feedback: f32,
    /// High-pass filter frequency in Hz.
    pub highpass_hz: f32,
    /// Low-pass filter frequency in Hz.
    pub lowpass_hz: f32,
    /// Sync to tempo.
    pub tempo_sync: bool,
    /// Note division (if tempo sync).
    pub division: NoteDivision,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            time_ms: 250.0,
            feedback: 0.3,
            highpass_hz: 100.0,
            lowpass_hz: 8000.0,
            tempo_sync: false,
            division: NoteDivision::Quarter,
        }
    }
}

/// Echo parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoParams {
    /// Left delay time in milliseconds.
    pub left_time_ms: f32,
    /// Right delay time in milliseconds.
    pub right_time_ms: f32,
    /// Feedback.
    pub feedback: f32,
    /// Cross-feedback.
    pub cross_feedback: f32,
}

impl Default for EchoParams {
    fn default() -> Self {
        Self {
            left_time_ms: 250.0,
            right_time_ms: 375.0,
            feedback: 0.3,
            cross_feedback: 0.1,
        }
    }
}

/// Note division for tempo-synced effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoteDivision {
    /// Whole note.
    Whole,
    /// Half note.
    Half,
    /// Quarter note.
    Quarter,
    /// Eighth note.
    Eighth,
    /// Sixteenth note.
    Sixteenth,
    /// Dotted quarter.
    DottedQuarter,
    /// Dotted eighth.
    DottedEighth,
    /// Triplet quarter.
    TripletQuarter,
    /// Triplet eighth.
    TripletEighth,
}

// ============================================================================
// Modulation Effects
// ============================================================================

/// Chorus parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChorusParams {
    /// Rate in Hz.
    pub rate_hz: f32,
    /// Depth (0.0 to 1.0).
    pub depth: f32,
    /// Number of voices.
    pub voices: u32,
    /// Feedback.
    pub feedback: f32,
    /// Delay time in milliseconds.
    pub delay_ms: f32,
}

impl Default for ChorusParams {
    fn default() -> Self {
        Self {
            rate_hz: 0.5,
            depth: 0.3,
            voices: 3,
            feedback: 0.2,
            delay_ms: 20.0,
        }
    }
}

/// Flanger parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlangerParams {
    /// Rate in Hz.
    pub rate_hz: f32,
    /// Depth (0.0 to 1.0).
    pub depth: f32,
    /// Feedback.
    pub feedback: f32,
    /// Delay time in milliseconds.
    pub delay_ms: f32,
}

impl Default for FlangerParams {
    fn default() -> Self {
        Self {
            rate_hz: 0.2,
            depth: 0.5,
            feedback: 0.5,
            delay_ms: 5.0,
        }
    }
}

/// Phaser parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaserParams {
    /// Rate in Hz.
    pub rate_hz: f32,
    /// Depth (0.0 to 1.0).
    pub depth: f32,
    /// Feedback.
    pub feedback: f32,
    /// Number of stages.
    pub stages: u32,
}

impl Default for PhaserParams {
    fn default() -> Self {
        Self {
            rate_hz: 0.3,
            depth: 0.5,
            feedback: 0.5,
            stages: 4,
        }
    }
}

/// Vibrato parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibratoParams {
    /// Rate in Hz.
    pub rate_hz: f32,
    /// Depth (0.0 to 1.0).
    pub depth: f32,
}

impl Default for VibratoParams {
    fn default() -> Self {
        Self {
            rate_hz: 5.0,
            depth: 0.5,
        }
    }
}

/// Tremolo parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremoloParams {
    /// Rate in Hz.
    pub rate_hz: f32,
    /// Depth (0.0 to 1.0).
    pub depth: f32,
    /// Waveform shape.
    pub waveform: Waveform,
}

impl Default for TremoloParams {
    fn default() -> Self {
        Self {
            rate_hz: 5.0,
            depth: 0.5,
            waveform: Waveform::Sine,
        }
    }
}

/// Ring modulator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingModulatorParams {
    /// Carrier frequency in Hz.
    pub frequency_hz: f32,
    /// Depth (0.0 to 1.0).
    pub depth: f32,
}

impl Default for RingModulatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: 440.0,
            depth: 1.0,
        }
    }
}

/// LFO waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Waveform {
    /// Sine wave.
    Sine,
    /// Triangle wave.
    Triangle,
    /// Square wave.
    Square,
    /// Sawtooth wave.
    Sawtooth,
}

// ============================================================================
// Distortion Effects
// ============================================================================

/// Saturation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationParams {
    /// Drive (input gain).
    pub drive: f32,
    /// Saturation amount.
    pub amount: f32,
    /// Output gain.
    pub output: f32,
}

impl Default for SaturationParams {
    fn default() -> Self {
        Self {
            drive: 1.0,
            amount: 0.5,
            output: 1.0,
        }
    }
}

/// Overdrive parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverdriveParams {
    /// Drive (input gain).
    pub drive: f32,
    /// Tone (high-frequency rolloff).
    pub tone: f32,
    /// Output level.
    pub level: f32,
}

impl Default for OverdriveParams {
    fn default() -> Self {
        Self {
            drive: 0.5,
            tone: 0.5,
            level: 1.0,
        }
    }
}

/// Distortion parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistortionParams {
    /// Drive (input gain).
    pub drive: f32,
    /// Distortion type.
    pub distortion_type: DistortionType,
    /// Output level.
    pub level: f32,
}

impl Default for DistortionParams {
    fn default() -> Self {
        Self {
            drive: 0.5,
            distortion_type: DistortionType::HardClip,
            level: 1.0,
        }
    }
}

/// Distortion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistortionType {
    /// Hard clipping.
    HardClip,
    /// Soft clipping.
    SoftClip,
    /// Tube saturation.
    Tube,
    /// Asymmetric clipping.
    Asymmetric,
}

/// Bit crusher parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitCrusherParams {
    /// Bit depth (1-16).
    pub bit_depth: u32,
    /// Sample rate reduction.
    pub sample_rate_reduction: f32,
}

impl Default for BitCrusherParams {
    fn default() -> Self {
        Self {
            bit_depth: 8,
            sample_rate_reduction: 1.0,
        }
    }
}

/// Wave shaper parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveShaperParams {
    /// Drive (input gain).
    pub drive: f32,
    /// Transfer curve points.
    pub curve: Vec<(f32, f32)>,
    /// Output gain.
    pub output: f32,
}

impl Default for WaveShaperParams {
    fn default() -> Self {
        Self {
            drive: 1.0,
            curve: vec![(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)],
            output: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_category() {
        let compressor = Effect::Compressor(CompressorParams::default());
        assert_eq!(compressor.category(), EffectCategory::Dynamics);
        assert_eq!(compressor.name(), "Compressor");

        let reverb = Effect::Reverb(ReverbParams::default());
        assert_eq!(reverb.category(), EffectCategory::TimeBased);
        assert_eq!(reverb.name(), "Reverb");
    }

    #[test]
    fn test_effect_slot() {
        let effect = Effect::Compressor(CompressorParams::default());
        let slot = EffectSlot::new(effect);

        assert!(!slot.bypassed);
        assert!((slot.mix - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compressor_defaults() {
        let params = CompressorParams::default();
        assert_eq!(params.threshold_db, -20.0);
        assert_eq!(params.ratio, 4.0);
        assert!(params.auto_makeup);
    }

    #[test]
    fn test_eq_band() {
        let band = EqBand::new(EqBandType::Bell, 1000.0, 3.0, 1.0);
        assert_eq!(band.band_type, EqBandType::Bell);
        assert_eq!(band.frequency_hz, 1000.0);
        assert_eq!(band.gain_db, 3.0);
        assert!(band.enabled);
    }

    #[test]
    fn test_parametric_eq_defaults() {
        let params = ParametricEqParams::default();
        assert_eq!(params.bands.len(), 6);
    }

    #[test]
    fn test_graphic_eq_defaults() {
        let params = GraphicEqParams::default();
        assert_eq!(params.bands.len(), 31);
    }
}
