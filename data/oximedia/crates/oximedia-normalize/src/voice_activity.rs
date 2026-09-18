//! Voice Activity Detection (VAD) for `OxiMedia` normalize crate.
//!
//! Detects speech segments within an audio stream for downstream processing
//! such as dynamic normalization, noise gating, or transcript alignment.

#![allow(dead_code)]

/// State produced by the VAD for each processed frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadState {
    /// Frame contains active speech.
    Speech,
    /// Frame is in the hangover period (recently ended speech).
    Hangover,
    /// Frame is silence / non-speech.
    Silence,
}

impl VadState {
    /// True if this state counts as speech output.
    pub fn is_speech(self) -> bool {
        matches!(self, Self::Speech | Self::Hangover)
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::Hangover => "hangover",
            Self::Silence => "silence",
        }
    }
}

/// Configuration for the voice activity detector.
#[derive(Clone, Debug)]
pub struct VadConfig {
    /// Energy threshold in dBFS below which frames are considered silence.
    pub energy_threshold_db: f32,
    /// Number of hangover frames to keep VAD active after speech ends.
    pub hangover_frames: usize,
    /// Minimum number of speech frames before declaring onset.
    pub min_speech_frames: usize,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Frame length in samples.
    pub frame_length: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold_db: -40.0,
            hangover_frames: 8,
            min_speech_frames: 2,
            sample_rate: 16_000.0,
            frame_length: 256,
        }
    }
}

impl VadConfig {
    /// Create a config tuned for 16 kHz narrowband speech.
    pub fn narrowband() -> Self {
        Self {
            sample_rate: 16_000.0,
            frame_length: 256,
            ..Default::default()
        }
    }

    /// Create a config tuned for 48 kHz wideband speech.
    pub fn wideband() -> Self {
        Self {
            sample_rate: 48_000.0,
            frame_length: 480,
            ..Default::default()
        }
    }

    /// Number of hangover frames before returning to silence.
    pub fn hangover_frames(&self) -> usize {
        self.hangover_frames
    }
}

/// Energy-based Voice Activity Detector.
///
/// Uses short-time energy with a simple threshold and hangover extension
/// to label audio frames as speech or silence.
pub struct VoiceActivityDetector {
    config: VadConfig,
    /// Current hangover countdown.
    hangover_remaining: usize,
    /// Consecutive frames exceeding the energy threshold.
    speech_frame_count: usize,
    /// Current VAD output state.
    current_state: VadState,
    /// Total speech frames counted.
    total_speech_frames: u64,
    /// Total frames processed.
    total_frames: u64,
}

impl VoiceActivityDetector {
    /// Create a new `VoiceActivityDetector`.
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            hangover_remaining: 0,
            speech_frame_count: 0,
            current_state: VadState::Silence,
            total_speech_frames: 0,
            total_frames: 0,
        }
    }

    /// Compute short-time energy in dBFS for a frame of samples.
    fn frame_energy_db(frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return -120.0;
        }
        let mean_sq: f32 = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
        if mean_sq < 1e-12 {
            -120.0
        } else {
            10.0 * mean_sq.log10()
        }
    }

    /// Process one frame of audio samples and return the detected `VadState`.
    pub fn process_frame(&mut self, frame: &[f32]) -> VadState {
        let energy_db = Self::frame_energy_db(frame);
        self.total_frames += 1;

        let above_threshold = energy_db >= self.config.energy_threshold_db;

        if above_threshold {
            self.speech_frame_count += 1;
        } else {
            self.speech_frame_count = 0;
        }

        let onset = self.speech_frame_count >= self.config.min_speech_frames;

        let state = if onset {
            self.hangover_remaining = self.config.hangover_frames;
            VadState::Speech
        } else if self.hangover_remaining > 0 {
            self.hangover_remaining -= 1;
            VadState::Hangover
        } else {
            VadState::Silence
        };

        if state.is_speech() {
            self.total_speech_frames += 1;
        }

        self.current_state = state;
        state
    }

    /// True if the last processed frame was classified as speech.
    pub fn is_speech(&self) -> bool {
        self.current_state.is_speech()
    }

    /// Ratio of speech frames to total frames processed (0.0–1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn speech_ratio(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.total_speech_frames as f32 / self.total_frames as f32
    }

    /// Current VAD state.
    pub fn current_state(&self) -> VadState {
        self.current_state
    }

    /// Reset detector to initial state.
    pub fn reset(&mut self) {
        self.hangover_remaining = 0;
        self.speech_frame_count = 0;
        self.current_state = VadState::Silence;
        self.total_speech_frames = 0;
        self.total_frames = 0;
    }

    /// Total frames processed.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Configuration accessor.
    pub fn config(&self) -> &VadConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(amplitude: f32, len: usize) -> Vec<f32> {
        vec![amplitude; len]
    }

    #[test]
    fn test_vad_state_is_speech_speech() {
        assert!(VadState::Speech.is_speech());
    }

    #[test]
    fn test_vad_state_is_speech_hangover() {
        assert!(VadState::Hangover.is_speech());
    }

    #[test]
    fn test_vad_state_is_speech_silence() {
        assert!(!VadState::Silence.is_speech());
    }

    #[test]
    fn test_vad_state_labels() {
        assert_eq!(VadState::Speech.label(), "speech");
        assert_eq!(VadState::Hangover.label(), "hangover");
        assert_eq!(VadState::Silence.label(), "silence");
    }

    #[test]
    fn test_config_hangover_frames() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.hangover_frames(), 8);
    }

    #[test]
    fn test_config_narrowband() {
        let cfg = VadConfig::narrowband();
        assert!((cfg.sample_rate - 16_000.0).abs() < 1.0);
    }

    #[test]
    fn test_config_wideband() {
        let cfg = VadConfig::wideband();
        assert!((cfg.sample_rate - 48_000.0).abs() < 1.0);
    }

    #[test]
    fn test_vad_silence_detection() {
        let cfg = VadConfig::default();
        let mut vad = VoiceActivityDetector::new(cfg);
        let silent = make_frame(1e-6, 256); // very low amplitude
        let state = vad.process_frame(&silent);
        assert_eq!(state, VadState::Silence);
    }

    #[test]
    fn test_vad_speech_detection() {
        let mut cfg = VadConfig::default();
        cfg.min_speech_frames = 1;
        let mut vad = VoiceActivityDetector::new(cfg);
        let loud = make_frame(0.5, 256); // well above -40 dBFS
        let state = vad.process_frame(&loud);
        assert_eq!(state, VadState::Speech);
    }

    #[test]
    fn test_vad_hangover_after_speech() {
        let mut cfg = VadConfig::default();
        cfg.min_speech_frames = 1;
        cfg.hangover_frames = 3;
        let mut vad = VoiceActivityDetector::new(cfg);
        let loud = make_frame(0.5, 256);
        let silent = make_frame(1e-6, 256);
        vad.process_frame(&loud);
        let state = vad.process_frame(&silent);
        assert_eq!(state, VadState::Hangover);
    }

    #[test]
    fn test_vad_silence_after_hangover_expires() {
        let mut cfg = VadConfig::default();
        cfg.min_speech_frames = 1;
        cfg.hangover_frames = 1;
        let mut vad = VoiceActivityDetector::new(cfg);
        let loud = make_frame(0.5, 256);
        let silent = make_frame(1e-6, 256);
        vad.process_frame(&loud);
        vad.process_frame(&silent); // hangover
        let state = vad.process_frame(&silent); // now silence
        assert_eq!(state, VadState::Silence);
    }

    #[test]
    fn test_vad_is_speech() {
        let mut cfg = VadConfig::default();
        cfg.min_speech_frames = 1;
        let mut vad = VoiceActivityDetector::new(cfg);
        vad.process_frame(&make_frame(0.5, 256));
        assert!(vad.is_speech());
    }

    #[test]
    fn test_vad_speech_ratio_zero_frames() {
        let cfg = VadConfig::default();
        let vad = VoiceActivityDetector::new(cfg);
        assert!((vad.speech_ratio()).abs() < 1e-6);
    }

    #[test]
    fn test_vad_speech_ratio_all_speech() {
        let mut cfg = VadConfig::default();
        cfg.min_speech_frames = 1;
        let mut vad = VoiceActivityDetector::new(cfg);
        for _ in 0..10 {
            vad.process_frame(&make_frame(0.5, 256));
        }
        assert!(vad.speech_ratio() > 0.9);
    }

    #[test]
    fn test_vad_reset() {
        let mut cfg = VadConfig::default();
        cfg.min_speech_frames = 1;
        let mut vad = VoiceActivityDetector::new(cfg);
        vad.process_frame(&make_frame(0.5, 256));
        vad.reset();
        assert_eq!(vad.total_frames(), 0);
        assert_eq!(vad.current_state(), VadState::Silence);
    }

    #[test]
    fn test_vad_total_frames_count() {
        let cfg = VadConfig::default();
        let mut vad = VoiceActivityDetector::new(cfg);
        let silent = make_frame(1e-6, 256);
        for _ in 0..5 {
            vad.process_frame(&silent);
        }
        assert_eq!(vad.total_frames(), 5);
    }
}
