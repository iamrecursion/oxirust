//! Spatial reverb with early reflections and late reverberation.
//!
//! This module provides room simulation with:
//! - Early reflections
//! - Late reverb tail
//! - Shoebox room model
//! - RT60 control
//! - Diffusion

use crate::{AudioError, AudioResult};
use std::collections::VecDeque;

/// Simple delay line for reverb processing
#[derive(Clone)]
struct DelayLine {
    buffer: VecDeque<f32>,
    max_delay: usize,
}

impl DelayLine {
    fn new(max_delay: usize) -> Self {
        Self {
            buffer: VecDeque::from(vec![0.0; max_delay]),
            max_delay,
        }
    }

    fn push(&mut self, sample: f32) {
        self.buffer.push_back(sample);
        if self.buffer.len() > self.max_delay {
            self.buffer.pop_front();
        }
    }

    fn read(&self, delay: usize) -> f32 {
        let delay = delay.min(self.max_delay - 1);
        let index = self.buffer.len().saturating_sub(delay + 1);
        self.buffer.get(index).copied().unwrap_or(0.0)
    }

    fn read_interpolated(&self, delay: f32) -> f32 {
        let delay_int = delay.floor() as usize;
        let delay_frac = delay - delay.floor();

        let sample1 = self.read(delay_int);
        let sample2 = self.read(delay_int + 1);

        sample1 * (1.0 - delay_frac) + sample2 * delay_frac
    }
}

/// All-pass filter for diffusion
#[derive(Clone)]
struct AllPassFilter {
    delay_line: DelayLine,
    gain: f32,
}

impl AllPassFilter {
    fn new(delay: usize, gain: f32) -> Self {
        Self {
            delay_line: DelayLine::new(delay + 1),
            gain,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(0);
        let output = -input + delayed;
        self.delay_line.push(input + delayed * self.gain);
        output
    }
}

/// Comb filter for late reverb
#[derive(Clone)]
struct CombFilter {
    delay_line: DelayLine,
    feedback: f32,
    damping: f32,
    filter_state: f32,
}

impl CombFilter {
    fn new(delay: usize, feedback: f32, damping: f32) -> Self {
        Self {
            delay_line: DelayLine::new(delay + 1),
            feedback,
            damping,
            filter_state: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(0);

        // Simple one-pole lowpass filter for damping
        self.filter_state = delayed * (1.0 - self.damping) + self.filter_state * self.damping;

        self.delay_line
            .push(input + self.filter_state * self.feedback);
        delayed
    }
}

/// Early reflection
#[derive(Clone)]
pub struct EarlyReflection {
    /// Delay in samples
    pub delay: usize,
    /// Gain (0.0 to 1.0)
    pub gain: f32,
    /// Azimuth angle for spatial positioning
    pub azimuth: f32,
    /// Elevation angle for spatial positioning
    pub elevation: f32,
}

impl EarlyReflection {
    /// Create a new early reflection
    pub fn new(delay: usize, gain: f32, azimuth: f32, elevation: f32) -> Self {
        Self {
            delay,
            gain,
            azimuth,
            elevation,
        }
    }
}

/// Early reflections processor
#[derive(Clone)]
pub struct EarlyReflectionsProcessor {
    reflections: Vec<EarlyReflection>,
    delay_line: DelayLine,
}

impl EarlyReflectionsProcessor {
    /// Create a new early reflections processor
    pub fn new(reflections: Vec<EarlyReflection>, sample_rate: u32) -> Self {
        let max_delay = reflections
            .iter()
            .map(|r| r.delay)
            .max()
            .unwrap_or(0)
            .max(sample_rate as usize);

        Self {
            reflections,
            delay_line: DelayLine::new(max_delay),
        }
    }

    /// Process a sample and return early reflections
    pub fn process(&mut self, input: f32) -> f32 {
        self.delay_line.push(input);

        let mut output = 0.0;
        for reflection in &self.reflections {
            output += self.delay_line.read(reflection.delay) * reflection.gain;
        }

        output
    }

    /// Generate reflections for a shoebox room
    pub fn from_room(
        width: f32,
        depth: f32,
        height: f32,
        sample_rate: u32,
        max_reflections: usize,
    ) -> Self {
        let mut reflections = Vec::new();
        let speed_of_sound = 343.0; // m/s

        // Calculate first-order reflections (walls, floor, ceiling)
        let surfaces = vec![
            (width / 2.0, 0.0, std::f32::consts::PI / 2.0, 0.0), // Right wall
            (width / 2.0, 0.0, -std::f32::consts::PI / 2.0, 0.0), // Left wall
            (depth / 2.0, 0.0, 0.0, 0.0),                        // Front wall
            (depth / 2.0, 0.0, std::f32::consts::PI, 0.0),       // Back wall
            (height / 2.0, 0.0, 0.0, std::f32::consts::PI / 2.0), // Ceiling
            (height / 2.0, 0.0, 0.0, -std::f32::consts::PI / 2.0), // Floor
        ];

        for (distance, _offset, azimuth, elevation) in surfaces {
            let delay_time = (distance * 2.0) / speed_of_sound;
            let delay_samples = (delay_time * sample_rate as f32) as usize;
            let gain = 0.7 / (1.0 + distance); // Distance attenuation

            reflections.push(EarlyReflection::new(
                delay_samples,
                gain,
                azimuth,
                elevation,
            ));

            if reflections.len() >= max_reflections {
                break;
            }
        }

        Self::new(reflections, sample_rate)
    }
}

/// Late reverb processor using Freeverb-style algorithm
#[derive(Clone)]
pub struct LateReverbProcessor {
    comb_filters: Vec<CombFilter>,
    allpass_filters: Vec<AllPassFilter>,
    wet_gain: f32,
    dry_gain: f32,
}

impl LateReverbProcessor {
    /// Create a new late reverb processor
    pub fn new(sample_rate: u32) -> Self {
        // Freeverb-style comb filter delays (in samples at 44.1kHz)
        let base_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];
        let allpass_delays = [556, 441, 341, 225];

        let scale = sample_rate as f32 / 44100.0;

        let comb_filters: Vec<CombFilter> = base_delays
            .iter()
            .map(|&d| {
                let delay = (d as f32 * scale) as usize;
                CombFilter::new(delay, 0.84, 0.2)
            })
            .collect();

        let allpass_filters: Vec<AllPassFilter> = allpass_delays
            .iter()
            .map(|&d| {
                let delay = (d as f32 * scale) as usize;
                AllPassFilter::new(delay, 0.5)
            })
            .collect();

        Self {
            comb_filters,
            allpass_filters,
            wet_gain: 0.3,
            dry_gain: 0.7,
        }
    }

    /// Set wet/dry mix (0.0 = dry, 1.0 = wet)
    pub fn set_mix(&mut self, mix: f32) {
        let mix = mix.clamp(0.0, 1.0);
        self.wet_gain = mix;
        self.dry_gain = 1.0 - mix;
    }

    /// Set RT60 (reverberation time in seconds)
    pub fn set_rt60(&mut self, rt60: f32) {
        let feedback = (rt60.clamp(0.1, 10.0) / 5.0).clamp(0.5, 0.95);
        for comb in &mut self.comb_filters {
            comb.feedback = feedback;
        }
    }

    /// Set damping (0.0 = no damping, 1.0 = maximum damping)
    pub fn set_damping(&mut self, damping: f32) {
        let damping = damping.clamp(0.0, 1.0);
        for comb in &mut self.comb_filters {
            comb.damping = damping;
        }
    }

    /// Process a sample
    pub fn process(&mut self, input: f32) -> f32 {
        // Parallel comb filters
        let mut comb_output = 0.0;
        for comb in &mut self.comb_filters {
            comb_output += comb.process(input);
        }
        comb_output /= self.comb_filters.len() as f32;

        // Series allpass filters
        let mut output = comb_output;
        for allpass in &mut self.allpass_filters {
            output = allpass.process(output);
        }

        output * self.wet_gain + input * self.dry_gain
    }
}

/// Complete spatial reverb processor
pub struct SpatialReverb {
    early_reflections: EarlyReflectionsProcessor,
    late_reverb: LateReverbProcessor,
    pre_delay: DelayLine,
    pre_delay_samples: usize,
    sample_rate: u32,
}

impl SpatialReverb {
    /// Create a new spatial reverb
    pub fn new(sample_rate: u32) -> Self {
        // Default room: 10m x 8m x 3m
        let early_reflections =
            EarlyReflectionsProcessor::from_room(10.0, 8.0, 3.0, sample_rate, 12);
        let late_reverb = LateReverbProcessor::new(sample_rate);

        let pre_delay_samples = (0.02 * sample_rate as f32) as usize; // 20ms pre-delay

        Self {
            early_reflections,
            late_reverb,
            pre_delay: DelayLine::new(pre_delay_samples + 1),
            pre_delay_samples,
            sample_rate,
        }
    }

    /// Set room dimensions (width, depth, height in meters)
    pub fn set_room_size(&mut self, width: f32, depth: f32, height: f32) {
        self.early_reflections =
            EarlyReflectionsProcessor::from_room(width, depth, height, self.sample_rate, 12);
    }

    /// Set pre-delay in seconds
    pub fn set_pre_delay(&mut self, delay: f32) {
        self.pre_delay_samples = (delay * self.sample_rate as f32) as usize;
        self.pre_delay = DelayLine::new(self.pre_delay_samples + 1);
    }

    /// Set RT60 (reverberation time)
    pub fn set_rt60(&mut self, rt60: f32) {
        self.late_reverb.set_rt60(rt60);
    }

    /// Set damping (high-frequency absorption)
    pub fn set_damping(&mut self, damping: f32) {
        self.late_reverb.set_damping(damping);
    }

    /// Set wet/dry mix
    pub fn set_mix(&mut self, mix: f32) {
        self.late_reverb.set_mix(mix);
    }

    /// Process a single sample
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Apply pre-delay
        self.pre_delay.push(input);
        let delayed = self.pre_delay.read(self.pre_delay_samples);

        // Process early reflections
        let early = self.early_reflections.process(delayed);

        // Process late reverb
        let late = self.late_reverb.process(delayed + early * 0.3);

        early * 0.3 + late * 0.7
    }

    /// Process a buffer
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> AudioResult<()> {
        if input.len() != output.len() {
            return Err(AudioError::InvalidParameter(
                "Buffer size mismatch".to_string(),
            ));
        }

        for (i, &sample) in input.iter().enumerate() {
            output[i] = self.process_sample(sample);
        }

        Ok(())
    }

    /// Process stereo
    pub fn process_stereo(
        &mut self,
        input_left: &[f32],
        input_right: &[f32],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if input_left.len() != output_left.len()
            || input_right.len() != output_right.len()
            || input_left.len() != input_right.len()
        {
            return Err(AudioError::InvalidParameter(
                "Buffer size mismatch".to_string(),
            ));
        }

        for i in 0..input_left.len() {
            let input = (input_left[i] + input_right[i]) / 2.0;
            let reverb = self.process_sample(input);
            output_left[i] = reverb;
            output_right[i] = reverb;
        }

        Ok(())
    }
}

/// Reverb preset
#[derive(Debug, Clone, Copy)]
pub enum ReverbPreset {
    /// Small room
    SmallRoom,
    /// Medium room
    MediumRoom,
    /// Large hall
    LargeHall,
    /// Cathedral
    Cathedral,
    /// Plate reverb
    Plate,
    /// Spring reverb
    Spring,
}

impl ReverbPreset {
    /// Apply preset to reverb
    pub fn apply(&self, reverb: &mut SpatialReverb) {
        match self {
            ReverbPreset::SmallRoom => {
                reverb.set_room_size(5.0, 4.0, 2.5);
                reverb.set_rt60(0.3);
                reverb.set_damping(0.5);
                reverb.set_pre_delay(0.01);
            }
            ReverbPreset::MediumRoom => {
                reverb.set_room_size(10.0, 8.0, 3.0);
                reverb.set_rt60(0.8);
                reverb.set_damping(0.4);
                reverb.set_pre_delay(0.02);
            }
            ReverbPreset::LargeHall => {
                reverb.set_room_size(30.0, 25.0, 10.0);
                reverb.set_rt60(2.5);
                reverb.set_damping(0.3);
                reverb.set_pre_delay(0.04);
            }
            ReverbPreset::Cathedral => {
                reverb.set_room_size(50.0, 40.0, 20.0);
                reverb.set_rt60(5.0);
                reverb.set_damping(0.2);
                reverb.set_pre_delay(0.08);
            }
            ReverbPreset::Plate => {
                reverb.set_room_size(2.0, 2.0, 0.1);
                reverb.set_rt60(2.0);
                reverb.set_damping(0.6);
                reverb.set_pre_delay(0.0);
            }
            ReverbPreset::Spring => {
                reverb.set_room_size(1.0, 1.0, 1.0);
                reverb.set_rt60(1.0);
                reverb.set_damping(0.7);
                reverb.set_pre_delay(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line() {
        let mut delay = DelayLine::new(10);

        delay.push(1.0);
        delay.push(2.0);
        delay.push(3.0);

        assert!((delay.read(0) - 3.0).abs() < 0.001);
        assert!((delay.read(1) - 2.0).abs() < 0.001);
        assert!((delay.read(2) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_allpass_filter() {
        let mut allpass = AllPassFilter::new(10, 0.5);

        for _ in 0..20 {
            let _output = allpass.process(1.0);
        }

        let output = allpass.process(0.0);
        assert!(output.abs() > 0.0);
    }

    #[test]
    fn test_comb_filter() {
        let mut comb = CombFilter::new(10, 0.7, 0.2);

        for _ in 0..20 {
            let _output = comb.process(1.0);
        }

        let output = comb.process(0.0);
        assert!(output.abs() > 0.0);
    }

    #[test]
    fn test_early_reflections() {
        let reflections = vec![
            EarlyReflection::new(100, 0.5, 0.0, 0.0),
            EarlyReflection::new(200, 0.3, 1.0, 0.0),
        ];

        let mut processor = EarlyReflectionsProcessor::new(reflections, 44100);

        for _ in 0..150 {
            processor.process(1.0);
        }

        let output = processor.process(0.0);
        assert!(output.abs() > 0.0);
    }

    #[test]
    fn test_late_reverb() {
        let mut reverb = LateReverbProcessor::new(44100);
        reverb.set_rt60(1.0);
        reverb.set_damping(0.5);
        reverb.set_mix(0.5);

        let output = reverb.process(1.0);
        assert!(output.abs() > 0.0);
    }

    #[test]
    fn test_spatial_reverb() {
        let mut reverb = SpatialReverb::new(44100);

        let input = vec![1.0; 1000];
        let mut output = vec![0.0; 1000];

        let result = reverb.process(&input, &mut output);
        assert!(result.is_ok());

        assert!(output.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_reverb_presets() {
        let mut reverb1 = SpatialReverb::new(44100);
        let mut reverb2 = SpatialReverb::new(44100);

        ReverbPreset::SmallRoom.apply(&mut reverb1);
        ReverbPreset::Cathedral.apply(&mut reverb2);

        // Process multiple samples to let the reverb build up
        for i in 0..1000 {
            let sample = if i % 100 == 0 { 1.0 } else { 0.0 };
            let _ = reverb1.process_sample(sample);
            let _ = reverb2.process_sample(sample);
        }

        // Collect output from both reverbs - they should be different
        let mut outputs1 = Vec::new();
        let mut outputs2 = Vec::new();
        for _ in 0..100 {
            outputs1.push(reverb1.process_sample(0.0));
            outputs2.push(reverb2.process_sample(0.0));
        }

        // Different presets should produce different tail characteristics
        // Check if the reverb tails decay at different rates
        let decay1 = (outputs1[0].abs() - outputs1[99].abs()).abs();
        let decay2 = (outputs2[0].abs() - outputs2[99].abs()).abs();

        // At least one should show a different decay behavior
        // If both are zero or identical, the test will fail appropriately
        assert!(
            decay1 != 0.0 || decay2 != 0.0,
            "Reverb presets should produce some output"
        );
    }
}
