//! Meter ballistics for realistic meter behavior.
//!
//! Implements various ballistic standards for audio meters including
//! VU meters, PPM meters (EBU, BBC, DIN), and custom attack/release characteristics.

/// Ballistic type for meter movement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BallisticType {
    /// VU meter: 300ms integration time to reach 99% of full scale.
    Vu,
    /// PPM EBU: 10ms integration, 20dB/1.5s return.
    PpmEbu,
    /// PPM BBC: 10ms integration, 24dB/2.8s return.
    PpmBbc,
    /// PPM DIN: 10ms integration, 20dB/1.5s return (similar to EBU).
    PpmDin,
    /// Fast peak: 1-2 sample attack, fast release.
    FastPeak,
    /// Custom ballistics.
    Custom {
        /// Attack time constant in seconds.
        attack_time: f64,
        /// Release time constant in seconds.
        release_time: f64,
    },
}

impl BallisticType {
    /// Get attack time constant in seconds.
    pub fn attack_time(&self) -> f64 {
        match self {
            Self::Vu => 0.3,                     // 300ms integration
            Self::PpmEbu | Self::PpmDin => 0.01, // 10ms integration
            Self::PpmBbc => 0.01,                // 10ms integration
            Self::FastPeak => 0.000_02,          // ~1 sample at 48kHz
            Self::Custom { attack_time, .. } => *attack_time,
        }
    }

    /// Get release time constant in seconds.
    pub fn release_time(&self) -> f64 {
        match self {
            Self::Vu => 0.3,                    // Same as attack for VU
            Self::PpmEbu | Self::PpmDin => 1.5, // 20dB/1.5s
            Self::PpmBbc => 2.8,                // 24dB/2.8s
            Self::FastPeak => 0.1,              // Fast release
            Self::Custom { release_time, .. } => *release_time,
        }
    }
}

/// Ballistic processor for applying attack/release characteristics to meter values.
pub struct BallisticProcessor {
    ballistic_type: BallisticType,
    sample_rate: f64,
    attack_coeff: f64,
    release_coeff: f64,
    current_value: f64,
    peak_hold_time: f64,
    peak_hold_samples: usize,
    peak_hold_counter: usize,
    peak_hold_value: f64,
}

impl BallisticProcessor {
    /// Create a new ballistic processor.
    ///
    /// # Arguments
    ///
    /// * `ballistic_type` - Type of ballistics to apply
    /// * `sample_rate` - Sample rate in Hz
    /// * `peak_hold_time` - Peak hold time in seconds (0.0 for no hold)
    pub fn new(ballistic_type: BallisticType, sample_rate: f64, peak_hold_time: f64) -> Self {
        let attack_time = ballistic_type.attack_time();
        let release_time = ballistic_type.release_time();

        // Calculate attack/release coefficients for exponential smoothing
        // coeff = exp(-1 / (time_constant * sample_rate))
        let attack_coeff = (-1.0 / (attack_time * sample_rate)).exp();
        let release_coeff = (-1.0 / (release_time * sample_rate)).exp();

        let peak_hold_samples = (peak_hold_time * sample_rate) as usize;

        Self {
            ballistic_type,
            sample_rate,
            attack_coeff,
            release_coeff,
            current_value: 0.0,
            peak_hold_time,
            peak_hold_samples,
            peak_hold_counter: 0,
            peak_hold_value: 0.0,
        }
    }

    /// Process a single input value and apply ballistics.
    ///
    /// # Arguments
    ///
    /// * `input` - Input value (linear scale)
    ///
    /// # Returns
    ///
    /// Ballistically filtered value
    pub fn process(&mut self, input: f64) -> f64 {
        // Apply attack or release based on whether input is rising or falling
        if input > self.current_value {
            // Attack: input is rising
            self.current_value = input + self.attack_coeff * (self.current_value - input);
        } else {
            // Release: input is falling
            self.current_value = input + self.release_coeff * (self.current_value - input);
        }

        // Update peak hold
        if input > self.peak_hold_value {
            self.peak_hold_value = input;
            self.peak_hold_counter = self.peak_hold_samples;
        } else if self.peak_hold_counter > 0 {
            self.peak_hold_counter -= 1;
        } else {
            self.peak_hold_value = self.current_value;
        }

        self.current_value
    }

    /// Get the current ballistically filtered value.
    pub fn current_value(&self) -> f64 {
        self.current_value
    }

    /// Get the peak hold value.
    pub fn peak_hold_value(&self) -> f64 {
        self.peak_hold_value
    }

    /// Reset the processor to initial state.
    pub fn reset(&mut self) {
        self.current_value = 0.0;
        self.peak_hold_counter = 0;
        self.peak_hold_value = 0.0;
    }

    /// Get the ballistic type.
    pub fn ballistic_type(&self) -> BallisticType {
        self.ballistic_type
    }
}

/// Multi-channel ballistic processor.
pub struct MultiChannelBallistics {
    channels: Vec<BallisticProcessor>,
}

impl MultiChannelBallistics {
    /// Create a new multi-channel ballistic processor.
    ///
    /// # Arguments
    ///
    /// * `num_channels` - Number of channels
    /// * `ballistic_type` - Type of ballistics to apply
    /// * `sample_rate` - Sample rate in Hz
    /// * `peak_hold_time` - Peak hold time in seconds
    pub fn new(
        num_channels: usize,
        ballistic_type: BallisticType,
        sample_rate: f64,
        peak_hold_time: f64,
    ) -> Self {
        let channels = (0..num_channels)
            .map(|_| BallisticProcessor::new(ballistic_type, sample_rate, peak_hold_time))
            .collect();

        Self { channels }
    }

    /// Process samples for all channels.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input values for each channel (linear scale)
    ///
    /// # Returns
    ///
    /// Ballistically filtered values for each channel
    pub fn process(&mut self, inputs: &[f64]) -> Vec<f64> {
        inputs
            .iter()
            .zip(&mut self.channels)
            .map(|(&input, processor)| processor.process(input))
            .collect()
    }

    /// Get current values for all channels.
    pub fn current_values(&self) -> Vec<f64> {
        self.channels
            .iter()
            .map(BallisticProcessor::current_value)
            .collect()
    }

    /// Get peak hold values for all channels.
    pub fn peak_hold_values(&self) -> Vec<f64> {
        self.channels
            .iter()
            .map(BallisticProcessor::peak_hold_value)
            .collect()
    }

    /// Reset all channels.
    pub fn reset(&mut self) {
        for channel in &mut self.channels {
            channel.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ballistic_attack() {
        let mut ballistics = BallisticProcessor::new(BallisticType::FastPeak, 48000.0, 0.0);

        // Apply step input
        let result = ballistics.process(1.0);

        // Should rise quickly but not instantly
        assert!(result > 0.5);
        assert!(result < 1.0);
    }

    #[test]
    fn test_ballistic_release() {
        let mut ballistics = BallisticProcessor::new(BallisticType::FastPeak, 48000.0, 0.0);

        // Rise to 1.0
        for _ in 0..1000 {
            ballistics.process(1.0);
        }

        // Then fall
        let result = ballistics.process(0.0);

        // Should fall but not instantly
        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_peak_hold() {
        let mut ballistics = BallisticProcessor::new(
            BallisticType::FastPeak,
            48000.0,
            1.0, // 1 second hold
        );

        ballistics.process(1.0);

        // Peak hold should maintain value
        for _ in 0..100 {
            ballistics.process(0.0);
        }

        assert_eq!(ballistics.peak_hold_value(), 1.0);
    }

    #[test]
    fn test_vu_ballistics() {
        let ballistics = BallisticProcessor::new(BallisticType::Vu, 48000.0, 0.0);

        assert_eq!(ballistics.ballistic_type().attack_time(), 0.3);
        assert_eq!(ballistics.ballistic_type().release_time(), 0.3);
    }

    #[test]
    fn test_ppm_ebu_ballistics() {
        let ballistics = BallisticProcessor::new(BallisticType::PpmEbu, 48000.0, 0.0);

        assert_eq!(ballistics.ballistic_type().attack_time(), 0.01);
        assert_eq!(ballistics.ballistic_type().release_time(), 1.5);
    }

    #[test]
    fn test_multi_channel_ballistics() {
        let mut ballistics = MultiChannelBallistics::new(2, BallisticType::FastPeak, 48000.0, 0.0);

        let inputs = vec![0.5, 1.0];
        let outputs = ballistics.process(&inputs);

        assert_eq!(outputs.len(), 2);
        assert!(outputs[0] > 0.0);
        assert!(outputs[1] > 0.0);
    }
}
