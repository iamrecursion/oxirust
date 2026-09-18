//! Ambisonics encoding and decoding.
//!
//! This module provides:
//! - B-format encoding (W, X, Y, Z)
//! - Higher-order Ambisonics (HOA) up to 3rd order
//! - Spherical harmonics computation
//! - ACN (Ambisonic Channel Number) ordering
//! - N3D (Normalized 3D) normalization
//! - Ambisonic rotation
//! - Decoding to speaker arrays

use crate::{AudioError, AudioResult};
use std::f32::consts::PI;

/// Ambisonic order
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AmbisonicOrder {
    /// First order (4 channels: W, X, Y, Z)
    First,
    /// Second order (9 channels)
    Second,
    /// Third order (16 channels)
    Third,
}

impl AmbisonicOrder {
    /// Get the number of channels for this order
    pub fn channel_count(&self) -> usize {
        match self {
            AmbisonicOrder::First => 4,
            AmbisonicOrder::Second => 9,
            AmbisonicOrder::Third => 16,
        }
    }

    /// Get the maximum degree for this order
    pub fn max_degree(&self) -> i32 {
        match self {
            AmbisonicOrder::First => 1,
            AmbisonicOrder::Second => 2,
            AmbisonicOrder::Third => 3,
        }
    }
}

/// Spherical coordinate
#[derive(Debug, Clone, Copy)]
pub struct SphericalCoord {
    /// Azimuth angle in radians (-π to π, 0 = front)
    pub azimuth: f32,
    /// Elevation angle in radians (-π/2 to π/2, 0 = horizontal)
    pub elevation: f32,
    /// Distance (typically normalized to 1.0)
    pub distance: f32,
}

impl SphericalCoord {
    /// Create a new spherical coordinate
    pub fn new(azimuth: f32, elevation: f32, distance: f32) -> Self {
        Self {
            azimuth,
            elevation,
            distance,
        }
    }

    /// Convert to Cartesian coordinates
    pub fn to_cartesian(&self) -> (f32, f32, f32) {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.cos() * self.azimuth.cos();
        let z = self.distance * self.elevation.sin();
        (x, y, z)
    }

    /// Create from Cartesian coordinates
    pub fn from_cartesian(x: f32, y: f32, z: f32) -> Self {
        let distance = (x * x + y * y + z * z).sqrt();
        let azimuth = x.atan2(y);
        let elevation = if distance > 0.0 {
            (z / distance).asin()
        } else {
            0.0
        };
        Self {
            azimuth,
            elevation,
            distance,
        }
    }
}

/// Compute factorial
fn factorial(n: u32) -> f64 {
    (1..=n).map(|x| x as f64).product()
}

/// Compute associated Legendre polynomial P_l^m(x)
fn associated_legendre(l: i32, m: i32, x: f32) -> f32 {
    let x = x.clamp(-1.0, 1.0);
    let m_abs = m.abs();

    // Base cases
    if l < m_abs {
        return 0.0;
    }

    if l == 0 && m == 0 {
        return 1.0;
    }

    // Use recurrence relations
    let mut pmm = 1.0;
    if m_abs > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let fact = 1.0;
        for _i in 1..=m_abs {
            pmm *= -fact * somx2;
        }
    }

    if l == m_abs {
        return pmm;
    }

    // Compute P_{m+1}^m
    let mut pmmp1 = x * (2 * m_abs + 1) as f32 * pmm;

    if l == m_abs + 1 {
        return pmmp1;
    }

    // Compute P_l^m for l > m+1 using recurrence
    let mut pll = 0.0;
    for ll in (m_abs + 2)..=l {
        pll =
            ((2 * ll - 1) as f32 * x * pmmp1 - (ll + m_abs - 1) as f32 * pmm) / (ll - m_abs) as f32;
        pmm = pmmp1;
        pmmp1 = pll;
    }

    pll
}

/// Compute spherical harmonic Y_l^m(θ, φ) with N3D normalization
fn spherical_harmonic_n3d(l: i32, m: i32, azimuth: f32, elevation: f32) -> f32 {
    // N3D normalization factor
    let normalization = if m == 0 {
        ((2 * l + 1) as f64 / (4.0 * PI as f64)).sqrt() as f32
    } else {
        let m_abs = m.abs();
        let factor = (2.0 * (2 * l + 1) as f64 * factorial((l - m_abs) as u32)
            / factorial((l + m_abs) as u32))
            / (4.0 * PI as f64);
        factor.sqrt() as f32
    };

    // Compute associated Legendre polynomial
    let sin_elevation = elevation.sin();
    let p_lm = associated_legendre(l, m.abs(), sin_elevation);

    // Compute azimuthal component
    let azimuthal = if m > 0 {
        (m as f32 * azimuth).cos()
    } else if m < 0 {
        (m.abs() as f32 * azimuth).sin()
    } else {
        1.0
    };

    normalization * p_lm * azimuthal
}

/// ACN (Ambisonic Channel Number) index for degree l and order m
fn acn_index(l: i32, m: i32) -> usize {
    (l * (l + 1) + m) as usize
}

/// Ambisonic encoder
pub struct AmbisonicEncoder {
    /// Ambisonic order
    order: AmbisonicOrder,
    /// Number of channels
    channel_count: usize,
}

impl AmbisonicEncoder {
    /// Create a new ambisonic encoder
    pub fn new(order: AmbisonicOrder) -> Self {
        let channel_count = order.channel_count();
        Self {
            order,
            channel_count,
        }
    }

    /// Encode a mono signal to ambisonic format
    pub fn encode(
        &self,
        input: &[f32],
        direction: SphericalCoord,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        if output.len() != self.channel_count {
            return Err(AudioError::InvalidParameter(format!(
                "Expected {} output channels, got {}",
                self.channel_count,
                output.len()
            )));
        }

        for channel in output.iter() {
            if channel.len() != input.len() {
                return Err(AudioError::InvalidParameter(
                    "Buffer size mismatch".to_string(),
                ));
            }
        }

        // Compute spherical harmonic coefficients
        let coefficients = self.compute_coefficients(direction);

        // Encode each sample
        for sample_idx in 0..input.len() {
            let sample = input[sample_idx];
            for (channel_idx, coefficient) in coefficients.iter().enumerate() {
                output[channel_idx][sample_idx] = sample * coefficient;
            }
        }

        Ok(())
    }

    /// Compute encoding coefficients for a direction
    pub fn compute_coefficients(&self, direction: SphericalCoord) -> Vec<f32> {
        let mut coefficients = vec![0.0; self.channel_count];
        let max_degree = self.order.max_degree();

        for l in 0..=max_degree {
            for m in -l..=l {
                let idx = acn_index(l, m);
                if idx < self.channel_count {
                    coefficients[idx] =
                        spherical_harmonic_n3d(l, m, direction.azimuth, direction.elevation);
                }
            }
        }

        coefficients
    }

    /// Encode with distance attenuation
    pub fn encode_with_distance(
        &self,
        input: &[f32],
        direction: SphericalCoord,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        // Apply distance attenuation (inverse square law)
        let attenuation = 1.0 / (direction.distance * direction.distance).max(0.01);
        let mut attenuated = vec![0.0; input.len()];
        for (i, &sample) in input.iter().enumerate() {
            attenuated[i] = sample * attenuation;
        }

        self.encode(&attenuated, direction, output)
    }

    /// Get the number of channels
    pub fn channel_count(&self) -> usize {
        self.channel_count
    }
}

/// Speaker configuration for ambisonic decoding
#[derive(Debug, Clone)]
pub struct SpeakerConfig {
    /// Speaker positions in spherical coordinates
    pub positions: Vec<SphericalCoord>,
}

impl SpeakerConfig {
    /// Create a new speaker configuration
    pub fn new(positions: Vec<SphericalCoord>) -> Self {
        Self { positions }
    }

    /// Stereo configuration
    pub fn stereo() -> Self {
        Self {
            positions: vec![
                SphericalCoord::new(-30.0_f32.to_radians(), 0.0, 1.0),
                SphericalCoord::new(30.0_f32.to_radians(), 0.0, 1.0),
            ],
        }
    }

    /// Quad configuration
    pub fn quad() -> Self {
        Self {
            positions: vec![
                SphericalCoord::new(-45.0_f32.to_radians(), 0.0, 1.0),
                SphericalCoord::new(45.0_f32.to_radians(), 0.0, 1.0),
                SphericalCoord::new(-135.0_f32.to_radians(), 0.0, 1.0),
                SphericalCoord::new(135.0_f32.to_radians(), 0.0, 1.0),
            ],
        }
    }

    /// 5.1 surround configuration
    pub fn surround_5_1() -> Self {
        Self {
            positions: vec![
                SphericalCoord::new(-30.0_f32.to_radians(), 0.0, 1.0), // L
                SphericalCoord::new(30.0_f32.to_radians(), 0.0, 1.0),  // R
                SphericalCoord::new(0.0, 0.0, 1.0),                    // C
                SphericalCoord::new(0.0, 0.0, 1.0), // LFE (same as C for simplicity)
                SphericalCoord::new(-110.0_f32.to_radians(), 0.0, 1.0), // LS
                SphericalCoord::new(110.0_f32.to_radians(), 0.0, 1.0), // RS
            ],
        }
    }

    /// Horizontal ring (for testing)
    pub fn horizontal_ring(count: usize) -> Self {
        let mut positions = Vec::new();
        for i in 0..count {
            let azimuth = 2.0 * PI * i as f32 / count as f32;
            positions.push(SphericalCoord::new(azimuth, 0.0, 1.0));
        }
        Self { positions }
    }
}

/// Ambisonic decoder using pseudoinverse decoding
pub struct AmbisonicDecoder {
    /// Ambisonic order
    order: AmbisonicOrder,
    /// Number of input channels
    input_channel_count: usize,
    /// Speaker configuration
    speaker_config: SpeakerConfig,
    /// Decoding matrix
    decode_matrix: Vec<Vec<f32>>,
}

impl AmbisonicDecoder {
    /// Create a new ambisonic decoder
    pub fn new(order: AmbisonicOrder, speaker_config: SpeakerConfig) -> Self {
        let input_channel_count = order.channel_count();
        let decode_matrix = Self::compute_decode_matrix(order, &speaker_config);

        Self {
            order,
            input_channel_count,
            speaker_config,
            decode_matrix,
        }
    }

    /// Compute decoding matrix using pseudoinverse method
    fn compute_decode_matrix(
        order: AmbisonicOrder,
        speaker_config: &SpeakerConfig,
    ) -> Vec<Vec<f32>> {
        let num_speakers = speaker_config.positions.len();
        let num_channels = order.channel_count();

        // Build encoding matrix (speakers x ambisonics)
        let mut encode_matrix = vec![vec![0.0; num_channels]; num_speakers];

        let encoder = AmbisonicEncoder::new(order);
        for (speaker_idx, position) in speaker_config.positions.iter().enumerate() {
            let coefficients = encoder.compute_coefficients(*position);
            encode_matrix[speaker_idx] = coefficients;
        }

        // For simplicity, use Moore-Penrose pseudoinverse approximation
        // Real implementation would use proper SVD
        let mut decode_matrix = vec![vec![0.0; num_speakers]; num_channels];

        // Simple transpose for now (works reasonably for well-distributed speakers)
        for i in 0..num_channels {
            for j in 0..num_speakers {
                decode_matrix[i][j] = encode_matrix[j][i];
            }
        }

        // Normalize
        let scale = 1.0 / num_speakers as f32;
        for row in &mut decode_matrix {
            for val in row {
                *val *= scale;
            }
        }

        decode_matrix
    }

    /// Decode ambisonic signal to speaker array
    pub fn decode(&self, input: &[Vec<f32>], output: &mut [Vec<f32>]) -> AudioResult<()> {
        if input.len() != self.input_channel_count {
            return Err(AudioError::InvalidParameter(format!(
                "Expected {} input channels, got {}",
                self.input_channel_count,
                input.len()
            )));
        }

        if output.len() != self.speaker_config.positions.len() {
            return Err(AudioError::InvalidParameter(format!(
                "Expected {} output channels, got {}",
                self.speaker_config.positions.len(),
                output.len()
            )));
        }

        let buffer_size = input[0].len();
        for channel in input.iter() {
            if channel.len() != buffer_size {
                return Err(AudioError::InvalidParameter(
                    "Input buffer size mismatch".to_string(),
                ));
            }
        }

        for channel in output.iter() {
            if channel.len() != buffer_size {
                return Err(AudioError::InvalidParameter(
                    "Output buffer size mismatch".to_string(),
                ));
            }
        }

        // Matrix multiplication: output = decode_matrix * input
        for sample_idx in 0..buffer_size {
            for (speaker_idx, speaker_output) in output.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (channel_idx, channel_input) in input.iter().enumerate() {
                    sum += self.decode_matrix[channel_idx][speaker_idx] * channel_input[sample_idx];
                }
                speaker_output[sample_idx] = sum;
            }
        }

        Ok(())
    }

    /// Get number of output speakers
    pub fn speaker_count(&self) -> usize {
        self.speaker_config.positions.len()
    }
}

/// Rotation angles
#[derive(Debug, Clone, Copy)]
pub struct RotationAngles {
    /// Yaw (rotation around Z axis)
    pub yaw: f32,
    /// Pitch (rotation around X axis)
    pub pitch: f32,
    /// Roll (rotation around Y axis)
    pub roll: f32,
}

impl RotationAngles {
    /// Create new rotation angles
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        Self { yaw, pitch, roll }
    }

    /// No rotation
    pub fn identity() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }
}

/// Ambisonic rotation (simplified first-order only)
pub struct AmbisonicRotator {
    /// Current rotation
    rotation: RotationAngles,
}

impl AmbisonicRotator {
    /// Create a new rotator
    pub fn new() -> Self {
        Self {
            rotation: RotationAngles::identity(),
        }
    }

    /// Set rotation
    pub fn set_rotation(&mut self, rotation: RotationAngles) {
        self.rotation = rotation;
    }

    /// Rotate first-order ambisonic signal (W, X, Y, Z)
    pub fn rotate_first_order(
        &self,
        input: &[Vec<f32>],
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        if input.len() != 4 || output.len() != 4 {
            return Err(AudioError::InvalidParameter(
                "First-order Ambisonics requires 4 channels".to_string(),
            ));
        }

        let buffer_size = input[0].len();

        // Build rotation matrix for first-order
        let (sin_yaw, cos_yaw) = (self.rotation.yaw.sin(), self.rotation.yaw.cos());
        let (sin_pitch, cos_pitch) = (self.rotation.pitch.sin(), self.rotation.pitch.cos());
        let (sin_roll, cos_roll) = (self.rotation.roll.sin(), self.rotation.roll.cos());

        // Combined rotation matrix (simplified)
        let m11 = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll;
        let m12 = -cos_yaw * sin_roll - sin_yaw * sin_pitch * cos_roll;
        let m13 = -sin_yaw * cos_pitch;

        let m21 = cos_pitch * sin_roll;
        let m22 = cos_pitch * cos_roll;
        let m23 = -sin_pitch;

        let m31 = sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll;
        let m32 = -sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll;
        let m33 = cos_yaw * cos_pitch;

        // Apply rotation: W stays the same, rotate X, Y, Z
        for i in 0..buffer_size {
            output[0][i] = input[0][i]; // W (omnidirectional)

            let x = input[1][i];
            let y = input[2][i];
            let z = input[3][i];

            output[1][i] = m11 * x + m12 * y + m13 * z; // X
            output[2][i] = m21 * x + m22 * y + m23 * z; // Y
            output[3][i] = m31 * x + m32 * y + m33 * z; // Z
        }

        Ok(())
    }
}

impl Default for AmbisonicRotator {
    fn default() -> Self {
        Self::new()
    }
}

/// Ambisonic field processor
pub struct AmbisonicProcessor {
    encoder: AmbisonicEncoder,
    decoder: AmbisonicDecoder,
    rotator: AmbisonicRotator,
    order: AmbisonicOrder,
}

impl AmbisonicProcessor {
    /// Create a new ambisonic processor
    pub fn new(order: AmbisonicOrder, speaker_config: SpeakerConfig) -> Self {
        Self {
            encoder: AmbisonicEncoder::new(order),
            decoder: AmbisonicDecoder::new(order, speaker_config),
            rotator: AmbisonicRotator::new(),
            order,
        }
    }

    /// Get the encoder
    pub fn encoder(&self) -> &AmbisonicEncoder {
        &self.encoder
    }

    /// Get the encoder (mutable)
    pub fn encoder_mut(&mut self) -> &mut AmbisonicEncoder {
        &mut self.encoder
    }

    /// Get the decoder
    pub fn decoder(&self) -> &AmbisonicDecoder {
        &self.decoder
    }

    /// Get the decoder (mutable)
    pub fn decoder_mut(&mut self) -> &mut AmbisonicDecoder {
        &mut self.decoder
    }

    /// Get the rotator
    pub fn rotator(&self) -> &AmbisonicRotator {
        &self.rotator
    }

    /// Get the rotator (mutable)
    pub fn rotator_mut(&mut self) -> &mut AmbisonicRotator {
        &mut self.rotator
    }

    /// Process mono source through encoding and decoding
    pub fn process_source(
        &mut self,
        input: &[f32],
        direction: SphericalCoord,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        // Encode to ambisonic
        let mut ambisonic_buffer: Vec<Vec<f32>> =
            vec![vec![0.0; input.len()]; self.order.channel_count()];
        self.encoder
            .encode(input, direction, &mut ambisonic_buffer)?;

        // Decode to speakers
        self.decoder.decode(&ambisonic_buffer, output)?;

        Ok(())
    }

    /// Process with rotation
    pub fn process_source_with_rotation(
        &mut self,
        input: &[f32],
        direction: SphericalCoord,
        rotation: RotationAngles,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        // Encode to ambisonic
        let mut ambisonic_buffer: Vec<Vec<f32>> =
            vec![vec![0.0; input.len()]; self.order.channel_count()];
        self.encoder
            .encode(input, direction, &mut ambisonic_buffer)?;

        // Rotate (first-order only)
        if self.order == AmbisonicOrder::First {
            let mut rotated_buffer = vec![vec![0.0; input.len()]; 4];
            self.rotator.set_rotation(rotation);
            self.rotator
                .rotate_first_order(&ambisonic_buffer, &mut rotated_buffer)?;
            ambisonic_buffer = rotated_buffer;
        }

        // Decode to speakers
        self.decoder.decode(&ambisonic_buffer, output)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ambisonic_order() {
        assert_eq!(AmbisonicOrder::First.channel_count(), 4);
        assert_eq!(AmbisonicOrder::Second.channel_count(), 9);
        assert_eq!(AmbisonicOrder::Third.channel_count(), 16);
    }

    #[test]
    fn test_spherical_coord() {
        let coord = SphericalCoord::new(0.0, 0.0, 1.0);
        let (x, y, z) = coord.to_cartesian();

        assert!(x.abs() < 0.01);
        assert!((y - 1.0).abs() < 0.01);
        assert!(z.abs() < 0.01);
    }

    #[test]
    fn test_acn_index() {
        assert_eq!(acn_index(0, 0), 0); // W
        assert_eq!(acn_index(1, -1), 1); // Y
        assert_eq!(acn_index(1, 0), 2); // Z
        assert_eq!(acn_index(1, 1), 3); // X
    }

    #[test]
    fn test_spherical_harmonic() {
        let y00 = spherical_harmonic_n3d(0, 0, 0.0, 0.0);
        assert!(y00 > 0.0);

        let y11 = spherical_harmonic_n3d(1, 1, 0.0, 0.0);
        assert!(y11 != 0.0);
    }

    #[test]
    fn test_ambisonic_encoder() {
        let encoder = AmbisonicEncoder::new(AmbisonicOrder::First);
        let direction = SphericalCoord::new(0.0, 0.0, 1.0);

        let coefficients = encoder.compute_coefficients(direction);
        assert_eq!(coefficients.len(), 4);
        assert!(coefficients[0] > 0.0); // W should be positive
    }

    #[test]
    fn test_encode_decode() {
        let order = AmbisonicOrder::First;
        let speaker_config = SpeakerConfig::stereo();

        let encoder = AmbisonicEncoder::new(order);
        let decoder = AmbisonicDecoder::new(order, speaker_config);

        let input = vec![1.0; 100];
        let direction = SphericalCoord::new(0.0, 0.0, 1.0);

        let mut ambisonic = vec![vec![0.0; 100]; 4];
        let result = encoder.encode(&input, direction, &mut ambisonic);
        assert!(result.is_ok());

        let mut output = vec![vec![0.0; 100]; 2];
        let result = decoder.decode(&ambisonic, &mut output);
        assert!(result.is_ok());

        assert!(output[0].iter().any(|&x| x != 0.0));
        assert!(output[1].iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_rotation() {
        let mut rotator = AmbisonicRotator::new();
        rotator.set_rotation(RotationAngles::new(PI / 2.0, 0.0, 0.0));

        let input = vec![vec![1.0; 10], vec![1.0; 10], vec![0.0; 10], vec![0.0; 10]];

        let mut output = vec![vec![0.0; 10], vec![0.0; 10], vec![0.0; 10], vec![0.0; 10]];

        let result = rotator.rotate_first_order(&input, &mut output);
        assert!(result.is_ok());

        // W should remain the same
        assert!((output[0][0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ambisonic_processor() {
        let order = AmbisonicOrder::First;
        let speaker_config = SpeakerConfig::quad();
        let mut processor = AmbisonicProcessor::new(order, speaker_config);

        let input = vec![1.0; 100];
        let direction = SphericalCoord::new(0.0, 0.0, 1.0);
        let mut output = vec![vec![0.0; 100]; 4];

        let result = processor.process_source(&input, direction, &mut output);
        assert!(result.is_ok());

        assert!(output.iter().any(|ch| ch.iter().any(|&x| x != 0.0)));
    }
}
