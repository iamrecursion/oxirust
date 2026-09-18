//! Multi-channel to binaural downmix via HRTF rendering.
//!
//! The `binauralizer` module converts any multi-channel audio format (stereo,
//! 5.1, 7.1, 7.1.4, Ambisonics, etc.) into a two-channel binaural output
//! suitable for headphone playback.
//!
//! # Algorithm
//!
//! Each input channel is associated with a loudspeaker direction in world space.
//! The binauralizer renders each channel through a pair of HRTFs (left ear, right
//! ear) at the corresponding speaker direction and sums the results:
//!
//! ```text
//! y_L[n] = Σ_i ( x_i[n] * h_L(az_i, el_i) )
//! y_R[n] = Σ_i ( x_i[n] * h_R(az_i, el_i) )
//! ```
//!
//! where `h_{L/R}(az, el)` are the left/right HRTF impulse responses at the
//! speaker azimuth and elevation, and `*` denotes convolution.
//!
//! # Supported Layouts
//!
//! - Stereo (2.0)
//! - 5.1 Surround
//! - 7.1 Surround
//! - Dolby Atmos 7.1.4
//! - First-order Ambisonics (4 channels, decoded via virtual speaker ring)
//! - Custom arbitrary layout

use crate::{
    binaural::{convolve, HrtfDatabase},
    SpatialError,
};

// ─── Types ────────────────────────────────────────────────────────────────────

/// A speaker direction for a single channel of the binauralizer.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerDirection {
    /// Azimuth in degrees (convention: 0 = front, 90 = left, 270 = right).
    pub azimuth_deg: f32,
    /// Elevation in degrees (0 = horizontal, +90 = above).
    pub elevation_deg: f32,
}

impl SpeakerDirection {
    pub fn new(azimuth_deg: f32, elevation_deg: f32) -> Self {
        Self {
            azimuth_deg,
            elevation_deg,
        }
    }
}

/// Predefined loudspeaker layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Two-channel stereo (L at 30°, R at 330°).
    Stereo,
    /// 5.1 surround (L, R, C, LFE, Ls, Rs).
    FiveOnOne,
    /// 7.1 surround (L, R, C, LFE, Ls, Rs, Lss, Rss).
    SevenOnOne,
    /// Dolby Atmos 7.1.4 (7.1 + 4 height channels).
    SevenOnFourAtmos,
    /// First-order Ambisonics decoded to 8-speaker virtual ring.
    AmbisonicsFirstOrder,
    /// Mono (single channel, front centre).
    Mono,
}

impl ChannelLayout {
    /// Return the speaker directions for this layout.
    ///
    /// Convention: 0° = front, increasing CCW (same as VBAP/OxiMedia convention).
    pub fn speaker_directions(self) -> Vec<SpeakerDirection> {
        match self {
            Self::Mono => vec![SpeakerDirection::new(0.0, 0.0)],
            Self::Stereo => vec![
                SpeakerDirection::new(30.0, 0.0),
                SpeakerDirection::new(330.0, 0.0),
            ],
            Self::FiveOnOne => vec![
                SpeakerDirection::new(30.0, 0.0),  // L
                SpeakerDirection::new(330.0, 0.0), // R
                SpeakerDirection::new(0.0, 0.0),   // C
                SpeakerDirection::new(0.0, -10.0), // LFE (low-frequency; near front)
                SpeakerDirection::new(110.0, 0.0), // Ls
                SpeakerDirection::new(250.0, 0.0), // Rs
            ],
            Self::SevenOnOne => vec![
                SpeakerDirection::new(30.0, 0.0),  // L
                SpeakerDirection::new(330.0, 0.0), // R
                SpeakerDirection::new(0.0, 0.0),   // C
                SpeakerDirection::new(0.0, -10.0), // LFE
                SpeakerDirection::new(110.0, 0.0), // Ls
                SpeakerDirection::new(250.0, 0.0), // Rs
                SpeakerDirection::new(60.0, 0.0),  // Lss
                SpeakerDirection::new(300.0, 0.0), // Rss
            ],
            Self::SevenOnFourAtmos => {
                let mut dirs = Self::SevenOnOne.speaker_directions();
                dirs.push(SpeakerDirection::new(45.0, 35.0)); // Ltf
                dirs.push(SpeakerDirection::new(315.0, 35.0)); // Rtf
                dirs.push(SpeakerDirection::new(135.0, 35.0)); // Ltr
                dirs.push(SpeakerDirection::new(225.0, 35.0)); // Rtr
                dirs
            }
            Self::AmbisonicsFirstOrder => {
                // Decode HOA first-order to 8 evenly-spaced virtual speakers.
                (0..8)
                    .map(|i| SpeakerDirection::new(i as f32 * 45.0, 0.0))
                    .collect()
            }
        }
    }

    /// Number of channels in this layout.
    pub fn num_channels(self) -> usize {
        self.speaker_directions().len()
    }
}

// ─── Binauralizer ────────────────────────────────────────────────────────────

/// Multi-channel to binaural converter.
///
/// Each input channel is associated with a speaker direction.  The converter
/// renders each channel through HRTFs at the corresponding direction, then sums
/// all rendered outputs into a stereo binaural mix.
#[derive(Debug, Clone)]
pub struct Binauralizer {
    /// HRTF database used for rendering.
    pub db: HrtfDatabase,
    /// Speaker directions (one per channel).
    pub speaker_dirs: Vec<SpeakerDirection>,
    /// Output gain applied after summing all channels.
    pub output_gain: f32,
    /// Per-channel gain weights (e.g., LFE at lower volume).
    pub channel_gains: Vec<f32>,
}

impl Binauralizer {
    /// Create a binauralizer for a standard predefined layout.
    ///
    /// # Parameters
    /// - `db`: HRTF database.
    /// - `layout`: channel layout enum.
    pub fn from_layout(db: HrtfDatabase, layout: ChannelLayout) -> Self {
        let speaker_dirs = layout.speaker_directions();
        let n = speaker_dirs.len();
        Self {
            db,
            speaker_dirs,
            output_gain: 1.0,
            channel_gains: vec![1.0; n],
        }
    }

    /// Create a binauralizer with custom speaker directions.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if `speaker_dirs` is empty.
    pub fn from_custom(
        db: HrtfDatabase,
        speaker_dirs: Vec<SpeakerDirection>,
    ) -> Result<Self, SpatialError> {
        if speaker_dirs.is_empty() {
            return Err(SpatialError::InvalidConfig(
                "Binauralizer requires at least one speaker direction".into(),
            ));
        }
        let n = speaker_dirs.len();
        Ok(Self {
            db,
            speaker_dirs,
            output_gain: 1.0,
            channel_gains: vec![1.0; n],
        })
    }

    /// Render multi-channel audio to binaural.
    ///
    /// # Parameters
    /// - `channels`: slice of mono audio buffers, one per speaker channel.
    ///   All buffers must have the same length.
    ///
    /// # Returns
    /// `(left_ear, right_ear)` — two output buffers of length
    /// `input_length + ir_length - 1` (linear convolution).
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if the number of channels does
    /// not match the number of speaker directions, or if the buffers have
    /// different lengths.
    pub fn render(&self, channels: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>), SpatialError> {
        if channels.len() != self.speaker_dirs.len() {
            return Err(SpatialError::InvalidConfig(format!(
                "Channel count mismatch: got {} channels, expected {}",
                channels.len(),
                self.speaker_dirs.len()
            )));
        }

        let input_len = match channels.first() {
            Some(ch) => ch.len(),
            None => return Ok((Vec::new(), Vec::new())),
        };

        for (i, ch) in channels.iter().enumerate() {
            if ch.len() != input_len {
                return Err(SpatialError::InvalidConfig(format!(
                    "Channel {i} has length {}, expected {input_len}",
                    ch.len()
                )));
            }
        }

        let out_len = input_len + self.db.ir_length - 1;
        let mut left_out = vec![0.0_f32; out_len];
        let mut right_out = vec![0.0_f32; out_len];

        for ((ch, dir), &ch_gain) in channels
            .iter()
            .zip(self.speaker_dirs.iter())
            .zip(self.channel_gains.iter())
        {
            if ch_gain.abs() < 1e-10 {
                continue; // Skip silent channels.
            }

            let hrtf = self
                .db
                .nearest_measurement(dir.azimuth_deg, dir.elevation_deg);

            let left_conv = convolve(ch, &hrtf.left_ir);
            let right_conv = convolve(ch, &hrtf.right_ir);

            let effective_gain = ch_gain * self.output_gain;
            for (i, (l, r)) in left_conv.iter().zip(right_conv.iter()).enumerate() {
                if i >= out_len {
                    break;
                }
                left_out[i] += l * effective_gain;
                right_out[i] += r * effective_gain;
            }
        }

        Ok((left_out, right_out))
    }

    /// Render a single mono signal at a given direction.
    ///
    /// Convenience wrapper when there is only one audio object.
    pub fn render_mono_at(
        &self,
        samples: &[f32],
        azimuth_deg: f32,
        elevation_deg: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let hrtf = self.db.nearest_measurement(azimuth_deg, elevation_deg);
        let left = convolve(samples, &hrtf.left_ir);
        let right = convolve(samples, &hrtf.right_ir);
        (left, right)
    }

    /// Apply a per-channel gain (e.g., lower LFE channel).
    pub fn set_channel_gain(&mut self, channel: usize, gain: f32) {
        if channel < self.channel_gains.len() {
            self.channel_gains[channel] = gain;
        }
    }

    /// Number of channels this binauralizer expects.
    pub fn num_channels(&self) -> usize {
        self.speaker_dirs.len()
    }

    /// IR length in samples.
    pub fn ir_length(&self) -> usize {
        self.db.ir_length
    }
}

// ─── Ambisonics first-order binaural decode ───────────────────────────────────

/// Decode first-order Ambisonics (4 channels: W, Y, Z, X) to binaural.
///
/// The decode is performed by:
/// 1. Distributing the W/Y/Z/X channels across 8 virtual loudspeakers at
///    0°, 45°, …, 315° azimuth using the standard 1st-order decoding matrix.
/// 2. Rendering each virtual speaker through the HRTF to produce binaural output.
///
/// # Parameters
/// - `ambi_channels`: slice of 4 HOA channel buffers `[W, Y, Z, X]` in ACN order.
/// - `db`: HRTF database.
///
/// # Returns
/// `(left_ear, right_ear)`.
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if the input does not have exactly 4 channels.
pub fn decode_foa_to_binaural(
    ambi_channels: &[Vec<f32>],
    db: &HrtfDatabase,
) -> Result<(Vec<f32>, Vec<f32>), SpatialError> {
    if ambi_channels.len() != 4 {
        return Err(SpatialError::InvalidConfig(format!(
            "FOA decode requires 4 channels (W, Y, Z, X), got {}",
            ambi_channels.len()
        )));
    }

    let input_len = ambi_channels[0].len();
    for (i, ch) in ambi_channels.iter().enumerate() {
        if ch.len() != input_len {
            return Err(SpatialError::InvalidConfig(format!(
                "Channel {i} length {} does not match channel 0 length {input_len}",
                ch.len()
            )));
        }
    }

    // 8 virtual speakers at 0°, 45°, … 315° on the horizontal plane.
    let n_virt = 8_usize;
    let decode_gain = 1.0 / n_virt as f32; // equal-loudness normalisation

    let out_len = input_len + db.ir_length - 1;
    let mut left_out = vec![0.0_f32; out_len];
    let mut right_out = vec![0.0_f32; out_len];

    for v in 0..n_virt {
        let az_deg = v as f32 * 45.0;
        let az_rad = az_deg.to_radians();

        // 1st-order decode coefficients for this virtual speaker:
        // W: 1/sqrt(2),  Y: sin(az),  Z: 0 (horizontal only),  X: cos(az)
        let w_gain = std::f32::consts::FRAC_1_SQRT_2;
        let y_gain = az_rad.sin();
        let x_gain = az_rad.cos();

        // Virtual speaker signal = Σ decode_coeff * channel.
        let virt_sig: Vec<f32> = (0..input_len)
            .map(|n| {
                let w = ambi_channels[0][n];
                let y = ambi_channels[1][n];
                let x = ambi_channels[3][n];
                (w * w_gain + y * y_gain + x * x_gain) * decode_gain
            })
            .collect();

        let hrtf = db.nearest_measurement(az_deg, 0.0);
        let left_conv = convolve(&virt_sig, &hrtf.left_ir);
        let right_conv = convolve(&virt_sig, &hrtf.right_ir);

        for (i, (&l, &r)) in left_conv.iter().zip(right_conv.iter()).enumerate() {
            if i >= out_len {
                break;
            }
            left_out[i] += l;
            right_out[i] += r;
        }
    }

    Ok((left_out, right_out))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binaural::HrtfDatabase;

    fn make_db() -> HrtfDatabase {
        HrtfDatabase::synthetic()
    }

    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
    }

    fn sine(n: usize, hz: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 48_000.0).sin())
            .collect()
    }

    // ── ChannelLayout ────────────────────────────────────────────────────────

    #[test]
    fn test_stereo_layout_two_speakers() {
        let dirs = ChannelLayout::Stereo.speaker_directions();
        assert_eq!(dirs.len(), 2);
    }

    #[test]
    fn test_five_one_layout_six_speakers() {
        assert_eq!(ChannelLayout::FiveOnOne.num_channels(), 6);
    }

    #[test]
    fn test_seven_one_layout_eight_speakers() {
        assert_eq!(ChannelLayout::SevenOnOne.num_channels(), 8);
    }

    #[test]
    fn test_atmos_layout_twelve_speakers() {
        assert_eq!(ChannelLayout::SevenOnFourAtmos.num_channels(), 12);
    }

    #[test]
    fn test_ambisonics_layout_eight_virtual_speakers() {
        assert_eq!(ChannelLayout::AmbisonicsFirstOrder.num_channels(), 8);
    }

    #[test]
    fn test_mono_layout_one_speaker() {
        assert_eq!(ChannelLayout::Mono.num_channels(), 1);
    }

    // ── Binauralizer construction ────────────────────────────────────────────

    #[test]
    fn test_from_layout_stereo() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::Stereo);
        assert_eq!(b.num_channels(), 2);
    }

    #[test]
    fn test_from_custom_empty_fails() {
        let result = Binauralizer::from_custom(make_db(), vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_custom_valid() {
        let dirs = vec![SpeakerDirection::new(45.0, 0.0)];
        let b = Binauralizer::from_custom(make_db(), dirs).expect("ok");
        assert_eq!(b.num_channels(), 1);
    }

    // ── Binauralizer::render ─────────────────────────────────────────────────

    #[test]
    fn test_render_stereo_output_lengths() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::Stereo);
        let channels = vec![sine(256, 440.0), sine(256, 660.0)];
        let (l, r) = b.render(&channels).expect("render ok");
        assert_eq!(l.len(), 256 + b.ir_length() - 1);
        assert_eq!(r.len(), 256 + b.ir_length() - 1);
    }

    #[test]
    fn test_render_stereo_has_energy() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::Stereo);
        let channels = vec![sine(512, 440.0), sine(512, 660.0)];
        let (l, r) = b.render(&channels).expect("render ok");
        assert!(rms(&l) > 0.0, "Left output should have energy");
        assert!(rms(&r) > 0.0, "Right output should have energy");
    }

    #[test]
    fn test_render_channel_count_mismatch_fails() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::Stereo);
        let channels = vec![sine(256, 440.0)]; // only 1 channel for a 2-ch layout
        assert!(b.render(&channels).is_err());
    }

    #[test]
    fn test_render_unequal_lengths_fails() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::Stereo);
        let channels = vec![sine(256, 440.0), sine(128, 660.0)];
        assert!(b.render(&channels).is_err());
    }

    #[test]
    fn test_render_five_one() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::FiveOnOne);
        let channels: Vec<Vec<f32>> = (0..6)
            .map(|i| sine(256, 200.0 + i as f32 * 100.0))
            .collect();
        let (l, r) = b.render(&channels).expect("5.1 render ok");
        assert!(rms(&l) > 0.0);
        assert!(rms(&r) > 0.0);
    }

    #[test]
    fn test_render_output_finite() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::FiveOnOne);
        let channels: Vec<Vec<f32>> = (0..6).map(|_| sine(256, 440.0)).collect();
        let (l, r) = b.render(&channels).expect("ok");
        for &s in l.iter().chain(r.iter()) {
            assert!(s.is_finite(), "Output must be finite, got {s}");
        }
    }

    #[test]
    fn test_set_channel_gain() {
        let mut b = Binauralizer::from_layout(make_db(), ChannelLayout::Stereo);
        b.set_channel_gain(0, 0.0); // mute left channel
        let channels = vec![vec![1.0_f32; 64], vec![0.0_f32; 64]];
        let (l, r) = b.render(&channels).expect("ok");
        // Muted left channel → both outputs should be near zero.
        assert!(
            rms(&l) < 0.1,
            "Muted channel should be silent: rms={}",
            rms(&l)
        );
        assert!(
            rms(&r) < 0.1,
            "Muted channel should give silent right ear: rms={}",
            rms(&r)
        );
    }

    #[test]
    fn test_render_mono_at_has_energy() {
        let b = Binauralizer::from_layout(make_db(), ChannelLayout::Mono);
        let sig = sine(256, 440.0);
        let (l, r) = b.render_mono_at(&sig, 45.0, 0.0);
        assert!(rms(&l) > 0.0);
        assert!(rms(&r) > 0.0);
    }

    // ── decode_foa_to_binaural ───────────────────────────────────────────────

    #[test]
    fn test_foa_decode_valid_four_channels() {
        let db = make_db();
        let channels: Vec<Vec<f32>> = (0..4).map(|i| sine(256, 100.0 + i as f32 * 50.0)).collect();
        let result = decode_foa_to_binaural(&channels, &db);
        assert!(result.is_ok(), "FOA decode should succeed");
        let (l, r) = result.expect("FOA decode result should be valid");
        assert!(!l.is_empty());
        assert!(!r.is_empty());
    }

    #[test]
    fn test_foa_decode_wrong_channel_count_fails() {
        let db = make_db();
        let channels: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0_f32; 64]).collect();
        assert!(decode_foa_to_binaural(&channels, &db).is_err());
    }

    #[test]
    fn test_foa_decode_output_has_energy() {
        let db = make_db();
        let channels: Vec<Vec<f32>> = (0..4).map(|_| sine(512, 440.0)).collect();
        let (l, r) = decode_foa_to_binaural(&channels, &db).expect("ok");
        assert!(rms(&l) > 0.0, "FOA left should have energy");
        assert!(rms(&r) > 0.0, "FOA right should have energy");
    }

    #[test]
    fn test_foa_decode_output_finite() {
        let db = make_db();
        let channels: Vec<Vec<f32>> = (0..4).map(|i| sine(256, 100.0 + i as f32 * 30.0)).collect();
        let (l, r) = decode_foa_to_binaural(&channels, &db).expect("ok");
        for &s in l.iter().chain(r.iter()) {
            assert!(s.is_finite(), "Output must be finite, got {s}");
        }
    }

    #[test]
    fn test_foa_decode_silent_input_gives_silent_output() {
        let db = make_db();
        let channels: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0_f32; 64]).collect();
        let (l, r) = decode_foa_to_binaural(&channels, &db).expect("ok");
        assert_eq!(rms(&l), 0.0, "Silent input should give silent output");
        assert_eq!(rms(&r), 0.0);
    }
}
