//! Channel folding: stereo-to-mono and mono-to-stereo conversion.
//!
//! The [`ChannelFolder`](crate::channel_folder::ChannelFolder) provides conversion between channel formats as
//! a processing chain option:
//!
//! - **Stereo to mono**: Sum left and right channels, divide by 2 to prevent
//!   clipping.
//! - **Mono to stereo**: Duplicate the mono signal to both channels.
//!
//! Additional modes support mid/side encoding and asymmetric fold-down.

// ---------------------------------------------------------------------------
// FoldMode
// ---------------------------------------------------------------------------

/// Defines how channels are folded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldMode {
    /// Sum L + R, divide by 2.
    StereoToMono,
    /// Duplicate mono to both L and R.
    MonoToStereo,
    /// Mid-side encoding: mid = (L+R)/2, side = (L-R)/2.
    StereoToMidSide,
    /// Mid-side decoding: L = mid + side, R = mid - side.
    MidSideToStereo,
}

// ---------------------------------------------------------------------------
// ChannelFolder
// ---------------------------------------------------------------------------

/// Channel folder for converting between mono and stereo (and mid/side).
#[derive(Debug, Clone)]
pub struct ChannelFolder {
    /// The folding mode.
    mode: FoldMode,
    /// Optional gain compensation after folding.
    gain: f32,
}

impl ChannelFolder {
    /// Create a new channel folder with the given mode.
    #[must_use]
    pub fn new(mode: FoldMode) -> Self {
        Self { mode, gain: 1.0 }
    }

    /// Set the post-fold gain compensation.
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(0.0, 4.0);
    }

    /// Get the current gain.
    #[must_use]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Get the current mode.
    #[must_use]
    pub fn mode(&self) -> FoldMode {
        self.mode
    }

    /// Set the folding mode.
    pub fn set_mode(&mut self, mode: FoldMode) {
        self.mode = mode;
    }

    /// Fold stereo to mono: output is `(L + R) / 2 * gain`.
    ///
    /// Returns a mono buffer.
    #[must_use]
    pub fn stereo_to_mono(&self, left: &[f32], right: &[f32]) -> Vec<f32> {
        let n = left.len().min(right.len());
        let mut mono = Vec::with_capacity(n);
        for i in 0..n {
            mono.push((left[i] + right[i]) * 0.5 * self.gain);
        }
        mono
    }

    /// Fold mono to stereo: duplicate mono to both channels.
    ///
    /// Returns `(left, right)` where both are copies scaled by `gain`.
    #[must_use]
    pub fn mono_to_stereo(&self, mono: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let left: Vec<f32> = mono.iter().map(|&s| s * self.gain).collect();
        let right = left.clone();
        (left, right)
    }

    /// Encode stereo to mid/side.
    ///
    /// Returns `(mid, side)` where `mid = (L+R)/2` and `side = (L-R)/2`.
    #[must_use]
    pub fn stereo_to_mid_side(&self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = left.len().min(right.len());
        let mut mid = Vec::with_capacity(n);
        let mut side = Vec::with_capacity(n);
        for i in 0..n {
            mid.push((left[i] + right[i]) * 0.5 * self.gain);
            side.push((left[i] - right[i]) * 0.5 * self.gain);
        }
        (mid, side)
    }

    /// Decode mid/side to stereo.
    ///
    /// Returns `(left, right)` where `L = mid + side` and `R = mid - side`.
    #[must_use]
    pub fn mid_side_to_stereo(&self, mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = mid.len().min(side.len());
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            left.push((mid[i] + side[i]) * self.gain);
            right.push((mid[i] - side[i]) * self.gain);
        }
        (left, right)
    }

    /// Process buffers according to the current mode.
    ///
    /// This is a convenience method that dispatches to the appropriate
    /// fold function.
    ///
    /// # Arguments
    /// * `input_a` — First input (left for stereo, mono for mono, mid for M/S).
    /// * `input_b` — Second input (right for stereo, unused/empty for mono, side for M/S).
    ///
    /// # Returns
    /// `(output_a, output_b)` — The folded output buffers.
    #[must_use]
    pub fn process(&self, input_a: &[f32], input_b: &[f32]) -> (Vec<f32>, Vec<f32>) {
        match self.mode {
            FoldMode::StereoToMono => {
                let mono = self.stereo_to_mono(input_a, input_b);
                (mono, Vec::new())
            }
            FoldMode::MonoToStereo => self.mono_to_stereo(input_a),
            FoldMode::StereoToMidSide => self.stereo_to_mid_side(input_a, input_b),
            FoldMode::MidSideToStereo => self.mid_side_to_stereo(input_a, input_b),
        }
    }

    /// Process buffers in-place where possible.
    ///
    /// For `StereoToMono`: writes mono to `left`, clears `right`.
    /// For `MonoToStereo`: copies `left` to `right`.
    /// For M/S: transforms in-place.
    pub fn process_in_place(&self, left: &mut [f32], right: &mut [f32]) {
        let n = left.len().min(right.len());
        match self.mode {
            FoldMode::StereoToMono => {
                for i in 0..n {
                    left[i] = (left[i] + right[i]) * 0.5 * self.gain;
                    right[i] = 0.0;
                }
            }
            FoldMode::MonoToStereo => {
                for i in 0..n {
                    let mono = left[i] * self.gain;
                    left[i] = mono;
                    right[i] = mono;
                }
            }
            FoldMode::StereoToMidSide => {
                for i in 0..n {
                    let l = left[i];
                    let r = right[i];
                    left[i] = (l + r) * 0.5 * self.gain; // mid
                    right[i] = (l - r) * 0.5 * self.gain; // side
                }
            }
            FoldMode::MidSideToStereo => {
                for i in 0..n {
                    let m = left[i];
                    let s = right[i];
                    left[i] = (m + s) * self.gain; // L
                    right[i] = (m - s) * self.gain; // R
                }
            }
        }
    }
}

impl Default for ChannelFolder {
    fn default() -> Self {
        Self::new(FoldMode::StereoToMono)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_to_mono() {
        let folder = ChannelFolder::new(FoldMode::StereoToMono);
        let left = vec![1.0_f32, 0.5, -0.5];
        let right = vec![0.0_f32, 0.5, 0.5];
        let mono = folder.stereo_to_mono(&left, &right);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6); // (1.0 + 0.0) / 2
        assert!((mono[1] - 0.5).abs() < 1e-6); // (0.5 + 0.5) / 2
        assert!((mono[2] - 0.0).abs() < 1e-6); // (-0.5 + 0.5) / 2
    }

    #[test]
    fn test_mono_to_stereo() {
        let folder = ChannelFolder::new(FoldMode::MonoToStereo);
        let mono = vec![0.5_f32, -0.3, 0.8];
        let (left, right) = folder.mono_to_stereo(&mono);
        assert_eq!(left.len(), 3);
        assert_eq!(right.len(), 3);
        for i in 0..3 {
            assert!(
                (left[i] - right[i]).abs() < f32::EPSILON,
                "L and R should be equal"
            );
            assert!(
                (left[i] - mono[i]).abs() < f32::EPSILON,
                "should match input"
            );
        }
    }

    #[test]
    fn test_stereo_to_mid_side() {
        let folder = ChannelFolder::new(FoldMode::StereoToMidSide);
        let left = vec![1.0_f32, 0.0];
        let right = vec![1.0_f32, 0.0];
        let (mid, side) = folder.stereo_to_mid_side(&left, &right);
        // Identical L/R: mid = 1.0, side = 0.0
        assert!((mid[0] - 1.0).abs() < 1e-6);
        assert!(side[0].abs() < 1e-6);
    }

    #[test]
    fn test_mid_side_to_stereo() {
        let folder = ChannelFolder::new(FoldMode::MidSideToStereo);
        let mid = vec![0.5_f32];
        let side = vec![0.25_f32];
        let (left, right) = folder.mid_side_to_stereo(&mid, &side);
        // L = mid + side = 0.75
        // R = mid - side = 0.25
        assert!((left[0] - 0.75).abs() < 1e-6);
        assert!((right[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_mid_side_roundtrip() {
        let folder_encode = ChannelFolder::new(FoldMode::StereoToMidSide);
        let folder_decode = ChannelFolder::new(FoldMode::MidSideToStereo);

        let left = vec![0.8_f32, 0.3, -0.5];
        let right = vec![0.2_f32, 0.7, 0.1];

        let (mid, side) = folder_encode.stereo_to_mid_side(&left, &right);
        let (l_out, r_out) = folder_decode.mid_side_to_stereo(&mid, &side);

        for i in 0..3 {
            assert!(
                (l_out[i] - left[i]).abs() < 1e-5,
                "L roundtrip failed at {i}"
            );
            assert!(
                (r_out[i] - right[i]).abs() < 1e-5,
                "R roundtrip failed at {i}"
            );
        }
    }

    #[test]
    fn test_gain_compensation() {
        let mut folder = ChannelFolder::new(FoldMode::StereoToMono);
        folder.set_gain(2.0);
        let left = vec![0.5_f32];
        let right = vec![0.5_f32];
        let mono = folder.stereo_to_mono(&left, &right);
        // (0.5 + 0.5) / 2 * 2.0 = 1.0
        assert!((mono[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_dispatch_stereo_to_mono() {
        let folder = ChannelFolder::new(FoldMode::StereoToMono);
        let (out_a, out_b) = folder.process(&[1.0, 0.0], &[0.0, 1.0]);
        assert_eq!(out_a.len(), 2);
        assert!(out_b.is_empty());
        assert!((out_a[0] - 0.5).abs() < 1e-6);
        assert!((out_a[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_process_dispatch_mono_to_stereo() {
        let folder = ChannelFolder::new(FoldMode::MonoToStereo);
        let (out_a, out_b) = folder.process(&[0.7], &[]);
        assert_eq!(out_a.len(), 1);
        assert_eq!(out_b.len(), 1);
        assert!((out_a[0] - 0.7).abs() < 1e-6);
        assert!((out_b[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_process_in_place_stereo_to_mono() {
        let folder = ChannelFolder::new(FoldMode::StereoToMono);
        let mut left = vec![1.0_f32, 0.5];
        let mut right = vec![0.0_f32, 0.5];
        folder.process_in_place(&mut left, &mut right);
        assert!((left[0] - 0.5).abs() < 1e-6);
        assert!((left[1] - 0.5).abs() < 1e-6);
        assert!(right[0].abs() < 1e-6);
        assert!(right[1].abs() < 1e-6);
    }

    #[test]
    fn test_process_in_place_mono_to_stereo() {
        let folder = ChannelFolder::new(FoldMode::MonoToStereo);
        let mut left = vec![0.3_f32, 0.7];
        let mut right = vec![0.0_f32, 0.0];
        folder.process_in_place(&mut left, &mut right);
        for i in 0..2 {
            assert!((left[i] - right[i]).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_process_in_place_mid_side_roundtrip() {
        let mut left = vec![0.8_f32, 0.3];
        let mut right = vec![0.2_f32, 0.7];
        let orig_l = left.clone();
        let orig_r = right.clone();

        let encode = ChannelFolder::new(FoldMode::StereoToMidSide);
        encode.process_in_place(&mut left, &mut right);

        let decode = ChannelFolder::new(FoldMode::MidSideToStereo);
        decode.process_in_place(&mut left, &mut right);

        for i in 0..2 {
            assert!((left[i] - orig_l[i]).abs() < 1e-5);
            assert!((right[i] - orig_r[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_different_length_inputs() {
        let folder = ChannelFolder::new(FoldMode::StereoToMono);
        let left = vec![1.0_f32; 10];
        let right = vec![0.0_f32; 5]; // Shorter
        let mono = folder.stereo_to_mono(&left, &right);
        assert_eq!(mono.len(), 5); // Uses min length
    }

    #[test]
    fn test_empty_inputs() {
        let folder = ChannelFolder::new(FoldMode::StereoToMono);
        let mono = folder.stereo_to_mono(&[], &[]);
        assert!(mono.is_empty());
    }

    #[test]
    fn test_gain_clamping() {
        let mut folder = ChannelFolder::new(FoldMode::StereoToMono);
        folder.set_gain(10.0);
        assert!(folder.gain() <= 4.0);
        folder.set_gain(-1.0);
        assert!(folder.gain() >= 0.0);
    }

    #[test]
    fn test_default() {
        let folder = ChannelFolder::default();
        assert_eq!(folder.mode(), FoldMode::StereoToMono);
        assert!((folder.gain() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_mode() {
        let mut folder = ChannelFolder::new(FoldMode::StereoToMono);
        folder.set_mode(FoldMode::MonoToStereo);
        assert_eq!(folder.mode(), FoldMode::MonoToStereo);
    }
}
