//! Audio routing matrix with panning laws and channel strip support.
//!
//! This module extends the basic routing found in `routing.rs` with panning
//! utilities and a full channel-strip model.

/// A flexible gain-based routing matrix (sources × destinations).
///
/// `gains[src * destinations + dst]` holds the linear gain for that connection.
#[derive(Debug, Clone)]
pub struct RoutingMatrix {
    /// Number of source channels.
    pub sources: usize,
    /// Number of destination channels.
    pub destinations: usize,
    gains: Vec<f32>,
}

impl RoutingMatrix {
    /// Create a new routing matrix initialised to silence.
    #[must_use]
    pub fn new(sources: usize, destinations: usize) -> Self {
        Self {
            sources,
            destinations,
            gains: vec![0.0_f32; sources * destinations],
        }
    }

    /// Set gain for the connection from `src` to `dst`.
    ///
    /// Out-of-range indices are silently ignored.
    pub fn set_gain(&mut self, src: usize, dst: usize, gain: f32) {
        if src < self.sources && dst < self.destinations {
            self.gains[src * self.destinations + dst] = gain;
        }
    }

    /// Get gain for the connection from `src` to `dst`.
    ///
    /// Returns `0.0` for out-of-range indices.
    #[must_use]
    pub fn get_gain(&self, src: usize, dst: usize) -> f32 {
        if src < self.sources && dst < self.destinations {
            self.gains[src * self.destinations + dst]
        } else {
            0.0
        }
    }

    /// Apply the routing matrix to a set of input buffers.
    ///
    /// Each element of `inputs` is the sample buffer for one source channel.
    /// Returns `destinations` output buffers, each with the same length as the
    /// shortest input buffer.
    #[must_use]
    pub fn route(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if inputs.is_empty() || self.destinations == 0 {
            return vec![Vec::new(); self.destinations];
        }
        let buf_len = inputs.iter().map(Vec::len).min().unwrap_or(0);
        let mut outputs = vec![vec![0.0_f32; buf_len]; self.destinations];

        for (src, in_buf) in inputs.iter().enumerate() {
            if src >= self.sources {
                break;
            }
            for (dst, out_buf) in outputs.iter_mut().enumerate() {
                let gain = self.gains[src * self.destinations + dst];
                if gain.abs() < 1e-10 {
                    continue;
                }
                for (o, &s) in out_buf.iter_mut().zip(in_buf.iter()) {
                    *o += s * gain;
                }
            }
        }
        outputs
    }
}

// ---------------------------------------------------------------------------
// Stereo Router / Pan Laws
// ---------------------------------------------------------------------------

/// Stereo routing utilities with multiple panning laws.
pub struct StereoRouter;

impl StereoRouter {
    /// Linear panning law.
    ///
    /// `pan` ranges from -1.0 (hard left) to +1.0 (hard right).
    /// Returns `(left_gain, right_gain)`.
    #[must_use]
    pub fn pan_law_linear(pan: f32) -> (f32, f32) {
        let p = pan.clamp(-1.0, 1.0);
        let right = (p + 1.0) / 2.0;
        let left = 1.0 - right;
        (left, right)
    }

    /// Square-root ("equal-power") panning law.
    ///
    /// Provides 3 dB boost at centre.
    #[must_use]
    pub fn pan_law_sqrt(pan: f32) -> (f32, f32) {
        let p = pan.clamp(-1.0, 1.0);
        let right_linear = (p + 1.0) / 2.0;
        let left_linear = 1.0 - right_linear;
        (left_linear.sqrt(), right_linear.sqrt())
    }

    /// Sine/cosine panning law (true equal-power).
    ///
    /// Uses a quarter-circle mapping for perceptually uniform loudness.
    #[must_use]
    pub fn pan_law_sine_cosine(pan: f32) -> (f32, f32) {
        use std::f32::consts::FRAC_PI_2;
        let p = pan.clamp(-1.0, 1.0);
        // Map [-1, 1] → [0, π/2]
        let angle = (p + 1.0) / 2.0 * FRAC_PI_2;
        (angle.cos(), angle.sin())
    }
}

// ---------------------------------------------------------------------------
// Channel Strip
// ---------------------------------------------------------------------------

/// A single channel strip with volume, pan, mute, and solo controls.
#[derive(Debug, Clone)]
pub struct ChannelStrip {
    /// Fader level (linear, 0.0 = -∞ dB, 1.0 = 0 dB).
    pub volume: f32,
    /// Pan position: -1.0 = hard left, 0.0 = centre, +1.0 = hard right.
    pub pan: f32,
    /// If `true` the channel produces no output.
    pub muted: bool,
    /// If `true` this channel is soloed.
    pub solo: bool,
}

impl Default for ChannelStrip {
    fn default() -> Self {
        Self {
            volume: 1.0,
            pan: 0.0,
            muted: false,
            solo: false,
        }
    }
}

impl ChannelStrip {
    /// Create a new channel strip with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the final `(left_gain, right_gain)` pair for this strip.
    ///
    /// `any_solo` should be `true` when *any* channel in the mix is soloed.
    /// Strips that are not soloed are silenced whenever another strip is.
    #[must_use]
    pub fn output_gain(&self, any_solo: bool) -> (f32, f32) {
        // Mute or solo-isolation silences the channel
        if self.muted || (any_solo && !self.solo) {
            return (0.0, 0.0);
        }

        let (left_pan, right_pan) = StereoRouter::pan_law_sqrt(self.pan);
        (self.volume * left_pan, self.volume * right_pan)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_matrix_new_is_silent() {
        let m = RoutingMatrix::new(4, 2);
        for src in 0..4 {
            for dst in 0..2 {
                assert_eq!(m.get_gain(src, dst), 0.0);
            }
        }
    }

    #[test]
    fn test_routing_matrix_set_and_get() {
        let mut m = RoutingMatrix::new(3, 3);
        m.set_gain(1, 2, 0.75);
        assert!((m.get_gain(1, 2) - 0.75).abs() < f32::EPSILON);
        assert_eq!(m.get_gain(0, 0), 0.0);
    }

    #[test]
    fn test_routing_matrix_out_of_range() {
        let mut m = RoutingMatrix::new(2, 2);
        m.set_gain(5, 5, 1.0); // should silently do nothing
        assert_eq!(m.get_gain(5, 5), 0.0);
    }

    #[test]
    fn test_routing_matrix_route_passthrough() {
        let mut m = RoutingMatrix::new(2, 2);
        m.set_gain(0, 0, 1.0);
        m.set_gain(1, 1, 1.0);
        let inputs = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let out = m.route(&inputs);
        assert_eq!(out.len(), 2);
        assert!((out[0][0] - 1.0).abs() < 1e-6);
        assert!((out[0][1] - 2.0).abs() < 1e-6);
        assert!((out[1][0] - 3.0).abs() < 1e-6);
        assert!((out[1][1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_routing_matrix_route_empty_inputs() {
        let m = RoutingMatrix::new(2, 2);
        let out = m.route(&[]);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_empty());
    }

    #[test]
    fn test_pan_law_linear_center() {
        let (l, r) = StereoRouter::pan_law_linear(0.0);
        assert!((l - 0.5).abs() < 1e-6);
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pan_law_linear_hard_left() {
        let (l, r) = StereoRouter::pan_law_linear(-1.0);
        assert!((l - 1.0).abs() < 1e-6);
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn test_pan_law_sqrt_energy_preserved() {
        let (l, r) = StereoRouter::pan_law_sqrt(0.0);
        // At centre, l == r and l² + r² == 1.0 (equal power)
        assert!((l * l + r * r - 1.0).abs() < 1e-5, "l={l} r={r}");
    }

    #[test]
    fn test_pan_law_sine_cosine_energy_preserved() {
        for pan in [-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let (l, r) = StereoRouter::pan_law_sine_cosine(pan);
            let energy = l * l + r * r;
            assert!((energy - 1.0).abs() < 1e-5, "pan={pan} energy={energy}");
        }
    }

    #[test]
    fn test_channel_strip_default() {
        let strip = ChannelStrip::default();
        assert_eq!(strip.volume, 1.0);
        assert_eq!(strip.pan, 0.0);
        assert!(!strip.muted);
        assert!(!strip.solo);
    }

    #[test]
    fn test_channel_strip_muted_silence() {
        let strip = ChannelStrip {
            muted: true,
            ..Default::default()
        };
        let (l, r) = strip.output_gain(false);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_channel_strip_solo_isolation() {
        // Strip is NOT soloed, but another strip is → should be silent
        let strip = ChannelStrip {
            solo: false,
            ..Default::default()
        };
        let (l, r) = strip.output_gain(true);
        assert_eq!(l, 0.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_channel_strip_solo_passes_through() {
        let strip = ChannelStrip {
            solo: true,
            ..Default::default()
        };
        let (l, r) = strip.output_gain(true);
        assert!(l > 0.0 && r > 0.0);
    }
}
