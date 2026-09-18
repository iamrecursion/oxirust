//! Schroeder / Freeverb-style algorithmic reverb and direct-convolution reverb.
//!
//! Provides two complementary reverb implementations:
//!
//! - [`SchroederReverb`] — CPU-efficient algorithmic reverb using the classic
//!   Freeverb architecture (8 parallel comb filters → 4 series all-pass filters).
//! - [`SimpleConvolutionReverb`] — Direct convolution reverb that applies an
//!   arbitrary impulse response using overlap-add in 512-sample blocks.

use crate::utils::delay_line::DelayLine;

/// Feedback comb filter with one-pole low-pass damping.
///
/// Core building block of the Schroeder / Freeverb reverb algorithm.
///
/// Internally uses a pre-allocated [`DelayLine`] ring buffer rather than a
/// raw `Vec<f32>` — no allocations occur during audio processing.
struct CombFilter {
    /// Circular delay line (size = comb delay + 1).
    delay_line: DelayLine,
    /// Delay length in samples.
    delay: usize,
    feedback: f32,
    damp: f32,
    filter_store: f32,
}

impl CombFilter {
    fn new(delay_samples: usize) -> Self {
        let size = delay_samples.max(1);
        Self {
            delay_line: DelayLine::new(size + 1),
            delay: size,
            feedback: 0.84,
            damp: 0.2,
            filter_store: 0.0,
        }
    }

    /// Process one input sample through the comb filter.
    ///
    /// Algorithm (Freeverb):
    /// ```text
    /// output       = read(delay)
    /// filter_store = output*(1-damp) + filter_store*damp
    /// write(input + filter_store*feedback)
    /// ```
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        // Read the sample written `delay` steps ago (oldest in the loop).
        let output = self.delay_line.read(self.delay);

        // One-pole low-pass damping on the feedback path.
        self.filter_store = output * (1.0 - self.damp) + self.filter_store * self.damp;

        // Write: input plus the filtered feedback.
        self.delay_line
            .write(input + self.filter_store * self.feedback);

        output
    }

    fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(0.0, 0.98);
    }

    fn set_damp(&mut self, d: f32) {
        self.damp = d.clamp(0.0, 1.0);
    }

    fn clear(&mut self) {
        self.delay_line.clear();
        self.filter_store = 0.0;
    }
}

/// Schroeder all-pass filter.
///
/// Adds diffusion without changing the frequency spectrum.
///
/// Internally uses a pre-allocated [`DelayLine`] ring buffer.
struct AllpassFilter {
    /// Circular delay line (size = allpass delay + 1).
    delay_line: DelayLine,
    /// Delay length in samples.
    delay: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize) -> Self {
        let size = delay_samples.max(1);
        Self {
            delay_line: DelayLine::new(size + 1),
            delay: size,
            feedback: 0.5,
        }
    }

    /// Process one sample through the all-pass filter.
    ///
    /// Algorithm:
    /// ```text
    /// buf_out = read(delay)
    /// write(input + buf_out * feedback)
    /// output  = buf_out - input
    /// ```
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buf_out = self.delay_line.read(self.delay);
        self.delay_line.write(input + buf_out * self.feedback);
        buf_out - input
    }

    fn clear(&mut self) {
        self.delay_line.clear();
    }
}

// ---------------------------------------------------------------------------
// Freeverb standard delay sizes (samples @ 44.1 kHz)
// ---------------------------------------------------------------------------
const COMB_DELAYS_BASE: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
/// Stereo offset (right channel) — 23 samples as in original Freeverb.
const STEREO_SPREAD: usize = 23;
const ALLPASS_DELAYS_BASE: [usize; 4] = [556, 441, 341, 225];

// ---------------------------------------------------------------------------
// SchroederReverb
// ---------------------------------------------------------------------------

/// High-quality algorithmic reverb based on the Freeverb / Schroeder architecture.
///
/// Uses 8 parallel feedback comb filters (per channel) followed by 4 series
/// all-pass filters for diffusion.  Left and right channels share the same
/// room parameters but use slightly different delay lengths for natural stereo.
///
/// # Parameters
///
/// | Field | Range | Description |
/// |-------|-------|-------------|
/// | `room_size` | 0.0–1.0 | Controls comb feedback (larger = longer decay) |
/// | `damping`   | 0.0–1.0 | High-frequency roll-off in comb filters |
/// | `wet_mix`   | 0.0–1.0 | Reverb wet level |
/// | `dry_mix`   | 0.0–1.0 | Direct (dry) level |
/// | `width`     | 0.0–1.0 | Stereo width of the reverb tail |
pub struct SchroederReverb {
    /// Room size (0.0–1.0). Larger values increase decay time.
    pub room_size: f32,
    /// High-frequency damping (0.0–1.0).
    pub damping: f32,
    /// Wet (reverb) mix level.
    pub wet_mix: f32,
    /// Dry (direct) mix level.
    pub dry_mix: f32,
    /// Stereo width of the reverb output.
    pub width: f32,

    // Left-channel DSP
    combs_l: Vec<CombFilter>,
    allpass_l: Vec<AllpassFilter>,

    // Right-channel DSP
    combs_r: Vec<CombFilter>,
    allpass_r: Vec<AllpassFilter>,
}

impl SchroederReverb {
    /// Create a new `SchroederReverb` initialised for `sample_rate`.
    ///
    /// Delay lengths are scaled linearly from the Freeverb 44.1 kHz constants.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f32;
        let scale = sr / 44100.0;

        let make_combs = |delays: &[usize]| -> Vec<CombFilter> {
            delays
                .iter()
                .map(|&d| {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let scaled = ((d as f32) * scale) as usize;
                    CombFilter::new(scaled.max(1))
                })
                .collect()
        };

        let make_allpasses = |delays: &[usize]| -> Vec<AllpassFilter> {
            delays
                .iter()
                .map(|&d| {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let scaled = ((d as f32) * scale) as usize;
                    AllpassFilter::new(scaled.max(1))
                })
                .collect()
        };

        let right_comb_delays: Vec<usize> = COMB_DELAYS_BASE
            .iter()
            .map(|&d| d + STEREO_SPREAD)
            .collect();
        let right_allpass_delays: Vec<usize> = ALLPASS_DELAYS_BASE
            .iter()
            .map(|&d| d + STEREO_SPREAD)
            .collect();

        let mut rv = Self {
            room_size: 0.5,
            damping: 0.5,
            wet_mix: 0.33,
            dry_mix: 0.67,
            width: 1.0,
            combs_l: make_combs(&COMB_DELAYS_BASE),
            allpass_l: make_allpasses(&ALLPASS_DELAYS_BASE),
            combs_r: make_combs(&right_comb_delays),
            allpass_r: make_allpasses(&right_allpass_delays),
        };

        rv.apply_params();
        rv
    }

    // -----------------------------------------------------------------------
    // Parameter setters
    // -----------------------------------------------------------------------

    /// Set room size (0.0–1.0); updates comb feedback immediately.
    pub fn set_room_size(&mut self, size: f32) {
        self.room_size = size.clamp(0.0, 1.0);
        self.apply_params();
    }

    /// Set high-frequency damping (0.0–1.0); updates comb LPF immediately.
    pub fn set_damping(&mut self, damp: f32) {
        self.damping = damp.clamp(0.0, 1.0);
        self.apply_params();
    }

    /// Propagate room_size and damping to all comb filters.
    fn apply_params(&mut self) {
        // Freeverb mapping: feedback = 0.7 + room_size*0.28
        let feedback = 0.7 + self.room_size * 0.28;
        // Freeverb mapping: damp coefficient = damping*0.4
        let damp = self.damping * 0.4;

        for c in &mut self.combs_l {
            c.set_feedback(feedback);
            c.set_damp(damp);
        }
        for c in &mut self.combs_r {
            c.set_feedback(feedback);
            c.set_damp(damp);
        }
    }

    // -----------------------------------------------------------------------
    // Processing
    // -----------------------------------------------------------------------

    /// Process interleaved stereo buffers and return (left_out, right_out).
    ///
    /// Both input slices must be the same length; extra samples in the longer
    /// slice are ignored.
    #[must_use]
    pub fn process_stereo(&mut self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = left.len().min(right.len());
        let mut out_l = Vec::with_capacity(n);
        let mut out_r = Vec::with_capacity(n);

        let wet1 = self.wet_mix * (self.width / 2.0 + 0.5);
        let wet2 = self.wet_mix * ((1.0 - self.width) / 2.0);

        for i in 0..n {
            let in_l = left[i];
            let in_r = right[i];

            // Parallel comb filters
            let comb_l: f32 = self.combs_l.iter_mut().map(|c| c.process(in_l)).sum();
            let comb_r: f32 = self.combs_r.iter_mut().map(|c| c.process(in_r)).sum();

            // Series all-pass diffusers
            let mut diff_l = comb_l;
            for ap in &mut self.allpass_l {
                diff_l = ap.process(diff_l);
            }
            let mut diff_r = comb_r;
            for ap in &mut self.allpass_r {
                diff_r = ap.process(diff_r);
            }

            // Stereo spread + wet/dry mix
            let o_l = diff_l * wet1 + diff_r * wet2 + in_l * self.dry_mix;
            let o_r = diff_r * wet1 + diff_l * wet2 + in_r * self.dry_mix;

            out_l.push(o_l);
            out_r.push(o_r);
        }

        (out_l, out_r)
    }

    /// Process a mono buffer (same input fed to both channels) and return
    /// a mono summed output (mean of left + right reverb tails).
    #[must_use]
    pub fn process_mono(&mut self, input: &[f32]) -> Vec<f32> {
        let (l, r) = self.process_stereo(input, input);
        l.iter().zip(r.iter()).map(|(a, b)| (a + b) * 0.5).collect()
    }

    /// Clear all internal delay-line state.
    pub fn reset(&mut self) {
        for c in &mut self.combs_l {
            c.clear();
        }
        for c in &mut self.combs_r {
            c.clear();
        }
        for ap in &mut self.allpass_l {
            ap.clear();
        }
        for ap in &mut self.allpass_r {
            ap.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// SimpleConvolutionReverb
// ---------------------------------------------------------------------------

const BLOCK_SIZE: usize = 512;

/// Direct convolution reverb using overlap-add in 512-sample blocks.
///
/// Convolves the input signal against a stored impulse response (IR).  For
/// short IRs (≤ a few thousand samples) this is fast enough for offline
/// processing; for real-time use with long IRs consider a partitioned FFT
/// convolver.
///
/// # Example
///
/// ```
/// use oximedia_effects::reverb::schroeder::SimpleConvolutionReverb;
///
/// let ir: Vec<f32> = (0..256).map(|i| 0.99_f32.powi(i as i32)).collect();
/// let reverb = SimpleConvolutionReverb::new(ir);
/// let input: Vec<f32> = vec![1.0; 512];
/// let output = reverb.process(&input);
/// assert_eq!(output.len(), 512);
/// ```
pub struct SimpleConvolutionReverb {
    /// Impulse response.
    pub ir: Vec<f32>,
    /// Wet (convolved) level.
    pub wet_mix: f32,
    /// Dry (direct) level.
    pub dry_mix: f32,
}

impl SimpleConvolutionReverb {
    /// Create a new convolution reverb from an impulse response.
    #[must_use]
    pub fn new(ir: Vec<f32>) -> Self {
        Self {
            ir,
            wet_mix: 0.5,
            dry_mix: 0.5,
        }
    }

    /// Process `input` using overlap-add convolution in 512-sample blocks.
    ///
    /// Returns a `Vec<f32>` of the same length as `input`.
    #[must_use]
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        if n == 0 || self.ir.is_empty() {
            return input.to_vec();
        }

        let ir_len = self.ir.len();
        // Output buffer large enough to hold the full linear convolution.
        let out_len = n + ir_len - 1;
        let mut conv_out = vec![0.0_f32; out_len];

        // Overlap-add: process input in BLOCK_SIZE chunks.
        let mut block_start = 0;
        while block_start < n {
            let block_end = (block_start + BLOCK_SIZE).min(n);
            // Direct convolution of this block with the full IR.
            for (bi, &x) in input[block_start..block_end].iter().enumerate() {
                if x == 0.0 {
                    continue;
                }
                let out_offset = block_start + bi;
                for (ii, &h) in self.ir.iter().enumerate() {
                    conv_out[out_offset + ii] += x * h;
                }
            }
            block_start = block_end;
        }

        // Mix wet + dry, clamp output to input length.
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let wet = conv_out[i];
            let dry = input[i];
            output.push(wet * self.wet_mix + dry * self.dry_mix);
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SchroederReverb tests ----

    #[test]
    fn test_schroeder_new_44100() {
        let rv = SchroederReverb::new(44100);
        assert_eq!(rv.combs_l.len(), 8);
        assert_eq!(rv.allpass_l.len(), 4);
    }

    #[test]
    fn test_schroeder_new_48000() {
        let rv = SchroederReverb::new(48000);
        // Scaled delay lengths should still create valid buffers.
        assert_eq!(rv.combs_l.len(), 8);
        assert_eq!(rv.allpass_l.len(), 4);
    }

    #[test]
    fn test_schroeder_process_stereo_length() {
        let mut rv = SchroederReverb::new(44100);
        let l = vec![0.5_f32; 256];
        let r = vec![-0.5_f32; 256];
        let (ol, or_) = rv.process_stereo(&l, &r);
        assert_eq!(ol.len(), 256);
        assert_eq!(or_.len(), 256);
    }

    #[test]
    fn test_schroeder_output_finite() {
        let mut rv = SchroederReverb::new(44100);
        // Drive with an impulse and let it ring.
        let mut l = vec![0.0_f32; 1024];
        let mut r = vec![0.0_f32; 1024];
        l[0] = 1.0;
        r[0] = 1.0;
        let (ol, or_) = rv.process_stereo(&l, &r);
        for (i, (&a, &b)) in ol.iter().zip(or_.iter()).enumerate() {
            assert!(a.is_finite(), "out_l[{i}] is not finite: {a}");
            assert!(b.is_finite(), "out_r[{i}] is not finite: {b}");
        }
    }

    #[test]
    fn test_schroeder_silence_in_silence_out() {
        let mut rv = SchroederReverb::new(44100);
        // With wet_mix and dry_mix both 0 the output should be zero.
        rv.wet_mix = 0.0;
        rv.dry_mix = 0.0;
        let l = vec![1.0_f32; 64];
        let r = vec![1.0_f32; 64];
        let (ol, or_) = rv.process_stereo(&l, &r);
        for &s in ol.iter().chain(or_.iter()) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_schroeder_set_room_size_clamp() {
        let mut rv = SchroederReverb::new(44100);
        rv.set_room_size(2.5);
        assert_eq!(rv.room_size, 1.0);
        rv.set_room_size(-1.0);
        assert_eq!(rv.room_size, 0.0);
    }

    #[test]
    fn test_schroeder_set_damping_clamp() {
        let mut rv = SchroederReverb::new(44100);
        rv.set_damping(1.5);
        assert_eq!(rv.damping, 1.0);
        rv.set_damping(-0.1);
        assert_eq!(rv.damping, 0.0);
    }

    #[test]
    fn test_schroeder_reset_clears_state() {
        let mut rv = SchroederReverb::new(44100);
        let impulse = {
            let mut v = vec![0.0_f32; 256];
            v[0] = 1.0;
            v
        };
        rv.process_stereo(&impulse, &impulse);
        rv.reset();
        // After reset, processing silence should give (near-) zero output.
        let silence = vec![0.0_f32; 64];
        let (ol, or_) = rv.process_stereo(&silence, &silence);
        for &s in ol.iter().chain(or_.iter()) {
            assert!(s.abs() < 1e-9, "Expected silence after reset, got {s}");
        }
    }

    #[test]
    fn test_schroeder_stereo_channels_differ() {
        let mut rv = SchroederReverb::new(44100);
        rv.width = 1.0;
        let mut l = vec![0.0_f32; 512];
        let r = vec![0.0_f32; 512];
        l[0] = 1.0; // only left impulse
        let (ol, or_) = rv.process_stereo(&l, &r);
        // With stereo spread the two channels should not be identical.
        let identical = ol.iter().zip(or_.iter()).all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(!identical, "Expected stereo channels to differ");
    }

    #[test]
    fn test_schroeder_process_mono() {
        let mut rv = SchroederReverb::new(44100);
        let input = vec![0.5_f32; 128];
        let out = rv.process_mono(&input);
        assert_eq!(out.len(), 128);
        for &s in &out {
            assert!(s.is_finite());
        }
    }

    // ---- SimpleConvolutionReverb tests ----

    #[test]
    fn test_conv_reverb_new() {
        let ir = vec![1.0_f32, 0.5, 0.25];
        let rv = SimpleConvolutionReverb::new(ir.clone());
        assert_eq!(rv.ir, ir);
        assert!((rv.wet_mix - 0.5).abs() < 1e-6);
        assert!((rv.dry_mix - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_conv_reverb_output_length() {
        let ir: Vec<f32> = (0..50).map(|i| 0.99_f32.powi(i)).collect();
        let rv = SimpleConvolutionReverb::new(ir);
        let input = vec![1.0_f32; 512];
        let out = rv.process(&input);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_conv_reverb_empty_input() {
        let ir = vec![1.0_f32, 0.5];
        let rv = SimpleConvolutionReverb::new(ir);
        let out = rv.process(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_conv_reverb_output_finite() {
        let ir: Vec<f32> = (0..128).map(|i| 0.97_f32.powi(i)).collect();
        let rv = SimpleConvolutionReverb::new(ir);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let out = rv.process(&input);
        for (i, &s) in out.iter().enumerate() {
            assert!(s.is_finite(), "out[{i}] not finite: {s}");
        }
    }

    /// Regression test for the `Vec<f32>` → [`DelayLine`] ring-buffer migration in
    /// `CombFilter` and `AllpassFilter`.
    ///
    /// Verifies that:
    /// 1. All output samples remain finite (no NaN / inf).
    /// 2. The reverb tail carries non-zero energy (reverb is actually doing work).
    /// 3. The total energy over a long tail window is finite (tail is not diverging).
    #[test]
    fn delay_line_migration_schroeder_bounded_and_decaying() {
        let mut rv = SchroederReverb::new(48000);

        // Feed a single impulse followed by a long silence.
        let mut impulse = vec![0.0_f32; 4096];
        impulse[0] = 1.0;

        let (ol, or_) = rv.process_stereo(&impulse, &impulse);
        for (i, (&l, &r)) in ol.iter().zip(or_.iter()).enumerate() {
            assert!(l.is_finite(), "out_l[{i}] not finite after migration: {l}");
            assert!(r.is_finite(), "out_r[{i}] not finite after migration: {r}");
        }

        // Drain a further 44100 samples (≈1 second) — all finite, finite total energy.
        let silence = vec![0.0_f32; 4096];
        let mut total_energy = 0.0_f32;
        let mut chunks = 0usize;
        while chunks * 4096 < 44_100 {
            let (chunk_l, chunk_r) = rv.process_stereo(&silence, &silence);
            for (i, (&l, &r)) in chunk_l.iter().zip(chunk_r.iter()).enumerate() {
                assert!(l.is_finite(), "late out_l[{i}] not finite: {l}");
                assert!(r.is_finite(), "late out_r[{i}] not finite: {r}");
            }
            let e: f32 = chunk_l.iter().chain(chunk_r.iter()).map(|s| s * s).sum();
            total_energy += e;
            chunks += 1;
        }

        assert!(
            total_energy > 1e-6,
            "Schroeder reverb tail should carry non-zero energy: {total_energy}"
        );
        assert!(
            total_energy < 1.0e6,
            "Schroeder reverb tail energy is diverging (migration bug?): {total_energy}"
        );
    }
}
