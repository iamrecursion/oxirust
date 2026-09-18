//! Audio vectorscope — stereo phase / Lissajous display.
//!
//! An audio vectorscope plots the left and right channel samples against each
//! other on an X/Y plane (Lissajous figure), making stereo width and phase
//! correlation immediately visible:
//!
//! - A **mono** signal (L == R) produces a single diagonal line at 45°.
//! - A **fully correlated anti-phase** signal (L == −R) produces a diagonal line
//!   at 135°.
//! - A **wide stereo** signal fills a roughly circular or elliptical area.
//!
//! The module also supports a **phase-coloured** mode where hue is derived from
//! the instantaneous phase angle (atan2), and an **intensity** mode where pixel
//! brightness encodes sample density.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

// ─── Colour mode ─────────────────────────────────────────────────────────────

/// Rendering style for the audio vectorscope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorscopeColorMode {
    /// Classic green-phosphor Lissajous figure.
    Lissajous,
    /// Hue encodes the instantaneous phase angle (atan2 of R/L).
    Phase,
    /// Pixel brightness is proportional to hit-count density.
    Intensity,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`AudioVectorscope`].
#[derive(Debug, Clone)]
pub struct AudioVectorscopeConfig {
    /// Output frame width in pixels.
    pub width: u32,
    /// Output frame height in pixels.
    pub height: u32,
    /// Persistence factor in `[0.0, 1.0)`.
    ///
    /// Each frame the accumulation buffer is multiplied by this value so that
    /// older samples fade out. `0.0` = no persistence (instant decay),
    /// `0.9` = slow fade.
    pub persistence: f32,
    /// Colour rendering mode.
    pub color_mode: VectorscopeColorMode,
    /// Input gain applied to samples before plotting (1.0 = unity).
    pub gain: f32,
}

impl Default for AudioVectorscopeConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            persistence: 0.85,
            color_mode: VectorscopeColorMode::Lissajous,
            gain: 1.0,
        }
    }
}

// ─── Vectorscope ─────────────────────────────────────────────────────────────

/// Audio vectorscope renderer with persistence.
///
/// Call [`render`](AudioVectorscope::render) repeatedly with successive audio
/// buffers. The internal accumulation buffer decays between calls according to
/// the configured `persistence` factor.
#[derive(Debug)]
pub struct AudioVectorscope {
    /// Rendering configuration.
    pub config: AudioVectorscopeConfig,
    /// Floating-point accumulation buffer — one value per pixel, representing
    /// accumulated energy at that position.  Three sub-channels (R, G, B) are
    /// stored interleaved for Phase mode; a single channel is used otherwise.
    frame_buffer: Vec<f32>,
}

impl AudioVectorscope {
    /// Creates a new vectorscope with the given configuration.
    #[must_use]
    pub fn new(config: AudioVectorscopeConfig) -> Self {
        let size = (config.width * config.height * 3) as usize;
        Self {
            config,
            frame_buffer: vec![0.0_f32; size],
        }
    }

    /// Clears the accumulation buffer.
    pub fn reset(&mut self) {
        for v in &mut self.frame_buffer {
            *v = 0.0;
        }
    }

    /// Renders one block of stereo audio into an RGBA frame.
    ///
    /// `left` and `right` must contain normalised samples in `[-1.0, 1.0]`.
    /// Shorter slices are silently zero-padded.  Returns `width × height × 4`
    /// bytes (RGBA, row-major, top-left origin).
    pub fn render(&mut self, left: &[f32], right: &[f32]) -> Vec<u8> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let cx = (w / 2) as f32;
        let cy = (h / 2) as f32;
        let half_w = cx;
        let half_h = cy;

        // ── Decay existing buffer ──────────────────────────────────────────
        let p = self.config.persistence.clamp(0.0, 0.999);
        for v in &mut self.frame_buffer {
            *v *= p;
        }

        // ── Plot new samples ───────────────────────────────────────────────
        let n = left.len().max(right.len());
        for i in 0..n {
            let l = if i < left.len() { left[i] } else { 0.0 };
            let r = if i < right.len() { right[i] } else { 0.0 };

            let l_scaled = (l * self.config.gain).clamp(-1.0, 1.0);
            let r_scaled = (r * self.config.gain).clamp(-1.0, 1.0);

            // X axis = right channel, Y axis = left channel (standard convention)
            // Centre of display = (0, 0), positive up/right
            let px = (cx + r_scaled * half_w * 0.95).round() as usize;
            let py = (cy - l_scaled * half_h * 0.95).round() as usize; // flip Y

            if px >= w || py >= h {
                continue;
            }

            let base = (py * w + px) * 3;

            match self.config.color_mode {
                VectorscopeColorMode::Lissajous => {
                    // Bright green dot
                    self.frame_buffer[base] += 0.0; // R
                    self.frame_buffer[base + 1] += 1.0; // G
                    self.frame_buffer[base + 2] += 0.0; // B
                }
                VectorscopeColorMode::Phase => {
                    // Map phase angle to hue
                    let angle = r_scaled.atan2(l_scaled); // −π .. +π
                    let hue = (angle / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
                    let (rr, gg, bb) = hsv_to_rgb(hue, 1.0, 1.0);
                    self.frame_buffer[base] += rr;
                    self.frame_buffer[base + 1] += gg;
                    self.frame_buffer[base + 2] += bb;
                }
                VectorscopeColorMode::Intensity => {
                    // Accumulate mono intensity; all three channels receive it
                    self.frame_buffer[base] += 1.0;
                    self.frame_buffer[base + 1] += 1.0;
                    self.frame_buffer[base + 2] += 1.0;
                }
            }
        }

        // ── Draw centre cross ──────────────────────────────────────────────
        let cross_color: f32 = 0.25;
        let cx_px = cx.round() as usize;
        let cy_px = cy.round() as usize;
        let cross_len = (w / 16).max(4);

        // Horizontal arm
        let y_base = cy_px * w;
        for dx in 0..cross_len {
            let x1 = cx_px.saturating_sub(dx);
            let x2 = (cx_px + dx).min(w - 1);
            let fade = cross_color * (1.0 - dx as f32 / cross_len as f32);
            if x1 < w {
                let idx = (y_base + x1) * 3;
                self.frame_buffer[idx] = self.frame_buffer[idx].max(fade);
                self.frame_buffer[idx + 1] = self.frame_buffer[idx + 1].max(fade);
                self.frame_buffer[idx + 2] = self.frame_buffer[idx + 2].max(fade);
            }
            if x2 < w {
                let idx = (y_base + x2) * 3;
                self.frame_buffer[idx] = self.frame_buffer[idx].max(fade);
                self.frame_buffer[idx + 1] = self.frame_buffer[idx + 1].max(fade);
                self.frame_buffer[idx + 2] = self.frame_buffer[idx + 2].max(fade);
            }
        }
        // Vertical arm
        for dy in 0..cross_len {
            let y1 = cy_px.saturating_sub(dy);
            let y2 = (cy_px + dy).min(h - 1);
            let fade = cross_color * (1.0 - dy as f32 / cross_len as f32);
            if y1 < h {
                let idx = (y1 * w + cx_px) * 3;
                self.frame_buffer[idx] = self.frame_buffer[idx].max(fade);
                self.frame_buffer[idx + 1] = self.frame_buffer[idx + 1].max(fade);
                self.frame_buffer[idx + 2] = self.frame_buffer[idx + 2].max(fade);
            }
            if y2 < h {
                let idx = (y2 * w + cx_px) * 3;
                self.frame_buffer[idx] = self.frame_buffer[idx].max(fade);
                self.frame_buffer[idx + 1] = self.frame_buffer[idx + 1].max(fade);
                self.frame_buffer[idx + 2] = self.frame_buffer[idx + 2].max(fade);
            }
        }

        // ── Convert accumulation buffer to RGBA ────────────────────────────
        // Find peak to normalise Intensity mode; other modes saturate at 1.0
        let peak = match self.config.color_mode {
            VectorscopeColorMode::Intensity => self
                .frame_buffer
                .iter()
                .cloned()
                .fold(0.0_f32, f32::max)
                .max(1.0),
            _ => 1.0_f32,
        };

        let mut out = vec![0u8; w * h * 4];
        for py2 in 0..h {
            for px2 in 0..w {
                let src = (py2 * w + px2) * 3;
                let dst = (py2 * w + px2) * 4;
                let r = ((self.frame_buffer[src] / peak).clamp(0.0, 1.0) * 255.0) as u8;
                let g = ((self.frame_buffer[src + 1] / peak).clamp(0.0, 1.0) * 255.0) as u8;
                let b = ((self.frame_buffer[src + 2] / peak).clamp(0.0, 1.0) * 255.0) as u8;
                out[dst] = r;
                out[dst + 1] = g;
                out[dst + 2] = b;
                out[dst + 3] = 255;
            }
        }
        out
    }
}

// ─── HSV → RGB helper ────────────────────────────────────────────────────────

/// Converts an HSV colour to linear RGB floats in `[0.0, 1.0]`.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (v, v, v);
    }
    let h6 = h * 6.0;
    let i = h6.floor() as u32;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn mono_ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect()
    }

    // ── Output dimensions ─────────────────────────────────────────────────

    #[test]
    fn test_render_returns_correct_size() {
        let cfg = AudioVectorscopeConfig {
            width: 128,
            height: 128,
            ..Default::default()
        };
        let mut vs = AudioVectorscope::new(cfg);
        let out = vs.render(&silence(256), &silence(256));
        assert_eq!(out.len(), 128 * 128 * 4);
    }

    #[test]
    fn test_render_default_size() {
        let mut vs = AudioVectorscope::new(AudioVectorscopeConfig::default());
        let out = vs.render(&silence(0), &silence(0));
        assert_eq!(out.len(), 512 * 512 * 4);
    }

    // ── Centre cross ──────────────────────────────────────────────────────

    #[test]
    fn test_centre_cross_pixels_nonzero() {
        let cfg = AudioVectorscopeConfig {
            width: 128,
            height: 128,
            persistence: 0.0,
            ..Default::default()
        };
        let mut vs = AudioVectorscope::new(cfg);
        let out = vs.render(&silence(0), &silence(0));
        // Centre pixel of a 128×128 canvas is (64, 64)
        let cx = 64usize;
        let cy = 64usize;
        let idx = (cy * 128 + cx) * 4;
        // At least one channel should be non-zero (the cross)
        let any_nonzero = out[idx] > 0 || out[idx + 1] > 0 || out[idx + 2] > 0;
        assert!(any_nonzero, "centre pixel should be non-zero from cross");
    }

    // ── Silence ───────────────────────────────────────────────────────────

    #[test]
    fn test_silence_centre_has_cross_only() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.0,
            ..Default::default()
        };
        let mut vs = AudioVectorscope::new(cfg);
        let out = vs.render(&silence(512), &silence(512));
        // Corner pixel should be black
        let corner = out[0..4].to_vec();
        assert_eq!(corner[0], 0);
        assert_eq!(corner[1], 0);
        assert_eq!(corner[2], 0);
    }

    // ── Mono signal ───────────────────────────────────────────────────────

    #[test]
    fn test_mono_signal_diagonal_energy() {
        let cfg = AudioVectorscopeConfig {
            width: 128,
            height: 128,
            persistence: 0.0,
            color_mode: VectorscopeColorMode::Lissajous,
            gain: 1.0,
        };
        let mut vs = AudioVectorscope::new(cfg);
        let signal: Vec<f32> = mono_ramp(1024);
        let out = vs.render(&signal, &signal); // identical L and R → diagonal

        // Sum green channel along main diagonal vs anti-diagonal
        let mut diag_sum: u64 = 0;
        let mut anti_sum: u64 = 0;
        for i in 0..128usize {
            // main diagonal (top-left to bottom-right, but recall Y is flipped)
            // When L==R, pixel lands on centre diagonal
            let idx_diag = (i * 128 + i) * 4;
            diag_sum += u64::from(out[idx_diag + 1]); // green
                                                      // anti-diagonal
            let j = 127 - i;
            let idx_anti = (i * 128 + j) * 4;
            anti_sum += u64::from(out[idx_anti + 1]);
        }
        assert!(
            diag_sum > anti_sum,
            "diagonal should have more energy than anti-diagonal for mono, diag={diag_sum} anti={anti_sum}"
        );
    }

    // ── Phase mode ────────────────────────────────────────────────────────

    #[test]
    fn test_phase_mode_produces_colour() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.0,
            color_mode: VectorscopeColorMode::Phase,
            gain: 1.0,
        };
        let mut vs = AudioVectorscope::new(cfg);
        let l: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let r: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05 + 0.7).sin()).collect();
        let out = vs.render(&l, &r);

        // At least some pixels should have different R and G channels (not grey)
        let has_colour = out.chunks_exact(4).any(|p| p[0] != p[1] || p[1] != p[2]);
        assert!(has_colour, "phase mode should produce coloured output");
    }

    // ── Intensity mode ────────────────────────────────────────────────────

    #[test]
    fn test_intensity_mode_output_is_greyscale_ish() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.0,
            color_mode: VectorscopeColorMode::Intensity,
            gain: 1.0,
        };
        let mut vs = AudioVectorscope::new(cfg);
        let signal = mono_ramp(512);
        let out = vs.render(&signal, &signal);
        // In Intensity mode R==G==B for all pixels
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], chunk[1], "intensity R should == G");
            assert_eq!(chunk[1], chunk[2], "intensity G should == B");
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────

    #[test]
    fn test_persistence_accumulates_over_frames() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.9,
            color_mode: VectorscopeColorMode::Lissajous,
            gain: 1.0,
        };
        let mut vs = AudioVectorscope::new(cfg);
        let signal = mono_ramp(256);

        let out1 = vs.render(&signal, &signal);
        let out2 = vs.render(&signal, &signal);

        // With persistence, second frame should have at least as bright pixels
        let sum1: u64 = out1.iter().map(|&v| u64::from(v)).sum();
        let sum2: u64 = out2.iter().map(|&v| u64::from(v)).sum();
        assert!(
            sum2 >= sum1,
            "persistence should not reduce brightness over frames"
        );
    }

    #[test]
    fn test_zero_persistence_clears_each_frame() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.0,
            color_mode: VectorscopeColorMode::Lissajous,
            gain: 1.0,
        };
        let mut vs = AudioVectorscope::new(cfg);
        let signal = mono_ramp(256);
        let first = vs.render(&signal, &signal);

        // Second render with silence (same silence both frames).
        // With zero persistence the accumulation buffer is fully cleared each frame,
        // so the second frame (silence) must not be brighter than the first (full signal).
        let second = vs.render(&silence(256), &silence(256));

        let sum_first: u64 = first.iter().map(|&v| u64::from(v)).sum();
        let sum_second: u64 = second.iter().map(|&v| u64::from(v)).sum();

        // The silent frame should be dimmer than the active-signal frame
        assert!(
            sum_second <= sum_first,
            "zero persistence: silence frame ({sum_second}) should be <= signal frame ({sum_first})"
        );
    }

    // ── Gain scaling ──────────────────────────────────────────────────────

    #[test]
    fn test_gain_clamps_to_display_boundary() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.0,
            color_mode: VectorscopeColorMode::Lissajous,
            gain: 10.0, // extreme gain → all samples should clamp to edges
        };
        let mut vs = AudioVectorscope::new(cfg);
        let signal: Vec<f32> = vec![0.5; 256];
        // Should not panic and should return correct size
        let out = vs.render(&signal, &signal);
        assert_eq!(out.len(), 64 * 64 * 4);
    }

    // ── Reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_buffer() {
        let cfg = AudioVectorscopeConfig {
            width: 64,
            height: 64,
            persistence: 0.9,
            color_mode: VectorscopeColorMode::Lissajous,
            gain: 1.0,
        };
        let mut vs = AudioVectorscope::new(cfg);
        let signal = mono_ramp(512);
        vs.render(&signal, &signal);
        vs.reset();
        // After reset, only the cross should be visible
        let out = vs.render(&silence(0), &silence(0));
        let bright: usize = out.chunks_exact(4).filter(|p| p[1] > 10).count();
        // Cross spans roughly cross_len*2 pixels; should be << total pixels
        assert!(bright < 200, "reset should clear most of the display");
    }

    // ── HSV helper ────────────────────────────────────────────────────────

    #[test]
    fn test_hsv_to_rgb_primary_hues() {
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0); // red
        assert!(r > 0.9 && g < 0.1 && b < 0.1);
        let (r, g, b) = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0); // green
        assert!(r < 0.1 && g > 0.9 && b < 0.1);
        let (r, g, b) = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0); // blue
        assert!(r < 0.1 && g < 0.1 && b > 0.9);
    }
}
