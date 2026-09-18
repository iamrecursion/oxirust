//! Flash and strobe detection for accessibility compliance.
//!
//! Detects rapid luminance changes (flashes / strobes) in a sequence of video
//! frames and reports whether the detected events fail the Harding photosensitive
//! epilepsy (PSE) standard (> 3 Hz within a 10° visual field with Δluma > 0.1).
//!
//! # References
//!
//! - Harding PSE standard — ITC / Ofcom Guidelines on Harmful Flash and Patterns
//!   in Television (2005)
//! - WCAG 2.1 — Success Criterion 2.3.1 Three Flashes or Below Threshold
//!
//! # Example
//!
//! ```
//! use oximedia_shots::flash_detection::{FlashDetector, FlashEvent};
//!
//! let fps = 25.0_f32;
//! let detector = FlashDetector::new(fps);
//! // frames: (frame_index, RGBA pixels f32 in [0, 1])
//! let frames: Vec<(u64, Vec<f32>)> = Vec::new();
//! let events = detector.detect(&frames);
//! let report = FlashDetector::compliance_report(&events);
//! println!("{report}");
//! ```

// ── Public types ─────────────────────────────────────────────────────────────

/// A detected flash / strobe event comprising one or more polarity reversals.
#[derive(Debug, Clone, PartialEq)]
pub struct FlashEvent {
    /// Index of the first frame in the flash sequence.
    pub start_frame: u64,
    /// Index of the last frame in the flash sequence.
    pub end_frame: u64,
    /// Maximum luminance delta (Δluma) observed across all transitions in the event.
    pub peak_luminance_change: f32,
    /// Semantic type of the flash.
    pub flash_type: FlashType,
    /// Estimated flash frequency in Hz (reversals / window duration).
    pub frequency_hz: f32,
    /// `true` when the event exceeds the Harding PSE threshold:
    /// `frequency_hz > harding_freq_limit` AND `peak_luminance_change > 0.1`.
    pub fails_harding_standard: bool,
}

/// Semantic classification of a detected flash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashType {
    /// General luminance flash (large Δluma, not classified as red).
    WhiteFlash,
    /// Predominantly red-channel flash (saturated red transitions).
    RedFlash,
    /// General flash that does not fit white or red classification.
    GeneralFlash,
}

/// Flash and strobe detector configuration.
#[derive(Debug, Clone)]
pub struct FlashDetector {
    /// Frames per second of the source video.
    pub fps: f32,
    /// Minimum mean-luminance change between consecutive frames to count as a
    /// flash transition (default `0.1`).
    pub luma_threshold: f32,
    /// Minimum mean-red-channel change between consecutive frames to count as a
    /// red-flash transition (default `0.1`).
    pub red_threshold: f32,
    /// Duration of the sliding analysis window in seconds (default `1.0`).
    pub window_secs: f32,
    /// Maximum flash frequency in Hz that is considered safe under the Harding
    /// PSE standard (default `3.0`).
    pub harding_freq_limit: f32,
    /// Visual-field area percentage used in the Harding threshold check
    /// (default `0.25` = 25 %).  Reserved for future spatial analysis.
    pub harding_area_pct: f32,
}

impl Default for FlashDetector {
    fn default() -> Self {
        Self {
            fps: 25.0,
            luma_threshold: 0.1,
            red_threshold: 0.1,
            window_secs: 1.0,
            harding_freq_limit: 3.0,
            harding_area_pct: 0.25,
        }
    }
}

impl FlashDetector {
    /// Create a detector with default parameters for `fps`.
    #[must_use]
    pub fn new(fps: f32) -> Self {
        Self {
            fps,
            ..Default::default()
        }
    }

    /// Analyse a sequence of `(frame_index, RGBA pixels f32 [0, 1])` tuples
    /// and return a list of detected flash events.
    ///
    /// Algorithm:
    /// 1. For each consecutive pair of frames, compute mean luminance and mean
    ///    red channel.
    /// 2. Record a *transition* whenever |Δluma| ≥ `luma_threshold`.
    /// 3. Detect luminance *polarity reversals* (sign change of Δluma) within a
    ///    sliding window of `window_secs`.
    /// 4. Cluster consecutive reversals into [`FlashEvent`]s.
    /// 5. Estimate frequency and apply Harding check.
    #[must_use]
    pub fn detect(&self, frames: &[(u64, Vec<f32>)]) -> Vec<FlashEvent> {
        if frames.len() < 2 {
            return Vec::new();
        }

        // ── Step 1: compute per-frame mean luma and mean red ──────────────────
        let frame_lumas: Vec<f32> = frames
            .iter()
            .map(|(_, pixels)| Self::luminance(pixels))
            .collect();
        let frame_reds: Vec<f32> = frames.iter().map(|(_, pixels)| mean_red(pixels)).collect();

        // ── Step 2: build delta sequence ─────────────────────────────────────
        // A Transition records the signed luma difference and whether it meets
        // the threshold.
        #[derive(Debug, Clone, Copy)]
        struct Transition {
            frame_a: u64,
            frame_b: u64,
            delta_luma: f32,
            delta_red: f32,
        }

        let mut transitions: Vec<Transition> = Vec::with_capacity(frames.len());
        for i in 1..frames.len() {
            let dl = frame_lumas[i] - frame_lumas[i - 1];
            let dr = frame_reds[i] - frame_reds[i - 1];
            if dl.abs() >= self.luma_threshold {
                transitions.push(Transition {
                    frame_a: frames[i - 1].0,
                    frame_b: frames[i].0,
                    delta_luma: dl,
                    delta_red: dr,
                });
            }
        }

        if transitions.len() < 2 {
            return Vec::new();
        }

        // ── Step 3: detect polarity reversals within the sliding window ───────
        let window_frames = (self.window_secs * self.fps).ceil() as usize;

        // A reversal is a pair (t-1, t) where sign(delta[t]) != sign(delta[t-1])
        // We collect reversal indices into flash clusters.
        let mut events: Vec<FlashEvent> = Vec::new();

        // Sliding window over `transitions`
        let mut window_start = 0_usize;
        let mut t = 1_usize;

        while t < transitions.len() {
            let prev = transitions[t - 1];
            let curr = transitions[t];

            // Check for polarity reversal
            let is_reversal = prev.delta_luma.signum() != curr.delta_luma.signum();
            if !is_reversal {
                t += 1;
                window_start = window_start.min(t.saturating_sub(1));
                continue;
            }

            // We have a reversal at `t`.  Look forward to collect contiguous
            // reversals within `window_frames`.
            let cluster_start_frame = prev.frame_a;
            let mut cluster_end_frame = curr.frame_b;
            let mut reversal_count = 1_usize;
            let mut peak_delta = curr.delta_luma.abs().max(prev.delta_luma.abs());
            let mut max_red_delta = curr.delta_red.abs().max(prev.delta_red.abs());

            let mut j = t + 1;
            while j < transitions.len() {
                let span_frames = transitions[j].frame_b.saturating_sub(cluster_start_frame);
                if span_frames as usize > window_frames {
                    break;
                }
                let prev_j = transitions[j - 1];
                let curr_j = transitions[j];
                if prev_j.delta_luma.signum() != curr_j.delta_luma.signum() {
                    reversal_count += 1;
                    cluster_end_frame = curr_j.frame_b;
                    let d = curr_j.delta_luma.abs();
                    if d > peak_delta {
                        peak_delta = d;
                    }
                    let rd = curr_j.delta_red.abs();
                    if rd > max_red_delta {
                        max_red_delta = rd;
                    }
                }
                j += 1;
            }

            // Duration in seconds
            let duration_secs = (cluster_end_frame.saturating_sub(cluster_start_frame) as f32)
                / self.fps.max(f32::EPSILON);
            let frequency_hz = if duration_secs > f32::EPSILON {
                reversal_count as f32 / duration_secs
            } else {
                self.fps
            };

            // Classify flash type
            let flash_type =
                if max_red_delta >= self.red_threshold && max_red_delta > peak_delta * 0.5 {
                    FlashType::RedFlash
                } else if peak_delta >= 0.2 {
                    FlashType::WhiteFlash
                } else {
                    FlashType::GeneralFlash
                };

            // Harding standard check
            let fails_harding = frequency_hz > self.harding_freq_limit && peak_delta > 0.1;

            events.push(FlashEvent {
                start_frame: cluster_start_frame,
                end_frame: cluster_end_frame,
                peak_luminance_change: peak_delta,
                flash_type,
                frequency_hz,
                fails_harding_standard: fails_harding,
            });

            // Advance past this cluster
            t = j;
            window_start = t.saturating_sub(1);
        }

        events
    }

    /// Compute the mean Y (luminance) value from a flat RGBA pixel buffer (f32,
    /// normalised to `[0, 1]`).
    ///
    /// Uses ITU-R BT.709 coefficients: `Y = 0.2126R + 0.7152G + 0.0722B`.
    /// The alpha channel is ignored.
    ///
    /// Returns `0.0` for an empty or malformed buffer.
    #[must_use]
    pub fn luminance(rgba: &[f32]) -> f32 {
        if rgba.len() < 4 {
            return 0.0;
        }
        let num_pixels = rgba.len() / 4;
        let total: f32 = rgba
            .chunks_exact(4)
            .map(|px| 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2])
            .sum();
        total / num_pixels as f32
    }

    /// Generate a human-readable compliance summary for a set of flash events.
    ///
    /// Summarises the number of events that fail the Harding PSE standard, their
    /// time ranges (in frame indices), peak luminance changes, and frequencies.
    #[must_use]
    pub fn compliance_report(events: &[FlashEvent]) -> String {
        let failing: Vec<&FlashEvent> =
            events.iter().filter(|e| e.fails_harding_standard).collect();

        if failing.is_empty() {
            return "Flash compliance report: PASS — no Harding PSE violations detected."
                .to_owned();
        }

        let mut report = format!(
            "Flash compliance report: FAIL — {} Harding PSE violation(s) detected.\n",
            failing.len()
        );
        for (i, event) in failing.iter().enumerate() {
            report.push_str(&format!(
                "  [{i}] frames {start}–{end}, type={type:?}, \
                 freq={freq:.2} Hz, peak_Δluma={delta:.3}\n",
                i = i + 1,
                start = event.start_frame,
                end = event.end_frame,
                type = event.flash_type,
                freq = event.frequency_hz,
                delta = event.peak_luminance_change,
            ));
        }
        report
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Mean value of the red channel from a flat RGBA f32 buffer.
fn mean_red(rgba: &[f32]) -> f32 {
    if rgba.len() < 4 {
        return 0.0;
    }
    let num_pixels = rgba.len() / 4;
    let total: f32 = rgba.chunks_exact(4).map(|px| px[0]).sum();
    total / num_pixels as f32
}

/// Build a test RGBA frame (all pixels set to `(r, g, b, 1.0)`).
#[cfg(test)]
fn solid_rgba_frame(r: f32, g: f32, b: f32, num_pixels: usize) -> Vec<f32> {
    let mut buf = Vec::with_capacity(num_pixels * 4);
    for _ in 0..num_pixels {
        buf.push(r);
        buf.push(g);
        buf.push(b);
        buf.push(1.0);
    }
    buf
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const PIXELS: usize = 100;

    // 1. Default constructor
    #[test]
    fn test_new_default_fps() {
        let det = FlashDetector::new(25.0);
        assert!((det.fps - 25.0).abs() < f32::EPSILON);
        assert!((det.luma_threshold - 0.1).abs() < f32::EPSILON);
        assert!((det.harding_freq_limit - 3.0).abs() < f32::EPSILON);
    }

    // 2. Empty input returns no events
    #[test]
    fn test_detect_empty_no_events() {
        let det = FlashDetector::new(25.0);
        assert!(det.detect(&[]).is_empty());
    }

    // 3. Single frame returns no events
    #[test]
    fn test_detect_single_frame_no_events() {
        let det = FlashDetector::new(25.0);
        let frames = vec![(0_u64, solid_rgba_frame(0.5, 0.5, 0.5, PIXELS))];
        assert!(det.detect(&frames).is_empty());
    }

    // 4. luminance of white frame ≈ 1.0
    #[test]
    fn test_luminance_white() {
        let frame = solid_rgba_frame(1.0, 1.0, 1.0, PIXELS);
        let luma = FlashDetector::luminance(&frame);
        assert!((luma - 1.0).abs() < 1e-4, "expected ~1.0, got {luma}");
    }

    // 5. luminance of black frame ≈ 0.0
    #[test]
    fn test_luminance_black() {
        let frame = solid_rgba_frame(0.0, 0.0, 0.0, PIXELS);
        let luma = FlashDetector::luminance(&frame);
        assert!(luma.abs() < 1e-6, "expected ~0.0, got {luma}");
    }

    // 6. luminance of empty buffer is 0.0
    #[test]
    fn test_luminance_empty() {
        assert_eq!(FlashDetector::luminance(&[]), 0.0);
    }

    // 7. compliance_report for no events is PASS
    #[test]
    fn test_compliance_report_no_events() {
        let report = FlashDetector::compliance_report(&[]);
        assert!(report.contains("PASS"), "expected PASS, got: {report}");
    }

    // 8. compliance_report with a failing event is FAIL
    #[test]
    fn test_compliance_report_fail() {
        let event = FlashEvent {
            start_frame: 0,
            end_frame: 10,
            peak_luminance_change: 0.8,
            flash_type: FlashType::WhiteFlash,
            frequency_hz: 10.0,
            fails_harding_standard: true,
        };
        let report = FlashDetector::compliance_report(&[event]);
        assert!(report.contains("FAIL"), "expected FAIL, got: {report}");
        assert!(report.contains("10.00 Hz"), "expected frequency in report");
    }

    // 9. Rapidly alternating bright/dark frames → flash event detected
    #[test]
    fn test_rapid_alternation_detected() {
        let fps = 25.0_f32;
        let det = FlashDetector {
            fps,
            luma_threshold: 0.1,
            red_threshold: 0.1,
            window_secs: 1.0,
            harding_freq_limit: 3.0,
            harding_area_pct: 0.25,
        };

        // Build 25 frames: alternating bright (luma≈1) and dark (luma≈0)
        let mut frames: Vec<(u64, Vec<f32>)> = Vec::new();
        for i in 0..25_u64 {
            let luma = if i % 2 == 0 { 1.0_f32 } else { 0.0_f32 };
            frames.push((i, solid_rgba_frame(luma, luma, luma, PIXELS)));
        }

        let events = det.detect(&frames);
        assert!(
            !events.is_empty(),
            "should detect flash events in rapidly alternating frames"
        );
    }

    // 10. Slow alternation (< 3 Hz) should not fail Harding standard
    #[test]
    fn test_slow_flash_passes_harding() {
        let fps = 25.0_f32;
        let det = FlashDetector::new(fps);

        // One bright→dark transition per second (1 Hz) over 4 seconds
        let mut frames: Vec<(u64, Vec<f32>)> = Vec::new();
        for i in 0..100_u64 {
            // Toggle once per 25 frames (= 1 Hz)
            let luma = if (i / 25) % 2 == 0 { 0.9_f32 } else { 0.1_f32 };
            frames.push((i, solid_rgba_frame(luma, luma, luma, PIXELS)));
        }

        let events = det.detect(&frames);
        // Any events that are detected should pass the Harding standard
        for event in &events {
            assert!(
                !event.fails_harding_standard,
                "slow flash at ~1 Hz should pass Harding, but event has freq={:.2}",
                event.frequency_hz
            );
        }
    }
}
