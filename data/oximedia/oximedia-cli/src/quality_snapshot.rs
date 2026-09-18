//! Quick no-reference quality snapshot for `probe --quality-snapshot`.
//!
//! Decodes **frame 0 only** of the probed file via
//! [`crate::frame_extract::extract_video_frame_rgb`] (a real container +
//! codec decode — Y4M natively; MP4/MKV/WebM/TS via demuxer + AV1/VP9/VP8),
//! converts the packed RGB24 result to a single-plane `Gray8`
//! [`oximedia_quality::Frame`] using BT.601 integer luma, and scores all
//! five no-reference metrics (Blur, Noise, Blockiness, BRISQUE, NIQE).
//!
//! Each metric is **independently optional** because the metrics have
//! different minimum-frame-size guards (Blur/Noise 8x8, Blockiness 16x16,
//! BRISQUE 32x32, NIQE 96x96) — a small frame legitimately drops some
//! metrics while keeping others.
//!
//! [`compute_quality_snapshot`] is deliberately **infallible**: any failure
//! (audio-only input, unsupported container, decode error) is encoded inside
//! the returned [`QualitySnapshot`] (`available == false` plus a `reason`)
//! so `probe` itself never fails because of the snapshot — mirroring the
//! graceful-degradation approach `probe --hash` uses.

use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use std::path::Path;

/// The frame index the snapshot is computed on.
///
/// Frame 0 is the only "quick" choice: the container decode path in
/// `frame_extract` has no seek support and decodes sequentially from the
/// start of the stream.
const SNAPSHOT_FRAME_INDEX: u64 = 0;

/// CSV columns appended to the probe CSV header when `--quality-snapshot`
/// is active. Must stay in sync with [`QualitySnapshot::csv_cells`].
pub(crate) const CSV_HEADER_SUFFIX: &str =
    ",quality_available,quality_blur,quality_noise,quality_blockiness,quality_brisque,quality_niqe";

/// Outcome of a single no-reference metric.
///
/// Exactly one of `score` / `unavailable` is `Some`: either the metric
/// produced a real score, or it was skipped with a human-readable reason
/// (typically a minimum-frame-size guard).
#[derive(Debug, Clone)]
pub(crate) struct MetricOutcome {
    /// The computed score, if the metric succeeded.
    pub(crate) score: Option<f64>,
    /// Why the metric could not be computed, if it failed.
    pub(crate) unavailable: Option<String>,
}

impl MetricOutcome {
    /// A successfully computed score.
    fn ok(score: f64) -> Self {
        Self {
            score: Some(score),
            unavailable: None,
        }
    }

    /// A metric that could not be computed (e.g. frame below its size guard).
    fn skipped(reason: String) -> Self {
        Self {
            score: None,
            unavailable: Some(reason),
        }
    }
}

/// No-reference quality snapshot of the first decoded video frame.
#[derive(Debug, Clone)]
pub(crate) struct QualitySnapshot {
    /// `true` when a real frame was decoded and metrics were attempted.
    pub(crate) available: bool,
    /// Why the whole snapshot is unavailable (audio-only file, unsupported
    /// container, decode failure, ...). `None` when `available` is `true`.
    pub(crate) reason: Option<String>,
    /// Index of the decoded frame (always frame 0).
    pub(crate) frame_index: u64,
    /// Decoded frame width in pixels (`None` when unavailable).
    pub(crate) width: Option<u32>,
    /// Decoded frame height in pixels (`None` when unavailable).
    pub(crate) height: Option<u32>,
    /// Sharpness via Laplacian/Tenengrad variance (higher = sharper,
    /// unbounded; near 0 = heavily blurred). Min 8x8.
    pub(crate) blur: MetricOutcome,
    /// Spatial high-pass noise power (higher = noisier, unbounded;
    /// 0 = locally smooth). Min 8x8.
    pub(crate) noise: MetricOutcome,
    /// Block-boundary edge excess in percent (>= 0, lower = better).
    /// Min 16x16.
    pub(crate) blockiness: MetricOutcome,
    /// BRISQUE (0-100, lower = better). Min 32x32.
    pub(crate) brisque: MetricOutcome,
    /// NIQE (lower = better; natural images ~3-5). Min 96x96.
    pub(crate) niqe: MetricOutcome,
}

impl QualitySnapshot {
    /// Build the "whole snapshot failed" value (frame never decoded).
    fn whole_frame_unavailable(reason: String) -> Self {
        let skipped = || MetricOutcome::skipped("no decoded frame".to_string());
        Self {
            available: false,
            reason: Some(reason),
            frame_index: SNAPSHOT_FRAME_INDEX,
            width: None,
            height: None,
            blur: skipped(),
            noise: skipped(),
            blockiness: skipped(),
            brisque: skipped(),
            niqe: skipped(),
        }
    }

    /// All five metrics as `(json_key, display_name, scale_hint, outcome)`
    /// rows, in stable output order.
    pub(crate) fn metric_rows(
        &self,
    ) -> [(&'static str, &'static str, &'static str, &MetricOutcome); 5] {
        [
            (
                "blur",
                "Blur",
                "sharpness variance (higher = sharper)",
                &self.blur,
            ),
            (
                "noise",
                "Noise",
                "noise power (higher = noisier)",
                &self.noise,
            ),
            (
                "blockiness",
                "Blockiness",
                "block-edge excess % (lower = better)",
                &self.blockiness,
            ),
            (
                "brisque",
                "BRISQUE",
                "0-100 (lower = better)",
                &self.brisque,
            ),
            ("niqe", "NIQE", "lower = better", &self.niqe),
        ]
    }

    /// JSON representation used by both `--format json` and `--ndjson`.
    ///
    /// When the snapshot is unavailable, `metrics` is `null` and `reason`
    /// explains why; individual metric entries carry `score`/`unavailable`
    /// so callers can tell "score is null because the frame is too small
    /// for this metric" apart from "metric succeeded".
    pub(crate) fn to_json(&self) -> serde_json::Value {
        let metrics = if self.available {
            let mut map = serde_json::Map::new();
            for (key, _name, scale, outcome) in self.metric_rows() {
                map.insert(
                    key.to_string(),
                    serde_json::json!({
                        "score": outcome.score,
                        "scale": scale,
                        "unavailable": outcome.unavailable.as_deref(),
                    }),
                );
            }
            serde_json::Value::Object(map)
        } else {
            serde_json::Value::Null
        };

        serde_json::json!({
            "available": self.available,
            "reason": self.reason.as_deref(),
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "metrics": metrics,
        })
    }

    /// CSV cells appended to the probe CSV data row (leading comma included).
    /// Column order must stay in sync with [`CSV_HEADER_SUFFIX`]; metrics
    /// that are unavailable produce empty cells.
    pub(crate) fn csv_cells(&self) -> String {
        let mut out = String::new();
        out.push(',');
        out.push_str(if self.available { "true" } else { "false" });
        for (_key, _name, _scale, outcome) in self.metric_rows() {
            out.push(',');
            if let Some(score) = outcome.score {
                out.push_str(&format!("{score:.6}"));
            }
        }
        out
    }
}

/// BT.601 integer luma of one RGB pixel: `y = (299*r + 587*g + 114*b) / 1000`.
///
/// Matches the integer formula in `oximedia_convert::color_convert::rgb_to_yuv`
/// (replicated locally so `oximedia-cli` does not grow an `oximedia-convert`
/// dependency for one line). Weights sum to 1000, so the result is already
/// in `0..=255`; the clamp is defensive only.
fn bt601_luma(r: u8, g: u8, b: u8) -> u8 {
    let y = (299 * i32::from(r) + 587 * i32::from(g) + 114 * i32::from(b)) / 1_000;
    y.clamp(0, 255) as u8
}

/// Convert packed RGB24 bytes to a single-plane `Gray8` quality [`Frame`].
///
/// All five no-reference metrics read only `frame.planes[0]` (the luma
/// plane), so a full RGB -> YUV conversion is unnecessary.
///
/// # Errors
///
/// Returns a human-readable message when the dimensions are zero, overflow
/// `usize`, or the RGB buffer is shorter than `width * height * 3`.
fn rgb24_to_gray_frame(rgb: &[u8], width: u32, height: u32) -> Result<Frame, String> {
    if width == 0 || height == 0 {
        return Err(format!(
            "decoded frame has zero dimensions ({width}x{height})"
        ));
    }

    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| format!("frame dimensions overflow: {width}x{height}"))?;
    let expected_len = pixel_count
        .checked_mul(3)
        .ok_or_else(|| format!("frame dimensions overflow: {width}x{height}"))?;
    if rgb.len() < expected_len {
        return Err(format!(
            "RGB buffer too short: got {} bytes, need {expected_len} for {width}x{height}",
            rgb.len()
        ));
    }

    let mut frame = Frame::new(width as usize, height as usize, PixelFormat::Gray8)
        .map_err(|e| format!("failed to allocate Gray8 frame: {e}"))?;
    let luma = frame.luma_mut();
    for (px, out) in rgb[..expected_len].chunks_exact(3).zip(luma.iter_mut()) {
        *out = bt601_luma(px[0], px[1], px[2]);
    }
    Ok(frame)
}

/// Score all five no-reference metrics on an already-decoded RGB24 frame.
///
/// Infallible: conversion failures produce a whole-snapshot-unavailable
/// value; per-metric failures (minimum-size guards) are recorded per metric
/// while the rest of the snapshot stays available.
fn snapshot_from_rgb(rgb: &[u8], width: u32, height: u32) -> QualitySnapshot {
    let frame = match rgb24_to_gray_frame(rgb, width, height) {
        Ok(frame) => frame,
        Err(reason) => return QualitySnapshot::whole_frame_unavailable(reason),
    };

    let assessor = QualityAssessor::new();
    let score_metric = |metric: MetricType| -> MetricOutcome {
        match assessor.assess_no_reference(&frame, metric) {
            Ok(score) => MetricOutcome::ok(score.score),
            Err(e) => MetricOutcome::skipped(e.to_string()),
        }
    };

    QualitySnapshot {
        available: true,
        reason: None,
        frame_index: SNAPSHOT_FRAME_INDEX,
        width: Some(width),
        height: Some(height),
        blur: score_metric(MetricType::Blur),
        noise: score_metric(MetricType::Noise),
        blockiness: score_metric(MetricType::Blockiness),
        brisque: score_metric(MetricType::Brisque),
        niqe: score_metric(MetricType::Niqe),
    }
}

/// Compute a quick no-reference quality snapshot on frame 0 of `path`.
///
/// This performs a **real decode** of the actual file content (never a
/// synthetic stand-in frame). The function never returns an error: on
/// audio-only or undecodable input the returned snapshot has
/// `available == false` and a descriptive `reason`, so `probe` degrades
/// gracefully instead of failing.
pub(crate) async fn compute_quality_snapshot(path: &Path) -> QualitySnapshot {
    match crate::frame_extract::extract_video_frame_rgb(path, SNAPSHOT_FRAME_INDEX).await {
        Ok((rgb, width, height)) => snapshot_from_rgb(&rgb, width, height),
        Err(e) => QualitySnapshot::whole_frame_unavailable(format!("{e:#}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic high-frequency texture so metrics see real structure
    /// (a flat or purely linear pattern would zero out Laplacian variance).
    fn textured_luma(x: usize, y: usize) -> u8 {
        ((x * 7 + y * 13) % 251) as u8
    }

    /// Packed RGB24 grey buffer whose per-pixel value equals `textured_luma`
    /// (r == g == b == v means BT.601 luma is exactly v).
    fn textured_rgb(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let v = textured_luma(x, y);
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        rgb
    }

    // -- bt601_luma ---------------------------------------------------------

    #[test]
    fn bt601_luma_known_values() {
        assert_eq!(bt601_luma(0, 0, 0), 0);
        assert_eq!(bt601_luma(255, 255, 255), 255);
        assert_eq!(bt601_luma(128, 128, 128), 128);
        // Primaries: floor((coeff * 255) / 1000)
        assert_eq!(bt601_luma(255, 0, 0), 76); // 299*255/1000 = 76.245
        assert_eq!(bt601_luma(0, 255, 0), 149); // 587*255/1000 = 149.685
        assert_eq!(bt601_luma(0, 0, 255), 29); // 114*255/1000 = 29.07
    }

    #[test]
    fn bt601_luma_grey_is_identity() {
        // Weights sum to exactly 1000, so grey pixels map to themselves.
        for v in [0u8, 1, 17, 100, 200, 254, 255] {
            assert_eq!(bt601_luma(v, v, v), v);
        }
    }

    // -- rgb24_to_gray_frame -------------------------------------------------

    #[test]
    fn rgb_to_gray_odd_dimensions_pixel_exact() {
        let (w, h) = (7usize, 5usize);
        let mut rgb = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h) {
            // Varied per-channel values, not just grey.
            let r = (i * 3 % 256) as u8;
            let g = (i * 5 % 256) as u8;
            let b = (i * 11 % 256) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }

        let frame = rgb24_to_gray_frame(&rgb, w as u32, h as u32).expect("conversion must succeed");
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        assert_eq!(frame.luma().len(), w * h);

        for (i, &luma) in frame.luma().iter().enumerate() {
            let px = &rgb[i * 3..i * 3 + 3];
            assert_eq!(
                luma,
                bt601_luma(px[0], px[1], px[2]),
                "luma mismatch at pixel {i}"
            );
        }
    }

    #[test]
    fn rgb_to_gray_rejects_short_buffer() {
        let rgb = vec![0u8; 4 * 4 * 3 - 1];
        let err = rgb24_to_gray_frame(&rgb, 4, 4).expect_err("short buffer must fail");
        assert!(err.contains("too short"), "unexpected message: {err}");
    }

    #[test]
    fn rgb_to_gray_rejects_zero_dimensions() {
        let err = rgb24_to_gray_frame(&[], 0, 16).expect_err("zero width must fail");
        assert!(err.contains("zero dimensions"), "unexpected message: {err}");
        let err = rgb24_to_gray_frame(&[], 16, 0).expect_err("zero height must fail");
        assert!(err.contains("zero dimensions"), "unexpected message: {err}");
    }

    // -- minimum-size guard tiers --------------------------------------------

    #[test]
    fn snapshot_96x96_scores_all_five_metrics() {
        let rgb = textured_rgb(96, 96);
        let snap = snapshot_from_rgb(&rgb, 96, 96);
        assert!(snap.available);
        assert_eq!(snap.reason, None);
        assert_eq!(snap.width, Some(96));
        assert_eq!(snap.height, Some(96));
        for (key, _name, _scale, outcome) in snap.metric_rows() {
            assert!(
                outcome.score.is_some(),
                "{key} must score on a 96x96 frame: {:?}",
                outcome.unavailable
            );
            let score = outcome.score.unwrap_or(f64::NAN);
            assert!(score.is_finite(), "{key} score must be finite: {score}");
        }
    }

    #[test]
    fn snapshot_32x32_drops_only_niqe() {
        let rgb = textured_rgb(32, 32);
        let snap = snapshot_from_rgb(&rgb, 32, 32);
        assert!(snap.available);
        assert!(snap.blur.score.is_some());
        assert!(snap.noise.score.is_some());
        assert!(snap.blockiness.score.is_some());
        assert!(snap.brisque.score.is_some());
        assert!(snap.niqe.score.is_none(), "NIQE needs 96x96");
        let reason = snap.niqe.unavailable.as_deref().unwrap_or("");
        assert!(reason.contains("too small"), "unexpected reason: {reason}");
    }

    #[test]
    fn snapshot_16x16_drops_brisque_and_niqe() {
        let rgb = textured_rgb(16, 16);
        let snap = snapshot_from_rgb(&rgb, 16, 16);
        assert!(snap.available);
        assert!(snap.blur.score.is_some());
        assert!(snap.noise.score.is_some());
        assert!(snap.blockiness.score.is_some());
        assert!(snap.brisque.score.is_none(), "BRISQUE needs 32x32");
        assert!(snap.niqe.score.is_none(), "NIQE needs 96x96");
    }

    #[test]
    fn snapshot_8x8_keeps_only_blur_and_noise() {
        let rgb = textured_rgb(8, 8);
        let snap = snapshot_from_rgb(&rgb, 8, 8);
        assert!(snap.available);
        assert!(snap.blur.score.is_some());
        assert!(snap.noise.score.is_some());
        assert!(snap.blockiness.score.is_none(), "blockiness needs 16x16");
        assert!(snap.brisque.score.is_none());
        assert!(snap.niqe.score.is_none());
    }

    #[test]
    fn snapshot_4x4_is_available_but_all_metrics_skipped() {
        let rgb = textured_rgb(4, 4);
        let snap = snapshot_from_rgb(&rgb, 4, 4);
        // The frame decoded fine — availability refers to the frame, not
        // to individual metrics.
        assert!(snap.available);
        for (key, _name, _scale, outcome) in snap.metric_rows() {
            assert!(outcome.score.is_none(), "{key} must be skipped at 4x4");
            assert!(outcome.unavailable.is_some());
        }
    }

    // -- JSON / CSV shapes ----------------------------------------------------

    #[test]
    fn to_json_available_shape() {
        let rgb = textured_rgb(96, 96);
        let snap = snapshot_from_rgb(&rgb, 96, 96);
        let json = snap.to_json();
        assert_eq!(json["available"], serde_json::json!(true));
        assert_eq!(json["reason"], serde_json::Value::Null);
        assert_eq!(json["frame_index"], serde_json::json!(0));
        assert_eq!(json["width"], serde_json::json!(96));
        assert_eq!(json["height"], serde_json::json!(96));
        let metrics = json["metrics"]
            .as_object()
            .expect("metrics must be an object when available");
        for key in ["blur", "noise", "blockiness", "brisque", "niqe"] {
            assert!(
                metrics[key]["score"].as_f64().is_some(),
                "{key} score must be a number: {}",
                metrics[key]
            );
            assert_eq!(metrics[key]["unavailable"], serde_json::Value::Null);
        }
    }

    #[test]
    fn to_json_unavailable_shape() {
        let snap = QualitySnapshot::whole_frame_unavailable("no video stream".to_string());
        let json = snap.to_json();
        assert_eq!(json["available"], serde_json::json!(false));
        assert_eq!(json["reason"], serde_json::json!("no video stream"));
        assert_eq!(json["width"], serde_json::Value::Null);
        assert_eq!(json["height"], serde_json::Value::Null);
        assert_eq!(json["metrics"], serde_json::Value::Null);
    }

    #[test]
    fn csv_cells_match_header_column_count() {
        let header_cols = CSV_HEADER_SUFFIX.split(',').count();

        let rgb = textured_rgb(96, 96);
        let available = snapshot_from_rgb(&rgb, 96, 96);
        assert_eq!(available.csv_cells().split(',').count(), header_cols);
        assert!(available.csv_cells().starts_with(",true,"));

        let unavailable = QualitySnapshot::whole_frame_unavailable("nope".to_string());
        assert_eq!(unavailable.csv_cells().split(',').count(), header_cols);
        assert!(unavailable.csv_cells().starts_with(",false,"));
        // No scores: every metric cell is empty.
        assert_eq!(unavailable.csv_cells(), ",false,,,,,");
    }

    // -- compute_quality_snapshot (real file paths) ---------------------------

    #[tokio::test]
    async fn compute_snapshot_missing_file_degrades_gracefully() {
        let path = std::env::temp_dir().join("oximedia_qsnap_no_such_file_777.y4m");
        let snap = compute_quality_snapshot(&path).await;
        assert!(!snap.available);
        let reason = snap.reason.as_deref().unwrap_or("");
        assert!(!reason.is_empty(), "reason must explain the failure");
    }

    #[tokio::test]
    async fn compute_snapshot_real_y4m_decodes_frame_zero() {
        // 16x16 C420jpeg Y4M: textured luma, neutral chroma (U=V=128).
        // With neutral chroma, YUV->RGB->BT.601 luma round-trips exactly,
        // so this exercises the full real-decode path deterministically.
        let (w, h) = (16usize, 16usize);
        let mut data = Vec::new();
        data.extend_from_slice(format!("YUV4MPEG2 W{w} H{h} F25:1 Ip C420jpeg\n").as_bytes());
        data.extend_from_slice(b"FRAME\n");
        for y in 0..h {
            for x in 0..w {
                data.push(textured_luma(x, y));
            }
        }
        let chroma_len = w.div_ceil(2) * h.div_ceil(2);
        data.extend(std::iter::repeat_n(128u8, chroma_len * 2));

        let path = std::env::temp_dir().join(format!(
            "oximedia_qsnap_unit_{}_16x16.y4m",
            std::process::id()
        ));
        std::fs::write(&path, &data).expect("write Y4M fixture");

        let snap = compute_quality_snapshot(&path).await;
        std::fs::remove_file(&path).ok();

        assert!(snap.available, "reason: {:?}", snap.reason);
        assert_eq!(snap.width, Some(16));
        assert_eq!(snap.height, Some(16));
        // 16x16 tier: blur/noise/blockiness score, brisque/niqe skipped.
        assert!(snap.blur.score.is_some());
        assert!(snap.noise.score.is_some());
        assert!(snap.blockiness.score.is_some());
        assert!(snap.brisque.score.is_none());
        assert!(snap.niqe.score.is_none());

        // Cross-check against the pure in-memory path: identical luma must
        // give identical scores (proves the decode fed real pixels through).
        let expected = snapshot_from_rgb(&textured_rgb(w, h), 16, 16);
        for ((key, _, _, got), (_, _, _, want)) in
            snap.metric_rows().iter().zip(expected.metric_rows().iter())
        {
            match (got.score, want.score) {
                (Some(a), Some(b)) => {
                    assert!((a - b).abs() < 1e-12, "{key}: {a} != {b}");
                }
                (None, None) => {}
                other => panic!("{key}: outcome mismatch: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn compute_snapshot_audio_only_wav_degrades_gracefully() {
        // Minimal WAV header — not a video, must not panic or error out.
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&36u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8000u32.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&0u32.to_le_bytes());

        let path = std::env::temp_dir().join(format!(
            "oximedia_qsnap_unit_{}_audio.wav",
            std::process::id()
        ));
        std::fs::write(&path, &wav).expect("write WAV fixture");

        let snap = compute_quality_snapshot(&path).await;
        std::fs::remove_file(&path).ok();

        assert!(!snap.available, "audio-only input must not yield a frame");
        assert!(snap.reason.is_some());
        assert!(snap.blur.score.is_none());
    }
}
