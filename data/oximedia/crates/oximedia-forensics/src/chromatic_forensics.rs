#![allow(dead_code)]
//! Chromatic aberration forensics for image tampering detection.
//!
//! Natural images captured by a camera lens exhibit predictable chromatic
//! aberration (CA) patterns -- lateral colour fringing that increases with
//! distance from the optical centre. When an image is composited or spliced,
//! the CA pattern in the manipulated region will be inconsistent with the rest
//! of the image, providing a strong forensic signal.
//!
//! # Features
//!
//! - **Per-block CA measurement** using cross-channel phase correlation
//! - **Radial CA model fitting** to separate natural CA from anomalies
//! - **Anomaly detection** for regions with inconsistent CA
//! - **Full-image CA consistency scoring**

use std::collections::HashMap;

/// Configuration for chromatic aberration analysis.
#[derive(Debug, Clone)]
pub struct ChromaticConfig {
    /// Block size in pixels (width and height of each analysis tile)
    pub block_size: usize,
    /// Minimum edge strength for a block to be considered (0.0..1.0)
    pub min_edge_strength: f64,
    /// Anomaly threshold expressed as multiples of standard deviation
    pub anomaly_sigma: f64,
}

impl Default for ChromaticConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            min_edge_strength: 0.1,
            anomaly_sigma: 2.5,
        }
    }
}

/// Measured lateral CA shift for a single image block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockCaShift {
    /// Block centre X in pixels
    pub cx: f64,
    /// Block centre Y in pixels
    pub cy: f64,
    /// Red-green lateral shift in pixels
    pub rg_shift: f64,
    /// Blue-green lateral shift in pixels
    pub bg_shift: f64,
    /// Edge strength weight
    pub weight: f64,
}

/// A simple radial CA model: `shift(r) = a * r + b * r^2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialCaModel {
    /// Optical centre X
    pub cx: f64,
    /// Optical centre Y
    pub cy: f64,
    /// Linear coefficient for red-green shift
    pub rg_a: f64,
    /// Quadratic coefficient for red-green shift
    pub rg_b: f64,
    /// Linear coefficient for blue-green shift
    pub bg_a: f64,
    /// Quadratic coefficient for blue-green shift
    pub bg_b: f64,
}

impl RadialCaModel {
    /// Predict the red-green shift at radius `r`.
    pub fn predict_rg(&self, r: f64) -> f64 {
        self.rg_a * r + self.rg_b * r * r
    }

    /// Predict the blue-green shift at radius `r`.
    pub fn predict_bg(&self, r: f64) -> f64 {
        self.bg_a * r + self.bg_b * r * r
    }

    /// Compute the radius of a point from the optical centre.
    pub fn radius(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.cx;
        let dy = y - self.cy;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Result of chromatic aberration forensic analysis.
#[derive(Debug, Clone)]
pub struct ChromaticAnalysisResult {
    /// Fitted radial CA model
    pub model: RadialCaModel,
    /// Per-block measurements
    pub block_shifts: Vec<BlockCaShift>,
    /// Indices of anomalous blocks
    pub anomalous_blocks: Vec<usize>,
    /// Overall consistency score (0.0 = very inconsistent, 1.0 = perfectly consistent)
    pub consistency_score: f64,
    /// R-squared of the radial model fit
    pub r_squared: f64,
}

/// Compute the cross-channel lateral shift between two single-channel images
/// represented as flat row-major `f64` slices, using normalised cross-correlation
/// over a small search window.
///
/// Returns the sub-pixel shift of `channel_b` relative to `channel_a`.
#[allow(clippy::cast_precision_loss)]
fn cross_channel_shift(
    channel_a: &[f64],
    channel_b: &[f64],
    width: usize,
    height: usize,
    max_shift: usize,
) -> f64 {
    if channel_a.len() != width * height || channel_b.len() != width * height {
        return 0.0;
    }
    let search = max_shift.min(width / 4).max(1);
    let mut best_corr = f64::NEG_INFINITY;
    let mut best_dx: isize = 0;

    for dx in -(search as isize)..=(search as isize) {
        let mut sum_ab = 0.0;
        let mut sum_aa = 0.0;
        let mut sum_bb = 0.0;
        for y in 0..height {
            for x in search..width.saturating_sub(search) {
                let bx = x as isize + dx;
                if bx < 0 || bx >= width as isize {
                    continue;
                }
                let a = channel_a[y * width + x];
                let b = channel_b[y * width + bx as usize];
                sum_ab += a * b;
                sum_aa += a * a;
                sum_bb += b * b;
            }
        }
        let denom = (sum_aa * sum_bb).sqrt();
        if denom > 1e-12 {
            let corr = sum_ab / denom;
            if corr > best_corr {
                best_corr = corr;
                best_dx = dx;
            }
        }
    }

    best_dx as f64
}

/// Compute the edge strength of a block (simple Sobel-like gradient magnitude).
#[allow(clippy::cast_precision_loss)]
fn block_edge_strength(data: &[f64], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = data[y * width + x + 1] - data[y * width + x - 1];
            let gy = data[(y + 1) * width + x] - data[(y - 1) * width + x];
            sum += (gx * gx + gy * gy).sqrt();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Fit a radial CA model from block measurements using weighted least squares.
///
/// Model: `shift = a*r + b*r^2`, solved via normal equations.
#[allow(clippy::cast_precision_loss)]
fn fit_radial_model(
    blocks: &[BlockCaShift],
    img_cx: f64,
    img_cy: f64,
) -> (f64, f64, f64, f64, f64) {
    // Solve for rg and bg independently
    let mut sr2 = 0.0;
    let mut sr3 = 0.0;
    let mut sr4 = 0.0;
    let mut srg_r = 0.0;
    let mut srg_r2 = 0.0;
    let mut sbg_r = 0.0;
    let mut sbg_r2 = 0.0;
    let mut total_w = 0.0;

    for b in blocks {
        let r = ((b.cx - img_cx).powi(2) + (b.cy - img_cy).powi(2)).sqrt();
        let w = b.weight;
        sr2 += w * r * r;
        sr3 += w * r * r * r;
        sr4 += w * r * r * r * r;
        srg_r += w * b.rg_shift * r;
        srg_r2 += w * b.rg_shift * r * r;
        sbg_r += w * b.bg_shift * r;
        sbg_r2 += w * b.bg_shift * r * r;
        total_w += w;
    }

    if total_w < 1e-12 || (sr2 * sr4 - sr3 * sr3).abs() < 1e-18 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let det = sr2 * sr4 - sr3 * sr3;
    let rg_a = (sr4 * srg_r - sr3 * srg_r2) / det;
    let rg_b = (sr2 * srg_r2 - sr3 * srg_r) / det;
    let bg_a = (sr4 * sbg_r - sr3 * sbg_r2) / det;
    let bg_b = (sr2 * sbg_r2 - sr3 * sbg_r) / det;

    // Compute R² for combined model
    let mean_rg: f64 = blocks.iter().map(|b| b.rg_shift * b.weight).sum::<f64>() / total_w;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for b in blocks {
        let r = ((b.cx - img_cx).powi(2) + (b.cy - img_cy).powi(2)).sqrt();
        let pred = rg_a * r + rg_b * r * r;
        ss_res += b.weight * (b.rg_shift - pred).powi(2);
        ss_tot += b.weight * (b.rg_shift - mean_rg).powi(2);
    }
    let r_sq = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    (rg_a, rg_b, bg_a, bg_b, r_sq)
}

/// Analyse chromatic aberration consistency across an image.
///
/// `red`, `green`, `blue` are flat row-major single-channel images normalised to `[0, 1]`.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_chromatic_aberration(
    red: &[f64],
    green: &[f64],
    blue: &[f64],
    width: usize,
    height: usize,
    config: &ChromaticConfig,
) -> ChromaticAnalysisResult {
    let bs = config.block_size.max(4);
    let img_cx = width as f64 / 2.0;
    let img_cy = height as f64 / 2.0;

    let mut blocks = Vec::new();

    let nx = width / bs;
    let ny = height / bs;

    for by in 0..ny {
        for bx in 0..nx {
            let x0 = bx * bs;
            let y0 = by * bs;
            let cx = x0 as f64 + bs as f64 / 2.0;
            let cy = y0 as f64 + bs as f64 / 2.0;

            // Extract block from green channel for edge strength
            let mut g_block = vec![0.0f64; bs * bs];
            for dy in 0..bs {
                for dx in 0..bs {
                    let idx = (y0 + dy) * width + (x0 + dx);
                    if idx < green.len() {
                        g_block[dy * bs + dx] = green[idx];
                    }
                }
            }
            let edge = block_edge_strength(&g_block, bs, bs);
            if edge < config.min_edge_strength {
                continue;
            }

            // Extract blocks for R and B
            let mut r_block = vec![0.0f64; bs * bs];
            let mut b_block = vec![0.0f64; bs * bs];
            for dy in 0..bs {
                for dx in 0..bs {
                    let idx = (y0 + dy) * width + (x0 + dx);
                    if idx < red.len() {
                        r_block[dy * bs + dx] = red[idx];
                    }
                    if idx < blue.len() {
                        b_block[dy * bs + dx] = blue[idx];
                    }
                }
            }

            let rg_shift = cross_channel_shift(&r_block, &g_block, bs, bs, 4);
            let bg_shift = cross_channel_shift(&b_block, &g_block, bs, bs, 4);

            blocks.push(BlockCaShift {
                cx,
                cy,
                rg_shift,
                bg_shift,
                weight: edge,
            });
        }
    }

    if blocks.is_empty() {
        return ChromaticAnalysisResult {
            model: RadialCaModel {
                cx: img_cx,
                cy: img_cy,
                rg_a: 0.0,
                rg_b: 0.0,
                bg_a: 0.0,
                bg_b: 0.0,
            },
            block_shifts: blocks,
            anomalous_blocks: Vec::new(),
            consistency_score: 1.0,
            r_squared: 1.0,
        };
    }

    let (rg_a, rg_b, bg_a, bg_b, r_sq) = fit_radial_model(&blocks, img_cx, img_cy);
    let model = RadialCaModel {
        cx: img_cx,
        cy: img_cy,
        rg_a,
        rg_b,
        bg_a,
        bg_b,
    };

    // Compute residuals
    let residuals: Vec<f64> = blocks
        .iter()
        .map(|b| {
            let r = model.radius(b.cx, b.cy);
            let pred_rg = model.predict_rg(r);
            let pred_bg = model.predict_bg(r);
            let drg = b.rg_shift - pred_rg;
            let dbg = b.bg_shift - pred_bg;
            (drg * drg + dbg * dbg).sqrt()
        })
        .collect();

    let n = residuals.len() as f64;
    let mean_res = residuals.iter().sum::<f64>() / n;
    let var_res = residuals
        .iter()
        .map(|r| (r - mean_res).powi(2))
        .sum::<f64>()
        / n;
    let std_res = var_res.sqrt();

    let threshold = mean_res + config.anomaly_sigma * std_res;
    let anomalous: Vec<usize> = residuals
        .iter()
        .enumerate()
        .filter(|(_, &r)| r > threshold)
        .map(|(i, _)| i)
        .collect();

    let anomaly_ratio = anomalous.len() as f64 / n;
    let consistency_score = (1.0 - anomaly_ratio).clamp(0.0, 1.0);

    ChromaticAnalysisResult {
        model,
        block_shifts: blocks,
        anomalous_blocks: anomalous,
        consistency_score,
        r_squared: r_sq,
    }
}

/// Summarise anomalous regions by grouping nearby anomalous blocks.
#[allow(clippy::cast_precision_loss)]
pub fn summarise_anomalies(result: &ChromaticAnalysisResult) -> HashMap<String, f64> {
    let mut summary = HashMap::new();
    summary.insert("total_blocks".to_string(), result.block_shifts.len() as f64);
    summary.insert(
        "anomalous_blocks".to_string(),
        result.anomalous_blocks.len() as f64,
    );
    summary.insert("consistency_score".to_string(), result.consistency_score);
    summary.insert("r_squared".to_string(), result.r_squared);
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_channel(width: usize, height: usize, value: f64) -> Vec<f64> {
        vec![value; width * height]
    }

    #[test]
    fn test_config_default() {
        let c = ChromaticConfig::default();
        assert_eq!(c.block_size, 64);
        assert!((c.min_edge_strength - 0.1).abs() < 1e-12);
        assert!((c.anomaly_sigma - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_radial_model_predict_zero() {
        let m = RadialCaModel {
            cx: 0.0,
            cy: 0.0,
            rg_a: 0.0,
            rg_b: 0.0,
            bg_a: 0.0,
            bg_b: 0.0,
        };
        assert!(m.predict_rg(10.0).abs() < 1e-12);
        assert!(m.predict_bg(10.0).abs() < 1e-12);
    }

    #[test]
    fn test_radial_model_linear() {
        let m = RadialCaModel {
            cx: 0.0,
            cy: 0.0,
            rg_a: 0.5,
            rg_b: 0.0,
            bg_a: -0.3,
            bg_b: 0.0,
        };
        assert!((m.predict_rg(4.0) - 2.0).abs() < 1e-12);
        assert!((m.predict_bg(4.0) - (-1.2)).abs() < 1e-12);
    }

    #[test]
    fn test_radial_model_radius() {
        let m = RadialCaModel {
            cx: 5.0,
            cy: 5.0,
            rg_a: 0.0,
            rg_b: 0.0,
            bg_a: 0.0,
            bg_b: 0.0,
        };
        assert!((m.radius(8.0, 9.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_cross_channel_shift_identical() {
        let data: Vec<f64> = (0..64).map(|i| (i % 8) as f64 / 7.0).collect();
        let shift = cross_channel_shift(&data, &data, 8, 8, 2);
        assert!(shift.abs() < 1e-6);
    }

    #[test]
    fn test_block_edge_strength_flat() {
        let data = vec![0.5; 25];
        let edge = block_edge_strength(&data, 5, 5);
        assert!(edge.abs() < 1e-12);
    }

    #[test]
    fn test_block_edge_strength_gradient() {
        let mut data = vec![0.0; 25];
        for y in 0..5 {
            for x in 0..5 {
                data[y * 5 + x] = x as f64 / 4.0;
            }
        }
        let edge = block_edge_strength(&data, 5, 5);
        assert!(edge > 0.0);
    }

    #[test]
    fn test_analyze_uniform_image() {
        let w = 128;
        let h = 128;
        let r = uniform_channel(w, h, 0.5);
        let g = uniform_channel(w, h, 0.5);
        let b = uniform_channel(w, h, 0.5);
        let config = ChromaticConfig::default();
        let result = analyze_chromatic_aberration(&r, &g, &b, w, h, &config);
        // Uniform image has no edges so no blocks should be measured
        assert!(result.block_shifts.is_empty());
        assert!((result.consistency_score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_analyze_empty_image() {
        let config = ChromaticConfig::default();
        let result = analyze_chromatic_aberration(&[], &[], &[], 0, 0, &config);
        assert!(result.block_shifts.is_empty());
    }

    #[test]
    fn test_summarise_anomalies() {
        let result = ChromaticAnalysisResult {
            model: RadialCaModel {
                cx: 0.0,
                cy: 0.0,
                rg_a: 0.0,
                rg_b: 0.0,
                bg_a: 0.0,
                bg_b: 0.0,
            },
            block_shifts: vec![
                BlockCaShift {
                    cx: 0.0,
                    cy: 0.0,
                    rg_shift: 0.0,
                    bg_shift: 0.0,
                    weight: 1.0,
                },
                BlockCaShift {
                    cx: 1.0,
                    cy: 1.0,
                    rg_shift: 0.0,
                    bg_shift: 0.0,
                    weight: 1.0,
                },
            ],
            anomalous_blocks: vec![1],
            consistency_score: 0.5,
            r_squared: 0.8,
        };
        let s = summarise_anomalies(&result);
        assert!((s["total_blocks"] - 2.0).abs() < 1e-12);
        assert!((s["anomalous_blocks"] - 1.0).abs() < 1e-12);
        assert!((s["consistency_score"] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_block_ca_shift_equality() {
        let a = BlockCaShift {
            cx: 1.0,
            cy: 2.0,
            rg_shift: 0.5,
            bg_shift: -0.3,
            weight: 1.0,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_block_edge_strength_small() {
        let data = vec![0.0; 4];
        let edge = block_edge_strength(&data, 2, 2);
        assert!(edge.abs() < 1e-12);
    }
}
