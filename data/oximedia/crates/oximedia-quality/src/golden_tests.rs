//! Golden reference tests comparing metric outputs against known-correct values.
//!
//! These tests assert that the quality metric implementations produce outputs
//! within a tight tolerance of analytically computable or externally verified
//! reference values.  They serve as regression guards: if a refactor changes
//! the numerical output, these tests will catch it immediately.
//!
//! # Reference derivations
//!
//! ## PSNR
//! For two uniform-grey frames of value A and B on an 8-bit plane:
//! - MSE = (A − B)²
//! - PSNR = 10 · log₁₀(255² / MSE)
//!
//! ## SSIM (uniform planes)
//! When both planes have constant pixel values μ₁ and μ₂ and σ = 0:
//! - SSIM = (2μ₁μ₂ + C1) / (μ₁² + μ₂² + C1)
//!
//! ## VIF
//! Identical uniform frames → VIF = 1.0 (lossless).

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::{
        Frame, FsimCalculator, MsSsimCalculator, PoolingMethod, PsnrCalculator, SsimCalculator,
        VifCalculator, VmafCalculator,
    };
    use oximedia_core::PixelFormat;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn uniform_yuv420(width: usize, height: usize, y: u8, cb: u8, cr: u8) -> Frame {
        let mut f =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        f.planes[0].fill(y);
        f.planes[1].fill(cb);
        f.planes[2].fill(cr);
        f
    }

    fn grey_yuv420(width: usize, height: usize, y: u8) -> Frame {
        uniform_yuv420(width, height, y, 128, 128)
    }

    /// Analytically compute weighted PSNR for two uniform YUV420 frames.
    ///
    /// For plane P, MSE = (y1 − y2)² and PSNR_P = 10·log10(255²/MSE).
    /// The luma (Y) plane uses weight 4/6; each chroma plane uses 1/6.
    fn expected_psnr(y1: u8, y2: u8, cb1: u8, cb2: u8, cr1: u8, cr2: u8) -> f64 {
        fn plane_psnr(a: u8, b: u8) -> f64 {
            let mse = (i32::from(a) - i32::from(b)).pow(2) as f64;
            if mse < 1e-10 {
                return 100.0;
            }
            10.0 * (255.0_f64.powi(2) / mse).log10()
        }
        let y_psnr = plane_psnr(y1, y2);
        let cb_psnr = plane_psnr(cb1, cb2);
        let cr_psnr = plane_psnr(cr1, cr2);
        (4.0 / 6.0) * y_psnr + (1.0 / 6.0) * (cb_psnr + cr_psnr)
    }

    /// Analytically compute SSIM for two uniform constant-value planes.
    fn expected_ssim_uniform(mu1: f64, mu2: f64) -> f64 {
        // σ = 0, σ_xy = 0
        let l = 255.0_f64;
        let c1 = (0.01 * l).powi(2);
        let c2 = (0.03 * l).powi(2);
        // luminance component
        let lum = (2.0 * mu1 * mu2 + c1) / (mu1.powi(2) + mu2.powi(2) + c1);
        // contrast + structure = C2 / C2 = 1 when σ = 0
        let cs = c2 / c2;
        lum * cs
    }

    // ── PSNR golden tests ─────────────────────────────────────────────────

    #[test]
    fn golden_psnr_uniform_diff_10() {
        // Y: 100 vs 110 → MSE = 100, PSNR_Y = 10·log10(65025/100) ≈ 28.13
        // Cb/Cr identical → PSNR = 100
        let calc = PsnrCalculator::new();
        let f1 = grey_yuv420(32, 32, 100);
        let f2 = grey_yuv420(32, 32, 110);

        let result = calc.calculate(&f1, &f2).expect("should succeed");
        let expected = expected_psnr(100, 110, 128, 128, 128, 128);

        assert!(
            (result.score - expected).abs() < 0.05,
            "PSNR={:.4} expected={:.4}",
            result.score,
            expected
        );
    }

    #[test]
    fn golden_psnr_identical_returns_high_value() {
        let calc = PsnrCalculator::new();
        let f = grey_yuv420(32, 32, 128);
        let result = calc.calculate(&f, &f).expect("should succeed");
        assert!(
            result.score >= 99.0,
            "Identical frames must yield very high PSNR, got {}",
            result.score
        );
    }

    #[test]
    fn golden_psnr_known_mse() {
        // Y: 0 vs 255 → MSE = 65025 → PSNR_Y = 10·log10(65025/65025) = 0 dB
        let calc = PsnrCalculator::new();
        let f1 = grey_yuv420(32, 32, 0);
        let f2 = grey_yuv420(32, 32, 255);

        let y_psnr = result_y_component(&calc.calculate(&f1, &f2).expect("should succeed"));
        assert!(
            y_psnr.abs() < 0.1,
            "PSNR for max difference must be ≈ 0 dB, got {y_psnr}"
        );
    }

    fn result_y_component(score: &crate::QualityScore) -> f64 {
        *score.components.get("Y").unwrap_or(&0.0)
    }

    #[test]
    fn golden_psnr_diff_1() {
        // Y: 127 vs 128 → MSE = 1 → PSNR_Y = 10·log10(65025) ≈ 48.13
        let calc = PsnrCalculator::new();
        let f1 = grey_yuv420(64, 64, 127);
        let f2 = grey_yuv420(64, 64, 128);

        let y_psnr = result_y_component(&calc.calculate(&f1, &f2).expect("should succeed"));
        let expected_y = 10.0 * (255.0_f64.powi(2) / 1.0).log10();
        assert!(
            (y_psnr - expected_y).abs() < 0.05,
            "PSNR_Y={y_psnr:.4} expected={expected_y:.4}"
        );
    }

    /// Every plane offset by exactly Δ=8 → each plane MSE = 8² = 64, so each
    /// plane PSNR = 10·log10(255²/64) = 20·log10(255/8).  Because the weighted
    /// score's weights sum to 1 (4/6·Y + 1/6·Cb + 1/6·Cr) and all three plane
    /// PSNRs are equal, the overall score collapses to that same value.
    ///
    /// This closes the gap left by the other PSNR golden tests, which offset
    /// only the Y plane and therefore leave Cb/Cr pinned at the identical-plane
    /// 100.0 cap — meaning the chroma branch of the weighted sum is never
    /// exercised against a real, non-degenerate plane MSE.
    #[test]
    fn golden_psnr_all_planes_offset_exact_20log10() {
        let calc = PsnrCalculator::new();
        // f1: Y=100, Cb=120, Cr=120 ; f2: every plane +8.
        let f1 = uniform_yuv420(64, 64, 100, 120, 120);
        let f2 = uniform_yuv420(64, 64, 108, 128, 128);

        let result = calc.calculate(&f1, &f2).expect("should succeed");

        // 10·log10(255²/64) == 20·log10(255/8) ≈ 30.0690
        let expected = 20.0 * (255.0_f64 / 8.0).log10();
        assert!(
            (result.score - expected).abs() < 0.02,
            "weighted PSNR={:.4} expected={:.4}",
            result.score,
            expected
        );

        // Each individual plane must independently equal the same value
        // (none of them hit the identical-plane 100.0 cap).
        for plane in ["Y", "Cb", "Cr"] {
            let p = *result.components.get(plane).unwrap_or(&0.0);
            assert!(
                (p - expected).abs() < 0.02,
                "PSNR_{plane}={p:.4} expected={expected:.4} (must not be capped at 100.0)"
            );
        }
    }

    // ── SSIM golden tests ─────────────────────────────────────────────────

    #[test]
    fn golden_ssim_identical_frames() {
        let calc = SsimCalculator::new();
        let f = grey_yuv420(64, 64, 128);
        let result = calc.calculate(&f, &f).expect("should succeed");
        assert!(
            (result.score - 1.0).abs() < 0.02,
            "SSIM of identical frames must be ~1.0, got {}",
            result.score
        );
    }

    #[test]
    fn golden_ssim_uniform_frames_analytical() {
        let calc = SsimCalculator::new();
        let f1 = grey_yuv420(64, 64, 100);
        let f2 = grey_yuv420(64, 64, 200);

        let result = calc.calculate(&f1, &f2).expect("should succeed");
        let y_ssim = *result.components.get("Y").unwrap_or(&0.0);

        let analytical = expected_ssim_uniform(100.0, 200.0);
        assert!(
            (y_ssim - analytical).abs() < 0.02,
            "SSIM_Y={y_ssim:.6} analytical={analytical:.6}"
        );
    }

    #[test]
    fn golden_ssim_max_difference_low() {
        let calc = SsimCalculator::new();
        let f1 = grey_yuv420(64, 64, 0);
        let f2 = grey_yuv420(64, 64, 255);
        let result = calc.calculate(&f1, &f2).expect("should succeed");
        // For max contrast difference, SSIM must be clearly below 0.5
        assert!(
            result.score < 0.5,
            "Max-difference SSIM must be < 0.5, got {}",
            result.score
        );
    }

    // ── VIF golden tests ──────────────────────────────────────────────────

    #[test]
    fn golden_vif_identical_frames_returns_one() {
        let calc = VifCalculator::new();
        let f = grey_yuv420(64, 64, 128);
        let result = calc.calculate(&f, &f).expect("should succeed");
        assert!(
            (result.score - 1.0).abs() < 0.05,
            "VIF of identical frames must be ~1.0, got {}",
            result.score
        );
    }

    #[test]
    fn golden_vif_distorted_less_than_identical() {
        let calc = VifCalculator::new();
        let ref_frame = grey_yuv420(64, 64, 128);
        let identical = grey_yuv420(64, 64, 128);
        let distorted = grey_yuv420(64, 64, 90);

        let vif_identical = calc
            .calculate(&ref_frame, &identical)
            .expect("should succeed")
            .score;
        let vif_distorted = calc
            .calculate(&ref_frame, &distorted)
            .expect("should succeed")
            .score;

        assert!(
            vif_identical >= vif_distorted,
            "identical VIF={vif_identical} must be >= distorted VIF={vif_distorted}"
        );
    }

    // ── VMAF golden tests ─────────────────────────────────────────────────

    #[test]
    fn golden_vmaf_in_range() {
        let calc = VmafCalculator::new();
        let f1 = grey_yuv420(128, 128, 128);
        let f2 = grey_yuv420(128, 128, 128);
        let result = calc.calculate(&f1, &f2).expect("should succeed");
        assert!(
            result.score >= 0.0 && result.score <= 100.0,
            "VMAF must be in [0, 100], got {}",
            result.score
        );
    }

    #[test]
    fn golden_vmaf_identical_frames_high() {
        let calc = VmafCalculator::new();
        let f = grey_yuv420(128, 128, 128);
        let result = calc.calculate(&f, &f).expect("should succeed");
        // Identical frames should yield a high VMAF
        assert!(
            result.score >= 50.0,
            "VMAF of identical frames must be >= 50, got {}",
            result.score
        );
    }

    #[test]
    fn golden_vmaf_components_present() {
        let calc = VmafCalculator::new();
        let f1 = grey_yuv420(64, 64, 100);
        let f2 = grey_yuv420(64, 64, 120);
        let result = calc.calculate(&f1, &f2).expect("should succeed");
        assert!(result.components.contains_key("VIF"));
        assert!(result.components.contains_key("DLM"));
        assert!(result.components.contains_key("Motion"));
    }

    // ── Region PSNR / SSIM golden tests ───────────────────────────────────

    #[test]
    fn golden_region_psnr_matches_full_for_uniform() {
        // For a uniform frame, PSNR over any region should equal PSNR over full frame.
        let calc = PsnrCalculator::new();
        let f1 = grey_yuv420(64, 64, 100);
        let f2 = grey_yuv420(64, 64, 110);

        let full = calc.calculate(&f1, &f2).expect("full frame");
        let region = calc
            .calculate_region(&f1, &f2, (8, 8, 32, 32))
            .expect("region");

        // For uniform frames the Y-plane PSNR should be identical regardless of region
        let full_y = *full.components.get("Y").unwrap_or(&0.0);
        let region_y = *region.components.get("Y").unwrap_or(&0.0);
        assert!(
            (full_y - region_y).abs() < 0.1,
            "full_Y={full_y:.4} region_Y={region_y:.4}"
        );
    }

    #[test]
    fn golden_region_ssim_identical_region_is_one() {
        let calc = SsimCalculator::new();
        let f = grey_yuv420(64, 64, 128);
        let result = calc
            .calculate_region(&f, &f, (4, 4, 32, 32))
            .expect("should succeed");
        assert!(
            (result.score - 1.0).abs() < 0.02,
            "SSIM of identical region must be ~1.0, got {}",
            result.score
        );
    }

    // ── Helper: varied (gradient) frame ──────────────────────────────────────

    /// Build a YUV420 frame whose Y plane is a diagonal gradient.
    ///
    /// pixel\[y\]\[x\] = ((x + y) * 2) % 240  →  values in \[0, 238\].
    /// This avoids the flat-plane degenerate case where all MSEs are 0.
    fn make_varied_frame(width: usize, height: usize) -> Frame {
        let mut f =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        for row in 0..height {
            for col in 0..width {
                f.planes[0][row * width + col] = (((col + row) * 2) % 240) as u8;
            }
        }
        f.planes[1].fill(128);
        f.planes[2].fill(128);
        f
    }

    /// Add pseudo-random noise of the given amplitude to the Y plane.
    ///
    /// Uses an LCG hash so noise is deterministic and cheap.
    fn add_noise_to_frame(frame: &Frame, amplitude: u8) -> Frame {
        let mut noisy = frame.clone();
        let amp = u64::from(amplitude.max(1));
        for (i, pixel) in noisy.planes[0].iter_mut().enumerate() {
            let hash = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (hash & 0xFF) % amp;
            *pixel = pixel.saturating_add(noise as u8);
        }
        noisy
    }

    // ── Monotonicity tests ────────────────────────────────────────────────────

    /// PSNR must decrease (or stay equal) as noise amplitude increases.
    #[test]
    fn test_psnr_monotone_with_increasing_distortion() {
        let reference = make_varied_frame(64, 64);
        let calc = PsnrCalculator::new();
        let noise_levels: &[u8] = &[0, 10, 30, 80];
        let scores: Vec<f64> = noise_levels
            .iter()
            .map(|&n| {
                let distorted = add_noise_to_frame(&reference, n);
                calc.calculate(&reference, &distorted)
                    .expect("psnr calc ok")
                    .score
            })
            .collect();
        for window in scores.windows(2) {
            assert!(
                window[0] >= window[1],
                "PSNR must decrease with more distortion: {:.4} >= {:.4}",
                window[0],
                window[1]
            );
        }
    }

    /// SSIM must decrease (or stay equal) as noise amplitude increases.
    #[test]
    fn test_ssim_monotone_with_increasing_distortion() {
        let reference = make_varied_frame(64, 64);
        let calc = SsimCalculator::new();
        let noise_levels: &[u8] = &[0, 10, 30, 80];
        let scores: Vec<f64> = noise_levels
            .iter()
            .map(|&n| {
                let distorted = add_noise_to_frame(&reference, n);
                calc.calculate(&reference, &distorted)
                    .expect("ssim calc ok")
                    .score
            })
            .collect();
        for window in scores.windows(2) {
            assert!(
                window[0] >= window[1] - 1e-9,
                "SSIM must not increase with more distortion: {:.6} >= {:.6}",
                window[0],
                window[1]
            );
        }
    }

    // ── VIF golden-reference tests ────────────────────────────────────────────

    /// VIF of perfectly identical frames must be close to 1.0.
    #[test]
    fn test_vif_identical_frames_near_one() {
        let calc = VifCalculator::new();
        let f = make_varied_frame(64, 64);
        let result = calc.calculate(&f, &f).expect("vif calc ok");
        assert!(
            (result.score - 1.0).abs() < 0.1,
            "VIF of identical frames must be ≈1.0, got {}",
            result.score
        );
    }

    /// VIF of a heavily-noised frame must be no larger than VIF of identical frames.
    #[test]
    fn test_vif_white_noise_lower_than_identical() {
        let calc = VifCalculator::new();
        let reference = make_varied_frame(64, 64);
        let noisy = add_noise_to_frame(&reference, 80);
        let vif_identical = calc
            .calculate(&reference, &reference)
            .expect("vif identical ok")
            .score;
        let vif_noisy = calc
            .calculate(&reference, &noisy)
            .expect("vif noisy ok")
            .score;
        assert!(
            vif_identical >= vif_noisy,
            "VIF identical ({:.4}) must be >= VIF noisy ({:.4})",
            vif_identical,
            vif_noisy
        );
    }

    // ── FSIM golden-reference tests ───────────────────────────────────────────

    /// FSIM of identical frames must be close to 1.0.
    #[test]
    fn test_fsim_identical_frames_maximum() {
        let calc = FsimCalculator::new();
        let f = make_varied_frame(64, 64);
        let result = calc.calculate(&f, &f).expect("fsim calc ok");
        assert!(
            result.score >= 0.95,
            "FSIM of identical frames must be near 1.0, got {}",
            result.score
        );
    }

    /// FSIM of distorted frames must be lower than (or equal to) identical frames.
    #[test]
    fn test_fsim_distorted_lower_than_identical() {
        let calc = FsimCalculator::new();
        let reference = make_varied_frame(64, 64);
        let noisy = add_noise_to_frame(&reference, 60);
        let fsim_identical = calc
            .calculate(&reference, &reference)
            .expect("fsim identical ok")
            .score;
        let fsim_noisy = calc
            .calculate(&reference, &noisy)
            .expect("fsim noisy ok")
            .score;
        assert!(
            fsim_identical >= fsim_noisy,
            "FSIM identical ({:.4}) must be >= FSIM noisy ({:.4})",
            fsim_identical,
            fsim_noisy
        );
    }

    // ── MS-SSIM golden-reference tests ───────────────────────────────────────

    /// MS-SSIM of identical frames must be close to 1.0.
    #[test]
    fn test_mssim_identical_frames_near_one() {
        let calc = MsSsimCalculator::new();
        // MS-SSIM needs room to downsample across 5 scales
        let f = make_varied_frame(128, 128);
        let result = calc.calculate(&f, &f).expect("mssim calc ok");
        assert!(
            (result.score - 1.0).abs() < 0.05,
            "MS-SSIM of identical frames must be ≈1.0, got {}",
            result.score
        );
    }

    /// MS-SSIM must decrease as distortion increases.
    #[test]
    fn test_mssim_monotone_with_increasing_distortion() {
        let reference = make_varied_frame(128, 128);
        let calc = MsSsimCalculator::new();
        let noise_levels: &[u8] = &[0, 15, 50];
        let scores: Vec<f64> = noise_levels
            .iter()
            .map(|&n| {
                let distorted = add_noise_to_frame(&reference, n);
                calc.calculate(&reference, &distorted)
                    .expect("mssim calc ok")
                    .score
            })
            .collect();
        for window in scores.windows(2) {
            assert!(
                window[0] >= window[1] - 1e-9,
                "MS-SSIM must not increase with more distortion: {:.6} >= {:.6}",
                window[0],
                window[1]
            );
        }
    }

    // ── PoolingMethod edge-case tests ─────────────────────────────────────────

    /// All pooling methods must handle a single-element slice correctly.
    #[test]
    fn test_pooling_method_single_score() {
        let v = 42.0_f64;
        assert!(
            (PoolingMethod::Mean.apply(&[v]) - v).abs() < 1e-9,
            "Mean of [42.0] must be 42.0"
        );
        assert!(
            (PoolingMethod::HarmonicMean.apply(&[v]) - v).abs() < 1e-9,
            "HarmonicMean of [42.0] must be 42.0"
        );
        assert!(
            (PoolingMethod::Min.apply(&[v]) - v).abs() < 1e-9,
            "Min of [42.0] must be 42.0"
        );
        assert!(
            (PoolingMethod::Percentile(50).apply(&[v]) - v).abs() < 1e-9,
            "Percentile(50) of [42.0] must be 42.0"
        );
    }

    /// Pooling 100 identical values must return that value.
    #[test]
    fn test_pooling_method_all_identical() {
        let scores = vec![7.5_f64; 100];
        assert!(
            (PoolingMethod::Mean.apply(&scores) - 7.5).abs() < 1e-9,
            "Mean of 100×7.5 must be 7.5"
        );
        assert!(
            (PoolingMethod::HarmonicMean.apply(&scores) - 7.5).abs() < 1e-9,
            "HarmonicMean of 100×7.5 must be 7.5"
        );
        assert!(
            (PoolingMethod::Min.apply(&scores) - 7.5).abs() < 1e-9,
            "Min of 100×7.5 must be 7.5"
        );
    }

    /// All pooling methods must not panic on an empty slice.
    #[test]
    fn test_pooling_method_empty_does_not_panic() {
        let _ = PoolingMethod::Mean.apply(&[]);
        let _ = PoolingMethod::HarmonicMean.apply(&[]);
        let _ = PoolingMethod::Min.apply(&[]);
        let _ = PoolingMethod::Percentile(50).apply(&[]);
    }

    /// Percentile(50) on a sorted list must return a value near the median.
    #[test]
    fn test_pooling_percentile_50_near_median() {
        let scores = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let result = PoolingMethod::Percentile(50).apply(&scores);
        // floor(0.5 × 5) = 2 → sorted[2] = 3.0
        assert!(
            (result - 3.0).abs() < 1.5,
            "Percentile(50) of [1..5] must be near median 3.0, got {result}"
        );
    }

    /// Percentile(0) returns the minimum; Percentile(100) returns a high value.
    #[test]
    fn test_pooling_percentile_extremes() {
        let scores = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
        let p0 = PoolingMethod::Percentile(0).apply(&scores);
        assert!(
            (p0 - 10.0).abs() < 1e-9,
            "Percentile(0) must be the minimum, got {p0}"
        );
        let p100 = PoolingMethod::Percentile(100).apply(&scores);
        assert!(
            p100 >= 40.0,
            "Percentile(100) must be at the high end, got {p100}"
        );
    }

    // ── High-value frame tests ────────────────────────────────────────────────

    /// Metrics must not panic when Y-plane pixels are at maximum (255).
    #[test]
    fn test_metric_max_value_frame_no_panic() {
        let mut f1 = Frame::new(32, 32, PixelFormat::Yuv420p).expect("should succeed in test");
        f1.planes[0].fill(255);
        f1.planes[1].fill(128);
        f1.planes[2].fill(128);
        let mut f2 = f1.clone();
        f2.planes[0]
            .iter_mut()
            .for_each(|p| *p = p.saturating_sub(20));

        let psnr_result = PsnrCalculator::new().calculate(&f1, &f2);
        assert!(
            psnr_result.is_ok(),
            "PSNR must handle max-value frames, got: {:?}",
            psnr_result.err()
        );
        let ssim_result = SsimCalculator::new().calculate(&f1, &f2);
        assert!(
            ssim_result.is_ok(),
            "SSIM must handle max-value frames, got: {:?}",
            ssim_result.err()
        );
    }

    /// Metrics must not panic when Y-plane pixels are all zero (dark frame).
    #[test]
    fn test_metric_zero_value_frame_no_panic() {
        let f1 = Frame::new(32, 32, PixelFormat::Yuv420p).expect("should succeed in test");
        let mut f2 = f1.clone();
        f2.planes[0].fill(10);
        let result = PsnrCalculator::new().calculate(&f1, &f2);
        assert!(result.is_ok(), "PSNR must handle all-zero frames");
    }

    /// PSNR result must be finite and positive for near-identical high-value frames.
    #[test]
    fn test_psnr_high_value_frame_finite() {
        let mut f1 = Frame::new(64, 64, PixelFormat::Yuv420p).expect("should succeed in test");
        f1.planes[0].fill(240);
        let mut f2 = f1.clone();
        f2.planes[0].fill(230);
        let result = PsnrCalculator::new()
            .calculate(&f1, &f2)
            .expect("should succeed");
        assert!(
            result.score.is_finite() && result.score > 0.0,
            "PSNR for near-identical high-value frames must be finite and positive, got {}",
            result.score
        );
    }
}
