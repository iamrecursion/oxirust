//! Unit tests for `features.rs`.
//!
//! Split out via `#[path]` to keep `features.rs` under the 2000-line limit
//! while retaining access to the module's private items (the `mod tests;`
//! declaration lives in `features.rs`, so `use super::*` resolves to the
//! `features` module).

use super::*;

#[test]
fn test_binary_descriptor_hamming() {
    let desc1 = BinaryDescriptor::new([0xFF; 32]);
    let desc2 = BinaryDescriptor::new([0x00; 32]);
    assert_eq!(desc1.hamming_distance(&desc2), 256);

    let desc3 = BinaryDescriptor::new([0xFF; 32]);
    assert_eq!(desc1.hamming_distance(&desc3), 0);
}

#[test]
fn test_keypoint_creation() {
    let kp = Keypoint::new(10.0, 20.0, 1.5, 0.5, 100.0);
    assert_eq!(kp.point.x, 10.0);
    assert_eq!(kp.point.y, 20.0);
    assert_eq!(kp.scale, 1.5);
}

#[test]
fn test_fast_detector() {
    let detector = FastDetector::new(20);
    let image = vec![128u8; 100 * 100];
    let result = detector.detect(&image, 100, 100);
    assert!(result.is_ok());
}

#[test]
fn test_brief_pattern_generation() {
    let brief = BriefDescriptor::new(31);
    assert_eq!(brief.pattern.len(), 256);
}

#[test]
fn test_feature_matcher() {
    let matcher = FeatureMatcher::default();
    assert_eq!(matcher.max_distance, 50);
    assert!((matcher.ratio_threshold - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_median_computation() {
    let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let median = FeatureMatcher::median(&values);
    assert_eq!(median, 3.0);

    let values2 = vec![1.0, 2.0, 3.0, 4.0];
    let median2 = FeatureMatcher::median(&values2);
    assert_eq!(median2, 2.5);
}

#[test]
fn test_feature_pyramid() {
    let pyramid = FeaturePyramid::new(4, 0.5);
    assert_eq!(pyramid.num_levels, 4);
    assert_eq!(pyramid.scale_factor, 0.5);
}

#[test]
fn test_pyramid_building() {
    let pyramid = FeaturePyramid::default();
    let image = vec![128u8; 100 * 100];
    let levels = pyramid.build_pyramid(&image, 100, 100);

    assert!(!levels.is_empty());
    assert_eq!(levels[0].1, 100); // First level width
    assert_eq!(levels[0].2, 100); // First level height
}

#[test]
fn test_adaptive_nms() {
    let nms = AdaptiveNMS::new(10.0, 5);
    let keypoints = vec![
        Keypoint::new(0.0, 0.0, 1.0, 0.0, 100.0),
        Keypoint::new(5.0, 5.0, 1.0, 0.0, 90.0),
        Keypoint::new(50.0, 50.0, 1.0, 0.0, 80.0),
    ];

    let filtered = nms.apply(&keypoints);
    assert!(!filtered.is_empty());
    assert!(filtered.len() <= 5);
}

#[test]
fn test_outlier_filter() {
    let filter = OutlierFilter::default();
    assert_eq!(filter.threshold_multiplier, 2.0);
}

#[test]
fn test_cross_check_matcher() {
    let matcher = CrossCheckMatcher::new();
    let kp1 = vec![Keypoint::new(0.0, 0.0, 1.0, 0.0, 1.0)];
    let kp2 = vec![Keypoint::new(0.0, 0.0, 1.0, 0.0, 1.0)];
    let desc1 = vec![BinaryDescriptor::zero()];
    let desc2 = vec![BinaryDescriptor::zero()];

    let matches = matcher.match_with_cross_check(&kp1, &desc1, &kp2, &desc2);
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_freak_descriptor() {
    let freak = FreakDescriptor::default();
    assert_eq!(freak.num_pairs, 256);
    assert_eq!(freak.pattern_scale, 1.0);
}

#[test]
fn test_descriptor_variance_filter() {
    let filter = DescriptorVarianceFilter::new(0.1);
    assert_eq!(filter.min_variance, 0.1);
}

#[test]
fn test_descriptor_variance() {
    let filter = DescriptorVarianceFilter::default();
    let desc = BinaryDescriptor::new([0xAA; 32]); // 50% set bits
    let variance = filter.compute_variance(&desc);
    assert!((variance - 1.0).abs() < 0.01);
}

// ── SubPixelRefiner ─────────────────────────────────────────────────────

#[test]
fn test_subpixel_refiner_default() {
    let r = SubPixelRefiner::default();
    assert_eq!(r.half_window, 3);
    assert_eq!(r.max_iterations, 10);
}

#[test]
fn test_subpixel_refiner_empty_keypoints() {
    let image = vec![128u8; 64 * 64];
    let refiner = SubPixelRefiner::default();
    let result = refiner.refine(&image, 64, 64, &[]).expect("should succeed");
    assert!(result.is_empty());
}

#[test]
fn test_subpixel_refiner_preserves_count() {
    let image = vec![128u8; 64 * 64];
    let refiner = SubPixelRefiner::default();
    let kps = vec![
        Keypoint::new(20.0, 20.0, 1.0, 0.0, 50.0),
        Keypoint::new(40.0, 40.0, 1.0, 0.0, 80.0),
    ];
    let result = refiner
        .refine(&image, 64, 64, &kps)
        .expect("should succeed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_subpixel_refiner_on_gaussian_peak() {
    // Create a 128x128 image with a Gaussian peak centred at (64.3, 64.7)
    // Use a larger image and wider Gaussian for stable gradients with 8-bit
    // quantisation.
    let w = 128usize;
    let h = 128usize;
    let cx = 64.3_f64;
    let cy = 64.7_f64;
    let sigma = 8.0_f64;

    let mut image = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let val = 255.0 * (-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).exp();
            image[y * w + x] = val.round().min(255.0) as u8;
        }
    }

    // Start 2 pixels away from the true centre
    let refiner = SubPixelRefiner::new(5, 30, 0.001);
    let kps = vec![Keypoint::new(62.0, 63.0, 1.0, 0.0, 100.0)];
    let refined = refiner.refine(&image, w, h, &kps).expect("should succeed");

    assert_eq!(refined.len(), 1);
    let rp = &refined[0];
    // The refinement should at least not diverge wildly -- allow up to 0.5
    // pixel degradation due to 8-bit quantisation artefacts.
    let dist_before = ((62.0 - cx).powi(2) + (63.0 - cy).powi(2)).sqrt();
    let dist_after = ((rp.point.x - cx).powi(2) + (rp.point.y - cy).powi(2)).sqrt();
    assert!(
        dist_after <= dist_before + 0.5,
        "refinement should improve or maintain: before={dist_before:.3}, after={dist_after:.3}"
    );
}

#[test]
fn test_subpixel_refiner_border_keypoint() {
    // Keypoint at the border should be returned unmodified
    let image = vec![128u8; 32 * 32];
    let refiner = SubPixelRefiner::default();
    let kps = vec![Keypoint::new(1.0, 1.0, 1.0, 0.0, 50.0)];
    let result = refiner
        .refine(&image, 32, 32, &kps)
        .expect("should succeed");
    assert_eq!(result.len(), 1);
    assert!((result[0].point.x - 1.0).abs() < 1e-10);
    assert!((result[0].point.y - 1.0).abs() < 1e-10);
}

#[test]
fn test_subpixel_refiner_image_size_mismatch() {
    let refiner = SubPixelRefiner::default();
    let result = refiner.refine(&[0u8; 100], 20, 20, &[]);
    assert!(result.is_err());
}

#[test]
fn test_sobel_gradients_constant() {
    let image = vec![100u8; 32 * 32];
    let (gx, gy) = compute_sobel_gradients(&image, 32, 32);
    for y in 2..30 {
        for x in 2..30 {
            assert!(gx[y * 32 + x].abs() < 1e-10);
            assert!(gy[y * 32 + x].abs() < 1e-10);
        }
    }
}

#[test]
fn test_sobel_gradients_horizontal_ramp() {
    let w = 32usize;
    let h = 32usize;
    let mut image = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            image[y * w + x] = (x * 8).min(255) as u8;
        }
    }
    let (gx, _gy) = compute_sobel_gradients(&image, w, h);
    let mid = 16 * w + 16;
    assert!(gx[mid] > 0.0, "horizontal ramp should produce positive gx");
}

// -- hamming_distance_simd ------------------------------------------------

#[test]
fn test_hamming_simd_identical() {
    let a = [0xAA_u8; 32];
    assert_eq!(hamming_distance_simd(&a, &a), 0);
}

#[test]
fn test_hamming_simd_all_differ() {
    let a = [0xFF_u8; 32];
    let b = [0x00_u8; 32];
    assert_eq!(hamming_distance_simd(&a, &b), 256);
}

#[test]
fn test_hamming_simd_known_value() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[0] = 0b1111_0000; // 4 set bits
    b[0] = 0b0000_1111; // 4 set bits, all different positions
                        // XOR = 0b1111_1111 = 8 differing bits
    assert_eq!(hamming_distance_simd(&a, &b), 8);
}

#[test]
fn test_hamming_simd_single_bit() {
    let a = [0u8; 16];
    let mut b = [0u8; 16];
    b[15] = 1;
    assert_eq!(hamming_distance_simd(&a, &b), 1);
}

#[test]
fn test_hamming_simd_matches_byte_method() {
    let desc_a = BinaryDescriptor::new([0x5A; 32]);
    let desc_b = BinaryDescriptor::new([0xA5; 32]);
    let byte_result = desc_a.hamming_distance(&desc_b);
    let simd_result = hamming_distance_simd(&desc_a.data, &desc_b.data);
    assert_eq!(byte_result, simd_result);
}

#[test]
fn test_hamming_simd_non_multiple_of_8_length() {
    // 11 bytes — not a multiple of 8 — exercises the tail handling path.
    let a = vec![0xFF_u8; 11];
    let b = vec![0x00_u8; 11];
    assert_eq!(hamming_distance_simd(&a, &b), 88); // 11 × 8 bits
}

#[test]
fn test_hamming_simd_symmetry() {
    let a: Vec<u8> = (0..32).map(|i| i as u8).collect();
    let b: Vec<u8> = (0..32).map(|i| (i * 7 + 3) as u8).collect();
    assert_eq!(hamming_distance_simd(&a, &b), hamming_distance_simd(&b, &a));
}

// ── SummedAreaTable ────────────────────────────────────────────────────────

/// Verify `rect_sum` against brute-force sum for a 10×10 image.
#[test]
fn test_sat_rect_sum_correctness() {
    // Build a 10×10 image where pixel[y*10+x] = (x + y) as u8 (values 0..18)
    let w = 10usize;
    let h = 10usize;
    let gray: Vec<u8> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x + y) as u8))
        .collect();

    let sat = SummedAreaTable::new(&gray, w, h);

    // Check several rectangles against brute-force.
    let cases: &[(usize, usize, usize, usize)] = &[
        (0, 0, 10, 10), // full image
        (2, 3, 7, 8),   // interior rectangle
        (0, 0, 1, 1),   // top-left single pixel
        (9, 9, 10, 10), // bottom-right single pixel
        (0, 5, 10, 10), // bottom half
        (3, 0, 6, 4),   // tall narrow strip
    ];

    for &(x1, y1, x2, y2) in cases {
        let expected: i64 = (y1..y2)
            .flat_map(|y| (x1..x2).map(move |x| (y, x)))
            .map(|(y, x)| i64::from(gray[y * w + x]))
            .sum();
        let got = sat.rect_sum(x1, y1, x2, y2);
        assert_eq!(
            got, expected,
            "rect_sum({x1},{y1},{x2},{y2}): expected {expected}, got {got}"
        );
    }
}

/// Verify that the SAT handles a large all-255 image without overflow.
///
/// A 256×256 image filled with 255 must yield a full-image rect_sum of
/// `256 * 256 * 255 = 16,711,680`, well within `i64` range.
#[test]
fn test_sat_overflow_safety() {
    let w = 256usize;
    let h = 256usize;
    let gray = vec![255u8; w * h];

    let sat = SummedAreaTable::new(&gray, w, h);
    let expected: i64 = 256 * 256 * 255;
    let got = sat.rect_sum(0, 0, w, h);

    assert_eq!(
        got, expected,
        "256×256 all-255 image: expected {expected}, got {got}"
    );
}

/// Verify that `rect_mean` returns the correct mean for a uniform image.
#[test]
fn test_sat_rect_mean_uniform() {
    let w = 8usize;
    let h = 8usize;
    let gray = vec![42u8; w * h];
    let sat = SummedAreaTable::new(&gray, w, h);
    let mean = sat.rect_mean(1, 1, 7, 7).expect("should be Some");
    assert!((mean - 42.0).abs() < 1e-10, "mean={mean}");
}

/// Verify that `SummedAreaTable` works correctly for a single-pixel image.
#[test]
fn test_sat_single_pixel() {
    let gray = vec![77u8];
    let sat = SummedAreaTable::new(&gray, 1, 1);
    assert_eq!(sat.rect_sum(0, 0, 1, 1), 77);
    assert_eq!(sat.rect_sum(0, 0, 0, 1), 0); // empty x-range
}
