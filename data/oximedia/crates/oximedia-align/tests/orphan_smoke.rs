//! Smoke tests for newly-wired orphan modules in oximedia-align.

#[test]
fn test_bundle_adjust_config() {
    use oximedia_align::bundle_adjust::BundleAdjustConfig;
    let config = BundleAdjustConfig::default();
    assert!(config.max_iterations > 0);
}

#[test]
fn test_illumination_invariant_descriptor() {
    use oximedia_align::illumination_invariant::{
        IlluminationInvariantConfig, IlluminationInvariantDescriptor,
    };
    let config = IlluminationInvariantConfig::default();
    let desc = IlluminationInvariantDescriptor::new(config);
    let _ = desc;
}

#[test]
fn test_image_stitch_config_default() {
    use oximedia_align::image_stitch::StitchConfig;
    let c = StitchConfig::default();
    assert!(c.max_features > 0);
}

#[test]
fn test_integral_image_build() {
    use oximedia_align::integral_image::IntegralImage;
    let pixels = vec![1u8; 4 * 4];
    let ii = IntegralImage::build(&pixels, 4, 4);
    // sum of entire 4x4 image = 16
    let total = ii.box_sum(0, 0, 4, 4);
    assert_eq!(total, 16);
}

#[test]
fn test_lens_calibration_checkerboard_detector() {
    use oximedia_align::lens_calibration::CheckerboardDetector;
    let _det = CheckerboardDetector::new(1.0);
}

#[test]
fn test_motion_interp_new() {
    use oximedia_align::motion_interp::MotionInterpolator;
    let _ = MotionInterpolator::new();
}

#[test]
fn test_parallel_ransac_construct() {
    use oximedia_align::parallel_ransac::{
        ParallelModelType, ParallelRansac, ParallelRansacConfig,
    };
    let config = ParallelRansacConfig::default();
    let _estimator = ParallelRansac::new(config, ParallelModelType::Homography);
}

#[test]
fn test_simd_hamming_zero_distance() {
    use oximedia_align::simd_hamming::hamming_u64_batch;
    let a = [0u8; 32];
    let b = [0u8; 32];
    assert_eq!(hamming_u64_batch(&a, &b), 0);
}
