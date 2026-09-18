//! Integration tests for `ImagePreprocessor` correctness.

use oximedia_ml::{ImagePreprocessor, InputRange, PixelLayout, TensorLayout};

mod fixtures;

#[test]
fn nchw_default_layout_matches_builder_contract() {
    let p = ImagePreprocessor::new(2, 2);
    assert_eq!(p.batch_shape(), vec![1, 3, 2, 2]);
}

#[test]
fn nhwc_layout_flips_axes() {
    let p = ImagePreprocessor::new(4, 8).with_tensor_layout(TensorLayout::Nhwc);
    assert_eq!(p.batch_shape(), vec![1, 8, 4, 3]);
}

#[test]
fn imagenet_normalization_on_white_pixel() {
    let p = ImagePreprocessor::new(1, 1).with_imagenet_normalization();
    let pixels = vec![255u8, 255u8, 255u8];
    let out = p.process_u8_rgb(&pixels, 1, 1).expect("ok");
    // (1.0 - mean) / std for each channel
    assert!((out[0] - ((1.0 - 0.485) / 0.229)).abs() < 1e-4);
    assert!((out[1] - ((1.0 - 0.456) / 0.224)).abs() < 1e-4);
    assert!((out[2] - ((1.0 - 0.406) / 0.225)).abs() < 1e-4);
}

#[test]
fn bgr_pixel_layout_swaps_channels() {
    let p = ImagePreprocessor::new(1, 1).with_pixel_layout(PixelLayout::Bgr);
    // BGR source: B=10, G=20, R=30 in memory order.
    let pixels = vec![10u8, 20u8, 30u8];
    let out = p.process_u8_rgb(&pixels, 1, 1).expect("ok");
    // After swap and U8→f32/255 (no mean/std), channels should be R=30/255, G=20/255, B=10/255.
    assert!((out[0] - 30.0 / 255.0).abs() < 1e-4);
    assert!((out[1] - 20.0 / 255.0).abs() < 1e-4);
    assert!((out[2] - 10.0 / 255.0).abs() < 1e-4);
}

#[test]
fn tiny_image_scales_without_error() {
    let (pixels, w, h) = fixtures::tiny_rgb_image();
    let p = ImagePreprocessor::new(4, 4);
    let out = p.process_u8_rgb(&pixels, w, h).expect("ok");
    assert_eq!(out.len(), 4 * 4 * 3);
}

#[test]
fn buffer_mismatch_is_an_error() {
    let p = ImagePreprocessor::new(4, 4);
    assert!(p.process_u8_rgb(&[0u8; 10], 2, 2).is_err());
}

#[test]
fn unit_float_range_skips_division() {
    let p = ImagePreprocessor::new(1, 1)
        .with_input_range(InputRange::UnitFloat)
        .with_mean([0.0, 0.0, 0.0])
        .with_std([1.0, 1.0, 1.0]);
    let pixels = vec![1u8, 2u8, 3u8];
    let out = p.process_u8_rgb(&pixels, 1, 1).expect("ok");
    // Values pass through u8 → f32 cast (no /255).
    assert!((out[0] - 1.0).abs() < 1e-4);
    assert!((out[1] - 2.0).abs() < 1e-4);
    assert!((out[2] - 3.0).abs() < 1e-4);
}
