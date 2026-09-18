use oximedia_compat_cv2::{imread, imwrite, Mat, MatType, IMREAD_COLOR, IMREAD_GRAYSCALE};
use std::env::temp_dir;

#[test]
fn test_png_round_trip_dimensions_and_content() {
    let tmp = temp_dir().join("oximedia_cv2_rt_test.png");
    // 2×2 image with distinct BGR values
    let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let mat_orig = Mat::from_bgr_bytes(data, 2, 2);
    imwrite(&tmp, &mat_orig).expect("imwrite PNG failed");
    let mat_loaded = imread(&tmp, IMREAD_COLOR).expect("imread PNG failed");
    assert_eq!(mat_loaded.rows, 2);
    assert_eq!(mat_loaded.cols, 2);
    assert_eq!(mat_loaded.mat_type, MatType::CV_8UC3);
    // PNG is lossless — pixel values must be preserved
    assert_eq!(mat_loaded.at_8u3(0, 0), mat_orig.at_8u3(0, 0));
    assert_eq!(mat_loaded.at_8u3(1, 1), mat_orig.at_8u3(1, 1));
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_imread_grayscale_produces_1ch() {
    let tmp = temp_dir().join("oximedia_cv2_gray_test.png");
    let mat = Mat::new_8uc3(4, 4);
    imwrite(&tmp, &mat).expect("imwrite failed");
    let gray = imread(&tmp, IMREAD_GRAYSCALE).expect("imread GRAYSCALE failed");
    assert_eq!(gray.rows, 4);
    assert_eq!(gray.cols, 4);
    assert_eq!(gray.mat_type, MatType::CV_8UC1);
    assert_eq!(gray.channels(), 1);
    let _ = std::fs::remove_file(&tmp);
}
