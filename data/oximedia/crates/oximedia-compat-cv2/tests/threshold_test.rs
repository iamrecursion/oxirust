use oximedia_compat_cv2::{threshold::threshold, Mat, THRESH_BINARY, THRESH_OTSU};

#[test]
fn test_threshold_binary() {
    let data: Vec<u8> = (0..=9).map(|i: u8| i * 25).collect();
    let mat = Mat::from_gray_bytes(data, 1, 10);
    let (_t, binary) = threshold(&mat, 127.0, 255.0, THRESH_BINARY).expect("threshold binary");
    // pixel 5 = 5*25 = 125, which is NOT > 127 → 0
    assert_eq!(binary.at_8u1(0, 5), 0);
    // pixel 6 = 6*25 = 150, which is > 127 → 255
    assert_eq!(binary.at_8u1(0, 6), 255);
}

#[test]
fn test_otsu_bimodal() {
    // Two clusters: 20 pixels at value 50, 20 pixels at value 200
    let mut data = vec![50u8; 20];
    data.extend(vec![200u8; 20]);
    let mat = Mat::from_gray_bytes(data, 1, 40);
    let (t, _) = threshold(&mat, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU).expect("otsu threshold");
    // Otsu threshold should fall between 50 and 200
    assert!(
        t > 50.0 && t < 200.0,
        "Otsu threshold {t} should be between 50 and 200"
    );
}

#[test]
fn test_threshold_returns_threshold_value() {
    let data = vec![128u8; 10];
    let mat = Mat::from_gray_bytes(data, 1, 10);
    let (t, _) = threshold(&mat, 100.0, 255.0, THRESH_BINARY).expect("threshold");
    // When not using Otsu, retval should equal the supplied thresh
    assert!(
        (t - 100.0).abs() < 1.0,
        "retval should be supplied thresh, got {t}"
    );
}

#[test]
fn test_threshold_wrong_dtype_returns_error() {
    let data = vec![100u8; 30];
    let mat = Mat::from_bgr_bytes(data, 1, 10);
    assert!(
        threshold(&mat, 127.0, 255.0, THRESH_BINARY).is_err(),
        "CV_8UC3 should return UnsupportedDtype"
    );
}
