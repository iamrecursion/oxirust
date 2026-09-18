use oximedia_compat_cv2::{edge::canny, Mat};

#[test]
fn test_canny_step_edge() {
    // 10×10 grayscale: left half=0, right half=255 (a step edge)
    let mut data = vec![0u8; 100];
    for row in 0..10 {
        for col in 5..10 {
            data[row * 10 + col] = 255;
        }
    }
    let mat = Mat::from_gray_bytes(data, 10, 10);
    let edges = canny(&mat, 50.0, 150.0, 3, false).expect("canny should succeed");
    assert_eq!(edges.rows, 10);
    assert_eq!(edges.cols, 10);
    // The vertical edge should be detected (at least one non-zero pixel near column 4 or 5)
    let has_edge = (0..10).any(|r| edges.at_8u1(r, 4) > 0 || edges.at_8u1(r, 5) > 0);
    assert!(has_edge, "step edge should produce detected edge pixels");
}

#[test]
fn test_canny_output_is_binary() {
    let data: Vec<u8> = (0..100u8).collect();
    let mat = Mat::from_gray_bytes(data, 10, 10);
    let edges = canny(&mat, 30.0, 100.0, 3, false).expect("canny");
    for &v in &edges.data {
        assert!(v == 0 || v == 255, "edge map must be binary, got {v}");
    }
}

#[test]
fn test_canny_uniform_image_no_edges() {
    // A completely flat image has no gradients — no edges should be found.
    let data = vec![128u8; 100];
    let mat = Mat::from_gray_bytes(data, 10, 10);
    let edges = canny(&mat, 50.0, 150.0, 3, false).expect("canny");
    assert!(
        edges.data.iter().all(|&v| v == 0),
        "uniform image should produce zero edges"
    );
}

#[test]
fn test_canny_accepts_color_mat() {
    // cv2.Canny should accept BGR input and convert to gray internally.
    let data = vec![100u8; 10 * 10 * 3];
    let mat = Mat::from_bgr_bytes(data, 10, 10);
    let result = canny(&mat, 50.0, 150.0, 3, false);
    assert!(result.is_ok(), "canny on BGR Mat should succeed");
}
