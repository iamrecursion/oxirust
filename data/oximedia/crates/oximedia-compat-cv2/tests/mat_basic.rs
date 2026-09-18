use oximedia_compat_cv2::{Mat, MatType};

#[test]
fn test_new_8uc3_data_size() {
    let mat = Mat::new_8uc3(10, 20);
    assert_eq!(mat.data.len(), 10 * 20 * 3);
    assert_eq!(mat.rows, 10);
    assert_eq!(mat.cols, 20);
    assert_eq!(mat.step, 20 * 3);
}

#[test]
fn test_at_8u3_reads_bgr() {
    let mut mat = Mat::new_8uc3(5, 5);
    let off = 2 * mat.step + 3 * 3;
    mat.data[off] = 10;
    mat.data[off + 1] = 20;
    mat.data[off + 2] = 30;
    assert_eq!(mat.at_8u3(2, 3), [10, 20, 30]);
}

#[test]
fn test_mattype_channels_depth() {
    assert_eq!(MatType::CV_8UC3.channels(), 3);
    assert_eq!(MatType::CV_8UC3.depth_bytes(), 1);
    assert_eq!(MatType::CV_8UC1.channels(), 1);
    assert_eq!(MatType::CV_32FC1.channels(), 1);
    assert_eq!(MatType::CV_32FC1.depth_bytes(), 4);
    assert_eq!(MatType::CV_32FC3.channels(), 3);
}

#[test]
fn test_to_rgb_bytes_swaps_rb() {
    let data = vec![10u8, 20, 30, 40, 50, 60];
    let mat = Mat::from_bgr_bytes(data, 1, 2);
    let rgb = mat.to_rgb_bytes().expect("to_rgb_bytes");
    assert_eq!(rgb, vec![30, 20, 10, 60, 50, 40]);
}
