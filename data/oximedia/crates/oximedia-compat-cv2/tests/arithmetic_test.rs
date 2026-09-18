use oximedia_compat_cv2::{
    arithmetic::{add_weighted, bitwise_and, bitwise_not},
    Mat,
};

#[test]
fn test_add_weighted_blend() {
    let a = Mat::from_gray_bytes(vec![100u8; 16], 4, 4);
    let b = Mat::from_gray_bytes(vec![200u8; 16], 4, 4);
    let blended = add_weighted(&a, 0.5, &b, 0.5, 0.0).unwrap();
    // 0.5*100 + 0.5*200 = 150
    assert_eq!(blended.at_8u1(0, 0), 150);
}

#[test]
fn test_bitwise_not() {
    let mat = Mat::from_gray_bytes(vec![0u8, 255, 128], 1, 3);
    let notted = bitwise_not(&mat).unwrap();
    assert_eq!(notted.at_8u1(0, 0), 255);
    assert_eq!(notted.at_8u1(0, 1), 0);
    assert_eq!(notted.at_8u1(0, 2), 127);
}

#[test]
fn test_bitwise_and_mask() {
    let a = Mat::from_gray_bytes(vec![0xFF, 0x0F, 0xAA], 1, 3);
    let b = Mat::from_gray_bytes(vec![0x0F, 0xFF, 0x55], 1, 3);
    let c = bitwise_and(&a, &b).unwrap();
    assert_eq!(c.at_8u1(0, 0), 0x0F);
    assert_eq!(c.at_8u1(0, 1), 0x0F);
    assert_eq!(c.at_8u1(0, 2), 0x00);
}

#[test]
fn test_size_mismatch() {
    use oximedia_compat_cv2::arithmetic::add;
    let a = Mat::from_gray_bytes(vec![0u8; 4], 2, 2);
    let b = Mat::from_gray_bytes(vec![0u8; 6], 2, 3);
    assert!(add(&a, &b).is_err(), "mismatched sizes should error");
}
