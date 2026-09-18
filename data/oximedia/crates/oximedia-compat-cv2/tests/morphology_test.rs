use oximedia_compat_cv2::{
    morphology::{dilate, erode, get_structuring_element, morphology_ex},
    Mat, MORPH_RECT,
};

#[test]
fn test_dilate_single_pixel() {
    // 7×7 gray image with single bright pixel in center
    let mut data = vec![0u8; 49];
    data[3 * 7 + 3] = 255; // center pixel
    let mat = Mat::from_gray_bytes(data, 7, 7);
    let kernel = get_structuring_element(MORPH_RECT, 3).expect("kernel");
    let dilated = dilate(&mat, &kernel, 1).expect("dilate");
    // The 3×3 neighborhood of center should all be 255
    assert_eq!(dilated.at_8u1(2, 2), 255);
    assert_eq!(dilated.at_8u1(3, 3), 255);
    assert_eq!(dilated.at_8u1(4, 4), 255);
    // Corners should still be 0
    assert_eq!(dilated.at_8u1(0, 0), 0);
    assert_eq!(dilated.at_8u1(6, 6), 0);
}

#[test]
fn test_erode_isolates_removal() {
    // Single bright pixel surrounded by dark — erode should remove it
    let mut data = vec![0u8; 49];
    data[3 * 7 + 3] = 255;
    let mat = Mat::from_gray_bytes(data, 7, 7);
    let kernel = get_structuring_element(MORPH_RECT, 3).expect("kernel");
    let eroded = erode(&mat, &kernel, 1).expect("erode");
    assert_eq!(
        eroded.at_8u1(3, 3),
        0,
        "isolated pixel should be eroded away"
    );
}

#[test]
fn test_morphology_ex_open_removes_noise() {
    use oximedia_compat_cv2::MORPH_OPEN;
    let mut data = vec![0u8; 49];
    data[3 * 7 + 3] = 255; // isolated noise pixel
    let mat = Mat::from_gray_bytes(data, 7, 7);
    let kernel = get_structuring_element(MORPH_RECT, 3).expect("kernel");
    let opened = morphology_ex(&mat, MORPH_OPEN, &kernel).expect("open");
    assert_eq!(
        opened.at_8u1(3, 3),
        0,
        "opening should remove isolated pixel"
    );
}

#[test]
fn test_erode_multiple_iterations() {
    // A 5×5 bright block — after 2 erode iterations with 3×3 kernel it should shrink to 1×1
    let mut data = vec![0u8; 11 * 11];
    for y in 3..8usize {
        for x in 3..8usize {
            data[y * 11 + x] = 255;
        }
    }
    let mat = Mat::from_gray_bytes(data, 11, 11);
    let kernel = get_structuring_element(MORPH_RECT, 3).expect("kernel");
    let out = erode(&mat, &kernel, 2).expect("erode 2");
    // Centre should survive 2 erosions of a 5×5 block
    assert_eq!(out.at_8u1(5, 5), 255);
    // The corners of the 5×5 block should be eroded away
    assert_eq!(out.at_8u1(3, 3), 0);
    assert_eq!(out.at_8u1(7, 7), 0);
}
