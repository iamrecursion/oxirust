use oximedia_compat_cv2::connected::connected_components;
use oximedia_compat_cv2::Mat;

#[test]
fn test_three_blobs_three_labels() {
    // 7×15 binary image with 3 isolated blobs
    let mut data = vec![0u8; 7 * 15];
    // blob1: columns 0-2 rows 0-2
    for r in 0..3 {
        for c in 0..3 {
            data[r * 15 + c] = 255;
        }
    }
    // blob2: columns 6-8 rows 0-2
    for r in 0..3 {
        for c in 6..9 {
            data[r * 15 + c] = 255;
        }
    }
    // blob3: columns 12-14 rows 0-2
    for r in 0..3 {
        for c in 12..15 {
            data[r * 15 + c] = 255;
        }
    }
    let mat = Mat::from_gray_bytes(data, 7, 15);
    let (num_labels, labels) = connected_components(&mat).unwrap();
    // Should find exactly 3 foreground labels + background = 4 total
    assert_eq!(num_labels, 4, "expected 3 blobs + background = 4 labels");
    // Background label is 0
    assert_eq!(
        labels[3 * 15 + 7],
        0,
        "middle row center should be background"
    );
}

#[test]
fn test_single_pixel_blob() {
    let mut data = vec![0u8; 5 * 5];
    data[2 * 5 + 2] = 255; // single center pixel
    let mat = Mat::from_gray_bytes(data, 5, 5);
    let (num_labels, labels) = connected_components(&mat).unwrap();
    assert_eq!(num_labels, 2, "background + 1 blob = 2");
    assert_eq!(labels[2 * 5 + 2], 1, "single pixel gets label 1");
    // All other pixels are background
    for i in 0..25 {
        if i != 2 * 5 + 2 {
            assert_eq!(labels[i], 0);
        }
    }
}

#[test]
fn test_all_zeros_is_background() {
    let mat = Mat::from_gray_bytes(vec![0u8; 10 * 10], 10, 10);
    let (num_labels, labels) = connected_components(&mat).unwrap();
    assert_eq!(num_labels, 1, "only background");
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn test_blobs_have_unique_labels() {
    // Two separate 1-pixel blobs
    let mut data = vec![0u8; 3 * 5];
    data[0] = 255; // row 0, col 0
    data[3 * 5 - 1] = 255; // row 2, col 4 (far corner)
    let mat = Mat::from_gray_bytes(data, 3, 5);
    let (num_labels, labels) = connected_components(&mat).unwrap();
    assert_eq!(num_labels, 3, "background + 2 blobs = 3");
    assert_ne!(labels[0], labels[3 * 5 - 1], "blobs get different labels");
    assert_ne!(labels[0], 0, "blob 1 is foreground");
    assert_ne!(labels[3 * 5 - 1], 0, "blob 2 is foreground");
}
