use image::{DynamicImage, RgbImage};
use scirs2_vision::color::{rgb_to_grayscale, ColorSpace};
use scirs2_vision::preprocessing::StructuringElement;
use scirs2_vision::segmentation::{threshold_binary, AdaptiveMethod};

#[test]
#[allow(dead_code)]
fn test_colorspace_enum() {
    assert_ne!(ColorSpace::RGB, ColorSpace::HSV);
    assert_ne!(ColorSpace::RGB, ColorSpace::LAB);
    assert_ne!(ColorSpace::RGB, ColorSpace::Gray);
    assert_ne!(ColorSpace::HSV, ColorSpace::LAB);
    assert_ne!(ColorSpace::HSV, ColorSpace::Gray);
    assert_ne!(ColorSpace::LAB, ColorSpace::Gray);
}

#[test]
#[allow(dead_code)]
fn test_create_structuring_elements() {
    let rect = StructuringElement::Rectangle(3, 3);
    let ellipse = StructuringElement::Ellipse(5, 5);
    let cross = StructuringElement::Cross(3);

    match rect {
        StructuringElement::Rectangle(w, h) => {
            assert_eq!(w, 3);
            assert_eq!(h, 3);
        }
        _ => panic!("Expected Rectangle"),
    }

    match ellipse {
        StructuringElement::Ellipse(w, h) => {
            assert_eq!(w, 5);
            assert_eq!(h, 5);
        }
        _ => panic!("Expected Ellipse"),
    }

    match cross {
        StructuringElement::Cross(size) => {
            assert_eq!(size, 3);
        }
        _ => panic!("Expected Cross"),
    }
}

#[test]
#[allow(dead_code)]
fn test_adaptive_method_enum() {
    match AdaptiveMethod::Mean {
        AdaptiveMethod::Mean => {}
        _ => panic!("Expected Mean"),
    }

    match AdaptiveMethod::Gaussian {
        AdaptiveMethod::Gaussian => {}
        _ => panic!("Expected Gaussian"),
    }
}

#[test]
#[allow(dead_code)]
fn test_grayscale_conversion_with_test_image() {
    // Create a simple test image
    let mut img = RgbImage::new(2, 2);
    img.put_pixel(0, 0, image::Rgb([100, 150, 200]));
    img.put_pixel(0, 1, image::Rgb([50, 100, 150]));
    img.put_pixel(1, 0, image::Rgb([200, 150, 100]));
    img.put_pixel(1, 1, image::Rgb([255, 255, 255]));

    let dynamic_img = DynamicImage::ImageRgb8(img);

    // Test with default weights
    let result = rgb_to_grayscale(&dynamic_img, None).expect("Operation failed");
    assert_eq!(result.width(), 2);
    assert_eq!(result.height(), 2);

    // Test with custom weights
    let custom_result =
        rgb_to_grayscale(&dynamic_img, Some([0.3, 0.4, 0.3])).expect("Operation failed");
    assert_eq!(custom_result.width(), 2);
    assert_eq!(custom_result.height(), 2);
}

#[test]
#[allow(dead_code)]
fn test_binary_threshold_with_test_image() {
    // Create a simple test image
    let mut img = RgbImage::new(2, 2);
    img.put_pixel(0, 0, image::Rgb([100, 100, 100]));
    img.put_pixel(0, 1, image::Rgb([50, 50, 50]));
    img.put_pixel(1, 0, image::Rgb([200, 200, 200]));
    img.put_pixel(1, 1, image::Rgb([150, 150, 150]));

    let dynamic_img = DynamicImage::ImageRgb8(img);

    // Test binary thresholding
    let result = threshold_binary(&dynamic_img, 0.5).expect("Operation failed"); // 0.5 = 127.5 in [0,255]

    // Only the pixel at (1,0) should be above threshold
    assert_eq!(result.get_pixel(0, 0)[0], 0);
    assert_eq!(result.get_pixel(0, 1)[0], 0);
    assert_eq!(result.get_pixel(1, 0)[0], 255);
    assert_eq!(result.get_pixel(1, 1)[0], 255);
}
