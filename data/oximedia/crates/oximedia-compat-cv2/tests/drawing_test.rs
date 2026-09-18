use oximedia_compat_cv2::{drawing::line, Mat, Point, Scalar};

#[test]
fn test_bresenham_line_sets_pixels() {
    let mut mat = Mat::new_8uc3(20, 20);
    // Draw a horizontal line from (5,5) to (15,5)
    line(
        &mut mat,
        Point { x: 5, y: 5 },
        Point { x: 15, y: 5 },
        Scalar(255.0, 0.0, 0.0, 0.0),
        1,
    )
    .unwrap();
    // Check that pixels along the line are set
    let px = mat.at_8u3(5, 10);
    assert_eq!(px[0], 255, "B channel should be 255 for blue BGR color");
}

#[test]
fn test_rectangle_draws_outline() {
    use oximedia_compat_cv2::drawing::rectangle;
    let mut mat = Mat::new_8uc3(20, 20);
    rectangle(
        &mut mat,
        Point { x: 2, y: 2 },
        Point { x: 8, y: 8 },
        Scalar(0.0, 255.0, 0.0, 0.0),
        1,
    )
    .unwrap();
    // Corner pixel should be set
    assert_eq!(
        mat.at_8u3(2, 2)[1],
        255,
        "G channel for green should be 255"
    );
}

#[test]
fn test_circle_draws_at_top() {
    use oximedia_compat_cv2::drawing::circle;
    let mut mat = Mat::new_8uc3(50, 50);
    circle(
        &mut mat,
        Point { x: 25, y: 25 },
        10,
        Scalar(0.0, 0.0, 255.0, 0.0),
        1,
    )
    .unwrap();
    // Top of circle at row=15, col=25
    let px = mat.at_8u3(15, 25);
    assert_eq!(px[2], 255, "R channel should be 255");
}

#[test]
fn test_fill_poly_fills_triangle() {
    use oximedia_compat_cv2::drawing::fill_poly;
    let mut mat = Mat::new_8uc1(20, 20);
    let pts = vec![
        Point { x: 5, y: 5 },
        Point { x: 15, y: 5 },
        Point { x: 10, y: 15 },
    ];
    fill_poly(&mut mat, &pts, Scalar(200.0, 0.0, 0.0, 0.0)).unwrap();
    // Centroid of triangle roughly at (10, 8)
    let v = mat.data[8 * 20 + 10];
    assert_eq!(v, 200, "interior pixel should be filled");
}

#[test]
fn test_put_text_does_not_panic() {
    use oximedia_compat_cv2::drawing::put_text;
    let mut mat = Mat::new_8uc3(50, 200);
    // Should not panic for any printable ASCII string
    put_text(
        &mut mat,
        "Hello World 0123",
        Point { x: 5, y: 40 },
        0,
        1.0,
        Scalar(255.0, 255.0, 255.0, 0.0),
        1,
    )
    .unwrap();
}

#[test]
fn test_arrow_line_draws_shaft() {
    use oximedia_compat_cv2::drawing::arrow_line;
    let mut mat = Mat::new_8uc3(50, 50);
    arrow_line(
        &mut mat,
        Point { x: 5, y: 25 },
        Point { x: 45, y: 25 },
        Scalar(255.0, 0.0, 0.0, 0.0),
        1,
    )
    .unwrap();
    // Midpoint of shaft
    let px = mat.at_8u3(25, 25);
    assert_eq!(px[0], 255, "shaft pixel should be drawn");
}
