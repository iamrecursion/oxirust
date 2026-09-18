use oximedia_compat_cv2::*;

#[test]
fn test_key_constant_values() {
    assert_eq!(IMREAD_COLOR, 1);
    assert_eq!(IMREAD_GRAYSCALE, 0);
    assert_eq!(COLOR_BGR2RGB, 4);
    assert_eq!(INTER_LINEAR, 1);
    assert_eq!(THRESH_BINARY, 0);
    assert_eq!(THRESH_OTSU, 8);
    assert_eq!(MORPH_RECT, 0);
    assert_eq!(MORPH_OPEN, 2);
    assert_eq!(RETR_EXTERNAL, 0);
    assert_eq!(CHAIN_APPROX_SIMPLE, 2);
    assert_eq!(BORDER_CONSTANT, 0);
    assert_eq!(LINE_8, 8);
    assert_eq!(FONT_HERSHEY_SIMPLEX, 0);
    assert_eq!(TM_SQDIFF, 0);
    assert_eq!(TM_CCOEFF_NORMED, 5);
    assert_eq!(CV_8U, 0);
    assert_eq!(FILLED, -1);
}

#[test]
fn test_constant_set_compiles() {
    let _ = [
        IMREAD_COLOR,
        IMREAD_GRAYSCALE,
        IMREAD_UNCHANGED,
        COLOR_BGR2RGB,
        COLOR_BGR2GRAY,
        COLOR_BGR2HSV,
        COLOR_BGR2Lab,
        INTER_NEAREST,
        INTER_LINEAR,
        INTER_CUBIC,
        INTER_AREA,
        INTER_LANCZOS4,
        THRESH_BINARY,
        THRESH_BINARY_INV,
        THRESH_OTSU,
        THRESH_TRIANGLE,
        MORPH_ERODE,
        MORPH_DILATE,
        MORPH_OPEN,
        MORPH_CLOSE,
        MORPH_RECT,
        MORPH_CROSS,
        MORPH_ELLIPSE,
        RETR_EXTERNAL,
        RETR_LIST,
        RETR_CCOMP,
        RETR_TREE,
        CHAIN_APPROX_NONE,
        CHAIN_APPROX_SIMPLE,
        BORDER_CONSTANT,
        BORDER_REPLICATE,
        BORDER_REFLECT,
        LINE_8,
        LINE_4,
        LINE_AA,
        FILLED,
        FONT_HERSHEY_SIMPLEX,
        TM_SQDIFF,
        TM_CCOEFF_NORMED,
        NORM_L2,
        NORM_MINMAX,
        CV_8U,
        CV_32F,
    ];
}
