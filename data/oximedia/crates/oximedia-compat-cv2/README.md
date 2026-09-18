# oximedia-compat-cv2

Pure-Rust OpenCV `cv2` API compatibility layer built on top of OxiMedia. Provides a
`Mat` type (BGR channel ordering by default, per OpenCV convention), ~170 OpenCV-compatible
constants, and image I/O functions (`imread`, `imwrite`, `imdecode`, `imencode`) that
dispatch into the OxiMedia image pipeline.

```rust
use oximedia_compat_cv2::{imread, imwrite, IMREAD_COLOR};

let mat = imread("input.png", IMREAD_COLOR).expect("imread failed");
imwrite("output.png", &mat).expect("imwrite failed");
```

This crate is part of the [OxiMedia](https://github.com/cool-japan/oximedia) workspace.
Geometry (resize, flip, `warp_affine`, `warp_perspective`, `get_perspective_transform`,
`get_affine_transform`, `remap`), filters, edge detection, morphology, features (ORB,
`BFMatcher` with NORM_L1/L2/L2SQR/Hamming), drawing (`draw_contours`, `draw_keypoints`,
`draw_matches`, `draw_marker`, `put_text_hershey`), shape analysis (`moments`, `hu_moments`,
`fit_ellipse`, `fit_line`, `min_area_rect`, `point_polygon_test`), Mat reductions
(`count_non_zero`, `sum_elems`, `mean_val`, `mean_std_dev`, `norm`, `norm_diff`,
`min_max_loc`, `split`, `merge`), dense optical flow (`calc_optical_flow_farneback`),
and DNN inference (`dnn` feature, gated on `oxionnx`) are provided across slices A–G
of the cv2 compatibility layer.
