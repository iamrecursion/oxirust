use oximedia_compat_cv2::{
    mat::{Mat, MatType},
    optical_flow::calc_optical_flow_farneback,
};

/// Synthetic test texture: checkerboard with a gentle linear ramp.
fn make_checkerboard(w: usize, h: usize, block: usize) -> Mat {
    let mut m = Mat::new(h, w, MatType::CV_8UC1);
    for y in 0..h {
        for x in 0..w {
            let base: u8 = if ((x / block) + (y / block)) % 2 == 0 {
                180
            } else {
                20
            };
            let gradient = ((x + y) * 25 / (w + h)) as u8;
            m.data[y * w + x] = base.saturating_add(gradient);
        }
    }
    m
}

fn make_shifted(src: &Mat, dx: i32, dy: i32) -> Mat {
    let w = src.cols;
    let h = src.rows;
    let mut m = Mat::new(h, w, MatType::CV_8UC1);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = (x - dx).clamp(0, w as i32 - 1) as usize;
            let sy = (y - dy).clamp(0, h as i32 - 1) as usize;
            m.data[y as usize * w + x as usize] = src.data[sy * w + sx];
        }
    }
    m
}

fn read_flow(flow: &Mat, idx: usize) -> (f32, f32) {
    let off = idx * 8;
    let dx = f32::from_ne_bytes([
        flow.data[off],
        flow.data[off + 1],
        flow.data[off + 2],
        flow.data[off + 3],
    ]);
    let dy = f32::from_ne_bytes([
        flow.data[off + 4],
        flow.data[off + 5],
        flow.data[off + 6],
        flow.data[off + 7],
    ]);
    (dx, dy)
}

#[test]
fn test_farneback_dtype_and_shape() {
    let prev = make_checkerboard(64, 64, 8);
    let next = prev.clone();
    let flow = calc_optical_flow_farneback(&prev, &next, 0.5, 3, 15, 3, 5, 1.1, 0).unwrap();
    assert_eq!(flow.mat_type, MatType::CV_32FC2);
    assert_eq!(flow.rows, 64);
    assert_eq!(flow.cols, 64);
}

#[test]
fn test_farneback_static_frame() {
    let prev = make_checkerboard(64, 64, 8);
    let next = prev.clone();
    let flow = calc_optical_flow_farneback(&prev, &next, 0.5, 3, 15, 3, 5, 1.1, 0).unwrap();
    let n = flow.rows * flow.cols;
    let mean_mag: f32 = (0..n)
        .map(|i| {
            let (dx, dy) = read_flow(&flow, i);
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f32>()
        / n as f32;
    assert!(
        mean_mag < 0.5,
        "static frame should have ~zero flow, got mean_mag={}",
        mean_mag
    );
}

#[test]
fn test_farneback_horizontal_shift() {
    let prev = make_checkerboard(80, 80, 10);
    let next = make_shifted(&prev, 5, 0);
    let flow = calc_optical_flow_farneback(&prev, &next, 0.5, 3, 15, 3, 5, 1.1, 0).unwrap();
    let w = flow.cols;
    let h = flow.rows;
    let mut dxs: Vec<f32> = Vec::new();
    for y in 10..h - 10 {
        for x in 10..w - 10 {
            let (dx, _) = read_flow(&flow, y * w + x);
            dxs.push(dx);
        }
    }
    dxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dx = dxs[dxs.len() / 2];
    assert!(
        (median_dx - 5.0).abs() < 1.5,
        "expected median dx≈5.0, got {}",
        median_dx
    );
}

#[test]
fn test_farneback_vertical_shift() {
    let prev = make_checkerboard(80, 80, 10);
    let next = make_shifted(&prev, 0, 3);
    let flow = calc_optical_flow_farneback(&prev, &next, 0.5, 3, 15, 3, 5, 1.1, 0).unwrap();
    let w = flow.cols;
    let h = flow.rows;
    let mut dys: Vec<f32> = Vec::new();
    for y in 10..h - 10 {
        for x in 10..w - 10 {
            let (_, dy) = read_flow(&flow, y * w + x);
            dys.push(dy);
        }
    }
    dys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dy = dys[dys.len() / 2];
    assert!(
        (median_dy - 3.0).abs() < 1.5,
        "expected median dy≈3.0, got {}",
        median_dy
    );
}

#[test]
fn test_farneback_size_mismatch() {
    let prev = Mat::new_8uc1(20, 20);
    let next = Mat::new_8uc1(30, 30);
    let result = calc_optical_flow_farneback(&prev, &next, 0.5, 3, 15, 3, 5, 1.1, 0);
    assert!(result.is_err());
}

#[test]
fn test_farneback_diagonal_shift() {
    // shift=(3,3): coarsest-level shift = 3*0.25 = 0.75px per axis.
    let prev = make_checkerboard(80, 80, 10);
    let next = make_shifted(&prev, 3, 3);
    let flow = calc_optical_flow_farneback(&prev, &next, 0.5, 3, 15, 3, 5, 1.1, 0).unwrap();
    let w = flow.cols;
    let h = flow.rows;
    let mut interior_dxs: Vec<f32> = Vec::new();
    let mut interior_dys: Vec<f32> = Vec::new();
    for y in 15..h - 15 {
        for x in 15..w - 15 {
            let (dx, dy) = read_flow(&flow, y * w + x);
            interior_dxs.push(dx);
            interior_dys.push(dy);
        }
    }
    interior_dxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    interior_dys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mdx = interior_dxs[interior_dxs.len() / 2];
    let mdy = interior_dys[interior_dys.len() / 2];
    assert!((mdx - 3.0).abs() < 1.5, "expected dx≈3.0, got {}", mdx);
    assert!((mdy - 3.0).abs() < 1.5, "expected dy≈3.0, got {}", mdy);
}
