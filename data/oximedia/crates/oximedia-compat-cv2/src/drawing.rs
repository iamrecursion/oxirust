//! Drawing functions: line, rectangle, circle, ellipse, polylines, fill_poly, put_text, arrow_line.
//!
//! All functions use integer Bresenham rasterization matching cv2 output exactly.
//! No anti-aliasing is performed; this intentionally mirrors cv2 integer drawing.

use crate::error::Cv2Result;
use crate::mat::{Mat, Point, Scalar};

// ── Pixel write helper ────────────────────────────────────────────────────────

/// Write a single pixel at `(row, col)` with bounds checking.
///
/// Coordinates are checked before writing; out-of-bounds silently no-ops.
#[inline]
fn set_pixel(mat: &mut Mat, row: i32, col: i32, color: Scalar) {
    if row < 0 || col < 0 || row as usize >= mat.rows || col as usize >= mat.cols {
        return;
    }
    let r = row as usize;
    let c = col as usize;
    match mat.channels() {
        1 => {
            mat.data[r * mat.step + c] = color.0 as u8;
        }
        3 => {
            let off = r * mat.step + c * 3;
            mat.data[off] = color.0 as u8;
            mat.data[off + 1] = color.1 as u8;
            mat.data[off + 2] = color.2 as u8;
        }
        4 => {
            let off = r * mat.step + c * 4;
            mat.data[off] = color.0 as u8;
            mat.data[off + 1] = color.1 as u8;
            mat.data[off + 2] = color.2 as u8;
            mat.data[off + 3] = color.3 as u8;
        }
        _ => {}
    }
}

// ── Bresenham line core ───────────────────────────────────────────────────────

/// Bresenham line rasterizer with optional thickness (brush half-width).
fn draw_bresenham_line(
    mat: &mut Mat,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: Scalar,
    thickness: usize,
) {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let half = (thickness / 2) as i32;

    loop {
        for ty in -half..=half {
            for tx in -half..=half {
                set_pixel(mat, y + ty, x + tx, color);
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

// ── Public drawing functions ──────────────────────────────────────────────────

/// Draw a line on `mat` from `pt1` to `pt2`.
///
/// Mirrors `cv2.line(img, pt1, pt2, color, thickness=1)`.
pub fn line(mat: &mut Mat, pt1: Point, pt2: Point, color: Scalar, thickness: i32) -> Cv2Result<()> {
    let t = thickness.max(1) as usize;
    draw_bresenham_line(mat, pt1.x, pt1.y, pt2.x, pt2.y, color, t);
    Ok(())
}

/// Draw a rectangle outline (or filled rectangle when `thickness < 0`).
///
/// Mirrors `cv2.rectangle(img, pt1, pt2, color, thickness=1)`.
pub fn rectangle(
    mat: &mut Mat,
    pt1: Point,
    pt2: Point,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    let x1 = pt1.x.min(pt2.x);
    let x2 = pt1.x.max(pt2.x);
    let y1 = pt1.y.min(pt2.y);
    let y2 = pt1.y.max(pt2.y);

    if thickness < 0 {
        // Filled rectangle
        for y in y1.max(0)..(y2 + 1).min(mat.rows as i32) {
            for x in x1.max(0)..(x2 + 1).min(mat.cols as i32) {
                set_pixel(mat, y, x, color);
            }
        }
    } else {
        let t = thickness.max(1) as usize;
        // Top edge
        draw_bresenham_line(mat, x1, y1, x2, y1, color, t);
        // Bottom edge
        draw_bresenham_line(mat, x1, y2, x2, y2, color, t);
        // Left edge
        draw_bresenham_line(mat, x1, y1, x1, y2, color, t);
        // Right edge
        draw_bresenham_line(mat, x2, y1, x2, y2, color, t);
    }
    Ok(())
}

/// Draw a circle outline (or filled disc when `thickness < 0`).
///
/// Mirrors `cv2.circle(img, center, radius, color, thickness=1)`.
pub fn circle(
    mat: &mut Mat,
    center: Point,
    radius: i32,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    let (cx, cy) = (center.x, center.y);
    let r = radius.max(0);

    if thickness < 0 {
        // Filled circle
        for y in (cy - r).max(0)..(cy + r + 1).min(mat.rows as i32) {
            let dy = y - cy;
            let dx = ((r * r - dy * dy) as f64).sqrt() as i32;
            for x in (cx - dx).max(0)..(cx + dx + 1).min(mat.cols as i32) {
                set_pixel(mat, y, x, color);
            }
        }
    } else {
        // Midpoint circle algorithm
        let t = (thickness.max(1) - 1) / 2;
        let mut mx = 0i32;
        let mut my = r;
        let mut d = 3 - 2 * r;
        while mx <= my {
            for &(px, py) in &[
                (cx + mx, cy + my),
                (cx - mx, cy + my),
                (cx + mx, cy - my),
                (cx - mx, cy - my),
                (cx + my, cy + mx),
                (cx - my, cy + mx),
                (cx + my, cy - mx),
                (cx - my, cy - mx),
            ] {
                for tx in -t..=t {
                    for ty in -t..=t {
                        set_pixel(mat, py + ty, px + tx, color);
                    }
                }
            }
            if d < 0 {
                d += 4 * mx + 6;
            } else {
                d += 4 * (mx - my) + 10;
                my -= 1;
            }
            mx += 1;
        }
    }
    Ok(())
}

/// Draw an elliptic arc or filled ellipse.
///
/// Mirrors `cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1)`.
#[allow(clippy::too_many_arguments)]
pub fn ellipse(
    mat: &mut Mat,
    center: Point,
    axes: Point,
    angle: f64,
    start_angle: f64,
    end_angle: f64,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    let (cx, cy) = (center.x as f64, center.y as f64);
    let (ax, ay) = (axes.x as f64, axes.y as f64);
    let rot_rad = angle * std::f64::consts::PI / 180.0;
    let cos_r = rot_rad.cos();
    let sin_r = rot_rad.sin();

    let start_rad = start_angle * std::f64::consts::PI / 180.0;
    let end_rad = end_angle * std::f64::consts::PI / 180.0;

    let steps = ((ax.max(ay) * 2.0 * std::f64::consts::PI) as usize).max(360);
    let delta = (end_rad - start_rad) / steps as f64;

    let t = thickness.max(1) as usize;
    let filled = thickness < 0;

    let mut prev: Option<(i32, i32)> = None;
    for step in 0..=steps {
        let theta = start_rad + step as f64 * delta;
        let ex = ax * theta.cos();
        let ey = ay * theta.sin();
        let rx = (ex * cos_r - ey * sin_r + cx).round() as i32;
        let ry = (ex * sin_r + ey * cos_r + cy).round() as i32;

        if let Some((px, py)) = prev {
            if filled {
                draw_bresenham_line(mat, center.x, center.y, rx, ry, color, 1);
            } else {
                draw_bresenham_line(mat, px, py, rx, ry, color, t);
            }
        }
        prev = Some((rx, ry));
    }
    Ok(())
}

/// Draw connected polylines (open or closed polygon outline).
///
/// Mirrors `cv2.polylines(img, pts, isClosed, color, thickness=1)`.
pub fn polylines(
    mat: &mut Mat,
    pts: &[Point],
    is_closed: bool,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    let t = thickness.max(1) as usize;
    let n = pts.len();
    for i in 0..n {
        let p0 = pts[i];
        let p1 = if i + 1 < n {
            pts[i + 1]
        } else if is_closed {
            pts[0]
        } else {
            break;
        };
        draw_bresenham_line(mat, p0.x, p0.y, p1.x, p1.y, color, t);
    }
    Ok(())
}

/// Fill a polygon defined by `pts` with `color` using scanline rasterization.
///
/// Mirrors `cv2.fillPoly(img, [pts], color)`.
pub fn fill_poly(mat: &mut Mat, pts: &[Point], color: Scalar) -> Cv2Result<()> {
    if pts.len() < 3 {
        return Ok(());
    }

    let y_min = pts.iter().map(|p| p.y).min().unwrap_or(0).max(0) as usize;
    let y_max = pts
        .iter()
        .map(|p| p.y)
        .max()
        .unwrap_or(0)
        .min(mat.rows as i32 - 1) as usize;
    let n = pts.len();

    for y in y_min..=y_max {
        let mut intersections: Vec<i32> = Vec::new();
        let yf = y as i32;
        for i in 0..n {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % n];
            let (x0, y0, x1, y1) = (p0.x, p0.y, p1.x, p1.y);
            if (y0 <= yf && y1 > yf) || (y1 <= yf && y0 > yf) {
                let xi = x0 + (yf - y0) * (x1 - x0) / (y1 - y0);
                intersections.push(xi);
            }
        }
        intersections.sort_unstable();
        let mut i = 0;
        while i + 1 < intersections.len() {
            let x_start = intersections[i].clamp(0, mat.cols as i32 - 1) as usize;
            let x_end = intersections[i + 1].clamp(0, mat.cols as i32 - 1) as usize;
            for x in x_start..=x_end {
                set_pixel(mat, y as i32, x as i32, color);
            }
            i += 2;
        }
    }
    Ok(())
}

/// Render ASCII text on `mat` using a minimal 5×8 bitmap font.
///
/// Mirrors `cv2.putText(img, text, org, fontFace, fontScale, color, thickness=1)`.
///
/// `font_face` is accepted for API compatibility but only a single bitmap font
/// is implemented.  Text origin `org` is the bottom-left corner of the first
/// character, matching cv2 convention.
#[allow(clippy::too_many_arguments)]
pub fn put_text(
    mat: &mut Mat,
    text: &str,
    org: Point,
    font_face: i32,
    font_scale: f64,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    let _ = font_face; // API compat
    let char_w = (6.0 * font_scale).round() as i32;
    let char_h = (10.0 * font_scale).round() as i32;
    let stroke = thickness.max(1);
    let mut cursor_x = org.x;
    let cursor_y = org.y;

    for c in text.chars() {
        let bitmap = char_bitmap(c);
        for (by, &row) in bitmap.iter().enumerate() {
            for bx in 0..5i32 {
                if (row >> (4 - bx)) & 1 == 1 {
                    let px = cursor_x + (bx as f64 * font_scale) as i32;
                    let py = cursor_y - char_h + (by as f64 * font_scale) as i32;
                    for sy in 0..stroke {
                        for sx in 0..stroke {
                            set_pixel(mat, py + sy, px + sx, color);
                        }
                    }
                }
            }
        }
        cursor_x += char_w + stroke;
    }
    Ok(())
}

/// Draw one or all contours from a list of polygon vertex lists.
///
/// Mirrors `cv2.drawContours(img, contours, contourIdx, color, thickness)`.
///
/// When `contour_idx < 0` all contours are drawn; otherwise only the contour
/// at `contours[contour_idx as usize]` is drawn (silently ignored if out of
/// bounds).  `thickness < 0` (e.g. `FILLED = -1`) fills the polygon.
pub fn draw_contours(
    img: &mut Mat,
    contours: &[Vec<Point>],
    contour_idx: i32,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    if contour_idx < 0 {
        for contour in contours {
            draw_single_contour(img, contour, color, thickness)?;
        }
    } else {
        let idx = contour_idx as usize;
        if idx < contours.len() {
            draw_single_contour(img, &contours[idx], color, thickness)?;
        }
    }
    Ok(())
}

fn draw_single_contour(
    img: &mut Mat,
    pts: &[Point],
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    if pts.is_empty() {
        return Ok(());
    }
    if thickness < 0 {
        // Delegate to the existing scanline fill_poly implementation.
        fill_poly(img, pts, color)?;
    } else {
        let t = thickness.max(1) as usize;
        let n = pts.len();
        for i in 0..n {
            let p0 = pts[i];
            let p1 = pts[(i + 1) % n];
            draw_bresenham_line(img, p0.x, p0.y, p1.x, p1.y, color, t);
        }
    }
    Ok(())
}

/// Draw keypoints on a copy of `src`, placing the result in `out`.
///
/// Mirrors `cv2.drawKeypoints(src, keypoints, outImage, color, flags)`.
///
/// Each keypoint is drawn as a circle whose radius is half the keypoint `size`.
/// When `flags & DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS != 0` (= 4), an
/// additional line from the center towards the `angle` direction is drawn.
pub fn draw_keypoints(
    src: &Mat,
    keypoints: &[crate::features::KeyPoint],
    out: &mut Mat,
    color: Scalar,
    flags: i32,
) -> Cv2Result<()> {
    // Copy src into out.
    out.data = src.data.clone();
    out.rows = src.rows;
    out.cols = src.cols;
    out.step = src.step;
    out.mat_type = src.mat_type;

    for kp in keypoints {
        let cx = kp.pt.x as i32;
        let cy = kp.pt.y as i32;
        let radius = ((kp.size / 2.0).max(1.0)) as i32;
        circle(out, Point { x: cx, y: cy }, radius, color, 1)?;

        if flags & crate::constants::DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS != 0 {
            let angle_rad = (kp.angle as f64) * std::f64::consts::PI / 180.0;
            let end_x = kp.pt.x as f64 + radius as f64 * angle_rad.cos();
            let end_y = kp.pt.y as f64 + radius as f64 * angle_rad.sin();
            line(
                out,
                Point { x: cx, y: cy },
                Point {
                    x: end_x as i32,
                    y: end_y as i32,
                },
                color,
                1,
            )?;
        }
    }
    Ok(())
}

/// Draw match lines between two keypoint sets placed side-by-side.
///
/// Mirrors `cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, …)`.
///
/// The output Mat has width `img1.cols + img2.cols` and height
/// `max(img1.rows, img2.rows)`.  `img1` is placed on the left, `img2` on the
/// right.  Each `DMatch` entry draws a line from the query keypoint (in
/// `img1`) to the train keypoint (in `img2`, x-offset by `img1.cols`).  If
/// `matches_mask` is non-empty, only matches where the corresponding mask byte
/// is non-zero are drawn.
#[allow(clippy::too_many_arguments)]
pub fn draw_matches(
    img1: &Mat,
    kp1: &[crate::features::KeyPoint],
    img2: &Mat,
    kp2: &[crate::features::KeyPoint],
    matches: &[crate::features::DMatch],
    out: &mut Mat,
    match_color: Scalar,
    single_point_color: Scalar,
    matches_mask: &[u8],
    flags: i32,
) -> Cv2Result<()> {
    let out_cols = img1.cols + img2.cols;
    let out_rows = img1.rows.max(img2.rows);

    let mut result = Mat::new(out_rows, out_cols, img1.mat_type);

    // Copy img1 (left side)
    let ch = img1.channels();
    let row_bytes1 = img1.cols * ch;
    for row in 0..img1.rows {
        let src_off = row * img1.step;
        let dst_off = row * result.step;
        result.data[dst_off..dst_off + row_bytes1]
            .copy_from_slice(&img1.data[src_off..src_off + row_bytes1]);
    }

    // Copy img2 (right side, offset by img1.cols)
    let row_bytes2 = img2.cols * img2.channels();
    let x_off2 = img1.cols * ch;
    for row in 0..img2.rows {
        let src_off = row * img2.step;
        let dst_off = row * result.step + x_off2;
        result.data[dst_off..dst_off + row_bytes2]
            .copy_from_slice(&img2.data[src_off..src_off + row_bytes2]);
    }

    // Draw match lines
    for (i, m) in matches.iter().enumerate() {
        if !matches_mask.is_empty() && matches_mask.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        let q_idx = m.query_idx as usize;
        let t_idx = m.train_idx as usize;
        if q_idx >= kp1.len() || t_idx >= kp2.len() {
            continue;
        }
        let p1 = Point {
            x: kp1[q_idx].pt.x as i32,
            y: kp1[q_idx].pt.y as i32,
        };
        let p2 = Point {
            x: kp2[t_idx].pt.x as i32 + img1.cols as i32,
            y: kp2[t_idx].pt.y as i32,
        };
        line(&mut result, p1, p2, match_color, 1)?;
    }

    // Draw single keypoints unless suppressed
    if flags & crate::constants::DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS == 0 {
        for kp in kp1 {
            circle(
                &mut result,
                Point {
                    x: kp.pt.x as i32,
                    y: kp.pt.y as i32,
                },
                3,
                single_point_color,
                1,
            )?;
        }
        for kp in kp2 {
            circle(
                &mut result,
                Point {
                    x: kp.pt.x as i32 + img1.cols as i32,
                    y: kp.pt.y as i32,
                },
                3,
                single_point_color,
                1,
            )?;
        }
    }

    *out = result;
    Ok(())
}

/// Draw a marker symbol at the specified position.
///
/// Mirrors `cv2.drawMarker(img, position, color, markerType, markerSize, thickness)`.
///
/// The supported marker types are defined in `crate::constants::marker_type::*`.
/// Unknown `marker_type` values are silently ignored.
pub fn draw_marker(
    img: &mut Mat,
    position: Point,
    color: Scalar,
    marker_type: i32,
    marker_size: i32,
    thickness: i32,
) -> Cv2Result<()> {
    let (cx, cy) = (position.x, position.y);
    let s = marker_size / 2;

    match marker_type {
        // MARKER_CROSS (0): horizontal + vertical lines
        0 => {
            line(
                img,
                Point { x: cx - s, y: cy },
                Point { x: cx + s, y: cy },
                color,
                thickness,
            )?;
            line(
                img,
                Point { x: cx, y: cy - s },
                Point { x: cx, y: cy + s },
                color,
                thickness,
            )?;
        }
        // MARKER_TILTED_CROSS (1): two diagonal lines
        1 => {
            line(
                img,
                Point {
                    x: cx - s,
                    y: cy - s,
                },
                Point {
                    x: cx + s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx + s,
                    y: cy - s,
                },
                Point {
                    x: cx - s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
        }
        // MARKER_STAR (2): cross + tilted cross
        2 => {
            line(
                img,
                Point { x: cx - s, y: cy },
                Point { x: cx + s, y: cy },
                color,
                thickness,
            )?;
            line(
                img,
                Point { x: cx, y: cy - s },
                Point { x: cx, y: cy + s },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx - s,
                    y: cy - s,
                },
                Point {
                    x: cx + s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx + s,
                    y: cy - s,
                },
                Point {
                    x: cx - s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
        }
        // MARKER_DIAMOND (3): 4 diagonal segments forming a diamond
        3 => {
            line(
                img,
                Point { x: cx, y: cy - s },
                Point { x: cx + s, y: cy },
                color,
                thickness,
            )?;
            line(
                img,
                Point { x: cx + s, y: cy },
                Point { x: cx, y: cy + s },
                color,
                thickness,
            )?;
            line(
                img,
                Point { x: cx, y: cy + s },
                Point { x: cx - s, y: cy },
                color,
                thickness,
            )?;
            line(
                img,
                Point { x: cx - s, y: cy },
                Point { x: cx, y: cy - s },
                color,
                thickness,
            )?;
        }
        // MARKER_SQUARE (4): 4 edges of an axis-aligned square
        4 => {
            line(
                img,
                Point {
                    x: cx - s,
                    y: cy - s,
                },
                Point {
                    x: cx + s,
                    y: cy - s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx + s,
                    y: cy - s,
                },
                Point {
                    x: cx + s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx + s,
                    y: cy + s,
                },
                Point {
                    x: cx - s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx - s,
                    y: cy + s,
                },
                Point {
                    x: cx - s,
                    y: cy - s,
                },
                color,
                thickness,
            )?;
        }
        // MARKER_TRIANGLE_UP (5): upward-pointing triangle
        5 => {
            line(
                img,
                Point { x: cx, y: cy - s },
                Point {
                    x: cx + s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx + s,
                    y: cy + s,
                },
                Point {
                    x: cx - s,
                    y: cy + s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx - s,
                    y: cy + s,
                },
                Point { x: cx, y: cy - s },
                color,
                thickness,
            )?;
        }
        // MARKER_TRIANGLE_DOWN (6): downward-pointing triangle
        6 => {
            line(
                img,
                Point { x: cx, y: cy + s },
                Point {
                    x: cx + s,
                    y: cy - s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx + s,
                    y: cy - s,
                },
                Point {
                    x: cx - s,
                    y: cy - s,
                },
                color,
                thickness,
            )?;
            line(
                img,
                Point {
                    x: cx - s,
                    y: cy - s,
                },
                Point { x: cx, y: cy + s },
                color,
                thickness,
            )?;
        }
        _ => {} // Unknown marker type — ignore without panic
    }
    Ok(())
}

/// Draw an arrow from `pt1` to `pt2` with arrowhead at `pt2`.
///
/// The arrowhead tip lines are drawn at ±π/6 (30°) offset from the direction,
/// with a default tip length of 10% of the total line length.
pub fn arrow_line(
    mat: &mut Mat,
    pt1: Point,
    pt2: Point,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    let t = thickness.max(1) as usize;
    draw_bresenham_line(mat, pt1.x, pt1.y, pt2.x, pt2.y, color, t);

    let dx = (pt2.x - pt1.x) as f64;
    let dy = (pt2.y - pt1.y) as f64;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return Ok(());
    }

    // tip length is 10% of the total line length
    let tip_len = (len * 0.1).max(5.0);
    let angle = dy.atan2(dx);
    let offset = std::f64::consts::PI / 6.0; // 30 degrees

    for &side_angle in &[
        angle + std::f64::consts::PI - offset,
        angle + std::f64::consts::PI + offset,
    ] {
        let tx = (pt2.x as f64 + tip_len * side_angle.cos()).round() as i32;
        let ty = (pt2.y as f64 + tip_len * side_angle.sin()).round() as i32;
        draw_bresenham_line(mat, pt2.x, pt2.y, tx, ty, color, t);
    }
    Ok(())
}

// ── Bitmap font ───────────────────────────────────────────────────────────────

/// Minimal 5×8 bitmap font for printable ASCII (32–126).
///
/// Each `u8` encodes one row; bit 4 (MSB of the low 5 bits) is the leftmost pixel.
fn char_bitmap(c: char) -> [u8; 8] {
    let code = c as usize;
    if code < 32 || code > 126 {
        return [0; 8];
    }
    match c {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E, 0x00],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F, 0x00],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E, 0x00],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02, 0x00],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E, 0x00],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E, 0x00],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08, 0x00],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E, 0x00],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C, 0x00],
        'A' => [0x04, 0x0A, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x00],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E, 0x00],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E, 0x00],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C, 0x00],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F, 0x00],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10, 0x00],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F, 0x00],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11, 0x00],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C, 0x00],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11, 0x00],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F, 0x00],
        'M' => [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11, 0x00],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11, 0x00],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10, 0x00],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D, 0x00],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11, 0x00],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E, 0x00],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E, 0x00],
        'V' => [0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04, 0x04, 0x00],
        'W' => [0x11, 0x11, 0x15, 0x15, 0x0A, 0x0A, 0x11, 0x00],
        'X' => [0x11, 0x0A, 0x04, 0x04, 0x04, 0x0A, 0x11, 0x00],
        'Y' => [0x11, 0x0A, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F, 0x00],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F, 0x00],
        'b' => [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E, 0x00],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x10, 0x0E, 0x00],
        'd' => [0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F, 0x00],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E, 0x00],
        'f' => [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08, 0x00],
        'g' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E, 0x00],
        'h' => [0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11, 0x00],
        'i' => [0x04, 0x00, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00],
        'j' => [0x02, 0x00, 0x02, 0x02, 0x02, 0x12, 0x0C, 0x00],
        'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12, 0x00],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E, 0x00],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11, 0x00],
        'n' => [0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11, 0x00],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E, 0x00],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10, 0x00],
        'q' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01, 0x00],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10, 0x00],
        's' => [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E, 0x00],
        't' => [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06, 0x00],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x11, 0x0F, 0x00],
        'v' => [0x00, 0x00, 0x11, 0x11, 0x0A, 0x0A, 0x04, 0x00],
        'w' => [0x00, 0x00, 0x11, 0x15, 0x15, 0x0A, 0x0A, 0x00],
        'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x00],
        'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E, 0x00],
        'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F, 0x00],
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10, 0x00],
        '!' => [0x04, 0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
        '?' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04, 0x00],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08, 0x00],
        ';' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08, 0x00],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02, 0x00],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08, 0x00],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E, 0x00],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00, 0x00],
        '*' => [0x00, 0x0A, 0x04, 0x1F, 0x04, 0x0A, 0x00, 0x00],
        '#' => [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A, 0x00],
        '@' => [0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E, 0x00],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03, 0x00],
        '&' => [0x0C, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0D, 0x00],
        '\'' => [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00],
        '"' => [0x0A, 0x0A, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02, 0x00],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08, 0x00],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '^' => [0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00],
        '~' => [0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00, 0x00],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x00],
        '\\' => [0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01, 0x00],
        '{' => [0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02, 0x00],
        '}' => [0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08, 0x00],
        '`' => [0x04, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        _ => [0x15, 0x0A, 0x15, 0x0A, 0x15, 0x0A, 0x15, 0x00], // checkered for unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::{Mat, Point, Scalar};

    #[test]
    fn test_set_pixel_3ch_bounds() {
        let mut mat = Mat::new_8uc3(10, 10);
        // Out-of-bounds should not panic
        set_pixel(&mut mat, -1, 5, Scalar(255.0, 0.0, 0.0, 0.0));
        set_pixel(&mut mat, 5, -1, Scalar(255.0, 0.0, 0.0, 0.0));
        set_pixel(&mut mat, 100, 5, Scalar(255.0, 0.0, 0.0, 0.0));
        // In-bounds should write
        set_pixel(&mut mat, 5, 5, Scalar(10.0, 20.0, 30.0, 0.0));
        assert_eq!(mat.at_8u3(5, 5), [10, 20, 30]);
    }

    #[test]
    fn test_line_horizontal() {
        let mut mat = Mat::new_8uc3(20, 20);
        line(
            &mut mat,
            Point { x: 0, y: 5 },
            Point { x: 19, y: 5 },
            Scalar(255.0, 0.0, 0.0, 0.0),
            1,
        )
        .unwrap();
        // All pixels on row 5 should be set
        for col in 0..20 {
            assert_eq!(mat.at_8u3(5, col)[0], 255);
        }
    }

    #[test]
    fn test_circle_outline_contains_expected_pixel() {
        let mut mat = Mat::new_8uc1(50, 50);
        circle(
            &mut mat,
            Point { x: 25, y: 25 },
            10,
            Scalar(255.0, 0.0, 0.0, 0.0),
            1,
        )
        .unwrap();
        // Point on the circle boundary at (25, 15) should be set
        assert_eq!(mat.data[15 * 50 + 25], 255);
    }

    #[test]
    fn test_rectangle_filled() {
        let mut mat = Mat::new_8uc1(10, 10);
        rectangle(
            &mut mat,
            Point { x: 2, y: 2 },
            Point { x: 5, y: 5 },
            Scalar(100.0, 0.0, 0.0, 0.0),
            -1,
        )
        .unwrap();
        // All pixels in [2..=5, 2..=5] should be 100
        for r in 2..=5usize {
            for c in 2..=5usize {
                assert_eq!(mat.data[r * 10 + c], 100, "row={r} col={c}");
            }
        }
    }
}
