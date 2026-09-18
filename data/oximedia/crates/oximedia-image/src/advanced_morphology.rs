//! Advanced morphological image processing operations.
//!
//! This module extends the basic morphology in `morphology.rs` with:
//!
//! - A flexible `StructuringElement` that supports arbitrary masks (rectangle,
//!   disk, cross, angled line).
//! - Grayscale erosion, dilation, opening, closing, morphological gradient,
//!   top-hat, and black-hat transforms on `u8` images.
//! - The hit-or-miss transform on binary (`bool`) images.
//! - Connected-component labelling via union-find (4-connectivity).
//!
//! All functions accept flat row-major pixel buffers and return new `Vec`s;
//! no in-place aliasing is required.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Structuring element
// ---------------------------------------------------------------------------

/// An arbitrary-shape structuring element for morphological operations.
///
/// The element is defined by a flat `bool` mask of size `width × height` and
/// an *origin* (anchor point) which is typically the centre pixel.  Active
/// pixels (`true`) participate in the morphological operation.
#[derive(Debug, Clone)]
pub struct StructuringElement {
    /// Flat, row-major mask.  `data[y * width + x]` is `true` if the pixel at
    /// offset `(x - origin.0, y - origin.1)` relative to the anchor is active.
    pub data: Vec<bool>,
    /// Width of the mask grid.
    pub width: u32,
    /// Height of the mask grid.
    pub height: u32,
    /// Anchor point (column, row) within the mask.  Usually the centre.
    pub origin: (u32, u32),
}

impl StructuringElement {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a fully-active rectangular structuring element of size `w × h`.
    ///
    /// The origin is placed at `(w/2, h/2)` (integer division).
    #[must_use]
    pub fn rectangle(w: u32, h: u32) -> Self {
        let count = (w as usize) * (h as usize);
        Self {
            data: vec![true; count],
            width: w,
            height: h,
            origin: (w / 2, h / 2),
        }
    }

    /// Create a disk-shaped (circular) structuring element.
    ///
    /// The bounding box is `(2*radius+1) × (2*radius+1)`.  A pixel at offset
    /// `(dx, dy)` from the centre is active when `dx²+dy² ≤ radius²`.
    #[must_use]
    pub fn circle(radius: u32) -> Self {
        let side = 2 * radius + 1;
        let r_sq = (radius as i64) * (radius as i64);
        let data: Vec<bool> = (0..side)
            .flat_map(|y| {
                (0..side).map(move |x| {
                    let dx = x as i64 - radius as i64;
                    let dy = y as i64 - radius as i64;
                    dx * dx + dy * dy <= r_sq
                })
            })
            .collect();
        Self {
            data,
            width: side,
            height: side,
            origin: (radius, radius),
        }
    }

    /// Create a cross (plus-sign) structuring element.
    ///
    /// `size` is the *arm length* in pixels extending from the centre; the
    /// bounding box is `(2*size+1) × (2*size+1)`.  Only pixels on the
    /// horizontal and vertical axes through the centre are active.
    #[must_use]
    pub fn cross(size: u32) -> Self {
        let side = 2 * size + 1;
        let data: Vec<bool> = (0..side)
            .flat_map(|y| (0..side).map(move |x| x == size || y == size))
            .collect();
        Self {
            data,
            width: side,
            height: side,
            origin: (size, size),
        }
    }

    /// Create an angled line structuring element.
    ///
    /// The line passes through the centre of a bounding box of approximately
    /// `length × length` pixels and is oriented at `angle_deg` degrees
    /// measured counter-clockwise from the positive x-axis.  Pixels that fall
    /// within half a pixel of the idealised line are activated.
    #[must_use]
    pub fn line(length: u32, angle_deg: f32) -> Self {
        let half = length as i32 / 2;
        let side = (2 * half + 1) as u32;
        let angle_rad = angle_deg.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let data: Vec<bool> = (0..side)
            .flat_map(|y| {
                (0..side).map(move |x| {
                    let dx = x as f32 - half as f32;
                    let dy = y as f32 - half as f32;
                    // Distance from the point to the line through the origin
                    // at angle `angle_rad`: |dx·sin - dy·cos|
                    let dist = (dx * sin_a - dy * cos_a).abs();
                    // Also restrict to the line segment (projection within ±half)
                    let proj = dx * cos_a + dy * sin_a;
                    dist <= 0.5 && proj.abs() <= half as f32 + 0.5
                })
            })
            .collect();
        Self {
            data,
            width: side,
            height: side,
            origin: (half as u32, half as u32),
        }
    }

    /// Number of active pixels in the element.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }
}

// ---------------------------------------------------------------------------
// Grayscale morphology on u8 images
// ---------------------------------------------------------------------------

/// Grayscale erosion: replace each pixel with the minimum value inside the
/// structuring element window.
///
/// Pixels outside the image boundary are treated as `u8::MAX` (neutral element
/// for the min operation), so the erosion does not artificially darken borders.
///
/// **Dispatch**: flat rectangular and axis-aligned line SEs use the O(1)
/// van Herk–Gil–Werman algorithm; all other shapes fall through to brute force.
#[must_use]
pub fn morphology_erode(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
) -> Vec<u8> {
    use crate::morphology_vhgw;
    if se_is_flat_rect(se) {
        return morphology_vhgw::erode_rect_vhgw(pixels, width, height, se.width, se.height);
    }
    if let Some((length, horizontal)) = se_is_axis_aligned_line(se) {
        return morphology_vhgw::erode_line_vhgw(pixels, width, height, length, horizontal);
    }
    morph_op(pixels, width, height, se, u8::MAX, u8::min)
}

/// Grayscale dilation: replace each pixel with the maximum value inside the
/// structuring element window.
///
/// Pixels outside the image boundary are treated as `u8::MIN` (neutral element
/// for the max operation).
///
/// **Dispatch**: flat rectangular and axis-aligned line SEs use the O(1)
/// van Herk–Gil–Werman algorithm; all other shapes fall through to brute force.
#[must_use]
pub fn morphology_dilate(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
) -> Vec<u8> {
    use crate::morphology_vhgw;
    if se_is_flat_rect(se) {
        return morphology_vhgw::dilate_rect_vhgw(pixels, width, height, se.width, se.height);
    }
    if let Some((length, horizontal)) = se_is_axis_aligned_line(se) {
        return morphology_vhgw::dilate_line_vhgw(pixels, width, height, length, horizontal);
    }
    morph_op(pixels, width, height, se, u8::MIN, u8::max)
}

/// Returns `true` when `se` is a fully-active flat rectangle whose origin is
/// at the centre (as produced by `StructuringElement::rectangle`).
fn se_is_flat_rect(se: &StructuringElement) -> bool {
    se.origin == (se.width / 2, se.height / 2) && se.data.iter().all(|&v| v)
}

/// Returns `Some((length, horizontal))` when `se` is an axis-aligned,
/// fully-active, centred line SE, or `None` otherwise.
fn se_is_axis_aligned_line(se: &StructuringElement) -> Option<(u32, bool)> {
    // Horizontal line: height == 1, all active, centred
    if se.height == 1 && se.origin == (se.width / 2, 0) && se.data.iter().all(|&v| v) {
        return Some((se.width, true));
    }
    // Vertical line: width == 1, all active, centred
    if se.width == 1 && se.origin == (0, se.height / 2) && se.data.iter().all(|&v| v) {
        return Some((se.height, false));
    }
    None
}

/// Grayscale opening: erosion followed by dilation.  Removes small bright
/// structures while preserving the overall shape of larger bright regions.
#[must_use]
pub fn morphology_open(pixels: &[u8], width: u32, height: u32, se: &StructuringElement) -> Vec<u8> {
    let eroded = morphology_erode(pixels, width, height, se);
    morphology_dilate(&eroded, width, height, se)
}

/// Grayscale closing: dilation followed by erosion.  Fills small dark holes
/// while preserving the overall shape of darker regions.
#[must_use]
pub fn morphology_close(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
) -> Vec<u8> {
    let dilated = morphology_dilate(pixels, width, height, se);
    morphology_erode(&dilated, width, height, se)
}

/// Morphological gradient: `dilate - erode` (saturating subtraction).
///
/// Highlights edges in the image.
#[must_use]
pub fn morphology_gradient(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
) -> Vec<u8> {
    let dilated = morphology_dilate(pixels, width, height, se);
    let eroded = morphology_erode(pixels, width, height, se);
    dilated
        .iter()
        .zip(eroded.iter())
        .map(|(&d, &e)| d.saturating_sub(e))
        .collect()
}

/// White top-hat transform: `original - open`.
///
/// Extracts small bright features that are smaller than the structuring
/// element.
#[must_use]
pub fn morphology_top_hat(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
) -> Vec<u8> {
    let opened = morphology_open(pixels, width, height, se);
    pixels
        .iter()
        .zip(opened.iter())
        .map(|(&p, &o)| p.saturating_sub(o))
        .collect()
}

/// Black top-hat (bottom-hat) transform: `close - original`.
///
/// Extracts small dark features that are smaller than the structuring element.
#[must_use]
pub fn morphology_black_hat(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
) -> Vec<u8> {
    let closed = morphology_close(pixels, width, height, se);
    closed
        .iter()
        .zip(pixels.iter())
        .map(|(&c, &p)| c.saturating_sub(p))
        .collect()
}

// ---------------------------------------------------------------------------
// Binary operations
// ---------------------------------------------------------------------------

/// Hit-or-miss transform on a binary image.
///
/// A pixel is set to `true` in the output when the foreground structuring
/// element `se_fg` matches the foreground (`true`) pixels *and* the background
/// structuring element `se_bg` matches the background (`false`) pixels in a
/// complementary region.
///
/// Both structuring elements must have the same dimensions and origin.
///
/// Out-of-bounds pixels are treated as `false` (background).
#[must_use]
pub fn hit_or_miss(
    pixels: &[bool],
    width: u32,
    height: u32,
    se_fg: &StructuringElement,
    se_bg: &StructuringElement,
) -> Vec<bool> {
    let w = width as usize;
    let h = height as usize;
    let mut output = vec![false; w * h];

    for cy in 0..h {
        for cx in 0..w {
            let hit_fg = se_matches(pixels, w, h, cx, cy, se_fg, true);
            let hit_bg = se_matches(pixels, w, h, cx, cy, se_bg, false);
            output[cy * w + cx] = hit_fg && hit_bg;
        }
    }

    output
}

/// Check whether `se` matches the expected `target` value for all active
/// pixels when centred at `(cx, cy)`.
fn se_matches(
    pixels: &[bool],
    w: usize,
    h: usize,
    cx: usize,
    cy: usize,
    se: &StructuringElement,
    target: bool,
) -> bool {
    let ox = se.origin.0 as isize;
    let oy = se.origin.1 as isize;
    for ky in 0..se.height as usize {
        for kx in 0..se.width as usize {
            if !se.data[ky * se.width as usize + kx] {
                continue;
            }
            let sx = cx as isize + kx as isize - ox;
            let sy = cy as isize + ky as isize - oy;
            let pixel = if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                pixels[sy as usize * w + sx as usize]
            } else {
                false
            };
            if pixel != target {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Connected-component labelling (union-find, 4-connectivity)
// ---------------------------------------------------------------------------

/// Label connected components in a binary image using 4-connectivity.
///
/// Returns `(label_image, component_count)` where:
/// - `label_image` has the same length as `binary` (one label per pixel).
/// - Background (`false`) pixels get label `0`.
/// - Foreground (`true`) pixels get labels `1..=component_count`.
///
/// The algorithm is a two-pass approach with union-find path compression.
#[must_use]
pub fn connected_components(binary: &[bool], width: u32, height: u32) -> (Vec<u32>, u32) {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    if n == 0 {
        return (vec![], 0);
    }

    // First pass: assign provisional labels and build equivalence table.
    let mut labels = vec![0u32; n];
    let mut parent: Vec<u32> = Vec::new(); // parent[0] is unused; real labels start at 1
    parent.push(0); // placeholder

    let mut next_label = 1u32;

    for y in 0..h {
        for x in 0..w {
            if !binary[y * w + x] {
                continue;
            }
            // Neighbour labels (left and top)
            let left = if x > 0 { labels[y * w + x - 1] } else { 0 };
            let top = if y > 0 { labels[(y - 1) * w + x] } else { 0 };

            match (left, top) {
                (0, 0) => {
                    // New component
                    labels[y * w + x] = next_label;
                    parent.push(next_label); // points to itself
                    next_label += 1;
                }
                (l, 0) | (0, l) => {
                    labels[y * w + x] = find(&mut parent, l);
                }
                (l, t) => {
                    let rl = find(&mut parent, l);
                    let rt = find(&mut parent, t);
                    if rl != rt {
                        union(&mut parent, rl, rt);
                    }
                    labels[y * w + x] = find(&mut parent, rl.min(rt));
                }
            }
        }
    }

    // Second pass: flatten labels and renumber.
    let mut remap = vec![0u32; parent.len()];
    let mut counter = 0u32;
    for i in 1..parent.len() {
        let root = find(&mut parent, i as u32);
        if remap[root as usize] == 0 {
            counter += 1;
            remap[root as usize] = counter;
        }
    }

    for label in labels.iter_mut() {
        if *label > 0 {
            let root = find(&mut parent, *label);
            *label = remap[root as usize];
        }
    }

    (labels, counter)
}

/// Find the root of the set containing `x`, with path compression.
fn find(parent: &mut Vec<u32>, x: u32) -> u32 {
    let mut root = x;
    // Walk to root
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    // Path compression
    let mut node = x;
    while parent[node as usize] != root {
        let next = parent[node as usize];
        parent[node as usize] = root;
        node = next;
    }
    root
}

/// Union two sets by pointing the larger root at the smaller root (union by
/// label index used as a simple rank proxy).
fn union(parent: &mut Vec<u32>, a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        // Attach larger label to smaller label to keep labelling stable
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generic morphological operator that iterates the structuring element and
/// folds pixel values with `combine`, starting from `init`.
fn morph_op<F>(
    pixels: &[u8],
    width: u32,
    height: u32,
    se: &StructuringElement,
    init: u8,
    combine: F,
) -> Vec<u8>
where
    F: Fn(u8, u8) -> u8,
{
    let w = width as usize;
    let h = height as usize;
    let ox = se.origin.0 as isize;
    let oy = se.origin.1 as isize;
    let mut output = vec![0u8; w * h];

    for cy in 0..h {
        for cx in 0..w {
            let mut acc = init;
            for ky in 0..se.height as usize {
                for kx in 0..se.width as usize {
                    if !se.data[ky * se.width as usize + kx] {
                        continue;
                    }
                    let sx = cx as isize + kx as isize - ox;
                    let sy = cy as isize + ky as isize - oy;
                    let val = if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                        pixels[sy as usize * w + sx as usize]
                    } else {
                        init // neutral element — doesn't affect the fold
                    };
                    acc = combine(acc, val);
                }
            }
            output[cy * w + cx] = acc;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // StructuringElement constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_rectangle_all_active() {
        let se = StructuringElement::rectangle(3, 5);
        assert_eq!(se.width, 3);
        assert_eq!(se.height, 5);
        assert_eq!(se.active_count(), 15);
        assert_eq!(se.origin, (1, 2));
    }

    #[test]
    fn test_circle_radius_0() {
        let se = StructuringElement::circle(0);
        assert_eq!(se.active_count(), 1);
        assert!(se.data[0]);
    }

    #[test]
    fn test_circle_radius_1() {
        let se = StructuringElement::circle(1);
        // Disk of radius 1 in a 3×3 grid: 5 pixels (plus-shape without diagonals)
        assert_eq!(se.active_count(), 5);
    }

    #[test]
    fn test_circle_radius_2_centre_active() {
        let se = StructuringElement::circle(2);
        let mid = se.origin.1 as usize * se.width as usize + se.origin.0 as usize;
        assert!(se.data[mid], "centre must be active");
    }

    #[test]
    fn test_cross_arm_length_1() {
        let se = StructuringElement::cross(1);
        assert_eq!(se.width, 3);
        assert_eq!(se.height, 3);
        // Cross: 5 pixels active (centre + 4 arms)
        assert_eq!(se.active_count(), 5);
    }

    #[test]
    fn test_line_horizontal() {
        // angle = 0° → horizontal line
        let se = StructuringElement::line(5, 0.0);
        // All pixels on the centre row should be active
        let mid_row = se.origin.1 as usize;
        let active_in_row: usize = (0..se.width as usize)
            .filter(|&x| se.data[mid_row * se.width as usize + x])
            .count();
        // At minimum the centre pixel should be active
        assert!(
            active_in_row >= 1,
            "horizontal line must have active pixels"
        );
    }

    // -----------------------------------------------------------------------
    // Grayscale morphology
    // -----------------------------------------------------------------------

    #[test]
    fn test_erode_uniform_image() {
        let se = StructuringElement::rectangle(3, 3);
        let pixels = vec![128u8; 25];
        let result = morphology_erode(&pixels, 5, 5, &se);
        assert!(
            result.iter().all(|&v| v == 128),
            "erode on uniform image should be unchanged"
        );
    }

    #[test]
    fn test_dilate_uniform_image() {
        let se = StructuringElement::rectangle(3, 3);
        let pixels = vec![200u8; 25];
        let result = morphology_dilate(&pixels, 5, 5, &se);
        assert!(
            result.iter().all(|&v| v == 200),
            "dilate on uniform image should be unchanged"
        );
    }

    #[test]
    fn test_dilate_single_bright_pixel() {
        let se = StructuringElement::rectangle(3, 3);
        let mut pixels = vec![0u8; 25];
        pixels[12] = 255; // centre of 5×5
        let result = morphology_dilate(&pixels, 5, 5, &se);
        // The 3×3 neighbourhood around centre should now all be 255
        assert_eq!(result[6], 255);
        assert_eq!(result[7], 255);
        assert_eq!(result[8], 255);
        assert_eq!(result[11], 255);
        assert_eq!(result[12], 255);
        assert_eq!(result[13], 255);
        assert_eq!(result[16], 255);
        assert_eq!(result[17], 255);
        assert_eq!(result[18], 255);
    }

    #[test]
    fn test_erode_single_dark_pixel() {
        let se = StructuringElement::rectangle(3, 3);
        let mut pixels = vec![255u8; 25];
        pixels[12] = 0; // single dark pixel in centre of 5×5
        let result = morphology_erode(&pixels, 5, 5, &se);
        // The 3×3 neighbourhood around the dark pixel should all become 0
        assert_eq!(result[12], 0);
    }

    #[test]
    fn test_open_removes_isolated_bright_pixel() {
        let se = StructuringElement::rectangle(3, 3);
        let mut pixels = vec![0u8; 25];
        pixels[12] = 255; // isolated bright pixel
        let result = morphology_open(&pixels, 5, 5, &se);
        // Opening should eliminate it
        assert!(
            result.iter().all(|&v| v == 0),
            "opening should remove isolated bright pixel"
        );
    }

    #[test]
    fn test_close_fills_isolated_dark_pixel() {
        let se = StructuringElement::rectangle(3, 3);
        let mut pixels = vec![255u8; 25];
        pixels[12] = 0; // isolated dark pixel
        let result = morphology_close(&pixels, 5, 5, &se);
        assert_eq!(result[12], 255, "closing should fill isolated dark pixel");
    }

    #[test]
    fn test_gradient_uniform_image_is_zero() {
        let se = StructuringElement::rectangle(3, 3);
        let pixels = vec![128u8; 25];
        let result = morphology_gradient(&pixels, 5, 5, &se);
        // Interior pixels: dilate == erode == 128 → gradient = 0
        // We only check the interior pixel
        assert_eq!(result[12], 0, "gradient of uniform image should be 0");
    }

    #[test]
    fn test_top_hat_uniform_image_is_zero() {
        let se = StructuringElement::rectangle(3, 3);
        let pixels = vec![100u8; 25];
        let result = morphology_top_hat(&pixels, 5, 5, &se);
        // top-hat of uniform image = 0 everywhere
        assert!(result.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_black_hat_uniform_image_is_zero() {
        let se = StructuringElement::rectangle(3, 3);
        let pixels = vec![100u8; 25];
        let result = morphology_black_hat(&pixels, 5, 5, &se);
        assert!(result.iter().all(|&v| v == 0));
    }

    // -----------------------------------------------------------------------
    // Hit-or-miss
    // -----------------------------------------------------------------------

    #[test]
    fn test_hit_or_miss_basic() {
        // A 3×3 image with a single foreground pixel at centre
        #[rustfmt::skip]
        let binary = vec![
            false, false, false,
            false, true,  false,
            false, false, false,
        ];
        // SE_fg: single pixel (detect isolated point)
        let se_fg = StructuringElement {
            data: vec![true],
            width: 1,
            height: 1,
            origin: (0, 0),
        };
        // SE_bg: not applicable (empty background requirement)
        let se_bg = StructuringElement {
            data: vec![false],
            width: 1,
            height: 1,
            origin: (0, 0),
        };
        let result = hit_or_miss(&binary, 3, 3, &se_fg, &se_bg);
        // The centre pixel should be detected
        assert!(result[4], "centre should be hit");
    }

    // -----------------------------------------------------------------------
    // Connected components
    // -----------------------------------------------------------------------

    #[test]
    fn test_connected_components_empty() {
        let binary = vec![false; 9];
        let (labels, count) = connected_components(&binary, 3, 3);
        assert_eq!(count, 0);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_connected_components_single_blob() {
        #[rustfmt::skip]
        let binary = vec![
            false, false, false,
            false, true,  false,
            false, false, false,
        ];
        let (labels, count) = connected_components(&binary, 3, 3);
        assert_eq!(count, 1);
        assert_eq!(labels[4], 1);
    }

    #[test]
    fn test_connected_components_two_blobs() {
        #[rustfmt::skip]
        let binary = vec![
            true,  false, true,
            false, false, false,
            false, false, false,
        ];
        let (labels, count) = connected_components(&binary, 3, 3);
        assert_eq!(count, 2, "should detect two separate blobs");
        assert_ne!(labels[0], labels[2], "blobs should have different labels");
        assert_eq!(labels[1], 0, "background pixel should be 0");
    }

    #[test]
    fn test_connected_components_connected_row() {
        // All true in a single row → one component
        let binary = vec![true; 5];
        let (labels, count) = connected_components(&binary, 5, 1);
        assert_eq!(count, 1);
        assert!(labels.iter().all(|&l| l == 1));
    }
}
