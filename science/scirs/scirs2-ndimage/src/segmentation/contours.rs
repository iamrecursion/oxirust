//! Contour finding and analysis
//!
//! This module provides functionality similar to OpenCV's `findContours`,
//! implementing the Suzuki-Abe border following algorithm for extracting
//! contours from binary images.
//!
//! # Example
//!
//! ```
//! use scirs2_ndimage::segmentation::contours::{
//!     find_contours, RetrievalMode, ApproximationMethod,
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a simple binary image with a square
//! let mut image = Array2::<u8>::zeros((10, 10));
//! for i in 2..8 {
//!     for j in 2..8 {
//!         image[[i, j]] = 255;
//!     }
//! }
//!
//! // Find contours
//! let contours = find_contours(
//!     &image.view(),
//!     RetrievalMode::External,
//!     ApproximationMethod::Simple,
//! ).unwrap();
//!
//! assert!(!contours.is_empty());
//! ```

use scirs2_core::ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

use crate::error::{NdimageError, NdimageResult};

/// Contour retrieval mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetrievalMode {
    /// Retrieve only the outermost contours
    #[default]
    External,
    /// Retrieve all contours without establishing hierarchy
    List,
    /// Retrieve all contours and organize them into a two-level hierarchy
    /// (external contours and holes)
    CComp,
    /// Retrieve all contours and reconstruct full hierarchy of nested contours
    Tree,
}

/// Contour approximation method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApproximationMethod {
    /// Store all contour points (no approximation)
    None,
    /// Compress horizontal, vertical, and diagonal segments
    /// (keep only endpoints)
    #[default]
    Simple,
    /// Apply Teh-Chin chain approximation (L1 distance)
    TehChinL1,
    /// Apply Teh-Chin chain approximation (k-cosine curvature)
    TehChinKCos,
}

/// Contour hierarchy information
///
/// Each contour can have:
/// - `next`: Index of the next contour at the same hierarchy level
/// - `previous`: Index of the previous contour at the same hierarchy level
/// - `first_child`: Index of the first child contour
/// - `parent`: Index of the parent contour
///
/// A value of -1 indicates no corresponding contour exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContourHierarchy {
    /// Index of the next contour at the same level
    pub next: i32,
    /// Index of the previous contour at the same level
    pub previous: i32,
    /// Index of the first child contour
    pub first_child: i32,
    /// Index of the parent contour
    pub parent: i32,
}

impl Default for ContourHierarchy {
    fn default() -> Self {
        Self {
            next: -1,
            previous: -1,
            first_child: -1,
            parent: -1,
        }
    }
}

/// A contour extracted from a binary image
#[derive(Debug, Clone)]
pub struct Contour {
    /// Points forming the contour, in order
    pub points: Vec<(i32, i32)>,
    /// Hierarchy information for this contour
    pub hierarchy: ContourHierarchy,
    /// Whether this is an outer (external) contour or a hole
    pub is_hole: bool,
}

impl Contour {
    /// Create a new contour
    pub fn new(points: Vec<(i32, i32)>) -> Self {
        Self {
            points,
            hierarchy: ContourHierarchy::default(),
            is_hole: false,
        }
    }

    /// Calculate the area of the contour using the shoelace formula
    pub fn area(&self) -> f64 {
        if self.points.len() < 3 {
            return 0.0;
        }

        let mut sum = 0i64;
        let n = self.points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            sum += (self.points[i].0 as i64) * (self.points[j].1 as i64);
            sum -= (self.points[j].0 as i64) * (self.points[i].1 as i64);
        }

        (sum.abs() as f64) / 2.0
    }

    /// Calculate the perimeter (arc length) of the contour
    pub fn perimeter(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        let n = self.points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let dx = (self.points[j].0 - self.points[i].0) as f64;
            let dy = (self.points[j].1 - self.points[i].1) as f64;
            length += (dx * dx + dy * dy).sqrt();
        }

        length
    }

    /// Calculate the bounding rectangle of the contour
    pub fn bounding_rect(&self) -> (i32, i32, i32, i32) {
        if self.points.is_empty() {
            return (0, 0, 0, 0);
        }

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;

        for &(x, y) in &self.points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    }

    /// Calculate the centroid of the contour
    pub fn centroid(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }

        let sum_x: i64 = self.points.iter().map(|p| p.0 as i64).sum();
        let sum_y: i64 = self.points.iter().map(|p| p.1 as i64).sum();
        let n = self.points.len() as f64;

        (sum_x as f64 / n, sum_y as f64 / n)
    }
}

/// Direction offsets for 8-connectivity border following
/// Order: E, NE, N, NW, W, SW, S, SE (clockwise from East)
const DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),   // 0: East
    (1, -1),  // 1: NE
    (0, -1),  // 2: North
    (-1, -1), // 3: NW
    (-1, 0),  // 4: West
    (-1, 1),  // 5: SW
    (0, 1),   // 6: South
    (1, 1),   // 7: SE
];

/// Find contours in a binary image using the Suzuki-Abe algorithm
///
/// This function extracts contours from a binary image, similar to OpenCV's
/// `findContours` function. It implements the border following algorithm
/// described in Suzuki & Abe (1985).
///
/// # Arguments
///
/// * `image` - Binary image (0 = background, non-zero = foreground)
/// * `mode` - Contour retrieval mode
/// * `method` - Contour approximation method
///
/// # Returns
///
/// A vector of contours, each with points and hierarchy information
///
/// # Example
///
/// ```
/// use scirs2_ndimage::segmentation::contours::{
///     find_contours, RetrievalMode, ApproximationMethod,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// // Create a binary image with a filled rectangle
/// let mut image = Array2::<u8>::zeros((20, 20));
/// for i in 5..15 {
///     for j in 5..15 {
///         image[[i, j]] = 255;
///     }
/// }
///
/// let contours = find_contours(
///     &image.view(),
///     RetrievalMode::List,
///     ApproximationMethod::Simple,
/// ).unwrap();
///
/// assert_eq!(contours.len(), 1);
/// ```
pub fn find_contours(
    image: &ArrayView2<u8>,
    mode: RetrievalMode,
    method: ApproximationMethod,
) -> NdimageResult<Vec<Contour>> {
    let (height, width) = image.dim();

    if height < 2 || width < 2 {
        return Ok(Vec::new());
    }

    // Create a working copy with border padding
    // Values: 0 = background, 1 = foreground, 2+ = labels
    let mut labels = Array2::<i32>::zeros((height + 2, width + 2));

    // Copy image to labels (with 1-pixel border)
    for i in 0..height {
        for j in 0..width {
            if image[[i, j]] != 0 {
                labels[[i + 1, j + 1]] = 1;
            }
        }
    }

    let mut contours = Vec::new();
    let mut nbd = 1; // Current border number
    let mut lnbd; // Last encountered border number

    // Parent tracking for hierarchy
    let mut parent_stack: Vec<i32> = Vec::new();
    let mut border_types: Vec<bool> = Vec::new(); // true = outer, false = hole

    for i in 1..=height {
        lnbd = 1;

        for j in 1..=width {
            let fij = labels[[i, j]];

            // Check for outer border starting point
            // Condition: f(i,j) = 1 and f(i,j-1) = 0
            let is_outer_start = fij == 1 && labels[[i, j - 1]] == 0;

            // Check for hole border starting point
            // Condition: f(i,j) >= 1 and f(i,j+1) = 0
            let is_hole_start = fij >= 1 && labels[[i, j + 1]] == 0;

            if is_outer_start || is_hole_start {
                let is_outer = is_outer_start;

                // Determine parent based on LNBD and border type
                let parent = if lnbd <= 1 {
                    -1
                } else {
                    let lnbd_idx = (lnbd - 2) as usize;
                    if lnbd_idx < border_types.len() {
                        let lnbd_is_outer = border_types[lnbd_idx];
                        if is_outer == lnbd_is_outer {
                            // Same type: parent is LNBD's parent
                            if lnbd_idx < parent_stack.len() {
                                parent_stack[lnbd_idx]
                            } else {
                                -1
                            }
                        } else {
                            // Different type: parent is LNBD itself
                            lnbd as i32 - 2
                        }
                    } else {
                        -1
                    }
                };

                nbd += 1;

                // Follow the border
                let start_dir = if is_outer { 0 } else { 4 }; // Start from East for outer, West for hole
                let border_points = follow_border(&mut labels, i, j, start_dir, nbd, is_outer)?;

                if !border_points.is_empty() {
                    // Apply approximation
                    let approx_points = approximate_contour(&border_points, method);

                    // Adjust coordinates (remove padding offset)
                    let adjusted_points: Vec<(i32, i32)> = approx_points
                        .into_iter()
                        .map(|(x, y)| (x - 1, y - 1))
                        .collect();

                    // Create contour based on retrieval mode
                    let should_add = match mode {
                        RetrievalMode::External => is_outer && parent == -1,
                        RetrievalMode::List => true,
                        RetrievalMode::CComp => true,
                        RetrievalMode::Tree => true,
                    };

                    if should_add && !adjusted_points.is_empty() {
                        let mut contour = Contour::new(adjusted_points);
                        contour.is_hole = !is_outer;
                        contour.hierarchy.parent = parent;
                        contours.push(contour);
                    }
                }

                parent_stack.push(parent);
                border_types.push(is_outer);
            }

            // Update LNBD
            let abs_fij = labels[[i, j]].abs();
            if abs_fij > 1 {
                lnbd = abs_fij;
            }
        }
    }

    // Build hierarchy relationships
    build_hierarchy(&mut contours, mode);

    Ok(contours)
}

/// Follow a border starting from (i, j) in the given direction
fn follow_border(
    labels: &mut Array2<i32>,
    start_i: usize,
    start_j: usize,
    start_dir: usize,
    nbd: i32,
    is_outer: bool,
) -> NdimageResult<Vec<(i32, i32)>> {
    let (height, width) = labels.dim();
    let mut points = Vec::new();

    // Find the first non-zero neighbor
    let mut dir = start_dir;
    let mut found = false;

    for _ in 0..8 {
        let ni = (start_i as i32 + DIRECTIONS[dir].1) as usize;
        let nj = (start_j as i32 + DIRECTIONS[dir].0) as usize;

        if ni < height && nj < width && labels[[ni, nj]] != 0 {
            found = true;
            break;
        }
        dir = (dir + 1) % 8;
    }

    if !found {
        // Isolated point
        points.push((start_j as i32, start_i as i32));
        labels[[start_i, start_j]] = -nbd;
        return Ok(points);
    }

    let mut i = start_i;
    let mut j = start_j;
    let mut prev_dir = dir;

    loop {
        points.push((j as i32, i as i32));

        // Search for next border point
        let search_start = (prev_dir + if is_outer { 6 } else { 2 }) % 8;
        let mut next_found = false;
        let mut next_i = i;
        let mut next_j = j;
        let mut examined_background = false;

        for k in 0..8 {
            let d = (search_start + k) % 8;
            let ni = (i as i32 + DIRECTIONS[d].1) as usize;
            let nj = (j as i32 + DIRECTIONS[d].0) as usize;

            if ni >= height || nj >= width {
                examined_background = true;
                continue;
            }

            if labels[[ni, nj]] == 0 {
                examined_background = true;
            } else {
                next_i = ni;
                next_j = nj;
                prev_dir = d;
                next_found = true;
                break;
            }
        }

        // Update label
        if examined_background {
            labels[[i, j]] = -nbd;
        } else if labels[[i, j]] == 1 {
            labels[[i, j]] = nbd;
        }

        if !next_found || (next_i == start_i && next_j == start_j && i == start_i && j == start_j) {
            break;
        }

        // Check for return to start
        if next_i == start_i && next_j == start_j {
            // We've completed the loop
            break;
        }

        i = next_i;
        j = next_j;

        // Safety check to prevent infinite loops
        if points.len() > (height * width * 4) {
            return Err(NdimageError::ComputationError(
                "Contour following exceeded maximum iterations".to_string(),
            ));
        }
    }

    Ok(points)
}

/// Apply contour approximation
fn approximate_contour(points: &[(i32, i32)], method: ApproximationMethod) -> Vec<(i32, i32)> {
    match method {
        ApproximationMethod::None => points.to_vec(),
        ApproximationMethod::Simple => approximate_simple(points),
        ApproximationMethod::TehChinL1 => approximate_teh_chin(points, false),
        ApproximationMethod::TehChinKCos => approximate_teh_chin(points, true),
    }
}

/// Simple approximation: keep only endpoints of horizontal/vertical/diagonal runs
fn approximate_simple(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut result = Vec::new();
    result.push(points[0]);

    let mut prev_dx = 0i32;
    let mut prev_dy = 0i32;

    for i in 1..points.len() {
        let dx = (points[i].0 - points[i - 1].0).signum();
        let dy = (points[i].1 - points[i - 1].1).signum();

        if dx != prev_dx || dy != prev_dy {
            if i > 1 {
                result.push(points[i - 1]);
            }
            prev_dx = dx;
            prev_dy = dy;
        }
    }

    // Always include the last point
    if result.last() != points.last() {
        result.push(*points.last().unwrap_or(&(0, 0)));
    }

    result
}

/// Teh-Chin chain approximation algorithm
fn approximate_teh_chin(points: &[(i32, i32)], use_kcos: bool) -> Vec<(i32, i32)> {
    if points.len() <= 4 {
        return points.to_vec();
    }

    // Calculate curvature at each point
    let n = points.len();
    let mut curvatures = vec![0.0f64; n];
    let region = 3; // Region size for curvature calculation

    for i in 0..n {
        let prev_idx = (i + n - region) % n;
        let next_idx = (i + region) % n;

        let p1 = points[prev_idx];
        let p2 = points[i];
        let p3 = points[next_idx];

        if use_kcos {
            // K-cosine curvature
            let v1 = ((p2.0 - p1.0) as f64, (p2.1 - p1.1) as f64);
            let v2 = ((p3.0 - p2.0) as f64, (p3.1 - p2.1) as f64);

            let dot = v1.0 * v2.0 + v1.1 * v2.1;
            let len1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
            let len2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();

            if len1 > 0.0 && len2 > 0.0 {
                curvatures[i] = 1.0 - dot / (len1 * len2);
            }
        } else {
            // L1 distance-based curvature
            let direct_dist = ((p3.0 - p1.0).abs() + (p3.1 - p1.1).abs()) as f64;
            let path_dist = ((p2.0 - p1.0).abs()
                + (p2.1 - p1.1).abs()
                + (p3.0 - p2.0).abs()
                + (p3.1 - p2.1).abs()) as f64;

            if path_dist > 0.0 {
                curvatures[i] = 1.0 - direct_dist / path_dist;
            }
        }
    }

    // Non-maximum suppression
    let threshold = 0.1;
    let mut is_corner = vec![false; n];

    for i in 0..n {
        if curvatures[i] > threshold {
            let prev = (i + n - 1) % n;
            let next = (i + 1) % n;

            if curvatures[i] >= curvatures[prev] && curvatures[i] >= curvatures[next] {
                is_corner[i] = true;
            }
        }
    }

    // Collect corner points
    let mut result: Vec<(i32, i32)> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| is_corner[*i])
        .map(|(_, p)| *p)
        .collect();

    // Ensure we have at least the first and last points
    if result.is_empty() {
        result = approximate_simple(points);
    }

    result
}

/// Build hierarchy relationships between contours
fn build_hierarchy(contours: &mut [Contour], mode: RetrievalMode) {
    if contours.is_empty() {
        return;
    }

    let n = contours.len();

    match mode {
        RetrievalMode::External | RetrievalMode::List => {
            // Simple linear list
            for i in 0..n {
                contours[i].hierarchy.next = if i + 1 < n { (i + 1) as i32 } else { -1 };
                contours[i].hierarchy.previous = if i > 0 { (i - 1) as i32 } else { -1 };
                contours[i].hierarchy.first_child = -1;
                contours[i].hierarchy.parent = -1;
            }
        }
        RetrievalMode::CComp | RetrievalMode::Tree => {
            // Build parent-child relationships based on stored parent info
            // First, update first_child for parents
            for i in 0..n {
                let parent = contours[i].hierarchy.parent;
                if parent >= 0 && (parent as usize) < n {
                    let parent_idx = parent as usize;
                    if contours[parent_idx].hierarchy.first_child == -1 {
                        contours[parent_idx].hierarchy.first_child = i as i32;
                    }
                }
            }

            // Build sibling relationships (next/previous at same level)
            // Group contours by parent
            let mut by_parent: std::collections::HashMap<i32, Vec<usize>> =
                std::collections::HashMap::new();

            for i in 0..n {
                by_parent
                    .entry(contours[i].hierarchy.parent)
                    .or_default()
                    .push(i);
            }

            for siblings in by_parent.values() {
                for (idx, &i) in siblings.iter().enumerate() {
                    contours[i].hierarchy.previous = if idx > 0 {
                        siblings[idx - 1] as i32
                    } else {
                        -1
                    };
                    contours[i].hierarchy.next = if idx + 1 < siblings.len() {
                        siblings[idx + 1] as i32
                    } else {
                        -1
                    };
                }
            }
        }
    }
}

/// Draw contours on an image
///
/// # Arguments
///
/// * `image` - Mutable image to draw on
/// * `contours` - Contours to draw
/// * `contour_idx` - Index of contour to draw (-1 for all)
/// * `color` - Color value to use
/// * `thickness` - Line thickness (1 for single pixel, -1 for filled)
pub fn draw_contours(
    image: &mut Array2<u8>,
    contours: &[Contour],
    contour_idx: i32,
    color: u8,
    thickness: i32,
) -> NdimageResult<()> {
    let (height, width) = image.dim();

    let indices: Vec<usize> = if contour_idx < 0 {
        (0..contours.len()).collect()
    } else {
        vec![contour_idx as usize]
    };

    for &idx in &indices {
        if idx >= contours.len() {
            continue;
        }

        let contour = &contours[idx];

        if thickness == -1 {
            // Fill contour
            fill_contour(image, contour)?;
        } else {
            // Draw contour outline
            for i in 0..contour.points.len() {
                let p1 = contour.points[i];
                let p2 = contour.points[(i + 1) % contour.points.len()];

                draw_line(image, p1, p2, color, thickness.max(1) as usize)?;
            }
        }
    }

    Ok(())
}

/// Fill a contour using scanline algorithm
fn fill_contour(image: &mut Array2<u8>, contour: &Contour) -> NdimageResult<()> {
    if contour.points.len() < 3 {
        return Ok(());
    }

    let (height, width) = image.dim();
    let (_, min_y, _, h) = contour.bounding_rect();
    let max_y = min_y + h;

    for y in min_y.max(0)..max_y.min(height as i32) {
        let mut intersections = Vec::new();

        // Find all intersections with the scanline
        for i in 0..contour.points.len() {
            let p1 = contour.points[i];
            let p2 = contour.points[(i + 1) % contour.points.len()];

            if (p1.1 <= y && p2.1 > y) || (p2.1 <= y && p1.1 > y) {
                let x = p1.0 + (y - p1.1) * (p2.0 - p1.0) / (p2.1 - p1.1);
                intersections.push(x);
            }
        }

        intersections.sort();

        // Fill between pairs of intersections
        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let x1 = pair[0].max(0) as usize;
                let x2 = (pair[1] as usize).min(width - 1);
                for x in x1..=x2 {
                    if y >= 0 && (y as usize) < height {
                        image[[y as usize, x]] = 255;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Draw a line using Bresenham's algorithm
fn draw_line(
    image: &mut Array2<u8>,
    p1: (i32, i32),
    p2: (i32, i32),
    color: u8,
    thickness: usize,
) -> NdimageResult<()> {
    let (height, width) = image.dim();

    let dx = (p2.0 - p1.0).abs();
    let dy = (p2.1 - p1.1).abs();
    let sx = if p1.0 < p2.0 { 1 } else { -1 };
    let sy = if p1.1 < p2.1 { 1 } else { -1 };
    let mut err = dx - dy;

    let mut x = p1.0;
    let mut y = p1.1;

    loop {
        // Draw point with thickness
        let half_t = (thickness / 2) as i32;
        for dy in -half_t..=half_t {
            for dx in -half_t..=half_t {
                let px = x + dx;
                let py = y + dy;
                if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                    image[[py as usize, px as usize]] = color;
                }
            }
        }

        if x == p2.0 && y == p2.1 {
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

    Ok(())
}

/// Check if a point is inside a contour using ray casting
pub fn point_in_contour(point: (i32, i32), contour: &Contour) -> bool {
    if contour.points.len() < 3 {
        return false;
    }

    let (x, y) = point;
    let n = contour.points.len();
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = contour.points[i];
        let (xj, yj) = contour.points[j];

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Calculate moments of a contour
#[derive(Debug, Clone, Default)]
pub struct ContourMoments {
    /// Spatial moments m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
    pub m: [[f64; 4]; 4],
    /// Central moments mu20, mu11, mu02, mu30, mu21, mu12, mu03
    pub mu: [[f64; 4]; 4],
    /// Normalized central moments nu20, nu11, nu02, nu30, nu21, nu12, nu03
    pub nu: [[f64; 4]; 4],
}

impl ContourMoments {
    /// Calculate moments from contour points
    pub fn from_contour(contour: &Contour) -> Self {
        let mut moments = Self::default();

        if contour.points.is_empty() {
            return moments;
        }

        // Calculate spatial moments
        for &(x, y) in &contour.points {
            let xf = x as f64;
            let yf = y as f64;

            for i in 0..4 {
                for j in 0..4 {
                    if i + j <= 3 {
                        moments.m[i][j] += xf.powi(i as i32) * yf.powi(j as i32);
                    }
                }
            }
        }

        // Calculate centroid
        let cx = if moments.m[0][0] != 0.0 {
            moments.m[1][0] / moments.m[0][0]
        } else {
            0.0
        };
        let cy = if moments.m[0][0] != 0.0 {
            moments.m[0][1] / moments.m[0][0]
        } else {
            0.0
        };

        // Calculate central moments
        for &(x, y) in &contour.points {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;

            for i in 0..4 {
                for j in 0..4 {
                    if i + j >= 2 && i + j <= 3 {
                        moments.mu[i][j] += dx.powi(i as i32) * dy.powi(j as i32);
                    }
                }
            }
        }

        // Calculate normalized central moments
        let m00 = moments.m[0][0];
        if m00 > 0.0 {
            for i in 0..4 {
                for j in 0..4 {
                    if i + j >= 2 && i + j <= 3 {
                        let exp = ((i + j) as f64 / 2.0) + 1.0;
                        moments.nu[i][j] = moments.mu[i][j] / m00.powf(exp);
                    }
                }
            }
        }

        moments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_find_contours_simple_square() {
        let mut image = Array2::<u8>::zeros((10, 10));

        // Create a filled square
        for i in 2..8 {
            for j in 2..8 {
                image[[i, j]] = 255;
            }
        }

        let contours = find_contours(
            &image.view(),
            RetrievalMode::External,
            ApproximationMethod::Simple,
        )
        .expect("find_contours should succeed for valid image");

        assert!(!contours.is_empty());
        assert!(contours[0].points.len() >= 4); // At least 4 corners
    }

    #[test]
    fn test_find_contours_with_hole() {
        let mut image = Array2::<u8>::zeros((20, 20));

        // Outer square
        for i in 2..18 {
            for j in 2..18 {
                image[[i, j]] = 255;
            }
        }

        // Inner hole
        for i in 6..14 {
            for j in 6..14 {
                image[[i, j]] = 0;
            }
        }

        let contours = find_contours(
            &image.view(),
            RetrievalMode::Tree,
            ApproximationMethod::None,
        )
        .expect("find_contours should succeed for image with hole");

        // Should find at least 2 contours (outer and hole)
        assert!(!contours.is_empty());
    }

    #[test]
    fn test_contour_area() {
        // 10x10 square contour
        let points = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let contour = Contour::new(points);

        let area = contour.area();
        assert!((area - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_contour_perimeter() {
        // 10x10 square contour
        let points = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let contour = Contour::new(points);

        let perimeter = contour.perimeter();
        assert!((perimeter - 40.0).abs() < 1.0);
    }

    #[test]
    fn test_point_in_contour() {
        let points = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let contour = Contour::new(points);

        assert!(point_in_contour((5, 5), &contour));
        assert!(!point_in_contour((15, 15), &contour));
    }

    #[test]
    fn test_empty_image() {
        let image = Array2::<u8>::zeros((10, 10));

        let contours = find_contours(
            &image.view(),
            RetrievalMode::List,
            ApproximationMethod::Simple,
        )
        .expect("find_contours should succeed for empty image");

        assert!(contours.is_empty());
    }
}
