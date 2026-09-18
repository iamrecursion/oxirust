//! Image segmentation algorithms: Otsu thresholding, watershed, and SLIC superpixels.
//!
//! All algorithms work on grayscale u8 images in row-major order.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// SegmentMap
// ---------------------------------------------------------------------------

/// Result of a segmentation operation.
///
/// Each pixel is assigned a label (u32). Label 0 is typically background or the
/// first segment, depending on the algorithm.
#[derive(Debug, Clone)]
pub struct SegmentMap {
    /// Per-pixel label (row-major).
    pub labels: Vec<u32>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Total number of distinct labels.
    pub num_labels: u32,
}

impl SegmentMap {
    /// Create a new segment map with all pixels labelled 0.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            labels: vec![0; (width as usize) * (height as usize)],
            width,
            height,
            num_labels: 1,
        }
    }

    /// Get the label at (x, y), or None if out of bounds.
    #[must_use]
    pub fn get_label(&self, x: u32, y: u32) -> Option<u32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.labels[(y as usize) * (self.width as usize) + (x as usize)])
    }

    /// Number of pixels in a given segment.
    #[must_use]
    pub fn segment_size(&self, label: u32) -> usize {
        self.labels.iter().filter(|&&l| l == label).count()
    }

    /// All unique labels present in the map.
    #[must_use]
    pub fn unique_labels(&self) -> Vec<u32> {
        let mut seen = std::collections::BTreeSet::new();
        for &l in &self.labels {
            seen.insert(l);
        }
        seen.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Otsu thresholding
// ---------------------------------------------------------------------------

/// Compute the optimal threshold using Otsu's method.
///
/// Minimizes intra-class variance (equivalently, maximizes inter-class variance)
/// over all possible thresholds 0–255.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn otsu_threshold(image: &[u8]) -> u8 {
    if image.is_empty() {
        return 128;
    }

    // Build histogram
    let mut hist = [0u64; 256];
    for &v in image {
        hist[v as usize] += 1;
    }

    let total = image.len() as f64;
    let mut sum_total = 0.0_f64;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut best_threshold = 0u8;
    let mut best_variance = 0.0_f64;

    let mut weight_bg = 0.0_f64;
    let mut sum_bg = 0.0_f64;

    for t in 0..=254u8 {
        weight_bg += hist[t as usize] as f64;
        if weight_bg < 1e-12 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg < 1e-12 {
            break;
        }

        sum_bg += t as f64 * hist[t as usize] as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;

        let between_var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if between_var > best_variance {
            best_variance = between_var;
            best_threshold = t;
        }
    }

    best_threshold
}

/// Thresholding method for adaptive segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMethod {
    /// Otsu's method: minimizes intra-class variance.
    Otsu,
    /// Triangle method: geometric approach using the histogram peak.
    Triangle,
}

/// Compute the optimal threshold using the triangle method.
///
/// The triangle algorithm works by finding the histogram peak, then drawing
/// a line from the peak to the farthest non-zero bin. The threshold is chosen
/// at the point of maximum perpendicular distance from this line to the
/// histogram envelope. Works well for unimodal distributions with a long tail.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn triangle_threshold(image: &[u8]) -> u8 {
    if image.is_empty() {
        return 128;
    }

    // Build histogram
    let mut hist = [0u64; 256];
    for &v in image {
        hist[v as usize] += 1;
    }

    // Find the histogram peak (mode)
    let mut peak_bin = 0usize;
    let mut peak_val = 0u64;
    for (i, &count) in hist.iter().enumerate() {
        if count > peak_val {
            peak_val = count;
            peak_bin = i;
        }
    }

    // Find the farthest non-zero bin from the peak
    let mut first_nonzero = 0usize;
    let mut last_nonzero = 255usize;
    for i in 0..256 {
        if hist[i] > 0 {
            first_nonzero = i;
            break;
        }
    }
    for i in (0..256).rev() {
        if hist[i] > 0 {
            last_nonzero = i;
            break;
        }
    }

    // Determine which end is farther from the peak
    let (start, end, flip) = if (peak_bin - first_nonzero) >= (last_nonzero - peak_bin) {
        // Peak is closer to the right end; line from peak to first_nonzero
        (first_nonzero, peak_bin, true)
    } else {
        // Peak is closer to the left end; line from peak to last_nonzero
        (peak_bin, last_nonzero, false)
    };

    if start == end {
        return peak_bin as u8;
    }

    // Line from (start, hist[start]) to (end, hist[end])
    // Find point of maximum perpendicular distance
    let x1 = start as f64;
    let y1 = hist[start] as f64;
    let x2 = end as f64;
    let y2 = hist[end] as f64;

    let line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
    if line_len < 1e-12 {
        return peak_bin as u8;
    }

    // Normal form: (y2-y1)*x - (x2-x1)*y + x2*y1 - x1*y2 = 0
    let a = y2 - y1;
    let b = -(x2 - x1);
    let c = x2 * y1 - x1 * y2;
    let norm = (a * a + b * b).sqrt();

    let mut best_threshold = start;
    let mut best_distance = 0.0_f64;

    let range_start = if flip { start } else { start + 1 };
    let range_end = if flip { end } else { end };

    for i in range_start..range_end {
        let xi = i as f64;
        let yi = hist[i] as f64;
        let dist = (a * xi + b * yi + c).abs() / norm;
        if dist > best_distance {
            best_distance = dist;
            best_threshold = i;
        }
    }

    best_threshold as u8
}

/// Automatically compute threshold using the specified method.
///
/// Returns the computed threshold value.
#[must_use]
pub fn auto_threshold(image: &[u8], method: ThresholdMethod) -> u8 {
    match method {
        ThresholdMethod::Otsu => otsu_threshold(image),
        ThresholdMethod::Triangle => triangle_threshold(image),
    }
}

/// Adaptive binary segmentation using the specified thresholding method.
///
/// Automatically computes the optimal threshold, then applies it.
#[must_use]
pub fn adaptive_threshold_segment(
    image: &[u8],
    width: u32,
    height: u32,
    method: ThresholdMethod,
) -> SegmentMap {
    let threshold = auto_threshold(image, method);
    threshold_segment(image, width, height, threshold)
}

/// Binary segmentation using a threshold.
///
/// Pixels below `threshold` get label 0, pixels >= threshold get label 1.
#[must_use]
pub fn threshold_segment(image: &[u8], width: u32, height: u32, threshold: u8) -> SegmentMap {
    let n = (width as usize) * (height as usize);
    let labels: Vec<u32> = if image.len() >= n {
        image[..n]
            .iter()
            .map(|&v| if v >= threshold { 1 } else { 0 })
            .collect()
    } else {
        vec![0; n]
    };

    SegmentMap {
        labels,
        width,
        height,
        num_labels: 2,
    }
}

// ---------------------------------------------------------------------------
// Watershed segmentation
// ---------------------------------------------------------------------------

/// Priority queue entry for watershed.
#[derive(Debug, Clone)]
struct WatershedEntry {
    priority: u8, // gradient magnitude (lower = higher priority)
    idx: usize,
    label: u32,
}

impl PartialEq for WatershedEntry {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.priority == other.priority
    }
}
impl Eq for WatershedEntry {}

impl PartialOrd for WatershedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WatershedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by priority
        other.priority.cmp(&self.priority)
    }
}

/// Compute the gradient magnitude image using Sobel-like differences.
#[allow(clippy::cast_precision_loss)]
fn gradient_magnitude(image: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let mut grad = vec![0u8; n];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let left = if x > 0 {
                image[idx - 1] as i32
            } else {
                image[idx] as i32
            };
            let right = if x + 1 < w {
                image[idx + 1] as i32
            } else {
                image[idx] as i32
            };
            let up = if y > 0 {
                image[idx - w] as i32
            } else {
                image[idx] as i32
            };
            let down = if y + 1 < h {
                image[idx + w] as i32
            } else {
                image[idx] as i32
            };

            let gx = (right - left).unsigned_abs();
            let gy = (down - up).unsigned_abs();
            let mag = ((gx + gy) as u32).min(255);
            grad[idx] = mag as u8;
        }
    }
    grad
}

/// Marker-based watershed segmentation.
///
/// `markers` is a label image where 0 means unlabelled and non-zero values are
/// initial seed labels. The algorithm uses BFS from seeds, prioritized by the
/// gradient magnitude, to grow regions.
///
/// Returns a `SegmentMap` where every pixel has a label.
#[must_use]
pub fn watershed_segment(image: &[u8], width: u32, height: u32, markers: &[u32]) -> SegmentMap {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    if n == 0 || image.len() < n || markers.len() < n {
        return SegmentMap::new(width, height);
    }

    let grad = gradient_magnitude(image, width, height);
    let mut labels = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut heap: BinaryHeap<WatershedEntry> = BinaryHeap::new();

    // Find max label for num_labels
    let mut max_label = 0u32;

    // Initialize with marker seeds
    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for i in 0..n {
        if markers[i] > 0 {
            labels[i] = markers[i];
            visited[i] = true;
            if markers[i] > max_label {
                max_label = markers[i];
            }
            // Enqueue neighbors
            let x = i % w;
            let y = i / w;
            for (dx, dy) in &dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = ny as usize * w + nx as usize;
                    if !visited[ni] {
                        heap.push(WatershedEntry {
                            priority: grad[ni],
                            idx: ni,
                            label: markers[i],
                        });
                    }
                }
            }
        }
    }

    // BFS by gradient priority
    while let Some(entry) = heap.pop() {
        let idx = entry.idx;
        if visited[idx] {
            continue;
        }
        visited[idx] = true;
        labels[idx] = entry.label;

        let x = idx % w;
        let y = idx / w;
        for (dx, dy) in &dirs {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                let ni = ny as usize * w + nx as usize;
                if !visited[ni] {
                    heap.push(WatershedEntry {
                        priority: grad[ni],
                        idx: ni,
                        label: entry.label,
                    });
                }
            }
        }
    }

    SegmentMap {
        labels,
        width,
        height,
        num_labels: max_label + 1,
    }
}

// ---------------------------------------------------------------------------
// SLIC superpixels
// ---------------------------------------------------------------------------

/// Cluster center for SLIC.
#[derive(Debug, Clone)]
struct SlicCenter {
    x: f64,
    y: f64,
    val: f64, // grayscale intensity
}

/// SLIC (Simple Linear Iterative Clustering) superpixel segmentation.
///
/// Produces approximately `n_clusters` superpixels by iterative k-means with
/// combined spatial and color (intensity) distance.
///
/// `compactness` controls the relative weight of spatial vs. color distance
/// (higher = more compact, spatially regular superpixels).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn superpixel_slic(
    image: &[u8],
    width: u32,
    height: u32,
    n_clusters: u32,
    compactness: f64,
) -> SegmentMap {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    if n == 0 || image.len() < n || n_clusters == 0 {
        return SegmentMap::new(width, height);
    }

    let nc = n_clusters.max(1) as usize;
    let step = ((n as f64) / (nc as f64)).sqrt().max(1.0);
    let step_i = step as usize;
    let m = compactness.max(1.0);

    // Initialize cluster centers on a grid
    let mut centers: Vec<SlicCenter> = Vec::new();
    let half_step = step_i / 2;
    let mut cy = half_step;
    while cy < h {
        let mut cx = half_step;
        while cx < w {
            let idx = cy * w + cx;
            centers.push(SlicCenter {
                x: cx as f64,
                y: cy as f64,
                val: image[idx] as f64,
            });
            cx += step_i;
        }
        cy += step_i;
    }

    if centers.is_empty() {
        // Fallback: single center
        centers.push(SlicCenter {
            x: (w / 2) as f64,
            y: (h / 2) as f64,
            val: image[(h / 2) * w + (w / 2)] as f64,
        });
    }

    let num_centers = centers.len();
    let mut labels = vec![0u32; n];
    let mut distances = vec![f64::MAX; n];

    let max_iterations = 10u32;
    let search_range = (step * 2.0) as i64;

    for _iter in 0..max_iterations {
        // Assignment step
        distances.fill(f64::MAX);

        for (ci, center) in centers.iter().enumerate() {
            let cx = center.x as i64;
            let cy_val = center.y as i64;

            let x_start = (cx - search_range).max(0) as usize;
            let x_end = ((cx + search_range) as usize).min(w);
            let y_start = (cy_val - search_range).max(0) as usize;
            let y_end = ((cy_val + search_range) as usize).min(h);

            for py in y_start..y_end {
                for px in x_start..x_end {
                    let idx = py * w + px;
                    let dc = (image[idx] as f64 - center.val).abs();
                    let ds_x = px as f64 - center.x;
                    let ds_y = py as f64 - center.y;
                    let ds = (ds_x * ds_x + ds_y * ds_y).sqrt();
                    let d = dc + (m / step) * ds;

                    if d < distances[idx] {
                        distances[idx] = d;
                        labels[idx] = ci as u32;
                    }
                }
            }
        }

        // Update step: recompute centers
        let mut sum_x = vec![0.0_f64; num_centers];
        let mut sum_y = vec![0.0_f64; num_centers];
        let mut sum_val = vec![0.0_f64; num_centers];
        let mut counts = vec![0u64; num_centers];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let ci = labels[idx] as usize;
                if ci < num_centers {
                    sum_x[ci] += x as f64;
                    sum_y[ci] += y as f64;
                    sum_val[ci] += image[idx] as f64;
                    counts[ci] += 1;
                }
            }
        }

        for ci in 0..num_centers {
            if counts[ci] > 0 {
                let c = counts[ci] as f64;
                centers[ci].x = sum_x[ci] / c;
                centers[ci].y = sum_y[ci] / c;
                centers[ci].val = sum_val[ci] / c;
            }
        }
    }

    // Enforce connectivity: relabel small orphan regions
    enforce_connectivity(&mut labels, w, h, num_centers);

    let actual_labels = {
        let mut seen = std::collections::BTreeSet::new();
        for &l in &labels {
            seen.insert(l);
        }
        seen.len() as u32
    };

    SegmentMap {
        labels,
        width,
        height,
        num_labels: actual_labels,
    }
}

/// Enforce spatial connectivity of SLIC labels by merging small orphan regions.
fn enforce_connectivity(labels: &mut [u32], w: usize, h: usize, _num_centers: usize) {
    let n = w * h;
    if n == 0 {
        return;
    }

    let min_size = (n / (_num_centers.max(1) * 4)).max(1);
    let mut visited = vec![false; n];
    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let label = labels[start];
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            let x = idx % w;
            let y = idx / w;
            for (dx, dy) in &dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = ny as usize * w + nx as usize;
                    if !visited[ni] && labels[ni] == label {
                        visited[ni] = true;
                        queue.push_back(ni);
                    }
                }
            }
        }

        // If component is too small, merge into neighbor label
        if component.len() < min_size {
            // Find adjacent label
            let mut neighbor_label = label;
            'find: for &idx in &component {
                let x = idx % w;
                let y = idx / w;
                for (dx, dy) in &dirs {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let ni = ny as usize * w + nx as usize;
                        if labels[ni] != label {
                            neighbor_label = labels[ni];
                            break 'find;
                        }
                    }
                }
            }
            if neighbor_label != label {
                for &idx in &component {
                    labels[idx] = neighbor_label;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Segment boundary mask
// ---------------------------------------------------------------------------

/// Create a boundary mask from a segment map.
///
/// A pixel is on a boundary if any of its 4-connected neighbors has a different label.
#[must_use]
pub fn segment_boundary_mask(map: &SegmentMap) -> Vec<bool> {
    let w = map.width as usize;
    let h = map.height as usize;
    let n = w * h;
    let mut boundary = vec![false; n];

    let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = map.labels[idx];
            for (dx, dy) in &dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = ny as usize * w + nx as usize;
                    if map.labels[ni] != label {
                        boundary[idx] = true;
                        break;
                    }
                }
            }
        }
    }
    boundary
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- SegmentMap ---

    #[test]
    fn test_segment_map_new() {
        let m = SegmentMap::new(4, 4);
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 4);
        assert_eq!(m.num_labels, 1);
        assert_eq!(m.labels.len(), 16);
    }

    #[test]
    fn test_segment_map_get_label() {
        let mut m = SegmentMap::new(3, 3);
        m.labels[4] = 5;
        assert_eq!(m.get_label(1, 1), Some(5));
        assert_eq!(m.get_label(0, 0), Some(0));
        assert_eq!(m.get_label(10, 10), None);
    }

    #[test]
    fn test_segment_map_segment_size() {
        let mut m = SegmentMap::new(4, 1);
        m.labels = vec![0, 1, 1, 0];
        assert_eq!(m.segment_size(0), 2);
        assert_eq!(m.segment_size(1), 2);
        assert_eq!(m.segment_size(99), 0);
    }

    #[test]
    fn test_segment_map_unique_labels() {
        let mut m = SegmentMap::new(4, 1);
        m.labels = vec![3, 1, 3, 2];
        let ul = m.unique_labels();
        assert_eq!(ul, vec![1, 2, 3]);
    }

    // --- Otsu threshold ---

    #[test]
    fn test_otsu_bimodal() {
        // Clear bimodal distribution: half at 50, half at 200
        let mut img = vec![50u8; 100];
        img.extend(vec![200u8; 100]);
        let t = otsu_threshold(&img);
        assert!(
            t > 40 && t < 210,
            "Otsu threshold {t} should be between modes"
        );
    }

    #[test]
    fn test_otsu_uniform() {
        let img = vec![128u8; 100];
        let _t = otsu_threshold(&img);
        // For uniform image, threshold is implementation-defined but should not panic
    }

    #[test]
    fn test_otsu_empty() {
        assert_eq!(otsu_threshold(&[]), 128);
    }

    #[test]
    fn test_otsu_single_value() {
        let img = vec![42u8; 50];
        let t = otsu_threshold(&img);
        // Should be 0 or close (all weight in one bin)
        assert!(t <= 42, "Threshold {t} should be <= single value 42");
    }

    // --- Threshold segment ---

    #[test]
    fn test_threshold_segment_basic() {
        let img = vec![10u8, 200, 50, 255];
        let m = threshold_segment(&img, 2, 2, 100);
        assert_eq!(m.labels, vec![0, 1, 0, 1]);
        assert_eq!(m.num_labels, 2);
    }

    #[test]
    fn test_threshold_segment_all_below() {
        let img = vec![10u8; 9];
        let m = threshold_segment(&img, 3, 3, 100);
        assert!(m.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_threshold_segment_all_above() {
        let img = vec![200u8; 4];
        let m = threshold_segment(&img, 2, 2, 100);
        assert!(m.labels.iter().all(|&l| l == 1));
    }

    // --- Watershed ---

    #[test]
    fn test_watershed_two_seeds() {
        // 5x1 image with two seeds at ends
        let img = vec![10u8, 20, 30, 20, 10];
        let markers = vec![1u32, 0, 0, 0, 2];
        let m = watershed_segment(&img, 5, 1, &markers);
        // Seeds should remain labelled
        assert_eq!(m.get_label(0, 0), Some(1));
        assert_eq!(m.get_label(4, 0), Some(2));
        // All pixels should be assigned
        assert!(m.labels.iter().all(|&l| l > 0));
    }

    #[test]
    fn test_watershed_single_seed() {
        let img = vec![128u8; 9];
        let mut markers = vec![0u32; 9];
        markers[4] = 1;
        let m = watershed_segment(&img, 3, 3, &markers);
        // All pixels should get label 1
        assert!(m.labels.iter().all(|&l| l == 1));
    }

    #[test]
    fn test_watershed_empty() {
        let m = watershed_segment(&[], 0, 0, &[]);
        assert_eq!(m.width, 0);
        assert_eq!(m.height, 0);
    }

    #[test]
    fn test_watershed_no_seeds() {
        let img = vec![50u8; 4];
        let markers = vec![0u32; 4];
        let m = watershed_segment(&img, 2, 2, &markers);
        // Without seeds, everything stays at 0
        assert!(m.labels.iter().all(|&l| l == 0));
    }

    // --- SLIC ---

    #[test]
    fn test_slic_produces_labels() {
        let img = vec![128u8; 16 * 16];
        let m = superpixel_slic(&img, 16, 16, 4, 10.0);
        // Should have at least 1 label
        assert!(m.num_labels >= 1);
        assert_eq!(m.labels.len(), 256);
    }

    #[test]
    fn test_slic_gradient_image() {
        // Gradient image: different intensities should produce distinct clusters
        let w = 20u32;
        let h = 20u32;
        let img: Vec<u8> = (0..(w * h))
            .map(|i| ((i % w) as u8).saturating_mul(12))
            .collect();
        let m = superpixel_slic(&img, w, h, 4, 10.0);
        assert!(
            m.num_labels >= 2,
            "Expected >= 2 labels, got {}",
            m.num_labels
        );
    }

    #[test]
    fn test_slic_empty() {
        let m = superpixel_slic(&[], 0, 0, 4, 10.0);
        assert_eq!(m.width, 0);
    }

    #[test]
    fn test_slic_single_cluster() {
        let img = vec![100u8; 4 * 4];
        let m = superpixel_slic(&img, 4, 4, 1, 10.0);
        assert!(m.num_labels >= 1);
    }

    // --- Boundary mask ---

    #[test]
    fn test_boundary_mask_uniform() {
        let m = SegmentMap {
            labels: vec![1; 9],
            width: 3,
            height: 3,
            num_labels: 1,
        };
        let b = segment_boundary_mask(&m);
        assert!(b.iter().all(|&v| !v), "No boundaries in uniform map");
    }

    #[test]
    fn test_boundary_mask_two_regions() {
        let m = SegmentMap {
            labels: vec![1, 1, 2, 2, 1, 1, 2, 2],
            width: 4,
            height: 2,
            num_labels: 2,
        };
        let b = segment_boundary_mask(&m);
        // Boundary should be at column 1 and 2 (where labels change)
        assert!(b[1]); // (1,0) is adjacent to (2,0) which is label 2
        assert!(b[2]); // (2,0) is adjacent to (1,0) which is label 1
        assert!(!b[0]); // (0,0) only neighbors same label
        assert!(!b[3]); // (3,0) only neighbors same label
    }

    #[test]
    fn test_boundary_mask_checkerboard() {
        let m = SegmentMap {
            labels: vec![0, 1, 1, 0],
            width: 2,
            height: 2,
            num_labels: 2,
        };
        let b = segment_boundary_mask(&m);
        // All pixels are on boundaries
        assert!(b.iter().all(|&v| v));
    }

    // --- Gradient magnitude ---

    #[test]
    fn test_gradient_uniform() {
        let img = vec![100u8; 9];
        let g = gradient_magnitude(&img, 3, 3);
        assert!(g.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_gradient_edge() {
        // Left half 0, right half 255
        let img = vec![0u8, 255, 0, 255];
        let g = gradient_magnitude(&img, 2, 2);
        // All pixels should have non-zero gradient
        assert!(g.iter().any(|&v| v > 0));
    }

    // --- Triangle threshold ---

    #[test]
    fn test_triangle_bimodal() {
        // Bimodal distribution: cluster at 30 (dominant) and cluster at 200
        let mut img = vec![30u8; 800];
        img.extend(vec![200u8; 200]);
        let t = triangle_threshold(&img);
        // Threshold should be between the two peaks
        assert!(
            t > 20 && t < 210,
            "Triangle threshold {t} should be between modes"
        );
    }

    #[test]
    fn test_triangle_unimodal_with_tail() {
        // Unimodal peak at 20 with a long tail towards 255
        let mut img = Vec::new();
        for _ in 0..500 {
            img.push(20);
        }
        for i in 21..=255u8 {
            let count = (255 - i) as usize / 4;
            for _ in 0..count.max(1) {
                img.push(i);
            }
        }
        let t = triangle_threshold(&img);
        // Should find a meaningful threshold in the tail region
        assert!(t > 10, "Triangle threshold {t} should be above noise floor");
        assert!(t < 200, "Triangle threshold {t} should be below far tail");
    }

    #[test]
    fn test_triangle_empty() {
        assert_eq!(triangle_threshold(&[]), 128);
    }

    #[test]
    fn test_triangle_single_value() {
        let img = vec![100u8; 50];
        let _t = triangle_threshold(&img);
        // Should not panic with uniform input
    }

    #[test]
    fn test_triangle_two_values() {
        let mut img = vec![0u8; 100];
        img.extend(vec![255u8; 100]);
        let t = triangle_threshold(&img);
        assert!(t < 255, "Threshold should be valid: {t}");
    }

    // --- Adaptive threshold segment ---

    #[test]
    fn test_adaptive_otsu() {
        // Use spread values so Otsu picks a threshold that splits both groups
        let mut img = Vec::new();
        for i in 40..=60 {
            for _ in 0..5 {
                img.push(i);
            }
        }
        for i in 190..=210 {
            for _ in 0..5 {
                img.push(i);
            }
        }
        let w = 21;
        let h = img.len() as u32 / w;
        let total = (w * h) as usize;
        img.truncate(total);

        let t = auto_threshold(&img, ThresholdMethod::Otsu);
        // Threshold should be somewhere between 39 and 211 (separating the clusters)
        assert!(
            t >= 40 && t <= 210,
            "Otsu threshold should separate clusters: {t}"
        );

        let m = adaptive_threshold_segment(&img, w, h, ThresholdMethod::Otsu);
        assert_eq!(m.num_labels, 2);
        // Verify segmentation produces meaningful result (has foreground pixels)
        let fg_count = m.labels.iter().filter(|&&l| l == 1).count();
        let bg_count = m.labels.iter().filter(|&&l| l == 0).count();
        assert!(fg_count > 0, "Should have foreground pixels (t={t})");
        assert!(bg_count > 0, "Should have background pixels (t={t})");
    }

    #[test]
    fn test_adaptive_triangle() {
        let mut img = vec![30u8; 150];
        img.extend(vec![200u8; 50]);
        let m = adaptive_threshold_segment(&img, 20, 10, ThresholdMethod::Triangle);
        assert_eq!(m.num_labels, 2);
    }

    #[test]
    fn test_auto_threshold_methods() {
        let mut img = vec![50u8; 100];
        img.extend(vec![200u8; 100]);
        let otsu_t = auto_threshold(&img, ThresholdMethod::Otsu);
        let tri_t = auto_threshold(&img, ThresholdMethod::Triangle);
        // Both should find a threshold between the two modes
        assert!(otsu_t > 40 && otsu_t < 210, "Otsu: {otsu_t}");
        assert!(tri_t > 0 && tri_t < 255, "Triangle: {tri_t}");
    }

    #[test]
    fn test_threshold_method_eq() {
        assert_eq!(ThresholdMethod::Otsu, ThresholdMethod::Otsu);
        assert_ne!(ThresholdMethod::Otsu, ThresholdMethod::Triangle);
    }
}

// ===========================================================================
// GrabCut-inspired segmentation
// ===========================================================================

/// Pixel label used by the GrabCut algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrabLabel {
    /// Definitely background (or seed background).
    Background,
    /// Definitely foreground (or seed foreground).
    Foreground,
    /// Probably background (uncertain).
    ProbBackground,
    /// Probably foreground (uncertain).
    ProbForeground,
}

/// Configuration for [`grabcut_segment`].
#[derive(Debug, Clone)]
pub struct GrabCutConfig {
    /// Number of EM-GMM + graph-cut iterations.
    pub iterations: u32,
    /// Seed rectangle (x, y, width, height) containing the foreground object.
    /// Pixels inside are initialised as probable foreground; outside as background.
    pub rect: (u32, u32, u32, u32),
    /// Number of Gaussian components per region (foreground / background).
    pub num_components: usize,
}

impl Default for GrabCutConfig {
    fn default() -> Self {
        Self {
            iterations: 5,
            rect: (0, 0, 0, 0),
            num_components: 2,
        }
    }
}

/// Result of a GrabCut segmentation.
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Per-pixel binary mask (255 = foreground, 0 = background), row-major.
    pub mask: Vec<u8>,
    /// Number of pixels classified as foreground.
    pub foreground_pixels: usize,
}

/// A simple 2-component GMM (one per class: fg/bg).
#[derive(Debug, Clone)]
struct SimpleGmm {
    means: Vec<f64>,     // num_components × 1 (grayscale)
    variances: Vec<f64>, // num_components × 1
    weights: Vec<f64>,   // num_components (sum to 1)
}

impl SimpleGmm {
    fn new(num_components: usize) -> Self {
        let k = num_components.max(1);
        Self {
            means: vec![128.0; k],
            variances: vec![1000.0; k],
            weights: vec![1.0 / k as f64; k],
        }
    }

    /// Fit via k-means-style EM on the supplied pixel values.
    fn fit(&mut self, pixels: &[f64]) {
        if pixels.is_empty() {
            return;
        }
        let k = self.means.len();

        // --- Init: spread means across pixel range ---
        let min_v = pixels.iter().cloned().fold(f64::MAX, f64::min);
        let max_v = pixels.iter().cloned().fold(f64::MIN, f64::max);
        let span = (max_v - min_v).max(1.0);
        for (i, m) in self.means.iter_mut().enumerate() {
            *m = min_v + (i as f64 + 0.5) / k as f64 * span;
        }

        // --- EM iterations ---
        let em_iters = 10;
        for _ in 0..em_iters {
            // E-step: assign each pixel to nearest component
            let mut sums = vec![0.0_f64; k];
            let mut counts = vec![0u64; k];
            for &v in pixels {
                let comp = nearest_component(&self.means, v);
                sums[comp] += v;
                counts[comp] += 1;
            }
            // M-step: update means, variances, weights
            let n = pixels.len() as f64;
            for c in 0..k {
                if counts[c] > 0 {
                    self.means[c] = sums[c] / counts[c] as f64;
                    let var: f64 = pixels
                        .iter()
                        .filter(|&&v| nearest_component(&self.means, v) == c)
                        .map(|&v| (v - self.means[c]).powi(2))
                        .sum::<f64>()
                        / counts[c] as f64;
                    self.variances[c] = var.max(1.0);
                    self.weights[c] = counts[c] as f64 / n;
                }
            }
        }
    }

    /// Log-likelihood that `v` was generated by this GMM.
    fn log_likelihood(&self, v: f64) -> f64 {
        let mut sum = 0.0_f64;
        for i in 0..self.means.len() {
            let sigma2 = self.variances[i].max(1e-6);
            let diff = v - self.means[i];
            let gauss = (-0.5 * diff * diff / sigma2).exp()
                / (2.0 * std::f64::consts::PI * sigma2).sqrt().max(1e-12);
            sum += self.weights[i] * gauss;
        }
        sum.max(1e-300).ln()
    }
}

fn nearest_component(means: &[f64], v: f64) -> usize {
    means
        .iter()
        .enumerate()
        .min_by(|a, b| {
            let da = (v - a.1).abs();
            let db = (v - b.1).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// GrabCut-inspired segmentation on a grayscale image.
///
/// Initialises fg/bg regions from the seed rectangle, fits two-component GMMs,
/// and iteratively refines the segmentation using energy minimisation.
///
/// `image` must be grayscale u8, row-major, of size `width × height`.
/// Returns a [`SegmentationResult`] or an empty result if inputs are invalid.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn grabcut_segment(
    image: &[u8],
    width: u32,
    height: u32,
    config: &GrabCutConfig,
) -> SegmentationResult {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    if n == 0 || image.len() < n {
        return SegmentationResult {
            mask: Vec::new(),
            foreground_pixels: 0,
        };
    }

    let (rx, ry, rw, rh) = config.rect;
    let rx2 = (rx + rw).min(width);
    let ry2 = (ry + rh).min(height);

    // --- Initialise labels ---
    let mut labels: Vec<GrabLabel> = (0..n)
        .map(|i| {
            let x = (i % w) as u32;
            let y = (i / w) as u32;
            if x >= rx && x < rx2 && y >= ry && y < ry2 {
                GrabLabel::ProbForeground
            } else {
                GrabLabel::Background
            }
        })
        .collect();

    let k = config.num_components.max(1);
    let mut fg_gmm = SimpleGmm::new(k);
    let mut bg_gmm = SimpleGmm::new(k);

    for _iter in 0..config.iterations {
        // --- Collect fg/bg pixel values ---
        let mut fg_pixels: Vec<f64> = Vec::new();
        let mut bg_pixels: Vec<f64> = Vec::new();
        for (i, &lbl) in labels.iter().enumerate() {
            let v = image[i] as f64;
            match lbl {
                GrabLabel::Foreground | GrabLabel::ProbForeground => fg_pixels.push(v),
                GrabLabel::Background | GrabLabel::ProbBackground => bg_pixels.push(v),
            }
        }

        // Avoid degenerate fits
        if fg_pixels.is_empty() || bg_pixels.is_empty() {
            break;
        }

        // --- Fit GMMs ---
        fg_gmm.fit(&fg_pixels);
        bg_gmm.fit(&bg_pixels);

        // --- Re-label uncertain pixels by energy minimisation ---
        // Energy = -log P(pixel | model) + smoothness term
        // Compute new labels into a separate buffer to avoid borrow conflicts.
        let mut new_labels = labels.clone();
        for i in 0..n {
            // Hard labels (Background / Foreground) are kept fixed.
            if labels[i] == GrabLabel::Background || labels[i] == GrabLabel::Foreground {
                continue;
            }
            let v = image[i] as f64;
            let ll_fg = fg_gmm.log_likelihood(v);
            let ll_bg = bg_gmm.log_likelihood(v);

            // Add a smoothness prior: prefer the majority label of 4-connected neighbours.
            let mut fg_neighbours = 0i32;
            let mut bg_neighbours = 0i32;
            let x = i % w;
            let y = i / w;
            let dirs: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dx, dy) in dirs {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let ni = ny as usize * w + nx as usize;
                    match labels[ni] {
                        GrabLabel::Foreground | GrabLabel::ProbForeground => fg_neighbours += 1,
                        GrabLabel::Background | GrabLabel::ProbBackground => bg_neighbours += 1,
                    }
                }
            }
            let smooth_weight = 0.5_f64;
            let energy_fg = -ll_fg - smooth_weight * fg_neighbours as f64;
            let energy_bg = -ll_bg - smooth_weight * bg_neighbours as f64;

            new_labels[i] = if energy_fg <= energy_bg {
                GrabLabel::ProbForeground
            } else {
                GrabLabel::ProbBackground
            };
        }
        labels = new_labels;
    }

    // --- Build output mask ---
    let mask: Vec<u8> = labels
        .iter()
        .map(|&l| match l {
            GrabLabel::Foreground | GrabLabel::ProbForeground => 255,
            _ => 0,
        })
        .collect();
    let foreground_pixels = mask.iter().filter(|&&v| v == 255).count();

    SegmentationResult {
        mask,
        foreground_pixels,
    }
}

// ---------------------------------------------------------------------------
// GrabCut tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod grabcut_tests {
    use super::*;

    fn make_image(w: u32, h: u32, fg_val: u8, bg_val: u8, rect: (u32, u32, u32, u32)) -> Vec<u8> {
        let (rx, ry, rw, rh) = rect;
        let rx2 = rx + rw;
        let ry2 = ry + rh;
        (0..(w * h) as usize)
            .map(|i| {
                let x = (i as u32) % w;
                let y = (i as u32) / w;
                if x >= rx && x < rx2 && y >= ry && y < ry2 {
                    fg_val
                } else {
                    bg_val
                }
            })
            .collect()
    }

    #[test]
    fn test_grabcut_returns_correct_length() {
        let w = 8u32;
        let h = 8u32;
        let img = vec![128u8; (w * h) as usize];
        let cfg = GrabCutConfig {
            iterations: 2,
            rect: (2, 2, 4, 4),
            num_components: 2,
        };
        let result = grabcut_segment(&img, w, h, &cfg);
        assert_eq!(result.mask.len(), (w * h) as usize);
    }

    #[test]
    fn test_grabcut_foreground_pixels_consistent() {
        let w = 6u32;
        let h = 6u32;
        let img = vec![100u8; (w * h) as usize];
        let cfg = GrabCutConfig {
            iterations: 2,
            rect: (1, 1, 4, 4),
            num_components: 2,
        };
        let result = grabcut_segment(&img, w, h, &cfg);
        let counted = result.mask.iter().filter(|&&v| v == 255).count();
        assert_eq!(counted, result.foreground_pixels);
    }

    #[test]
    fn test_grabcut_binary_mask_values() {
        let img = vec![80u8; 25];
        let cfg = GrabCutConfig {
            iterations: 1,
            rect: (1, 1, 3, 3),
            num_components: 2,
        };
        let result = grabcut_segment(&img, 5, 5, &cfg);
        for &v in &result.mask {
            assert!(v == 0 || v == 255, "mask must be binary, got {v}");
        }
    }

    #[test]
    fn test_grabcut_empty_image() {
        let cfg = GrabCutConfig::default();
        let result = grabcut_segment(&[], 0, 0, &cfg);
        assert!(result.mask.is_empty());
        assert_eq!(result.foreground_pixels, 0);
    }

    #[test]
    fn test_grabcut_rect_outside_image() {
        let img = vec![50u8; 9];
        let cfg = GrabCutConfig {
            iterations: 2,
            rect: (100, 100, 10, 10), // completely outside
            num_components: 2,
        };
        let result = grabcut_segment(&img, 3, 3, &cfg);
        // All pixels become background
        assert_eq!(result.foreground_pixels, 0);
    }

    #[test]
    fn test_grabcut_zero_iterations_gives_rect_as_fg() {
        let w = 6u32;
        let h = 6u32;
        let img = vec![128u8; (w * h) as usize];
        let cfg = GrabCutConfig {
            iterations: 0,
            rect: (1, 1, 4, 4),
            num_components: 2,
        };
        let result = grabcut_segment(&img, w, h, &cfg);
        // With 0 iterations, initial rect pixels should be foreground
        assert!(
            result.foreground_pixels > 0,
            "Rect pixels should be fg with 0 iters"
        );
    }

    #[test]
    fn test_grabcut_high_contrast_separates_regions() {
        let w = 10u32;
        let h = 10u32;
        let rect = (3, 3, 4, 4);
        // fg=220, bg=30 — very high contrast
        let img = make_image(w, h, 220, 30, rect);
        let cfg = GrabCutConfig {
            iterations: 5,
            rect,
            num_components: 2,
        };
        let result = grabcut_segment(&img, w, h, &cfg);
        assert!(
            result.foreground_pixels > 0,
            "High-contrast fg region should have fg pixels"
        );
    }

    #[test]
    fn test_grabcut_single_pixel() {
        let img = vec![200u8];
        let cfg = GrabCutConfig {
            iterations: 1,
            rect: (0, 0, 1, 1),
            num_components: 1,
        };
        let result = grabcut_segment(&img, 1, 1, &cfg);
        assert_eq!(result.mask.len(), 1);
    }

    #[test]
    fn test_grabcut_full_rect_all_fg() {
        let w = 4u32;
        let h = 4u32;
        let img = vec![100u8; (w * h) as usize];
        let cfg = GrabCutConfig {
            iterations: 1,
            rect: (0, 0, w, h), // covers entire image
            num_components: 2,
        };
        let result = grabcut_segment(&img, w, h, &cfg);
        // All pixels are initialised as probable foreground
        assert_eq!(result.foreground_pixels, (w * h) as usize);
    }

    #[test]
    fn test_grabcut_num_components_one() {
        let img = vec![128u8; 16];
        let cfg = GrabCutConfig {
            iterations: 2,
            rect: (1, 1, 2, 2),
            num_components: 1,
        };
        let result = grabcut_segment(&img, 4, 4, &cfg);
        assert_eq!(result.mask.len(), 16);
    }

    #[test]
    fn test_grabcut_result_struct_fields() {
        let img = vec![50u8; 9];
        let cfg = GrabCutConfig {
            iterations: 2,
            rect: (0, 0, 3, 2),
            num_components: 2,
        };
        let r = grabcut_segment(&img, 3, 3, &cfg);
        assert_eq!(r.mask.len(), 9);
        assert!(r.foreground_pixels <= 9);
    }

    #[test]
    fn test_grabcut_default_config() {
        let cfg = GrabCutConfig::default();
        assert_eq!(cfg.iterations, 5);
        assert_eq!(cfg.num_components, 2);
    }

    #[test]
    fn test_grabcut_label_enum() {
        assert_ne!(GrabLabel::Foreground, GrabLabel::Background);
        assert_eq!(GrabLabel::Foreground, GrabLabel::Foreground);
    }

    #[test]
    fn test_grabcut_short_image_returns_empty() {
        let cfg = GrabCutConfig {
            iterations: 1,
            rect: (0, 0, 4, 4),
            num_components: 2,
        };
        // Image too short for 4×4
        let result = grabcut_segment(&[1u8, 2, 3], 4, 4, &cfg);
        assert!(result.mask.is_empty());
    }

    #[test]
    fn test_grabcut_many_iterations_stable() {
        let w = 8u32;
        let h = 8u32;
        let img: Vec<u8> = (0..(w * h)).map(|i| ((i * 3) % 256) as u8).collect();
        let cfg = GrabCutConfig {
            iterations: 20,
            rect: (2, 2, 4, 4),
            num_components: 2,
        };
        let result = grabcut_segment(&img, w, h, &cfg);
        assert_eq!(result.mask.len(), (w * h) as usize);
    }
}
