//! Color quantization and palette extraction for image processing.
//!
//! Implements the median-cut algorithm, octree quantization, and k-means
//! quantization for reducing a set of colors to a representative palette, plus
//! utilities for nearest-palette-color lookup and dithering error computation.

/// An sRGB color with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgb8 {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl Rgb8 {
    /// Creates a new 8-bit RGB color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Returns the color as an `[r, g, b]` array.
    #[must_use]
    pub const fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// Squared Euclidean distance in RGB space.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_sq(self, other: Self) -> u32 {
        let dr = i32::from(self.r) - i32::from(other.r);
        let dg = i32::from(self.g) - i32::from(other.g);
        let db = i32::from(self.b) - i32::from(other.b);
        (dr * dr + dg * dg + db * db) as u32
    }
}

/// Which color channel to split on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Red,
    Green,
    Blue,
}

/// A bounding box (bucket) of colors used by median-cut.
#[derive(Debug, Clone)]
struct ColorBox {
    colors: Vec<Rgb8>,
}

impl ColorBox {
    /// Creates a new color box.
    fn new(colors: Vec<Rgb8>) -> Self {
        Self { colors }
    }

    /// Returns the channel with the widest range.
    fn widest_channel(&self) -> Channel {
        let (mut rmin, mut rmax) = (u8::MAX, u8::MIN);
        let (mut gmin, mut gmax) = (u8::MAX, u8::MIN);
        let (mut bmin, mut bmax) = (u8::MAX, u8::MIN);

        for c in &self.colors {
            rmin = rmin.min(c.r);
            rmax = rmax.max(c.r);
            gmin = gmin.min(c.g);
            gmax = gmax.max(c.g);
            bmin = bmin.min(c.b);
            bmax = bmax.max(c.b);
        }

        let r_range = rmax - rmin;
        let g_range = gmax - gmin;
        let b_range = bmax - bmin;

        if r_range >= g_range && r_range >= b_range {
            Channel::Red
        } else if g_range >= b_range {
            Channel::Green
        } else {
            Channel::Blue
        }
    }

    /// Returns the average color of this box.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn average_color(&self) -> Rgb8 {
        if self.colors.is_empty() {
            return Rgb8::new(0, 0, 0);
        }
        let n = self.colors.len() as f64;
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        for c in &self.colors {
            r_sum += u64::from(c.r);
            g_sum += u64::from(c.g);
            b_sum += u64::from(c.b);
        }
        Rgb8::new(
            (r_sum as f64 / n).round() as u8,
            (g_sum as f64 / n).round() as u8,
            (b_sum as f64 / n).round() as u8,
        )
    }

    /// Splits this box into two along the widest channel at the median.
    fn split(mut self) -> (Self, Self) {
        let ch = self.widest_channel();
        match ch {
            Channel::Red => self.colors.sort_by_key(|c| c.r),
            Channel::Green => self.colors.sort_by_key(|c| c.g),
            Channel::Blue => self.colors.sort_by_key(|c| c.b),
        }
        let mid = self.colors.len() / 2;
        let right = self.colors.split_off(mid);
        (Self::new(self.colors), Self::new(right))
    }
}

/// Extracts a palette from a list of colors using the median-cut algorithm.
///
/// # Arguments
///
/// * `colors` - Input colors (can contain duplicates)
/// * `palette_size` - Desired number of palette entries (will be rounded to nearest power of 2 in iterations)
///
/// # Returns
///
/// A vector of representative palette colors.
#[must_use]
pub fn median_cut(colors: &[Rgb8], palette_size: usize) -> Vec<Rgb8> {
    if colors.is_empty() || palette_size == 0 {
        return Vec::new();
    }
    if palette_size >= colors.len() {
        let mut unique: Vec<Rgb8> = colors.to_vec();
        unique.dedup();
        return unique;
    }

    let mut boxes = vec![ColorBox::new(colors.to_vec())];

    while boxes.len() < palette_size {
        // Find box with most colors
        let idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colors.len() > 1)
            .max_by_key(|(_, b)| b.colors.len())
            .map(|(i, _)| i);

        let Some(idx) = idx else { break };

        let biggest = boxes.remove(idx);
        let (a, b) = biggest.split();
        if !a.colors.is_empty() {
            boxes.push(a);
        }
        if !b.colors.is_empty() {
            boxes.push(b);
        }
    }

    boxes.iter().map(ColorBox::average_color).collect()
}

/// A quantized palette with fast nearest-color lookup.
#[derive(Debug, Clone)]
pub struct Palette {
    /// The palette colors.
    entries: Vec<Rgb8>,
}

impl Palette {
    /// Creates a palette from a set of representative colors.
    #[must_use]
    pub fn new(entries: Vec<Rgb8>) -> Self {
        Self { entries }
    }

    /// Creates a palette by extracting representative colors from input data.
    #[must_use]
    pub fn from_colors(colors: &[Rgb8], size: usize) -> Self {
        Self {
            entries: median_cut(colors, size),
        }
    }

    /// Returns the palette entries.
    #[must_use]
    pub fn entries(&self) -> &[Rgb8] {
        &self.entries
    }

    /// Returns the number of entries in the palette.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the palette has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Finds the nearest palette color to the given color (brute-force).
    ///
    /// Returns the index and the color.
    #[must_use]
    pub fn nearest(&self, color: Rgb8) -> Option<(usize, Rgb8)> {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(_, &c)| color.distance_sq(c))
            .map(|(i, &c)| (i, c))
    }

    /// Quantizes a list of colors to palette indices.
    #[must_use]
    pub fn quantize(&self, colors: &[Rgb8]) -> Vec<usize> {
        colors
            .iter()
            .map(|&c| self.nearest(c).map_or(0, |(i, _)| i))
            .collect()
    }
}

/// Computes the quantization error (mean squared error) over a set of colors.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn quantization_mse(palette: &Palette, colors: &[Rgb8]) -> f64 {
    if colors.is_empty() || palette.is_empty() {
        return 0.0;
    }
    let total: u64 = colors
        .iter()
        .map(|&c| {
            let (_, nearest) = palette.nearest(c).unwrap_or((0, Rgb8::new(0, 0, 0)));
            u64::from(c.distance_sq(nearest))
        })
        .sum();
    total as f64 / colors.len() as f64
}

/// Computes the Floyd-Steinberg dithering error for a single pixel.
///
/// Returns `(error_r, error_g, error_b)` as signed values.
#[must_use]
pub fn dithering_error(original: Rgb8, quantized: Rgb8) -> (i16, i16, i16) {
    (
        i16::from(original.r) - i16::from(quantized.r),
        i16::from(original.g) - i16::from(quantized.g),
        i16::from(original.b) - i16::from(quantized.b),
    )
}

/// Applies a dithering error to a pixel with clamping.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_error(pixel: Rgb8, error: (i16, i16, i16), factor: f64) -> Rgb8 {
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    Rgb8::new(
        clamp(f64::from(pixel.r) + f64::from(error.0) * factor),
        clamp(f64::from(pixel.g) + f64::from(error.1) * factor),
        clamp(f64::from(pixel.b) + f64::from(error.2) * factor),
    )
}

/// Computes the color histogram from a list of colors.
///
/// Returns a vector of `(color, count)` pairs sorted by count descending.
#[must_use]
pub fn color_histogram(colors: &[Rgb8]) -> Vec<(Rgb8, usize)> {
    use std::collections::HashMap;
    let mut map: HashMap<Rgb8, usize> = HashMap::new();
    for &c in colors {
        *map.entry(c).or_insert(0) += 1;
    }
    let mut hist: Vec<_> = map.into_iter().collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1));
    hist
}

// ── Octree quantization (Gervautz & Purgathofer 1988) ─────────────────────────

/// Internal octree node (arena-allocated).
struct OctreeNode {
    /// Accumulated red sum.
    sum_r: u64,
    /// Accumulated green sum.
    sum_g: u64,
    /// Accumulated blue sum.
    sum_b: u64,
    /// Number of pixels represented.
    count: u64,
    /// Eight children (indices into the arena, `None` = leaf with no child).
    children: [Option<usize>; 8],
    /// True when this node is a leaf (no further subdivision needed).
    is_leaf: bool,
    /// Current depth in the tree.
    depth: u8,
}

impl OctreeNode {
    fn new(depth: u8) -> Self {
        Self {
            sum_r: 0,
            sum_g: 0,
            sum_b: 0,
            count: 0,
            children: [None; 8],
            is_leaf: false,
            depth,
        }
    }
}

/// Octree-based color quantizer.
struct Octree {
    /// Arena storage for nodes (index 0 = root).
    nodes: Vec<OctreeNode>,
    /// Indices of leaf nodes, grouped by depth for efficient merging.
    leaves_by_depth: Vec<Vec<usize>>,
    /// Current leaf count.
    leaf_count: usize,
}

impl Octree {
    const MAX_DEPTH: u8 = 7;

    fn new() -> Self {
        let mut nodes = Vec::with_capacity(4096);
        nodes.push(OctreeNode::new(0)); // root = index 0
        Self {
            nodes,
            leaves_by_depth: vec![Vec::new(); (Self::MAX_DEPTH + 1) as usize],
            leaf_count: 0,
        }
    }

    /// Map an RGB color to an octree child index at a given bit level.
    #[inline]
    fn child_index(r: u8, g: u8, b: u8, bit: u8) -> usize {
        let r_bit = usize::from((r >> bit) & 1);
        let g_bit = usize::from((g >> bit) & 1);
        let b_bit = usize::from((b >> bit) & 1);
        (r_bit << 2) | (g_bit << 1) | b_bit
    }

    /// Insert a pixel into the octree.
    fn insert(&mut self, r: u8, g: u8, b: u8) {
        let mut node_idx = 0usize;
        for level in 0..=Self::MAX_DEPTH {
            let bit = Self::MAX_DEPTH - level;
            let child = Self::child_index(r, g, b, bit);

            if self.nodes[node_idx].children[child].is_none() {
                let new_idx = self.nodes.len();
                self.nodes.push(OctreeNode::new(level + 1));

                if level == Self::MAX_DEPTH {
                    // Make it a leaf immediately
                    self.nodes[new_idx].is_leaf = true;
                    self.nodes[new_idx].depth = level + 1;
                    let depth_idx = self.nodes[new_idx].depth as usize;
                    self.leaves_by_depth[depth_idx.min(Self::MAX_DEPTH as usize)].push(new_idx);
                    self.leaf_count += 1;
                }
                self.nodes[node_idx].children[child] = Some(new_idx);
            }

            node_idx = match self.nodes[node_idx].children[child] {
                Some(idx) => idx,
                None => break,
            };

            if self.nodes[node_idx].is_leaf {
                self.nodes[node_idx].sum_r += u64::from(r);
                self.nodes[node_idx].sum_g += u64::from(g);
                self.nodes[node_idx].sum_b += u64::from(b);
                self.nodes[node_idx].count += 1;
                return;
            }
        }
        // Accumulate at this node if we reach max depth
        self.nodes[node_idx].sum_r += u64::from(r);
        self.nodes[node_idx].sum_g += u64::from(g);
        self.nodes[node_idx].sum_b += u64::from(b);
        self.nodes[node_idx].count += 1;
    }

    /// Merge the leaf node at `idx` into its parent `parent_idx`.
    fn merge_leaf_into_parent(&mut self, parent_idx: usize, leaf_idx: usize) {
        self.nodes[parent_idx].sum_r += self.nodes[leaf_idx].sum_r;
        self.nodes[parent_idx].sum_g += self.nodes[leaf_idx].sum_g;
        self.nodes[parent_idx].sum_b += self.nodes[leaf_idx].sum_b;
        self.nodes[parent_idx].count += self.nodes[leaf_idx].count;
        // Disconnect leaf
        for slot in &mut self.nodes[parent_idx].children {
            if *slot == Some(leaf_idx) {
                *slot = None;
            }
        }
    }

    /// Reduce to at most `k` leaves by merging least-populated siblings.
    fn reduce_to(&mut self, k: usize) {
        while self.leaf_count > k {
            // Find the deepest level with leaves
            let deepest = (0..=Self::MAX_DEPTH as usize)
                .rev()
                .find(|&d| !self.leaves_by_depth[d].is_empty());
            let Some(d) = deepest else { break };

            let leaf_idx = match self.leaves_by_depth[d].pop() {
                Some(idx) => idx,
                None => break,
            };

            // Find parent by scanning nodes (brute-force; tree is small)
            let parent_idx = (0..self.nodes.len())
                .find(|&i| self.nodes[i].children.iter().any(|&c| c == Some(leaf_idx)));

            if let Some(parent) = parent_idx {
                self.merge_leaf_into_parent(parent, leaf_idx);
                // If parent now has no children, promote it to leaf
                if self.nodes[parent].children.iter().all(|c| c.is_none())
                    && self.nodes[parent].count > 0
                {
                    self.nodes[parent].is_leaf = true;
                    let parent_depth = self.nodes[parent].depth as usize;
                    self.leaves_by_depth[parent_depth.min(Self::MAX_DEPTH as usize)].push(parent);
                }
                self.leaf_count -= 1;
            } else {
                break;
            }
        }
    }

    /// Extract the palette from all leaf nodes.
    fn palette(&self) -> Vec<[u8; 3]> {
        let mut result = Vec::new();
        for node in &self.nodes {
            if (node.is_leaf || node.count > 0) && node.count > 0 {
                let r = (node.sum_r / node.count) as u8;
                let g = (node.sum_g / node.count) as u8;
                let b = (node.sum_b / node.count) as u8;
                let entry = [r, g, b];
                // Deduplicate approximate leaves
                if !result.contains(&entry) {
                    result.push(entry);
                }
            }
        }
        result
    }
}

/// Quantize a slice of RGB pixels to at most `k` palette colors using an octree.
///
/// Implements the Gervautz & Purgathofer (1988) octree color quantization algorithm:
/// pixels are inserted into an 8-level octree; when there are more than `k` leaves,
/// the deepest leaves are merged into their parents until the palette fits in `k` entries.
///
/// # Arguments
///
/// * `pixels` — input RGB pixels as `[r, g, b]` bytes.
/// * `k` — maximum number of palette colors (> 0).
///
/// # Returns
///
/// A vector of at most `k` representative `[r, g, b]` palette entries.
#[must_use]
pub fn quantize_octree(pixels: &[[u8; 3]], k: usize) -> Vec<[u8; 3]> {
    if pixels.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut tree = Octree::new();
    for &[r, g, b] in pixels {
        tree.insert(r, g, b);
    }
    tree.reduce_to(k);
    let mut palette = tree.palette();
    palette.truncate(k);
    palette
}

// ── k-means color quantization (Lloyd 1957 / k-means++) ──────────────────────

/// Quantize a slice of RGB pixels to `k` palette colors using k-means clustering
/// with k-means++ seeding.
///
/// # Algorithm
///
/// 1. **k-means++ seeding** — first center chosen deterministically from the first
///    pixel; subsequent centers chosen with probability proportional to squared
///    distance from the nearest existing center.
/// 2. **Lloyd iterations** — alternating assignment and centroid recomputation, up
///    to `max_iterations` rounds; stops early when max centroid movement < `tolerance`.
///
/// The initial seed index is computed deterministically from the first pixel value
/// so results are reproducible for identical inputs.
///
/// # Arguments
///
/// * `pixels` — input RGB pixels as `[r, g, b]` bytes.
/// * `k` — number of clusters (> 0; if `k >= pixels.len()`, returns unique colors).
/// * `max_iterations` — upper bound on Lloyd iterations.
/// * `tolerance` — early-stop threshold in RGB Euclidean units.
///
/// # Returns
///
/// `k` representative `[r, g, b]` palette entries.
#[must_use]
pub fn quantize_kmeans(
    pixels: &[[u8; 3]],
    k: usize,
    max_iterations: u32,
    tolerance: f32,
) -> Vec<[u8; 3]> {
    if pixels.is_empty() || k == 0 {
        return Vec::new();
    }
    if k >= pixels.len() {
        let mut unique: Vec<[u8; 3]> = pixels.to_vec();
        unique.dedup();
        unique.truncate(k);
        return unique;
    }

    // ── k-means++ seeding ─────────────────────────────────────────────────────
    let mut centers: Vec<[f32; 3]> = Vec::with_capacity(k);

    // Deterministic first center based on first pixel
    let seed_idx = ((pixels[0][0] as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1)) as usize
        % pixels.len();
    centers.push([
        pixels[seed_idx][0] as f32,
        pixels[seed_idx][1] as f32,
        pixels[seed_idx][2] as f32,
    ]);

    // Successive centers chosen proportional to distance^2
    let mut weights = vec![0.0f32; pixels.len()];
    for _ in 1..k {
        // Update weights: squared distance to nearest center
        let mut weight_sum = 0.0f32;
        for (i, &px) in pixels.iter().enumerate() {
            let px_f = [px[0] as f32, px[1] as f32, px[2] as f32];
            let min_dist_sq = centers
                .iter()
                .map(|c| {
                    let dr = px_f[0] - c[0];
                    let dg = px_f[1] - c[1];
                    let db = px_f[2] - c[2];
                    dr * dr + dg * dg + db * db
                })
                .fold(f32::INFINITY, f32::min);
            weights[i] = min_dist_sq;
            weight_sum += min_dist_sq;
        }
        if weight_sum < f32::EPSILON {
            // All pixels are identical; duplicate the last center
            centers.push(*centers.last().unwrap_or(&[0.0; 3]));
            continue;
        }
        // Deterministic weighted pick using a linear-congruential RNG seeded on
        // current center count and first pixel
        let rng_seed = (centers.len() as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(pixels[0][0] as u64);
        let threshold = ((rng_seed >> 11) as f32 / (u64::MAX >> 11) as f32) * weight_sum;
        let mut cumulative = 0.0f32;
        let mut chosen = pixels.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centers.push([
            pixels[chosen][0] as f32,
            pixels[chosen][1] as f32,
            pixels[chosen][2] as f32,
        ]);
    }

    // ── Lloyd iterations ──────────────────────────────────────────────────────
    let mut assignments = vec![0usize; pixels.len()];
    let tol_sq = tolerance * tolerance;

    for _iter in 0..max_iterations {
        // Assignment step
        for (i, &px) in pixels.iter().enumerate() {
            let px_f = [px[0] as f32, px[1] as f32, px[2] as f32];
            let nearest = centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = {
                        let dr = px_f[0] - a[0];
                        let dg = px_f[1] - a[1];
                        let db = px_f[2] - a[2];
                        dr * dr + dg * dg + db * db
                    };
                    let db = {
                        let dr = px_f[0] - b[0];
                        let dg = px_f[1] - b[1];
                        let db_ = px_f[2] - b[2];
                        dr * dr + dg * dg + db_ * db_
                    };
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            assignments[i] = nearest;
        }

        // Update step
        let mut new_centers = vec![[0.0f32; 3]; k];
        let mut counts = vec![0u64; k];

        for (i, &px) in pixels.iter().enumerate() {
            let c = assignments[i];
            new_centers[c][0] += px[0] as f32;
            new_centers[c][1] += px[1] as f32;
            new_centers[c][2] += px[2] as f32;
            counts[c] += 1;
        }

        let mut max_movement_sq = 0.0f32;
        for (ci, cnt) in counts.iter().enumerate() {
            if *cnt > 0 {
                let n = *cnt as f32;
                new_centers[ci][0] /= n;
                new_centers[ci][1] /= n;
                new_centers[ci][2] /= n;
            } else {
                // Empty cluster: keep previous center
                new_centers[ci] = centers[ci];
            }
            let dr = new_centers[ci][0] - centers[ci][0];
            let dg = new_centers[ci][1] - centers[ci][1];
            let db = new_centers[ci][2] - centers[ci][2];
            let mv = dr * dr + dg * dg + db * db;
            if mv > max_movement_sq {
                max_movement_sq = mv;
            }
        }

        centers = new_centers;

        if max_movement_sq < tol_sq {
            break;
        }
    }

    centers
        .into_iter()
        .map(|c| {
            [
                c[0].clamp(0.0, 255.0).round() as u8,
                c[1].clamp(0.0, 255.0).round() as u8,
                c[2].clamp(0.0, 255.0).round() as u8,
            ]
        })
        .collect()
}

// ── Unified quantizer API ─────────────────────────────────────────────────────

/// Selects the color-quantization algorithm to use.
#[derive(Debug, Clone)]
pub enum QuantizerAlgorithm {
    /// Heckbert (1982) median-cut: splits the bounding box along the widest channel.
    MedianCut,
    /// Gervautz & Purgathofer (1988) octree: inserts into an 8-level octree and
    /// merges deepest siblings to fit within the palette size.
    Octree,
    /// Lloyd (1957) k-means with k-means++ seeding.
    KMeans {
        /// Maximum number of Lloyd iterations.
        max_iterations: u32,
        /// Early-stop threshold in RGB Euclidean distance units.
        tolerance: f32,
    },
}

/// Quantize a set of RGB pixels to a palette of `k` colors using the specified algorithm.
///
/// # Arguments
///
/// * `pixels` — input RGB pixels as `[r, g, b]` bytes.
/// * `k` — desired palette size (> 0).
/// * `alg` — algorithm to use.
///
/// # Returns
///
/// A vector of up to `k` representative `[r, g, b]` palette entries.
#[must_use]
pub fn quantize_palette(pixels: &[[u8; 3]], k: usize, alg: QuantizerAlgorithm) -> Vec<[u8; 3]> {
    match alg {
        QuantizerAlgorithm::MedianCut => {
            // Convert to Rgb8 and back
            let rgb8: Vec<Rgb8> = pixels.iter().map(|&[r, g, b]| Rgb8::new(r, g, b)).collect();
            median_cut(&rgb8, k)
                .into_iter()
                .map(|c| c.to_array())
                .collect()
        }
        QuantizerAlgorithm::Octree => quantize_octree(pixels, k),
        QuantizerAlgorithm::KMeans {
            max_iterations,
            tolerance,
        } => quantize_kmeans(pixels, k, max_iterations, tolerance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb8_new() {
        let c = Rgb8::new(10, 20, 30);
        assert_eq!(c.r, 10);
        assert_eq!(c.g, 20);
        assert_eq!(c.b, 30);
    }

    #[test]
    fn test_rgb8_to_array() {
        let c = Rgb8::new(1, 2, 3);
        assert_eq!(c.to_array(), [1, 2, 3]);
    }

    #[test]
    fn test_distance_sq_same() {
        let c = Rgb8::new(100, 100, 100);
        assert_eq!(c.distance_sq(c), 0);
    }

    #[test]
    fn test_distance_sq_known() {
        let a = Rgb8::new(0, 0, 0);
        let b = Rgb8::new(3, 4, 0);
        assert_eq!(a.distance_sq(b), 25); // 9 + 16
    }

    #[test]
    fn test_median_cut_empty() {
        let result = median_cut(&[], 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_median_cut_single() {
        let colors = vec![Rgb8::new(128, 128, 128)];
        let result = median_cut(&colors, 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_median_cut_two_clusters() {
        let mut colors = Vec::new();
        for _ in 0..50 {
            colors.push(Rgb8::new(200, 50, 50));
        }
        for _ in 0..50 {
            colors.push(Rgb8::new(50, 50, 200));
        }
        let palette = median_cut(&colors, 2);
        assert_eq!(palette.len(), 2);
    }

    #[test]
    fn test_palette_nearest() {
        let p = Palette::new(vec![Rgb8::new(0, 0, 0), Rgb8::new(255, 255, 255)]);
        let (idx, _) = p
            .nearest(Rgb8::new(200, 200, 200))
            .expect("nearest color should be found");
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_palette_quantize() {
        let p = Palette::new(vec![Rgb8::new(0, 0, 0), Rgb8::new(255, 255, 255)]);
        let indices = p.quantize(&[Rgb8::new(10, 10, 10), Rgb8::new(245, 245, 245)]);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_quantization_mse_exact() {
        let colors = vec![Rgb8::new(100, 100, 100)];
        let p = Palette::new(vec![Rgb8::new(100, 100, 100)]);
        let mse = quantization_mse(&p, &colors);
        assert!((mse).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dithering_error() {
        let orig = Rgb8::new(200, 100, 50);
        let quant = Rgb8::new(190, 110, 60);
        let (er, eg, eb) = dithering_error(orig, quant);
        assert_eq!(er, 10);
        assert_eq!(eg, -10);
        assert_eq!(eb, -10);
    }

    #[test]
    fn test_apply_error_clamping() {
        let pixel = Rgb8::new(250, 5, 128);
        let error = (20, -20, 0);
        let result = apply_error(pixel, error, 1.0);
        assert_eq!(result.r, 255); // clamped
        assert_eq!(result.g, 0); // clamped
        assert_eq!(result.b, 128);
    }

    #[test]
    fn test_color_histogram() {
        let colors = vec![
            Rgb8::new(1, 1, 1),
            Rgb8::new(2, 2, 2),
            Rgb8::new(1, 1, 1),
            Rgb8::new(1, 1, 1),
        ];
        let hist = color_histogram(&colors);
        assert_eq!(hist[0].0, Rgb8::new(1, 1, 1));
        assert_eq!(hist[0].1, 3);
        assert_eq!(hist[1].0, Rgb8::new(2, 2, 2));
        assert_eq!(hist[1].1, 1);
    }

    #[test]
    fn test_palette_from_colors() {
        let colors = vec![
            Rgb8::new(10, 10, 10),
            Rgb8::new(20, 20, 20),
            Rgb8::new(240, 240, 240),
            Rgb8::new(250, 250, 250),
        ];
        let p = Palette::from_colors(&colors, 2);
        assert_eq!(p.len(), 2);
    }

    // ── Octree quantization tests ─────────────────────────────────────────────

    #[test]
    fn test_octree_quantize_basic() {
        // 256 distinct colors → quantize to 16 → at most 16 palette entries
        let pixels: Vec<[u8; 3]> = (0..=255u8).map(|i| [i, 255 - i, i / 2]).collect();
        let palette = quantize_octree(&pixels, 16);
        assert!(
            !palette.is_empty(),
            "octree should produce a non-empty palette"
        );
        assert!(
            palette.len() <= 16,
            "palette should have at most 16 entries, got {}",
            palette.len()
        );
    }

    #[test]
    fn test_octree_quantize_empty() {
        let palette = quantize_octree(&[], 8);
        assert!(palette.is_empty());
    }

    #[test]
    fn test_octree_quantize_single_color() {
        let pixels = vec![[200u8, 100, 50]; 100];
        let palette = quantize_octree(&pixels, 8);
        assert!(
            !palette.is_empty(),
            "single-color input should produce palette"
        );
        // All entries should be close to the input color
        for entry in &palette {
            let dr = i32::from(entry[0]) - 200;
            let dg = i32::from(entry[1]) - 100;
            let db = i32::from(entry[2]) - 50;
            let dist = ((dr * dr + dg * dg + db * db) as f32).sqrt();
            assert!(
                dist < 30.0,
                "palette entry {:?} far from source color",
                entry
            );
        }
    }

    #[test]
    fn test_octree_quantize_k_zero_returns_empty() {
        let pixels = vec![[1u8, 2, 3]; 10];
        let palette = quantize_octree(&pixels, 0);
        assert!(palette.is_empty());
    }

    // ── k-means quantization tests ────────────────────────────────────────────

    #[test]
    fn test_kmeans_quantize_basic_k4() {
        // 4 well-separated clusters of 50 pixels each
        let mut pixels: Vec<[u8; 3]> = Vec::new();
        for _ in 0..50 {
            pixels.push([20, 20, 20]);
        }
        for _ in 0..50 {
            pixels.push([200, 20, 20]);
        }
        for _ in 0..50 {
            pixels.push([20, 200, 20]);
        }
        for _ in 0..50 {
            pixels.push([20, 20, 200]);
        }
        let palette = quantize_kmeans(&pixels, 4, 50, 1.0);
        assert_eq!(palette.len(), 4, "k-means should return exactly k centers");
    }

    #[test]
    fn test_kmeans_quantize_empty() {
        let palette = quantize_kmeans(&[], 4, 10, 1.0);
        assert!(palette.is_empty());
    }

    #[test]
    fn test_kmeans_quantize_k_zero() {
        let pixels = vec![[1u8, 2, 3]; 10];
        let palette = quantize_kmeans(&pixels, 0, 10, 1.0);
        assert!(palette.is_empty());
    }

    #[test]
    fn test_kmeans_deterministic() {
        let pixels: Vec<[u8; 3]> = (0..100u8).map(|i| [i, 255 - i, i]).collect();
        let a = quantize_kmeans(&pixels, 4, 20, 1.0);
        let b = quantize_kmeans(&pixels, 4, 20, 1.0);
        assert_eq!(a, b, "k-means should be deterministic for same input");
    }

    // ── quantize_palette dispatch tests ──────────────────────────────────────

    #[test]
    fn test_quantize_palette_median_cut() {
        let pixels: Vec<[u8; 3]> = (0..64u8).map(|i| [i * 4, 255 - i * 4, 128]).collect();
        let palette = quantize_palette(&pixels, 8, QuantizerAlgorithm::MedianCut);
        assert!(palette.len() <= 8 && !palette.is_empty());
    }

    #[test]
    fn test_quantize_palette_octree() {
        let pixels: Vec<[u8; 3]> = (0..64u8).map(|i| [i * 4, 255 - i * 4, 128]).collect();
        let palette = quantize_palette(&pixels, 8, QuantizerAlgorithm::Octree);
        assert!(palette.len() <= 8 && !palette.is_empty());
    }

    #[test]
    fn test_quantize_palette_kmeans() {
        let pixels: Vec<[u8; 3]> = (0..64u8).map(|i| [i * 4, 255 - i * 4, 128]).collect();
        let palette = quantize_palette(
            &pixels,
            8,
            QuantizerAlgorithm::KMeans {
                max_iterations: 20,
                tolerance: 1.0,
            },
        );
        assert_eq!(palette.len(), 8);
    }
}
