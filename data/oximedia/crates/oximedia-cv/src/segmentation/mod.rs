//! Image segmentation: connected components, region growing, and a
//! watershed-like approximation.
//!
//! All algorithms operate on flat, row-major grayscale `f32` or label `u32`
//! images.

pub mod person_bg;

/// A per-instance binary mask.
///
/// Each `InstanceMask` represents one detected object instance with its own
/// label, bounding box, and pixel mask.
#[derive(Debug, Clone)]
pub struct InstanceMask {
    /// Instance ID (1-based).
    pub instance_id: u32,
    /// Semantic class label (e.g. 1 = person, 2 = car, …).
    pub class_label: u32,
    /// Confidence score from the segmenter (0.0–1.0).
    pub score: f32,
    /// Bounding box `(x_min, y_min, x_max, y_max)`.
    pub bbox: (usize, usize, usize, usize),
    /// Binary mask for the bounding box region.
    ///
    /// Length == `(x_max - x_min + 1) * (y_max - y_min + 1)`.
    /// `true` = foreground (object), `false` = background.
    pub mask: Vec<bool>,
    /// Width of the mask (== `x_max - x_min + 1`).
    pub mask_width: usize,
    /// Height of the mask (== `y_max - y_min + 1`).
    pub mask_height: usize,
}

impl InstanceMask {
    /// Create a new instance mask from a full-image boolean slice.
    ///
    /// The `full_mask` must have `image_width * image_height` elements.
    #[must_use]
    pub fn from_full_mask(
        instance_id: u32,
        class_label: u32,
        score: f32,
        full_mask: &[bool],
        image_width: usize,
        image_height: usize,
    ) -> Self {
        // Compute bounding box
        let mut x_min = usize::MAX;
        let mut y_min = usize::MAX;
        let mut x_max = 0;
        let mut y_max = 0;
        let mut found = false;

        for y in 0..image_height {
            for x in 0..image_width {
                if y * image_width + x < full_mask.len() && full_mask[y * image_width + x] {
                    x_min = x_min.min(x);
                    y_min = y_min.min(y);
                    x_max = x_max.max(x);
                    y_max = y_max.max(y);
                    found = true;
                }
            }
        }

        if !found {
            return Self {
                instance_id,
                class_label,
                score,
                bbox: (0, 0, 0, 0),
                mask: Vec::new(),
                mask_width: 0,
                mask_height: 0,
            };
        }

        let mw = x_max - x_min + 1;
        let mh = y_max - y_min + 1;
        let mut mask = vec![false; mw * mh];

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let src_idx = y * image_width + x;
                let dst_idx = (y - y_min) * mw + (x - x_min);
                if src_idx < full_mask.len() {
                    mask[dst_idx] = full_mask[src_idx];
                }
            }
        }

        Self {
            instance_id,
            class_label,
            score,
            bbox: (x_min, y_min, x_max, y_max),
            mask,
            mask_width: mw,
            mask_height: mh,
        }
    }

    /// Number of foreground pixels in the mask.
    #[must_use]
    pub fn area(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }

    /// Check whether pixel `(x, y)` (image coordinates) is foreground.
    #[must_use]
    pub fn contains(&self, x: usize, y: usize) -> bool {
        let (x_min, y_min, x_max, y_max) = self.bbox;
        if x < x_min || x > x_max || y < y_min || y > y_max {
            return false;
        }
        let mx = x - x_min;
        let my = y - y_min;
        let idx = my * self.mask_width + mx;
        idx < self.mask.len() && self.mask[idx]
    }
}

/// Configuration for instance segmentation.
#[derive(Debug, Clone)]
pub struct InstanceSegConfig {
    /// Minimum area (pixels) for an instance to be kept.
    pub min_area: usize,
    /// Confidence threshold for accepting a detected instance.
    pub min_score: f32,
    /// Maximum number of instances to return.
    pub max_instances: usize,
    /// Non-maximum suppression IoU threshold for merging overlapping instances.
    pub nms_iou_threshold: f32,
}

impl Default for InstanceSegConfig {
    fn default() -> Self {
        Self {
            min_area: 100,
            min_score: 0.5,
            max_instances: 32,
            nms_iou_threshold: 0.5,
        }
    }
}

/// CPU-based instance segmentation using connected-component analysis on a
/// class-probability map.
///
/// This implementation does not require a neural network at runtime.  It
/// uses the following pipeline:
///
/// 1. Threshold the probability map to produce a binary foreground mask.
/// 2. Run connected-component labelling to identify individual instances.
/// 3. Filter components by area and score.
/// 4. Apply greedy non-maximum suppression on bounding boxes by IoU.
pub struct InstanceSegmenter {
    config: InstanceSegConfig,
}

impl InstanceSegmenter {
    /// Create a new instance segmenter with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: InstanceSegConfig::default(),
        }
    }

    /// Create a new instance segmenter with custom configuration.
    #[must_use]
    pub fn with_config(config: InstanceSegConfig) -> Self {
        Self { config }
    }

    /// Segment instances from a probability/score map.
    ///
    /// # Arguments
    ///
    /// * `prob_map` – Per-pixel foreground probability (0.0–1.0), row-major.
    /// * `width`, `height` – Image dimensions.
    /// * `class_label` – Semantic class to assign to all detected instances.
    ///
    /// Returns a vector of detected `InstanceMask`, sorted by score descending.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn segment(
        &self,
        prob_map: &[f32],
        width: usize,
        height: usize,
        class_label: u32,
    ) -> Vec<InstanceMask> {
        if prob_map.is_empty() || width == 0 || height == 0 {
            return Vec::new();
        }

        let threshold = self.config.min_score;

        // Step 1: threshold to binary
        let binary: Vec<f32> = prob_map
            .iter()
            .map(|&p| if p >= threshold { 1.0 } else { 0.0 })
            .collect();

        // Step 2: connected components
        let label_map = connected_components(&binary, width, height, 0.5);
        let n_labels = label_map.num_labels();
        if n_labels == 0 {
            return Vec::new();
        }

        // Step 3: build instance masks and compute per-instance mean probability
        let mut instances: Vec<InstanceMask> = Vec::new();

        for label in 1..=(n_labels as u32) {
            // Build full-image bool mask for this label
            let full_mask: Vec<bool> = label_map.labels.iter().map(|&l| l == label).collect();
            let area = full_mask.iter().filter(|&&b| b).count();
            if area < self.config.min_area {
                continue;
            }

            // Compute mean probability as score
            let score = prob_map
                .iter()
                .zip(full_mask.iter())
                .filter(|(_, &fg)| fg)
                .map(|(&p, _)| p)
                .sum::<f32>()
                / area as f32;

            let mask =
                InstanceMask::from_full_mask(label, class_label, score, &full_mask, width, height);

            instances.push(mask);
        }

        // Step 4: sort by score descending
        instances.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 5: greedy IoU-based NMS
        let instances = self.nms(instances);

        // Step 6: limit count
        instances
            .into_iter()
            .take(self.config.max_instances)
            .collect()
    }

    /// Segment multiple classes simultaneously.
    ///
    /// `prob_maps` is a slice of `(prob_map, class_label)` pairs.
    /// Results are merged and re-sorted by score.
    #[must_use]
    pub fn segment_multiclass(
        &self,
        prob_maps: &[(&[f32], u32)],
        width: usize,
        height: usize,
    ) -> Vec<InstanceMask> {
        let mut all: Vec<InstanceMask> = prob_maps
            .iter()
            .flat_map(|&(map, cls)| self.segment(map, width, height, cls))
            .collect();

        // Renumber instance IDs globally
        for (i, m) in all.iter_mut().enumerate() {
            m.instance_id = i as u32 + 1;
        }

        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.into_iter().take(self.config.max_instances).collect()
    }

    /// Greedy NMS: remove instances with IoU > threshold with a higher-scored instance.
    fn nms(&self, sorted_instances: Vec<InstanceMask>) -> Vec<InstanceMask> {
        let mut kept: Vec<InstanceMask> = Vec::new();
        let threshold = self.config.nms_iou_threshold;

        'outer: for candidate in sorted_instances {
            for existing in &kept {
                if bbox_iou(candidate.bbox, existing.bbox) > threshold {
                    continue 'outer;
                }
            }
            kept.push(candidate);
        }

        kept
    }
}

impl Default for InstanceSegmenter {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute intersection-over-union for two bounding boxes.
fn bbox_iou(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> f32 {
    let ix_min = a.0.max(b.0);
    let iy_min = a.1.max(b.1);
    let ix_max = a.2.min(b.2);
    let iy_max = a.3.min(b.3);

    if ix_min > ix_max || iy_min > iy_max {
        return 0.0;
    }

    let intersection = ((ix_max - ix_min + 1) * (iy_max - iy_min + 1)) as f32;
    let area_a = ((a.2 - a.0 + 1) * (a.3 - a.1 + 1)) as f32;
    let area_b = ((b.2 - b.0 + 1) * (b.3 - b.1 + 1)) as f32;
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}

/// A 2-D label map produced by segmentation algorithms.
#[derive(Debug, Clone)]
pub struct LabelMap {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Label for each pixel (0 = background / unlabelled).
    pub labels: Vec<u32>,
}

impl LabelMap {
    /// Create a new all-zero label map.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            labels: vec![0u32; width * height],
        }
    }

    /// Get the label at pixel `(x, y)`.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> u32 {
        self.labels[y * self.width + x]
    }

    /// Set the label at pixel `(x, y)`.
    pub fn set(&mut self, x: usize, y: usize, label: u32) {
        self.labels[y * self.width + x] = label;
    }

    /// Number of distinct non-zero labels.
    #[must_use]
    pub fn num_labels(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for &l in &self.labels {
            if l != 0 {
                seen.insert(l);
            }
        }
        seen.len()
    }

    /// Count pixels belonging to `label`.
    #[must_use]
    pub fn count_label(&self, label: u32) -> usize {
        self.labels.iter().filter(|&&l| l == label).count()
    }

    /// Bounding box of a label: `(x_min, y_min, x_max, y_max)`.
    #[must_use]
    pub fn bounding_box(&self, label: u32) -> Option<(usize, usize, usize, usize)> {
        let mut x_min = usize::MAX;
        let mut y_min = usize::MAX;
        let mut x_max = 0;
        let mut y_max = 0;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) == label {
                    x_min = x_min.min(x);
                    y_min = y_min.min(y);
                    x_max = x_max.max(x);
                    y_max = y_max.max(y);
                    found = true;
                }
            }
        }

        found.then_some((x_min, y_min, x_max, y_max))
    }
}

/// Connected-component labeling using 4-connectivity (BFS / union-find hybrid).
///
/// Input: binary image where pixels > `threshold` are foreground.
#[must_use]
pub fn connected_components(
    image: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() || width == 0 || height == 0 {
        return map;
    }

    let mut current_label = 0u32;
    let mut stack = Vec::new();

    for sy in 0..height {
        for sx in 0..width {
            if image[sy * width + sx] <= threshold || map.get(sx, sy) != 0 {
                continue;
            }
            // New component
            current_label += 1;
            map.set(sx, sy, current_label);
            stack.push((sx, sy));

            while let Some((x, y)) = stack.pop() {
                // 4-connected neighbours
                let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dx, dy) in neighbours {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if image[ny * width + nx] > threshold && map.get(nx, ny) == 0 {
                        map.set(nx, ny, current_label);
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }

    map
}

/// Region-growing segmentation starting from a seed pixel.
///
/// Grows while the absolute difference from the seed value is within
/// `tolerance`.
#[must_use]
pub fn region_growing(
    image: &[f32],
    width: usize,
    height: usize,
    seed_x: usize,
    seed_y: usize,
    tolerance: f32,
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() || seed_x >= width || seed_y >= height {
        return map;
    }

    let seed_val = image[seed_y * width + seed_x];
    let label = 1u32;
    let mut queue = std::collections::VecDeque::new();
    map.set(seed_x, seed_y, label);
    queue.push_back((seed_x, seed_y));

    while let Some((x, y)) = queue.pop_front() {
        let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in neighbours {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;
            if map.get(nx, ny) == 0 {
                let val = image[ny * width + nx];
                if (val - seed_val).abs() <= tolerance {
                    map.set(nx, ny, label);
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    map
}

/// Watershed-like segmentation approximation using distance-based flooding.
///
/// Seeds are provided as `(x, y)` pairs and each gets a unique label ≥ 1.
/// Pixels are flooded in order of increasing distance to any seed.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn watershed_approx(
    image: &[f32],
    width: usize,
    height: usize,
    seeds: &[(usize, usize)],
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() || seeds.is_empty() {
        return map;
    }

    // Priority queue: (distance * 1000 as u64, x, y, label)
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut heap: BinaryHeap<Reverse<(u64, usize, usize, u32)>> = BinaryHeap::new();

    for (i, &(sx, sy)) in seeds.iter().enumerate() {
        if sx < width && sy < height {
            let label = i as u32 + 1;
            map.set(sx, sy, label);
            heap.push(Reverse((0, sx, sy, label)));
        }
    }

    while let Some(Reverse((_, x, y, label))) = heap.pop() {
        let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in neighbours {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;
            if map.get(nx, ny) == 0 {
                // Distance metric: image gradient magnitude (higher = harder to cross)
                let weight = (image[ny * width + nx] * 1000.0) as u64;
                map.set(nx, ny, label);
                heap.push(Reverse((weight, nx, ny, label)));
            }
        }
    }

    map
}

/// Simple mean-shift-inspired superpixel cluster (single pass, approximate).
///
/// Groups pixels whose value is within `bandwidth` of cluster centres.
/// Returns a label map (centre-based), capped at `max_clusters` clusters.
#[must_use]
pub fn mean_shift_simple(
    image: &[f32],
    width: usize,
    height: usize,
    bandwidth: f32,
    max_clusters: usize,
) -> LabelMap {
    let mut map = LabelMap::new(width, height);
    if image.is_empty() {
        return map;
    }

    let mut centres: Vec<f32> = Vec::new();
    let mut next_label = 1u32;

    for y in 0..height {
        for x in 0..width {
            let val = image[y * width + x];
            // Find nearest cluster centre
            let closest = centres
                .iter()
                .enumerate()
                .find(|(_, &c)| (c - val).abs() <= bandwidth);

            let label = if let Some((i, _)) = closest {
                i as u32 + 1
            } else if centres.len() < max_clusters {
                centres.push(val);
                let l = next_label;
                next_label += 1;
                l
            } else {
                // Assign to nearest centre regardless
                centres
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        ((*a - val).abs())
                            .partial_cmp(&((*b - val).abs()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(1, |(i, _)| i as u32 + 1)
            };

            map.set(x, y, label);
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(val: f32, w: usize, h: usize) -> Vec<f32> {
        vec![val; w * h]
    }

    fn binary_image() -> (Vec<f32>, usize, usize) {
        // 5x5 image with two separate blobs
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        (data, 5, 5)
    }

    #[test]
    fn test_label_map_new() {
        let m = LabelMap::new(4, 4);
        assert_eq!(m.width, 4);
        assert_eq!(m.height, 4);
        assert!(m.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_label_map_set_get() {
        let mut m = LabelMap::new(5, 5);
        m.set(2, 3, 7);
        assert_eq!(m.get(2, 3), 7);
    }

    #[test]
    fn test_label_map_num_labels() {
        let mut m = LabelMap::new(3, 3);
        m.set(0, 0, 1);
        m.set(1, 1, 2);
        m.set(2, 2, 2);
        assert_eq!(m.num_labels(), 2);
    }

    #[test]
    fn test_label_map_count_label() {
        let mut m = LabelMap::new(3, 3);
        m.set(0, 0, 1);
        m.set(1, 1, 1);
        m.set(2, 2, 2);
        assert_eq!(m.count_label(1), 2);
        assert_eq!(m.count_label(2), 1);
        assert_eq!(m.count_label(3), 0);
    }

    #[test]
    fn test_label_map_bounding_box() {
        let mut m = LabelMap::new(10, 10);
        m.set(2, 3, 1);
        m.set(5, 7, 1);
        let bb = m.bounding_box(1).expect("bounding_box should succeed");
        assert_eq!(bb, (2, 3, 5, 7));
    }

    #[test]
    fn test_label_map_bounding_box_missing() {
        let m = LabelMap::new(5, 5);
        assert!(m.bounding_box(42).is_none());
    }

    #[test]
    fn test_connected_components_two_blobs() {
        let (img, w, h) = binary_image();
        let map = connected_components(&img, w, h, 0.5);
        assert_eq!(map.num_labels(), 2);
    }

    #[test]
    fn test_connected_components_blank_image() {
        let map = connected_components(&uniform(0.0, 5, 5), 5, 5, 0.5);
        assert_eq!(map.num_labels(), 0);
    }

    #[test]
    fn test_connected_components_full_image() {
        let map = connected_components(&uniform(1.0, 4, 4), 4, 4, 0.5);
        assert_eq!(map.num_labels(), 1);
    }

    #[test]
    fn test_region_growing_uniform() {
        let img = uniform(0.5, 6, 6);
        let map = region_growing(&img, 6, 6, 2, 2, 0.1);
        // All pixels should be in region 1
        assert_eq!(map.count_label(1), 36);
    }

    #[test]
    fn test_region_growing_limited() {
        let mut img = uniform(0.0, 6, 6);
        // Seed area with value 0.5, rest 0.0
        for x in 0..3 {
            img[0 * 6 + x] = 0.5;
            img[1 * 6 + x] = 0.5;
        }
        let map = region_growing(&img, 6, 6, 0, 0, 0.1);
        // Only the 0.5 patch should be included
        assert!(map.count_label(1) <= 6);
    }

    #[test]
    fn test_watershed_two_seeds() {
        let img = uniform(0.5, 10, 10);
        let seeds = vec![(1, 1), (8, 8)];
        let map = watershed_approx(&img, 10, 10, &seeds);
        assert_eq!(map.num_labels(), 2);
        // Every pixel should be labelled
        assert_eq!(map.labels.iter().filter(|&&l| l == 0).count(), 0);
    }

    #[test]
    fn test_watershed_no_seeds_returns_blank() {
        let img = uniform(0.5, 5, 5);
        let map = watershed_approx(&img, 5, 5, &[]);
        assert_eq!(map.num_labels(), 0);
    }

    #[test]
    fn test_mean_shift_uniform_image() {
        let img = uniform(0.5, 5, 5);
        let map = mean_shift_simple(&img, 5, 5, 0.1, 10);
        // All same value → single cluster
        assert_eq!(map.num_labels(), 1);
    }

    #[test]
    fn test_mean_shift_two_clusters() {
        let mut img = vec![0.0f32; 10];
        // First 5 pixels near 0.0, last 5 near 1.0
        for i in 5..10 {
            img[i] = 1.0;
        }
        let map = mean_shift_simple(&img, 10, 1, 0.1, 10);
        assert_eq!(map.num_labels(), 2);
    }

    // ── Instance segmentation tests ─────────────────────────────────────────

    fn two_blob_prob_map() -> (Vec<f32>, usize, usize) {
        let w = 20usize;
        let h = 20usize;
        let mut map = vec![0.0f32; w * h];
        // Blob 1: top-left 6×6 square
        for y in 1..7 {
            for x in 1..7 {
                map[y * w + x] = 0.9;
            }
        }
        // Blob 2: bottom-right 6×6 square
        for y in 13..19 {
            for x in 13..19 {
                map[y * w + x] = 0.8;
            }
        }
        (map, w, h)
    }

    #[test]
    fn test_instance_mask_from_full_mask_empty() {
        let mask = vec![false; 100];
        let im = InstanceMask::from_full_mask(1, 1, 0.5, &mask, 10, 10);
        assert_eq!(im.area(), 0);
    }

    #[test]
    fn test_instance_mask_area() {
        let mut mask = vec![false; 100];
        for i in 10..20 {
            mask[i] = true;
        }
        let im = InstanceMask::from_full_mask(1, 1, 0.9, &mask, 10, 10);
        assert_eq!(im.area(), 10);
    }

    #[test]
    fn test_instance_mask_contains() {
        let mut mask = vec![false; 100];
        mask[55] = true; // x=5, y=5
        let im = InstanceMask::from_full_mask(1, 1, 0.9, &mask, 10, 10);
        assert!(im.contains(5, 5));
        assert!(!im.contains(0, 0));
    }

    #[test]
    fn test_instance_segmenter_two_blobs() {
        let (map, w, h) = two_blob_prob_map();
        let config = InstanceSegConfig {
            min_area: 10,
            min_score: 0.5,
            max_instances: 10,
            nms_iou_threshold: 0.5,
        };
        let segmenter = InstanceSegmenter::with_config(config);
        let instances = segmenter.segment(&map, w, h, 1);
        assert_eq!(instances.len(), 2, "expected two instances");
    }

    #[test]
    fn test_instance_segmenter_min_area_filter() {
        let (map, w, h) = two_blob_prob_map();
        let config = InstanceSegConfig {
            min_area: 1000, // Much larger than any blob
            ..InstanceSegConfig::default()
        };
        let segmenter = InstanceSegmenter::with_config(config);
        let instances = segmenter.segment(&map, w, h, 1);
        assert!(instances.is_empty(), "all instances below min_area");
    }

    #[test]
    fn test_instance_segmenter_empty_input() {
        let segmenter = InstanceSegmenter::new();
        let instances = segmenter.segment(&[], 0, 0, 1);
        assert!(instances.is_empty());
    }

    #[test]
    fn test_instance_segmenter_sorted_by_score() {
        let (map, w, h) = two_blob_prob_map();
        let config = InstanceSegConfig {
            min_area: 5,
            ..InstanceSegConfig::default()
        };
        let segmenter = InstanceSegmenter::with_config(config);
        let instances = segmenter.segment(&map, w, h, 1);
        // Verify descending score order
        for pair in instances.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = (0, 0, 5, 5);
        let b = (10, 10, 15, 15);
        assert!((bbox_iou(a, b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_iou_full_overlap() {
        let a = (0, 0, 9, 9);
        let b = (0, 0, 9, 9);
        assert!((bbox_iou(a, b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_instance_segmenter_multiclass() {
        let w = 20usize;
        let h = 20usize;
        let mut map1 = vec![0.0f32; w * h];
        for y in 1..6 {
            for x in 1..6 {
                map1[y * w + x] = 0.9;
            }
        }
        let mut map2 = vec![0.0f32; w * h];
        for y in 13..18 {
            for x in 13..18 {
                map2[y * w + x] = 0.85;
            }
        }
        let config = InstanceSegConfig {
            min_area: 5,
            ..Default::default()
        };
        let seg = InstanceSegmenter::with_config(config);
        let instances = seg.segment_multiclass(&[(&map1, 1), (&map2, 2)], w, h);
        assert_eq!(instances.len(), 2);
        // Classes should be 1 and 2
        let classes: std::collections::HashSet<u32> =
            instances.iter().map(|m| m.class_label).collect();
        assert!(classes.contains(&1));
        assert!(classes.contains(&2));
    }
}
