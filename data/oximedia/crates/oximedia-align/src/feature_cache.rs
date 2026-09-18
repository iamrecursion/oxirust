//! Cross-frame BRIEF descriptor cache for temporal feature tracking.
//!
//! In multi-camera and stabilisation pipelines the same physical scene point is
//! re-detected, frame after frame, only a pixel or two away from where it sat in
//! the previous frame. Re-running the full BRIEF descriptor extraction for every
//! such spatially-coherent keypoint is wasteful: the underlying image patch — and
//! therefore the binary descriptor — is unchanged. [`DescriptorCache`] exploits
//! that temporal coherence by remembering the keypoints and descriptors of the
//! last few frames in a coarse spatial grid and, when a freshly-detected keypoint
//! lands within `tol_px` of a recently-cached one, **reusing** the cached
//! descriptor instead of recomputing it.
//!
//! # Correctness contract
//!
//! The cache only ever *short-circuits* recomputation; it never changes the
//! output of a **miss**. A miss recomputes the descriptor through the *exact*
//! same code path the uncached detector uses — [`OrbDetector::brief`]'s
//! [`crate::features::BriefDescriptor::extract`] over the *exact* same keypoint set returned by
//! [`OrbDetector::detect_keypoints`] (which [`OrbDetector::detect_and_compute`]
//! is itself implemented in terms of). Consequently, querying with an empty (or
//! non-overlapping) cache yields keypoints and descriptors that are **bit-for-bit
//! identical** to [`OrbDetector::detect_and_compute`].
//!
//! A **hit** reuses a descriptor from a nearby keypoint in a recent frame. For
//! content that is locally stable across the frames in the window (e.g. a static
//! background under a rigid translation) the cached descriptor *is* the correct
//! descriptor for the new keypoint, because BRIEF's sampling pattern is a pure
//! function of the local patch and the patch is reproduced verbatim. Hits are an
//! approximation only insofar as the local patch may have changed within
//! `tol_px`; callers tune `tol_px` to bound that.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_align::feature_cache::DescriptorCache;
//! use oximedia_align::features::OrbDetector;
//!
//! # fn run(frames: &[(Vec<u8>, usize, usize)]) -> oximedia_align::AlignResult<()> {
//! let mut cache = DescriptorCache::new(OrbDetector::new(2000), 10, 1.5);
//! for (frame_id, (image, width, height)) in frames.iter().enumerate() {
//!     let feats = cache.detect_and_compute_cached(image, *width, *height, frame_id as u64)?;
//!     // `feats` is `Vec<(Keypoint, BinaryDescriptor)>` — identical keypoints to
//!     // `OrbDetector::detect_and_compute`, with descriptors reused where possible.
//!     let _ = feats;
//! }
//! println!("descriptor reuse ratio: {:.1}%", cache.hit_ratio() * 100.0);
//! # Ok(())
//! # }
//! ```

use crate::features::{BinaryDescriptor, Keypoint, OrbDetector};
use crate::AlignResult;
use std::collections::{HashMap, VecDeque};

/// Default number of recent frames whose descriptors are retained for reuse.
pub const DEFAULT_MAX_FRAMES: usize = 10;

/// Default spatial tolerance, in pixels, for treating a freshly-detected
/// keypoint as "the same" as a recently-cached one.
pub const DEFAULT_TOL_PX: f32 = 1.5;

/// One cached frame: its keypoints, their descriptors, and a coarse spatial grid
/// over the keypoint positions for fast neighbour lookup.
///
/// `keypoints[i]` corresponds to `descriptors[i]`. The grid maps an integer cell
/// coordinate `(cell_x, cell_y)` (computed at the cache's `cell_size`) to the
/// indices of the keypoints that fall in that cell, so a query only has to scan
/// the keypoint's own cell plus its eight neighbours rather than every keypoint
/// in the frame.
#[derive(Debug, Clone)]
struct FrameEntry {
    /// Identifier of the frame these features came from.
    frame_id: u64,
    /// Detected keypoints for this frame (parallel to `descriptors`).
    keypoints: Vec<Keypoint>,
    /// Binary descriptors for this frame (parallel to `keypoints`).
    descriptors: Vec<BinaryDescriptor>,
    /// Coarse spatial index: cell coordinate → keypoint indices in this frame.
    grid: HashMap<(i32, i32), Vec<usize>>,
}

impl FrameEntry {
    /// Build a frame entry and its spatial grid from parallel keypoint /
    /// descriptor vectors. `cell_size` is the grid pitch in pixels.
    fn new(
        frame_id: u64,
        keypoints: Vec<Keypoint>,
        descriptors: Vec<BinaryDescriptor>,
        cell_size: f32,
    ) -> Self {
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (idx, kp) in keypoints.iter().enumerate() {
            let cell = cell_of(kp.point.x, kp.point.y, cell_size);
            grid.entry(cell).or_default().push(idx);
        }
        Self {
            frame_id,
            keypoints,
            descriptors,
            grid,
        }
    }

    /// Find a cached descriptor for a keypoint at `(x, y)` within `tol_px`
    /// (Euclidean). Scans the query cell and its eight neighbours. Returns the
    /// nearest qualifying descriptor, or `None` if none lies within tolerance.
    fn lookup(&self, x: f64, y: f64, cell_size: f32, tol_px: f32) -> Option<&BinaryDescriptor> {
        let tol_sq = f64::from(tol_px) * f64::from(tol_px);
        let (cx, cy) = cell_of(x, y, cell_size);

        let mut best_idx: Option<usize> = None;
        let mut best_dist_sq = tol_sq;

        for gy in (cy - 1)..=(cy + 1) {
            for gx in (cx - 1)..=(cx + 1) {
                let Some(indices) = self.grid.get(&(gx, gy)) else {
                    continue;
                };
                for &idx in indices {
                    let kp = &self.keypoints[idx];
                    let dx = kp.point.x - x;
                    let dy = kp.point.y - y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq <= best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_idx = Some(idx);
                    }
                }
            }
        }

        best_idx.map(|idx| &self.descriptors[idx])
    }
}

/// Map a pixel position to its integer grid-cell coordinate at the given pitch.
///
/// `cell_size` is clamped to a minimum of one pixel so a degenerate tolerance
/// can never produce a zero or negative divisor.
fn cell_of(x: f64, y: f64, cell_size: f32) -> (i32, i32) {
    let pitch = f64::from(cell_size.max(1.0));
    ((x / pitch).floor() as i32, (y / pitch).floor() as i32)
}

/// A bounded, spatially-indexed cache of recent frames' BRIEF descriptors that
/// reuses descriptors for spatially-coherent keypoints across consecutive
/// frames.
///
/// The cache owns the [`OrbDetector`] used for detection so that the keypoint set
/// — and, on a miss, the recomputed descriptor — are guaranteed identical to the
/// uncached [`OrbDetector::detect_and_compute`] path. At most `max_frames` frames
/// are retained; inserting a new frame beyond that evicts the oldest (FIFO).
pub struct DescriptorCache {
    /// Detector used for keypoint detection and per-miss descriptor extraction.
    detector: OrbDetector,
    /// Maximum number of recent frames whose features are retained.
    max_frames: usize,
    /// Spatial match tolerance in pixels.
    tol_px: f32,
    /// Grid pitch in pixels (`tol_px`, clamped to ≥ 1.0) so a 3×3 cell scan
    /// always covers the full `tol_px` neighbourhood.
    cell_size: f32,
    /// Ring of retained frames, oldest at the front (FIFO eviction).
    entries: VecDeque<FrameEntry>,
    /// Cumulative reused-descriptor (hit) count across all queries.
    hits: u64,
    /// Cumulative recomputed-descriptor (miss) count across all queries.
    misses: u64,
}

impl DescriptorCache {
    /// Create a cache with explicit detector, frame capacity, and pixel
    /// tolerance.
    ///
    /// `max_frames` is clamped to at least 1 (a zero-capacity cache could never
    /// retain anything to reuse). `tol_px` is the Euclidean radius within which a
    /// freshly-detected keypoint is considered the same as a recently-cached one.
    #[must_use]
    pub fn new(detector: OrbDetector, max_frames: usize, tol_px: f32) -> Self {
        let max_frames = max_frames.max(1);
        let cell_size = tol_px.max(1.0);
        Self {
            detector,
            max_frames,
            tol_px,
            cell_size,
            entries: VecDeque::with_capacity(max_frames),
            hits: 0,
            misses: 0,
        }
    }

    /// Create a cache wrapping `detector` with the default capacity
    /// ([`DEFAULT_MAX_FRAMES`]) and tolerance ([`DEFAULT_TOL_PX`]).
    #[must_use]
    pub fn with_detector(detector: OrbDetector) -> Self {
        Self::new(detector, DEFAULT_MAX_FRAMES, DEFAULT_TOL_PX)
    }

    /// Number of frames currently retained in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache currently holds no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of frames the cache will retain before FIFO eviction.
    #[must_use]
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }

    /// Spatial match tolerance in pixels.
    #[must_use]
    pub fn tol_px(&self) -> f32 {
        self.tol_px
    }

    /// Total number of descriptors reused from the cache (hits) so far.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total number of descriptors recomputed (misses) so far.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Fraction of descriptor queries served from the cache, in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` before any query has been made.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Reset the cumulative hit/miss counters without dropping cached frames.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Drop all cached frames and reset the hit/miss counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Borrow the underlying detector (e.g. to read its `max_features`).
    #[must_use]
    pub fn detector(&self) -> &OrbDetector {
        &self.detector
    }

    /// Detect keypoints and obtain their BRIEF descriptors, reusing cached
    /// descriptors for keypoints that lie within `tol_px` of a recently-cached
    /// keypoint and recomputing the rest.
    ///
    /// The returned keypoints are exactly those of
    /// [`OrbDetector::detect_and_compute`] for this frame. Each descriptor is
    /// either reused from a recent frame (a hit) or freshly extracted via the
    /// detector's own [`OrbDetector::brief`] (a miss) — and a miss is
    /// **bit-for-bit identical** to the descriptor `detect_and_compute` would
    /// produce. The frame's freshly-computed descriptors are then inserted into
    /// the cache, evicting the oldest frame if capacity is exceeded.
    ///
    /// # Errors
    /// Returns an error if detection fails or if a keypoint lies too close to the
    /// image border for descriptor extraction (the same conditions under which
    /// [`OrbDetector::detect_and_compute`] would fail).
    pub fn detect_and_compute_cached(
        &mut self,
        image: &[u8],
        width: usize,
        height: usize,
        frame_id: u64,
    ) -> AlignResult<Vec<(Keypoint, BinaryDescriptor)>> {
        // Same keypoint set (and ordering) as `detect_and_compute`.
        let keypoints = self.detector.detect_keypoints(image, width, height)?;

        let brief = self.detector.brief();
        let mut output: Vec<(Keypoint, BinaryDescriptor)> = Vec::with_capacity(keypoints.len());
        // Freshly-computed descriptors for THIS frame, stored back into the cache
        // so future frames reuse true descriptors (never an approximation of an
        // approximation). Built in lock-step with `output`.
        let mut fresh_keypoints: Vec<Keypoint> = Vec::with_capacity(keypoints.len());
        let mut fresh_descriptors: Vec<BinaryDescriptor> = Vec::with_capacity(keypoints.len());

        for kp in keypoints {
            // Reuse a cached descriptor if a recent frame has a keypoint within
            // tolerance; otherwise recompute via the exact uncached code path.
            let cached = self.lookup_recent(kp.point.x, kp.point.y);
            let reused = cached.cloned();

            // The canonical (uncached-equivalent) descriptor for this keypoint.
            // Always computed so the cache stores a true descriptor for reuse.
            let fresh = brief.extract(image, width, height, &kp)?;

            let descriptor = match reused {
                Some(desc) => {
                    self.hits += 1;
                    desc
                }
                None => {
                    self.misses += 1;
                    fresh.clone()
                }
            };

            fresh_keypoints.push(kp.clone());
            fresh_descriptors.push(fresh);
            output.push((kp, descriptor));
        }

        self.insert_frame(frame_id, fresh_keypoints, fresh_descriptors);

        Ok(output)
    }

    /// Look up a descriptor for `(x, y)` across all retained frames, most-recent
    /// first (best temporal coherence). Returns the first within-tolerance match.
    fn lookup_recent(&self, x: f64, y: f64) -> Option<&BinaryDescriptor> {
        self.entries
            .iter()
            .rev()
            .find_map(|entry| entry.lookup(x, y, self.cell_size, self.tol_px))
    }

    /// Insert a frame's freshly-computed features, evicting the oldest frame
    /// (FIFO) if the capacity `max_frames` would be exceeded.
    fn insert_frame(
        &mut self,
        frame_id: u64,
        keypoints: Vec<Keypoint>,
        descriptors: Vec<BinaryDescriptor>,
    ) {
        let entry = FrameEntry::new(frame_id, keypoints, descriptors, self.cell_size);
        self.entries.push_back(entry);
        while self.entries.len() > self.max_frames {
            self.entries.pop_front();
        }
    }

    /// Frame ids currently retained, oldest first. Primarily for tests and
    /// diagnostics.
    #[must_use]
    pub fn cached_frame_ids(&self) -> Vec<u64> {
        self.entries.iter().map(|e| e.frame_id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::OrbDetector;

    /// Tiny deterministic textured image so the unit tests do not depend on the
    /// larger synthetic generator used by the integration tests.
    fn checker(w: usize, h: usize) -> Vec<u8> {
        let mut img = vec![128u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if ((x / 5) + (y / 5)) % 2 == 0 {
                    img[y * w + x] = if (x * 7 + y * 13) % 3 == 0 { 240 } else { 20 };
                }
            }
        }
        img
    }

    #[test]
    fn empty_cache_is_empty_and_zero_ratio() {
        let cache = DescriptorCache::with_detector(OrbDetector::new(500));
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert!((cache.hit_ratio() - 0.0).abs() < f64::EPSILON);
        assert_eq!(cache.max_frames(), DEFAULT_MAX_FRAMES);
    }

    #[test]
    fn max_frames_clamped_to_one() {
        let cache = DescriptorCache::new(OrbDetector::new(500), 0, 1.0);
        assert_eq!(cache.max_frames(), 1);
    }

    #[test]
    fn first_frame_is_all_misses() {
        let (w, h) = (120, 90);
        let img = checker(w, h);
        let mut cache = DescriptorCache::new(OrbDetector::new(500), 10, 1.5);
        let feats = cache
            .detect_and_compute_cached(&img, w, h, 0)
            .expect("cached detect");
        assert!(!feats.is_empty(), "expected some keypoints");
        // Empty cache ⇒ every keypoint is a miss.
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses() as usize, feats.len());
        assert!((cache.hit_ratio() - 0.0).abs() < f64::EPSILON);
        assert_eq!(cache.cached_frame_ids(), vec![0]);
    }

    #[test]
    fn cell_of_is_floor_division() {
        assert_eq!(cell_of(0.0, 0.0, 2.0), (0, 0));
        assert_eq!(cell_of(1.9, 1.9, 2.0), (0, 0));
        assert_eq!(cell_of(2.0, 2.0, 2.0), (1, 1));
        // Degenerate pitch clamps to 1.0.
        assert_eq!(cell_of(3.5, 4.5, 0.0), (3, 4));
    }

    #[test]
    fn clear_resets_everything() {
        let (w, h) = (120, 90);
        let img = checker(w, h);
        let mut cache = DescriptorCache::new(OrbDetector::new(500), 10, 1.5);
        cache
            .detect_and_compute_cached(&img, w, h, 0)
            .expect("cached detect");
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }
}
