// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Tile-based rendering support.
//!
//! Provides tile decomposition, assignment tracking, reassembly ordering, and
//! configurable border overlap (guard-band) for seamless multi-worker renders.
//!
//! This module is distinct from [`super::tile_rendering`] which handles
//! distribution logistics; this module focuses on the geometry and lifecycle.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Geometry primitives
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-D rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Pixel area of the rectangle.
    #[must_use]
    pub const fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Expand the rectangle outward by `border` pixels on every side, clamped
    /// to `(0, 0, frame_w, frame_h)`.
    #[must_use]
    pub fn expand(&self, border: u32, frame_w: u32, frame_h: u32) -> Self {
        let x = self.x.saturating_sub(border);
        let y = self.y.saturating_sub(border);
        let x2 = (self.x + self.w + border).min(frame_w);
        let y2 = (self.y + self.h + border).min(frame_h);
        Self::new(x, y, x2 - x, y2 - y)
    }

    /// Returns `true` if this rectangle contains the point `(px, py)`.
    #[must_use]
    pub const fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for splitting a frame into tiles.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Desired tile width (tiles at the right edge may be narrower).
    pub tile_width: u32,
    /// Desired tile height (tiles at the bottom edge may be shorter).
    pub tile_height: u32,
    /// Border overlap (guard-band) in pixels.
    pub border_overlap: u32,
}

impl TileConfig {
    /// Create a tile configuration with no border overlap.
    #[must_use]
    pub const fn new(
        frame_width: u32,
        frame_height: u32,
        tile_width: u32,
        tile_height: u32,
    ) -> Self {
        Self {
            frame_width,
            frame_height,
            tile_width,
            tile_height,
            border_overlap: 0,
        }
    }

    /// Set the border overlap.
    #[must_use]
    pub const fn with_border(mut self, border: u32) -> Self {
        self.border_overlap = border;
        self
    }

    /// Number of tile columns.
    #[must_use]
    pub fn columns(&self) -> u32 {
        self.frame_width.div_ceil(self.tile_width)
    }

    /// Number of tile rows.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.frame_height.div_ceil(self.tile_height)
    }

    /// Total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        self.columns() * self.rows()
    }
}

/// Decompose a frame into tiles according to `config`.
///
/// Returns a vector of `(tile_id, render_rect, crop_rect)` where:
/// - `render_rect` is the area that should be rendered (including border overlap)
/// - `crop_rect`  is the final contribution area within the full frame
#[must_use]
pub fn decompose(config: &TileConfig) -> Vec<(u32, Rect, Rect)> {
    let cols = config.columns();
    let rows = config.rows();
    let mut tiles = Vec::with_capacity((cols * rows) as usize);

    for row in 0..rows {
        for col in 0..cols {
            let tile_id = row * cols + col;

            let x = col * config.tile_width;
            let y = row * config.tile_height;
            let w = config.tile_width.min(config.frame_width - x);
            let h = config.tile_height.min(config.frame_height - y);

            let crop_rect = Rect::new(x, y, w, h);
            let render_rect = crop_rect.expand(
                config.border_overlap,
                config.frame_width,
                config.frame_height,
            );

            tiles.push((tile_id, render_rect, crop_rect));
        }
    }

    tiles
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile assignment and lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/// State of a single tile within a render job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileStatus {
    /// Waiting to be assigned.
    Pending,
    /// Assigned to a worker (stores worker ID).
    Assigned(String),
    /// Successfully rendered.
    Done,
    /// Failed (stores error message).
    Failed(String),
}

/// Assignment record for a tile.
#[derive(Debug, Clone)]
pub struct TileAssignment {
    /// Zero-based tile identifier.
    pub tile_id: u32,
    /// Render region (with overlap).
    pub render_rect: Rect,
    /// Crop region (final contribution).
    pub crop_rect: Rect,
    /// Current tile status.
    pub status: TileStatus,
    /// Number of times this tile has been attempted.
    pub attempt_count: u32,
}

impl TileAssignment {
    /// Create a pending tile assignment.
    #[must_use]
    pub fn new(tile_id: u32, render_rect: Rect, crop_rect: Rect) -> Self {
        Self {
            tile_id,
            render_rect,
            crop_rect,
            status: TileStatus::Pending,
            attempt_count: 0,
        }
    }

    /// Assign this tile to a worker.
    pub fn assign(&mut self, worker_id: impl Into<String>) {
        self.attempt_count += 1;
        self.status = TileStatus::Assigned(worker_id.into());
    }

    /// Mark this tile as successfully rendered.
    pub fn complete(&mut self) {
        self.status = TileStatus::Done;
    }

    /// Mark this tile as failed.
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = TileStatus::Failed(reason.into());
    }

    /// Returns `true` if the tile has been rendered successfully.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.status == TileStatus::Done
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile job manager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the full set of tile assignments for one render frame.
#[derive(Debug)]
pub struct TileJob {
    /// Frame configuration.
    pub config: TileConfig,
    /// Map from `tile_id` → assignment.
    assignments: HashMap<u32, TileAssignment>,
}

impl TileJob {
    /// Create a new tile job by decomposing the frame.
    #[must_use]
    pub fn new(config: TileConfig) -> Self {
        let tiles = decompose(&config);
        let assignments = tiles
            .into_iter()
            .map(|(id, render, crop)| (id, TileAssignment::new(id, render, crop)))
            .collect();
        Self {
            config,
            assignments,
        }
    }

    /// Total number of tiles in this job.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.assignments.len()
    }

    /// Return all pending tile IDs.
    #[must_use]
    pub fn pending_tiles(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .assignments
            .values()
            .filter(|a| a.status == TileStatus::Pending)
            .map(|a| a.tile_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Assign the next pending tile to a worker.
    ///
    /// Returns the tile ID assigned, or `None` if no pending tiles remain.
    pub fn assign_next(&mut self, worker_id: impl Into<String>) -> Option<u32> {
        let id = {
            let pending = self
                .assignments
                .values()
                .filter(|a| a.status == TileStatus::Pending)
                .map(|a| a.tile_id)
                .min()?;
            pending
        };
        self.assignments.get_mut(&id)?.assign(worker_id);
        Some(id)
    }

    /// Mark a tile as complete.
    ///
    /// Returns `false` if the tile ID is unknown.
    pub fn complete_tile(&mut self, tile_id: u32) -> bool {
        if let Some(a) = self.assignments.get_mut(&tile_id) {
            a.complete();
            true
        } else {
            false
        }
    }

    /// Mark a tile as failed.
    pub fn fail_tile(&mut self, tile_id: u32, reason: impl Into<String>) -> bool {
        if let Some(a) = self.assignments.get_mut(&tile_id) {
            a.fail(reason);
            true
        } else {
            false
        }
    }

    /// Re-queue a failed tile by resetting its status to `Pending`.
    pub fn retry_tile(&mut self, tile_id: u32) -> bool {
        if let Some(a) = self.assignments.get_mut(&tile_id) {
            if matches!(a.status, TileStatus::Failed(_)) {
                a.status = TileStatus::Pending;
                return true;
            }
        }
        false
    }

    /// Returns `true` when every tile has been rendered successfully.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.assignments.values().all(TileAssignment::is_done)
    }

    /// Progress ratio: `done / total` in `[0.0, 1.0]`.
    #[must_use]
    pub fn progress(&self) -> f32 {
        let done = self.assignments.values().filter(|a| a.is_done()).count();
        done as f32 / self.assignments.len() as f32
    }

    /// Return tiles in row-major order for reassembly.
    ///
    /// Only returns tiles that are `Done`.
    #[must_use]
    pub fn reassembly_order(&self) -> Vec<&TileAssignment> {
        let cols = self.config.columns();
        let mut done: Vec<&TileAssignment> =
            self.assignments.values().filter(|a| a.is_done()).collect();
        done.sort_by_key(|a| {
            let row = a.tile_id / cols;
            let col = a.tile_id % cols;
            (row, col)
        });
        done
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TileCompositor — parallel 8K tile merge
// ─────────────────────────────────────────────────────────────────────────────

use rayon::prelude::*;

/// A rendered tile's pixel data ready for compositing into a frame buffer.
pub struct TilePixels {
    /// Tile geometry within the full frame (pixel coordinates).
    pub rect: Rect,
    /// Raw pixel bytes: `bytes_per_pixel` bytes per pixel, row-major order.
    pub data: Vec<u8>,
}

/// Composites multiple rendered tiles into a single flat frame buffer.
///
/// The `merge` method uses rayon for parallelism. Tiles are required to be
/// non-overlapping; each tile's rows are written to disjoint byte ranges.
///
/// # Parallelism strategy
///
/// Because `#![forbid(unsafe_code)]` is in effect for this crate, we cannot
/// use raw pointer writes across threads. Instead, we group tiles by row band
/// and process each band (set of tiles sharing the same vertical span) in
/// parallel by giving each tile exclusive ownership of its pixel columns via
/// `chunks_mut` / `split_at_mut` decomposition.
///
/// For tiles that share the same row range, we perform column-level splits
/// using a per-row serial copy within each parallel rayon task. This remains
/// safe and avoids aliasing.
pub struct TileCompositor {
    /// Full frame width in pixels.
    pub frame_width: u32,
    /// Full frame height in pixels.
    pub frame_height: u32,
    /// Bytes per pixel (e.g. 3 for RGB, 4 for RGBA).
    pub bytes_per_pixel: u32,
}

impl TileCompositor {
    /// Create a new compositor for the given frame dimensions.
    pub fn new(frame_width: u32, frame_height: u32, bytes_per_pixel: u32) -> Self {
        Self {
            frame_width,
            frame_height,
            bytes_per_pixel,
        }
    }

    /// Merge all tiles into `output`, resizing it to `frame_width × frame_height × bpp`.
    ///
    /// Returns `Err(String)` if any tile extends beyond the frame boundaries.
    ///
    /// # Parallelism
    ///
    /// Tiles are processed in parallel using rayon. Each tile owns disjoint
    /// rows, so we partition the output buffer into per-row-band slices and
    /// hand each tile its exclusive slice via `split_at_mut`.  Within a single
    /// row band that holds multiple tiles at different column offsets we copy
    /// sequentially (the common case is that tiles occupy non-overlapping
    /// column ranges of the same row).
    pub fn merge(&self, tiles: &[TilePixels], output: &mut Vec<u8>) -> Result<(), String> {
        let stride = self.frame_width as usize * self.bytes_per_pixel as usize;
        let total = stride * self.frame_height as usize;
        output.resize(total, 0u8);

        // Validate all tiles before touching output.
        for tile in tiles {
            let x_end = tile
                .rect
                .x
                .checked_add(tile.rect.w)
                .ok_or_else(|| format!("tile x overflow: {:?}", tile.rect))?;
            let y_end = tile
                .rect
                .y
                .checked_add(tile.rect.h)
                .ok_or_else(|| format!("tile y overflow: {:?}", tile.rect))?;
            if x_end > self.frame_width || y_end > self.frame_height {
                return Err(format!(
                    "tile ({},{} {}x{}) exceeds frame {}x{}",
                    tile.rect.x,
                    tile.rect.y,
                    tile.rect.w,
                    tile.rect.h,
                    self.frame_width,
                    self.frame_height
                ));
            }
        }

        // Build row-work items: (dst_row_byte_offset, x_byte_start, row_data)
        // grouped by destination row. Rayon parallelises over individual
        // tile-rows; since tiles are non-overlapping every item writes to a
        // distinct byte range.
        let bpp = self.bytes_per_pixel as usize;

        // Collect all (dst_byte_start, src_row_slice) pairs.
        let row_copies: Vec<(usize, &[u8])> = tiles
            .iter()
            .flat_map(|tile| {
                let tile_stride = tile.rect.w as usize * bpp;
                (0..tile.rect.h as usize).map(move |row| {
                    let src_start = row * tile_stride;
                    let src_end = src_start + tile_stride;
                    let dst_row = (tile.rect.y as usize + row) * stride;
                    let dst_start = dst_row + tile.rect.x as usize * bpp;
                    (dst_start, &tile.data[src_start..src_end])
                })
            })
            .collect();

        // We need mutable access to disjoint regions of `output` across
        // parallel threads.  Safe Rust requires us to express the
        // non-aliasing.  We use chunks_mut over individual output rows, then
        // for each row copy the tile data into the correct column offset.
        //
        // Index the row_copies by destination row for efficient lookup.
        let frame_height = self.frame_height as usize;
        let mut per_row: Vec<Vec<(usize, &[u8])>> = vec![Vec::new(); frame_height];
        for (dst_byte_start, src) in &row_copies {
            let row_idx = dst_byte_start / stride;
            let col_byte = dst_byte_start % stride;
            per_row[row_idx].push((col_byte, src));
        }

        // Parallel over rows: each row owns a disjoint `&mut [u8]` of `output`.
        output
            .par_chunks_mut(stride)
            .zip(per_row.into_par_iter())
            .for_each(|(row_buf, copies)| {
                for (col_byte, src) in copies {
                    row_buf[col_byte..col_byte + src.len()].copy_from_slice(src);
                }
            });

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0, 0, 100, 50);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_rect_contains() {
        let r = Rect::new(10, 10, 20, 20);
        assert!(r.contains(10, 10));
        assert!(r.contains(29, 29));
        assert!(!r.contains(30, 30));
        assert!(!r.contains(9, 10));
    }

    #[test]
    fn test_rect_expand_within_frame() {
        let r = Rect::new(10, 10, 100, 100);
        let expanded = r.expand(5, 1920, 1080);
        assert_eq!(expanded.x, 5);
        assert_eq!(expanded.y, 5);
        assert_eq!(expanded.w, 110);
        assert_eq!(expanded.h, 110);
    }

    #[test]
    fn test_rect_expand_clamps_to_frame() {
        let r = Rect::new(0, 0, 1920, 1080);
        let expanded = r.expand(32, 1920, 1080);
        assert_eq!(expanded.x, 0);
        assert_eq!(expanded.y, 0);
        assert_eq!(expanded.w, 1920);
        assert_eq!(expanded.h, 1080);
    }

    #[test]
    fn test_tile_config_count() {
        let cfg = TileConfig::new(1920, 1080, 256, 256);
        // ceil(1920/256)=8, ceil(1080/256)=5 → 40
        assert_eq!(cfg.columns(), 8);
        assert_eq!(cfg.rows(), 5);
        assert_eq!(cfg.tile_count(), 40);
    }

    #[test]
    fn test_tile_config_exact_division() {
        let cfg = TileConfig::new(800, 600, 100, 100);
        assert_eq!(cfg.columns(), 8);
        assert_eq!(cfg.rows(), 6);
        assert_eq!(cfg.tile_count(), 48);
    }

    #[test]
    fn test_decompose_no_overlap() {
        let cfg = TileConfig::new(400, 200, 100, 100);
        let tiles = decompose(&cfg);
        assert_eq!(tiles.len(), 8);
        // All crop areas should cover the full frame with no gaps
        let total_area: u64 = tiles.iter().map(|(_, _, crop)| crop.area()).sum();
        assert_eq!(total_area, 400 * 200);
    }

    #[test]
    fn test_decompose_with_overlap_render_rect_larger() {
        let cfg = TileConfig::new(400, 200, 200, 200).with_border(16);
        let tiles = decompose(&cfg);
        for (_, render, crop) in &tiles {
            assert!(render.w >= crop.w);
            assert!(render.h >= crop.h);
        }
    }

    #[test]
    fn test_tile_job_assign_and_complete() {
        let cfg = TileConfig::new(200, 200, 100, 100);
        let mut job = TileJob::new(cfg);
        assert_eq!(job.tile_count(), 4);
        let id = job.assign_next("worker-1").expect("should succeed in test");
        assert!(job.complete_tile(id));
        assert_eq!(job.progress(), 0.25);
    }

    #[test]
    fn test_tile_job_all_complete() {
        let cfg = TileConfig::new(200, 100, 100, 100);
        let mut job = TileJob::new(cfg);
        while let Some(id) = job.assign_next("w") {
            job.complete_tile(id);
        }
        assert!(job.is_complete());
        assert!((job.progress() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tile_job_fail_and_retry() {
        let cfg = TileConfig::new(200, 100, 100, 100);
        let mut job = TileJob::new(cfg);
        let id = job.assign_next("w1").expect("should succeed in test");
        job.fail_tile(id, "GPU crash");
        assert!(matches!(job.assignments[&id].status, TileStatus::Failed(_)));
        assert!(job.retry_tile(id));
        assert_eq!(job.assignments[&id].status, TileStatus::Pending);
    }

    #[test]
    fn test_tile_job_reassembly_order() {
        let cfg = TileConfig::new(200, 200, 100, 100);
        let mut job = TileJob::new(cfg);
        // Complete tiles out of order: 1, 3, 0, 2
        for id in [1u32, 3, 0, 2] {
            if let Some(a) = job.assignments.get_mut(&id) {
                a.status = TileStatus::Done;
            }
        }
        let order: Vec<u32> = job.reassembly_order().iter().map(|a| a.tile_id).collect();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_tile_assignment_attempt_count() {
        let mut a = TileAssignment::new(0, Rect::new(0, 0, 100, 100), Rect::new(0, 0, 100, 100));
        a.assign("w1");
        a.fail("timeout");
        a.status = TileStatus::Pending;
        a.assign("w2");
        assert_eq!(a.attempt_count, 2);
    }

    #[test]
    fn test_tile_job_pending_tiles_sorted() {
        let cfg = TileConfig::new(300, 100, 100, 100);
        let mut job = TileJob::new(cfg);
        // Mark tile 1 as done to leave 0 and 2 pending
        job.assign_next("w").expect("should succeed in test"); // assigns tile 0
        job.complete_tile(0);
        job.assign_next("w").expect("should succeed in test"); // assigns tile 1
                                                               // tile 2 is still pending
        let pending = job.pending_tiles();
        assert_eq!(pending, vec![2]);
    }

    // --- TileCompositor ---

    #[test]
    fn test_tile_compositor_basic_merge() {
        let compositor = TileCompositor::new(4, 4, 1);
        // Two 2x4 tiles side by side (left half / right half)
        let left = TilePixels {
            rect: Rect::new(0, 0, 2, 4),
            data: vec![1u8; 8],
        };
        let right = TilePixels {
            rect: Rect::new(2, 0, 2, 4),
            data: vec![2u8; 8],
        };
        let mut output = Vec::new();
        compositor
            .merge(&[left, right], &mut output)
            .expect("merge should succeed");
        assert_eq!(output.len(), 16);
        // Left half pixels = 1, right half pixels = 2 in each row
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 1);
        assert_eq!(output[2], 2);
        assert_eq!(output[3], 2);
    }

    #[test]
    fn test_tile_compositor_out_of_bounds_error() {
        let compositor = TileCompositor::new(100, 100, 4);
        let bad = TilePixels {
            rect: Rect::new(90, 90, 20, 20), // x_end = 110 > 100
            data: vec![0u8; 20 * 20 * 4],
        };
        let mut output = Vec::new();
        assert!(compositor.merge(&[bad], &mut output).is_err());
    }

    #[test]
    fn test_tile_compositor_single_tile() {
        let compositor = TileCompositor::new(2, 2, 3);
        let tile = TilePixels {
            rect: Rect::new(0, 0, 2, 2),
            data: vec![255u8; 12],
        };
        let mut output = Vec::new();
        compositor
            .merge(&[tile], &mut output)
            .expect("merge should succeed");
        assert_eq!(output, vec![255u8; 12]);
    }

    #[test]
    fn test_tile_compositor_empty_tiles() {
        let compositor = TileCompositor::new(4, 4, 1);
        let mut output = Vec::new();
        compositor.merge(&[], &mut output).expect("empty merge ok");
        assert_eq!(output.len(), 16);
        assert!(output.iter().all(|&b| b == 0));
    }
}
