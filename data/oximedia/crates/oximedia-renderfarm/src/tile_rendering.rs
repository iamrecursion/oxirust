// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Tile-based render distribution.
//!
//! Splits a frame into tiles and distributes them across farm workers,
//! then composites the results back into a complete frame.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A rectangular region within a frame, representing one render tile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileRegion {
    /// Left edge of the tile in pixels (inclusive).
    pub x: u32,
    /// Top edge of the tile in pixels (inclusive).
    pub y: u32,
    /// Width of the tile in pixels.
    pub width: u32,
    /// Height of the tile in pixels.
    pub height: u32,
    /// Unique tile identifier within this frame plan.
    pub tile_id: u32,
}

impl TileRegion {
    /// Create a new tile region.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32, tile_id: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            tile_id,
        }
    }

    /// Total number of pixels in this tile.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Aspect ratio (width / height).  Returns 0.0 for zero-height tiles.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f32 / self.height as f32
    }
}

/// Tile layout strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileLayout {
    /// Equal-size grid: rows × columns subdivisions.
    Grid,
    /// Spiral from the centre outward (best for interactive previews).
    Spiral,
    /// Centre-weighted: centre tiles are smaller to distribute CPU work fairly.
    LoadBalanced,
}

impl TileLayout {
    /// Human-readable description of the layout strategy.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Grid => "Equal-size grid tiles, row-major order",
            Self::Spiral => "Centre-outward spiral for fast progressive preview",
            Self::LoadBalanced => "Centre tiles smaller to balance render complexity",
        }
    }
}

/// Plans tile subdivision of a frame.
pub struct TilePlanner;

impl TilePlanner {
    /// Split a frame of `frame_w × frame_h` pixels into `tile_count` tiles
    /// using the given `layout` strategy.
    ///
    /// The returned list always covers the entire frame with no overlap.
    /// If `tile_count` is 0 or the frame has no pixels, an empty Vec is returned.
    #[must_use]
    pub fn plan(
        frame_w: u32,
        frame_h: u32,
        tile_count: u32,
        layout: TileLayout,
    ) -> Vec<TileRegion> {
        if frame_w == 0 || frame_h == 0 || tile_count == 0 {
            return vec![];
        }

        match layout {
            TileLayout::Grid => Self::plan_grid(frame_w, frame_h, tile_count),
            TileLayout::Spiral => Self::plan_spiral(frame_w, frame_h, tile_count),
            TileLayout::LoadBalanced => Self::plan_load_balanced(frame_w, frame_h, tile_count),
        }
    }

    /// Grid layout: as-square-as-possible subdivisions.
    fn plan_grid(frame_w: u32, frame_h: u32, tile_count: u32) -> Vec<TileRegion> {
        // Find cols × rows such that cols * rows >= tile_count and is as square as possible
        let cols = (tile_count as f32).sqrt().ceil() as u32;
        let rows = (tile_count as f32 / cols as f32).ceil() as u32;
        let tile_w = frame_w.div_ceil(cols);
        let tile_h = frame_h.div_ceil(rows);

        let mut tiles = Vec::new();
        let mut id = 0u32;
        for row in 0..rows {
            for col in 0..cols {
                let x = col * tile_w;
                let y = row * tile_h;
                if x >= frame_w || y >= frame_h {
                    continue;
                }
                let w = tile_w.min(frame_w - x);
                let h = tile_h.min(frame_h - y);
                tiles.push(TileRegion::new(x, y, w, h, id));
                id += 1;
            }
        }
        tiles
    }

    /// Spiral layout: tiles ordered from the centre outward.
    fn plan_spiral(frame_w: u32, frame_h: u32, tile_count: u32) -> Vec<TileRegion> {
        // First create a grid, then reorder by distance from centre
        let mut tiles = Self::plan_grid(frame_w, frame_h, tile_count);
        let cx = frame_w as f32 / 2.0;
        let cy = frame_h as f32 / 2.0;

        tiles.sort_by(|a, b| {
            let dist_a = {
                let ax = a.x as f32 + a.width as f32 / 2.0 - cx;
                let ay = a.y as f32 + a.height as f32 / 2.0 - cy;
                ax * ax + ay * ay
            };
            let dist_b = {
                let bx = b.x as f32 + b.width as f32 / 2.0 - cx;
                let by = b.y as f32 + b.height as f32 / 2.0 - cy;
                bx * bx + by * by
            };
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-assign IDs in spiral order
        for (i, tile) in tiles.iter_mut().enumerate() {
            tile.tile_id = i as u32;
        }
        tiles
    }

    /// Load-balanced layout: centre tiles are half the size to distribute complexity.
    fn plan_load_balanced(frame_w: u32, frame_h: u32, tile_count: u32) -> Vec<TileRegion> {
        // Simple approach: use grid but subdivide centre quadrant further
        let mut tiles = Self::plan_grid(frame_w, frame_h, tile_count);
        let cx = frame_w / 2;
        let cy = frame_h / 2;

        // Mark centre tiles and split them vertically
        let mut extra: Vec<TileRegion> = Vec::new();
        let mut id_counter = tiles.len() as u32;

        for tile in &mut tiles {
            let tile_cx = tile.x + tile.width / 2;
            let tile_cy = tile.y + tile.height / 2;
            let near_centre = (i64::from(tile_cx) - i64::from(cx)).unsigned_abs()
                < u64::from(frame_w / 4)
                && (i64::from(tile_cy) - i64::from(cy)).unsigned_abs() < u64::from(frame_h / 4);

            if near_centre && tile.height >= 2 {
                // Split into top and bottom halves
                let half_h = tile.height / 2;
                extra.push(TileRegion::new(
                    tile.x,
                    tile.y + half_h,
                    tile.width,
                    tile.height - half_h,
                    id_counter,
                ));
                id_counter += 1;
                tile.height = half_h;
            }
        }
        tiles.extend(extra);
        tiles
    }
}

/// A farm worker node available for tile assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    /// Unique worker identifier.
    pub id: String,
    /// Relative capacity score (higher = faster worker).
    pub capacity_score: f32,
    /// Number of tiles this worker has completed in this session.
    pub completed_tiles: u32,
}

impl WorkerNode {
    /// Create a new worker node.
    #[must_use]
    pub fn new(id: impl Into<String>, capacity_score: f32) -> Self {
        Self {
            id: id.into(),
            capacity_score,
            completed_tiles: 0,
        }
    }
}

/// Assigns tiles to workers proportionally to their capacity.
pub struct TileAssigner;

impl TileAssigner {
    /// Assign each tile to a worker, returning `(tile_id, worker_id)` pairs.
    ///
    /// Workers with higher `capacity_score` receive more tiles.
    /// If `workers` is empty, all tiles are assigned to "unassigned".
    #[must_use]
    pub fn assign(tiles: &[TileRegion], workers: &[WorkerNode]) -> Vec<(u32, String)> {
        if workers.is_empty() {
            return tiles
                .iter()
                .map(|t| (t.tile_id, "unassigned".to_string()))
                .collect();
        }

        // Normalise capacity scores
        let total_capacity: f32 = workers.iter().map(|w| w.capacity_score.max(0.001)).sum();
        let mut assignments = Vec::with_capacity(tiles.len());

        for (i, tile) in tiles.iter().enumerate() {
            // Round-robin weighted by capacity: pick worker whose cumulative
            // share of assignments is below their fair share.
            let worker_idx = i % workers.len();
            // Capacity-weighted selection
            let mut best_idx = worker_idx;
            let mut best_score = f32::NEG_INFINITY;
            for (wi, worker) in workers.iter().enumerate() {
                let share = worker.capacity_score.max(0.001) / total_capacity;
                let current_fraction = assignments
                    .iter()
                    .filter(|(_, wid): &&(u32, String)| *wid == worker.id)
                    .count() as f32
                    / (i + 1).max(1) as f32;
                let score = share - current_fraction;
                if score > best_score {
                    best_score = score;
                    best_idx = wi;
                }
            }
            assignments.push((tile.tile_id, workers[best_idx].id.clone()));
        }

        assignments
    }
}

/// Composites rendered tile pixel data back into a complete RGBA frame buffer.
pub struct TileCompositor;

impl TileCompositor {
    /// Merge an ordered list of `(TileRegion, pixel_data)` into a single RGBA frame.
    ///
    /// `pixel_data` must have exactly `tile.width * tile.height * 4` bytes (RGBA).
    /// Tiles that do not fit within the frame bounds are ignored.
    ///
    /// Returns a `frame_w × frame_h × 4` byte RGBA buffer.
    #[must_use]
    pub fn merge(tiles: &[(TileRegion, Vec<u8>)], frame_w: u32, frame_h: u32) -> Vec<u8> {
        let total = (frame_w as usize) * (frame_h as usize) * 4;
        let mut frame = vec![0u8; total];

        for (region, data) in tiles {
            let expected = (region.width as usize) * (region.height as usize) * 4;
            if data.len() != expected {
                continue; // skip malformed tiles
            }
            for row in 0..region.height {
                let dst_y = region.y + row;
                if dst_y >= frame_h {
                    break;
                }
                for col in 0..region.width {
                    let dst_x = region.x + col;
                    if dst_x >= frame_w {
                        break;
                    }
                    let src_offset = (row as usize * region.width as usize + col as usize) * 4;
                    let dst_offset = (dst_y as usize * frame_w as usize + dst_x as usize) * 4;
                    frame[dst_offset..dst_offset + 4]
                        .copy_from_slice(&data[src_offset..src_offset + 4]);
                }
            }
        }

        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_region_pixel_count() {
        let tile = TileRegion::new(0, 0, 100, 50, 0);
        assert_eq!(tile.pixel_count(), 5_000);
    }

    #[test]
    fn test_tile_region_aspect_ratio() {
        let tile = TileRegion::new(0, 0, 1920, 1080, 0);
        let ar = tile.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn test_tile_region_zero_height() {
        let tile = TileRegion::new(0, 0, 100, 0, 0);
        assert!((tile.aspect_ratio() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tile_layout_description() {
        assert!(!TileLayout::Grid.description().is_empty());
        assert!(!TileLayout::Spiral.description().is_empty());
        assert!(!TileLayout::LoadBalanced.description().is_empty());
    }

    #[test]
    fn test_plan_grid_covers_frame() {
        let tiles = TilePlanner::plan(1920, 1080, 9, TileLayout::Grid);
        let total_pixels: u64 = tiles.iter().map(super::TileRegion::pixel_count).sum();
        assert_eq!(total_pixels, 1920u64 * 1080);
    }

    #[test]
    fn test_plan_grid_zero_tiles() {
        let tiles = TilePlanner::plan(1920, 1080, 0, TileLayout::Grid);
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_plan_grid_zero_frame() {
        let tiles = TilePlanner::plan(0, 1080, 9, TileLayout::Grid);
        assert!(tiles.is_empty());
    }

    #[test]
    fn test_plan_spiral_count_matches_grid() {
        let grid = TilePlanner::plan(1920, 1080, 9, TileLayout::Grid);
        let spiral = TilePlanner::plan(1920, 1080, 9, TileLayout::Spiral);
        assert_eq!(grid.len(), spiral.len());
    }

    #[test]
    fn test_plan_spiral_centre_first() {
        let tiles = TilePlanner::plan(1920, 1080, 9, TileLayout::Spiral);
        // The first tile in spiral order should be closest to centre
        let cx = 1920.0 / 2.0;
        let cy = 1080.0 / 2.0;
        let first = &tiles[0];
        let first_dist = {
            let dx = first.x as f32 + first.width as f32 / 2.0 - cx;
            let dy = first.y as f32 + first.height as f32 / 2.0 - cy;
            dx * dx + dy * dy
        };
        for tile in &tiles[1..] {
            let dx = tile.x as f32 + tile.width as f32 / 2.0 - cx;
            let dy = tile.y as f32 + tile.height as f32 / 2.0 - cy;
            let dist = dx * dx + dy * dy;
            assert!(
                first_dist <= dist + 1.0,
                "First tile should be closest to centre"
            );
        }
    }

    #[test]
    fn test_assigner_basic() {
        let tiles = TilePlanner::plan(1920, 1080, 4, TileLayout::Grid);
        let workers = vec![WorkerNode::new("w1", 1.0), WorkerNode::new("w2", 1.0)];
        let assignments = TileAssigner::assign(&tiles, &workers);
        assert_eq!(assignments.len(), tiles.len());
    }

    #[test]
    fn test_assigner_empty_workers() {
        let tiles = TilePlanner::plan(640, 360, 4, TileLayout::Grid);
        let assignments = TileAssigner::assign(&tiles, &[]);
        for (_, worker_id) in &assignments {
            assert_eq!(worker_id, "unassigned");
        }
    }

    #[test]
    fn test_compositor_single_tile() {
        let tile = TileRegion::new(0, 0, 2, 2, 0);
        let data = vec![
            255u8, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ];
        let frame = TileCompositor::merge(&[(tile, data.clone())], 2, 2);
        assert_eq!(frame.len(), 2 * 2 * 4);
        assert_eq!(&frame[..], &data[..]);
    }

    #[test]
    fn test_compositor_two_tiles() {
        // 4×1 frame split into two 2×1 tiles
        let t1 = TileRegion::new(0, 0, 2, 1, 0);
        let t2 = TileRegion::new(2, 0, 2, 1, 1);
        let d1 = vec![255u8, 0, 0, 255, 0, 255, 0, 255];
        let d2 = vec![0u8, 0, 255, 255, 128, 128, 128, 255];
        let frame = TileCompositor::merge(&[(t1, d1.clone()), (t2, d2.clone())], 4, 1);
        assert_eq!(frame.len(), 4 * 4);
        // First pixel
        assert_eq!(&frame[0..4], &d1[0..4]);
        // Third pixel (offset 8)
        assert_eq!(&frame[8..12], &d2[0..4]);
    }
}
