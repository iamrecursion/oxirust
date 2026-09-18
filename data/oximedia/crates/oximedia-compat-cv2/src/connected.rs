//! Connected-component labeling using two-pass union-find.
//!
//! Implements `cv2.connectedComponents` and `cv2.connectedComponentsWithStats`
//! equivalents operating on `CV_8UC1` binary `Mat`s.
//!
//! Background pixels (value 0) always receive label 0.
//! Foreground pixels (value > 0) receive labels 1 … N.

use crate::error::{Cv2Error, Cv2Result};
use crate::mat::{Mat, MatType};

// ── Union-Find ────────────────────────────────────────────────────────────────

/// Path-halving find with a flat parent array.
fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        // Path halving: two steps at once
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Union by setting the root of `a` to point to the root of `b`.
fn uf_union(parent: &mut [u32], a: u32, b: u32) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra as usize] = rb;
    }
}

// ── Core labeling pass ────────────────────────────────────────────────────────

/// Run the two-pass 8-connected union-find labeling algorithm.
///
/// Returns `(num_labels, labels_vec)` where:
/// - `num_labels` = number of foreground components + 1 (for background label 0)
/// - `labels_vec` has length `src.rows * src.cols`; background = 0, foreground = 1..N
///
/// The input `src` must be `CV_8UC1`.
fn label_image(src: &Mat) -> Cv2Result<(usize, Vec<i32>)> {
    if src.mat_type != MatType::CV_8UC1 {
        return Err(Cv2Error::UnsupportedDtype {
            mat_type: src.mat_type,
        });
    }

    let h = src.rows;
    let w = src.cols;
    let n = h * w;

    // binary[i] = true for foreground pixel
    let binary: Vec<bool> = src.data.iter().map(|&v| v > 0).collect();

    // parent array; index 0 is unused (background sentinel)
    let mut parent: Vec<u32> = (0..=(n as u32)).collect();
    let mut provisional: Vec<u32> = vec![0u32; n]; // provisional label per pixel
    let mut next_label = 1u32;

    // ── Pass 1: Assign provisional labels with union-find merging ─────────────
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if !binary[idx] {
                continue;
            }

            // Collect foreground 8-connected neighbours that were already labeled
            // (top-left, top, top-right, left)
            let mut neighbors: Vec<u32> = Vec::with_capacity(4);

            if y > 0 {
                // top
                let top = (y - 1) * w + x;
                if binary[top] && provisional[top] > 0 {
                    neighbors.push(provisional[top]);
                }
                // top-left
                if x > 0 {
                    let tl = (y - 1) * w + (x - 1);
                    if binary[tl] && provisional[tl] > 0 {
                        neighbors.push(provisional[tl]);
                    }
                }
                // top-right
                if x + 1 < w {
                    let tr = (y - 1) * w + (x + 1);
                    if binary[tr] && provisional[tr] > 0 {
                        neighbors.push(provisional[tr]);
                    }
                }
            }
            // left
            if x > 0 {
                let left = y * w + (x - 1);
                if binary[left] && provisional[left] > 0 {
                    neighbors.push(provisional[left]);
                }
            }

            if neighbors.is_empty() {
                // New label
                provisional[idx] = next_label;
                next_label += 1;
            } else {
                // Find minimum root among neighbors
                let min_root = neighbors
                    .iter()
                    .map(|&lbl| uf_find(&mut parent, lbl))
                    .min()
                    .unwrap_or(next_label); // safe: neighbors non-empty

                provisional[idx] = min_root;
                // Union all neighbor roots to min_root
                for &nb in &neighbors {
                    uf_union(&mut parent, nb, min_root);
                }
            }
        }
    }

    // ── Pass 2: Flatten union-find and assign compact contiguous IDs ──────────
    // Map root → compact label
    let mut root_to_label: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
    let mut component_count: i32 = 0;

    let mut labels = vec![0i32; n];
    for i in 0..n {
        if !binary[i] {
            // background stays 0
            continue;
        }
        let root = uf_find(&mut parent, provisional[i]);
        let lbl = root_to_label.entry(root).or_insert_with(|| {
            component_count += 1;
            component_count
        });
        labels[i] = *lbl;
    }

    // num_labels = foreground component count + 1 (for background)
    let num_labels = component_count as usize + 1;
    Ok((num_labels, labels))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Label connected regions in a binary `CV_8UC1` Mat using 8-connectivity.
///
/// Background pixels (value 0) receive label 0.  Foreground pixels (value > 0)
/// receive labels 1 … N in an unspecified but stable order.
///
/// Returns `(num_labels, labels)` where:
/// - `num_labels` = N + 1  (includes background label 0)
/// - `labels` = flat `Vec<i32>` with length `src.rows * src.cols`, row-major
///
/// Mirrors `cv2.connectedComponents(image)`.
pub fn connected_components(src: &Mat) -> Cv2Result<(usize, Vec<i32>)> {
    label_image(src)
}

/// Label connected regions and compute bounding-box statistics.
///
/// Returns `(num_labels, labels, stats, centroids)` where:
/// - `num_labels` = N + 1 (background is label 0)
/// - `labels` = flat `Vec<i32>` row-major, length `rows * cols`
/// - `stats[i]` = `[x, y, width, height, area]` for label `i`
/// - `centroids[i]` = `[cx, cy]` for label `i`
///
/// Label 0 (background) is included in `stats` and `centroids`.
///
/// Mirrors `cv2.connectedComponentsWithStats(image)`.
pub fn connected_components_with_stats(
    src: &Mat,
) -> Cv2Result<(usize, Vec<i32>, Vec<[i32; 5]>, Vec<[f64; 2]>)> {
    let (num_labels, labels) = label_image(src)?;
    let h = src.rows;
    let w = src.cols;

    // Accumulate stats per label
    let mut min_x = vec![i32::MAX; num_labels];
    let mut min_y = vec![i32::MAX; num_labels];
    let mut max_x = vec![i32::MIN; num_labels];
    let mut max_y = vec![i32::MIN; num_labels];
    let mut area = vec![0i32; num_labels];
    let mut sum_x = vec![0f64; num_labels];
    let mut sum_y = vec![0f64; num_labels];

    for row in 0..h {
        for col in 0..w {
            let lbl = labels[row * w + col] as usize;
            let x = col as i32;
            let y = row as i32;
            if min_x[lbl] > x {
                min_x[lbl] = x;
            }
            if min_y[lbl] > y {
                min_y[lbl] = y;
            }
            if max_x[lbl] < x {
                max_x[lbl] = x;
            }
            if max_y[lbl] < y {
                max_y[lbl] = y;
            }
            area[lbl] += 1;
            sum_x[lbl] += x as f64;
            sum_y[lbl] += y as f64;
        }
    }

    let stats: Vec<[i32; 5]> = (0..num_labels)
        .map(|lbl| {
            if area[lbl] == 0 {
                [0, 0, 0, 0, 0]
            } else {
                let bx = min_x[lbl];
                let by = min_y[lbl];
                let bw = max_x[lbl] - min_x[lbl] + 1;
                let bh = max_y[lbl] - min_y[lbl] + 1;
                [bx, by, bw, bh, area[lbl]]
            }
        })
        .collect();

    let centroids: Vec<[f64; 2]> = (0..num_labels)
        .map(|lbl| {
            if area[lbl] == 0 {
                [0.0, 0.0]
            } else {
                let n = area[lbl] as f64;
                [sum_x[lbl] / n, sum_y[lbl] / n]
            }
        })
        .collect();

    Ok((num_labels, labels, stats, centroids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat::Mat;

    fn make_binary(data: Vec<u8>, rows: usize, cols: usize) -> Mat {
        Mat::from_gray_bytes(data, rows, cols)
    }

    #[test]
    fn test_all_background() {
        let src = make_binary(vec![0u8; 4 * 4], 4, 4);
        let (num_labels, labels) = connected_components(&src).unwrap();
        // Only background = 1 total label
        assert_eq!(num_labels, 1);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_all_foreground() {
        let src = make_binary(vec![255u8; 4 * 4], 4, 4);
        let (num_labels, labels) = connected_components(&src).unwrap();
        // All connected → 2 labels (background + 1 foreground)
        assert_eq!(num_labels, 2);
        assert!(labels.iter().all(|&l| l == 1));
    }

    #[test]
    fn test_two_separate_blobs() {
        // 1×6 row: [255, 255, 0, 0, 255, 255]
        let data = vec![255u8, 255, 0, 0, 255, 255];
        let src = make_binary(data, 1, 6);
        let (num_labels, labels) = connected_components(&src).unwrap();
        assert_eq!(num_labels, 3, "background + 2 blobs = 3");
        assert_ne!(labels[0], labels[4], "blobs have different labels");
        assert_eq!(labels[0], labels[1], "first blob pixels same label");
        assert_eq!(labels[4], labels[5], "second blob pixels same label");
        assert_eq!(labels[2], 0, "gap is background");
    }

    #[test]
    fn test_three_blobs_stats() {
        // 3×9 image: three 3×1 columns separated by gaps
        let mut data = vec![0u8; 3 * 9];
        for r in 0..3 {
            data[r * 9] = 255; // col 0
            data[r * 9 + 4] = 255; // col 4
            data[r * 9 + 8] = 255; // col 8
        }
        let src = make_binary(data, 3, 9);
        let (num_labels, _labels, stats, centroids) =
            connected_components_with_stats(&src).unwrap();
        assert_eq!(num_labels, 4, "background + 3 blobs = 4");
        // Each blob has area = 3
        let total_fg_area: i32 = stats[1..].iter().map(|s| s[4]).sum();
        assert_eq!(total_fg_area, 9);
        // Centroid row of each blob should be ~1.0 (middle of 3 rows)
        for ci in 1..num_labels {
            assert!(
                (centroids[ci][1] - 1.0).abs() < 0.01,
                "centroid y should be 1.0"
            );
        }
    }
}
