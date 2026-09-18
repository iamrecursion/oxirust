//! SIMD-optimized Hamming distance for binary descriptors.
//!
//! Uses `u64::count_ones()` for batch processing of Hamming distance
//! computations. On modern CPUs, `count_ones()` maps to a single
//! hardware `POPCNT` instruction, making this significantly faster
//! than byte-by-byte XOR + popcount.
//!
//! # Usage
//!
//! ```
//! use oximedia_align::simd_hamming::{hamming_u64_batch, batch_nearest_neighbor};
//! use oximedia_align::features::BinaryDescriptor;
//!
//! let desc1 = BinaryDescriptor::new([0xFF; 32]);
//! let desc2 = BinaryDescriptor::new([0x00; 32]);
//! assert_eq!(hamming_u64_batch(&desc1.data, &desc2.data), 256);
//! ```

use crate::features::{BinaryDescriptor, Keypoint, MatchPair};

/// Compute Hamming distance between two 256-bit descriptors using u64 batch
/// processing.
///
/// The 32-byte descriptor is reinterpreted as 4 × u64 values. Each pair is
/// XORed and the popcount is accumulated. This is 4× fewer loop iterations
/// than byte-by-byte processing and leverages the hardware POPCNT instruction.
#[must_use]
pub fn hamming_u64_batch(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut total = 0u32;

    // Process 8 bytes at a time as u64
    let mut offset = 0;
    while offset + 8 <= 32 {
        let a_u64 = u64::from_le_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let b_u64 = u64::from_le_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);

        total += (a_u64 ^ b_u64).count_ones();
        offset += 8;
    }

    total
}

/// Find the nearest neighbor for a query descriptor in a set of target
/// descriptors using the optimized Hamming distance.
///
/// Returns `(best_index, best_distance, second_best_distance)`.
///
/// Returns `None` if `targets` is empty.
#[must_use]
pub fn find_nearest_neighbor(
    query: &BinaryDescriptor,
    targets: &[BinaryDescriptor],
) -> Option<(usize, u32, u32)> {
    if targets.is_empty() {
        return None;
    }

    let mut best_dist = u32::MAX;
    let mut second_best_dist = u32::MAX;
    let mut best_idx = 0;

    for (j, target) in targets.iter().enumerate() {
        let dist = hamming_u64_batch(&query.data, &target.data);

        if dist < best_dist {
            second_best_dist = best_dist;
            best_dist = dist;
            best_idx = j;
        } else if dist < second_best_dist {
            second_best_dist = dist;
        }
    }

    Some((best_idx, best_dist, second_best_dist))
}

/// Batch nearest-neighbor matching using SIMD Hamming distance.
///
/// For each descriptor in `descs1`, finds the best match in `descs2` using
/// the ratio test (Lowe's ratio).
///
/// Returns matched pairs that pass both the distance threshold and ratio test.
#[must_use]
pub fn batch_nearest_neighbor(
    keypoints1: &[Keypoint],
    descs1: &[BinaryDescriptor],
    keypoints2: &[Keypoint],
    descs2: &[BinaryDescriptor],
    max_distance: u32,
    ratio_threshold: f32,
) -> Vec<MatchPair> {
    let mut matches = Vec::new();

    for (i, desc1) in descs1.iter().enumerate() {
        if let Some((best_idx, best_dist, second_best)) = find_nearest_neighbor(desc1, descs2) {
            if best_dist <= max_distance && second_best > 0 {
                let ratio = best_dist as f32 / second_best as f32;
                if ratio < ratio_threshold {
                    if i < keypoints1.len() && best_idx < keypoints2.len() {
                        matches.push(MatchPair::new(
                            i,
                            best_idx,
                            best_dist,
                            keypoints1[i].point,
                            keypoints2[best_idx].point,
                        ));
                    }
                }
            }
        }
    }

    matches
}

/// Compute Hamming distances between all pairs of two descriptor sets.
///
/// Returns a flat row-major matrix of distances with dimensions
/// `descs1.len() × descs2.len()`.
#[must_use]
pub fn pairwise_distances(descs1: &[BinaryDescriptor], descs2: &[BinaryDescriptor]) -> Vec<u32> {
    let n1 = descs1.len();
    let n2 = descs2.len();
    let mut distances = vec![0u32; n1 * n2];

    for (i, d1) in descs1.iter().enumerate() {
        for (j, d2) in descs2.iter().enumerate() {
            distances[i * n2 + j] = hamming_u64_batch(&d1.data, &d2.data);
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_identical() {
        let a = [0xAA_u8; 32];
        let b = [0xAA_u8; 32];
        assert_eq!(hamming_u64_batch(&a, &b), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let a = [0xFF_u8; 32];
        let b = [0x00_u8; 32];
        assert_eq!(hamming_u64_batch(&a, &b), 256);
    }

    #[test]
    fn test_hamming_known_value() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        a[0] = 0b1111_0000; // 4 ones
        b[0] = 0b0000_1111; // 4 ones, all different from a
        assert_eq!(hamming_u64_batch(&a, &b), 8); // 8 bits differ
    }

    #[test]
    fn test_hamming_single_bit() {
        let a = [0u8; 32];
        let mut b = [0u8; 32];
        b[0] = 1;
        assert_eq!(hamming_u64_batch(&a, &b), 1);
    }

    #[test]
    fn test_hamming_consistency_with_byte_method() {
        // Compare with the byte-by-byte method
        let a = BinaryDescriptor::new([0x5A; 32]);
        let b = BinaryDescriptor::new([0xA5; 32]);

        let byte_result = a.hamming_distance(&b);
        let u64_result = hamming_u64_batch(&a.data, &b.data);
        assert_eq!(byte_result, u64_result);
    }

    #[test]
    fn test_find_nearest_neighbor_empty() {
        let query = BinaryDescriptor::new([0; 32]);
        assert!(find_nearest_neighbor(&query, &[]).is_none());
    }

    #[test]
    fn test_find_nearest_neighbor_single() {
        let query = BinaryDescriptor::new([0; 32]);
        let targets = vec![BinaryDescriptor::new([0xFF; 32])];
        let result = find_nearest_neighbor(&query, &targets);
        assert!(result.is_some());
        let (idx, dist, _) = result.expect("should exist");
        assert_eq!(idx, 0);
        assert_eq!(dist, 256);
    }

    #[test]
    fn test_find_nearest_neighbor_best_match() {
        let query = BinaryDescriptor::new([0; 32]);
        let targets = vec![
            BinaryDescriptor::new([0xFF; 32]), // dist = 256
            BinaryDescriptor::new([0x01; 32]), // dist = 32 (1 bit per byte)
            BinaryDescriptor::new([0x0F; 32]), // dist = 128
        ];
        let (idx, dist, _) = find_nearest_neighbor(&query, &targets).expect("should exist");
        assert_eq!(idx, 1);
        assert_eq!(dist, 32);
    }

    #[test]
    fn test_batch_nearest_neighbor() {
        let kp1 = vec![
            Keypoint::new(10.0, 10.0, 1.0, 0.0, 1.0),
            Keypoint::new(20.0, 20.0, 1.0, 0.0, 1.0),
        ];
        let kp2 = vec![
            Keypoint::new(11.0, 11.0, 1.0, 0.0, 1.0),
            Keypoint::new(21.0, 21.0, 1.0, 0.0, 1.0),
        ];

        let d1 = vec![
            BinaryDescriptor::new([0; 32]),
            BinaryDescriptor::new([0xFF; 32]),
        ];
        let d2 = vec![
            BinaryDescriptor::new([0; 32]),    // matches d1[0]
            BinaryDescriptor::new([0xFF; 32]), // matches d1[1]
        ];

        let matches = batch_nearest_neighbor(&kp1, &d1, &kp2, &d2, 256, 0.9);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_batch_nearest_neighbor_with_threshold() {
        let kp1 = vec![Keypoint::new(10.0, 10.0, 1.0, 0.0, 1.0)];
        let kp2 = vec![
            Keypoint::new(11.0, 11.0, 1.0, 0.0, 1.0),
            Keypoint::new(12.0, 12.0, 1.0, 0.0, 1.0),
        ];

        let d1 = vec![BinaryDescriptor::new([0xFF; 32])];
        let d2 = vec![
            BinaryDescriptor::new([0x00; 32]), // dist = 256
            BinaryDescriptor::new([0x01; 32]), // dist = 224
        ];

        // With max_distance=10, nothing should match
        let matches = batch_nearest_neighbor(&kp1, &d1, &kp2, &d2, 10, 0.9);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_pairwise_distances() {
        let d1 = vec![
            BinaryDescriptor::new([0; 32]),
            BinaryDescriptor::new([0xFF; 32]),
        ];
        let d2 = vec![
            BinaryDescriptor::new([0; 32]),
            BinaryDescriptor::new([0xFF; 32]),
        ];

        let dists = pairwise_distances(&d1, &d2);
        assert_eq!(dists.len(), 4);
        assert_eq!(dists[0], 0); // [0] vs [0]
        assert_eq!(dists[1], 256); // [0] vs [FF]
        assert_eq!(dists[2], 256); // [FF] vs [0]
        assert_eq!(dists[3], 0); // [FF] vs [FF]
    }

    #[test]
    fn test_pairwise_distances_empty() {
        let d1: Vec<BinaryDescriptor> = vec![];
        let d2 = vec![BinaryDescriptor::new([0; 32])];
        let dists = pairwise_distances(&d1, &d2);
        assert!(dists.is_empty());
    }

    #[test]
    fn test_symmetry() {
        let a = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x01, 0x02, 0x03, 0x04,
            0x05, 0x06, 0x07, 0x08,
        ];
        let b = [
            0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99,
            0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB,
            0xFA, 0xF9, 0xF8, 0xF7,
        ];
        assert_eq!(hamming_u64_batch(&a, &b), hamming_u64_batch(&b, &a));
    }
}
