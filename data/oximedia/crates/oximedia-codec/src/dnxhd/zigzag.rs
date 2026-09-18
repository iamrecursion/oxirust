//! MPEG-2 progressive zigzag scan table for DNxHD coefficient ordering.
//!
//! DNxHD (VC-3 / SMPTE ST 2019-1) uses the standard MPEG-2 progressive
//! zigzag scan for 8×8 DCT block coefficient ordering. This table is
//! identical to the `PROGRESSIVE_ZIGZAG` used by ProRes and JPEG.

/// MPEG-2 progressive zigzag scan: `ZIGZAG_SCAN[scan_index] = raster_position`.
///
/// Coefficient `k` (0..64) is stored at raster position `ZIGZAG_SCAN[k]`
/// in the 8×8 block (where raster position = `row * 8 + col`).
pub const ZIGZAG_SCAN: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse-scan a 64-entry coefficient array from zigzag order to raster order.
///
/// `coeffs_in_scan_order[k]` is placed at raster position `ZIGZAG_SCAN[k]`.
#[must_use]
pub fn inverse_zigzag(coeffs: &[i32; 64]) -> [i32; 64] {
    let mut raster = [0i32; 64];
    for (scan_idx, &raster_idx) in ZIGZAG_SCAN.iter().enumerate() {
        raster[raster_idx] = coeffs[scan_idx];
    }
    raster
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_scan_is_a_permutation() {
        let mut seen = [false; 64];
        for &raster in &ZIGZAG_SCAN {
            assert!(raster < 64, "raster index {raster} out of range");
            assert!(!seen[raster], "duplicate raster index {raster}");
            seen[raster] = true;
        }
        assert!(seen.iter().all(|s| *s), "not all raster positions covered");
    }

    #[test]
    fn dc_at_index_zero() {
        // DC coefficient (raster position 0) is first in scan order.
        assert_eq!(ZIGZAG_SCAN[0], 0);
    }

    #[test]
    fn highest_frequency_at_end() {
        // Bottom-right of 8×8 (raster 63) is last in progressive scan.
        assert_eq!(ZIGZAG_SCAN[63], 63);
    }

    #[test]
    fn inverse_zigzag_round_trip() {
        // Build a raster array where position i has value i.
        // Forward scan: scan_buf[scan_idx] = raster[ZIGZAG_SCAN[scan_idx]].
        let raster: [i32; 64] = std::array::from_fn(|i| i as i32);
        let mut scan_buf = [0i32; 64];
        for (scan_idx, &raster_idx) in ZIGZAG_SCAN.iter().enumerate() {
            scan_buf[scan_idx] = raster[raster_idx];
        }
        // Inverse back.
        let recovered = inverse_zigzag(&scan_buf);
        assert_eq!(recovered, raster);
    }
}
