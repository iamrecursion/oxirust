//! MPEG-2 inverse-scan tables (ISO/IEC 13818-2 §7.3 / Figures 7-2 and 7-3).
//!
//! Two scan orders exist:
//!
//! - **Progressive / zig-zag scan** (Figure 7-2), selected when
//!   `alternate_scan == 0`. Identical to the JPEG / MPEG-1 zig-zag.
//! - **Alternate scan** (Figure 7-3), selected when `alternate_scan == 1`
//!   (typically for interlaced content), which prioritises vertical frequency.
//!
//! Both tables map *scan index* → *raster position* (`row * 8 + col`). The
//! decoder reads run/level pairs in scan order and places each level at the
//! raster position given by the active scan.

/// Progressive (zig-zag) scan, ISO/IEC 13818-2 Figure 7-2.
///
/// `SCAN_PROGRESSIVE[scan_index] = raster_position`.
pub const SCAN_PROGRESSIVE: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Alternate scan, ISO/IEC 13818-2 Figure 7-3.
///
/// `SCAN_ALTERNATE[scan_index] = raster_position`.
pub const SCAN_ALTERNATE: [usize; 64] = [
    0, 8, 16, 24, 1, 9, 2, 10, 17, 25, 32, 40, 48, 56, 57, 49, 41, 33, 26, 18, 3, 11, 4, 12, 19,
    27, 34, 42, 50, 58, 35, 43, 51, 59, 20, 28, 5, 13, 6, 14, 21, 29, 36, 44, 52, 60, 37, 45, 53,
    61, 22, 30, 7, 15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63,
];

/// Select the active scan table given the `alternate_scan` flag.
#[must_use]
pub fn scan_table(alternate_scan: bool) -> &'static [usize; 64] {
    if alternate_scan {
        &SCAN_ALTERNATE
    } else {
        &SCAN_PROGRESSIVE
    }
}

/// Place a level at `scan_index` into the raster-ordered block, using the
/// supplied scan table. Out-of-range scan indices are ignored.
pub fn place_in_raster(block: &mut [i32; 64], scan: &[usize; 64], scan_index: usize, value: i32) {
    if scan_index < 64 {
        block[scan[scan_index]] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_permutation(table: &[usize; 64], name: &str) {
        let mut seen = [false; 64];
        for &raster in table {
            assert!(raster < 64, "{name}: raster index {raster} out of range");
            assert!(!seen[raster], "{name}: duplicate raster index {raster}");
            seen[raster] = true;
        }
        assert!(seen.iter().all(|s| *s), "{name}: not all positions covered");
    }

    #[test]
    fn progressive_is_permutation() {
        assert_permutation(&SCAN_PROGRESSIVE, "progressive");
    }

    #[test]
    fn alternate_is_permutation() {
        assert_permutation(&SCAN_ALTERNATE, "alternate");
    }

    #[test]
    fn dc_is_first_in_both() {
        assert_eq!(SCAN_PROGRESSIVE[0], 0);
        assert_eq!(SCAN_ALTERNATE[0], 0);
    }

    #[test]
    fn highest_is_last_in_both() {
        assert_eq!(SCAN_PROGRESSIVE[63], 63);
        assert_eq!(SCAN_ALTERNATE[63], 63);
    }

    #[test]
    fn scan_table_selects_correctly() {
        assert_eq!(scan_table(false)[1], SCAN_PROGRESSIVE[1]);
        assert_eq!(scan_table(true)[1], SCAN_ALTERNATE[1]);
    }

    #[test]
    fn place_in_raster_writes_at_scan_position() {
        let mut block = [0i32; 64];
        place_in_raster(&mut block, &SCAN_PROGRESSIVE, 1, 42);
        // Scan index 1 → raster position 1.
        assert_eq!(block[1], 42);
        place_in_raster(&mut block, &SCAN_ALTERNATE, 1, 7);
        // Alternate scan index 1 → raster position 8.
        assert_eq!(block[8], 7);
    }

    #[test]
    fn place_in_raster_ignores_out_of_range() {
        let mut block = [0i32; 64];
        place_in_raster(&mut block, &SCAN_PROGRESSIVE, 64, 99);
        assert!(block.iter().all(|&v| v == 0));
    }
}
