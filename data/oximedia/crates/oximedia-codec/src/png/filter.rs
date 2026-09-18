//! PNG filter implementation.
//!
//! Implements the five PNG filter types as defined in the PNG specification:
//! - Filter 0: None
//! - Filter 1: Sub
//! - Filter 2: Up
//! - Filter 3: Average
//! - Filter 4: Paeth
//!
//! Also includes filter selection heuristics for optimal compression.

use crate::error::{CodecError, CodecResult};

/// PNG filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FilterType {
    /// No filtering.
    None = 0,
    /// Difference from left pixel (Sub).
    Sub = 1,
    /// Difference from above pixel (Up).
    Up = 2,
    /// Average of left and above pixels.
    Average = 3,
    /// Paeth predictor.
    Paeth = 4,
}

impl FilterType {
    /// Create filter type from byte.
    ///
    /// # Errors
    ///
    /// Returns error if filter type is invalid.
    pub fn from_u8(value: u8) -> CodecResult<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Up),
            3 => Ok(Self::Average),
            4 => Ok(Self::Paeth),
            _ => Err(CodecError::InvalidData(format!(
                "Invalid filter type: {value}"
            ))),
        }
    }

    /// Convert to byte value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Apply PNG filter to a scanline.
///
/// # Arguments
///
/// * `filter_type` - Type of filter to apply
/// * `scanline` - Current scanline data
/// * `prev_scanline` - Previous scanline data (for Up, Average, Paeth)
/// * `bytes_per_pixel` - Number of bytes per pixel
///
/// # Returns
///
/// Filtered scanline data.
#[allow(clippy::needless_pass_by_value)]
pub fn apply_filter(
    filter_type: FilterType,
    scanline: &[u8],
    prev_scanline: Option<&[u8]>,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    match filter_type {
        FilterType::None => scanline.to_vec(),
        FilterType::Sub => apply_sub_filter(scanline, bytes_per_pixel),
        FilterType::Up => apply_up_filter(scanline, prev_scanline.unwrap_or(&[])),
        FilterType::Average => {
            apply_average_filter(scanline, prev_scanline.unwrap_or(&[]), bytes_per_pixel)
        }
        FilterType::Paeth => {
            apply_paeth_filter(scanline, prev_scanline.unwrap_or(&[]), bytes_per_pixel)
        }
    }
}

/// Apply Sub filter.
fn apply_sub_filter(scanline: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut filtered = Vec::with_capacity(scanline.len());

    for i in 0..scanline.len() {
        let left = if i >= bytes_per_pixel {
            scanline[i - bytes_per_pixel]
        } else {
            0
        };
        filtered.push(scanline[i].wrapping_sub(left));
    }

    filtered
}

/// Apply Up filter.
fn apply_up_filter(scanline: &[u8], prev_scanline: &[u8]) -> Vec<u8> {
    let mut filtered = Vec::with_capacity(scanline.len());

    for i in 0..scanline.len() {
        let above = if i < prev_scanline.len() {
            prev_scanline[i]
        } else {
            0
        };
        filtered.push(scanline[i].wrapping_sub(above));
    }

    filtered
}

/// Apply Average filter.
fn apply_average_filter(scanline: &[u8], prev_scanline: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut filtered = Vec::with_capacity(scanline.len());

    for i in 0..scanline.len() {
        let left = if i >= bytes_per_pixel {
            scanline[i - bytes_per_pixel]
        } else {
            0
        };
        let above = if i < prev_scanline.len() {
            prev_scanline[i]
        } else {
            0
        };
        let avg = ((u16::from(left) + u16::from(above)) / 2) as u8;
        filtered.push(scanline[i].wrapping_sub(avg));
    }

    filtered
}

/// Apply Paeth filter.
fn apply_paeth_filter(scanline: &[u8], prev_scanline: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut filtered = Vec::with_capacity(scanline.len());

    for i in 0..scanline.len() {
        let left = if i >= bytes_per_pixel {
            scanline[i - bytes_per_pixel]
        } else {
            0
        };
        let above = if i < prev_scanline.len() {
            prev_scanline[i]
        } else {
            0
        };
        let upper_left = if i >= bytes_per_pixel && i < prev_scanline.len() {
            prev_scanline[i - bytes_per_pixel]
        } else {
            0
        };

        let paeth = paeth_predictor(left, above, upper_left);
        filtered.push(scanline[i].wrapping_sub(paeth));
    }

    filtered
}

/// Paeth predictor function.
///
/// Returns the value among a, b, c that is closest to p = a + b - c.
#[allow(clippy::cast_possible_wrap)]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = i32::from(a);
    let b = i32::from(b);
    let c = i32::from(c);

    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();

    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

/// Unfilter a scanline.
///
/// # Arguments
///
/// * `filter_type` - Type of filter applied
/// * `filtered` - Filtered scanline data
/// * `prev_scanline` - Previous scanline data (for Up, Average, Paeth)
/// * `bytes_per_pixel` - Number of bytes per pixel
///
/// # Errors
///
/// Returns error if unfiltering fails.
pub fn unfilter(
    filter_type: FilterType,
    filtered: &[u8],
    prev_scanline: Option<&[u8]>,
    bytes_per_pixel: usize,
) -> CodecResult<Vec<u8>> {
    match filter_type {
        FilterType::None => Ok(filtered.to_vec()),
        FilterType::Sub => Ok(unfilter_sub(filtered, bytes_per_pixel)),
        FilterType::Up => Ok(unfilter_up(filtered, prev_scanline.unwrap_or(&[]))),
        FilterType::Average => Ok(unfilter_average(
            filtered,
            prev_scanline.unwrap_or(&[]),
            bytes_per_pixel,
        )),
        FilterType::Paeth => Ok(unfilter_paeth(
            filtered,
            prev_scanline.unwrap_or(&[]),
            bytes_per_pixel,
        )),
    }
}

/// Unfilter Sub filter.
fn unfilter_sub(filtered: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut unfiltered = Vec::with_capacity(filtered.len());

    for i in 0..filtered.len() {
        let left = if i >= bytes_per_pixel {
            unfiltered[i - bytes_per_pixel]
        } else {
            0
        };
        unfiltered.push(filtered[i].wrapping_add(left));
    }

    unfiltered
}

/// Unfilter Up filter.
fn unfilter_up(filtered: &[u8], prev_scanline: &[u8]) -> Vec<u8> {
    let mut unfiltered = Vec::with_capacity(filtered.len());

    for i in 0..filtered.len() {
        let above = if i < prev_scanline.len() {
            prev_scanline[i]
        } else {
            0
        };
        unfiltered.push(filtered[i].wrapping_add(above));
    }

    unfiltered
}

/// Unfilter Average filter.
fn unfilter_average(filtered: &[u8], prev_scanline: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut unfiltered = Vec::with_capacity(filtered.len());

    for i in 0..filtered.len() {
        let left = if i >= bytes_per_pixel {
            unfiltered[i - bytes_per_pixel]
        } else {
            0
        };
        let above = if i < prev_scanline.len() {
            prev_scanline[i]
        } else {
            0
        };
        let avg = ((u16::from(left) + u16::from(above)) / 2) as u8;
        unfiltered.push(filtered[i].wrapping_add(avg));
    }

    unfiltered
}

/// Unfilter Paeth filter.
fn unfilter_paeth(filtered: &[u8], prev_scanline: &[u8], bytes_per_pixel: usize) -> Vec<u8> {
    let mut unfiltered = Vec::with_capacity(filtered.len());

    for i in 0..filtered.len() {
        let left = if i >= bytes_per_pixel {
            unfiltered[i - bytes_per_pixel]
        } else {
            0
        };
        let above = if i < prev_scanline.len() {
            prev_scanline[i]
        } else {
            0
        };
        let upper_left = if i >= bytes_per_pixel && i < prev_scanline.len() {
            prev_scanline[i - bytes_per_pixel]
        } else {
            0
        };

        let paeth = paeth_predictor(left, above, upper_left);
        unfiltered.push(filtered[i].wrapping_add(paeth));
    }

    unfiltered
}

/// Calculate sum of absolute differences for a filtered scanline.
///
/// Used for heuristic filter selection.
#[must_use]
pub fn sum_abs_diff(filtered: &[u8]) -> u64 {
    filtered.iter().map(|&b| u64::from(b.abs_diff(128))).sum()
}

/// Select best filter type for a scanline using heuristic evaluation.
///
/// Tries all filter types and selects the one with minimum sum of absolute differences.
///
/// # Arguments
///
/// * `scanline` - Current scanline data
/// * `prev_scanline` - Previous scanline data
/// * `bytes_per_pixel` - Number of bytes per pixel
///
/// # Returns
///
/// Tuple of (best filter type, filtered data).
#[must_use]
pub fn select_best_filter(
    scanline: &[u8],
    prev_scanline: Option<&[u8]>,
    bytes_per_pixel: usize,
) -> (FilterType, Vec<u8>) {
    let mut best_filter = FilterType::None;
    let mut best_data = scanline.to_vec();
    let mut best_score = sum_abs_diff(&best_data);

    let filters = [
        FilterType::None,
        FilterType::Sub,
        FilterType::Up,
        FilterType::Average,
        FilterType::Paeth,
    ];

    for &filter in &filters {
        let filtered = apply_filter(filter, scanline, prev_scanline, bytes_per_pixel);
        let score = sum_abs_diff(&filtered);

        if score < best_score {
            best_score = score;
            best_filter = filter;
            best_data = filtered;
        }
    }

    (best_filter, best_data)
}

/// Fast filter selection that only considers None, Sub, and Up filters.
///
/// This is faster than `select_best_filter` but may not achieve optimal compression.
#[must_use]
pub fn select_fast_filter(
    scanline: &[u8],
    prev_scanline: Option<&[u8]>,
    bytes_per_pixel: usize,
) -> (FilterType, Vec<u8>) {
    let mut best_filter = FilterType::None;
    let mut best_data = scanline.to_vec();
    let mut best_score = sum_abs_diff(&best_data);

    // Try Sub filter
    let sub_filtered = apply_sub_filter(scanline, bytes_per_pixel);
    let sub_score = sum_abs_diff(&sub_filtered);
    if sub_score < best_score {
        best_score = sub_score;
        best_filter = FilterType::Sub;
        best_data = sub_filtered;
    }

    // Try Up filter if previous scanline available
    if let Some(prev) = prev_scanline {
        let up_filtered = apply_up_filter(scanline, prev);
        let up_score = sum_abs_diff(&up_filtered);
        if up_score < best_score {
            best_filter = FilterType::Up;
            best_data = up_filtered;
        }
    }

    (best_filter, best_data)
}

/// Filter strategy for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrategy {
    /// No filtering (fastest).
    None,
    /// Use Sub filter only.
    Sub,
    /// Use Up filter only.
    Up,
    /// Use Average filter only.
    Average,
    /// Use Paeth filter only.
    Paeth,
    /// Fast heuristic selection (None, Sub, Up).
    Fast,
    /// Best compression (tries all filters).
    Best,
}

impl FilterStrategy {
    /// Apply filter strategy to a scanline.
    ///
    /// # Returns
    ///
    /// Tuple of (filter type, filtered data).
    #[must_use]
    pub fn apply(
        &self,
        scanline: &[u8],
        prev_scanline: Option<&[u8]>,
        bytes_per_pixel: usize,
    ) -> (FilterType, Vec<u8>) {
        match self {
            Self::None => (FilterType::None, scanline.to_vec()),
            Self::Sub => (FilterType::Sub, apply_sub_filter(scanline, bytes_per_pixel)),
            Self::Up => (
                FilterType::Up,
                apply_up_filter(scanline, prev_scanline.unwrap_or(&[])),
            ),
            Self::Average => (
                FilterType::Average,
                apply_average_filter(scanline, prev_scanline.unwrap_or(&[]), bytes_per_pixel),
            ),
            Self::Paeth => (
                FilterType::Paeth,
                apply_paeth_filter(scanline, prev_scanline.unwrap_or(&[]), bytes_per_pixel),
            ),
            Self::Fast => select_fast_filter(scanline, prev_scanline, bytes_per_pixel),
            Self::Best => select_best_filter(scanline, prev_scanline, bytes_per_pixel),
        }
    }
}

impl Default for FilterStrategy {
    fn default() -> Self {
        Self::Fast
    }
}
