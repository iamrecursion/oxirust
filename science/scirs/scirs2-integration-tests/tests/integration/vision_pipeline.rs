// Integration tests for scirs2-ndimage computer vision pipeline
// Tests: image filtering → edge detection, morphological operations, histogram analysis

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::Array2;
use scirs2_ndimage::BorderMode;
use scirs2_ndimage::{binary_dilation, binary_erosion, gaussian_filter, histogram, sobel};

use crate::common::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Build a 32×32 synthetic grayscale image (values in [0.0, ~1.0]).
/// Contains a bright 10×10 square in the center plus a subtle background gradient.
fn synthetic_image_32() -> Array2<f64> {
    let mut img = Array2::<f64>::zeros((32, 32));
    // Bright 10×10 square: rows 11..21, cols 11..21
    for r in 11..21 {
        for c in 11..21 {
            img[[r, c]] = 0.9;
        }
    }
    // Faint background gradient
    for r in 0..32_usize {
        for c in 0..32_usize {
            img[[r, c]] += (r as f64 + c as f64) / 128.0 * 0.1;
        }
    }
    img
}

/// Build a 16×16 binary image with a 5×5 foreground block.
fn binary_image_16() -> Array2<bool> {
    let mut img = Array2::<bool>::from_elem((16, 16), false);
    // Foreground block: rows 5..10, cols 5..10
    for r in 5..10 {
        for c in 5..10 {
            img[[r, c]] = true;
        }
    }
    img
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Gaussian smoothing → Sobel edge detection
// ─────────────────────────────────────────────────────────────────────────────

/// End-to-end test: Gaussian blur followed by Sobel edge detection.
/// Verifies output shape matches input and edge magnitudes are reasonable.
#[test]
fn test_vision_pipeline_basic_image_ops() -> TestResult<()> {
    let image = synthetic_image_32();
    let (rows, cols) = (image.nrows(), image.ncols());

    // 1. Gaussian smoothing — sigma = 1.5
    let smoothed = gaussian_filter(&image, 1.5, None, None)
        .map_err(|e| format!("gaussian_filter failed: {}", e))?;

    assert_eq!(
        smoothed.shape(),
        &[rows, cols],
        "Smoothed image shape mismatch"
    );

    // Values must remain in expected range after smoothing
    let max_val = smoothed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = smoothed.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min_val >= -1e-6, "Smoothed values below 0: {}", min_val);
    assert!(
        max_val <= 1.5,
        "Smoothed values above expected max: {}",
        max_val
    );

    // 2. Sobel edge detection along axis=0 (vertical edges)
    // Signature: sobel(input, axis: usize, mode: Option<BorderMode>)
    let edges = sobel(&smoothed, 0, Some(BorderMode::Reflect))
        .map_err(|e| format!("sobel failed: {}", e))?;

    assert_eq!(edges.shape(), &[rows, cols], "Edge image shape mismatch");

    // Interior pixel (15, 15) — well inside the 10×10 block → low edge magnitude
    let interior_edge = edges[[15, 15]].abs();
    assert!(
        interior_edge < 0.5,
        "Interior edge magnitude too large: {}",
        interior_edge
    );

    // At least some pixels should have non-trivial edge response
    let max_edge = edges.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_edge > 0.0,
        "No edges detected (max edge = {})",
        max_edge
    );

    println!(
        "Vision pipeline (Gaussian + Sobel): min={:.4}, max={:.4}, max_edge={:.4}",
        min_val, max_val, max_edge
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Morphological dilation increases foreground, erosion decreases it
// ─────────────────────────────────────────────────────────────────────────────

/// Verify binary morphological duality: dilation grows foreground, erosion shrinks it.
#[test]
fn test_vision_pipeline_morphology() -> TestResult<()> {
    let img = binary_image_16();

    // Count original foreground pixels
    let fg_orig = img.iter().filter(|&&v| v).count();
    assert_eq!(fg_orig, 25, "Expected 5×5 = 25 foreground pixels");

    // 3×3 structuring element (all-ones / box)
    let struct_elem = Array2::<bool>::from_elem((3, 3), true);

    // binary_dilation(input: &Array<bool,D>, structure, iterations, mask, border_value, origin, brute_force)
    let dilated = binary_dilation(
        &img,
        Some(&struct_elem),
        None, // iterations
        None, // mask
        None, // border_value
        None, // origin
        None, // brute_force
    )
    .map_err(|e| format!("binary_dilation failed: {}", e))?;

    let fg_dilated = dilated.iter().filter(|&&v| v).count();
    assert!(
        fg_dilated >= fg_orig,
        "Dilation must not reduce foreground: before={}, after={}",
        fg_orig,
        fg_dilated
    );

    let eroded = binary_erosion(
        &img,
        Some(&struct_elem),
        None, // iterations
        None, // mask
        None, // border_value
        None, // origin
        None, // brute_force
    )
    .map_err(|e| format!("binary_erosion failed: {}", e))?;

    let fg_eroded = eroded.iter().filter(|&&v| v).count();
    assert!(
        fg_eroded <= fg_orig,
        "Erosion must not increase foreground: before={}, after={}",
        fg_orig,
        fg_eroded
    );

    // After dilation: 5×5 block grows with a 3×3 SE → more pixels
    assert!(
        fg_dilated > fg_orig,
        "Expected dilation to grow the 5×5 block; got {} (same as {})",
        fg_dilated,
        fg_orig
    );

    // After erosion: 5×5 block shrinks → fewer pixels
    assert!(
        fg_eroded < fg_orig,
        "Expected erosion to shrink the 5×5 block; got {} (same as {})",
        fg_eroded,
        fg_orig
    );

    println!(
        "Morphology pipeline: orig={}, dilated={}, eroded={}",
        fg_orig, fg_dilated, fg_eroded
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Histogram sums to total pixel count
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that the image histogram counts all pixels exactly once.
#[test]
fn test_vision_pipeline_histogram() -> TestResult<()> {
    let image = synthetic_image_32();
    let total_pixels = image.len();

    // histogram(input, min, max, bins, labels: Option<&Array<usize,D>>, index: Option<&[usize]>)
    // Returns: NdimageResult<(Array1<usize>, Array1<T>)>
    // We use (counts, bin_edges) — we only care about counts here.
    let n_bins = 32usize;
    let min_val = 0.0f64;
    let max_val = 1.2f64;

    let (hist, _bin_edges) = histogram(
        &image,
        min_val,
        max_val,
        n_bins,
        None::<&Array2<usize>>,
        None::<&[usize]>,
    )
    .map_err(|e| format!("histogram failed: {}", e))?;

    assert_eq!(
        hist.len(),
        n_bins,
        "Expected {} histogram bins, got {}",
        n_bins,
        hist.len()
    );

    let hist_sum: usize = hist.iter().sum();
    assert_eq!(
        hist_sum, total_pixels,
        "Histogram sum ({}) must equal total pixel count ({})",
        hist_sum, total_pixels
    );

    // At least 2 non-empty bins
    let non_empty_bins = hist.iter().filter(|&&c| c > 0).count();
    assert!(
        non_empty_bins >= 2,
        "Expected at least 2 non-empty histogram bins, got {}",
        non_empty_bins
    );

    // Low-intensity pixels should be in the first few bins
    let low_bin_count: usize = hist.iter().take(5).sum();
    assert!(
        low_bin_count > 0,
        "Expected some pixels in the low-intensity bins"
    );

    println!(
        "Histogram: {} bins, sum={}, non-empty={}",
        n_bins, hist_sum, non_empty_bins
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Memory-efficiency smoke test
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that pipeline stages on a moderate image don't exceed memory budget.
#[test]
fn test_vision_pipeline_memory_smoke() -> TestResult<()> {
    assert_memory_efficient(
        || {
            let img = synthetic_image_32();
            let smoothed = gaussian_filter(&img, 1.0, None, None)
                .map_err(|e| format!("gaussian_filter: {}", e))?;
            let _edges = sobel(&smoothed, 0, None).map_err(|e| format!("sobel: {}", e))?;
            Ok(())
        },
        10.0, // 10 MB max for a 32×32 pipeline
        "Vision pipeline smoke",
    )?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Gaussian near-identity property
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian filter with very small sigma should preserve values closely.
#[test]
fn test_vision_gaussian_near_identity() -> TestResult<()> {
    let img = synthetic_image_32();
    // sigma=0.01 — extremely small; output should be very close to input
    let tiny_sigma =
        gaussian_filter(&img, 0.01, None, None).map_err(|e| format!("gaussian_filter: {}", e))?;

    for (&orig, &filt) in img.iter().zip(tiny_sigma.iter()) {
        assert_abs_diff_eq!(orig, filt, epsilon = 1e-3);
    }
    println!("Gaussian near-identity property verified");
    Ok(())
}
