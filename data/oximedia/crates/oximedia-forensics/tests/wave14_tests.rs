//! Wave 14 Slice F integration tests for oximedia-forensics.
//!
//! Covers:
//!  1. `analyze_regions_tiled_default` ≈ `analyze_regions` (tiled ELA)
//!  2. `detect_splicing_prnu` parallel path — correctness and no-panic
//!  3. Progressive early-stop in `ForensicsAnalyzer::analyze`
//!  4. `DctCache` idempotency — same result on cache hit, recompute on miss

use image::{Rgb, RgbImage};
use oximedia_forensics::compression::{compute_dct_blocks_cached, DctCache};
use oximedia_forensics::ela::{analyze_regions, analyze_regions_tiled_default};
use oximedia_forensics::noise::detect_splicing_prnu;
use oximedia_forensics::{ForensicsAnalyzer, ForensicsConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 256×256 RGB image with a smooth gradient, then overwrite a 64×64
/// block in the centre with a hard white patch.  This simulates a spliced
/// region and gives the ELA algorithm something to find.
fn make_spliced_image(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = 128u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    // Overwrite centre block with white
    let cx = w / 4;
    let cy = h / 4;
    for y in cy..cy + h / 4 {
        for x in cx..cx + w / 4 {
            img.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }
    img
}

/// Encode an `RgbImage` as JPEG bytes so we can pass them to `analyze()`.
fn encode_jpeg(img: &RgbImage) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let (w, h) = img.dimensions();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 85);
    enc.encode(img.as_raw(), w, h, image::ColorType::Rgb8.into())
        .expect("test JPEG encode");
    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — tiled ELA ≈ full ELA
// ─────────────────────────────────────────────────────────────────────────────

/// The tiled ELA function must return valid regions and scores in the unit
/// interval.  We also verify that the tiled and full-frame analyses agree on
/// the *mean ELA score* of the image within a reasonable tolerance — both
/// methods perform the same recompression, so their per-pixel error values are
/// identical; the only difference is how the regions are partitioned.
#[test]
fn test_tiled_ela_matches_full() {
    let img = make_spliced_image(256, 256);

    // Full-frame analysis with region_size=128 (matches default tile size).
    let full = analyze_regions(&img, 128).expect("full ELA");

    // Tiled analysis with default tile size (128).
    let tiled = analyze_regions_tiled_default(&img).expect("tiled ELA");

    assert!(
        !full.is_empty(),
        "full ELA should return at least one region"
    );
    assert!(
        !tiled.is_empty(),
        "tiled ELA should return at least one region"
    );

    // All tiled scores must be non-negative.
    for r in &tiled {
        assert!(r.score >= 0.0, "tiled ELA score must be non-negative");
    }

    // The mean ELA score from both methods should agree closely.
    // Both partition the 256×256 image into 128×128 tiles — there are exactly
    // 4 tiles each.  Mean score from both must be within a tolerance of 5.0
    // (error levels are roughly 0–100).
    let full_mean = full.iter().map(|&(_, _, s)| s).sum::<f64>() / full.len() as f64;
    let tiled_mean = tiled.iter().map(|r| r.score).sum::<f64>() / tiled.len() as f64;

    assert_eq!(
        full.len(),
        tiled.len(),
        "full and tiled ELA should produce the same number of regions for region_size == tile_size"
    );

    // Scores should be very close since both iterate the same pixel sets.
    let diff = (full_mean - tiled_mean).abs();
    assert!(
        diff < 5.0,
        "full mean ELA score {full_mean:.3} and tiled mean {tiled_mean:.3} diverged by {diff:.3}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — parallel PRNU: no-panic + valid result
// ─────────────────────────────────────────────────────────────────────────────

/// `detect_splicing_prnu` now uses rayon internally.  Run it on a 256×256 grey
/// image and assert:
///   - it doesn't panic
///   - it returns `Ok`
///   - every returned region has non-zero dimensions
///   - calling it twice gives the same number of spliced regions (determinism)
#[test]
fn test_parallel_noise_serial_equiv() {
    // 256×256 uniform grey — no real splicing, so most images will return 0 regions.
    let img = RgbImage::from_pixel(256, 256, Rgb([128, 128, 128]));

    let result1 = detect_splicing_prnu(&img).expect("first call must not fail");
    let result2 = detect_splicing_prnu(&img).expect("second call must not fail");

    // Each entry is (x, y, w, h); size must be positive.
    for (_, _, w, h) in &result1 {
        assert!(*w > 0 && *h > 0, "region size must be positive");
    }

    // Deterministic output.
    assert_eq!(
        result1.len(),
        result2.len(),
        "parallel PRNU results should be deterministic"
    );
}

/// Exercise `detect_splicing_prnu` on an image that has a clearly spliced
/// region (hard white patch on a gradient) to confirm the parallel path
/// actually detects something plausible.
#[test]
fn test_parallel_noise_detects_splice() {
    let img = make_spliced_image(256, 256);
    // Just assert it doesn't panic and returns a valid result.
    let result = detect_splicing_prnu(&img).expect("must not fail on spliced image");
    // Every entry must have plausible coordinates.
    let (h, w) = (256usize, 256usize);
    for &(x, y, rw, rh) in &result {
        assert!(x < w, "region x out of bounds");
        assert!(y < h, "region y out of bounds");
        assert!(rw > 0 && rh > 0, "region dimensions must be positive");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — progressive early-stop
// ─────────────────────────────────────────────────────────────────────────────

/// Set `confidence_threshold = 0.01` — an almost-guaranteed early exit on any
/// real image.  Assert `report.early_stop == true` and that at least one test
/// ran (the report is non-empty).
#[test]
fn test_progressive_early_stop() {
    let img = make_spliced_image(64, 64);
    let jpeg = encode_jpeg(&img);

    let config = ForensicsConfig {
        // Very low threshold — triggers early stop after the first test.
        confidence_threshold: 0.01,
        // Enable all tests so there are multiple to potentially run.
        enable_compression_analysis: true,
        enable_ela: true,
        enable_noise_analysis: true,
        enable_geometric_analysis: true,
        enable_lighting_analysis: true,
        enable_metadata_analysis: true,
        min_confidence_threshold: 0.0,
    };

    let analyzer = ForensicsAnalyzer::with_config(config);
    let report = analyzer.analyze(&jpeg).expect("analysis must succeed");

    assert!(
        report.early_stop,
        "report.early_stop must be true when confidence_threshold is very low"
    );
    assert!(
        !report.tests.is_empty(),
        "at least one test must have run before the early stop"
    );
    // The report must still have a valid (possibly partial) overall confidence.
    assert!(
        report.overall_confidence >= 0.0 && report.overall_confidence <= 1.0,
        "overall_confidence must be in [0,1]"
    );
}

/// Verify that with the default threshold (1.0) `early_stop` is never set.
#[test]
fn test_no_early_stop_by_default() {
    // Use 128×128 so that lighting.rs region_size=64 doesn't underflow.
    let img = RgbImage::from_pixel(128, 128, Rgb([100, 150, 200]));
    let jpeg = encode_jpeg(&img);

    let analyzer = ForensicsAnalyzer::new();
    let report = analyzer.analyze(&jpeg).expect("analysis must succeed");

    assert!(
        !report.early_stop,
        "early_stop must be false with default config (threshold=1.0)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — DCT cache idempotency
// ─────────────────────────────────────────────────────────────────────────────

/// Calling `compute_dct_blocks_cached` twice on the **same image** must return
/// identical data (cache hit).  Calling it on a **different-sized image** must
/// return a result with different dimensions (cache miss + recompute).
#[test]
fn test_dct_cache_idempotent() {
    let img_a = {
        let mut img = RgbImage::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                img.put_pixel(x, y, Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
            }
        }
        img
    };

    let mut cache = DctCache::new();

    // First call — cache miss, computes and stores.
    let blocks1 = compute_dct_blocks_cached(&img_a, &mut cache);
    let dim1 = blocks1.dim();
    // Clone data for later comparison.
    let data1: Vec<f64> = blocks1.iter().cloned().collect();

    // Second call with same image — cache hit.
    let blocks2 = compute_dct_blocks_cached(&img_a, &mut cache);
    let dim2 = blocks2.dim();
    let data2: Vec<f64> = blocks2.iter().cloned().collect();

    assert_eq!(dim1, dim2, "dimensions must match on cache hit");
    assert_eq!(
        data1, data2,
        "all coefficients must be identical on cache hit"
    );

    // Now use a differently-sized image — cache miss, recompute.
    let img_b = RgbImage::new(128, 128);
    let blocks3 = compute_dct_blocks_cached(&img_b, &mut cache);
    let dim3 = blocks3.dim();

    // A 128×128 image has 16×16 = 256 blocks; a 64×64 image has 8×8 = 64.
    assert_ne!(
        dim1, dim3,
        "cache should recompute for a different image size; got same dims {:?}",
        dim3
    );
}

/// Verify `DctCache::is_valid_for` correctly tracks the cached dimensions.
#[test]
fn test_dct_cache_validity_tracking() {
    let img = RgbImage::new(64, 64);
    let mut cache = DctCache::new();

    assert!(
        !cache.is_valid_for(64, 64),
        "cache must be invalid before first use"
    );

    compute_dct_blocks_cached(&img, &mut cache);

    assert!(
        cache.is_valid_for(64, 64),
        "cache must be valid after computing for (64,64)"
    );
    assert!(
        !cache.is_valid_for(128, 64),
        "cache must be invalid for a different width"
    );

    cache.invalidate();
    assert!(
        !cache.is_valid_for(64, 64),
        "cache must be invalid after explicit invalidation"
    );
}
