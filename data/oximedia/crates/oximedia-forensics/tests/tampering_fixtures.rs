//! Ground-truth tampering fixtures.
//!
//! This suite procedurally generates three canonical classes of image
//! tampering — **splicing** (foreign content pasted in), **cloning**
//! (copy-move duplication of a region within the same image), and
//! **retouching** (localized denoising/smoothing that erases a region's
//! natural sensor noise) — together with an exact ground-truth mask of the
//! tampered pixels. Each fixture is fed to the relevant
//! [`oximedia_forensics`] detector (and, for splicing, the full
//! [`ForensicsAnalyzer`] pipeline) and we assert that:
//!
//! 1. The detector actually flags tampering (`tampering_detected == true` /
//!    `overall_confidence` above a floor for the whole-analyzer case).
//! 2. The detector's per-pixel anomaly map is *localized* — the mean
//!    anomaly value inside the known-tampered mask is meaningfully higher
//!    than the mean value outside of it. This is the core "ground truth
//!    mask" check: it verifies the detector points at the right pixels, not
//!    just that it fires at all.
//!
//! No external image assets are used — every fixture is synthesized in-code
//! with a small deterministic xorshift PRNG so the tests are fully
//! reproducible and hermetic (see repository policy: tests use
//! `std::env::temp_dir()` and never depend on network/external files).

use image::{Rgb, RgbImage};
use oximedia_forensics::flat_array2::FlatArray2;
use oximedia_forensics::{ela, geometric, noise, ForensicsAnalyzer};

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic PRNG (xorshift32) — no external `rand` dependency needed.
// ─────────────────────────────────────────────────────────────────────────────

struct Xorshift32(u32);

impl Xorshift32 {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Signed jitter in `[-amplitude, amplitude]`.
    fn jitter(&mut self, amplitude: i32) -> i32 {
        if amplitude == 0 {
            return 0;
        }
        let span = (2 * amplitude + 1) as u32;
        (self.next_u32() % span) as i32 - amplitude
    }
}

/// A rectangular ground-truth region, `(x, y, width, height)` in pixels.
type Mask = Vec<(u32, u32, u32, u32)>;

/// Mean value of `map` inside vs. outside the given ground-truth `mask`
/// rectangles. Returns `(mean_inside, mean_outside)`.
fn mask_overlap(map: &FlatArray2<f64>, mask: &Mask) -> (f64, f64) {
    let (rows, cols) = map.dim();
    let mut inside_sum = 0.0;
    let mut inside_count = 0usize;
    let mut outside_sum = 0.0;
    let mut outside_count = 0usize;

    for y in 0..rows {
        for x in 0..cols {
            let in_mask = mask.iter().any(|&(mx, my, mw, mh)| {
                let x = x as u32;
                let y = y as u32;
                x >= mx && x < mx + mw && y >= my && y < my + mh
            });
            let v = map[[y, x]];
            if in_mask {
                inside_sum += v;
                inside_count += 1;
            } else {
                outside_sum += v;
                outside_count += 1;
            }
        }
    }

    let mean_inside = if inside_count > 0 {
        inside_sum / inside_count as f64
    } else {
        0.0
    };
    let mean_outside = if outside_count > 0 {
        outside_sum / outside_count as f64
    } else {
        0.0
    };
    (mean_inside, mean_outside)
}

/// Encode an `RgbImage` as JPEG bytes.
fn encode_jpeg(img: &RgbImage, quality: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let (w, h) = img.dimensions();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    enc.encode(img.as_raw(), w, h, image::ColorType::Rgb8.into())
        .expect("test JPEG encode must succeed");
    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 1 — SPLICED image (foreign flat-color patch pasted onto a photo)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a smooth photographic-looking gradient with a hard-edged, flat-color
/// "spliced-in" patch of known bounds. Splicing a foreign element in almost
/// always breaks local compression-error consistency (ELA) because the
/// pasted content did not share the same JPEG generation history as the
/// host image.
fn make_spliced_fixture(w: u32, h: u32) -> (RgbImage, Mask) {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = 128u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // Splice patch: a hard flat-color block in the upper-left quadrant.
    let (px, py, pw, ph) = (w / 4, h / 4, w / 4, h / 4);
    for y in py..py + ph {
        for x in px..px + pw {
            img.put_pixel(x, y, Rgb([250, 245, 10]));
        }
    }

    (img, vec![(px, py, pw, ph)])
}

#[test]
fn test_spliced_image_ela_localizes_ground_truth_mask() {
    let (img, mask) = make_spliced_fixture(256, 256);

    let test = ela::perform_ela(&img).expect("ELA must succeed on a valid RGB image");

    assert!(
        test.tampering_detected,
        "ELA should flag the spliced flat-color patch as anomalous"
    );

    let anomaly_map = test
        .anomaly_map
        .as_ref()
        .expect("ELA must attach an error map");
    let (mean_inside, mean_outside) = mask_overlap(anomaly_map, &mask);

    assert!(
        mean_inside > mean_outside,
        "ELA error should be concentrated inside the spliced ground-truth mask \
         (inside={mean_inside:.3}, outside={mean_outside:.3})"
    );
}

#[test]
fn test_spliced_image_full_analyzer_confidence() {
    let (img, _mask) = make_spliced_fixture(256, 256);
    // Re-encode through JPEG so the full pipeline (which parses image bytes)
    // sees a realistic compressed asset, then splice detection through ELA
    // (recompression at a different quality) has something to find.
    let jpeg = encode_jpeg(&img, 92);

    let analyzer = ForensicsAnalyzer::new();
    let report = analyzer.analyze(&jpeg).expect("full analysis must succeed");

    let ela_test = report
        .tests
        .get("Error Level Analysis (ELA)")
        .expect("ELA test must be present in the report");
    assert!(
        ela_test.tampering_detected,
        "full analyzer's ELA sub-test must flag the spliced patch"
    );
    assert!(
        report.overall_confidence > 0.0,
        "overall confidence must reflect the ELA detection"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 2 — CLONED image (copy-move duplication within the same image)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a "quilted" procedural texture (spatially varying, but not
/// perceptually periodic within the image bounds) so that block-matching
/// copy-move detection has a non-trivial background, then duplicate a
/// 64×64 source block onto a destination location far enough away to avoid
/// self-matching (> 40px, per `geometric::detect_copy_move`'s
/// `min_distance`).
fn make_cloned_fixture(w: u32, h: u32) -> (RgbImage, Mask) {
    let mut img = RgbImage::new(w, h);
    let mut rng = Xorshift32::new(0xC10E_D001);

    // Coarse 16×16 "quilt" cells with a distinct base color per cell, plus a
    // small per-pixel jitter so texture isn't perfectly flat within a cell
    // (avoids trivial matches from uniform-noise-free flat blocks).
    let cell = 16u32;
    for y in 0..h {
        for x in 0..w {
            let cell_seed = (x / cell) * 977 + (y / cell) * 131 + 1;
            let mut cell_rng = Xorshift32::new(cell_seed);
            let base_r = (cell_rng.next_u32() % 200) as i32 + 20;
            let base_g = (cell_rng.next_u32() % 200) as i32 + 20;
            let base_b = (cell_rng.next_u32() % 200) as i32 + 20;
            let jr = rng.jitter(4);
            let jg = rng.jitter(4);
            let jb = rng.jitter(4);
            let r = (base_r + jr).clamp(0, 255) as u8;
            let g = (base_g + jg).clamp(0, 255) as u8;
            let b = (base_b + jb).clamp(0, 255) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // Clone a 64×64 source region onto a destination far away (diagonal
    // offset well beyond the 40px min_distance threshold).
    let (sx, sy, size) = (16u32, 16u32, 64u32);
    let (dx, dy) = (w - size - 16, h - size - 16);

    let source: Vec<Rgb<u8>> = (0..size)
        .flat_map(|dyy| (0..size).map(move |dxx| (dxx, dyy)))
        .map(|(dxx, dyy)| *img.get_pixel(sx + dxx, sy + dyy))
        .collect();

    let mut idx = 0;
    for dyy in 0..size {
        for dxx in 0..size {
            img.put_pixel(dx + dxx, dy + dyy, source[idx]);
            idx += 1;
        }
    }

    (img, vec![(sx, sy, size, size), (dx, dy, size, size)])
}

#[test]
fn test_cloned_image_geometric_localizes_ground_truth_mask() {
    let (img, mask) = make_cloned_fixture(256, 256);

    let test = geometric::detect_copy_move(&img).expect("copy-move detection must succeed");

    assert!(
        test.tampering_detected,
        "copy-move detector must flag the duplicated 64x64 region"
    );

    let anomaly_map = test
        .anomaly_map
        .as_ref()
        .expect("copy-move detection must attach an anomaly map");
    let (mean_inside, mean_outside) = mask_overlap(anomaly_map, &mask);

    assert!(
        mean_inside > mean_outside,
        "copy-move anomaly map should be concentrated inside the source/destination \
         ground-truth mask (inside={mean_inside:.3}, outside={mean_outside:.3})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 3 — RETOUCHED image (localized denoise/smoothing erases sensor
// noise inside a region, a classic "blemish removal" retouch signature)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a uniformly noisy image (simulating natural sensor noise across the
/// whole frame) and then flatten a single 64×64 region to its local mean
/// (simulating a retouching / skin-smoothing tool that removes texture in
/// one spot but leaves the rest of the sensor noise untouched).
fn make_retouched_fixture(w: u32, h: u32) -> (RgbImage, Mask) {
    let mut img = RgbImage::new(w, h);
    let mut rng = Xorshift32::new(0x2E70_11C1);

    for y in 0..h {
        for x in 0..w {
            let base = 130i32;
            let n = rng.jitter(28);
            let v = (base + n).clamp(0, 255) as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }

    // Retouch region: flatten to a constant value (all texture/noise
    // removed), simulating aggressive local smoothing.
    let (rx, ry, rw, rh) = (w / 2 - 32, h / 2 - 32, 64u32, 64u32);
    for y in ry..ry + rh {
        for x in rx..rx + rw {
            img.put_pixel(x, y, Rgb([130, 130, 130]));
        }
    }

    (img, vec![(rx, ry, rw, rh)])
}

#[test]
fn test_retouched_image_noise_localizes_ground_truth_mask() {
    let (img, mask) = make_retouched_fixture(256, 256);

    let test = noise::analyze_noise(&img).expect("noise analysis must succeed");

    assert!(
        test.tampering_detected,
        "noise analysis must flag the artificially flattened (retouched) region \
         as a noise-inconsistency outlier"
    );

    let anomaly_map = test
        .anomaly_map
        .as_ref()
        .expect("noise analysis must attach an anomaly map");
    let (mean_inside, mean_outside) = mask_overlap(anomaly_map, &mask);

    assert!(
        mean_inside > mean_outside,
        "noise-inconsistency anomaly map should be concentrated on the flattened \
         ground-truth region (inside={mean_inside:.3}, outside={mean_outside:.3})"
    );
}
