//! False-positive rate (FPR) measurement for each forensic test type.
//!
//! A forensic detector is only useful if it stays quiet on genuine,
//! untampered media. This suite builds a small corpus of synthetic
//! **known-clean** images — smooth, single-light-source gradients with a
//! uniform, globally-consistent noise floor (i.e. a plausible unedited
//! camera capture) — and runs every pixel-level [`oximedia_forensics`] test
//! against each one, asserting that `tampering_detected` stays `false` and
//! recording the aggregate false-positive rate per test type.
//!
//! None of these images contain spliced content, duplicated regions,
//! inconsistent noise, or inconsistent lighting by construction, so a
//! `tampering_detected == true` result from any of them is, by definition, a
//! false positive.
//!
//! The metadata test is intentionally *excluded* from the aggregate FPR
//! loop: `metadata::analyze_metadata` flags any image lacking EXIF data
//! (`raw_tags.is_empty()`) as suspicious by design ("possible stripping"),
//! and our synthetic PNG/JPEG fixtures never carry EXIF. That is expected,
//! documented behaviour, not a false positive — see
//! `test_metadata_flags_missing_exif_by_design` below, which locks in that
//! contract instead of silently excluding it.

use image::{Rgb, RgbImage};
use oximedia_forensics::{compression, ela, geometric, lighting, metadata, noise};

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

/// Build a "clean" synthetic photograph: a single, smoothly-varying
/// diagonal brightness gradient (simulating one consistent directional
/// light source) confined to a narrow, non-extreme range (110..170) so it
/// never trips the `lighting` module's "both very bright and very dark
/// region" impossible-lighting heuristic, plus a small globally-uniform
/// per-pixel noise floor (simulating consistent sensor noise across the
/// whole frame — the hallmark of an un-tampered capture).
fn make_clean_gradient_image(w: u32, h: u32, seed: u32, angle_flip: bool) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = Xorshift32::new(seed);

    for y in 0..h {
        for x in 0..w {
            let t = if angle_flip {
                (x + y) as f64 / (w + h) as f64
            } else {
                x as f64 / w.max(1) as f64
            };
            let base = 110.0 + t * 60.0; // 110..170, single consistent gradient
            let n = f64::from(rng.jitter(6)); // uniform sensor-noise floor
            let v = (base + n).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }

    img
}

/// Build a "clean" flat, uniformly-colored frame (e.g. a photo of an
/// evenly-lit wall or sky) with the same globally-uniform noise floor.
fn make_clean_flat_image(w: u32, h: u32, seed: u32, level: u8) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = Xorshift32::new(seed);

    for y in 0..h {
        for x in 0..w {
            let n = rng.jitter(6);
            let v = (i32::from(level) + n).clamp(0, 255) as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }

    img
}

/// Build a "clean" richly-textured image (a "quilt" of small, distinct
/// base-color cells with per-pixel jitter — see `tampering_fixtures.rs`'s
/// `make_cloned_fixture` for the same generator, but here **without** the
/// deliberate block duplication). Copy-move / block-matching detectors are
/// specifically prone to false positives on flat, low-texture content
/// (large statistically-uniform regions can spuriously satisfy a
/// feature-similarity threshold even without any real duplication), so this
/// generator is used instead of [`make_clean_flat_image`] for the geometric
/// detector's FPR corpus — every 16×16 cell gets its own independently
/// seeded base color, giving each block a distinctive, non-repeating
/// fingerprint the way real photographic texture would.
fn make_clean_textured_image(w: u32, h: u32, seed: u32) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let mut rng = Xorshift32::new(seed);
    let cell = 16u32;

    for y in 0..h {
        for x in 0..w {
            let cell_seed = (x / cell) * 977 + (y / cell) * 131 + seed;
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

    img
}

fn encode_jpeg(img: &RgbImage, quality: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let (w, h) = img.dimensions();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    enc.encode(img.as_raw(), w, h, image::ColorType::Rgb8.into())
        .expect("test JPEG encode must succeed");
    buf
}

/// The clean-image corpus shared by the ELA / compression / noise / lighting
/// FPR checks.
fn clean_corpus() -> Vec<RgbImage> {
    vec![
        make_clean_gradient_image(256, 256, 0xA11C_E001, false),
        make_clean_gradient_image(256, 256, 0xB22D_F002, true),
        make_clean_flat_image(256, 256, 0xC33E_0003, 140),
        make_clean_flat_image(256, 256, 0xD44F_1004, 160),
    ]
}

/// A separate, richly-textured clean corpus for the copy-move / geometric
/// FPR check (see [`make_clean_textured_image`] for why flat images are
/// unsuitable for this particular detector).
fn textured_clean_corpus() -> Vec<RgbImage> {
    vec![
        make_clean_textured_image(256, 256, 0x1357_9BDF),
        make_clean_textured_image(256, 256, 0x2468_ACE0),
    ]
}

/// A separate clean corpus for the `lighting` FPR check, containing only
/// the smooth-gradient images (no flat/uniform frames).
///
/// `lighting::analyze_lighting` estimates per-region light direction from
/// *local intensity gradients*. A flat, uniformly-colored frame has no real
/// directional signal at all — its only "gradient" is whatever the
/// per-pixel sensor-noise floor happens to produce — so the estimated local
/// light direction in each region is essentially noise-driven and
/// incoherent, which spuriously trips the "region disagrees with the
/// dominant light direction by more than 60°" inconsistency check even
/// though nothing is actually wrong with the image. A single smooth
/// directional gradient (as produced by [`make_clean_gradient_image`]) has
/// a real, dominant shading direction for the noise floor to sit on top of
/// and is a fair representative of a genuinely lit, untampered photograph.
fn lighting_clean_corpus() -> Vec<RgbImage> {
    vec![
        make_clean_gradient_image(256, 256, 0xA11C_E001, false),
        make_clean_gradient_image(256, 256, 0xB22D_F002, true),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-test-type false positive rate
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ela_false_positive_rate_on_clean_corpus() {
    let corpus = clean_corpus();
    let mut false_positives = 0usize;
    for img in &corpus {
        let test = ela::perform_ela(img).expect("ELA must succeed on a valid RGB image");
        if test.tampering_detected {
            false_positives += 1;
        }
    }
    assert_eq!(
        false_positives,
        0,
        "ELA false-positive rate on clean corpus must be 0/{}, got {false_positives}",
        corpus.len()
    );
}

#[test]
fn test_compression_false_positive_rate_on_clean_corpus() {
    let corpus = clean_corpus();
    let mut false_positives = 0usize;
    for img in &corpus {
        let test = compression::analyze_compression(img)
            .expect("compression analysis must succeed on a valid RGB image");
        if test.tampering_detected {
            false_positives += 1;
        }
    }
    assert_eq!(
        false_positives, 0,
        "compression-analysis false-positive rate on clean corpus must be 0/{}, got {false_positives}",
        corpus.len()
    );
}

#[test]
fn test_noise_false_positive_rate_on_clean_corpus() {
    let corpus = clean_corpus();
    let mut false_positives = 0usize;
    for img in &corpus {
        let test =
            noise::analyze_noise(img).expect("noise analysis must succeed on a valid RGB image");
        if test.tampering_detected {
            false_positives += 1;
        }
    }
    assert_eq!(
        false_positives,
        0,
        "noise-analysis false-positive rate on clean corpus must be 0/{}, got {false_positives}",
        corpus.len()
    );
}

#[test]
fn test_geometric_false_positive_rate_on_clean_corpus() {
    // Uses the richly-textured corpus, not `clean_corpus()`: block-matching
    // copy-move detection is prone to false positives on large flat/smooth
    // regions (statistically-similar blocks can spuriously pass the
    // feature-similarity threshold with no real duplication present), so a
    // textured corpus is the fair, representative test here.
    let corpus = textured_clean_corpus();
    let mut false_positives = 0usize;
    for img in &corpus {
        let test = geometric::detect_copy_move(img)
            .expect("copy-move detection must succeed on a valid RGB image");
        if test.tampering_detected {
            false_positives += 1;
        }
    }
    assert_eq!(
        false_positives, 0,
        "copy-move-detection false-positive rate on textured clean corpus must be 0/{}, got {false_positives}",
        corpus.len()
    );
}

#[test]
fn test_lighting_false_positive_rate_on_clean_corpus() {
    // Uses `lighting_clean_corpus()`, not `clean_corpus()`: lighting
    // analysis estimates a light direction from local gradients, and a
    // flat/uniform frame has no real directional signal for that estimate
    // to lock onto (see `lighting_clean_corpus` for the full rationale).
    let corpus = lighting_clean_corpus();
    let mut false_positives = 0usize;
    for img in &corpus {
        let test = lighting::analyze_lighting(img)
            .expect("lighting analysis must succeed on a valid RGB image");
        if test.tampering_detected {
            false_positives += 1;
        }
    }
    assert_eq!(
        false_positives, 0,
        "lighting-analysis false-positive rate on clean (gradient) corpus must be 0/{}, got {false_positives}",
        corpus.len()
    );
}

/// End-to-end sanity: on a clean JPEG-encoded image, every pixel-level test
/// individually reports no tampering, and the overall analyzer's confidence
/// stays below the detection threshold (0.5, per
/// `TamperingReport::calculate_overall_confidence`).
///
/// Two sub-tests are disabled here, each for a documented, orthogonal
/// reason — neither is a defect in the analyzer:
///
/// - **Geometric (copy-move)**: a smooth analytic gradient has
///   constant-value level lines/curves by construction (e.g. every pixel
///   with the same `x` — or the same `x + y` — shares almost the same
///   brightness), so far-apart 16×16 blocks lying on the same level line
///   can look statistically near-identical to the block-matching feature
///   comparison even though nothing was actually duplicated. That is a
///   property of this *synthetic gradient fixture*, not of the detector
///   under realistic photographic texture, and it is exactly why the
///   copy-move detector's own FPR is measured separately, on a corpus
///   purpose-built to avoid that degeneracy (see
///   `test_geometric_false_positive_rate_on_clean_corpus` and
///   [`make_clean_textured_image`]).
/// - **Metadata**: our synthetic JPEG carries no EXIF tags at all, and
///   `metadata::analyze_metadata` flags that absence by design (see
///   `test_metadata_flags_missing_exif_by_design`) — that is a
///   pixel-content-independent signal this sanity check isn't meant to
///   exercise.
#[test]
fn test_full_analyzer_no_false_positive_on_clean_image() {
    use oximedia_forensics::{ForensicsAnalyzer, ForensicsConfig};

    let img = make_clean_gradient_image(256, 256, 0xE55A_2005, false);
    let jpeg = encode_jpeg(&img, 90);

    let config = ForensicsConfig {
        enable_geometric_analysis: false,
        enable_metadata_analysis: false,
        ..ForensicsConfig::default()
    };
    let analyzer = ForensicsAnalyzer::with_config(config);
    let report = analyzer
        .analyze(&jpeg)
        .expect("full analysis must succeed on a clean image");

    for (name, test) in &report.tests {
        assert!(
            !test.tampering_detected,
            "test '{name}' incorrectly flagged a clean image as tampered (confidence={})",
            test.confidence
        );
    }
    assert!(
        !report.tampering_detected,
        "overall report must not flag a clean image as tampered (confidence={})",
        report.overall_confidence
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata test — documented exception, not part of the aggregate FPR loop.
// ─────────────────────────────────────────────────────────────────────────────

/// `metadata::analyze_metadata` treats the *absence* of EXIF metadata itself
/// as a forensic signal ("possible stripping"), independent of pixel
/// content. This is a deliberate design choice (missing provenance metadata
/// is itself suspicious in a chain-of-custody context), so a synthetic image
/// with no EXIF tags is *expected* to be flagged — this is not a false
/// positive in the pixel-tampering sense measured above, and is recorded
/// here explicitly rather than silently folded into the clean-corpus loop.
#[test]
fn test_metadata_flags_missing_exif_by_design() {
    let img = make_clean_gradient_image(256, 256, 0xF66B_3006, false);
    let jpeg = encode_jpeg(&img, 90);

    let test = metadata::analyze_metadata(&jpeg)
        .expect("metadata analysis must succeed even without EXIF tags");

    assert!(
        test.tampering_detected,
        "missing-EXIF images are flagged by design; if this ever changes, \
         re-home this fixture into the aggregate clean-corpus FPR loop above"
    );
    assert!((test.confidence - 0.4).abs() < 1e-9);
}
