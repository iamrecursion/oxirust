//! FAST + ORB feature-detection throughput / correctness across resolutions.
//!
//! These tests pin the REAL behaviour of the patent-free FAST corner detector,
//! the ORB detector/descriptor, and the brute-force `FeatureMatcher` across a
//! VGA → 4K resolution sweep. They are deliberately written against the public
//! API only (no private-helper access) so they live in `tests/` and keep
//! `src/features.rs` under the 2000-line refactor limit.
//!
//! The synthetic images place features on a lattice whose pitch is fixed in
//! PIXELS, so the corner density per unit area is CONSTANT and keypoint count
//! scales with image AREA. Each lattice cell gets a LOCALLY UNIQUE blob pattern
//! (seeded by an inline LCG) so BRIEF descriptors are discriminative and survive
//! Lowe's ratio test — a purely periodic texture would make every descriptor
//! ambiguous and the matcher would (correctly) return nothing.
//!
//! The always-on tests assert strict monotonicity VGA < HD < 2K; the 2K → 4K
//! leg plus the un-clamped 4K path live in the `#[ignore]`d `timing_liveness_4k`
//! so a slow debug-build 4K detection cannot make the default suite flaky.
//!
//! NOTE on parallel RANSAC (already implemented — NOT exercised here):
//! `spatial.rs::HomographyEstimator::estimate` is rayon-parallel (shared
//! `AtomicUsize`) and `parallel_ransac.rs` carries the adaptive
//! `log(1-p)/log(1-w^s)` early-termination bound via
//! `prosac::adaptive_max_iterations`. This file is the FAST+ORB feature test.

use oximedia_align::features::{BinaryDescriptor, FastDetector, FeatureMatcher, OrbDetector};
use std::time::Instant;

/// Resolution sweep: (width, height, label). Areas grow strictly so keypoint
/// counts (∝ area for our constant-density pattern) grow strictly too.
const RESOLUTIONS: [(usize, usize, &str); 4] = [
    (640, 480, "VGA"),
    (1280, 720, "HD"),
    (2560, 1440, "2K"),
    (3840, 2160, "4K"),
];

/// Liveness ceiling — never assert speed, only that detection makes progress.
/// Generous so the test is non-flaky even on a loaded debug-build CI machine.
const LIVENESS_LIMIT_SECS: u64 = 60;

/// Feature-lattice pitch in PIXELS — fixed, so corner density is constant per
/// unit area and keypoint count scales with image AREA. A coarse 64 px pitch
/// keeps the total keypoint count modest (HD ≈ 220 cells, 4K ≈ 1980 cells, so
/// even at a few corners per cell 4K stays well under the 50 000 `max_features`
/// cap) and the brute-force O(N²) matcher stays fast, while still giving strict
/// VGA < HD < 2K < 4K monotonicity.
const FEATURE_PITCH: usize = 64;

/// Deterministic 32-bit hash of a lattice cell's `(cx, cy)` coordinate. Seeds an
/// inline LCG so each cell gets a UNIQUE local appearance (and therefore a
/// discriminative BRIEF descriptor that survives Lowe's ratio test). Mixing is a
/// splitmix-style finaliser — no external RNG / dev-deps.
fn cell_seed(cx: usize, cy: usize) -> u32 {
    let mut z = (cx as u32).wrapping_mul(0x9E37_79B1) ^ (cy as u32).wrapping_mul(0x85EB_CA77);
    z = (z ^ (z >> 16)).wrapping_mul(0x7FEB_352D);
    z = (z ^ (z >> 15)).wrapping_mul(0x846C_A68B);
    z ^ (z >> 16)
}

/// One step of a 32-bit LCG (Numerical Recipes constants).
fn lcg_step(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// Generate a deterministic, strongly-textured grayscale image whose corner
/// density is CONSTANT per unit area AND whose local features are LOCALLY UNIQUE.
///
/// The pattern is a pure function of the absolute pixel coordinate `(x, y)`, so
/// (a) enlarging the canvas reveals MORE feature cells of the same density
/// (keypoint count ∝ area), and (b) a rigid translation reproduces every
/// feature's local neighbourhood verbatim, so BRIEF descriptors match exactly
/// across a translated pair.
///
/// Construction — on a fixed `FEATURE_PITCH` lattice, each cell stamps a small
/// cluster of high-contrast square blobs whose count, positions, sizes and
/// polarities are drawn from an LCG seeded by the cell's coordinate hash. The
/// per-cell randomness makes neighbouring features DISTINCT, so the matcher's
/// ratio test accepts the correct nearest neighbour instead of rejecting an
/// ambiguous near-tie (the failure mode of a purely periodic texture). The sharp
/// blob corners are exactly what FAST-9 fires on.
fn synth_textured_image(w: usize, h: usize) -> Vec<u8> {
    // Mid-grey background so blobs of either polarity create a strong step.
    let mut img = vec![128u8; w * h];

    let cells_x = w / FEATURE_PITCH;
    let cells_y = h / FEATURE_PITCH;

    for cy in 0..cells_y {
        for cx in 0..cells_x {
            // Node = top-left of an interior margin inside the cell, so blobs
            // never straddle a cell boundary and stay clear of the image edge.
            let node_x = cx * FEATURE_PITCH;
            let node_y = cy * FEATURE_PITCH;
            let mut rng = cell_seed(cx, cy);

            // 3–5 blobs per cell — count varies per cell for extra distinctness.
            let blob_count = 3 + (lcg_step(&mut rng) % 3) as usize;
            for _ in 0..blob_count {
                // Blob top-left within an 8..(PITCH-12) sub-window of the cell.
                let span = FEATURE_PITCH - 20; // room for the largest blob
                let bx = node_x + 8 + (lcg_step(&mut rng) % span as u32) as usize;
                let by = node_y + 8 + (lcg_step(&mut rng) % span as u32) as usize;
                let size = 3 + (lcg_step(&mut rng) % 4) as usize; // 3..6 px square
                let level: u8 = if lcg_step(&mut rng) & 1 == 0 { 245 } else { 10 };

                for dy in 0..size {
                    let py = by + dy;
                    if py >= h {
                        break;
                    }
                    for dx in 0..size {
                        let px = bx + dx;
                        if px >= w {
                            break;
                        }
                        img[py * w + px] = level;
                    }
                }
            }
        }
    }

    img
}

/// Translate `src` by an integer offset `(dx, dy)`, zero-filling the exposed
/// border. A destination pixel `(x, y)` samples the source pixel
/// `(x - dx, y - dy)`; out-of-range samples become 0.
fn translate_image(src: &[u8], w: usize, h: usize, dx: isize, dy: isize) -> Vec<u8> {
    assert_eq!(src.len(), w * h, "translate_image: size mismatch");
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let sy = y as isize - dy;
        if sy < 0 || sy >= h as isize {
            continue;
        }
        for x in 0..w {
            let sx = x as isize - dx;
            if sx < 0 || sx >= w as isize {
                continue;
            }
            out[y * w + x] = src[sy as usize * w + sx as usize];
        }
    }
    out
}

/// Median of an `isize` slice (mean of the two central values, rounded toward
/// zero, for even counts). Returns 0 for an empty slice.
fn median_isize(values: &[isize]) -> isize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2
    } else {
        sorted[mid]
    }
}

/// Build an ORB detector sized so 4K is NOT clamped by the default-500
/// truncation. This is THE footgun the resolution-scaling assertion would hit:
/// with `OrbDetector::new(500)` every resolution would top out at 500 keypoints
/// and monotonicity would silently break at HD upward.
fn orb() -> OrbDetector {
    OrbDetector::new(50_000)
}

/// Modest FAST threshold — the blob/background step is ~117 grey levels, so a
/// value of 25 fires cleanly on every blob corner without picking up noise
/// (there is none in this synthetic image).
const FAST_THRESHOLD: u8 = 25;

/// 1. Keypoint count scales with resolution.
///
/// Detect at VGA / HD / 2K and require strict monotonic growth VGA < HD < 2K,
/// every count > 0, and none clamped to `max_features`. Three points already
/// prove that keypoint count tracks image AREA for the constant-density texture.
///
/// 4K is detected only in the `#[ignore]`d `timing_liveness_4k`, where the
/// 2K → 4K monotonicity and no-clamp assertions are repeated — that keeps this
/// always-on test fast (~10 s) instead of spending ~50 s on a debug-build 4K
/// detection.
#[test]
fn keypoint_count_scales_with_resolution() {
    let detector = orb();
    let mut counts: Vec<(usize, usize)> = Vec::with_capacity(3); // (area, count)

    for &(w, h, label) in RESOLUTIONS.iter().take(3) {
        let img = synth_textured_image(w, h);
        let (kps, _desc) = detector
            .detect_and_compute(&img, w, h)
            .unwrap_or_else(|e| panic!("{label}: detect_and_compute failed: {e}"));
        eprintln!(
            "[scale] {label:>3} {w}x{h} (area={}) -> {} keypoints",
            w * h,
            kps.len()
        );
        assert!(!kps.is_empty(), "{label}: expected > 0 keypoints");
        assert!(
            kps.len() < detector.max_features,
            "{label}: keypoints ({}) hit max_features ({}) — clamped, scaling test invalid",
            kps.len(),
            detector.max_features
        );
        counts.push((w * h, kps.len()));
    }

    // Strict monotonicity across the (area-increasing) sweep.
    for win in counts.windows(2) {
        let (a0, c0) = win[0];
        let (a1, c1) = win[1];
        assert!(a1 > a0, "areas must strictly increase: {a0} !< {a1}");
        assert!(
            c1 > c0,
            "keypoint count must strictly increase with area: \
             area {a0}->{a1} gave counts {c0}->{c1}"
        );
    }
}

/// 2. Descriptor count == keypoint count at VGA / HD / 2K.
///
/// The 4K case is covered by the `#[ignore]`d `timing_liveness_4k` (it asserts
/// the same `kps.len() == desc.len()` invariant) so this always-on test does not
/// pay for a slow debug-build 4K detection.
#[test]
fn descriptor_count_matches_keypoint_count() {
    let detector = orb();
    for &(w, h, label) in RESOLUTIONS.iter().take(3) {
        let img = synth_textured_image(w, h);
        let (kps, desc) = detector
            .detect_and_compute(&img, w, h)
            .unwrap_or_else(|e| panic!("{label}: detect_and_compute failed: {e}"));
        assert_eq!(
            kps.len(),
            desc.len(),
            "{label}: descriptor count ({}) != keypoint count ({})",
            desc.len(),
            kps.len()
        );
        assert!(!desc.is_empty(), "{label}: expected > 0 descriptors");
    }
}

/// 3. Matcher recovers a known integer translation.
///
/// `img2 = translate_image(img1, 12, 8)`; run ORB on both; match with
/// `FeatureMatcher::new(50, 0.8)`; require several matches whose MEDIAN
/// displacement `(point2 - point1)` equals (12, 8) within ±2 px.
#[test]
fn matcher_recovers_known_translation() {
    const DX: isize = 12;
    const DY: isize = 8;

    // HD is large enough for a dense, unambiguous match set without being slow.
    let (w, h) = (1280, 720);
    let img1 = synth_textured_image(w, h);
    let img2 = translate_image(&img1, w, h, DX, DY);

    let detector = orb();
    let (kp1, d1) = detector
        .detect_and_compute(&img1, w, h)
        .expect("img1 detect_and_compute");
    let (kp2, d2) = detector
        .detect_and_compute(&img2, w, h)
        .expect("img2 detect_and_compute");

    let matcher = FeatureMatcher::new(50, 0.8);
    let matches = matcher.match_features(&kp1, &d1, &kp2, &d2);

    eprintln!(
        "[translate] {} matches over {}+{} keypoints",
        matches.len(),
        kp1.len(),
        kp2.len()
    );
    assert!(
        matches.len() >= 8,
        "expected several matches for a clean translation, got {}",
        matches.len()
    );

    // Median displacement is robust to the handful of ratio-test outliers that
    // survive on a periodic texture.
    let dxs: Vec<isize> = matches
        .iter()
        .map(|m| (m.point2.x - m.point1.x).round() as isize)
        .collect();
    let dys: Vec<isize> = matches
        .iter()
        .map(|m| (m.point2.y - m.point1.y).round() as isize)
        .collect();
    let med_dx = median_isize(&dxs);
    let med_dy = median_isize(&dys);

    eprintln!("[translate] median displacement = ({med_dx}, {med_dy}), expected ({DX}, {DY})");
    assert!(
        (med_dx - DX).abs() <= 2,
        "median dx {med_dx} not within ±2 of {DX}"
    );
    assert!(
        (med_dy - DY).abs() <= 2,
        "median dy {med_dy} not within ±2 of {DY}"
    );
}

/// 4. Self-match identity.
///
/// Matching an image's descriptors against themselves must put a large fraction
/// of matches at Hamming distance 0 and displacement (0, 0): each descriptor's
/// nearest neighbour is itself.
#[test]
fn self_match_is_identity() {
    let (w, h) = (1280, 720);
    let img = synth_textured_image(w, h);

    let detector = orb();
    let (kps, desc) = detector
        .detect_and_compute(&img, w, h)
        .expect("detect_and_compute");
    assert!(
        kps.len() >= 50,
        "need a healthy keypoint set, got {}",
        kps.len()
    );

    // A permissive ratio threshold: on a periodic texture some descriptors have
    // a near-tie second neighbour, so a strict ratio test would reject the
    // (correct) zero-distance self-match. We only need the identity structure.
    let matcher = FeatureMatcher::new(50, 0.99);
    let matches = matcher.match_features(&kps, &desc, &kps, &desc);

    let zero_dist = matches.iter().filter(|m| m.distance == 0).count();
    let zero_disp = matches
        .iter()
        .filter(|m| {
            (m.point2.x - m.point1.x).abs() < 1e-9 && (m.point2.y - m.point1.y).abs() < 1e-9
        })
        .count();

    eprintln!(
        "[self] {} matches, {} at distance 0, {} at displacement (0,0)",
        matches.len(),
        zero_dist,
        zero_disp
    );

    assert!(!matches.is_empty(), "self-match produced no matches");
    // "Large fraction" — at least half the matches are exact self-hits.
    assert!(
        zero_dist * 2 >= matches.len(),
        "expected ≥50% zero-distance self-matches, got {zero_dist}/{}",
        matches.len()
    );
    assert!(
        zero_disp * 2 >= matches.len(),
        "expected ≥50% zero-displacement self-matches, got {zero_disp}/{}",
        matches.len()
    );
}

/// 5. FAST count ≥ ORB retained count at the same resolution.
///
/// ORB runs FAST then sorts by response and truncates to `max_features`, so the
/// raw FAST keypoint count is an upper bound on the ORB-retained count. (With a
/// huge `max_features` cap there is no truncation, so the two are equal; the
/// invariant is `FAST >= ORB`.)
#[test]
fn fast_count_ge_orb_retained_count() {
    let (w, h) = (1280, 720);
    let img = synth_textured_image(w, h);

    let fast = FastDetector::new(FAST_THRESHOLD);
    let fast_kps = fast.detect(&img, w, h).expect("FAST detect");

    // OrbDetector uses the DEFAULT FastDetector internally (threshold 20), so
    // compare against an ORB whose internal FAST is at least as permissive.
    // The invariant FAST_retained-upper-bound >= ORB_retained holds regardless,
    // because ORB only ever truncates the FAST set, never grows it.
    let detector = OrbDetector::new(50_000);
    let (orb_kps, _d) = detector
        .detect_and_compute(&img, w, h)
        .expect("ORB detect_and_compute");

    eprintln!(
        "[fast>=orb] FAST(thr={FAST_THRESHOLD})={}, ORB-retained={}",
        fast_kps.len(),
        orb_kps.len()
    );

    // ORB's internal FAST (threshold 20) yields the candidate set it truncates.
    // To make the "FAST >= ORB" bound self-contained, also detect with ORB's
    // exact internal threshold and confirm it bounds the retained count.
    let fast_default = FastDetector::new(20);
    let fast_default_kps = fast_default.detect(&img, w, h).expect("FAST(20) detect");
    assert!(
        fast_default_kps.len() >= orb_kps.len(),
        "ORB-retained ({}) exceeds its own FAST candidate count ({}) — impossible",
        orb_kps.len(),
        fast_default_kps.len()
    );
    assert!(
        !fast_kps.is_empty(),
        "FAST(thr={FAST_THRESHOLD}) found no corners on a textured image"
    );
}

/// 6. Timing recorded, not asserted (liveness only).
///
/// Wrap each resolution's ORB detection in `Instant`, print the elapsed time
/// (visible with `--nocapture`), and assert ONLY that it finishes under the
/// generous liveness ceiling — a non-flaky progress check, never a speed gate.
///
/// VGA / HD / 2K run unconditionally; the 4K case is split into its own
/// `#[ignore]`d test so a slow debug build cannot make the always-on suite
/// flaky, while still being runnable on demand.
#[test]
fn timing_liveness_vga_hd_2k() {
    let detector = orb();
    for &(w, h, label) in RESOLUTIONS.iter().take(3) {
        let img = synth_textured_image(w, h);
        let t = Instant::now();
        let (kps, desc) = detector
            .detect_and_compute(&img, w, h)
            .unwrap_or_else(|e| panic!("{label}: detect_and_compute failed: {e}"));
        let elapsed = t.elapsed();
        eprintln!(
            "[timing] {label:>3} {w}x{h}: {:?} ({} kps, {} desc)",
            elapsed,
            kps.len(),
            desc.len()
        );
        assert!(
            elapsed.as_secs() < LIVENESS_LIMIT_SECS,
            "{label}: detection took {elapsed:?}, exceeds {LIVENESS_LIMIT_SECS}s liveness limit"
        );
        assert!(!kps.is_empty(), "{label}: no keypoints");
    }
}

/// 6b. 4K full-sweep coverage — `#[ignore]`d to keep the always-on suite fast in
/// debug builds. Run with `cargo test -- --ignored` (or `--nocapture`) to see
/// the 4K timing and exercise the un-clamped large-image path.
///
/// This is where the always-on `keypoint_count_scales_with_resolution` and
/// `descriptor_count_matches_keypoint_count` tests (which stop at 2K for speed)
/// get their 4K coverage: it repeats the `kps.len() == desc.len()` and no-clamp
/// invariants AND asserts the 2K → 4K leg of the strict monotonic growth.
#[test]
#[ignore = "4K is slow in debug builds; run with --ignored for the full sweep"]
fn timing_liveness_4k() {
    let (w2k, h2k, _) = RESOLUTIONS[2];
    let (w, h, label) = RESOLUTIONS[3];
    assert_eq!(label, "4K");
    let detector = orb();

    // 2K reference count for the 2K → 4K monotonicity leg.
    let img2k = synth_textured_image(w2k, h2k);
    let (kps2k, _) = detector
        .detect_and_compute(&img2k, w2k, h2k)
        .expect("2K detect_and_compute");

    let img = synth_textured_image(w, h);
    let t = Instant::now();
    let (kps, desc): (_, Vec<BinaryDescriptor>) = detector
        .detect_and_compute(&img, w, h)
        .expect("4K detect_and_compute");
    let elapsed = t.elapsed();
    eprintln!(
        "[timing] {label} {w}x{h}: {:?} ({} kps, {} desc); 2K had {} kps",
        elapsed,
        kps.len(),
        desc.len(),
        kps2k.len()
    );
    assert_eq!(kps.len(), desc.len(), "4K: desc/kp count mismatch");
    assert!(
        kps.len() < detector.max_features,
        "4K: keypoints ({}) clamped to max_features ({})",
        kps.len(),
        detector.max_features
    );
    assert!(
        kps.len() > kps2k.len(),
        "4K keypoints ({}) must strictly exceed 2K keypoints ({})",
        kps.len(),
        kps2k.len()
    );
    assert!(
        elapsed.as_secs() < LIVENESS_LIMIT_SECS,
        "4K: detection took {elapsed:?}, exceeds {LIVENESS_LIMIT_SECS}s liveness limit"
    );
}
