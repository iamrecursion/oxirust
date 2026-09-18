//! Cross-frame BRIEF descriptor-cache behaviour for temporal feature tracking.
//!
//! These tests pin the REAL behaviour of [`DescriptorCache`] against the public
//! API only, so they live in `tests/` and keep `src/feature_cache.rs` small.
//!
//! The synthetic image generator and integer translator mirror the ones used by
//! `fast_orb_resolution.rs`: the texture is a pure function of the absolute pixel
//! coordinate `(x, y)`, so a rigid translation reproduces every feature's local
//! neighbourhood VERBATIM. That is exactly what makes the cache's
//! stationary-overlap reuse meaningful — within the un-shifted region a keypoint
//! reappears at (almost) the same coordinate with an identical local patch, so
//! its previously-cached descriptor is the correct one to reuse.
//!
//! Coverage:
//!   1. same-frame re-query  ⇒ pure cache hits; descriptors identical to uncached.
//!   2. 8 px global translation ⇒ stationary-overlap region reuses ≥ 80%.
//!   3. 11 frames, max_frames = 10 ⇒ oldest evicted; capacity respected.
//!   4. miss descriptors are bit-exact vs the uncached `detect_and_compute`.

use oximedia_align::feature_cache::DescriptorCache;
use oximedia_align::features::{BinaryDescriptor, Keypoint, OrbDetector};

/// Feature-lattice pitch in PIXELS — fixed, so corner density is constant per
/// unit area. Mirrors the proven value from `fast_orb_resolution.rs`.
const FEATURE_PITCH: usize = 64;

/// Deterministic 32-bit hash of a lattice cell's `(cx, cy)` coordinate. Seeds an
/// inline LCG so each cell gets a UNIQUE local appearance (and therefore a
/// discriminative BRIEF descriptor). Splitmix-style finaliser; no dev-deps.
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

/// Generate a deterministic, strongly-textured grayscale image whose features
/// are LOCALLY UNIQUE (discriminative BRIEF) and whose appearance is a pure
/// function of the absolute pixel coordinate (so a rigid translation reproduces
/// every local neighbourhood verbatim).
fn synth_textured_image(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![128u8; w * h];
    let cells_x = w / FEATURE_PITCH;
    let cells_y = h / FEATURE_PITCH;

    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let node_x = cx * FEATURE_PITCH;
            let node_y = cy * FEATURE_PITCH;
            let mut rng = cell_seed(cx, cy);

            let blob_count = 3 + (lcg_step(&mut rng) % 3) as usize;
            for _ in 0..blob_count {
                let span = FEATURE_PITCH - 20;
                let bx = node_x + 8 + (lcg_step(&mut rng) % span as u32) as usize;
                let by = node_y + 8 + (lcg_step(&mut rng) % span as u32) as usize;
                let size = 3 + (lcg_step(&mut rng) % 4) as usize;
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

/// A detector sized so a busy texture is not clamped by truncation.
fn detector() -> OrbDetector {
    OrbDetector::new(50_000)
}

/// 1. Re-querying the IDENTICAL frame yields pure cache hits.
///
/// Frame 0 is all misses (empty cache). Re-querying the SAME image as frame 1
/// finds, for every keypoint, an exact-coordinate match in frame 0 (distance 0
/// ≤ tol), so every descriptor is a hit. The returned keypoints/descriptors are
/// still identical to the uncached `detect_and_compute` (a hit reuses frame 0's
/// descriptor, which for the identical image equals the fresh one).
#[test]
fn same_frame_requery_is_all_hits() {
    let (w, h) = (640, 480);
    let img = synth_textured_image(w, h);

    let mut cache = DescriptorCache::new(detector(), 10, 1.5);

    // Frame 0 — empty cache ⇒ all misses.
    let f0 = cache
        .detect_and_compute_cached(&img, w, h, 0)
        .expect("frame 0 cached detect");
    assert!(!f0.is_empty(), "expected keypoints on a textured image");
    assert_eq!(cache.hits(), 0, "frame 0 must be all misses");
    let frame0_misses = cache.misses();
    assert_eq!(frame0_misses as usize, f0.len());

    // Frame 1 — identical image ⇒ every keypoint hits its frame-0 twin.
    let f1 = cache
        .detect_and_compute_cached(&img, w, h, 1)
        .expect("frame 1 cached detect");
    assert_eq!(f0.len(), f1.len(), "same image ⇒ same keypoint count");

    let frame1_hits = cache.hits();
    let frame1_misses = cache.misses() - frame0_misses;
    eprintln!(
        "[same-frame] frame1: {} hits, {} misses over {} keypoints",
        frame1_hits,
        frame1_misses,
        f1.len()
    );
    assert_eq!(
        frame1_hits as usize,
        f1.len(),
        "re-querying the identical frame must be ALL hits"
    );
    assert_eq!(
        frame1_misses, 0,
        "re-querying the identical frame: zero misses"
    );

    // The cached path's keypoints/descriptors equal the uncached path exactly.
    let (kp_ref, desc_ref) = detector()
        .detect_and_compute(&img, w, h)
        .expect("uncached detect_and_compute");
    assert_eq!(kp_ref.len(), f1.len());
    for (i, ((kp, desc), (kpr, descr))) in f1
        .iter()
        .zip(kp_ref.iter().zip(desc_ref.iter()))
        .enumerate()
    {
        assert!(
            keypoints_equal(kp, kpr),
            "keypoint {i} differs from uncached path"
        );
        assert_eq!(
            desc.data, descr.data,
            "descriptor {i} differs from uncached path on a same-image hit"
        );
    }
}

/// 2. A slow 8 px pan reuses ≥ 80% of descriptors over the stationary overlap.
///
/// A keypoint that drifts by 1 px per frame moves √2 ≈ 1.41 px between frames —
/// within the 1.5 px tolerance — so on the stationary-overlap region (the vast
/// bulk of a 1280×720 frame) its descriptor from the previous frame is reused.
/// Frame 0 primes the cache (all misses, discounted via `reset_stats`); frames
/// 1..=8 each shift the base image by one more pixel, accumulating to an 8 px
/// total displacement while every consecutive pair stays within tolerance. The
/// texture is translation-invariant, so each reused descriptor is the correct
/// one for the re-detected keypoint.
#[test]
fn small_per_frame_translation_reuses_overlap() {
    let (w, h) = (1280, 720);
    let base = synth_textured_image(w, h);

    // tol = 1.5 px so a 1 px-per-frame drift stays within tolerance frame to
    // frame, while the texture's translation-invariance makes the reused
    // descriptor the correct one.
    let mut cache = DescriptorCache::new(detector(), 10, 1.5);

    // Prime with frame 0 (no shift) — all misses.
    let _ = cache
        .detect_and_compute_cached(&base, w, h, 0)
        .expect("frame 0");
    let baseline_misses = cache.misses();
    cache.reset_stats();

    // Frames 1..=8: shift by 1 px per frame, up to 8 px total.
    for step in 1..=8isize {
        let shifted = translate_image(&base, w, h, step, step);
        let feats = cache
            .detect_and_compute_cached(&shifted, w, h, step as u64)
            .unwrap_or_else(|e| panic!("frame {step}: {e}"));
        assert!(!feats.is_empty(), "frame {step}: no keypoints");
    }

    let hit_ratio = cache.hit_ratio();
    eprintln!(
        "[translate] baseline misses (frame 0) = {}, steady-state hit_ratio = {:.3} \
         ({} hits / {} misses)",
        baseline_misses,
        hit_ratio,
        cache.hits(),
        cache.misses()
    );

    // Across the eight 1-px steps the stationary-overlap region (the vast bulk of
    // a 1280×720 frame) re-detects its keypoints within tolerance of the prior
    // frame, so reuse dominates.
    assert!(
        hit_ratio >= 0.80,
        "expected ≥80% descriptor reuse on a slow pan, got {:.3}",
        hit_ratio
    );
}

/// 3. Capacity is respected and the oldest frame is FIFO-evicted.
///
/// Feed 11 distinct frames into a cache with `max_frames = 10`. After each
/// insert `len()` is bounded by 10, and once 11 frames have been seen the oldest
/// (frame 0) is gone while the newest (frame 10) is retained.
#[test]
fn capacity_respected_and_oldest_evicted() {
    let (w, h) = (320, 240);
    let mut cache = DescriptorCache::new(detector(), 10, 1.5);

    for fid in 0..11u64 {
        // Distinct per-frame translation so frames are genuinely different.
        let shifted = translate_image(
            &synth_textured_image(w, h),
            w,
            h,
            fid as isize,
            (fid % 3) as isize,
        );
        let feats = cache
            .detect_and_compute_cached(&shifted, w, h, fid)
            .unwrap_or_else(|e| panic!("frame {fid}: {e}"));
        assert!(!feats.is_empty(), "frame {fid}: no keypoints");
        assert!(
            cache.len() <= 10,
            "frame {fid}: cache len {} exceeds max_frames 10",
            cache.len()
        );
    }

    assert_eq!(cache.len(), 10, "cache should be full at capacity");
    let ids = cache.cached_frame_ids();
    eprintln!("[capacity] retained frame ids = {ids:?}");
    assert_eq!(
        ids,
        (1..=10).collect::<Vec<u64>>(),
        "oldest (0) must be evicted, newest (10) kept"
    );
    assert!(!ids.contains(&0), "frame 0 must have been evicted");
    assert!(ids.contains(&10), "frame 10 (newest) must be retained");
}

/// 4. On a miss the cached path is BIT-EXACT vs uncached `detect_and_compute`.
///
/// With a fresh (empty) cache, EVERY keypoint of the first queried frame is a
/// miss, so the cached path must reproduce both the keypoint set and the
/// descriptor bytes of `OrbDetector::detect_and_compute` exactly.
#[test]
fn miss_path_is_bit_exact_with_uncached() {
    let (w, h) = (800, 600);
    let img = synth_textured_image(w, h);

    let mut cache = DescriptorCache::new(detector(), 10, 1.5);
    let cached = cache
        .detect_and_compute_cached(&img, w, h, 0)
        .expect("cached detect");

    // Empty cache ⇒ every keypoint is a miss.
    assert_eq!(cache.hits(), 0, "fresh cache must produce only misses");
    assert_eq!(cache.misses() as usize, cached.len());

    let (kp_ref, desc_ref) = detector()
        .detect_and_compute(&img, w, h)
        .expect("uncached detect_and_compute");

    assert_eq!(
        cached.len(),
        kp_ref.len(),
        "cached miss path produced a different keypoint count than uncached"
    );
    assert_eq!(kp_ref.len(), desc_ref.len());

    for (i, ((kp, desc), (kpr, descr))) in cached
        .iter()
        .zip(kp_ref.iter().zip(desc_ref.iter()))
        .enumerate()
    {
        assert!(
            keypoints_equal(kp, kpr),
            "keypoint {i}: cached {:?} != uncached {:?}",
            (kp.point.x, kp.point.y, kp.response),
            (kpr.point.x, kpr.point.y, kpr.response)
        );
        assert_eq!(
            desc.data, descr.data,
            "descriptor {i}: cached MISS bytes differ from uncached detect_and_compute"
        );
    }
}

/// Exact keypoint equality on the fields `detect_and_compute` fills in
/// deterministically (position, scale, orientation, response).
fn keypoints_equal(a: &Keypoint, b: &Keypoint) -> bool {
    a.point.x.to_bits() == b.point.x.to_bits()
        && a.point.y.to_bits() == b.point.y.to_bits()
        && a.scale.to_bits() == b.scale.to_bits()
        && a.orientation.to_bits() == b.orientation.to_bits()
        && a.response.to_bits() == b.response.to_bits()
}

/// Sanity: a `BinaryDescriptor` round-trips its bytes (guards the test's
/// reliance on the `data` field being the full descriptor).
#[test]
fn descriptor_bytes_roundtrip() {
    let mut bytes = [0u8; 32];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i * 7 + 1) as u8;
    }
    let desc = BinaryDescriptor::new(bytes);
    assert_eq!(desc.data, bytes);
}
