//! Integration tests for `perceptual_hash` robustness across common image transforms.
//!
//! Generates synthetic 64×64 grayscale gradient images and verifies that the
//! perceptual hash Hamming distance remains ≤ 15 after resize simulation,
//! brightness adjustments, and crop/pad operations.

use oximedia_dedup::perceptual_hash::compute_dhash;

/// Generate a 64×64 grayscale gradient image as raw pixel bytes.
/// Pixel[y][x] = ((x + y) % 256) as u8  — a smooth diagonal ramp.
fn make_gradient_64x64() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64);
    for y in 0..64usize {
        for x in 0..64usize {
            pixels.push(((x + y) % 256) as u8);
        }
    }
    pixels
}

/// Simulate a resize by down-sampling a 64×64 image to 56×56 (nearest-neighbour)
/// then padding back to 64×64 with the edge value.
fn simulate_resize(src: &[u8], src_w: usize, src_h: usize) -> Vec<u8> {
    let dst_w = 56usize;
    let dst_h = 56usize;
    // Step 1: nearest-neighbour down-sample to 56×56
    let mut small = Vec::with_capacity(dst_w * dst_h);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = dx * src_w / dst_w;
            let sy = dy * src_h / dst_h;
            small.push(src[sy * src_w + sx]);
        }
    }
    // Step 2: nearest-neighbour up-sample back to 64×64
    let mut resized = Vec::with_capacity(src_w * src_h);
    for y in 0..src_h {
        for x in 0..src_w {
            let sx = x * dst_w / src_w;
            let sy = y * dst_h / src_h;
            let idx = sy.min(dst_h - 1) * dst_w + sx.min(dst_w - 1);
            resized.push(small[idx]);
        }
    }
    resized
}

/// Apply a brightness offset (+/- 10) with saturation at [0, 255].
fn apply_brightness(src: &[u8], delta: i16) -> Vec<u8> {
    src.iter()
        .map(|&p| {
            let v = p as i16 + delta;
            v.clamp(0, 255) as u8
        })
        .collect()
}

/// Simulate a centre crop: take the inner 48×48 region and up-sample it back
/// to 64×64 using nearest-neighbour interpolation.
fn simulate_centre_crop(src: &[u8], src_w: usize, src_h: usize) -> Vec<u8> {
    let crop_w = 48usize;
    let crop_h = 48usize;
    let ox = (src_w - crop_w) / 2;
    let oy = (src_h - crop_h) / 2;

    // Extract the crop
    let mut crop = Vec::with_capacity(crop_w * crop_h);
    for y in 0..crop_h {
        for x in 0..crop_w {
            crop.push(src[(oy + y) * src_w + (ox + x)]);
        }
    }

    // Up-sample crop back to 64×64 using nearest-neighbour
    let mut result = Vec::with_capacity(src_w * src_h);
    for y in 0..src_h {
        for x in 0..src_w {
            let sx = x * crop_w / src_w;
            let sy = y * crop_h / src_h;
            let idx = sy.min(crop_h - 1) * crop_w + sx.min(crop_w - 1);
            result.push(crop[idx]);
        }
    }
    result
}

#[test]
fn test_phash_robustness_resize() {
    let original = make_gradient_64x64();
    let transformed = simulate_resize(&original, 64, 64);

    let hash_original = compute_dhash(&original, 64, 64);
    let hash_transformed = compute_dhash(&transformed, 64, 64);

    let dist = hash_original.hamming_distance(&hash_transformed);
    assert!(
        dist <= 15,
        "resize: Hamming distance {dist} > 15 — perceptual hash too sensitive to resize"
    );
}

#[test]
fn test_phash_robustness_brightness_increase() {
    let original = make_gradient_64x64();
    let brighter = apply_brightness(&original, 10);

    let hash_original = compute_dhash(&original, 64, 64);
    let hash_brighter = compute_dhash(&brighter, 64, 64);

    let dist = hash_original.hamming_distance(&hash_brighter);
    assert!(
        dist <= 15,
        "brightness +10: Hamming distance {dist} > 15 — hash too sensitive to brightness"
    );
}

#[test]
fn test_phash_robustness_brightness_decrease() {
    let original = make_gradient_64x64();
    let darker = apply_brightness(&original, -10);

    let hash_original = compute_dhash(&original, 64, 64);
    let hash_darker = compute_dhash(&darker, 64, 64);

    let dist = hash_original.hamming_distance(&hash_darker);
    assert!(
        dist <= 15,
        "brightness -10: Hamming distance {dist} > 15 — hash too sensitive to darkness"
    );
}

#[test]
fn test_phash_robustness_centre_crop() {
    let original = make_gradient_64x64();
    let cropped = simulate_centre_crop(&original, 64, 64);

    let hash_original = compute_dhash(&original, 64, 64);
    let hash_cropped = compute_dhash(&cropped, 64, 64);

    let dist = hash_original.hamming_distance(&hash_cropped);
    assert!(
        dist <= 15,
        "centre crop: Hamming distance {dist} > 15 — hash too sensitive to centre crop"
    );
}

#[test]
fn test_phash_identical_images_zero_distance() {
    let original = make_gradient_64x64();
    let copy = original.clone();

    let h1 = compute_dhash(&original, 64, 64);
    let h2 = compute_dhash(&copy, 64, 64);

    assert_eq!(
        h1.hamming_distance(&h2),
        0,
        "identical images must produce identical hashes (Hamming = 0)"
    );
}

#[test]
fn test_phash_completely_different_images_large_distance() {
    // A gradient vs its inverse should produce a very different hash.
    let original = make_gradient_64x64();
    let inverted: Vec<u8> = original.iter().map(|&p| 255 - p).collect();

    let h1 = compute_dhash(&original, 64, 64);
    let h2 = compute_dhash(&inverted, 64, 64);

    let dist = h1.hamming_distance(&h2);
    // Inverted gradient should differ significantly (≥ 20 bits for dHash).
    assert!(
        dist >= 20,
        "inverted gradient: Hamming distance {dist} < 20 — hashes are not discriminative enough"
    );
}
