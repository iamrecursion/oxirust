// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the metric of [`super::distance_transform`] (it is
//! Euclidean, not city-block), the explicit city-block variant
//! [`super::distance_transform_cityblock`] and the consumers of the transform
//! ([`super::medial_axis_transform`], [`super::reconstruct_from_mat`]).

use super::{
    BinaryImage, MatPoint, distance_transform, distance_transform_cityblock,
    distance_transform_exact, medial_axis_transform, reconstruct_from_mat,
};

/// Deterministic xorshift64* generator so the random images are reproducible.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random_image(rng: &mut XorShift, w: usize, h: usize, fg_per_mille: u64) -> BinaryImage {
    let mut img = BinaryImage::new(w, h, false);
    for r in 0..h {
        for c in 0..w {
            img.set(c, r, rng.next_u64() % 1000 < fg_per_mille);
        }
    }
    img
}

/// Brute-force distance to the nearest background pixel under `metric` (a
/// function of the integer pixel offsets); `f64::INFINITY` everywhere when the
/// image has no background pixel.
fn brute_force(img: &BinaryImage, metric: impl Fn(i64, i64) -> f64) -> Vec<f64> {
    let (w, h) = (img.width, img.height);
    let background: Vec<(i64, i64)> = (0..h)
        .flat_map(|r| (0..w).map(move |c| (c, r)))
        .filter(|&(c, r)| !img.get(c, r))
        .map(|(c, r)| (c as i64, r as i64))
        .collect();
    let mut out = vec![0.0; w * h];
    for r in 0..h {
        for c in 0..w {
            out[r * w + c] = background
                .iter()
                .map(|&(bc, br)| metric(bc - c as i64, br - r as i64))
                .fold(f64::INFINITY, f64::min);
        }
    }
    out
}

fn euclid(dx: i64, dy: i64) -> f64 {
    ((dx * dx + dy * dy) as f64).sqrt()
}

fn manhattan(dx: i64, dy: i64) -> f64 {
    (dx.abs() + dy.abs()) as f64
}

/// 21x21 foreground image with a single background pixel at `(10, 10)`.
fn bullseye_image() -> BinaryImage {
    let mut img = BinaryImage::new(21, 21, true);
    img.set(10, 10, false);
    img
}

#[test]
fn distance_transform_is_euclidean_not_manhattan() {
    let img = bullseye_image();
    let expected_corner = 10.0 * 2.0_f64.sqrt();
    for (name, dt) in [
        ("distance_transform", distance_transform(&img)),
        ("distance_transform_exact", distance_transform_exact(&img)),
    ] {
        assert!(
            (dt[0] - expected_corner).abs() < 1e-9,
            "{name}: corner reads {} instead of 10*sqrt(2) = {expected_corner}",
            dt[0]
        );
        assert!((dt[21 * 21 - 1] - expected_corner).abs() < 1e-9, "{name}");
        assert_eq!(dt[10 * 21 + 10], 0.0, "{name}: background pixel");
        assert!((dt[10] - 10.0).abs() < 1e-12, "{name}: straight above");
        assert!((dt[10 * 21] - 10.0).abs() < 1e-12, "{name}: straight left");
    }
}

#[test]
fn distance_transform_corner_example_reads_three_root_two() {
    // The documented example: with background only at (0, 0) the pixel (3, 3)
    // is 3*sqrt(2) = 4.243 away, not 6.
    let mut img = BinaryImage::new(7, 7, true);
    img.set(0, 0, false);
    let dt = distance_transform(&img);
    assert!((dt[3 * 7 + 3] - 18.0_f64.sqrt()).abs() < 1e-12);
    let l1 = distance_transform_cityblock(&img);
    assert_eq!(l1[3 * 7 + 3], 6.0);
}

#[test]
fn distance_transform_matches_euclidean_brute_force() {
    let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
    for &(w, h) in &[(16_usize, 16_usize), (9, 14), (14, 9), (1, 8), (8, 1)] {
        for density in [400_u64, 800, 950] {
            let mut img = random_image(&mut rng, w, h, density);
            img.set(w / 2, h / 2, false);
            let dt = distance_transform(&img);
            let reference = brute_force(&img, euclid);
            assert_eq!(dt.len(), reference.len());
            for (idx, (a, b)) in dt.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-12,
                    "pixel {idx} of {w}x{h}: {a} vs brute force {b}"
                );
            }
        }
    }
}

#[test]
fn cityblock_matches_manhattan_brute_force() {
    let mut rng = XorShift(0x0BAD_5EED_1234_5678);
    for &(w, h) in &[(16_usize, 16_usize), (11, 6), (6, 11), (1, 7), (7, 1)] {
        for density in [400_u64, 800, 950] {
            let mut img = random_image(&mut rng, w, h, density);
            img.set(w / 2, h / 2, false);
            let dt = distance_transform_cityblock(&img);
            let reference = brute_force(&img, manhattan);
            for (idx, (a, b)) in dt.iter().zip(reference.iter()).enumerate() {
                assert_eq!(a, b, "pixel {idx} of {w}x{h}");
            }
        }
    }
    let l1 = distance_transform_cityblock(&bullseye_image());
    assert_eq!(l1[0], 20.0, "corner of the bullseye image is 10 + 10 away");
}

#[test]
fn distance_transforms_report_no_distance_without_background() {
    let img = BinaryImage::new(5, 4, true);
    for dt in [
        distance_transform(&img),
        distance_transform_exact(&img),
        distance_transform_cityblock(&img),
    ] {
        assert_eq!(dt.len(), 20);
        assert!(dt.iter().all(|d| d.is_infinite()));
    }
    let empty = BinaryImage::new(0, 3, true);
    assert!(distance_transform(&empty).is_empty());
    assert!(distance_transform_cityblock(&empty).is_empty());
    let all_background = BinaryImage::new(4, 4, false);
    assert!(
        distance_transform_cityblock(&all_background)
            .iter()
            .all(|&d| d == 0.0)
    );
}

/// Filled disc of radius `radius` centred in a `size` x `size` image.
fn disc_image(size: usize, radius: f64) -> BinaryImage {
    let mut img = BinaryImage::new(size, size, false);
    let centre = (size / 2) as f64;
    for r in 0..size {
        for c in 0..size {
            let (dx, dy) = (c as f64 - centre, r as f64 - centre);
            img.set(c, r, dx * dx + dy * dy <= radius * radius);
        }
    }
    img
}

#[test]
fn medial_axis_transform_radii_are_euclidean_distances() {
    let img = disc_image(41, 15.0);
    let mat = medial_axis_transform(&img);
    assert!(!mat.is_empty());
    let euclid_dt = brute_force(&img, euclid);
    let l1_dt = brute_force(&img, manhattan);
    let mut distinguishing_points = 0;
    for mp in &mat {
        let idx = mp.row * img.width + mp.col;
        assert!(
            (mp.radius - euclid_dt[idx]).abs() < 1e-12,
            "MAT pixel ({}, {}): radius {} vs Euclidean {}",
            mp.col,
            mp.row,
            mp.radius,
            euclid_dt[idx]
        );
        if (l1_dt[idx] - euclid_dt[idx]).abs() > 0.5 {
            distinguishing_points += 1;
        }
    }
    // Guard against a vacuous test: some skeleton pixels must have an L1 and
    // an L2 distance that differ clearly.
    assert!(
        distinguishing_points > 0,
        "none of the {} MAT pixels separates the L1 and L2 metrics",
        mat.len()
    );
    // The skeleton of a disc is near its centre where the radius is largest.
    let max_radius = mat.iter().map(|m| m.radius).fold(0.0_f64, f64::max);
    assert!(max_radius > 14.0 && max_radius < 16.0, "{max_radius}");
}

#[test]
fn medial_axis_transform_without_background_has_infinite_radius_and_reconstructs() {
    let img = BinaryImage::new(9, 7, true);
    let mat = medial_axis_transform(&img);
    assert!(!mat.is_empty(), "the border ring survives thinning");
    assert!(mat.iter().all(|m| m.radius.is_infinite()));
    // An infinite radius used to saturate `radius.ceil() as isize` and loop
    // over an astronomically large square; it must terminate and fill the
    // image.
    let recon = reconstruct_from_mat(&mat, 9, 7);
    assert_eq!(recon.foreground_count(), 9 * 7);
}

#[test]
fn reconstruct_from_mat_clips_discs_and_skips_invalid_radii() {
    let point = |col: usize, row: usize, radius: f64| MatPoint { col, row, radius };

    // Disc of radius 2 centred on the corner: the quarter inside the image.
    let recon = reconstruct_from_mat(&[point(0, 0, 2.0)], 6, 6);
    for r in 0..6 {
        for c in 0..6 {
            assert_eq!(recon.get(c, r), c * c + r * r <= 4, "pixel ({c}, {r})");
        }
    }

    // Negative and NaN radii paint nothing; zero paints the centre pixel.
    let recon = reconstruct_from_mat(&[point(2, 2, -1.0), point(3, 3, f64::NAN)], 6, 6);
    assert_eq!(recon.foreground_count(), 0);
    let recon = reconstruct_from_mat(&[point(2, 2, 0.0)], 6, 6);
    assert_eq!(recon.foreground_count(), 1);
    assert!(recon.get(2, 2));

    // Huge finite and infinite radii fill the image without looping forever.
    for radius in [1e18, f64::INFINITY] {
        let recon = reconstruct_from_mat(&[point(1, 4, radius)], 8, 5);
        assert_eq!(recon.foreground_count(), 40, "radius {radius}");
    }

    // A point outside the image paints nothing and does not panic.
    let recon = reconstruct_from_mat(&[point(50, 50, 3.0)], 6, 6);
    assert_eq!(recon.foreground_count(), 0);
}
