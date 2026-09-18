// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`extract_centerline_approx`]: input validation (it
//! used to loop forever for a non-positive slice spacing and index the mask
//! unchecked), the per-slice grouping and the cross-section radius.

use super::*;
use crate::error::Error;
use std::f64::consts::PI;

/// Mask of a cylinder with integer-radius disc cross-sections around the
/// voxel column `(cx, cy)`, running along z through all `nz` slices.
fn cylinder_mask(nx: usize, ny: usize, nz: usize, cx: usize, cy: usize, radius: i64) -> Vec<bool> {
    let mut mask = vec![false; nx * ny * nz];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dx = ix as i64 - cx as i64;
                let dy = iy as i64 - cy as i64;
                mask[iz * ny * nx + iy * nx + ix] = dx * dx + dy * dy <= radius * radius;
            }
        }
    }
    mask
}

fn disc_voxel_count(radius: i64) -> usize {
    let mut count = 0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn cylinder_mask_gives_axis_points_and_cross_section_radius() {
    let (nx, ny, nz) = (15, 15, 12);
    let mask = cylinder_mask(nx, ny, nz, 7, 7, 5);
    let cl = extract_centerline_approx(&mask, nx, ny, nz, [1.0, 1.0, 1.0], Vec3::zero())
        .expect("valid cylinder mask");
    assert_eq!(cl.points.len(), nz);
    // A digitised disc of radius 5 voxels has 81 voxels: equal-area radius
    // sqrt(81 / pi) = 5.08.
    assert_eq!(disc_voxel_count(5), 81);
    let expected_radius = (81.0 / PI).sqrt();
    for (iz, point) in cl.points.iter().enumerate() {
        assert!(
            (point.radius - 5.0).abs() < 0.5,
            "slice {iz}: radius {}",
            point.radius
        );
        assert!((point.radius - expected_radius).abs() < 1e-12);
        assert!((point.position.x - 7.0).abs() < 1e-12);
        assert!((point.position.y - 7.0).abs() < 1e-12);
        assert!((point.position.z - iz as f64).abs() < 1e-12);
    }
    assert!((cl.arc_length() - 11.0).abs() < 1e-9);
    assert!((cl.tortuosity() - 1.0).abs() < 1e-12);
}

#[test]
fn anisotropic_voxels_and_origin_scale_positions_and_radius() {
    let (nx, ny, nz) = (15, 15, 4);
    let mask = cylinder_mask(nx, ny, nz, 7, 7, 5);
    let origin = Vec3::new(-10.0, 20.0, 5.0);
    let cl =
        extract_centerline_approx(&mask, nx, ny, nz, [0.5, 0.5, 2.0], origin).expect("valid mask");
    assert_eq!(cl.points.len(), 4);
    for (iz, point) in cl.points.iter().enumerate() {
        assert!((point.position.x - (-10.0 + 7.0 * 0.5)).abs() < 1e-12);
        assert!((point.position.y - (20.0 + 7.0 * 0.5)).abs() < 1e-12);
        assert!((point.position.z - (5.0 + 2.0 * iz as f64)).abs() < 1e-12);
        // Radius in millimetres: 5.08 voxels of 0.5 mm.
        assert!((point.radius - 0.5 * (81.0 / PI).sqrt()).abs() < 1e-12);
    }
    assert!((cl.arc_length() - 6.0).abs() < 1e-9);
}

#[test]
fn non_positive_or_non_finite_voxel_size_is_rejected_without_hanging() {
    let mask = cylinder_mask(5, 5, 5, 2, 2, 1);
    let origin = Vec3::zero();
    let cases: [([f64; 3], &str); 7] = [
        ([1.0, 1.0, 0.0], "voxel_size[2]"),
        ([1.0, 1.0, -2.0], "voxel_size[2]"),
        ([1.0, 1.0, f64::NAN], "voxel_size[2]"),
        ([0.0, 1.0, 1.0], "voxel_size[0]"),
        ([1.0, f64::NEG_INFINITY, 1.0], "voxel_size[1]"),
        ([f64::INFINITY, 1.0, 1.0], "voxel_size[0]"),
        ([1.0, -0.0, 1.0], "voxel_size[1]"),
    ];
    for (voxel_size, expected_name) in cases {
        match extract_centerline_approx(&mask, 5, 5, 5, voxel_size, origin) {
            Err(Error::OutOfRange { name, .. }) => {
                assert_eq!(name, expected_name, "{voxel_size:?}");
            }
            other => panic!("{voxel_size:?}: expected OutOfRange, got {other:?}"),
        }
    }
}

#[test]
fn mask_length_must_match_the_volume_dimensions() {
    let origin = Vec3::zero();
    let size = [1.0; 3];
    let short = vec![true; 7];
    assert!(matches!(
        extract_centerline_approx(&short, 2, 2, 2, size, origin),
        Err(Error::LengthMismatch {
            expected: 8,
            actual: 7
        })
    ));
    let long = vec![true; 9];
    assert!(matches!(
        extract_centerline_approx(&long, 2, 2, 2, size, origin),
        Err(Error::LengthMismatch {
            expected: 8,
            actual: 9
        })
    ));
    // An empty mask does not match a non-empty volume.
    assert!(matches!(
        extract_centerline_approx(&[], 3, 3, 3, size, origin),
        Err(Error::LengthMismatch { .. })
    ));
}

#[test]
fn oversized_dimensions_are_an_error_not_an_overflow_panic() {
    let r = extract_centerline_approx(&[true], usize::MAX, 2, 1, [1.0; 3], Vec3::zero());
    assert!(matches!(r, Err(Error::General(_))), "{r:?}");
    let half = usize::MAX / 2;
    let r = extract_centerline_approx(&[true], half, half, half, [1.0; 3], Vec3::zero());
    assert!(matches!(r, Err(Error::General(_))), "{r:?}");
}

#[test]
fn non_finite_origin_is_rejected() {
    let mask = vec![true; 8];
    for origin in [
        Vec3::new(f64::NAN, 0.0, 0.0),
        Vec3::new(0.0, f64::INFINITY, 0.0),
        Vec3::new(0.0, 0.0, f64::NEG_INFINITY),
    ] {
        assert!(matches!(
            extract_centerline_approx(&mask, 2, 2, 2, [1.0; 3], origin),
            Err(Error::General(_))
        ));
    }
}

#[test]
fn empty_slices_are_skipped_and_slices_stay_ordered() {
    let (nx, ny, nz) = (4, 3, 6);
    let mut mask = vec![false; nx * ny * nz];
    // Foreground only in slices 1 and 4 (and a second blob in slice 4).
    mask[ny * nx + nx + 1] = true; // slice 1, voxel (1, 1)
    mask[4 * ny * nx + 2] = true; // slice 4, voxel (2, 0)
    mask[4 * ny * nx + 2 * nx + 2] = true; // slice 4, voxel (2, 2)
    let origin = Vec3::new(1.0, 2.0, 3.0);
    let cl = extract_centerline_approx(&mask, nx, ny, nz, [2.0, 3.0, 0.25], origin)
        .expect("sparse mask");
    assert_eq!(cl.points.len(), 2);

    let first = cl.points[0];
    assert!((first.position.x - (1.0 + 1.0 * 2.0)).abs() < 1e-12);
    assert!((first.position.y - (2.0 + 1.0 * 3.0)).abs() < 1e-12);
    assert!((first.position.z - (3.0 + 0.25)).abs() < 1e-12);
    // A single voxel reads its equal-area radius, not the voxel edge length.
    assert!((first.radius - (2.0 * 3.0 / PI).sqrt()).abs() < 1e-12);

    let second = cl.points[1];
    assert!((second.position.x - (1.0 + 2.0 * 2.0)).abs() < 1e-12);
    assert!((second.position.y - (2.0 + 1.0 * 3.0)).abs() < 1e-12);
    assert!((second.position.z - (3.0 + 4.0 * 0.25)).abs() < 1e-12);
    assert!((second.radius - (2.0 * 2.0 * 3.0 / PI).sqrt()).abs() < 1e-12);
}

#[test]
fn off_axis_blob_centroid_is_exact_for_every_slice() {
    // A 3 x 2 block at x in 2..5, y in 1..3, present in every slice.
    let (nx, ny, nz) = (8, 6, 5);
    let mut mask = vec![false; nx * ny * nz];
    for iz in 0..nz {
        for iy in 1..3 {
            for ix in 2..5 {
                mask[iz * ny * nx + iy * nx + ix] = true;
            }
        }
    }
    let cl =
        extract_centerline_approx(&mask, nx, ny, nz, [1.0, 1.0, 1.0], Vec3::zero()).expect("block");
    assert_eq!(cl.points.len(), nz);
    for (iz, point) in cl.points.iter().enumerate() {
        assert!((point.position.x - 3.0).abs() < 1e-12);
        assert!((point.position.y - 1.5).abs() < 1e-12);
        assert!((point.position.z - iz as f64).abs() < 1e-12);
        assert!((point.radius - (6.0 / PI).sqrt()).abs() < 1e-12);
    }
}

#[test]
fn degenerate_volume_dimensions_give_an_empty_centerline() {
    let size = [1.0; 3];
    for (nx, ny, nz) in [(0, 4, 4), (4, 0, 4), (4, 4, 0)] {
        let cl =
            extract_centerline_approx(&[], nx, ny, nz, size, Vec3::zero()).expect("empty volume");
        assert!(cl.points.is_empty());
    }
}
