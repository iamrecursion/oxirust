//! Regression tests for the medical visualization fixes: dose-volume
//! quantiles, bounds-safe volume sampling, ray clipping and compositing order
//! of the volume renderers, and the table-driven single-cube isosurface.

use super::helpers::ray_unit_cube_interval;
use super::*;
use crate::isosurface::{EDGE_TABLE, TRI_TABLE};
use std::collections::BTreeSet;

// ─────────────────────────────────────────────────────────────────────────────
// DoseDistribution::dose_at_volume
// ─────────────────────────────────────────────────────────────────────────────

/// A dose grid holding `values` along x.
fn dose_grid(values: &[f32]) -> DoseDistribution {
    let mut dose = DoseDistribution::new([values.len(), 1, 1], [1.0; 3], [0.0; 3]);
    for (i, &value) in values.iter().enumerate() {
        dose.set(i, 0, 0, value);
    }
    dose
}

/// Doses 1..=10 Gy in a scrambled voxel order.
const TEN_DOSES: [f32; 10] = [7.0, 2.0, 10.0, 4.0, 1.0, 9.0, 5.0, 3.0, 8.0, 6.0];

#[test]
fn dose_at_volume_is_the_minimum_dose_of_the_hottest_fraction() {
    let dose = dose_grid(&TEN_DOSES);
    // D100 is the minimum dose, D90 leaves out the coolest voxel, D50 is the
    // 5th hottest voxel and D10 the maximum.
    assert_eq!(dose.dose_at_volume(1.0), 1.0);
    assert_eq!(dose.dose_at_volume(0.9), 2.0);
    assert_eq!(dose.dose_at_volume(0.5), 6.0);
    assert_eq!(dose.dose_at_volume(0.1), 10.0);
    // 0.3 * 10 = 3.0000000000000004 and 0.7 * 10 = 7.000000000000001 must not
    // round up to one voxel more.
    assert_eq!(dose.dose_at_volume(0.3), 8.0);
    assert_eq!(dose.dose_at_volume(0.7), 4.0);
    // 25 % of 10 voxels needs ceil(2.5) = 3 voxels: 10, 9 and 8 Gy.
    assert_eq!(dose.dose_at_volume(0.25), 8.0);
}

#[test]
fn dose_at_volume_edge_cases() {
    let dose = dose_grid(&TEN_DOSES);
    assert_eq!(dose.dose_at_volume(0.0), 10.0, "D0 is the maximum dose");
    assert_eq!(dose.dose_at_volume(-0.5), 10.0);
    assert_eq!(
        dose.dose_at_volume(1.5),
        1.0,
        "fractions above 1 count as 1"
    );
    assert_eq!(dose.dose_at_volume(f64::INFINITY), 1.0);
    assert_eq!(dose.dose_at_volume(f64::NAN), 0.0);

    let empty = DoseDistribution::new([0, 0, 0], [1.0; 3], [0.0; 3]);
    assert_eq!(empty.dose_at_volume(0.5), 0.0);

    let single = dose_grid(&[42.0]);
    for fraction in [0.0, 0.3, 1.0] {
        assert_eq!(single.dose_at_volume(fraction), 42.0);
    }
}

#[test]
fn dose_at_volume_is_the_largest_dose_that_covers_the_volume() {
    let grids: [&[f32]; 3] = [
        &TEN_DOSES,
        // Tied doses.
        &[5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        // 13 voxels with fractional doses and a cold background.
        &[
            0.0, 0.0, 60.0, 59.5, 59.5, 30.0, 12.25, 0.0, 45.0, 45.0, 45.0, 1.0, 2.0,
        ],
    ];
    for values in grids {
        let dose = dose_grid(values);
        for step in 1..=100 {
            let fraction = f64::from(step) / 100.0;
            let d = dose.dose_at_volume(fraction);
            // At least `fraction` of the voxels receive `d` or more ...
            assert!(
                dose.volume_at_dose(d) >= fraction - 1e-12,
                "V(D_{fraction}) = {} < {fraction} for {values:?}",
                dose.volume_at_dose(d)
            );
            // ... and no larger dose does.
            assert!(
                dose.volume_at_dose(d + 1e-4) < fraction,
                "D_{fraction} = {d} is not maximal for {values:?}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VolumeData indexing and sampling
// ─────────────────────────────────────────────────────────────────────────────

/// A cubic volume whose value is the z index of the voxel.
fn z_ramp(n: usize) -> VolumeData {
    let mut vol = VolumeData::new([n, n, n], [1.0; 3]);
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                vol.set(ix, iy, iz, iz as f32);
            }
        }
    }
    vol
}

#[test]
fn volume_get_and_set_ignore_positions_past_any_axis() {
    let mut vol = VolumeData::new([3, 4, 5], [1.0; 3]);
    vol.set(2, 3, 4, 7.0);
    assert_eq!(vol.get(2, 3, 4), 7.0);
    // iz == nz used to index past the data.
    assert_eq!(vol.get(0, 0, 5), 0.0);
    assert_eq!(vol.get(0, 0, 500), 0.0);
    assert_eq!(vol.get(3, 0, 0), 0.0);
    assert_eq!(vol.get(0, 4, 0), 0.0);

    let before = vol.data.clone();
    vol.set(0, 0, 5, 9.0);
    vol.set(3, 0, 0, 9.0);
    vol.set(0, 4, 0, 9.0);
    assert_eq!(vol.data, before, "writes outside the volume are ignored");
}

#[test]
fn volume_get_survives_data_shorter_than_dims() {
    let vol = VolumeData {
        data: vec![1.0; 10],
        dims: [4, 4, 4],
        spacing: [1.0; 3],
    };
    assert_eq!(vol.get(1, 1, 0), 1.0);
    assert_eq!(vol.get(3, 3, 3), 0.0);
    // Sampling the far corner reads the missing voxels as zero.
    assert_eq!(vol.sample(1.0, 1.0, 1.0), 0.0);
}

#[test]
fn volume_sample_clamps_to_the_boundary() {
    let vol = z_ramp(4);
    assert!((vol.sample(0.5, 0.5, 0.5) - 1.5).abs() < 1e-6);
    assert!((vol.sample(0.5, 0.5, 1.0) - 3.0).abs() < 1e-6);
    // Past the volume: edge replication instead of an out-of-range index.
    assert!((vol.sample(0.5, 0.5, 2.0) - 3.0).abs() < 1e-6);
    assert!(vol.sample(0.5, 0.5, -1.0).abs() < 1e-6);
    assert!((vol.sample(5.0, 5.0, 0.5) - 1.5).abs() < 1e-6);
    assert!((vol.sample(-5.0, -5.0, 0.5) - 1.5).abs() < 1e-6);
    // A NaN coordinate counts as 0.
    assert!(vol.sample(f32::NAN, 0.5, f32::NAN).abs() < 1e-6);
}

#[test]
fn volume_sample_of_an_empty_or_single_voxel_volume() {
    for dims in [[0, 0, 0], [0, 3, 3], [3, 0, 3], [3, 3, 0]] {
        let empty = VolumeData::new(dims, [1.0; 3]);
        assert_eq!(empty.sample(0.5, 0.5, 0.5), 0.0, "dims {dims:?}");
    }
    let one = VolumeData {
        data: vec![2.5],
        dims: [1, 1, 1],
        spacing: [1.0; 3],
    };
    assert_eq!(one.sample(0.3, 0.9, 0.1), 2.5);
    assert_eq!(one.sample(7.0, -2.0, 0.5), 2.5);
}

#[test]
fn slice_plane_past_the_volume_is_clamped_instead_of_panicking() {
    let vol = z_ramp(4);
    let beyond = SlicePlane::axial(1.5, 4, 4).extract(&vol);
    assert_eq!(beyond.len(), 16);
    assert!(beyond.iter().all(|&p| (p - 3.0).abs() < 1e-6), "{beyond:?}");
    let before = SlicePlane::axial(-0.5, 4, 4).extract(&vol);
    assert!(before.iter().all(|&p| p.abs() < 1e-6), "{before:?}");
}

#[test]
fn volume_gradient_uses_the_true_stencil_width_at_the_boundary() {
    // The value is the x fraction, so the gradient is (1, 0, 0) everywhere.
    let mut vol = VolumeData::new([5, 4, 3], [1.0; 3]);
    for iz in 0..3 {
        for iy in 0..4 {
            for ix in 0..5 {
                vol.set(ix, iy, iz, ix as f32 / 4.0);
            }
        }
    }
    for (fx, fy, fz) in [
        (0.0, 0.5, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.99, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ] {
        let g = vol.gradient(fx, fy, fz, 0.05);
        assert!(
            (g[0] - 1.0).abs() < 1e-3,
            "gx at ({fx}, {fy}, {fz}) = {}",
            g[0]
        );
        assert!(g[1].abs() < 1e-4 && g[2].abs() < 1e-4, "{g:?}");
    }
    // Past the volume the gradient is that of the boundary; a vanishing step
    // gives a zero gradient instead of a division by zero.
    assert_eq!(
        vol.gradient(1.5, 0.5, 0.5, 0.05),
        vol.gradient(1.0, 0.5, 0.5, 0.05)
    );
    assert_eq!(vol.gradient(0.5, 0.5, 0.5, 0.0), [0.0; 3]);
    assert_eq!(vol.gradient(0.5, 0.5, 0.5, f32::NAN), [0.0; 3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray clipping
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ray_unit_cube_interval_clips_rays_against_the_cube() {
    let close = |got: Option<(f32, f32)>, want: (f32, f32)| {
        let (near, far) = got.expect("the ray hits the cube");
        assert!(
            (near - want.0).abs() < 1e-5 && (far - want.1).abs() < 1e-5,
            "got ({near}, {far}), want {want:?}"
        );
    };
    // Straight through the cube from outside.
    close(
        ray_unit_cube_interval([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]),
        (1.0, 2.0),
    );
    // Starting inside.
    close(
        ray_unit_cube_interval([0.5, 0.5, 0.5], [0.0, 0.0, 1.0]),
        (0.0, 0.5),
    );
    // Along the space diagonal.
    let diagonal = 1.0 / 3.0_f32.sqrt();
    close(
        ray_unit_cube_interval([0.0, 0.0, 0.0], [diagonal; 3]),
        (0.0, 3.0_f32.sqrt()),
    );
    // Grazing a cube edge counts as a hit.
    close(
        ray_unit_cube_interval([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
        (1.0, 2.0),
    );
    // Misses.
    assert!(ray_unit_cube_interval([0.5, 0.5, 2.0], [0.0, 0.0, 1.0]).is_none());
    assert!(ray_unit_cube_interval([0.5, 1.5, -1.0], [0.0, 0.0, 1.0]).is_none());
    assert!(ray_unit_cube_interval([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]).is_none());
    assert!(ray_unit_cube_interval([-1.0, 0.5, 0.5], [0.0, 1.0, 0.0]).is_none());
    // Zero direction and non-finite input.
    assert!(ray_unit_cube_interval([0.5; 3], [0.0; 3]).is_none());
    assert!(ray_unit_cube_interval([f32::NAN, 0.5, 0.5], [1.0, 0.0, 0.0]).is_none());
    assert!(ray_unit_cube_interval([0.5; 3], [f32::INFINITY, 0.0, 0.0]).is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// VolumeRenderer
// ─────────────────────────────────────────────────────────────────────────────

/// A transfer function that is `low` for normalised values up to 0.49 and
/// `high` from 0.51 on.
fn two_level_transfer(low: Rgba, high: Rgba) -> ColorTransferFunc {
    let mut transfer = ColorTransferFunc::new();
    transfer.add_point(0.0, low);
    transfer.add_point(0.49, low);
    transfer.add_point(0.51, high);
    transfer.add_point(1.0, high);
    transfer
}

/// An 8^3 volume that is 0 for z < 0.5 and 1 for z >= 0.5.
fn half_filled_volume() -> VolumeData {
    let mut vol = VolumeData::new([8, 8, 8], [1.0; 3]);
    for iz in 0..8 {
        for iy in 0..8 {
            for ix in 0..8 {
                let z = iz as f32 / 7.0;
                vol.set(ix, iy, iz, if z < 0.5 { 0.0 } else { 1.0 });
            }
        }
    }
    vol
}

/// An 8^3 volume that is 1 on the block of voxels 5..8 in every direction
/// (fractional positions from 5/7 to 1) and 0 elsewhere.
fn corner_block_volume() -> VolumeData {
    let mut vol = VolumeData::new([8, 8, 8], [1.0; 3]);
    for iz in 5..8 {
        for iy in 5..8 {
            for ix in 5..8 {
                vol.set(ix, iy, iz, 1.0);
            }
        }
    }
    vol
}

const RED: (f32, f32, f32) = (1.0, 0.0, 0.0);
const BLUE: (f32, f32, f32) = (0.0, 0.0, 1.0);

fn colour(rgb: (f32, f32, f32), alpha: f32) -> Rgba {
    Rgba::new(rgb.0, rgb.1, rgb.2, alpha)
}

#[test]
fn opaque_near_layer_hides_the_far_layer() {
    let vol = half_filled_volume();
    let transfer = two_level_transfer(colour(RED, 1.0), colour(BLUE, 1.0));
    let renderer = VolumeRenderer::new(32);
    let c = renderer.cast_ray(&vol, &transfer, [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]);
    assert!(c.r > 0.5 && c.b < 0.1, "{c:?}");
    // Seen from the other side the blue layer is in front.
    let back = renderer.cast_ray(&vol, &transfer, [0.5, 0.5, 1.0], [0.0, 0.0, -1.0]);
    assert!(back.b > 0.5 && back.r < 0.1, "{back:?}");
}

#[test]
fn near_samples_are_composited_in_front_of_far_samples() {
    // A thin translucent red layer in front of an opaque blue one: the colour
    // keeps the red share of the front layer.  With the far samples drawn over
    // the near ones the result used to be pure blue.
    let vol = half_filled_volume();
    let transfer = two_level_transfer(colour(RED, 0.25), colour(BLUE, 1.0));
    let renderer = VolumeRenderer::new(8);
    let c = renderer.cast_ray(&vol, &transfer, [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]);
    assert!(c.a > 0.99, "the blue layer is opaque: {c:?}");
    assert!(c.r > 0.5, "the red front layer must show: {c:?}");
    assert!(c.b < 0.3, "the blue back layer is mostly hidden: {c:?}");
    assert!(c.r > c.b, "{c:?}");
    // Reversing the ray puts the opaque blue layer in front.
    let back = renderer.cast_ray(&vol, &transfer, [0.5, 0.5, 1.0], [0.0, 0.0, -1.0]);
    assert!(back.b > 0.5 && back.r < 0.1, "{back:?}");
}

#[test]
fn diagonal_ray_reaches_a_feature_in_the_far_corner() {
    // The ray must march along its whole chord (length sqrt(3)), not stop
    // after one cube edge, to reach the block at (0.9, 0.9, 0.9).
    let vol = corner_block_volume();
    let mut transfer = ColorTransferFunc::new();
    transfer.add_point(0.0, Rgba::new(1.0, 1.0, 1.0, 0.0));
    transfer.add_point(1.0, Rgba::new(1.0, 1.0, 1.0, 1.0));
    let renderer = VolumeRenderer::new(64);

    let hit = renderer.cast_ray(&vol, &transfer, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(hit.a > 0.99, "the far-corner block is opaque: {hit:?}");
    // A ray in the z = 0 plane never meets the block.
    let miss = renderer.cast_ray(&vol, &transfer, [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
    assert!(miss.a < 1e-6, "{miss:?}");
    // The same diagonal entered from outside the cube.
    let outside = renderer.cast_ray(&vol, &transfer, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    assert!(outside.a > 0.99, "{outside:?}");
}

#[test]
fn rays_that_miss_the_volume_are_transparent() {
    let vol = half_filled_volume();
    let transfer = two_level_transfer(colour(RED, 1.0), colour(BLUE, 1.0));
    let renderer = VolumeRenderer::new(32);
    let transparent = Rgba::transparent();
    // Beyond the cube and pointing away, passing beside it, and a zero direction.
    assert_eq!(
        renderer.cast_ray(&vol, &transfer, [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),
        transparent
    );
    assert_eq!(
        renderer.cast_ray(&vol, &transfer, [-1.0, 0.5, 0.5], [-1.0, 0.0, 0.0]),
        transparent
    );
    assert_eq!(
        renderer.cast_ray(&vol, &transfer, [-1.0, 2.0, 0.5], [1.0, 0.0, 0.0]),
        transparent
    );
    assert_eq!(
        renderer.cast_ray(&vol, &transfer, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]),
        transparent
    );
    // No steps, and an empty volume.
    assert_eq!(
        VolumeRenderer::new(0).cast_ray(&vol, &transfer, [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),
        transparent
    );
    let empty = VolumeData::new([0, 0, 0], [1.0; 3]);
    assert_eq!(
        renderer.cast_ray(&empty, &transfer, [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),
        transparent
    );
    // Entering from outside the cube does hit.
    let hit = renderer.cast_ray(&vol, &transfer, [0.5, 0.5, -2.0], [0.0, 0.0, 1.0]);
    assert!(hit.a > 0.99 && hit.r > 0.5, "{hit:?}");
}

#[test]
fn ray_starting_inside_the_volume_marches_to_the_exit_face() {
    let vol = half_filled_volume();
    let transfer = two_level_transfer(colour(RED, 1.0), colour(BLUE, 1.0));
    let renderer = VolumeRenderer::new(32);
    // Starting in the middle (z = 0.5) the ray only sees the blue half.
    let c = renderer.cast_ray(&vol, &transfer, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);
    assert!(c.b > 0.5 && c.r < 0.1, "{c:?}");
}

#[test]
fn opacity_follows_the_chord_length_through_the_volume() {
    // A uniform translucent medium: the transfer alpha is the opacity of one
    // nominal step (1 / num_steps of the cube edge), so a chord of length L
    // crossed in n steps has the total opacity 1 - (1 - alpha)^(n * L).
    let mut vol = VolumeData::new([4, 4, 4], [1.0; 3]);
    vol.data.fill(0.3);
    let mut transfer = ColorTransferFunc::new();
    transfer.add_point(0.0, Rgba::new(1.0, 1.0, 1.0, 0.1));
    transfer.add_point(1.0, Rgba::new(1.0, 1.0, 1.0, 0.1));
    let mut renderer = VolumeRenderer::new(10);
    renderer.early_termination_alpha = 2.0;
    let expected = |chord: f32| 1.0 - 0.9_f32.powf(10.0 * chord);
    let cases = [
        ([0.5, 0.5, -1.0], [0.0, 0.0, 1.0], 1.0),
        ([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 0.5),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3.0_f32.sqrt()),
    ];
    for (origin, dir, chord) in cases {
        let c = renderer.cast_ray(&vol, &transfer, origin, dir);
        assert!(
            (c.a - expected(chord)).abs() < 1e-4,
            "chord {chord}: alpha {} vs {}",
            c.a,
            expected(chord)
        );
    }
}

#[test]
fn cast_ray_with_range_normalises_with_the_supplied_range() {
    // The red channel of the transfer function is the normalised value.
    let mut transfer = ColorTransferFunc::new();
    transfer.add_point(0.0, Rgba::new(0.0, 0.0, 0.0, 1.0));
    transfer.add_point(1.0, Rgba::new(1.0, 0.0, 0.0, 1.0));
    let mut vol = VolumeData::new([3, 3, 3], [1.0; 3]);
    vol.data.fill(0.25);
    let renderer = VolumeRenderer::new(16);
    let origin = [0.5, 0.5, 0.0];
    let dir = [0.0, 0.0, 1.0];

    // A constant volume has no range of its own and maps to the middle.
    let own = renderer.cast_ray(&vol, &transfer, origin, dir);
    assert!((own.r - 0.5).abs() < 1e-6, "{own:?}");
    let full = renderer.cast_ray_with_range(&vol, &transfer, (0.0, 1.0), origin, dir);
    assert!((full.r - 0.25).abs() < 1e-6, "{full:?}");
    let shifted = renderer.cast_ray_with_range(&vol, &transfer, (0.25, 0.75), origin, dir);
    assert!(shifted.r.abs() < 1e-6, "{shifted:?}");

    // cast_ray is cast_ray_with_range with the volume's own range.
    let mut varying = z_ramp(4);
    varying.data[5] = 9.0;
    let a = renderer.cast_ray(&varying, &transfer, origin, dir);
    let b = renderer.cast_ray_with_range(&varying, &transfer, varying.value_range(), origin, dir);
    assert_eq!(a, b);
}

// ─────────────────────────────────────────────────────────────────────────────
// MaximumIntensityProj
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mip_includes_the_exit_point_of_the_ray() {
    let mut vol = VolumeData::new([2, 2, 4], [1.0; 3]);
    vol.set(0, 0, 3, 0.9);
    let mip = MaximumIntensityProj::new(50);
    // The last voxel is only reached at the exit face of the cube.
    assert_eq!(mip.cast_ray(&vol, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), 0.9);
    assert_eq!(mip.cast_ray(&vol, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]), 0.9);
}

#[test]
fn mip_marches_the_whole_diagonal_chord() {
    let vol = corner_block_volume();
    let mip = MaximumIntensityProj::new(64);
    assert_eq!(mip.cast_ray(&vol, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1.0);
    assert_eq!(mip.cast_ray(&vol, [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]), 0.0);
    // Entering from outside the cube.
    assert_eq!(mip.cast_ray(&vol, [-2.0, -2.0, -2.0], [1.0, 1.0, 1.0]), 1.0);
}

#[test]
fn mip_of_a_ray_that_misses_is_zero() {
    let mut vol = VolumeData::new([4, 4, 4], [1.0; 3]);
    vol.data.fill(0.7);
    let mip = MaximumIntensityProj::new(50);
    for (origin, dir) in [
        ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),
        ([-1.0, 0.5, 0.5], [-1.0, 0.0, 0.0]),
        ([-1.0, 2.0, 0.5], [1.0, 0.0, 0.0]),
        ([0.5, 0.5, 0.5], [0.0, 0.0, 0.0]),
    ] {
        let v = mip.cast_ray(&vol, origin, dir);
        assert_eq!(v, 0.0, "origin {origin:?} dir {dir:?}");
    }
    assert_eq!(
        MaximumIntensityProj::new(0).cast_ray(&vol, [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]),
        0.0
    );
    // A ray that does hit sees the volume.
    let hit = mip.cast_ray(&vol, [0.5, 0.5, -3.0], [0.0, 0.0, 1.0]);
    assert!((hit - 0.7).abs() < 1e-6, "{hit}");
}

// ─────────────────────────────────────────────────────────────────────────────
// IsosurfaceRenderer
// ─────────────────────────────────────────────────────────────────────────────

/// Corner positions of the unit cube in marching-cubes order.
const UNIT_CUBE: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

/// Corner values with 0 at the `low` corners and 1 at the others.
fn corner_values_with_low(low: &[usize]) -> [f32; 8] {
    std::array::from_fn(|corner| if low.contains(&corner) { 0.0 } else { 1.0 })
}

/// Case index (bit `c` set for a low corner `c`).
fn case_of(low: &[usize]) -> usize {
    low.iter().fold(0, |case, &corner| case | (1 << corner))
}

/// Doubled coordinates of a point of the unit cube, for exact set comparison
/// of edge midpoints.
fn key(p: [f32; 3]) -> [i32; 3] {
    p.map(|c| (c * 2.0).round() as i32)
}

fn midpoint_key_of_edge(edge: usize) -> [i32; 3] {
    let (a, b) = crate::isosurface::CUBE_EDGES[edge];
    std::array::from_fn(|k| (UNIT_CUBE[a][k] + UNIT_CUBE[b][k]).round() as i32)
}

fn vertex_keys(tris: &[IsoTriangle]) -> BTreeSet<[i32; 3]> {
    tris.iter().flat_map(|t| t.vertices).map(key).collect()
}

#[test]
fn one_low_corner_gives_one_triangle_around_it() {
    let renderer = IsosurfaceRenderer::new(0.5);
    let tris = renderer.marching_cube_triangles(corner_values_with_low(&[0]), UNIT_CUBE);
    assert_eq!(tris.len(), 1);
    let expected: BTreeSet<[i32; 3]> = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].into_iter().collect();
    assert_eq!(vertex_keys(&tris), expected);

    // The same cell with the roles of low and high exchanged: corners 1..8 low.
    let mut values = [0.0_f32; 8];
    values[0] = 1.0;
    let tris = renderer.marching_cube_triangles(values, UNIT_CUBE);
    assert_eq!(tris.len(), 1);
    assert_eq!(vertex_keys(&tris), expected);
}

#[test]
fn two_adjacent_low_corners_give_a_quad_of_two_triangles() {
    let renderer = IsosurfaceRenderer::new(0.5);
    let tris = renderer.marching_cube_triangles(corner_values_with_low(&[0, 1]), UNIT_CUBE);
    // One triangle (the old pairing of crossed edges) left a hole.
    assert_eq!(tris.len(), 2);
    assert_eq!(vertex_keys(&tris).len(), 4, "{tris:?}");
}

#[test]
fn opposite_low_corners_give_two_disjoint_triangles() {
    let renderer = IsosurfaceRenderer::new(0.5);
    let tris = renderer.marching_cube_triangles(corner_values_with_low(&[0, 6]), UNIT_CUBE);
    assert_eq!(tris.len(), 2);
    assert_eq!(vertex_keys(&tris).len(), 6, "the triangles share no vertex");
}

#[test]
fn cells_on_one_side_of_the_iso_value_have_no_surface() {
    let renderer = IsosurfaceRenderer::new(0.5);
    assert!(
        renderer
            .marching_cube_triangles([1.0; 8], UNIT_CUBE)
            .is_empty()
    );
    assert!(
        renderer
            .marching_cube_triangles([0.0; 8], UNIT_CUBE)
            .is_empty()
    );
    // A corner exactly at the iso-value is not below it.
    assert!(
        renderer
            .marching_cube_triangles([0.5; 8], UNIT_CUBE)
            .is_empty()
    );
}

#[test]
fn every_cell_case_is_triangulated_by_the_shared_tables() {
    let renderer = IsosurfaceRenderer::new(0.5);
    for case in 0..256usize {
        let low: Vec<usize> = (0..8).filter(|&c| case & (1 << c) != 0).collect();
        assert_eq!(case_of(&low), case);
        let values = corner_values_with_low(&low);
        let tris = renderer.marching_cube_triangles(values, UNIT_CUBE);
        let live = TRI_TABLE[case].iter().take_while(|&&e| e != -1).count();
        assert_eq!(tris.len(), live / 3, "case {case}");
        assert!(tris.len() <= 5, "case {case}: {} triangles", tris.len());
        if case == 0 || case == 255 {
            assert!(tris.is_empty());
            continue;
        }
        let expected: BTreeSet<[i32; 3]> = (0..12usize)
            .filter(|&edge| EDGE_TABLE[case] & (1 << edge) != 0)
            .map(midpoint_key_of_edge)
            .collect();
        assert_eq!(
            vertex_keys(&tris),
            expected,
            "case {case}: the vertices are the midpoints of the crossed edges"
        );
        for tri in &tris {
            // Unit normals shared by the three vertices, along (v1-v0) x (v2-v0)
            // and pointing towards decreasing values.
            let n = tri.normals[0];
            assert_eq!(tri.normals, [n; 3]);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "case {case}: |n| = {len}");
            let [v0, v1, v2] = tri.vertices;
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            assert!(
                n[0] * cross[0] + n[1] * cross[1] + n[2] * cross[2] > 0.0,
                "case {case}: the normal follows the winding"
            );
            let centroid = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ];
            let g = trilinear_gradient(&values, centroid);
            assert!(
                n[0] * g[0] + n[1] * g[1] + n[2] * g[2] < 0.0,
                "case {case}: the normal {n:?} must point towards decreasing values (grad {g:?})"
            );
        }
    }
}

/// Gradient of the trilinear interpolant of `values` over the unit cube.
fn trilinear_gradient(values: &[f32; 8], p: [f32; 3]) -> [f32; 3] {
    let mut g = [0.0_f32; 3];
    for (corner, offset) in UNIT_CUBE.iter().enumerate() {
        let weight = |axis: usize| {
            if offset[axis] > 0.5 {
                p[axis]
            } else {
                1.0 - p[axis]
            }
        };
        let slope = |axis: usize| if offset[axis] > 0.5 { 1.0 } else { -1.0 };
        g[0] += values[corner] * slope(0) * weight(1) * weight(2);
        g[1] += values[corner] * weight(0) * slope(1) * weight(2);
        g[2] += values[corner] * weight(0) * weight(1) * slope(2);
    }
    g
}

#[test]
fn crossings_are_interpolated_linearly_in_the_cell_geometry() {
    // A cell of edge 2 at an offset; corner 0 is 0 and the others are 1, so
    // with the iso-value 0.25 every crossing sits a quarter along its edge.
    let positions: [[f32; 3]; 8] = std::array::from_fn(|c| UNIT_CUBE[c].map(|x| 10.0 + 2.0 * x));
    let renderer = IsosurfaceRenderer::new(0.25);
    let tris = renderer.marching_cube_triangles(corner_values_with_low(&[0]), positions);
    assert_eq!(tris.len(), 1);
    let mut got: Vec<[f32; 3]> = tris[0].vertices.to_vec();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let want = [[10.0, 10.0, 10.5], [10.0, 10.5, 10.0], [10.5, 10.0, 10.0]];
    for (g, w) in got.iter().zip(want.iter()) {
        for (gk, wk) in g.iter().zip(w.iter()) {
            assert!((gk - wk).abs() < 1e-5, "{got:?} vs {want:?}");
        }
    }
}

#[test]
fn degenerate_cells_give_no_triangles() {
    let renderer = IsosurfaceRenderer::new(0.5);
    let low = corner_values_with_low(&[0, 6]);
    // Non-finite values or iso-value.
    let mut values = low;
    values[3] = f32::NAN;
    assert!(
        renderer
            .marching_cube_triangles(values, UNIT_CUBE)
            .is_empty()
    );
    values[3] = f32::INFINITY;
    assert!(
        renderer
            .marching_cube_triangles(values, UNIT_CUBE)
            .is_empty()
    );
    assert!(
        IsosurfaceRenderer::new(f32::NAN)
            .marching_cube_triangles(low, UNIT_CUBE)
            .is_empty()
    );
    // Collapsed cell: every triangle has zero area.
    assert!(
        renderer
            .marching_cube_triangles(low, [[1.0, 2.0, 3.0]; 8])
            .is_empty()
    );
}

#[test]
fn normals_of_a_tiny_cell_are_still_unit_length() {
    let positions: [[f32; 3]; 8] = std::array::from_fn(|c| UNIT_CUBE[c].map(|x| x * 1e-12));
    let renderer = IsosurfaceRenderer::new(0.5);
    let tris = renderer.marching_cube_triangles(corner_values_with_low(&[0, 1]), positions);
    assert_eq!(tris.len(), 2);
    for tri in &tris {
        let n = tri.normals[0];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "{n:?}");
    }
}
