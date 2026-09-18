// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the additive-manufacturing overhang filter and its
//! adjoint: support propagation layer by layer in all four build directions,
//! the dependence on the overhang angle, and the exact sensitivities.

use super::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

/// An overhang angle a hair below a right angle, tan of which (about 1e4) is
/// far larger than any layer.
const NEAR_RIGHT_ANGLE: f64 = FRAC_PI_2 - 1e-4;

/// Deterministic uniform numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn filters(nx: usize, ny: usize, angle: f64, direction: usize) -> ManufacturingFilters {
    ManufacturingFilters::new(GridSize::new(nx, ny), 2.0, angle).with_build_direction(direction)
}

fn layers_of(filter: &ManufacturingFilters) -> BuildLayers {
    BuildLayers {
        nx: filter.grid.nx,
        ny: filter.grid.ny,
        direction: filter.build_direction,
    }
}

/// A density that is one at the given `(layer, position)` cells (layer 0 is
/// on the base plate) and zero elsewhere.
fn solid_at(filter: &ManufacturingFilters, cells: &[(usize, usize)]) -> Vec<f64> {
    let layers = layers_of(filter);
    let mut density = vec![0.0; filter.grid.n_elem()];
    for &(layer, position) in cells {
        density[layers.index(layer, position)] = 1.0;
    }
    density
}

fn at(filter: &ManufacturingFilters, field: &[f64], layer: usize, position: usize) -> f64 {
    field[layers_of(filter).index(layer, position)]
}

/// `smin(1, xi)`, the printed density of a fully designed element whose
/// support is `xi`.
fn printed_from_support(xi: f64) -> f64 {
    0.5 * (1.0 + xi + 1e-2 - ((1.0 - xi).powi(2) + 1e-4).sqrt())
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[test]
fn layers_enumerate_the_grid_from_the_base_plate() {
    // 3 columns, 2 rows: flat index = row * 3 + col.
    let layers = |direction: usize| BuildLayers {
        nx: 3,
        ny: 2,
        direction,
    };
    assert_eq!((layers(0).count(), layers(0).width()), (2, 3));
    assert_eq!((layers(1).count(), layers(1).width()), (2, 3));
    assert_eq!((layers(2).count(), layers(2).width()), (3, 2));
    assert_eq!((layers(3).count(), layers(3).width()), (3, 2));
    // +y: the base plate is row 0.
    assert_eq!((layers(0).index(0, 0), layers(0).index(1, 2)), (0, 5));
    // -y: the base plate is the last row.
    assert_eq!((layers(1).index(0, 0), layers(1).index(1, 2)), (3, 2));
    // +x: the base plate is column 0.
    assert_eq!(
        (
            layers(2).index(0, 0),
            layers(2).index(1, 0),
            layers(2).index(2, 1)
        ),
        (0, 1, 5)
    );
    // -x: the base plate is the last column (and anything above 3 means -x).
    assert_eq!((layers(3).index(0, 0), layers(3).index(2, 1)), (2, 3));
    assert_eq!(layers(9).index(1, 1), layers(3).index(1, 1));
    for direction in 0..4 {
        let mut seen = [false; 6];
        for layer in 0..layers(direction).count() {
            for position in 0..layers(direction).width() {
                seen[layers(direction).index(layer, position)] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "direction {direction} misses elements"
        );
    }
}

#[test]
fn footprint_weights_follow_the_tangent_of_the_overhang_angle() {
    let weights = |angle: f64, width: usize| footprint_weights(angle, width);
    assert_eq!(weights(0.0, 10), Some(vec![1.0]));
    assert_eq!(weights(FRAC_PI_4, 10), Some(vec![1.0, 1.0, 1.0]));
    assert_eq!(weights(2.0f64.atan(), 10), Some(vec![1.0; 5]));
    assert_eq!(weights(3.0f64.atan(), 10), Some(vec![1.0; 7]));
    let fractional = weights(1.5f64.atan(), 10).unwrap();
    assert_eq!(fractional.len(), 5);
    for (got, want) in fractional.iter().zip([0.5, 1.0, 1.0, 1.0, 0.5]) {
        assert!((got - want).abs() < 1e-12, "{fractional:?}");
    }
    let shallow = weights(0.6f64.atan(), 10).unwrap();
    for (got, want) in shallow.iter().zip([0.6, 1.0, 0.6]) {
        assert!((got - want).abs() < 1e-12, "{shallow:?}");
    }
    // Angles that cannot lean at all, or are not numbers, allow vertical walls only.
    assert_eq!(weights(-0.3, 10), Some(vec![1.0]));
    assert_eq!(weights(f64::NAN, 10), Some(vec![1.0]));
    // A right angle or more is unconstrained; a near-right angle is clamped to the layer.
    assert_eq!(weights(FRAC_PI_2, 10), None);
    assert_eq!(weights(3.0, 10), None);
    assert_eq!(weights(NEAR_RIGHT_ANGLE, 4), Some(vec![1.0; 7]));
    assert_eq!(weights(NEAR_RIGHT_ANGLE, 1), Some(vec![1.0]));
}

#[test]
fn smooth_minimum_and_its_derivatives_behave() {
    assert!((smooth_min(1.0, 1.0) - 1.0).abs() < 1e-15);
    assert!(smooth_min(0.0, 0.0).abs() < 1e-15);
    // Close to the minimum, never below it and at most sqrt(eps)/2 above it.
    for &(x, y) in &[(1.0, 0.0), (0.3, 0.9), (0.5, 0.5), (0.0, 1.0), (0.7, 0.71)] {
        let value = smooth_min(x, y);
        assert!(value >= f64::min(x, y) - 1e-15);
        assert!(value <= f64::min(x, y) + 0.005 + 1e-15);
        assert!((value - smooth_min(y, x)).abs() < 1e-15);
        let (dx, dy) = smooth_min_partials(x, y);
        assert!((0.0..=1.0).contains(&dx) && (0.0..=1.0).contains(&dy));
        assert!((dx + dy - 1.0).abs() < 1e-14);
        let h = 1e-7;
        let fd_x = (smooth_min(x + h, y) - smooth_min(x - h, y)) / (2.0 * h);
        let fd_y = (smooth_min(x, y + h) - smooth_min(x, y - h)) / (2.0 * h);
        assert!((dx - fd_x).abs() < 1e-6 && (dy - fd_y).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Overhang filter
// ---------------------------------------------------------------------------

#[test]
fn the_base_layer_is_printed_as_designed_in_every_direction() {
    for direction in 0..4 {
        let filter = filters(6, 5, FRAC_PI_4, direction);
        let mut rng = Lcg(7);
        let density: Vec<f64> = (0..30).map(|_| rng.next_unit()).collect();
        let printed = filter.overhang_filter(&density);
        let layers = layers_of(&filter);
        for position in 0..layers.width() {
            let i = layers.index(0, position);
            assert_eq!(
                printed[i], density[i],
                "direction {direction}, position {position}"
            );
        }
    }
}

#[test]
fn the_build_direction_selects_the_base_plate() {
    // A single solid element in the top row, and one in the left column.
    let mut top = [0.0; 16];
    top[3 * 4 + 1] = 1.0;
    let mut left = [0.0; 16];
    left[2 * 4] = 1.0;
    // +y builds from row 0, so the top element floats; -y builds from the top row.
    assert!(filters(4, 4, FRAC_PI_4, 0).overhang_filter(&top)[13] < 0.01);
    assert_eq!(filters(4, 4, FRAC_PI_4, 1).overhang_filter(&top)[13], 1.0);
    // +x builds from column 0, -x from the last column (as does any larger value).
    assert_eq!(filters(4, 4, FRAC_PI_4, 2).overhang_filter(&left)[8], 1.0);
    assert!(filters(4, 4, FRAC_PI_4, 3).overhang_filter(&left)[8] < 0.01);
    assert!(filters(4, 4, FRAC_PI_4, 7).overhang_filter(&left)[8] < 0.01);
}

#[test]
fn a_floating_block_is_printed_as_void_in_every_direction() {
    for direction in 0..4 {
        let filter = filters(10, 10, FRAC_PI_4, direction);
        let block: Vec<(usize, usize)> = (3..6)
            .flat_map(|layer| (2..5).map(move |position| (layer, position)))
            .collect();
        let printed = filter.overhang_filter(&solid_at(&filter, &block));
        for &(layer, position) in &block {
            let value = at(&filter, &printed, layer, position);
            assert!(
                value < 0.02,
                "direction {direction}: floating element ({layer}, {position}) printed {value}"
            );
        }
        // The smooth minimum leaks about sqrt(eps)/2 = 0.005 per layer, so the
        // three layers of the block print at 0.005, 0.010 and 0.015 at most.
        assert!(printed.iter().all(|&p| (0.0..0.02).contains(&p)));
    }
}

#[test]
fn missing_support_cuts_off_everything_above_it() {
    // A column on the base plate with a gap in the middle: the part above the
    // gap is unsupported all the way up, not only just above the gap.
    for direction in 0..4 {
        let filter = filters(9, 9, FRAC_PI_4, direction);
        let column: Vec<(usize, usize)> = (0..9)
            .filter(|&layer| layer != 3)
            .map(|layer| (layer, 4))
            .collect();
        let printed = filter.overhang_filter(&solid_at(&filter, &column));
        for layer in 0..3 {
            assert!(
                at(&filter, &printed, layer, 4) > 0.99,
                "direction {direction}, layer {layer}"
            );
        }
        // The smooth minimum lets about sqrt(eps)/2 = 0.005 leak through per
        // layer, so the top of the unsupported column prints at about 0.03
        // instead of the designed 1.
        let mut previous = 0.0;
        for layer in 4..9 {
            let value = at(&filter, &printed, layer, 4);
            assert!(
                value < 0.04,
                "direction {direction}, layer {layer}: {value}"
            );
            assert!(value >= previous, "the leak only accumulates upwards");
            previous = value;
        }
    }
}

#[test]
fn a_forty_five_degree_staircase_from_the_base_is_retained() {
    for direction in 0..4 {
        let filter = filters(10, 10, FRAC_PI_4, direction);
        // Every layer extends the one below by one element.
        let cells: Vec<(usize, usize)> = (0..5)
            .flat_map(|layer| (0..4 + layer).map(move |position| (layer, position)))
            .collect();
        let printed = filter.overhang_filter(&solid_at(&filter, &cells));
        for &(layer, position) in &cells {
            let value = at(&filter, &printed, layer, position);
            assert!(
                (0.99..=1.0051).contains(&value),
                "direction {direction}: ({layer}, {position}) printed {value}"
            );
        }
        // Beyond the staircase there is nothing to print.
        assert!(at(&filter, &printed, 4, 9) < 0.01);
    }
}

#[test]
fn a_shallower_overhang_needs_a_larger_overhang_angle() {
    let staircase = |filter: &ManufacturingFilters| {
        // Every layer extends the one below by two elements.
        let cells: Vec<(usize, usize)> = (0..3)
            .flat_map(|layer| (0..4 + 2 * layer).map(move |position| (layer, position)))
            .collect();
        solid_at(filter, &cells)
    };
    for direction in 0..4 {
        let strict = filters(12, 12, FRAC_PI_4, direction);
        let printed = strict.overhang_filter(&staircase(&strict));
        // One element beyond the edge is within reach of the diagonal, two are not.
        assert!(at(&strict, &printed, 1, 4) > 0.99, "direction {direction}");
        assert!(at(&strict, &printed, 1, 5) < 0.02, "direction {direction}");
        // The cut-off propagates to the next layer: (2, 5) is still carried by
        // the retained element (1, 4), but (2, 6) and (2, 7) have nothing
        // substantial underneath.
        assert!(at(&strict, &printed, 2, 5) > 0.99, "direction {direction}");
        assert!(at(&strict, &printed, 2, 6) < 0.02, "direction {direction}");
        assert!(at(&strict, &printed, 2, 7) < 0.02, "direction {direction}");

        // With tan(angle) = 2 the whole staircase is retained.
        let relaxed = filters(12, 12, 2.0f64.atan(), direction);
        let printed = relaxed.overhang_filter(&staircase(&relaxed));
        for layer in 0..3 {
            for position in 0..4 + 2 * layer {
                let value = at(&relaxed, &printed, layer, position);
                assert!(
                    value > 0.99,
                    "direction {direction}: ({layer}, {position}) {value}"
                );
            }
        }
    }
}

#[test]
fn with_no_overhang_allowed_only_vertical_walls_are_printed() {
    for direction in 0..4 {
        let filter = filters(10, 10, 0.0, direction);
        let printed = filter.overhang_filter(&solid_at(&filter, &[(0, 5), (1, 5), (1, 6), (2, 6)]));
        assert!(at(&filter, &printed, 0, 5) > 0.99);
        assert!(at(&filter, &printed, 1, 5) > 0.99);
        // Offset by one element, resting only diagonally.
        assert!(at(&filter, &printed, 1, 6) < 0.02, "direction {direction}");
        assert!(at(&filter, &printed, 2, 6) < 0.02, "direction {direction}");
    }
}

#[test]
fn fractional_tangents_give_partial_support_at_the_edge_of_the_footprint() {
    // tan = 1.5: reach 2, the outermost elements count with weight 0.5.
    let filter = filters(12, 6, 1.5f64.atan(), 0);
    let printed = filter.overhang_filter(&solid_at(
        &filter,
        &[(0, 5), (1, 5), (1, 6), (1, 7), (1, 8), (1, 3), (1, 2)],
    ));
    for position in [5, 6] {
        assert!(
            at(&filter, &printed, 1, position) > 0.99,
            "position {position}"
        );
    }
    // Two elements away: half support.
    let half = printed_from_support(0.5);
    assert!(
        (at(&filter, &printed, 1, 7) - half).abs() < 1e-9,
        "{}",
        at(&filter, &printed, 1, 7)
    );
    assert!((at(&filter, &printed, 1, 3) - half).abs() < 1e-9);
    // Three elements away: out of reach.
    assert!(at(&filter, &printed, 1, 8) < 0.01);
    assert!(at(&filter, &printed, 1, 2) < 0.01);

    // tan = 0.6: reach 1 with the diagonal elements at weight 0.6.
    let filter = filters(12, 6, 0.6f64.atan(), 0);
    let printed = filter.overhang_filter(&solid_at(
        &filter,
        &[(0, 5), (1, 5), (1, 6), (1, 4), (1, 7)],
    ));
    assert!(at(&filter, &printed, 1, 5) > 0.99);
    let partial = printed_from_support(0.6);
    assert!((at(&filter, &printed, 1, 6) - partial).abs() < 1e-9);
    assert!((at(&filter, &printed, 1, 4) - partial).abs() < 1e-9);
    assert!(at(&filter, &printed, 1, 7) < 0.01);
}

#[test]
fn the_printed_density_is_monotone_in_the_overhang_angle() {
    let mut rng = Lcg(99);
    let density: Vec<f64> = (0..60).map(|_| rng.next_unit()).collect();
    for direction in 0..4 {
        let mut previous: Option<Vec<f64>> = None;
        for angle in [0.0, 0.3, 0.6, FRAC_PI_4, 1.0, 2.0f64.atan(), 1.3, 1.5] {
            let printed = filters(10, 6, angle, direction).overhang_filter(&density);
            if let Some(before) = &previous {
                for (a, b) in before.iter().zip(printed.iter()) {
                    assert!(
                        b + 1e-12 >= *a,
                        "direction {direction}, angle {angle}: {a} > {b}"
                    );
                }
            }
            previous = Some(printed);
        }
    }
}

#[test]
fn right_angles_and_wrong_sizes_leave_the_density_unchanged() {
    let mut rng = Lcg(5);
    let density: Vec<f64> = (0..16).map(|_| rng.next_unit()).collect();
    for angle in [FRAC_PI_2, 2.0, 100.0, f64::INFINITY] {
        for direction in 0..4 {
            let filter = filters(4, 4, angle, direction);
            assert_eq!(filter.overhang_filter(&density), density, "angle {angle}");
        }
    }
    let filter = filters(4, 4, FRAC_PI_4, 0);
    assert_eq!(filter.overhang_filter(&density[..15]), density[..15]);
    assert!(filter.overhang_filter(&[]).is_empty());
    let empty = filters(0, 0, FRAC_PI_4, 2);
    assert!(empty.overhang_filter(&[]).is_empty());
    assert!(filters(4, 0, FRAC_PI_4, 0).overhang_filter(&[]).is_empty());
}

#[test]
fn a_nearly_horizontal_limit_supports_a_whole_layer_without_panicking() {
    // tan of the angle is about 1e4, far wider than the layer: every element of
    // the layer above is within reach of the one solid element below.
    for direction in 0..4 {
        let filter = filters(8, 8, NEAR_RIGHT_ANGLE, direction);
        let mut cells = vec![(0, 0)];
        cells.extend((0..8).map(|position| (1, position)));
        let printed = filter.overhang_filter(&solid_at(&filter, &cells));
        for position in 0..8 {
            assert!(
                at(&filter, &printed, 1, position) > 0.99,
                "direction {direction}, position {position}"
            );
        }
    }
}

#[test]
fn printed_densities_stay_within_the_smoothing_margin_of_the_design() {
    let mut rng = Lcg(2024);
    for direction in 0..4 {
        for angle in [0.0, 0.4, FRAC_PI_4, 1.1, 2.0f64.atan(), 1.4] {
            let filter = filters(7, 6, angle, direction);
            let density: Vec<f64> = (0..42).map(|_| rng.next_unit()).collect();
            let printed = filter.overhang_filter(&density);
            for (design, now) in density.iter().zip(printed.iter()) {
                assert!(*now >= 0.0, "negative printed density {now}");
                assert!(
                    *now <= design + 0.005 + 1e-12,
                    "printed {now} exceeds design {design}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Adjoint
// ---------------------------------------------------------------------------

#[test]
fn the_sensitivity_is_the_exact_derivative_of_the_filter() {
    let (nx, ny) = (6, 5);
    let mut rng = Lcg(31337);
    for direction in 0..4 {
        for angle in [FRAC_PI_4, 2.0f64.atan(), 0.9, 0.0, 0.3, 1.2] {
            let filter = filters(nx, ny, angle, direction);
            let density: Vec<f64> = (0..nx * ny).map(|_| 0.1 + 0.8 * rng.next_unit()).collect();
            let weights: Vec<f64> = (0..nx * ny).map(|_| 2.0 * rng.next_unit() - 1.0).collect();
            let analytic = filter.overhang_filter_sensitivity(&density, &weights);
            assert_eq!(analytic.len(), density.len());

            let objective = |field: &[f64]| -> f64 {
                filter
                    .overhang_filter(field)
                    .iter()
                    .zip(weights.iter())
                    .map(|(printed, weight)| printed * weight)
                    .sum()
            };
            let h = 1e-6;
            for i in 0..nx * ny {
                let mut plus = density.clone();
                plus[i] += h;
                let mut minus = density.clone();
                minus[i] -= h;
                let numeric = (objective(&plus) - objective(&minus)) / (2.0 * h);
                assert!(
                    (numeric - analytic[i]).abs() <= 1e-5 * analytic[i].abs().max(1.0),
                    "direction {direction}, angle {angle}, element {i}: analytic {} vs numeric {numeric}",
                    analytic[i]
                );
            }
        }
    }
}

#[test]
fn increasing_any_design_density_never_lowers_the_printed_total() {
    let mut rng = Lcg(77);
    for direction in 0..4 {
        for angle in [0.0, FRAC_PI_4, 1.3] {
            let filter = filters(8, 7, angle, direction);
            let density: Vec<f64> = (0..56).map(|_| 0.05 + 0.9 * rng.next_unit()).collect();
            let gradient = filter.overhang_filter_sensitivity(&density, &[1.0; 56]);
            assert!(
                gradient.iter().all(|&g| g >= -1e-12),
                "direction {direction}, angle {angle}: {gradient:?}"
            );
            // Elements of the base layer always count at least once.
            let layers = layers_of(&filter);
            for position in 0..layers.width() {
                assert!(gradient[layers.index(0, position)] >= 1.0 - 1e-12);
            }
        }
    }
}

#[test]
fn the_sensitivity_of_a_floating_element_flows_to_its_supports() {
    // One designed element above one supporter on the base plate: the printed
    // value is smin(rho, xi) with xi = rho_support, so for rho > support the
    // derivative is carried almost entirely by the support.
    let filter = filters(3, 2, 0.0, 0);
    let density = [0.4, 0.0, 0.0, 0.9, 0.0, 0.0];
    let gradient = filter.overhang_filter_sensitivity(&density, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert!(gradient[3] < 0.01, "{}", gradient[3]);
    assert!(gradient[0] > 0.99, "{}", gradient[0]);
    assert!(gradient[1].abs() < 1e-12 && gradient[2].abs() < 1e-12);
}

#[test]
fn degenerate_sensitivity_requests_return_the_input() {
    let filter = filters(4, 4, FRAC_PI_4, 0);
    let density = [0.5; 16];
    let weights: Vec<f64> = (0..16).map(|i| i as f64).collect();
    // Wrong sizes.
    assert_eq!(
        filter.overhang_filter_sensitivity(&density[..15], &weights[..15]),
        weights[..15]
    );
    assert_eq!(
        filter.overhang_filter_sensitivity(&density, &weights[..15]),
        weights[..15]
    );
    // The identity filter maps sensitivities to themselves.
    let identity = filters(4, 4, FRAC_PI_2, 0);
    assert_eq!(
        identity.overhang_filter_sensitivity(&density, &weights),
        weights
    );
    // Without any material the supports receive no sensitivity but the base
    // layer still passes its own through.
    let void = [0.0; 16];
    let gradient = filter.overhang_filter_sensitivity(&void, &[1.0; 16]);
    assert!(gradient.iter().all(|g| g.is_finite()));
    for (position, &entry) in gradient.iter().take(4).enumerate() {
        assert!(entry >= 1.0 - 1e-12, "base layer element {position}");
    }
}
