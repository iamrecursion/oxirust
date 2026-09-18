// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Behavioural tests of the sand simulation: things the previous ballistic
//! free-fall integrator could not do (stand on the floor, hold a slope, flow
//! when the slope is too steep, spread and settle, slide with friction).
//!
//! Except for the cylinder collapse at the end, all runs are plane-strain slabs
//! one cell thick (a 2-D problem at a fraction of the 3-D cost) on a grid of
//! 25 mm cells with two particles per cell edge.
//! Note that the Drucker-Prager cone is calibrated to the Mohr-Coulomb friction
//! angle in *triaxial compression*; in plane strain it admits steeper heaps than
//! `φ` (a heap of `φ = 30°` sand stands up to about 45°). The stability tests
//! therefore compare against the plane-strain friction angle derived from the
//! cone (`plane_strain_friction_angle_deg`) or against outcomes at slopes well
//! away from the boundary, never against `tan φ`. A dynamic collapse does not
//! stop at the limit slope either: the flow overshoots, so the settled flank of
//! a collapsed column is flatter than the friction angle.

use super::SandSimulation;
use super::test_support::{
    assert_close, cylinder_column, half_width, max_height, max_radius, mean_height, slab_column,
    wedge,
};

const SPACING: f64 = 0.0125;
const HALF_BASE: f64 = 0.1;
const WEDGE_TIME: f64 = 0.6;

/// Height retained by a pre-built heap after `WEDGE_TIME`: final / initial
/// maximum height (1 = unmoved, → 0 = flowed out flat).
fn retained_height(friction_angle_deg: f64, slope_deg: f64) -> f64 {
    let mut sim = wedge(friction_angle_deg, slope_deg, HALF_BASE, SPACING);
    let initial = max_height(&sim);
    assert!(initial > 0.0 && sim.particles.len() > 30);
    let mut elapsed = 0.0;
    while elapsed < WEDGE_TIME - 1e-12 {
        assert!(sim.try_step(0.05).is_ok());
        elapsed += 0.05;
    }
    assert!(
        sim.particles
            .iter()
            .all(|p| p.position.iter().all(|x| x.is_finite()))
    );
    max_height(&sim) / initial
}

// ---------------------------------------------------------------------------
// Standing on the floor
// ---------------------------------------------------------------------------

#[test]
fn a_flat_block_rests_on_the_floor() {
    // Ballistic free fall would drop every particle to y = 0 (mean height 0).
    let mut sim = slab_column(30.0, 0.2, 0.04, SPACING);
    let initial_mean = mean_height(&sim);
    let initial_top = max_height(&sim);
    for _ in 0..10 {
        assert!(sim.try_step(0.05).is_ok());
    }
    assert!(
        mean_height(&sim) > 0.98 * initial_mean,
        "block sank: {}",
        mean_height(&sim) / initial_mean
    );
    assert!(max_height(&sim) > 0.97 * initial_top);
    assert!(sim.kinetic_energy() < 1.0e-5, "KE {}", sim.kinetic_energy());
    assert!(
        half_width(&sim) < 0.11,
        "a wide flat block barely spreads: {}",
        half_width(&sim)
    );
    assert!(sim.particles.iter().all(|p| p.position[1] >= 0.0));
}

// ---------------------------------------------------------------------------
// Slope stability depends on the friction angle
// ---------------------------------------------------------------------------

#[test]
fn heaps_well_below_the_limit_stand() {
    // φ = 30° at 20°, φ = 40° at 30°: 10° below the friction angle.
    for (phi, slope) in [(30.0, 20.0), (40.0, 30.0)] {
        let retained = retained_height(phi, slope);
        assert!(
            retained > 0.98,
            "phi={phi} slope={slope}: retained {retained}"
        );
    }
}

#[test]
fn a_heap_slumps_when_the_friction_angle_is_too_small_for_its_slope() {
    // Same 40° heap: it flows for φ = 15° and stands for φ = 30°.
    let weak = retained_height(15.0, 40.0);
    let strong = retained_height(30.0, 40.0);
    assert!(weak < 0.8, "φ=15° must slump: retained {weak}");
    assert!(strong > 0.97, "φ=30° must stand: retained {strong}");
}

#[test]
fn retained_height_grows_with_the_friction_angle() {
    let retained: Vec<f64> = [15.0, 20.0, 25.0, 30.0]
        .iter()
        .map(|&phi| retained_height(phi, 40.0))
        .collect();
    for pair in retained.windows(2) {
        assert!(pair[1] > pair[0] + 0.03, "not increasing: {retained:?}");
    }
}

#[test]
fn a_steeper_heap_than_any_friction_angle_supports_collapses() {
    // 60° is beyond the plane-strain limit of φ = 30° ...
    let collapsed = retained_height(30.0, 60.0);
    assert!(
        collapsed < 0.85,
        "60° heap of φ=30° sand must flow: retained {collapsed}"
    );
    // ... but φ = 40° sand still carries it.
    let stands = retained_height(40.0, 60.0);
    assert!(stands > 0.95, "φ=40° sand must hold 60°: retained {stands}");
}

/// Friction angle \[degrees\] of a plane-strain heap whose stress state just
/// reaches the Drucker-Prager cone calibrated to the triaxial friction angle
/// `phi_deg`.
///
/// For the plane-strain stress state `σ₂ = (σ₁ + σ₃)/2` the cone
/// `√J₂ = α I₁` with `α = 2 sin φ / (√3 (3 − sin φ))` becomes
/// `(σ₁ − σ₃)/(σ₁ + σ₃) = 3 α = 2 √3 sin φ / (3 − sin φ)`, while Mohr-Coulomb in
/// plane strain reads `(σ₁ − σ₃)/(σ₁ + σ₃) = sin φ_ps`.
fn plane_strain_friction_angle_deg(phi_deg: f64) -> f64 {
    let sin_phi = phi_deg.to_radians().sin();
    (2.0 * 3.0_f64.sqrt() * sin_phi / (3.0 - sin_phi))
        .asin()
        .to_degrees()
}

#[test]
fn the_stability_limit_of_a_heap_follows_the_plane_strain_friction_angle() {
    // The limit slope of a cohesionless heap is the friction angle of its
    // stress state, here `plane_strain_friction_angle_deg(φ)` (26.5°, 34.6°,
    // 43.9° and 55.0° for φ = 20°, 25°, 30° and 35°). A heap 8° below it
    // stands; one 18° above it flows. The window is asymmetric because the
    // visible failure is delayed on a coarse lattice. Measured with this
    // fixture, the retained height first drops below 0.9 between 7° and 12°
    // above the limit on the 12.5 mm lattice used here and between 3° and 8°
    // above it on a 6.25 mm lattice, i.e. the bias shrinks with the lattice
    // spacing; a heap only slightly over the limit also needs time to fail.
    // The slopes that must stand and flow move with φ by 28° across the four
    // friction angles, so the test fails if the cone is mis-calibrated by more
    // than the window.
    for phi in [20.0, 25.0, 30.0, 35.0] {
        let limit = plane_strain_friction_angle_deg(phi);
        let standing = retained_height(phi, limit - 8.0);
        let flowing = retained_height(phi, limit + 18.0);
        assert!(
            standing > 0.97,
            "φ={phi}°: a {:.1}° heap (limit {limit:.1}°) must stand, retained {standing}",
            limit - 8.0
        );
        assert!(
            flowing < 0.9,
            "φ={phi}°: a {:.1}° heap (limit {limit:.1}°) must flow, retained {flowing}",
            limit + 18.0
        );
    }
}

#[test]
fn the_plane_strain_friction_angle_matches_its_closed_form_values() {
    // sin φ_ps = 2 √3 sin φ / (3 − sin φ): 30° -> asin(0.6928) = 43.85°.
    assert_close(plane_strain_friction_angle_deg(30.0), 43.85, 1e-3, "φ=30°");
    // Weak sand: φ_ps stays close to φ (small-angle limit 2 √3 / 3 = 1.155 φ).
    assert!(plane_strain_friction_angle_deg(5.0) > 5.0);
    assert!(plane_strain_friction_angle_deg(5.0) < 6.5);
    // The cone is never reached in plane strain once 2 √3 sin φ ≥ 3 − sin φ,
    // i.e. φ ≥ 42.2°: the angle is undefined (NaN) there.
    assert!(plane_strain_friction_angle_deg(42.0).is_finite());
    assert!(plane_strain_friction_angle_deg(43.0).is_nan());
}

// ---------------------------------------------------------------------------
// Collapse of a column
// ---------------------------------------------------------------------------

#[test]
fn a_column_collapses_spreads_and_settles() {
    let mut sim = slab_column(30.0, 0.12, 0.18, SPACING);
    let mass = sim.total_mass();
    let initial_half_width = half_width(&sim);
    let initial_top = max_height(&sim);
    let mut peak = 0.0f64;
    for _ in 0..12 {
        assert!(sim.try_step(0.05).is_ok());
        peak = peak.max(sim.kinetic_energy());
    }
    assert_eq!(sim.total_mass(), mass);
    assert!(
        max_height(&sim) < 0.6 * initial_top,
        "top {} of {initial_top}",
        max_height(&sim)
    );
    assert!(
        half_width(&sim) > 2.5 * initial_half_width,
        "half-width {}",
        half_width(&sim)
    );
    assert!(peak > 0.01, "the collapse must release energy: {peak}");
    assert!(
        sim.kinetic_energy() < 1.0e-3 * peak,
        "not at rest: {} vs {peak}",
        sim.kinetic_energy()
    );
    assert!(sim.particles.iter().all(|p| p.position[1] >= 0.0));
    // Spreading loosened the sand: dilation was recorded and never decreases.
    assert!(sim.particles.iter().any(|p| p.plastic_strain_vol > 0.0));
    assert!(sim.particles.iter().all(|p| p.plastic_strain_vol >= 0.0));
}

#[test]
fn runout_decreases_and_the_heap_grows_taller_with_the_friction_angle() {
    let mut outcomes = Vec::new();
    for phi in [20.0, 30.0, 40.0] {
        let mut sim = slab_column(phi, 0.12, 0.18, SPACING);
        for _ in 0..12 {
            assert!(sim.try_step(0.05).is_ok());
        }
        outcomes.push((half_width(&sim), max_height(&sim)));
    }
    for pair in outcomes.windows(2) {
        assert!(
            pair[1].0 < pair[0].0 - 0.015,
            "runout must shrink: {outcomes:?}"
        );
        assert!(
            pair[1].1 > pair[0].1 + 0.005,
            "heap must grow taller: {outcomes:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Floor friction
// ---------------------------------------------------------------------------

#[test]
fn a_sliding_block_decelerates_at_mu_times_g() {
    // A block of high-friction sand sliding on a floor with Coulomb friction µ:
    // the floor force is µ M g, so the mean velocity falls as v0 − µ g t.
    let (mu, v0) = (0.4, 1.0);
    let mut sim = slab_column(45.0, 0.2, 0.05, SPACING);
    sim.floor_friction = mu;
    for p in &mut sim.particles {
        p.velocity = [v0, 0.0, 0.0];
    }
    let mass = sim.total_mass();
    let mut t = 0.0;
    for _ in 0..4 {
        assert!(sim.try_step(0.05).is_ok());
        t += 0.05;
        let mean_velocity = sim.total_momentum()[0] / mass;
        assert_close(
            mean_velocity,
            v0 - mu * 9.81 * t,
            0.012,
            &format!("mean velocity at t={t}"),
        );
    }
    // It comes to rest and stays there.
    for _ in 0..4 {
        assert!(sim.try_step(0.05).is_ok());
    }
    assert!((sim.total_momentum()[0] / mass).abs() < 0.01);
}

#[test]
fn a_frictionless_floor_does_not_slow_a_sliding_block_and_a_sticky_one_stops_it() {
    let run = |floor_friction: f64| {
        let mut sim = slab_column(45.0, 0.2, 0.05, SPACING);
        sim.floor_friction = floor_friction;
        for p in &mut sim.particles {
            p.velocity = [1.0, 0.0, 0.0];
        }
        let mass = sim.total_mass();
        for _ in 0..4 {
            assert!(sim.try_step(0.05).is_ok());
        }
        sim.total_momentum()[0] / mass
    };
    assert!(
        run(0.0) > 0.99,
        "a frictionless floor must not brake the block"
    );
    assert!(
        run(f64::INFINITY).abs() < 0.02,
        "a sticky floor must stop it"
    );
}

// ---------------------------------------------------------------------------
// Collapse of a cylinder (full 3-D)
// ---------------------------------------------------------------------------

const CYLINDER_RADIUS: f64 = 0.1;
const CYLINDER_HEIGHT: f64 = 0.2;
const CYLINDER_SPACING: f64 = 0.02;

/// Collapses a cylinder of `friction_angle_deg` sand for two seconds and
/// returns it together with the peak kinetic energy of the collapse.
fn collapse_cylinder(friction_angle_deg: f64) -> (SandSimulation, f64) {
    let mut sim = cylinder_column(
        friction_angle_deg,
        CYLINDER_RADIUS,
        CYLINDER_HEIGHT,
        CYLINDER_SPACING,
    );
    let mut peak = 0.0f64;
    for _ in 0..40 {
        assert!(sim.try_step(0.05).is_ok());
        peak = peak.max(sim.kinetic_energy());
    }
    (sim, peak)
}

/// Mass-weighted first moments `(⟨x⟩, ⟨z⟩)` and second moments `(⟨x²⟩, ⟨z²⟩)`
/// of the particle positions about the vertical axis.
fn horizontal_moments(sim: &SandSimulation) -> ([f64; 2], [f64; 2]) {
    let mut first = [0.0; 2];
    let mut second = [0.0; 2];
    for p in &sim.particles {
        first[0] += p.mass * p.position[0];
        first[1] += p.mass * p.position[2];
        second[0] += p.mass * p.position[0] * p.position[0];
        second[1] += p.mass * p.position[2] * p.position[2];
    }
    let mass = sim.total_mass();
    (first.map(|v| v / mass), second.map(|v| v / mass))
}

#[test]
fn a_cylinder_collapses_axisymmetrically_and_runs_out_less_with_more_friction() {
    let mass =
        cylinder_column(20.0, CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_SPACING).total_mass();
    let (weak, weak_peak) = collapse_cylinder(20.0);
    let (strong, strong_peak) = collapse_cylinder(40.0);
    // Measured with this fixture (800 particles): φ = 20° ends 76 mm tall and
    // 259 mm wide, φ = 40° 99 mm tall and 216 mm wide (from a 200 mm column of
    // radius 100 mm); the centroid stays on the axis to within 1e-15 m.
    for (name, sim, peak) in [("φ=20°", &weak, weak_peak), ("φ=40°", &strong, strong_peak)] {
        assert_eq!(sim.total_mass(), mass, "{name}: mass changed");
        assert!(
            sim.particles
                .iter()
                .all(|p| p.position.iter().all(|c| c.is_finite()) && p.position[1] >= 0.0),
            "{name}: a particle left the domain"
        );
        // The column slumps and spreads, releases its potential energy and
        // comes to rest.
        assert!(
            max_height(sim) < 0.65 * CYLINDER_HEIGHT,
            "{name}: top {}",
            max_height(sim)
        );
        assert!(
            max_radius(sim) > 1.5 * CYLINDER_RADIUS,
            "{name}: radius {}",
            max_radius(sim)
        );
        assert!(peak > 0.1, "{name}: peak kinetic energy {peak}");
        assert!(
            sim.kinetic_energy() < 1.0e-3 * peak,
            "{name}: not at rest, {} vs {peak}",
            sim.kinetic_energy()
        );
        // The start is symmetric under x -> -x, z -> -z and x <-> z and so is
        // the scheme: the deposit stays on the axis and is round.
        let (first, second) = horizontal_moments(sim);
        assert!(
            first[0].abs() < 1.0e-9 && first[1].abs() < 1.0e-9,
            "{name}: centroid left the axis {first:?}"
        );
        assert_close(
            second[0],
            second[1],
            1.0e-9,
            &format!("{name}: <x²> vs <z²>"),
        );
        // Spreading loosened the sand.
        assert!(
            sim.particles.iter().any(|p| p.plastic_strain_vol > 0.0),
            "{name}: no dilation recorded"
        );
    }
    // More friction: shorter runout, taller heap.
    assert!(
        max_radius(&weak) > max_radius(&strong) + 0.02,
        "runout must shrink with friction: {} vs {}",
        max_radius(&weak),
        max_radius(&strong)
    );
    assert!(
        max_height(&strong) > max_height(&weak) + 0.01,
        "the heap must grow taller with friction: {} vs {}",
        max_height(&weak),
        max_height(&strong)
    );
}
