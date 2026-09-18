//! Regression tests for the 3-D colour-function SPH operators of
//! [`ColorFunctionSph`]: full-support colour smoothing and gradient, interface
//! normals, the curvature `κ = −∇·n̂` and the continuum surface force that is
//! built on it.

use super::functions::cubic_spline_kernel;
use super::types::{ColorFunctionSph, MultiphaseSphParticle};
use std::f64::consts::PI;

const RHO_L: f64 = 1000.0;
const RHO_G: f64 = 1.225;
const SIGMA: f64 = 0.072;

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `n³` lattice of spacing `dx` centred on the origin (`n` odd puts a particle
/// at the origin); phase 0 (liquid) inside the sphere of radius `radius`,
/// phase 1 (gas) outside.  Masses follow the phase densities, `m = ρ dx³`.
fn sphere_lattice(n: usize, dx: f64, radius: f64) -> Vec<MultiphaseSphParticle> {
    let offset = 0.5 * (n as f64 - 1.0) * dx;
    let mut particles = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let pos = [
                    i as f64 * dx - offset,
                    j as f64 * dx - offset,
                    k as f64 * dx - offset,
                ];
                let (phase, rho) = if norm(pos) < radius {
                    (0, RHO_L)
                } else {
                    (1, RHO_G)
                };
                particles.push(MultiphaseSphParticle::new(
                    pos,
                    phase,
                    rho * dx * dx * dx,
                    rho,
                ));
            }
        }
    }
    particles
}

/// Particles of the interface band, i.e. those that received a normal.
fn band(particles: &[MultiphaseSphParticle]) -> Vec<usize> {
    (0..particles.len())
        .filter(|&i| norm(particles[i].interface_normal) > 0.5)
        .collect()
}

/// Colour-gradient-weighted mean curvature of the interface band.
fn weighted_mean_curvature(
    csf: &ColorFunctionSph,
    particles: &[MultiphaseSphParticle],
    members: &[usize],
) -> f64 {
    let (mut weighted, mut weight) = (0.0_f64, 0.0_f64);
    for &i in members {
        let magnitude = norm(csf.color_gradient(particles, i));
        weighted += particles[i].curvature * magnitude;
        weight += magnitude;
    }
    assert!(weight > 0.0, "the band must carry a colour gradient");
    weighted / weight
}

// ---------------------------------------------------------------------------
// Full-support colour smoothing and gradient
// ---------------------------------------------------------------------------

#[test]
fn color_gradient_uses_the_full_kernel_support() {
    // Two particles 1.5 h apart: inside the support radius 2h but beyond h, so
    // the old cutoff at h dropped the interaction entirely.
    let h = 0.1;
    let csf = ColorFunctionSph::new(SIGMA, h);
    let mass = 2.0e-3;
    let particles = vec![
        MultiphaseSphParticle::new([0.0, 0.0, 0.0], 0, mass, RHO_L),
        MultiphaseSphParticle::new([1.5 * h, 0.0, 0.0], 1, mass, RHO_G),
    ];
    // dW/dr of the 3-D cubic spline at q = 1.5 is -0.75 (2 - q)^2 / (pi h^4).
    let dw_dr = -0.75 * 0.25 / (PI * h.powi(4));
    // ∇C_0 = V_1 (C_1 - C_0) dW/dr (x_0 - x_1)/r with C_1 - C_0 = -1 and
    // (x_0 - x_1)/r = -x̂: the two minus signs cancel, so the gradient is
    // V_1 dW/dr x̂ with dW/dr < 0, i.e. it points towards -x, away from the gas.
    let expected = (mass / RHO_G) * dw_dr;
    let grad = csf.color_gradient(&particles, 0);
    assert!(expected < 0.0);
    assert!(
        (grad[0] - expected).abs() <= 1e-12 * expected.abs(),
        "{} vs {expected}",
        grad[0]
    );
    assert_eq!(grad[1], 0.0);
    assert_eq!(grad[2], 0.0);
    // Beyond 2h nothing is left.
    let far = vec![
        MultiphaseSphParticle::new([0.0, 0.0, 0.0], 0, mass, RHO_L),
        MultiphaseSphParticle::new([2.0 * h * 1.0001, 0.0, 0.0], 1, mass, RHO_G),
    ];
    assert_eq!(csf.color_gradient(&far, 0), [0.0; 3]);
}

#[test]
fn smooth_color_sees_the_neighbours_out_to_two_smoothing_lengths() {
    let h = 0.1;
    let csf = ColorFunctionSph::new(SIGMA, h);
    let mass = 1.0e-3;
    let particles = vec![
        MultiphaseSphParticle::new([0.0; 3], 0, mass, RHO_L),
        MultiphaseSphParticle::new([1.5 * h, 0.0, 0.0], 1, mass, RHO_G),
    ];
    let v_l = mass / RHO_L;
    let v_g = mass / RHO_G;
    let w_self = cubic_spline_kernel(0.0, h);
    let w_pair = cubic_spline_kernel(1.5 * h, h);
    let expected = v_l * w_self / (v_l * w_self + v_g * w_pair);
    let smoothed = csf.smooth_color(&particles, 0);
    assert!(smoothed < 1.0, "the gas neighbour at 1.5 h must contribute");
    assert!(
        (smoothed - expected).abs() < 1e-12,
        "{smoothed} vs {expected}"
    );
}

// ---------------------------------------------------------------------------
// Curvature
// ---------------------------------------------------------------------------

#[test]
fn curvature_of_a_liquid_sphere_is_two_over_the_radius() {
    let (n, dx, radius) = (21, 0.01, 0.06);
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let mut particles = sphere_lattice(n, dx, radius);
    csf.compute_curvatures(&mut particles);
    let members = band(&particles);
    assert!(members.len() > 200, "band of {} particles", members.len());
    let mean = weighted_mean_curvature(&csf, &particles, &members);
    let expected = 2.0 / radius;
    assert!(
        (mean - expected).abs() < 0.08 * expected,
        "mean interface curvature {mean} vs 2/R = {expected}"
    );
    // Almost every band particle sees a convex liquid.
    let convex = members
        .iter()
        .filter(|&&i| particles[i].curvature > 0.0)
        .count();
    assert!(
        convex * 10 >= members.len() * 9,
        "{convex} of {}",
        members.len()
    );
}

#[test]
fn curvature_of_a_liquid_disc_in_a_plane_is_the_inverse_radius() {
    // A disc of radius R in the plane z = 0 has κ = 1/R; the normalised
    // divergence is insensitive to the 3-D kernel's normalisation.
    let (n, dx, radius) = (41, 0.02, 0.2);
    let offset = 0.5 * (n as f64 - 1.0) * dx;
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let mut particles = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let pos = [i as f64 * dx - offset, j as f64 * dx - offset, 0.0];
            let (phase, rho) = if norm(pos) < radius {
                (0, RHO_L)
            } else {
                (1, RHO_G)
            };
            particles.push(MultiphaseSphParticle::new(pos, phase, rho * dx * dx, rho));
        }
    }
    csf.compute_curvatures(&mut particles);
    let members = band(&particles);
    assert!(members.len() > 40);
    let mean = weighted_mean_curvature(&csf, &particles, &members);
    let expected = 1.0 / radius;
    assert!(
        (mean - expected).abs() < 0.10 * expected,
        "mean interface curvature {mean} vs 1/R = {expected}"
    );
}

#[test]
fn flat_interface_has_no_curvature() {
    // Half-width 10 dx: the central particles and the neighbours of their
    // neighbours (within 2 x 3 dx) all keep complete kernel neighbourhoods.
    let (n, dx) = (21, 0.01);
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let offset = 0.5 * (n as f64 - 1.0) * dx;
    let mut particles = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let pos = [
                    i as f64 * dx - offset,
                    j as f64 * dx - offset,
                    k as f64 * dx - offset,
                ];
                // The interface is the plane z = 0.5 dx (between two layers).
                let (phase, rho) = if pos[2] < 0.25 * dx {
                    (0, RHO_L)
                } else {
                    (1, RHO_G)
                };
                particles.push(MultiphaseSphParticle::new(
                    pos,
                    phase,
                    rho * dx * dx * dx,
                    rho,
                ));
            }
        }
    }
    csf.compute_curvatures(&mut particles);
    let mut checked = 0;
    for p in &particles {
        let central = p.position[0].abs() < 3.0 * dx && p.position[1].abs() < 3.0 * dx;
        if central && p.position[2].abs() < 2.5 * dx {
            assert!(
                p.curvature.abs() < 1e-8 / csf.h,
                "flat interface curvature {} at {:?}",
                p.curvature,
                p.position
            );
            if norm(p.interface_normal) > 0.5 {
                // Normals point into the liquid (towards -z).
                assert!(p.interface_normal[2] < -0.999, "{:?}", p.interface_normal);
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "only {checked} band particles were checked");
}

#[test]
fn single_phase_region_has_neither_normals_nor_curvature() {
    let (n, dx) = (9, 0.01);
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let mut particles = sphere_lattice(n, dx, 1.0);
    for p in particles.iter_mut() {
        p.curvature = 7.0;
        p.interface_normal = [0.0, 0.0, 1.0];
    }
    csf.compute_curvatures(&mut particles);
    for p in &particles {
        assert_eq!(p.curvature, 0.0);
        assert_eq!(p.interface_normal, [0.0; 3]);
    }
}

#[test]
fn curvature_does_not_depend_on_the_storage_order() {
    let (n, dx, radius) = (13, 0.01, 0.04);
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let mut forward = sphere_lattice(n, dx, radius);
    let mut reversed: Vec<MultiphaseSphParticle> = forward.iter().rev().cloned().collect();
    csf.compute_curvatures(&mut forward);
    csf.compute_curvatures(&mut reversed);
    let count = forward.len();
    let mut scale = 0.0_f64;
    for (i, p) in forward.iter().enumerate() {
        let q = &reversed[count - 1 - i];
        scale = scale.max(p.curvature.abs());
        assert!(
            (p.curvature - q.curvature).abs() <= 1e-9 * (1.0 + p.curvature.abs()),
            "particle {i}: {} vs {}",
            p.curvature,
            q.curvature
        );
        for k in 0..3 {
            assert!((p.interface_normal[k] - q.interface_normal[k]).abs() < 1e-9);
        }
    }
    assert!(scale > 1.0, "the sphere must carry curvature: {scale}");
}

#[test]
fn compute_curvatures_handles_degenerate_input() {
    // Empty and single-particle systems.
    let csf = ColorFunctionSph::new(SIGMA, 0.02);
    let mut none: Vec<MultiphaseSphParticle> = Vec::new();
    csf.compute_curvatures(&mut none);
    let mut one = vec![MultiphaseSphParticle::new([0.0; 3], 0, 1e-3, RHO_L)];
    one[0].curvature = 3.0;
    csf.compute_curvatures(&mut one);
    assert_eq!(one[0].curvature, 0.0);
    // A non-positive, NaN or infinite smoothing length clears the fields.
    for h in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let mut particles = sphere_lattice(7, 0.01, 0.02);
        for p in particles.iter_mut() {
            p.curvature = 5.0;
            p.interface_normal = [1.0, 0.0, 0.0];
        }
        ColorFunctionSph::new(SIGMA, h).compute_curvatures(&mut particles);
        for p in &particles {
            assert_eq!(p.curvature, 0.0, "h = {h}");
            assert_eq!(p.interface_normal, [0.0; 3], "h = {h}");
        }
    }
    // Coincident particles of different phase and non-finite positions do not
    // poison the result.
    let mut messy = sphere_lattice(7, 0.01, 0.02);
    messy.push(MultiphaseSphParticle::new([0.0; 3], 1, 1e-6, RHO_G));
    messy.push(MultiphaseSphParticle::new(
        [f64::NAN, 0.0, 0.0],
        0,
        1e-6,
        RHO_L,
    ));
    ColorFunctionSph::new(SIGMA, 0.015).compute_curvatures(&mut messy);
    assert!(messy.iter().all(|p| p.curvature.is_finite()));
}

// ---------------------------------------------------------------------------
// Continuum surface force
// ---------------------------------------------------------------------------

#[test]
fn csf_body_force_is_dead_until_the_curvature_is_computed() {
    let (n, dx, radius) = (21, 0.01, 0.06);
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let mut particles = sphere_lattice(n, dx, radius);
    // Curvature still at its initial value 0 for every particle.
    let probe = (0..particles.len())
        .find(|&i| {
            let r = norm(particles[i].position);
            (radius - 1.5 * dx..radius).contains(&r)
        })
        .expect("a particle just inside the interface");
    assert_eq!(csf.csf_body_force(&particles, probe), [0.0; 3]);
    csf.compute_curvatures(&mut particles);
    let force = csf.csf_body_force(&particles, probe);
    assert!(norm(force) > 0.0, "force {force:?}");
}

#[test]
fn csf_body_force_pulls_a_liquid_sphere_inwards_with_the_laplace_force() {
    let (n, dx, radius) = (21, 0.01, 0.06);
    let csf = ColorFunctionSph::new(SIGMA, 1.5 * dx);
    let mut particles = sphere_lattice(n, dx, radius);
    csf.compute_curvatures(&mut particles);
    let members = band(&particles);

    // Every band particle is pulled inwards, along the interface normal.
    let mut radial_force = 0.0_f64;
    let mut net = [0.0_f64; 3];
    for &i in &members {
        let p = &particles[i];
        let r = norm(p.position);
        let inward = [-p.position[0] / r, -p.position[1] / r, -p.position[2] / r];
        // The normal points into the liquid.
        assert!(
            dot(p.interface_normal, inward) > 0.9,
            "normal {:?}",
            p.interface_normal
        );
        let f = csf.csf_body_force(&particles, i);
        if p.curvature > 0.0 {
            let magnitude = norm(f);
            assert!(magnitude > 0.0);
            assert!(
                dot(f, inward) / magnitude > 0.9,
                "force {f:?} at {:?}",
                p.position
            );
        }
        // Force on the particle's mass: m f = σ κ V ∇C.
        for k in 0..3 {
            net[k] += p.mass * f[k];
        }
        radial_force += p.mass * dot(f, inward);
    }
    // The total inward force on the drop is σ κ A = 2σ/R · 4πR² = 8π σ R.
    let expected = 8.0 * PI * SIGMA * radius;
    assert!(
        radial_force > 0.95 * expected && radial_force < 1.05 * expected,
        "inward CSF force {radial_force} N vs Laplace estimate {expected} N"
    );
    // The lattice has cubic symmetry, so the net force on the drop vanishes.
    assert!(norm(net) < 1e-9 * radial_force, "net force {net:?}");

    // Particles deep inside the liquid and far out in the gas feel nothing.
    let centre = particles
        .iter()
        .position(|p| norm(p.position) < 1e-12)
        .expect("a particle at the origin");
    assert_eq!(csf.csf_body_force(&particles, centre), [0.0; 3]);
    assert_eq!(csf.csf_body_force(&particles, 0), [0.0; 3]);
}

#[test]
fn csf_body_force_scales_linearly_with_the_surface_tension() {
    let (n, dx, radius) = (13, 0.01, 0.04);
    let h = 1.5 * dx;
    let mut particles = sphere_lattice(n, dx, radius);
    ColorFunctionSph::new(0.0, h).compute_curvatures(&mut particles);
    let members = band(&particles);
    let one = ColorFunctionSph::new(0.05, h);
    let three = ColorFunctionSph::new(0.15, h);
    for &i in &members {
        let f1 = one.csf_body_force(&particles, i);
        let f3 = three.csf_body_force(&particles, i);
        for k in 0..3 {
            assert!((f3[k] - 3.0 * f1[k]).abs() <= 1e-12 * (1.0 + f1[k].abs()));
        }
    }
}
