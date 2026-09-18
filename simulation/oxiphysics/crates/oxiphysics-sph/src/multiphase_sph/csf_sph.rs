//! Continuum-surface-force (CSF) operators for the 3-D
//! [`MultiphaseSphParticle`] system of
//! [`ColorFunctionSph`](super::types::ColorFunctionSph).
//!
//! All sums run over the full support radius `2h` of the 3-D cubic-spline
//! kernel of smoothing length `h` and use the particle volumes `V_j = m_j / ρ_j`:
//!
//! * colour gradient `∇Cᵢ = Σⱼ Vⱼ (Cⱼ − Cᵢ) ∇ᵢWᵢⱼ`, which points towards the
//!   larger colour value, i.e. into phase 0 (`C = 1`);
//! * unit interface normal `n̂ᵢ = ∇Cᵢ/|∇Cᵢ|`, defined where `|∇Cᵢ| > 0.01/h`;
//! * curvature `κᵢ = −∇·n̂ᵢ` in the form of Morris (2000), see
//!   [`update_normals_and_curvatures`].
//!
//! The surface-tension acceleration `σ κᵢ ∇Cᵢ / ρᵢ` of Brackbill, Kothe & Zemach
//! (1992) is assembled from these by
//! [`ColorFunctionSph::csf_body_force`](super::types::ColorFunctionSph::csf_body_force).

use super::functions::{cubic_kernel_3d, cubic_kernel_grad_3d};
use super::types::MultiphaseSphParticle;
use crate::neighbor::SpatialHash3D;

/// Relative threshold on `|∇C|` (in units of `1/h`) below which a particle is
/// regarded as lying away from every interface and receives no normal.
const NORMAL_THRESHOLD: f64 = 0.01;

/// Smallest density used when forming the particle volume `m / ρ`.
const MIN_DENSITY: f64 = 1.0e-30;

/// `true` for a usable (positive and finite) smoothing length.
#[inline]
fn valid_smoothing_length(h: f64) -> bool {
    h > 0.0 && h.is_finite()
}

/// SPH volume `m / ρ` of a particle.
#[inline]
fn volume(p: &MultiphaseSphParticle) -> f64 {
    p.mass / p.density.max(MIN_DENSITY)
}

/// Separation `x_i − x_j` of two particles and its length.
#[inline]
fn separation(pi: &MultiphaseSphParticle, pj: &MultiphaseSphParticle) -> ([f64; 3], f64) {
    let rij = [
        pi.position[0] - pj.position[0],
        pi.position[1] - pj.position[1],
        pi.position[2] - pj.position[2],
    ];
    let r = (rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]).sqrt();
    (rij, r)
}

/// Neighbour lists of `particles` within the kernel support radius `2h`
/// (self excluded, indices sorted).
///
/// A non-positive or non-finite `h` yields empty lists.
pub(super) fn support_neighbors(particles: &[MultiphaseSphParticle], h: f64) -> Vec<Vec<usize>> {
    let radius = 2.0 * h;
    if !(radius > 0.0 && radius.is_finite()) {
        return vec![Vec::new(); particles.len()];
    }
    let positions: Vec<[f64; 3]> = particles.iter().map(|p| p.position).collect();
    SpatialHash3D::all_neighbors(&positions, radius)
}

/// Colour gradient `∇Cᵢ = Σⱼ Vⱼ (Cⱼ − Cᵢ) ∇ᵢWᵢⱼ` of particle `i`, summed over the
/// particles listed in `candidates`.
///
/// Entries equal to `i`, entries outside `particles`, coincident particles and
/// particles beyond the support radius `2h` contribute nothing.  The zero vector
/// is returned for an out-of-range `i` or a non-positive or non-finite `h`.
pub(super) fn colour_gradient_over<I>(
    particles: &[MultiphaseSphParticle],
    i: usize,
    candidates: I,
    h: f64,
) -> [f64; 3]
where
    I: IntoIterator<Item = usize>,
{
    let mut grad = [0.0_f64; 3];
    let Some(pi) = particles.get(i) else {
        return grad;
    };
    if !valid_smoothing_length(h) {
        return grad;
    }
    for j in candidates {
        if j == i {
            continue;
        }
        let Some(pj) = particles.get(j) else {
            continue;
        };
        let (rij, r) = separation(pi, pj);
        if !(r > 1.0e-14 && r < 2.0 * h) {
            continue;
        }
        let grad_w = cubic_kernel_grad_3d(rij, r, h);
        let weight = volume(pj) * (pj.color_function - pi.color_function);
        for (g, w) in grad.iter_mut().zip(grad_w) {
            *g += weight * w;
        }
    }
    grad
}

/// Interface normals `n̂ᵢ = ∇Cᵢ/|∇Cᵢ|` and curvatures `κᵢ = −∇·n̂ᵢ` of all
/// particles, stored in [`MultiphaseSphParticle::interface_normal`] and
/// [`MultiphaseSphParticle::curvature`].
///
/// The normal exists only where `|∇Cᵢ| > 0.01/h`; every other particle gets the
/// zero normal and `κ = 0`.  Following Morris (2000) the divergence is taken
/// over the neighbours that have a normal and normalised by the SPH estimate of
/// unity over that same set,
///
/// `κᵢ = −[Σⱼ Vⱼ (n̂ⱼ − n̂ᵢ)·∇ᵢWᵢⱼ] / [Vᵢ W(0) + Σⱼ Vⱼ Wᵢⱼ]`,
///
/// so the normal-free particles outside the interface band do not count as
/// "zero normals" (which would produce a large spurious curvature at the edge of
/// the band, even for a flat interface).  The normalisation also makes `κ`
/// independent of the kernel's normalisation constant.  `κ` is positive where
/// phase 0 is convex (`2/R` on a sphere of radius `R`, `1/R` on a circle of
/// radius `R` in a plane of particles) and vanishes on a flat interface.
///
/// A non-positive or non-finite `h` clears the normals and curvatures.
pub(super) fn update_normals_and_curvatures(particles: &mut [MultiphaseSphParticle], h: f64) {
    if !valid_smoothing_length(h) {
        for p in particles.iter_mut() {
            p.curvature = 0.0;
            p.interface_normal = [0.0; 3];
        }
        return;
    }
    let neighbors = support_neighbors(particles, h);
    let threshold = NORMAL_THRESHOLD / h;
    let normals: Vec<Option<[f64; 3]>> = (0..particles.len())
        .map(|i| {
            let grad = colour_gradient_over(particles, i, neighbors[i].iter().copied(), h);
            let magnitude = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();
            (magnitude > threshold).then(|| {
                [
                    grad[0] / magnitude,
                    grad[1] / magnitude,
                    grad[2] / magnitude,
                ]
            })
        })
        .collect();
    let self_weight = cubic_kernel_3d(0.0, h);
    let curvatures: Vec<f64> = particles
        .iter()
        .enumerate()
        .map(|(i, pi)| {
            let Some(n_i) = normals[i] else {
                return 0.0;
            };
            let mut divergence = 0.0_f64;
            let mut unity = volume(pi) * self_weight;
            for &j in &neighbors[i] {
                let (Some(pj), Some(n_j)) = (particles.get(j), normals.get(j).copied().flatten())
                else {
                    continue;
                };
                let (rij, r) = separation(pi, pj);
                if !(r > 1.0e-14 && r < 2.0 * h) {
                    continue;
                }
                let grad_w = cubic_kernel_grad_3d(rij, r, h);
                let dn = [n_j[0] - n_i[0], n_j[1] - n_i[1], n_j[2] - n_i[2]];
                let v_j = volume(pj);
                divergence += v_j * (dn[0] * grad_w[0] + dn[1] * grad_w[1] + dn[2] * grad_w[2]);
                unity += v_j * cubic_kernel_3d(r, h);
            }
            if unity > 0.0 {
                -divergence / unity
            } else {
                0.0
            }
        })
        .collect();
    for ((p, normal), kappa) in particles.iter_mut().zip(normals).zip(curvatures) {
        p.interface_normal = normal.unwrap_or([0.0; 3]);
        p.curvature = kappa;
    }
}
