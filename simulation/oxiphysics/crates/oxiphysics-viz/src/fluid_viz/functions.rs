//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[inline]
pub(super) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub(super) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub(super) fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}
#[inline]
pub(super) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub(super) fn len3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}
#[inline]
pub(super) fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = len3(v);
    if l < 1e-12 {
        [0.0; 3]
    } else {
        scale3(v, 1.0 / l)
    }
}
/// Linear interpolation between two RGBA colours.
pub fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}
/// Compute the Q-criterion from the velocity gradient Jacobian (row-major 3×3).
///
/// Q = 0.5 * (||Ω||_F² - ||S||_F²)
///
/// where Ω = antisymmetric part, S = symmetric part of J.
/// Positive Q indicates vortex cores.
pub fn compute_q_criterion(j: [f64; 9]) -> f64 {
    let mut norm_s2 = 0.0;
    let mut norm_omega2 = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            let j_ik = j[i * 3 + k];
            let j_ki = j[k * 3 + i];
            let s = 0.5 * (j_ik + j_ki);
            let omega = 0.5 * (j_ik - j_ki);
            norm_s2 += s * s;
            norm_omega2 += omega * omega;
        }
    }
    0.5 * (norm_omega2 - norm_s2)
}
/// Compute the λ2 criterion for vortex identification.
///
/// λ2 is the second eigenvalue of S²+Ω². Negative λ2 indicates a vortex core.
/// This function returns an approximate value using the Frobenius norms.
pub fn compute_lambda2_approx(j: [f64; 9]) -> f64 {
    -compute_q_criterion(j)
}
/// Compute the turbulent kinetic energy (TKE) from a velocity ensemble.
///
/// TKE = 0.5 * (σ_u² + σ_v² + σ_w²)
pub fn compute_turbulent_kinetic_energy(velocities: &[[f64; 3]]) -> f64 {
    let n = velocities.len();
    if n == 0 {
        return 0.0;
    }
    let mean: [f64; 3] = {
        let mut s = [0.0f64; 3];
        for v in velocities {
            s[0] += v[0];
            s[1] += v[1];
            s[2] += v[2];
        }
        [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64]
    };
    let mut variance = [0.0f64; 3];
    for v in velocities {
        for k in 0..3 {
            let d = v[k] - mean[k];
            variance[k] += d * d;
        }
    }
    for v in variance.iter_mut() {
        *v /= n as f64;
    }
    0.5 * (variance[0] + variance[1] + variance[2])
}
/// Compute the 1-D turbulent kinetic energy spectrum of a velocity record.
///
/// `velocities` is a uniformly sampled 1-D signal, for example the velocity
/// probed at equally spaced points along a line.  Each component has its mean
/// removed and is transformed with a real FFT (`oxifft::rfft`).  With
/// `û_c(k)` the transform of component `c`, the energy of the integer
/// wavenumber `k` (cycles per record, `k = 0..=n/2`) is
///
/// `E(k) = 0.5 * Σ_c |û_c(k)|² / n²`,
///
/// doubled for `0 < k < n/2` because the negative wavenumbers fold onto the
/// positive ones.  By Parseval's theorem the energies add up to the
/// fluctuating kinetic energy `compute_turbulent_kinetic_energy(velocities)`.
///
/// The wavenumbers `[0, n/2]` are split into `num_bins` equal bins and the
/// result lists `(bin centre, summed energy)` for each of them.  An empty
/// list is returned for an empty record or `num_bins == 0`.
pub fn compute_tke_spectrum(velocities: &[[f64; 3]], num_bins: usize) -> Vec<(f64, f64)> {
    let n = velocities.len();
    if n == 0 || num_bins == 0 {
        return Vec::new();
    }
    let max_k = n / 2;
    let bin_size = max_k as f64 / num_bins as f64;
    let mut bins = vec![0.0f64; num_bins];
    if n >= 2 {
        let norm = 0.5 / (n as f64 * n as f64);
        for component in 0..3 {
            let mean = velocities.iter().map(|v| v[component]).sum::<f64>() / n as f64;
            let fluctuation: Vec<f64> = velocities.iter().map(|v| v[component] - mean).collect();
            for (k, coefficient) in oxifft::rfft(&fluctuation).iter().enumerate() {
                // The DC and (even n) Nyquist coefficients have no mirror image.
                let folding = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
                let bin = ((k as f64 / bin_size) as usize).min(num_bins - 1);
                let power = coefficient.re * coefficient.re + coefficient.im * coefficient.im;
                bins[bin] += folding * norm * power;
            }
        }
    }
    (0..num_bins)
        .map(|i| (i as f64 * bin_size + bin_size * 0.5, bins[i]))
        .collect()
}
/// Compute a voxelized density map from particle positions and masses.
///
/// Uses a simple nearest-cell splatting approach.
pub fn compute_density_map(
    positions: &[[f64; 3]],
    masses: &[f64],
    grid_res: usize,
    domain_size: f64,
    smoothing_radius: f64,
) -> Vec<f64> {
    let n = grid_res;
    let h = domain_size / n as f64;
    let mut density = vec![0.0f64; n * n * n];
    for (pos, &mass) in positions.iter().zip(masses.iter()) {
        let ix = (pos[0] / domain_size * n as f64).clamp(0.0, (n - 1) as f64) as usize;
        let iy = (pos[1] / domain_size * n as f64).clamp(0.0, (n - 1) as f64) as usize;
        let iz = (pos[2] / domain_size * n as f64).clamp(0.0, (n - 1) as f64) as usize;
        let r_cells = (smoothing_radius / h) as i64 + 1;
        for di in -r_cells..=r_cells {
            for dj in -r_cells..=r_cells {
                for dk in -r_cells..=r_cells {
                    let gx = ix as i64 + di;
                    let gy = iy as i64 + dj;
                    let gz = iz as i64 + dk;
                    if gx < 0
                        || gy < 0
                        || gz < 0
                        || gx >= n as i64
                        || gy >= n as i64
                        || gz >= n as i64
                    {
                        continue;
                    }
                    let cx = gx as f64 * h + h * 0.5;
                    let cy = gy as f64 * h + h * 0.5;
                    let cz = gz as f64 * h + h * 0.5;
                    let dx = pos[0] - cx;
                    let dy = pos[1] - cy;
                    let dz = pos[2] - cz;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < smoothing_radius {
                        let w = 1.0 - dist / smoothing_radius;
                        let gidx = gx as usize * n * n + gy as usize * n + gz as usize;
                        density[gidx] += mass * w;
                    }
                }
            }
        }
    }
    density
}
/// Compute the gradient of a scalar field at a grid cell using central differences.
pub fn field_gradient(
    field: &[f64],
    ix: usize,
    iy: usize,
    iz: usize,
    res: usize,
    h: f64,
) -> [f64; 3] {
    let idx = |x: usize, y: usize, z: usize| x * res * res + y * res + z;
    let n = res;
    let dx = if ix + 1 < n && ix > 0 {
        (field[idx(ix + 1, iy, iz)] - field[idx(ix - 1, iy, iz)]) / (2.0 * h)
    } else {
        0.0
    };
    let dy = if iy + 1 < n && iy > 0 {
        (field[idx(ix, iy + 1, iz)] - field[idx(ix, iy - 1, iz)]) / (2.0 * h)
    } else {
        0.0
    };
    let dz = if iz + 1 < n && iz > 0 {
        (field[idx(ix, iy, iz + 1)] - field[idx(ix, iy, iz - 1)]) / (2.0 * h)
    } else {
        0.0
    };
    [dx, dy, dz]
}
/// Compute the curvature `κ = ∇ · (∇φ / |∇φ|)` of the level sets of a field.
///
/// `κ` is the sum of the two principal curvatures of the level surface through
/// the grid node `(ix, iy, iz)`, so a sphere of radius `R` has `κ = 2 / R`
/// (positive where `φ` grows away from the centre).  It is evaluated with
/// second-order central differences on the `res³` grid (indexed
/// `x * res² + y * res + z`, spacing `h`):
///
/// `κ = [φxx (φy² + φz²) + φyy (φx² + φz²) + φzz (φx² + φy²)
///       - 2 (φx φy φxy + φx φz φxz + φy φz φyz)] / |∇φ|³`.
///
/// Returns `0.0` on the boundary layer of nodes (which lack a central
/// stencil), where `|∇φ| < 1e-12`, and when `phi` holds fewer than `res³`
/// samples.
pub fn level_set_mean_curvature(
    phi: &[f64],
    ix: usize,
    iy: usize,
    iz: usize,
    res: usize,
    h: f64,
) -> f64 {
    let interior = |i: usize| i >= 1 && i + 1 < res;
    let has_samples = res.checked_pow(3).is_some_and(|total| phi.len() >= total);
    if !(interior(ix) && interior(iy) && interior(iz) && has_samples) {
        return 0.0;
    }
    let at = |x: usize, y: usize, z: usize| phi[x * res * res + y * res + z];
    let centre = at(ix, iy, iz);
    let (xp, xm) = (at(ix + 1, iy, iz), at(ix - 1, iy, iz));
    let (yp, ym) = (at(ix, iy + 1, iz), at(ix, iy - 1, iz));
    let (zp, zm) = (at(ix, iy, iz + 1), at(ix, iy, iz - 1));

    let inv_2h = 0.5 / h;
    let (px, py, pz) = ((xp - xm) * inv_2h, (yp - ym) * inv_2h, (zp - zm) * inv_2h);
    let inv_h2 = 1.0 / (h * h);
    let pxx = (xp - 2.0 * centre + xm) * inv_h2;
    let pyy = (yp - 2.0 * centre + ym) * inv_h2;
    let pzz = (zp - 2.0 * centre + zm) * inv_h2;
    let inv_4h2 = 0.25 * inv_h2;
    let pxy = (at(ix + 1, iy + 1, iz) - at(ix + 1, iy - 1, iz) - at(ix - 1, iy + 1, iz)
        + at(ix - 1, iy - 1, iz))
        * inv_4h2;
    let pxz = (at(ix + 1, iy, iz + 1) - at(ix + 1, iy, iz - 1) - at(ix - 1, iy, iz + 1)
        + at(ix - 1, iy, iz - 1))
        * inv_4h2;
    let pyz = (at(ix, iy + 1, iz + 1) - at(ix, iy + 1, iz - 1) - at(ix, iy - 1, iz + 1)
        + at(ix, iy - 1, iz - 1))
        * inv_4h2;

    let grad_sq = px * px + py * py + pz * pz;
    let grad_len = grad_sq.sqrt();
    if grad_len < 1e-12 {
        return 0.0;
    }
    let numerator =
        pxx * (py * py + pz * pz) + pyy * (px * px + pz * pz) + pzz * (px * px + py * py)
            - 2.0 * (px * py * pxy + px * pz * pxz + py * pz * pyz);
    numerator / (grad_len * grad_sq)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_visualization::LicRenderer;
    use crate::fluid_viz::Bubble;
    use crate::fluid_viz::BubbleRenderer;
    use crate::fluid_viz::FluidStats;
    use crate::fluid_viz::FoamRenderer;
    use crate::fluid_viz::FreeSurfaceExtractor;
    use crate::fluid_viz::IsosurfaceViz;
    use crate::fluid_viz::Metaball;
    use crate::fluid_viz::MetaballField;
    use crate::fluid_viz::ParticleSplat;
    use crate::fluid_viz::PathlineIntegrator;
    use crate::fluid_viz::PressureColormap;
    use crate::fluid_viz::Splat;
    use crate::fluid_viz::StreaklineRenderer;
    use crate::fluid_viz::StreamlineRenderer;
    use crate::fluid_viz::VelocityArrows;
    use crate::fluid_viz::VolumeSlice;
    use crate::fluid_viz::VorticityViz;
    use crate::fluid_viz::WaveViz;
    #[test]
    fn test_velocity_arrow_below_min() {
        let mut va = VelocityArrows::new(1.0, 10.0);
        va.min_magnitude = 5.0;
        assert!(va.make_arrow([0.0; 3], [1.0, 0.0, 0.0]).is_none());
    }
    #[test]
    fn test_velocity_arrow_above_min() {
        let va = VelocityArrows::new(1.0, 10.0);
        assert!(va.make_arrow([0.0; 3], [10.0, 0.0, 0.0]).is_some());
    }
    #[test]
    fn test_velocity_arrow_tip_displaced() {
        let va = VelocityArrows::new(2.0, 10.0);
        let arrow = va.make_arrow([0.0; 3], [1.0, 0.0, 0.0]).unwrap();
        assert!((arrow.tip[0] - 2.0).abs() < 1e-9);
    }
    #[test]
    fn test_lerp_color_t0() {
        let a = [1.0_f32, 0.0, 0.0, 1.0];
        let b = [0.0_f32, 1.0, 0.0, 1.0];
        let c = lerp_color(a, b, 0.0);
        assert!((c[0] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_lerp_color_t1() {
        let a = [1.0_f32, 0.0, 0.0, 1.0];
        let b = [0.0_f32, 1.0, 0.0, 1.0];
        let c = lerp_color(a, b, 1.0);
        assert!((c[1] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_pressure_iso_contours() {
        let pc = PressureColormap::new(0.0, 100.0);
        assert_eq!(pc.iso_contours().len(), 11);
    }
    #[test]
    fn test_pressure_map_range() {
        let pc = PressureColormap::new(0.0, 100.0);
        let c0 = pc.map(0.0);
        let c1 = pc.map(100.0);
        assert!(c0 != c1);
    }
    #[test]
    fn test_streamline_trace_one_point() {
        let sr = StreamlineRenderer::new(5, 0.1);
        let sl = sr.trace([0.0; 3], |_| [1.0, 0.0, 0.0]);
        assert!(!sl.points.is_empty());
    }
    #[test]
    fn test_streamline_trace_steps() {
        let sr = StreamlineRenderer::new(10, 0.1);
        let sl = sr.trace([0.0; 3], |_| [1.0, 0.0, 0.0]);
        assert_eq!(sl.points.len(), 11);
    }
    #[test]
    fn test_streamline_stops_zero_vel() {
        let sr = StreamlineRenderer::new(100, 0.1);
        let sl = sr.trace([0.0; 3], |_| [0.0; 3]);
        assert_eq!(sl.points.len(), 1);
    }
    #[test]
    fn test_vorticity_color_zero() {
        let vv = VorticityViz::new(10.0);
        let c = vv.color(0.0);
        assert!((c[2] - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_vorticity_identity_jacobian() {
        let vv = VorticityViz::new(1.0);
        let j = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let w = vv.vorticity_from_jacobian(j);
        assert!(w.iter().all(|x| x.abs() < 1e-12));
    }
    #[test]
    fn test_vortex_core_cells() {
        let vv = VorticityViz::new(1.0);
        let lambda2 = vec![-0.5, 0.1, -0.2, 0.3];
        let cells = vv.vortex_core_cells(&lambda2);
        assert_eq!(cells, vec![0, 2]);
    }
    #[test]
    fn test_splat_gaussian_peak() {
        let ps = ParticleSplat::new(100.0);
        assert!((ps.gaussian_weight(0.0, 1.0) - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_splat_gaussian_decreases() {
        let ps = ParticleSplat::new(100.0);
        assert!(ps.gaussian_weight(0.5, 1.0) > ps.gaussian_weight(1.0, 1.0));
    }
    #[test]
    fn test_iso_edge_interp_midpoint() {
        let iso = IsosurfaceViz::new(0.5, 8, 1.0);
        let p = iso.edge_interp([0.0; 3], 0.0, [1.0, 0.0, 0.0], 1.0);
        assert!((p[0] - 0.5).abs() < 1e-9);
    }
    #[test]
    fn test_volume_slice_extract() {
        let mut vs = VolumeSlice::new(4, 2, 0.5, 0.25);
        let field: Vec<f64> = (0..64).map(|i| i as f64).collect();
        vs.extract(&field, 4, 1.0);
        assert!(vs.pixels.iter().any(|&x| x != 0.0));
    }
    #[test]
    fn test_volume_slice_sample() {
        let mut vs = VolumeSlice::new(4, 2, 0.5, 0.25);
        vs.pixels = vec![1.0; 16];
        let v = vs.sample_bilinear(0.5, 0.5);
        assert!((v - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_fluid_stats_reynolds() {
        let mut fs = FluidStats::new();
        fs.compute_reynolds(1000.0, 1.0, 0.1, 0.001);
        assert!((fs.reynolds - 100000.0).abs() < 1.0);
    }
    #[test]
    fn test_fluid_stats_format() {
        let fs = FluidStats::new();
        assert!(!fs.format().is_empty());
    }
    #[test]
    fn test_wave_fresnel_at_zero() {
        let wv = WaveViz::new(4);
        assert!((wv.fresnel(1.0) - 0.02).abs() < 1e-9);
    }
    #[test]
    fn test_wave_update_normals() {
        let mut wv = WaveViz::new(4);
        let heights: Vec<f64> = (0..16).map(|i| (i as f64).sin()).collect();
        wv.update_normals(&heights, 0.25);
        assert!(
            wv.normal_map
                .iter()
                .any(|n| n[0].abs() > 0.0 || n[2].abs() > 0.0)
        );
    }
    #[test]
    fn test_wave_foam_range() {
        let mut wv = WaveViz::new(4);
        let heights = vec![
            0.0, 0.5, 1.0, 2.0, 0.1, 0.3, 0.8, 1.5, 0.2, 0.4, 0.9, 0.6, 0.7, 1.1, 0.3, 0.5,
        ];
        wv.update_foam(&heights);
        assert!(wv.foam_mask.iter().all(|&f| (0.0..=1.0).contains(&f)));
    }
    #[test]
    fn test_bubble_max_count() {
        let mut br = BubbleRenderer::new(3);
        for _ in 0..5 {
            br.add_bubble(Bubble::new([0.0; 3], 0.01));
        }
        assert_eq!(br.bubbles.len(), 3);
    }
    #[test]
    fn test_bubble_prune() {
        let mut br = BubbleRenderer::new(10);
        br.add_bubble(Bubble::new([0.0, 5.0, 0.0], 0.01));
        br.add_bubble(Bubble::new([0.0, -1.0, 0.0], 0.01));
        br.prune_above_surface(0.0);
        assert_eq!(br.bubbles.len(), 1);
    }
    #[test]
    fn test_bubble_total_internal_reflection() {
        let b = Bubble::new([0.0; 3], 0.01);
        let result = b.refract([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.5);
        let _ = result;
    }
    #[test]
    fn test_pressure_apply_count() {
        let pc = PressureColormap::new(0.0, 100.0);
        let positions = vec![[0.0; 3]; 5];
        let pressures = vec![10.0, 20.0, 50.0, 80.0, 100.0];
        let samples = pc.apply(&positions, &pressures);
        assert_eq!(samples.len(), 5);
    }
    #[test]
    fn test_velocity_arrow_build_field() {
        let va = VelocityArrows::new(1.0, 5.0);
        let origins = vec![[0.0; 3]; 4];
        let vels = vec![[1.0, 0.0, 0.0]; 4];
        let arrows = va.build_field(&origins, &vels);
        assert_eq!(arrows.len(), 4);
    }
    #[test]
    fn test_splat_depth_sort() {
        let ps = ParticleSplat::new(100.0);
        let mut splats = vec![
            Splat {
                position: [0.0; 2],
                radius: 1.0,
                color: [1.0; 4],
                depth: 3.0,
            },
            Splat {
                position: [0.0; 2],
                radius: 1.0,
                color: [1.0; 4],
                depth: 1.0,
            },
            Splat {
                position: [0.0; 2],
                radius: 1.0,
                color: [1.0; 4],
                depth: 2.0,
            },
        ];
        let _ = ps;
        ParticleSplat::depth_sort(&mut splats);
        assert!(splats[0].depth >= splats[1].depth);
        assert!(splats[1].depth >= splats[2].depth);
    }
    #[test]
    fn test_iso_crossings_monotone() {
        let iso = IsosurfaceViz::new(0.5, 4, 1.0);
        let field: Vec<f64> = (0..64).map(|i| i as f64 / 63.0).collect();
        let crossings = iso.extract_crossings(&field);
        assert!(!crossings.is_empty());
    }
    #[test]
    fn test_metaball_field_at_center() {
        let field = MetaballField::new(
            vec![Metaball {
                center: [0.0; 3],
                radius: 1.0,
                strength: 1.0,
            }],
            0.5,
        );
        let v = field.evaluate([0.0, 0.0, 0.0]);
        assert!(v > 1.0);
    }
    #[test]
    fn test_metaball_field_far() {
        let field = MetaballField::new(
            vec![Metaball {
                center: [0.0; 3],
                radius: 0.1,
                strength: 0.1,
            }],
            0.5,
        );
        let v = field.evaluate([100.0, 0.0, 0.0]);
        assert!(v < 0.01);
    }
    #[test]
    fn test_lic_output_size() {
        use crate::flow_visualization::{LicConfig, Vec2, VectorField};
        let lic = LicRenderer::with_config(LicConfig {
            kernel_length: 5,
            step_size: 0.1,
            contrast: 1.0,
        });
        let mut field = VectorField::new_2d(8, 8, 1.0, 1.0, Vec2::zero());
        for j in 0..8 {
            for i in 0..8 {
                field.set(i, j, 0, crate::flow_visualization::Vec3::new(1.0, 0.0, 0.0));
            }
        }
        let noise: Vec<f64> = (0..64).map(|i| i as f64 / 64.0).collect();
        let img = lic.render(&field, Some(&noise));
        assert_eq!(img.len(), 64);
    }
    #[test]
    fn test_lic_uniform_flow_nonzero() {
        use crate::flow_visualization::{LicConfig, Vec2, VectorField};
        let lic = LicRenderer::with_config(LicConfig {
            kernel_length: 10,
            step_size: 0.1,
            contrast: 1.0,
        });
        let mut field = VectorField::new_2d(16, 16, 1.0, 1.0, Vec2::zero());
        for j in 0..16 {
            for i in 0..16 {
                field.set(i, j, 0, crate::flow_visualization::Vec3::new(1.0, 0.0, 0.0));
            }
        }
        let noise: Vec<f64> = (0..256)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let img = lic.render(&field, Some(&noise));
        let nonzero = img.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero > 0);
    }
    #[test]
    fn test_q_criterion_positive_at_vortex() {
        let omega = 2.0_f64;
        let j = [0.0, -omega, 0.0, omega, 0.0, 0.0, 0.0, 0.0, 0.0];
        let q = compute_q_criterion(j);
        assert!(q > 0.0, "Q={q}");
    }
    #[test]
    fn test_q_criterion_negative_at_strain() {
        let j = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let q = compute_q_criterion(j);
        assert!(q < 0.0, "Q={q}");
    }
    #[test]
    fn test_pathline_length_increases() {
        let mut pl = PathlineIntegrator::new(5, 0.1);
        pl.add_seed([0.0; 3]);
        pl.step(|_| [1.0, 0.0, 0.0]);
        pl.step(|_| [1.0, 0.0, 0.0]);
        let path = &pl.paths[0];
        assert_eq!(path.len(), 3);
    }
    #[test]
    fn test_streakline_initial_empty() {
        let sr = StreaklineRenderer::new(20);
        assert!(sr.positions.is_empty());
    }
    #[test]
    fn test_streakline_emit() {
        let mut sr = StreaklineRenderer::new(20);
        sr.emit([1.0, 2.0, 3.0]);
        assert_eq!(sr.positions.len(), 1);
    }
    #[test]
    fn test_streakline_max_length() {
        let mut sr = StreaklineRenderer::new(5);
        for i in 0..10 {
            sr.emit([i as f64, 0.0, 0.0]);
        }
        assert_eq!(sr.positions.len(), 5);
    }
    #[test]
    fn test_free_surface_empty() {
        let fse = FreeSurfaceExtractor::new(4, 1.0);
        let phi = vec![1.0_f64; 64];
        let tris = fse.extract(&phi);
        assert!(tris.is_empty());
    }
    #[test]
    fn test_free_surface_crossings_exist() {
        let fse = FreeSurfaceExtractor::new(4, 1.0);
        let mut phi = vec![1.0_f64; 64];
        phi[0] = -1.0;
        let tris = fse.extract(&phi);
        assert!(!tris.is_empty());
    }
    #[test]
    fn test_tke_spectrum_nonempty() {
        let velocities: Vec<[f64; 3]> = (0..16)
            .map(|i| [(i as f64).sin(), (i as f64 * 0.5).cos(), 0.0])
            .collect();
        let spectrum = compute_tke_spectrum(&velocities, 16);
        assert!(!spectrum.is_empty());
    }
    #[test]
    fn test_tke_nonnegative() {
        let velocities: Vec<[f64; 3]> = (0..8).map(|i| [(i as f64).sin(); 3]).collect();
        let tke = compute_turbulent_kinetic_energy(&velocities);
        assert!(tke >= 0.0);
    }
    #[test]
    fn test_foam_no_low_velocity() {
        let fr = FoamRenderer::new(0.5, 0.8);
        let count = fr.should_foam(0.1);
        assert!(!count);
    }
    #[test]
    fn test_foam_high_velocity() {
        let fr = FoamRenderer::new(0.5, 0.8);
        let count = fr.should_foam(1.0);
        assert!(count);
    }
    #[test]
    fn test_density_map_positive() {
        let positions = vec![[0.5_f64; 3]; 8];
        let masses = vec![1.0_f64; 8];
        let dm = compute_density_map(&positions, &masses, 4, 1.0, 0.3);
        let total: f64 = dm.iter().sum();
        assert!(total > 0.0, "density map should be nonzero");
    }
    #[test]
    fn test_velocity_arrows_all_zero() {
        let mut va = VelocityArrows::new(1.0, 1.0);
        va.min_magnitude = 1e-9;
        let origins = vec![[0.0; 3]; 4];
        let vels = vec![[0.0; 3]; 4];
        let arrows = va.build_field(&origins, &vels);
        assert!(
            arrows.is_empty(),
            "zero velocities should produce no arrows"
        );
    }
    #[test]
    fn test_vorticity_rotation() {
        let vv = VorticityViz::new(10.0);
        let j = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let w = vv.vorticity_from_jacobian(j);
        assert!((w[2] - 2.0).abs() < 1e-10, "w_z={}", w[2]);
    }
    #[test]
    fn test_streamline_trace_many() {
        let sr = StreamlineRenderer::new(5, 0.1);
        let seeds = vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let lines = sr.trace_many(&seeds, |_| [0.5, 0.0, 0.0]);
        assert_eq!(lines.len(), 3);
    }
    #[test]
    fn test_bubble_visible_count() {
        let mut br = BubbleRenderer::new(10);
        let mut b1 = Bubble::new([0.0; 3], 0.1);
        b1.opacity = 0.5;
        let mut b2 = Bubble::new([1.0; 3], 0.1);
        b2.opacity = 0.01;
        br.add_bubble(b1);
        br.add_bubble(b2);
        assert_eq!(br.visible_count(0.1), 1);
    }
    #[test]
    fn test_fluid_stats_cfl() {
        let mut fs = FluidStats::new();
        fs.compute_cfl(10.0, 0.001, 0.1);
        assert!((fs.cfl - 0.1).abs() < 1e-10);
    }
    #[test]
    fn test_fluid_stats_mach() {
        let mut fs = FluidStats::new();
        fs.compute_mach(340.0, 340.0);
        assert!((fs.mach - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_wave_fresnel_increases() {
        let wv = WaveViz::new(4);
        let f0 = wv.fresnel(1.0);
        let f90 = wv.fresnel(0.0);
        assert!(f90 > f0, "fresnel should increase at grazing angle");
    }
    #[test]
    fn test_pressure_colormap_valid_range() {
        let pc = PressureColormap::new(-100.0, 100.0);
        for v in [-100.0, -50.0, 0.0, 50.0, 100.0] {
            let c = pc.map(v);
            for component in c.iter().take(3) {
                assert!(
                    *component >= 0.0 && *component <= 1.0,
                    "component out of range: {component}"
                );
            }
        }
    }

    // ── TKE spectrum ──────────────────────────────────────────────────────

    /// Pseudo-random numbers in `[-1, 1)` from a xorshift generator.
    fn noise(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    #[test]
    fn test_tke_spectrum_single_tone_lands_in_its_wavenumber_bin() {
        let n = 64;
        let velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * 8.0 * i as f64 / n as f64;
                [phase.sin(), 0.0, 0.0]
            })
            .collect();
        // 32 bins over the wavenumbers [0, 32] are one wavenumber wide.
        let spectrum = compute_tke_spectrum(&velocities, 32);
        assert_eq!(spectrum.len(), 32);
        for (bin, &(centre, energy)) in spectrum.iter().enumerate() {
            assert!(
                (centre - (bin as f64 + 0.5)).abs() < 1e-12,
                "bin {bin}: {centre}"
            );
            if bin == 8 {
                // Variance 1/2, so the kinetic energy is 1/4.
                assert!((energy - 0.25).abs() < 1e-12, "k = 8 carries {energy}");
            } else {
                assert!(energy < 1e-20, "bin {bin} carries {energy}");
            }
        }
        let total: f64 = spectrum.iter().map(|&(_, e)| e).sum();
        let tke = compute_turbulent_kinetic_energy(&velocities);
        assert!((total - tke).abs() < 1e-9, "sum {total} vs TKE {tke}");
    }

    #[test]
    fn test_tke_spectrum_separates_tones_per_component_and_ignores_the_mean() {
        let n = 128;
        let tau = 2.0 * std::f64::consts::PI;
        let velocities: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                [
                    5.0 + (tau * 3.0 * t).sin(),
                    -2.0 + 2.0 * (tau * 5.0 * t + 0.3).sin(),
                    7.0 + 0.5 * (tau * 11.0 * t).cos(),
                ]
            })
            .collect();
        // 64 bins over [0, 64] are one wavenumber wide.
        let spectrum = compute_tke_spectrum(&velocities, 64);
        let expected = [(3usize, 0.25), (5, 1.0), (11, 0.0625)];
        for (bin, &(_, energy)) in spectrum.iter().enumerate() {
            let want = expected
                .iter()
                .find(|&&(k, _)| k == bin)
                .map_or(0.0, |&(_, e)| e);
            assert!(
                (energy - want).abs() < 1e-10,
                "bin {bin}: {energy} vs {want}"
            );
        }
        let total: f64 = spectrum.iter().map(|&(_, e)| e).sum();
        let tke = compute_turbulent_kinetic_energy(&velocities);
        assert!((total - tke).abs() < 1e-9, "sum {total} vs TKE {tke}");
    }

    #[test]
    fn test_tke_spectrum_obeys_parseval_for_any_record_length() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for n in [2usize, 3, 7, 16, 50, 51, 97, 128] {
            let velocities: Vec<[f64; 3]> = (0..n)
                .map(|_| {
                    [
                        noise(&mut state),
                        3.0 * noise(&mut state),
                        noise(&mut state) - 4.0,
                    ]
                })
                .collect();
            for num_bins in [1usize, 5, 40] {
                let spectrum = compute_tke_spectrum(&velocities, num_bins);
                assert_eq!(spectrum.len(), num_bins);
                let total: f64 = spectrum.iter().map(|&(_, e)| e).sum();
                let tke = compute_turbulent_kinetic_energy(&velocities);
                assert!(
                    (total - tke).abs() < 1e-12 * tke.max(1.0),
                    "n = {n}, bins = {num_bins}: {total} vs {tke}"
                );
                assert!(spectrum.iter().all(|&(_, e)| e >= 0.0));
            }
        }
    }

    #[test]
    fn test_tke_spectrum_nyquist_mode_is_not_doubled() {
        // u_i = (-1)^i has all its energy in the Nyquist wavenumber n / 2.
        let velocities: Vec<[f64; 3]> = (0..16)
            .map(|i| [if i % 2 == 0 { 1.0 } else { -1.0 }, 0.0, 0.0])
            .collect();
        let spectrum = compute_tke_spectrum(&velocities, 8);
        let (last_centre, last_energy) = spectrum[7];
        assert!((last_centre - 7.5).abs() < 1e-12);
        assert!(
            (last_energy - 0.5).abs() < 1e-12,
            "Nyquist energy {last_energy}"
        );
        assert!(spectrum[..7].iter().all(|&(_, e)| e < 1e-20));
    }

    #[test]
    fn test_tke_spectrum_degenerate_records() {
        assert!(compute_tke_spectrum(&[], 4).is_empty());
        assert!(compute_tke_spectrum(&[[1.0, 2.0, 3.0]; 4], 0).is_empty());
        let single = compute_tke_spectrum(&[[1.0, 2.0, 3.0]], 3);
        assert_eq!(single.len(), 3);
        assert!(single.iter().all(|&(_, e)| e == 0.0));
        // A constant record has no fluctuations at all.
        let constant = compute_tke_spectrum(&[[1.0, 2.0, 3.0]; 10], 5);
        assert!(constant.iter().all(|&(_, e)| e < 1e-25));
    }

    // ── Level-set curvature ───────────────────────────────────────────────

    /// Sample `f(x, y, z)` at the nodes of a `res`-cubed grid of spacing `h`.
    fn level_set_grid(res: usize, h: f64, f: impl Fn(f64, f64, f64) -> f64) -> Vec<f64> {
        let mut phi = Vec::with_capacity(res * res * res);
        for ix in 0..res {
            for iy in 0..res {
                for iz in 0..res {
                    phi.push(f(ix as f64 * h, iy as f64 * h, iz as f64 * h));
                }
            }
        }
        phi
    }

    /// Relative error of the curvature at every node within `2h` of the zero
    /// level set; returns the largest error and the number of nodes checked.
    fn near_surface_curvature_error(
        phi: &[f64],
        res: usize,
        h: f64,
        expected: impl Fn(usize, usize, usize) -> f64,
    ) -> (f64, usize) {
        let (mut worst, mut checked) = (0.0_f64, 0);
        for ix in 1..res - 1 {
            for iy in 1..res - 1 {
                for iz in 1..res - 1 {
                    if phi[ix * res * res + iy * res + iz].abs() >= 2.0 * h {
                        continue;
                    }
                    let kappa = level_set_mean_curvature(phi, ix, iy, iz, res, h);
                    let want = expected(ix, iy, iz);
                    worst = worst.max((kappa - want).abs() / want);
                    checked += 1;
                }
            }
        }
        (worst, checked)
    }

    #[test]
    fn test_level_set_curvature_of_a_sphere_is_two_over_radius() {
        let (res, radius) = (64usize, 0.3);
        let h = 1.0 / res as f64;
        let phi = level_set_grid(res, h, |x, y, z| {
            ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt() - radius
        });
        let (worst, checked) = near_surface_curvature_error(&phi, res, h, |ix, iy, iz| {
            2.0 / (phi[ix * res * res + iy * res + iz] + radius)
        });
        assert!(checked > 5000, "only {checked} nodes near the surface");
        assert!(worst < 0.05, "worst relative error {worst}");
        // On the surface itself the curvature is 2 / R, not |grad phi| = 1.
        let kappa = level_set_mean_curvature(&phi, 32 + 19, 32, 32, res, h);
        assert!(
            (kappa - 2.0 / (0.30 + 0.001)).abs() < 0.3,
            "kappa = {kappa}"
        );
    }

    #[test]
    fn test_level_set_curvature_of_a_cylinder_is_one_over_radius() {
        let (res, radius) = (64usize, 0.25);
        let h = 1.0 / res as f64;
        let phi = level_set_grid(res, h, |x, y, _| {
            ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt() - radius
        });
        let (worst, checked) = near_surface_curvature_error(&phi, res, h, |ix, iy, iz| {
            1.0 / (phi[ix * res * res + iy * res + iz] + radius)
        });
        assert!(checked > 5000, "only {checked} nodes near the surface");
        assert!(worst < 0.05, "worst relative error {worst}");
    }

    #[test]
    fn test_level_set_curvature_is_independent_of_the_gradient_magnitude() {
        let (res, radius) = (32usize, 0.3);
        let h = 1.0 / res as f64;
        let distance = |x: f64, y: f64, z: f64| {
            ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt() - radius
        };
        let signed = level_set_grid(res, h, distance);
        let scaled = level_set_grid(res, h, |x, y, z| 3.0 * distance(x, y, z));
        for (ix, iy, iz) in [(24, 16, 16), (16, 25, 16), (20, 20, 20)] {
            let a = level_set_mean_curvature(&signed, ix, iy, iz, res, h);
            let b = level_set_mean_curvature(&scaled, ix, iy, iz, res, h);
            assert!(a > 3.0, "curvature {a} should be near 2 / R = 6.7");
            assert!((a - b).abs() < 1e-9 * a, "{a} vs {b}");
        }
    }

    #[test]
    fn test_level_set_curvature_of_a_plane_and_degenerate_inputs_is_zero() {
        let (res, h) = (16usize, 1.0 / 16.0);
        let plane = level_set_grid(res, h, |_, _, z| z - 0.5);
        for (ix, iy, iz) in [(3, 4, 5), (8, 8, 8), (14, 1, 14)] {
            let kappa = level_set_mean_curvature(&plane, ix, iy, iz, res, h);
            assert!(kappa.abs() < 1e-12, "plane curvature {kappa}");
        }
        // Boundary nodes have no central stencil.
        let sphere = level_set_grid(res, h, |x, y, z| {
            ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt() - 0.3
        });
        for (ix, iy, iz) in [
            (0, 8, 8),
            (8, 0, 8),
            (8, 8, 0),
            (15, 8, 8),
            (8, 15, 8),
            (8, 8, 15),
        ] {
            assert_eq!(level_set_mean_curvature(&sphere, ix, iy, iz, res, h), 0.0);
        }
        // Vanishing gradient and short input.
        let flat = vec![1.0; res * res * res];
        assert_eq!(level_set_mean_curvature(&flat, 8, 8, 8, res, h), 0.0);
        assert_eq!(
            level_set_mean_curvature(&sphere[..100], 8, 8, 8, res, h),
            0.0
        );
    }

    // ── Free surface extraction ───────────────────────────────────────────

    /// Right-hand-rule normal `(v1 - v0) x (v2 - v0)` of a triangle (not normalised).
    fn triangle_winding(verts: &[[f64; 3]; 3]) -> [f64; 3] {
        let a = sub3(verts[1], verts[0]);
        let b = sub3(verts[2], verts[0]);
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    #[test]
    fn test_free_surface_plane_is_a_flat_sheet_facing_up() {
        let n = 8;
        let fse = FreeSurfaceExtractor::new(n, 1.0);
        let h = 1.0 / n as f64;
        // 0.5 lies on a node plane, 0.55 between two of them.
        for level in [0.5, 0.55] {
            let phi = level_set_grid(n, h, |_, y, _| y - level);
            let tris = fse.extract(&phi);
            assert!(!tris.is_empty(), "level {level}");
            let mut area = 0.0;
            for t in &tris {
                for v in t.verts {
                    assert!((v[1] - level).abs() < 1e-12, "level {level}: {v:?}");
                }
                assert!(t.normal[0].abs() < 1e-12 && t.normal[2].abs() < 1e-12);
                assert!((t.normal[1] - 1.0).abs() < 1e-12, "normal {:?}", t.normal);
                // The winding agrees with the stored normal.
                let winding = triangle_winding(&t.verts);
                assert!(dot3(winding, t.normal) > 0.0);
                area += 0.5 * len3(winding);
            }
            let exact = (7.0 * h) * (7.0 * h);
            assert!(
                (area - exact).abs() < 1e-12,
                "level {level}: area {area} vs {exact}"
            );
        }
    }

    #[test]
    fn test_free_surface_sphere_has_outward_normals_and_the_right_area() {
        let (n, radius) = (32usize, 0.3);
        let fse = FreeSurfaceExtractor::new(n, 1.0);
        let h = 1.0 / n as f64;
        // The fluid (phi < 0) is the ball of radius 0.3 around node 16.
        let phi = level_set_grid(n, h, |x, y, z| {
            ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt() - radius
        });
        let tris = fse.extract(&phi);
        assert!(!tris.is_empty());
        let mut area = 0.0;
        for t in &tris {
            let centroid = scale3(add3(add3(t.verts[0], t.verts[1]), t.verts[2]), 1.0 / 3.0);
            let radial = norm3(sub3(centroid, [0.5; 3]));
            assert!(
                dot3(t.normal, radial) > 0.5,
                "normal {:?} points into the fluid",
                t.normal
            );
            let winding = triangle_winding(&t.verts);
            assert!(
                dot3(winding, t.normal) > 0.0,
                "winding disagrees with the normal"
            );
            area += 0.5 * len3(winding);
        }
        let exact = 4.0 * std::f64::consts::PI * radius * radius;
        assert!(
            (area - exact).abs() / exact < 0.03,
            "area {area} vs {exact}"
        );
    }

    #[test]
    fn test_free_surface_rejects_short_fields_and_ignores_uniform_ones() {
        let fse = FreeSurfaceExtractor::new(4, 1.0);
        assert!(fse.extract(&[0.0; 10]).is_empty());
        assert!(fse.extract(&[]).is_empty());
        assert!(fse.extract(&[-1.0; 64]).is_empty());
        assert!(FreeSurfaceExtractor::new(0, 1.0).extract(&[]).is_empty());
    }
}
