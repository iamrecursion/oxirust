//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// D2Q9 weights.
pub(super) const W9: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
/// D2Q9 velocity vectors ex.
pub(super) const EX9: [i32; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
/// D2Q9 velocity vectors ey.
pub(super) const EY9: [i32; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
/// Maximum (random close) packing solid volume fraction of monodisperse spheres.
pub(super) const PHI_MAX_PACKING: f64 = 0.64;
/// Compute D2Q9 equilibrium distribution.
pub fn feq_d2q9(rho: f64, ux: f64, uy: f64) -> [f64; 9] {
    let mut feq = [0.0f64; 9];
    let usq = ux * ux + uy * uy;
    for q in 0..9 {
        let eu = EX9[q] as f64 * ux + EY9[q] as f64 * uy;
        feq[q] = W9[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
    }
    feq
}
/// Compute Stokes settling velocity.
///
/// Returns terminal velocity in m/s for a sphere settling under gravity.
pub fn stokes_settling(radius: f64, rho_p: f64, rho_f: f64, mu: f64, g: f64) -> f64 {
    2.0 / 9.0 * (rho_p - rho_f) * g * radius * radius / mu
}
/// Compute Richardson-Zaki exponent from particle Reynolds number.
///
/// Returns exponent n for hindered settling correlation Ut*(1-phi)^n.
pub fn richardson_zaki_exponent(re_t: f64) -> f64 {
    if re_t < 0.2 {
        4.65
    } else if re_t < 1.0 {
        4.35 * re_t.powf(-0.03)
    } else if re_t < 500.0 {
        4.45 * re_t.powf(-0.1)
    } else {
        2.39
    }
}
/// Compute drag coefficient for a sphere.
///
/// Uses Schiller-Naumann correlation.
pub fn drag_coeff_particle(re: f64) -> f64 {
    if re < 1e-10 {
        return 1e6;
    }
    if re < 1000.0 {
        24.0 / re * (1.0 + 0.15 * re.powf(0.687))
    } else {
        0.44
    }
}
/// Compute mixture viscosity using Krieger-Dougherty model.
///
/// Returns relative viscosity mu_mix/mu_0.
pub fn mixture_viscosity(phi: f64, phi_max: f64) -> f64 {
    let phi = phi.clamp(0.0, phi_max * 0.999);
    (1.0 - phi / phi_max).powf(-2.5 * phi_max)
}
/// Godunov numerical flux for the Kynch batch-flux conservation law.
///
/// The solids move from the cell with fraction `phi_from` into the cell with
/// fraction `phi_to` with the flux function `q(phi) = phi |v_s(phi)|`.  It is
/// zero at `phi = 0`, rises to a single maximum `q_peak` at the fraction
/// `phi_peak = 1 / (n + 1)` and falls again.  `q_from = q(phi_from)` and
/// `q_to = q(phi_to)` are the flux function evaluated in the two cells.  The
/// Godunov flux of the exact Riemann problem between the two states is
///
/// ```text
/// phi_from >  phi_to :  q_peak                  if phi_to <= phi_peak <= phi_from
///                       max(q_from, q_to)       otherwise
/// phi_from <= phi_to :  min(q_from, q_to)
/// ```
///
/// For dilute states (`phi <= phi_peak`) this is the donor-cell flux `q_from`.
/// For dense states (`phi > phi_peak`) the characteristics of the batch flux
/// point back towards the source and the flux is taken from the receiving cell
/// (`q_to`); a plain donor-cell flux is a downwind difference there and
/// amplifies perturbations.
pub(super) fn godunov_batch_flux(
    phi_from: f64,
    phi_to: f64,
    q_from: f64,
    q_to: f64,
    phi_peak: f64,
    q_peak: f64,
) -> f64 {
    if phi_from > phi_to {
        if phi_to <= phi_peak && phi_peak <= phi_from {
            q_peak
        } else {
            q_from.max(q_to)
        }
    } else {
        q_from.min(q_to)
    }
}
/// Downward face fluxes of one settling column.
///
/// `phi` and `vs` hold the solid fraction and the signed settling velocity
/// (m/s, positive towards cell 0) of the cells of one column, bottom first;
/// `flux` must have `phi.len() + 1` entries.  On return `flux[j]` is the volume
/// flux of solids (m/s) through the lower face of cell `j`, positive when
/// solids move from cell `j` to cell `j - 1`; the bottom wall `flux[0]` and
/// the top wall `flux[ny]` carry nothing.
///
/// Interior faces use the Godunov flux ([`godunov_batch_flux`]) of the batch
/// flux `q(phi) = phi |v_s|` with its maximum `q_peak` at `phi_peak`:
/// sedimenting solids (`settling`) come from the upper cell of the face,
/// creaming solids from the lower one.  The flux is then limited so that the
/// source cell cannot lose more than it holds in one step (`ratio = dt / dz`)
/// and the receiving cell cannot exceed the maximum packing
/// [`PHI_MAX_PACKING`].
pub(super) fn column_face_fluxes(
    phi: &[f64],
    vs: &[f64],
    ratio: f64,
    peak: (f64, f64),
    settling: bool,
    flux: &mut [f64],
) {
    let (phi_peak, q_peak) = peak;
    flux.fill(0.0);
    let cells = phi.len().min(vs.len());
    for (j, face) in flux.iter_mut().enumerate().take(cells).skip(1) {
        let (lower, upper) = (j - 1, j);
        *face = if settling {
            let q_from = phi[upper] * vs[upper];
            let q_to = phi[lower] * vs[lower];
            let wanted = godunov_batch_flux(phi[upper], phi[lower], q_from, q_to, phi_peak, q_peak);
            let room = (PHI_MAX_PACKING - phi[lower]).max(0.0);
            wanted.min(phi[upper] / ratio).min(room / ratio).max(0.0)
        } else {
            let q_from = -phi[lower] * vs[lower];
            let q_to = -phi[upper] * vs[upper];
            let wanted = godunov_batch_flux(phi[lower], phi[upper], q_from, q_to, phi_peak, q_peak);
            let room = (PHI_MAX_PACKING - phi[upper]).max(0.0);
            -wanted.min(phi[lower] / ratio).min(room / ratio).max(0.0)
        };
    }
}
/// 3D vector magnitude.
pub(super) fn mag3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sedimentation::types::*;
    #[test]
    fn test_sediment_particle_mass() {
        let p = SedimentParticle::new([0.0, 0.0, 0.0], 1e-3, 2500.0);
        let vol = 4.0 / 3.0 * std::f64::consts::PI * 1e-9;
        assert!((p.mass() - 2500.0 * vol).abs() < 1e-20);
    }
    #[test]
    fn test_sediment_particle_volume() {
        let p = SedimentParticle::new([0.0, 0.0, 0.0], 1e-3, 2500.0);
        let vol = 4.0 / 3.0 * std::f64::consts::PI * 1e-9;
        assert!((p.volume() - vol).abs() < 1e-25);
    }
    #[test]
    fn test_hindrance_factor_zero_phi() {
        let h = HindranceFactor::new(1e-3, 4.65);
        assert!((h.hindered_velocity(0.0) - 1e-3).abs() < 1e-15);
    }
    #[test]
    fn test_hindrance_factor_high_phi() {
        let h = HindranceFactor::new(1e-3, 4.65);
        let v = h.hindered_velocity(0.9);
        assert!(v < h.hindered_velocity(0.0));
        assert!(v > 0.0);
    }
    #[test]
    fn test_hindrance_max_flux() {
        let h = HindranceFactor::new(1.0, 4.65);
        let phi_max = h.max_flux_phi();
        assert!((phi_max - 1.0 / 5.65).abs() < 1e-10);
    }
    #[test]
    fn test_batch_flux_zero_at_extremes() {
        let h = HindranceFactor::new(1.0, 4.65);
        assert!(h.batch_flux(0.0).abs() < 1e-15);
        assert!(h.batch_flux(1.0).abs() < 1e-15);
    }
    #[test]
    fn test_stokes_settling_positive() {
        let v = stokes_settling(1e-4, 2500.0, 1000.0, 1e-3, 9.81);
        assert!(v > 0.0);
    }
    #[test]
    fn test_stokes_settling_scales_r2() {
        let v1 = stokes_settling(1e-4, 2500.0, 1000.0, 1e-3, 9.81);
        let v2 = stokes_settling(2e-4, 2500.0, 1000.0, 1e-3, 9.81);
        assert!((v2 / v1 - 4.0).abs() < 1e-10);
    }
    #[test]
    fn test_richardson_zaki_low_re() {
        let n = richardson_zaki_exponent(0.1);
        assert!((n - 4.65).abs() < 1e-10);
    }
    #[test]
    fn test_richardson_zaki_high_re() {
        let n = richardson_zaki_exponent(1000.0);
        assert!((n - 2.39).abs() < 1e-10);
    }
    #[test]
    fn test_drag_coeff_stokes() {
        let cd = drag_coeff_particle(1e-4);
        let stokes = 24.0 / 1e-4;
        assert!((cd / stokes - 1.0).abs() < 0.01, "cd={cd}, stokes={stokes}");
    }
    #[test]
    fn test_drag_coeff_newton() {
        let cd = drag_coeff_particle(5000.0);
        assert!((cd - 0.44).abs() < 1e-10);
    }
    #[test]
    fn test_mixture_viscosity_zero_phi() {
        let mu = mixture_viscosity(0.0, 0.64);
        assert!((mu - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_mixture_viscosity_increases_with_phi() {
        let mu1 = mixture_viscosity(0.1, 0.64);
        let mu2 = mixture_viscosity(0.3, 0.64);
        assert!(mu2 > mu1);
    }
    #[test]
    fn test_sedi_lbm_creation() {
        let sim = SediLbm::new(8, 8, 1.7, 1.0, 2.5, 1.0e-4);
        assert_eq!(sim.nx, 8);
        assert_eq!(sim.ny, 8);
        assert_eq!(sim.body_force.len(), 8);
        assert!(sim.body_force.iter().flatten().all(|f| *f == [0.0, 0.0]));
    }
    /// Lattice-unit mixture buoyancy: one step injects exactly the body force
    /// density `(rho_s - rho_f) * phi * g` of downward momentum per cell and
    /// leaves the density untouched.
    #[test]
    fn test_sedi_lbm_step() {
        let g_latt = 1.0e-4;
        let mut sim = SediLbm::new(8, 8, 1.7, 1.0, 2.5, g_latt);
        sim.set_uniform_phi(0.1);
        sim.step();
        let expected_vy = -(2.5 - 1.0) * 0.1 * g_latt;
        for y in 0..sim.ny {
            for x in 0..sim.nx {
                assert!((sim.rho[y][x] - 1.0).abs() < 1e-12);
                assert!((sim.u[y][x][1] - expected_vy).abs() < 1e-12);
                assert!(sim.u[y][x][0].abs() < 1e-12);
            }
        }
    }
    #[test]
    fn test_forced_convection_terminal_velocity() {
        let fcs = ForcedConvectionSedi::new(1000.0, 1e-3, [0.0, -9.81, 0.0], [1.0, 1.0, 1.0], 1e-4);
        let p = SedimentParticle::new([0.5, 0.5, 0.5], 1e-4, 2500.0);
        let vt = fcs.terminal_velocity(&p);
        assert!(vt > 0.0);
    }
    #[test]
    fn test_forced_convection_drag_opposes_velocity() {
        let fcs = ForcedConvectionSedi::new(1000.0, 1e-3, [0.0, -9.81, 0.0], [1.0, 1.0, 1.0], 1e-4);
        let mut p = SedimentParticle::new([0.5, 0.5, 0.5], 1e-4, 2500.0);
        p.velocity = [1.0, 0.0, 0.0];
        let fd = fcs.drag_force(&p);
        assert!(fd[0] < 0.0);
    }
    #[test]
    fn test_bed_formation_deposit() {
        let mut bed = BedFormation::new(10, 1e-3, 1000.0, 2500.0);
        let h_before = bed.bed_height[5];
        bed.deposit(5, 1e-6);
        assert!(bed.bed_height[5] > h_before);
    }
    #[test]
    fn test_bed_formation_avg_height() {
        let bed = BedFormation::new(10, 1e-3, 1000.0, 2500.0);
        assert!(bed.avg_height() >= 0.0);
    }
    #[test]
    fn test_fluidization_umf_positive() {
        let umf = FluidizationLbm::calc_umf(1e-3, 2500.0, 1000.0, 1e-3, 0.4);
        assert!(umf > 0.0);
    }
    #[test]
    fn test_fluidization_regime_packed() {
        let sim = FluidizationLbm::new(8, 16, 1e-3, 2500.0, 1000.0, 1e-3, 0.0);
        assert_eq!(sim.regime(), FluidizationRegime::PackedBed);
    }
    #[test]
    fn test_sedimentation3d_creation() {
        let sim = Sedimentation3D::new([4, 8, 4], 1e-4, 2500.0, 1000.0, 1e-3, 1e-3, 1e-3);
        assert_eq!(sim.dims, [4, 8, 4]);
        assert!((sim.dz - 1e-3).abs() < 1e-15);
    }
    #[test]
    fn test_sedimentation3d_step() {
        let mut sim = Sedimentation3D::new([4, 4, 4], 1e-4, 2500.0, 1000.0, 1e-3, 1e-3, 1e-3);
        sim.step();
        for &phi in &sim.phi {
            assert!((0.0..=0.64 + 1e-12).contains(&phi), "phi = {phi}");
        }
    }
    #[test]
    fn test_darcy_lbm_creation() {
        let sim = DarcyFlowLbm::new(8, 8, 1.7, 0.4, 1e-10, 0.01);
        assert_eq!(sim.nx, 8);
    }
    #[test]
    fn test_darcy_lbm_step() {
        let mut sim = DarcyFlowLbm::new(8, 8, 1.7, 0.4, 1e-10, 0.01);
        sim.step();
        for y in 0..sim.ny {
            for x in 0..sim.nx {
                assert!(sim.rho[y][x].is_finite());
            }
        }
    }
    #[test]
    fn test_sedi_statistics_settling() {
        let mut stats = SediStatistics::new(1000.0, 1e-3, 9.81, 0.1, 0.01);
        stats.add_particle(1e-4, 2500.0);
        stats.add_particle(2e-4, 2500.0);
        let vs = stats.settling_velocities();
        assert_eq!(vs.len(), 2);
        assert!(vs[1] > vs[0]);
    }
    #[test]
    fn test_sedi_statistics_std() {
        let mut stats = SediStatistics::new(1000.0, 1e-3, 9.81, 0.1, 0.01);
        stats.add_particle(1e-4, 2500.0);
        stats.add_particle(2e-4, 2500.0);
        let std = stats.std_settling_velocity();
        assert!(std > 0.0);
    }
    #[test]
    fn test_turbulent_suspension_creation() {
        let ts = TurbulentSuspension::new(20, 1.0, 0.05, 1.5e-5, 0.1);
        assert_eq!(ts.ny, 20);
    }
    #[test]
    fn test_turbulent_suspension_step() {
        let mut ts = TurbulentSuspension::new(20, 1.0, 0.05, 1.5e-5, 0.1);
        ts.step();
        for &c in &ts.concentration {
            assert!(c >= 0.0);
        }
    }
    #[test]
    fn test_bed_shields_parameter() {
        let bed = BedFormation::new(10, 1e-3, 1000.0, 2500.0);
        let s = bed.shields_parameter(5, 0.1, 1e-3);
        assert!(s >= 0.0);
    }
    #[test]
    fn test_mixture_density() {
        let sim = SediLbm::new(4, 4, 1.7, 1.0, 2.5, 1.0e-4);
        let rho_mix = sim.mixture_density(0.5);
        assert!((rho_mix - 1.75).abs() < 1e-12);
    }
    #[test]
    fn test_floc_diameter_increases_with_n() {
        let f1 = Floc::new(1.0, 1e-6, 2.0, 2500.0);
        let f2 = Floc::new(8.0, 1e-6, 2.0, 2500.0);
        assert!(f2.diameter() > f1.diameter());
    }
    #[test]
    fn test_floc_effective_density_between_fluid_solid() {
        let f = Floc::new(100.0, 1e-6, 2.0, 2500.0);
        let rho_eff = f.effective_density(1000.0, 2500.0);
        assert!(rho_eff > 1000.0);
        assert!(rho_eff < 2500.0);
    }
    #[test]
    fn test_floc_settling_velocity_positive() {
        let f = Floc::new(10.0, 1e-6, 2.0, 2500.0);
        let vs = f.settling_velocity(1000.0, 2500.0, 1e-3, 9.81);
        assert!(vs > 0.0);
    }
    #[test]
    fn test_floc_dynamics_kolmogorov_scale() {
        let fd = FlocDynamics::new(1e-3, 1.0, 0.01, 1e-4);
        let eta = fd.kolmogorov_scale(1000.0);
        assert!(eta > 0.0);
        assert!(eta < 1.0);
    }
    #[test]
    fn test_floc_dynamics_aggregation_kernel_positive() {
        let fd = FlocDynamics::new(1e-3, 1.0, 0.01, 1e-4);
        let beta = fd.aggregation_kernel(1e-6, 1e-6, 1000.0);
        assert!(beta > 0.0);
    }
    #[test]
    fn test_floc_dynamics_breakup_increases_with_turbulence() {
        let fd_low = FlocDynamics::new(1e-3, 1.0, 0.01, 1e-5);
        let fd_high = FlocDynamics::new(1e-3, 1.0, 0.01, 1e-2);
        let b_low = fd_low.turbulent_breakup_rate(1e-4, 1000.0);
        let b_high = fd_high.turbulent_breakup_rate(1e-4, 1000.0);
        assert!(b_high > b_low);
    }
    #[test]
    fn test_free_settling_sphere_correction_unity() {
        let fsc = FreeSettlingCorrection::sphere();
        let factor = fsc.velocity_correction_factor(1.0);
        assert!(factor > 0.0);
    }
    #[test]
    fn test_free_settling_non_spherical_slower() {
        let sphere = FreeSettlingCorrection::sphere();
        let flat = FreeSettlingCorrection::new(0.7, 0.7, 1.0);
        let v_sphere = sphere.corrected_velocity(1e-3, 1.0);
        let v_flat = flat.corrected_velocity(1e-3, 1.0);
        assert!(v_sphere > v_flat);
    }
    #[test]
    fn test_dem_particle_mass_positive() {
        let p = DemParticle::new([0.0; 3], 1e-3, 2500.0, 0);
        assert!(p.mass() > 0.0);
    }
    #[test]
    fn test_dem_particle_overlap_zero_far() {
        let p1 = DemParticle::new([0.0, 0.0, 0.0], 1e-3, 2500.0, 0);
        let p2 = DemParticle::new([0.1, 0.0, 0.0], 1e-3, 2500.0, 1);
        let ov = p1.overlap(&p2);
        assert!(ov < 1e-10);
    }
    #[test]
    fn test_dem_particle_overlap_positive_when_close() {
        let p1 = DemParticle::new([0.0, 0.0, 0.0], 1e-3, 2500.0, 0);
        let p2 = DemParticle::new([1e-3, 0.0, 0.0], 1e-3, 2500.0, 1);
        let ov = p1.overlap(&p2);
        assert!(ov > 0.0);
    }
    #[test]
    fn test_lbm_dem_solver_creation() {
        let solver = LbmDemSolver::new(8, 8, 1.7, 1000.0, 2500.0, 1e-3, 1e-5);
        assert_eq!(solver.lbm.nx, 8);
        // The embedded LBM runs in lattice units, not with SI densities.
        assert!((solver.lbm.rho_fluid - 1.0).abs() < 1e-15);
        assert!((solver.lbm.rho_solid - 2.5).abs() < 1e-15);
        assert!(
            solver
                .lbm
                .rho
                .iter()
                .flatten()
                .all(|r| (r - 1.0).abs() < 1e-15)
        );
        // Default gravity and its lattice-unit value g dt^2 / dx.
        assert_eq!(solver.gravity, [0.0, -9.81, 0.0]);
        assert!((solver.gravity_latt[1] - (-9.81 * 1e-10 / 1e-3)).abs() < 1e-18);
        assert_eq!(solver.gravity_latt[0], 0.0);
    }
    #[test]
    fn test_volume_fraction_field_step_conserves_mass() {
        let mut vf = VolumeFractionField::new(20, 0.01, 1e-3, 4.65, 1e-8);
        vf.set_uniform(0.1);
        let mass_before = vf.total_solid_volume();
        vf.step(1e-4);
        let mass_after = vf.total_solid_volume();
        assert!((mass_after - mass_before).abs() < mass_before * 0.1);
    }
    #[test]
    fn test_volume_fraction_front_position() {
        let mut vf = VolumeFractionField::new(20, 0.01, 1e-3, 4.65, 1e-8);
        vf.set_uniform(0.05);
        let fp = vf.front_position(0.01);
        assert!(fp > 0.0);
    }
    #[test]
    fn test_turbulent_dispersion_stokes_number() {
        let td = TurbulentDispersion::new(0.01, 1e-4, 1.5e-5, 0.001);
        let st = td.stokes_number_kol();
        assert!(st > 0.0);
    }
    #[test]
    fn test_turbulent_dispersion_diffusivity_decreases_with_st() {
        let td_low = TurbulentDispersion::new(0.01, 1e-4, 1.5e-5, 1e-4);
        let td_high = TurbulentDispersion::new(0.01, 1e-4, 1.5e-5, 1.0);
        assert!(td_low.turbulent_diffusivity() > td_high.turbulent_diffusivity());
    }
    #[test]
    fn test_density_current_front_speed_positive() {
        let dc = DensityCurrent::new(1000.0, 1100.0, 0.5, 9.81);
        assert!(dc.front_speed() > 0.0);
    }
    #[test]
    fn test_density_current_step_advances() {
        let mut dc = DensityCurrent::new(1000.0, 1100.0, 0.5, 9.81);
        dc.step(1.0);
        assert!(dc.length > 0.0);
        assert!(dc.time > 0.0);
    }
    #[test]
    fn test_krieger_dougherty_zero_phi() {
        let kd = KriegerDougherty::spheres(1e-3);
        let mu = kd.mixture_viscosity(0.0);
        assert!((mu - 1e-3).abs() < 1e-15);
    }
    #[test]
    fn test_krieger_dougherty_increases_with_phi() {
        let kd = KriegerDougherty::spheres(1e-3);
        let mu1 = kd.mixture_viscosity(0.1);
        let mu2 = kd.mixture_viscosity(0.3);
        assert!(mu2 > mu1);
    }
    #[test]
    fn test_krieger_dougherty_einstein_approx() {
        let kd = KriegerDougherty::spheres(1e-3);
        let mu_kd = kd.mixture_viscosity(0.01);
        let mu_ein = kd.einstein_viscosity(0.01);
        assert!((mu_kd - mu_ein).abs() / mu_ein < 0.1);
    }
    #[test]
    fn test_sedimentation_front_initial_state() {
        let sf = SedimentationFront::new(1.0, 0.1, 4.65, 1e-3);
        assert!((sf.upper_front - 1.0).abs() < 1e-10);
        assert!((sf.lower_front).abs() < 1e-10);
    }
    #[test]
    fn test_sedimentation_front_step_reduces_upper() {
        let mut sf = SedimentationFront::new(1.0, 0.1, 4.65, 1e-3);
        sf.step(1.0);
        assert!(sf.upper_front < 1.0);
    }
    #[test]
    fn test_bed_compaction_porosity() {
        let bc = BedCompaction::new(0.8, 0.3, 1e-12, 2500.0, 1000.0, 0.1, 9.81);
        let n = bc.porosity();
        assert!(n > 0.0 && n < 1.0);
        assert!((n - 0.8 / 1.8).abs() < 1e-10);
    }
    #[test]
    fn test_bed_compaction_consolidate_reduces_void_ratio() {
        let mut bc = BedCompaction::new(0.8, 0.3, 1e-12, 2500.0, 1000.0, 0.1, 9.81);
        let e_before = bc.void_ratio;
        bc.consolidate(100.0);
        assert!(bc.void_ratio < e_before);
    }
}
