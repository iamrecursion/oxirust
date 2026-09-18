//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Compute the trace of a 3×3 matrix.
pub(super) fn trace3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] + m[1][1] + m[2][2]
}
/// Compute the determinant of a 3×3 matrix.
pub(super) fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
/// Multiply two 3×3 matrices.
pub(super) fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}
/// Transpose a 3×3 matrix.
pub(super) fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = m[j][i];
        }
    }
    t
}
/// Compute the Right Cauchy-Green deformation tensor C = F^T F.
pub(super) fn right_cauchy_green(f: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let ft = transpose3(f);
    matmul3(&ft, f)
}
/// Identity matrix 3×3.
pub(super) fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
/// Scalar multiply 3×3 matrix.
pub(super) fn scale3(s: f64, m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = s * m[i][j];
        }
    }
    out
}
/// Add two 3×3 matrices.
pub(super) fn add3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}
/// Second invariant I2 of C = 0.5*(tr(C)^2 - tr(C^2)).
pub(super) fn second_invariant_c(c: &[[f64; 3]; 3]) -> f64 {
    let c2 = matmul3(c, c);
    let tr_c = trace3(c);
    let tr_c2 = trace3(&c2);
    0.5 * (tr_c * tr_c - tr_c2)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::biomechanics::BloodVesselModel;
    use crate::biomechanics::BoneModel;
    use crate::biomechanics::CardiacMuscleModel;
    use crate::biomechanics::CartilageModel;
    use crate::biomechanics::CorneaModel;
    use crate::biomechanics::CorticalBone;
    use crate::biomechanics::FailureMode;
    use crate::biomechanics::FiberReinforcedTissue;
    use crate::biomechanics::HeartValveTissue;
    use crate::biomechanics::HuxleyModel;
    use crate::biomechanics::IntervertebralDiscModel;
    use crate::biomechanics::LigamentWrapModel;
    use crate::biomechanics::MeniscusModel;
    use crate::biomechanics::MuscleModel;
    use crate::biomechanics::SkinModel;
    use crate::biomechanics::SoftTissueMaterial;
    use crate::biomechanics::TendonLigamentModel;
    use crate::biomechanics::TissueFailureCriteria;
    use crate::biomechanics::TrabecularBone;
    #[test]
    fn test_soft_tissue_strain_energy_zero() {
        let mat = SoftTissueMaterial::liver();
        let w = mat.strain_energy_density(3.0, 3.0, 1.0);
        assert!(w.abs() < 1e-10, "W should be 0 at reference: {w}");
    }
    #[test]
    fn test_soft_tissue_strain_energy_positive() {
        let mat = SoftTissueMaterial::liver();
        let w = mat.strain_energy_density(3.5, 3.2, 1.05);
        assert!(w > 0.0, "Strain energy must be positive under deformation");
    }
    #[test]
    fn test_soft_tissue_brain_params() {
        let mat = SoftTissueMaterial::brain();
        assert!(mat.c1 > 0.0);
        assert!(mat.c2 > 0.0);
    }
    #[test]
    fn test_soft_tissue_q_exponent() {
        let mat = SoftTissueMaterial::new(1.0, 1.0, 0.5, 100.0);
        let q = mat.q_exponent(3.5, 3.2, 1.0);
        let expected = 1.0 * 0.5 + 0.5 * 0.2;
        assert!(
            (q - expected).abs() < 1e-10,
            "Q expected {expected}, got {q}"
        );
    }
    #[test]
    fn test_soft_tissue_cauchy_stress_identity() {
        // Regression: F = I + E and raw invariants left 2 c1 (c2 + 2 α) =
        // 539 Pa on every diagonal entry of the reference configuration.
        let zero_strain = [[0.0f64; 3]; 3];
        for mat in [
            SoftTissueMaterial::liver(),
            SoftTissueMaterial::brain(),
            SoftTissueMaterial::new(12.0, 3.0, 0.7, 40.0),
        ] {
            let stress = mat.cauchy_stress_green_lagrange(zero_strain);
            let mag: f64 = stress
                .iter()
                .flat_map(|r| r.iter())
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(mag < 1e-9, "Stress at zero strain must vanish: {mag}");
        }
    }
    /// Strain energy of the diagonal Green–Lagrange strain `e`
    /// (`C = diag(1 + 2 e_i)`), through the public invariant interface.
    fn soft_tissue_energy(mat: &SoftTissueMaterial, e: [f64; 3]) -> f64 {
        let c = [1.0 + 2.0 * e[0], 1.0 + 2.0 * e[1], 1.0 + 2.0 * e[2]];
        let i1 = c[0] + c[1] + c[2];
        let i2 = c[0] * c[1] + c[1] * c[2] + c[2] * c[0];
        let j = (c[0] * c[1] * c[2]).sqrt();
        mat.strain_energy_density(i1, i2, j)
    }
    fn diagonal_strain(e: [f64; 3]) -> [[f64; 3]; 3] {
        [[e[0], 0.0, 0.0], [0.0, e[1], 0.0], [0.0, 0.0, e[2]]]
    }
    #[test]
    fn test_soft_tissue_energy_is_isochoric_plus_volumetric() {
        let mat = SoftTissueMaterial::liver();
        // Reference state.
        assert!(soft_tissue_energy(&mat, [0.0; 3]).abs() < 1e-12);
        // Hydrostatic strain: Q̄ = 0, so only the volumetric term remains.
        let e = 0.01;
        let w = soft_tissue_energy(&mat, [e; 3]);
        let j = (1.0_f64 + 2.0 * e).powf(1.5);
        assert!(
            (w - mat.c1 * mat.beta * (j - 1.0).powi(2)).abs() < 1e-9 * w,
            "W = {w}"
        );
        // Isochoric uniaxial stretch (J = 1): the volumetric term vanishes.
        let lambda = 1.3_f64;
        let w_iso = soft_tissue_energy(
            &mat,
            [
                0.5 * (lambda * lambda - 1.0),
                0.5 * (1.0 / lambda - 1.0),
                0.5 * (1.0 / lambda - 1.0),
            ],
        );
        let i1 = lambda * lambda + 2.0 / lambda;
        let i2 = 2.0 * lambda + 1.0 / (lambda * lambda);
        let q = mat.c2 * (i1 - 3.0) + mat.alpha * (i2 - 3.0);
        assert!((w_iso - mat.c1 * (q.exp() - 1.0)).abs() < 1e-9 * w_iso);
    }
    #[test]
    fn test_soft_tissue_hydrostatic_strain_gives_only_a_pressure() {
        let mat = SoftTissueMaterial::liver();
        let e = 0.01;
        let stress = mat.cauchy_stress_green_lagrange(diagonal_strain([e; 3]));
        let j = (1.0_f64 + 2.0 * e).powf(1.5);
        let pressure = 2.0 * mat.c1 * mat.beta * (j - 1.0);
        assert!(pressure > 2.0e4, "pressure {pressure}");
        for (i, row) in stress.iter().enumerate() {
            for (k, &s) in row.iter().enumerate() {
                let expected = if i == k { pressure } else { 0.0 };
                assert!((s - expected).abs() < 1e-9 * pressure, "σ[{i}][{k}] = {s}");
            }
        }
    }
    #[test]
    fn test_soft_tissue_uniaxial_isochoric_stretch_matches_closed_form() {
        // J = 1: σ_xx − σ_yy = 2 (λ² − 1/λ) (Ŵ1 + Ŵ2/λ) and σ_yy = σ_zz.
        let mat = SoftTissueMaterial::liver();
        for &lambda in &[0.8_f64, 1.1, 1.3, 1.6] {
            let strain = diagonal_strain([
                0.5 * (lambda * lambda - 1.0),
                0.5 * (1.0 / lambda - 1.0),
                0.5 * (1.0 / lambda - 1.0),
            ]);
            let stress = mat.cauchy_stress_green_lagrange(strain);
            let i1 = lambda * lambda + 2.0 / lambda;
            let i2 = 2.0 * lambda + 1.0 / (lambda * lambda);
            let exp_q = (mat.c2 * (i1 - 3.0) + mat.alpha * (i2 - 3.0)).exp();
            let w1 = mat.c1 * mat.c2 * exp_q;
            let w2 = mat.c1 * mat.alpha * exp_q;
            let expected = 2.0 * (lambda * lambda - 1.0 / lambda) * (w1 + w2 / lambda);
            let difference = stress[0][0] - stress[1][1];
            assert!(
                (difference - expected).abs() < 1e-9 * expected.abs(),
                "λ = {lambda}: {difference} vs {expected}"
            );
            assert!((stress[1][1] - stress[2][2]).abs() < 1e-9 * expected.abs());
            // The deviatoric stress is traceless (J = 1: no pressure).
            let trace = stress[0][0] + stress[1][1] + stress[2][2];
            assert!(trace.abs() < 1e-9 * expected.abs(), "tr σ = {trace}");
        }
    }
    #[test]
    fn test_soft_tissue_stress_is_the_derivative_of_the_strain_energy() {
        // For F = U (diagonal strain) the Cauchy stress is σ = F S Fᵀ / J with
        // the second Piola-Kirchhoff stress S = ∂W/∂E: S_ii = J σ_ii / (1 + 2 e_i).
        let mat = SoftTissueMaterial::liver();
        let states = [
            [0.05, -0.02, 0.01],
            [0.2, 0.1, -0.1],
            [-0.1, -0.1, -0.1],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, -0.25],
        ];
        for e in states {
            let stress = mat.cauchy_stress_green_lagrange(diagonal_strain(e));
            let j = ((1.0 + 2.0 * e[0]) * (1.0 + 2.0 * e[1]) * (1.0 + 2.0 * e[2])).sqrt();
            for i in 0..3 {
                let h = 1.0e-5;
                let (mut plus, mut minus) = (e, e);
                plus[i] += h;
                minus[i] -= h;
                let numeric =
                    (soft_tissue_energy(&mat, plus) - soft_tissue_energy(&mat, minus)) / (2.0 * h);
                let analytic = j * stress[i][i] / (1.0 + 2.0 * e[i]);
                assert!(
                    (numeric - analytic).abs() < 1e-6 * analytic.abs().max(1.0),
                    "E = {e:?}, component {i}: dW/dE = {numeric}, J σ / λ² = {analytic}"
                );
                for (k, &entry) in stress[i].iter().enumerate() {
                    if k != i {
                        assert_eq!(entry, 0.0, "diagonal strain gives a diagonal stress");
                    }
                }
            }
        }
    }
    #[test]
    fn test_soft_tissue_stress_is_objective_under_rotation_of_the_strain() {
        // σ(Q E Qᵀ) = Q σ(E) Qᵀ for the isotropic material; this exercises
        // the full (non-diagonal) tensor algebra.
        let mat = SoftTissueMaterial::brain();
        let e = [
            [0.12, 0.03, -0.02],
            [0.03, -0.04, 0.05],
            [-0.02, 0.05, 0.08],
        ];
        let (sa, ca) = 0.52_f64.sin_cos();
        let (sb, cb) = (-0.9_f64).sin_cos();
        let rz = [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]];
        let rx = [[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]];
        let q = matmul3(&rz, &rx);
        let rotate = |m: &[[f64; 3]; 3]| matmul3(&matmul3(&q, m), &transpose3(&q));
        let stress = mat.cauchy_stress_green_lagrange(e);
        let rotated_stress = mat.cauchy_stress_green_lagrange(rotate(&e));
        let expected = rotate(&stress);
        let scale = stress.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(scale > 10.0, "scale {scale}");
        for i in 0..3 {
            for k in 0..3 {
                assert!(
                    (rotated_stress[i][k] - expected[i][k]).abs() < 1e-9 * scale,
                    "[{i}][{k}]: {} vs {}",
                    rotated_stress[i][k],
                    expected[i][k]
                );
                // The stress of a symmetric strain is symmetric.
                assert!((stress[i][k] - stress[k][i]).abs() < 1e-9 * scale);
            }
        }
    }
    #[test]
    fn test_soft_tissue_shear_strain_gives_shear_stress_of_equal_sign() {
        let mat = SoftTissueMaterial::liver();
        let strain = [[0.0, 0.05, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let stress = mat.cauchy_stress_green_lagrange(strain);
        assert!(stress[0][1] > 0.0 && stress[0][1] == stress[1][0]);
        let negative = mat.cauchy_stress_green_lagrange([
            [0.0, -0.05, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);
        assert!((negative[0][1] + stress[0][1]).abs() < 1e-9 * stress[0][1]);
    }
    #[test]
    fn test_soft_tissue_inadmissible_strain_is_reported() {
        let mat = SoftTissueMaterial::liver();
        // E_xx <= −1/2 collapses the x direction (C_xx <= 0).
        let collapsed = diagonal_strain([-0.6, 0.0, 0.0]);
        assert!(
            mat.cauchy_stress_green_lagrange(collapsed)
                .iter()
                .flatten()
                .all(|s| s.is_nan())
        );
        assert!(matches!(
            mat.try_cauchy_stress_green_lagrange(collapsed),
            Err(crate::Error::InadmissibleState(_))
        ));
        // Indefinite C = diag(−0.6, −0.6, 2) has a positive determinant, so
        // the determinant alone does not identify an admissible strain.
        let indefinite = [[-0.8, 0.0, 0.0], [0.0, -0.8, 0.0], [0.0, 0.0, 0.5]];
        assert!(mat.try_cauchy_stress_green_lagrange(indefinite).is_err());
        // Non-finite strain.
        let mut bad = diagonal_strain([0.1, 0.0, 0.0]);
        bad[1][1] = f64::NAN;
        assert!(mat.try_cauchy_stress_green_lagrange(bad).is_err());
        // An admissible strain gives the same result through both entry points.
        let ok = diagonal_strain([0.05, 0.0, 0.0]);
        assert_eq!(
            mat.try_cauchy_stress_green_lagrange(ok).ok(),
            Some(mat.cauchy_stress_green_lagrange(ok))
        );
        // Only the symmetric part of the strain matters.
        let skew = [[0.0, 0.02, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let sym = [[0.0, 0.01, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]];
        assert_eq!(
            mat.cauchy_stress_green_lagrange(skew),
            mat.cauchy_stress_green_lagrange(sym)
        );
    }
    #[test]
    fn test_soft_tissue_cauchy_stress_positive_stretch() {
        let mat = SoftTissueMaterial::liver();
        let strain = [[0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let stress = mat.cauchy_stress_green_lagrange(strain);
        assert!(
            stress[0][0] > 0.0,
            "σ_xx should be positive under tensile strain"
        );
    }
    #[test]
    fn test_fiber_strain_energy_no_tension() {
        let tissue = FiberReinforcedTissue::aortic_tissue();
        let w = tissue.fiber_strain_energy(0.9);
        assert_eq!(w, 0.0, "No fiber energy in compression");
    }
    #[test]
    fn test_fiber_strain_energy_tension() {
        let tissue = FiberReinforcedTissue::aortic_tissue();
        let w = tissue.fiber_strain_energy(1.1);
        assert!(w > 0.0, "Fiber energy should be positive in tension");
    }
    #[test]
    fn test_fiber_total_strain_energy() {
        let tissue = FiberReinforcedTissue::aortic_tissue();
        let w = tissue.total_strain_energy(3.5, 1.1);
        assert!(w > 0.0);
    }
    #[test]
    fn test_fiber_strain_energy_at_ref() {
        let tissue = FiberReinforcedTissue::aortic_tissue();
        let w = tissue.fiber_strain_energy(1.0);
        assert!(w.abs() < 1e-10, "At I4=1 (undeformed), fiber energy=0");
    }
    #[test]
    fn test_fiber_compute_i4_identity() {
        let tissue = FiberReinforcedTissue::new(1.0, [0.0, 1.0, 0.0], 0.1, 1.0, 1.0);
        let f_id = identity3();
        let i4 = tissue.compute_i4(&f_id);
        assert!(
            (i4 - 1.0).abs() < 1e-10,
            "I4 = 1 for identity deformation with unit fiber"
        );
    }
    #[test]
    fn test_cartilage_consolidation_coeff() {
        let cart = CartilageModel::articular_cartilage();
        let cv = cart.consolidation_coefficient();
        assert!(cv > 0.0);
    }
    #[test]
    fn test_cartilage_creep_compliance_zero_time() {
        let cart = CartilageModel::articular_cartilage();
        let j = cart.creep_compliance(0.0);
        assert!(j.abs() < 1e-6, "At t=0, creep compliance ~ 0 (got {j:.3e})");
    }
    #[test]
    fn test_cartilage_creep_compliance_large_time() {
        let cart = CartilageModel::articular_cartilage();
        let j_inf = cart.creep_compliance(1e8);
        let j_eq = 1.0 / cart.solid_modulus_h_a;
        assert!(
            (j_inf - j_eq).abs() / j_eq < 0.01,
            "Equilibrium compliance at large t"
        );
    }
    #[test]
    fn test_cartilage_stress_relaxation_zero() {
        let cart = CartilageModel::articular_cartilage();
        let g = cart.stress_relaxation(0.0);
        assert!((g - 1.0).abs() < 0.02, "G(0) ≈ 1 (got {g:.4})");
    }
    #[test]
    fn test_cartilage_stress_relaxation_large_time() {
        let cart = CartilageModel::articular_cartilage();
        let g = cart.stress_relaxation(1e8);
        assert!(g < 0.01, "G(∞) ≈ 0");
    }
    #[test]
    fn test_cortical_bone_modulus() {
        let bone = CorticalBone::femur_cortical();
        let e = bone.elastic_modulus();
        assert!(e > 1000.0 && e < 50000.0, "Modulus {e} MPa out of range");
    }
    #[test]
    fn test_cortical_bone_compressive_strength() {
        let bone = CorticalBone::femur_cortical();
        let sc = bone.strength_compressive();
        assert!(sc > 50.0 && sc < 300.0, "Compressive strength {sc} MPa");
    }
    #[test]
    fn test_cortical_bone_tensile_strength() {
        let bone = CorticalBone::femur_cortical();
        let st = bone.strength_tensile();
        assert!(st > 50.0 && st < 300.0, "Tensile strength {st} MPa");
    }
    #[test]
    fn test_trabecular_bone_modulus() {
        let bone = TrabecularBone::vertebral();
        let e = bone.elastic_modulus_trabecular();
        assert!(e > 0.0 && e < 5000.0, "Modulus {e} MPa");
    }
    #[test]
    fn test_trabecular_bone_yield_stress() {
        let bone = TrabecularBone::vertebral();
        let sy = bone.yield_stress_trabecular();
        assert!(sy > 0.0 && sy < 20.0, "Yield stress {sy} MPa");
    }
    #[test]
    fn test_trabecular_modulus_increases_with_bvtv() {
        let bone_low = TrabecularBone::new(0.10, 0.55);
        let bone_high = TrabecularBone::new(0.30, 0.55);
        assert!(
            bone_high.elastic_modulus_trabecular() > bone_low.elastic_modulus_trabecular(),
            "Higher BV/TV → higher stiffness"
        );
    }
    #[test]
    fn test_bone_model_effective_modulus() {
        let bone = BoneModel::femoral_neck();
        let e = bone.effective_axial_modulus();
        assert!(e > 0.0, "Effective modulus must be positive");
    }
    #[test]
    fn test_muscle_active_fl_at_opt() {
        let muscle = MuscleModel::tibialis_anterior();
        let fl = muscle.active_force_length(1.0);
        assert!((fl - 1.0).abs() < 1e-10, "fl(1.0) = 1 at optimal length");
    }
    #[test]
    fn test_muscle_active_fl_decreases_off_opt() {
        let muscle = MuscleModel::tibialis_anterior();
        let fl_opt = muscle.active_force_length(1.0);
        let fl_off = muscle.active_force_length(0.5);
        assert!(fl_off < fl_opt, "fl decreases away from optimal");
    }
    #[test]
    fn test_muscle_passive_fl_zero_at_opt() {
        let muscle = MuscleModel::tibialis_anterior();
        let fp = muscle.passive_force_length(1.0);
        assert_eq!(fp, 0.0, "Passive force = 0 at optimal length");
    }
    #[test]
    fn test_muscle_passive_fl_positive_above_opt() {
        let muscle = MuscleModel::tibialis_anterior();
        let fp = muscle.passive_force_length(1.3);
        assert!(fp > 0.0, "Passive force > 0 above optimal");
    }
    #[test]
    fn test_muscle_force_velocity_isometric() {
        let muscle = MuscleModel::tibialis_anterior();
        let fv = muscle.force_velocity(0.0);
        assert!((fv - 1.0).abs() < 1e-10, "fv = 1 at isometric (v=0)");
    }
    #[test]
    fn test_muscle_force_velocity_concentric() {
        let muscle = MuscleModel::tibialis_anterior();
        let fv = muscle.force_velocity(-0.5);
        assert!(fv < 1.0, "Force decreases during shortening");
        assert!(fv > 0.0, "Force must be positive");
    }
    #[test]
    fn test_muscle_force_full_activation() {
        let muscle = MuscleModel::tibialis_anterior();
        let f = muscle.muscle_force(muscle.l_opt, 0.0, 1.0);
        assert!(f > 0.0 && f <= muscle.f_max, "Force within bounds");
    }
    #[test]
    fn test_muscle_force_zero_activation() {
        let muscle = MuscleModel::tibialis_anterior();
        let f = muscle.muscle_force(muscle.l_opt, 0.0, 0.0);
        assert!(
            f.abs() < 1.0,
            "Near-zero force at zero activation and optimal length"
        );
    }
    #[test]
    fn test_muscle_max_power() {
        let muscle = MuscleModel::tibialis_anterior();
        let p = muscle.max_power();
        assert!(p > 0.0);
    }
    #[test]
    fn test_skin_effective_modulus_shallow() {
        let skin = SkinModel::human_forearm();
        let e = skin.effective_modulus(0.0);
        assert!(
            (e - skin.e_epidermis).abs() < 1e-6,
            "At zero depth: epidermis modulus"
        );
    }
    #[test]
    fn test_skin_effective_modulus_deep() {
        let skin = SkinModel::human_forearm();
        let e = skin.effective_modulus(skin.thickness_e);
        assert!(
            (e - skin.e_dermis).abs() < 1e-6,
            "At h_e depth: dermis modulus"
        );
    }
    #[test]
    fn test_skin_hertz_force_positive() {
        let skin = SkinModel::human_forearm();
        let e_eff = skin.effective_modulus(0.5e-3);
        let f = skin.indentation_force_hertz(1e-3, e_eff, 0.5e-3);
        assert!(f > 0.0, "Hertz force must be positive");
    }
    #[test]
    fn test_skin_total_thickness() {
        let skin = SkinModel::human_forearm();
        let t = skin.total_thickness();
        assert!((t - skin.thickness_e - skin.thickness_d).abs() < 1e-12);
    }
    #[test]
    fn test_vessel_laplace_stress() {
        let vessel = BloodVesselModel::aorta();
        let p = 16000.0;
        let sigma = vessel.laplace_wall_stress(p);
        assert!(sigma > 0.0, "Wall stress must be positive");
    }
    #[test]
    fn test_vessel_compliance_positive() {
        let vessel = BloodVesselModel::aorta();
        let c = vessel.compliance(16000.0);
        assert!(c > 0.0, "Compliance must be positive");
    }
    #[test]
    fn test_vessel_pwv_aorta() {
        let vessel = BloodVesselModel::aorta();
        let rho = 1060.0;
        let pwv = vessel.pulse_wave_velocity(rho);
        assert!(
            pwv > 1.0 && pwv < 30.0,
            "PWV {pwv} m/s out of typical range"
        );
    }
    #[test]
    fn test_vessel_radius_at_pressure() {
        let vessel = BloodVesselModel::femoral_artery();
        let r = vessel.radius_at_pressure(13000.0);
        assert!(r > vessel.r0, "Radius under pressure > reference radius");
    }
    #[test]
    fn test_tendon_stress_zero_strain() {
        let tendon = TendonLigamentModel::patellar_tendon();
        assert_eq!(tendon.stress_strain_nonlinear(0.0), 0.0);
    }
    #[test]
    fn test_tendon_stress_negative_strain() {
        let tendon = TendonLigamentModel::patellar_tendon();
        assert_eq!(
            tendon.stress_strain_nonlinear(-0.01),
            0.0,
            "No compressive load"
        );
    }
    #[test]
    fn test_tendon_stress_toe_region() {
        let tendon = TendonLigamentModel::patellar_tendon();
        let s = tendon.stress_strain_nonlinear(0.01);
        assert!(s > 0.0 && s < tendon.stress_strain_nonlinear(tendon.transition_strain));
    }
    #[test]
    fn test_tendon_stress_linear_region() {
        let tendon = TendonLigamentModel::patellar_tendon();
        let s = tendon.stress_strain_nonlinear(0.05);
        assert!(s > 0.0);
    }
    #[test]
    fn test_tendon_relaxation_at_zero() {
        let tendon = TendonLigamentModel::patellar_tendon();
        let g = tendon.relaxation_function(0.0);
        assert!((g - 1.0).abs() < 1e-10, "G(0) = 1");
    }
    #[test]
    fn test_tendon_relaxation_decreases_with_time() {
        let tendon = TendonLigamentModel::patellar_tendon();
        let g1 = tendon.relaxation_function(0.1);
        let g2 = tendon.relaxation_function(10.0);
        assert!(g2 < g1, "G(t) decreases with time");
    }
    #[test]
    fn test_tendon_viscoelastic_stress() {
        let tendon = TendonLigamentModel::patellar_tendon();
        let s = tendon.viscoelastic_stress(0.03, 1.0);
        assert!(s > 0.0);
    }
    #[test]
    fn test_disc_stiffness_positive() {
        let disc = IntervertebralDiscModel::lumbar_l4_l5();
        let k = disc.compressive_stiffness();
        assert!(k > 0.0, "Disc compressive stiffness must be positive");
    }
    #[test]
    fn test_disc_intradiscal_pressure() {
        let disc = IntervertebralDiscModel::lumbar_l4_l5();
        let p = disc.intradiscal_pressure(500.0);
        assert!(p > 0.0);
    }
    #[test]
    fn test_meniscus_hoop_stress() {
        let meniscus = MeniscusModel::medial_meniscus();
        let s = meniscus.hoop_stress(1000.0, 50e-6);
        assert!(s > 0.0);
    }
    #[test]
    fn test_heart_valve_bending_stiffness() {
        let valve = HeartValveTissue::aortic_valve();
        let d = valve.bending_stiffness(0.45);
        assert!(d > 0.0);
    }
    #[test]
    fn test_heart_valve_anisotropy() {
        let valve = HeartValveTissue::aortic_valve();
        let r = valve.anisotropy_ratio();
        assert!(r > 1.0, "Circumferential stiffness > radial in valves");
    }
    #[test]
    fn test_cornea_hoop_stress_iop() {
        let cornea = CorneaModel::healthy_cornea();
        let s = cornea.hoop_stress_iop();
        assert!(s > 0.0);
    }
    #[test]
    fn test_cornea_tangent_modulus() {
        let cornea = CorneaModel::healthy_cornea();
        let e_small = cornea.tangent_modulus(0.01);
        let e_large = cornea.tangent_modulus(0.1);
        assert_eq!(e_small, cornea.e_small);
        assert_eq!(e_large, cornea.e_large);
    }
    #[test]
    fn test_ligament_wrap_no_force_slack() {
        let lig = LigamentWrapModel::new(0.1, 1000.0, 0.09);
        let f = lig.restraint_force(0.085);
        assert_eq!(f, 0.0, "No force below slack length");
    }
    #[test]
    fn test_ligament_wrap_force_positive() {
        let lig = LigamentWrapModel::new(0.1, 1000.0, 0.09);
        let f = lig.restraint_force(0.12);
        assert!(f > 0.0, "Force positive above slack");
    }
    #[test]
    fn test_ligament_strain() {
        let lig = LigamentWrapModel::new(0.1, 1000.0, 0.09);
        let eps = lig.strain(0.11);
        assert!((eps - 0.1).abs() < 1e-10);
    }
    #[test]
    fn test_muscle_quadriceps_params() {
        let m = MuscleModel::quadriceps();
        assert!(m.f_max > 0.0);
        assert!(m.l_opt > 0.0);
        assert!(m.v_max > 0.0);
    }
    #[test]
    fn test_acl_ligament_stress_strain() {
        let lig = TendonLigamentModel::acl();
        let s1 = lig.stress_strain_nonlinear(0.01);
        let s2 = lig.stress_strain_nonlinear(0.05);
        assert!(s2 > s1, "Stress increases with strain");
    }
    #[test]
    fn test_vessel_femoral_pwv() {
        let vessel = BloodVesselModel::femoral_artery();
        let pwv = vessel.pulse_wave_velocity(1060.0);
        assert!(pwv > 1.0);
    }
    #[test]
    fn test_bone_model_cortical_trabecular_consistent() {
        let bone = BoneModel::femoral_neck();
        assert!(
            bone.cortical.elastic_modulus() > bone.trabecular.elastic_modulus_trabecular(),
            "Cortical bone stiffer than trabecular"
        );
    }
    #[test]
    fn test_fiber_tissue_monotone_energy() {
        let tissue = FiberReinforcedTissue::aortic_tissue();
        let w1 = tissue.fiber_strain_energy(1.05);
        let w2 = tissue.fiber_strain_energy(1.10);
        assert!(w2 > w1, "Fiber energy increases with stretch");
    }
    #[test]
    fn test_trabecular_isotropy_bounded() {
        let bone = TrabecularBone::new(0.2, 0.55);
        let r = bone.isotropy_ratio();
        assert!((0.0..=1.0).contains(&r), "Isotropy ratio must be in [0,1]");
    }
    #[test]
    fn test_huxley_isometric_force_positive() {
        let m = HuxleyModel::fast_twitch();
        let f0 = m.isometric_force();
        assert!(f0 > 0.0, "Isometric force must be positive: {f0}");
    }
    #[test]
    fn test_huxley_force_velocity_isometric() {
        let m = HuxleyModel::fast_twitch();
        let ratio = m.force_velocity_ratio(0.0);
        assert!((ratio - 1.0).abs() < 1e-10, "At v=0, F/F0 should be 1");
    }
    #[test]
    fn test_huxley_force_velocity_shortening_reduced() {
        let m = HuxleyModel::fast_twitch();
        let ratio = m.force_velocity_ratio(1e-3);
        assert!(ratio < 1.0, "During shortening, force should be reduced");
    }
    #[test]
    fn test_huxley_force_velocity_lengthening_enhanced() {
        let m = HuxleyModel::fast_twitch();
        let ratio = m.force_velocity_ratio(-1e-3);
        assert!(ratio > 1.0, "During lengthening, force should be enhanced");
    }
    #[test]
    fn test_huxley_steady_state_fraction_range() {
        let m = HuxleyModel::fast_twitch();
        let n_mid = m.steady_state_fraction(m.h / 2.0);
        let n_out = m.steady_state_fraction(-1e-9);
        assert!(n_mid > 0.0 && n_mid <= 1.0);
        assert!(n_out == 0.0);
    }
    #[test]
    fn test_huxley_power_zero_at_zero_velocity() {
        let m = HuxleyModel::fast_twitch();
        let p = m.power_output(0.0);
        assert!(p.abs() < 1e-6, "Power should be ~0 at zero velocity");
    }
    #[test]
    fn test_huxley_optimal_velocity_positive() {
        let m = HuxleyModel::fast_twitch();
        let v_opt = m.optimal_velocity(1e-2);
        assert!(
            v_opt > 0.0,
            "Optimal shortening velocity should be positive"
        );
        assert!(
            v_opt < 1e-2,
            "Optimal velocity should be within search range"
        );
    }
    #[test]
    fn test_cardiac_passive_energy_zero_at_reference() {
        let m = CardiacMuscleModel::left_ventricle();
        let w = m.passive_strain_energy(0.0, 0.0, 0.0, 0.0);
        assert!(w.abs() < 1e-10, "Passive energy at zero strain should be 0");
    }
    #[test]
    fn test_cardiac_passive_energy_positive_under_fiber_strain() {
        let m = CardiacMuscleModel::left_ventricle();
        let w = m.passive_strain_energy(0.05, 0.0, 0.0, 0.0);
        assert!(
            w > 0.0,
            "Passive energy must be positive under fiber tension"
        );
    }
    #[test]
    fn test_cardiac_active_stress_zero_below_l0() {
        let m = CardiacMuscleModel::left_ventricle();
        let t = m.active_fiber_stress(m.l0 - 0.1, 1.0);
        assert_eq!(t, 0.0, "No active stress below l0");
    }
    #[test]
    fn test_cardiac_active_stress_increases_with_calcium() {
        let m = CardiacMuscleModel::left_ventricle();
        let t1 = m.active_fiber_stress(2.0, 0.5);
        let t2 = m.active_fiber_stress(2.0, 2.0);
        assert!(t2 > t1, "Active stress increases with calcium");
    }
    #[test]
    fn test_cardiac_total_fiber_stress_positive() {
        let m = CardiacMuscleModel::left_ventricle();
        let t = m.total_fiber_stress(0.05, 0.01, 0.01, 0.0, 2.0, 1.5);
        assert!(t > 0.0, "Total fiber stress should be positive");
    }
    #[test]
    fn test_cardiac_stroke_work_positive() {
        let m = CardiacMuscleModel::left_ventricle();
        let w = m.stroke_work(140e-6, 70e-6, 12000.0);
        assert!(w > 0.0, "Stroke work should be positive");
    }
    #[test]
    fn test_failure_no_failure_below_uts() {
        let c = TissueFailureCriteria::cortical_bone();
        let mode = c.max_principal_stress(50e6, 10e6, 0.0);
        assert_eq!(mode, FailureMode::NoFailure);
    }
    #[test]
    fn test_failure_tensile_above_uts() {
        let c = TissueFailureCriteria::cortical_bone();
        let mode = c.max_principal_stress(200e6, 0.0, 0.0);
        assert_eq!(mode, FailureMode::TensileFailure);
    }
    #[test]
    fn test_failure_compressive_above_ucs() {
        let c = TissueFailureCriteria::cortical_bone();
        let mode = c.max_principal_stress(-250e6, 0.0, 0.0);
        assert_eq!(mode, FailureMode::CompressiveFailure);
    }
    #[test]
    fn test_safety_factor_infinite_for_compression() {
        let c = TissueFailureCriteria::ligament();
        let sf = c.tensile_safety_factor(-1e6);
        assert!(sf.is_infinite());
    }
    #[test]
    fn test_tsai_wu_safe_region() {
        let c = TissueFailureCriteria::cortical_bone();
        let idx = c.tsai_wu_index(1e6, 0.5e6, 0.1e6);
        assert!(
            idx < 1.0,
            "Tsai-Wu index should be < 1 for safe state: {idx}"
        );
    }
    #[test]
    fn test_fatigue_life_finite() {
        let c = TissueFailureCriteria::cortical_bone();
        let nf = c.fatigue_life_cycles(80e6);
        assert!(
            nf > 0.0 && nf.is_finite(),
            "Fatigue life should be finite for nonzero load"
        );
    }
    #[test]
    fn test_fatigue_life_infinite_for_zero_load() {
        let c = TissueFailureCriteria::cortical_bone();
        let nf = c.fatigue_life_cycles(0.0);
        assert!(nf.is_infinite());
    }
    #[test]
    fn test_fracture_criterion_no_failure() {
        let c = TissueFailureCriteria::cortical_bone();
        let mode = c.fracture_criterion(1e6);
        assert_eq!(mode, FailureMode::NoFailure);
    }
    #[test]
    fn test_fracture_criterion_failure() {
        let c = TissueFailureCriteria::cortical_bone();
        let mode = c.fracture_criterion(3e6);
        assert_eq!(mode, FailureMode::FractureFailure);
    }
    #[test]
    fn test_energy_release_rate_positive() {
        let c = TissueFailureCriteria::cortical_bone();
        let g = c.energy_release_rate(2e6, 17e9, 0.3);
        assert!(g > 0.0, "Energy release rate must be positive");
    }
    #[test]
    fn test_toughness_criterion_no_failure() {
        let c = TissueFailureCriteria::cortical_bone();
        let mode = c.toughness_criterion(0.5e6, 17e9, 0.3);
        assert_eq!(mode, FailureMode::NoFailure);
    }
}
