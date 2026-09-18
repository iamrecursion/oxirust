//! Tests of the FCC slip-system table and of the Taylor model (F1116).

use super::taylor::deviatoric_coordinates;
use super::*;

fn sqrt6() -> f64 {
    6.0_f64.sqrt()
}

/// Deterministic splitmix64 generator.
struct SplitMix(u64);

impl SplitMix {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal deviate (Box-Muller).
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Uniformly distributed unit vector.
    fn unit_vector(&mut self) -> [f64; 3] {
        loop {
            let v = [self.gaussian(), self.gaussian(), self.gaussian()];
            let length = dot3(v, v).sqrt();
            if length > 1e-6 {
                return [v[0] / length, v[1] / length, v[2] / length];
            }
        }
    }

    /// Uniformly distributed rotation matrix (row-major) from a Gaussian quaternion.
    fn rotation(&mut self) -> [f64; 9] {
        let q = [
            self.gaussian(),
            self.gaussian(),
            self.gaussian(),
            self.gaussian(),
        ];
        let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    }
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    dot3(a, b).abs() > 1.0 - 1e-12
}

fn apply(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

// ---------------------------------------------------------------------------
// FCC slip systems
// ---------------------------------------------------------------------------

#[test]
fn fcc_has_twelve_distinct_slip_systems() {
    // Regression: the former generator emitted two anti-parallel duplicates and
    // omitted <110> on the (-111) and (1-11) planes (only 10 distinct systems).
    let fcc = FccCrystal::new();
    let s = &fcc.slip_systems;
    assert_eq!(s.len(), 12);
    for sys in s {
        assert!(dot3(sys.slip_direction, sys.slip_normal).abs() < 1e-12);
        assert!((dot3(sys.slip_direction, sys.slip_direction) - 1.0).abs() < 1e-12);
        assert!((dot3(sys.slip_normal, sys.slip_normal) - 1.0).abs() < 1e-12);
    }
    for i in 0..s.len() {
        let same_plane = s
            .iter()
            .filter(|o| parallel(s[i].slip_normal, o.slip_normal))
            .count();
        let same_direction = s
            .iter()
            .filter(|o| parallel(s[i].slip_direction, o.slip_direction))
            .count();
        assert_eq!(
            same_plane, 3,
            "system {i}: each {{111}} plane carries 3 systems"
        );
        assert_eq!(same_direction, 2, "system {i}: each <110> lies on 2 planes");
        for j in (i + 1)..s.len() {
            let coincide = parallel(s[i].slip_normal, s[j].slip_normal)
                && parallel(s[i].slip_direction, s[j].slip_direction);
            assert!(!coincide, "systems {i} and {j} are the same up to sign");
        }
    }
}

#[test]
fn fcc_slip_system_set_is_invariant_under_the_cubic_generators() {
    // 90° about z, 90° about x and 120° about [111] generate the cubic group; a
    // complete {111}<110> set must map onto itself (up to the signs of s and n).
    let generators: [[[f64; 3]; 3]; 3] = [
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ];
    let fcc = FccCrystal::new();
    for (g, m) in generators.iter().enumerate() {
        for (i, sys) in fcc.slip_systems.iter().enumerate() {
            let direction = apply(m, sys.slip_direction);
            let normal = apply(m, sys.slip_normal);
            let image_exists = fcc
                .slip_systems
                .iter()
                .any(|o| parallel(direction, o.slip_direction) && parallel(normal, o.slip_normal));
            assert!(
                image_exists,
                "generator {g} maps system {i} outside the set"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Taylor factor of single crystals
// ---------------------------------------------------------------------------

#[test]
fn taylor_solver_finds_the_384_independent_five_system_combinations() {
    let solver = TaylorSolver::new(&FccCrystal::new().slip_systems).unwrap();
    assert_eq!(
        solver.n_combinations(),
        384,
        "792 five-system subsets of the 12 FCC systems, 384 of them non-singular"
    );
}

#[test]
fn taylor_factor_of_high_symmetry_fcc_orientations() {
    let fcc = FccCrystal::new();
    let solver = TaylorSolver::new(&fcc.slip_systems).unwrap();
    // [100]: M = √6; [111] and [110]: M = 3√6/2; [112]: M = 5√6/4.
    let cases: [([f64; 3], f64); 7] = [
        ([1.0, 0.0, 0.0], sqrt6()),
        ([0.0, -3.0, 0.0], sqrt6()),
        ([1.0, 1.0, 1.0], 1.5 * sqrt6()),
        ([-1.0, 1.0, 1.0], 1.5 * sqrt6()),
        ([1.0, 1.0, 0.0], 1.5 * sqrt6()),
        ([2.0, 1.0, 1.0], 1.25 * sqrt6()),
        ([1.0, 2.0, 3.0], 3.149_343_955_006_977),
    ];
    for (axis, expected) in cases {
        let m = solver.uniaxial_taylor_factor(axis).unwrap();
        assert!(
            (m - expected).abs() <= 1e-9 * expected,
            "axis {axis:?}: M = {m}, expected {expected}"
        );
        assert!((fcc.taylor_factor(axis).unwrap() - m).abs() < 1e-12);
    }
}

#[test]
fn taylor_factor_is_invariant_under_cubic_symmetry_axis_sign_and_scale() {
    let solver = TaylorSolver::new(&FccCrystal::new().slip_systems).unwrap();
    let generators: [[[f64; 3]; 3]; 3] = [
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ];
    let mut rng = SplitMix(17);
    for _ in 0..40 {
        let a = rng.unit_vector();
        let m = solver.uniaxial_taylor_factor(a).unwrap();
        for g in &generators {
            let image = solver.uniaxial_taylor_factor(apply(g, a)).unwrap();
            assert!((image - m).abs() < 1e-9, "{a:?}: {m} vs {image}");
        }
        let flipped = solver
            .uniaxial_taylor_factor([-a[0], -a[1], -a[2]])
            .unwrap();
        let scaled = solver
            .uniaxial_taylor_factor([7.0 * a[0], 7.0 * a[1], 7.0 * a[2]])
            .unwrap();
        assert!((flipped - m).abs() < 1e-9 && (scaled - m).abs() < 1e-9);
    }
}

#[test]
fn taylor_factor_lies_between_the_sachs_and_octahedral_bounds() {
    // Sachs (lower) bound 1/m_max; the largest uniaxial Taylor factor of FCC is
    // 3√6/2 (⟨111⟩, ⟨110⟩); the smallest sits near ⟨116⟩ at about 2.27.
    let fcc = FccCrystal::new();
    let solver = TaylorSolver::new(&fcc.slip_systems).unwrap();
    let mut rng = SplitMix(23);
    for _ in 0..300 {
        let a = rng.unit_vector();
        let m = solver.uniaxial_taylor_factor(a).unwrap();
        let sachs = 1.0 / fcc.max_schmid_factor(a);
        assert!(m >= sachs - 1e-9, "{a:?}: Taylor {m} below Sachs {sachs}");
        assert!(m >= 2.25, "{a:?}: {m}");
        assert!(m <= 1.5 * sqrt6() + 1e-9, "{a:?}: {m}");
    }
}

/// Gaussian elimination with partial pivoting for the 5×5 system `a x = b`.
fn gauss_solve5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> Option<[f64; 5]> {
    for k in 0..5 {
        let pivot = (k..5).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[pivot][k].abs() < 1e-10 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let pivot_row = a[k];
        let pivot_rhs = b[k];
        for r in (k + 1)..5 {
            let f = a[r][k] / pivot_row[k];
            for (entry, &value) in a[r].iter_mut().zip(pivot_row.iter()).skip(k) {
                *entry -= f * value;
            }
            b[r] -= f * pivot_rhs;
        }
    }
    let mut x = [0.0; 5];
    for i in (0..5).rev() {
        let tail: f64 = ((i + 1)..5).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - tail) / a[i][i];
    }
    Some(x)
}

#[test]
fn taylor_minimum_is_certified_by_a_dual_stress_state() {
    // Independent optimality proof (linear-programming duality): for the best
    // five-system combination with slip rates γ there must be a stress state σ
    // with resolved shear stress sign(γ_α) on the active systems and |τ_β| ≤ 1
    // on all twelve systems.  Then σ:D = Σ|γ| is the maximum plastic work and
    // no other slip combination can need less shear.
    let fcc = FccCrystal::new();
    let solver = TaylorSolver::new(&fcc.slip_systems).unwrap();
    let schmid: Vec<[f64; 5]> = fcc
        .slip_systems
        .iter()
        .map(|s| deviatoric_coordinates(&s.schmid_tensor))
        .collect();
    let mut rng = SplitMix(41);
    for _ in 0..25 {
        let a = rng.unit_vector();
        let mut d_tensor = [0.0; 9];
        for (k, e) in d_tensor.iter_mut().enumerate() {
            let (i, j) = (k / 3, k % 3);
            *e = 1.5 * a[i] * a[j] - if i == j { 0.5 } else { 0.0 };
        }
        let d = deviatoric_coordinates(&d_tensor);
        let mut best: Option<(f64, [usize; 5], [f64; 5])> = None;
        for i0 in 0..12 {
            for i1 in (i0 + 1)..12 {
                for i2 in (i1 + 1)..12 {
                    for i3 in (i2 + 1)..12 {
                        for i4 in (i3 + 1)..12 {
                            let subset = [i0, i1, i2, i3, i4];
                            let mut matrix = [[0.0; 5]; 5];
                            for (c, &system) in subset.iter().enumerate() {
                                for (r, row) in matrix.iter_mut().enumerate() {
                                    row[c] = schmid[system][r];
                                }
                            }
                            let Some(gamma) = gauss_solve5(matrix, d) else {
                                continue;
                            };
                            let total: f64 = gamma.iter().map(|g| g.abs()).sum();
                            if best.as_ref().is_none_or(|(t, _, _)| total < *t) {
                                best = Some((total, subset, gamma));
                            }
                        }
                    }
                }
            }
        }
        let (total, subset, gamma) = best.expect("some combination is non-singular");
        let mut rows = [[0.0; 5]; 5];
        let mut signs = [0.0; 5];
        for (k, &system) in subset.iter().enumerate() {
            rows[k] = schmid[system];
            signs[k] = gamma[k].signum();
        }
        let sigma = gauss_solve5(rows, signs).expect("dual system is regular");
        let worst_tau = schmid
            .iter()
            .map(|p| {
                p.iter()
                    .zip(sigma.iter())
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
                    .abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            worst_tau <= 1.0 + 1e-9,
            "{a:?}: the stress certificate violates the yield condition ({worst_tau})"
        );
        let work: f64 = sigma.iter().zip(d.iter()).map(|(s, e)| s * e).sum();
        assert!(
            (work - total).abs() <= 1e-9 * total,
            "{a:?}: dual work {work} != primal shear {total}"
        );
        let eps_eq = (2.0_f64 / 3.0).sqrt() * d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let m = solver.uniaxial_taylor_factor(a).unwrap();
        assert!(
            (m - total / eps_eq).abs() <= 1e-9 * m,
            "{a:?}: solver {m} vs certified {}",
            total / eps_eq
        );
    }
}

#[test]
fn taylor_factor_of_a_general_strain_rate_is_scale_free_and_deviatoric() {
    let solver = TaylorSolver::new(&FccCrystal::new().slip_systems).unwrap();
    // Plane-strain compression D = diag(1, 0, −1).
    let plane = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
    let m = solver.taylor_factor(&plane).unwrap();
    assert!(m > 2.0 && m < 4.5, "plane-strain Taylor factor {m}");
    // The magnitude of the strain rate is irrelevant.
    let scaled = plane.map(|x| 3.0 * x);
    assert!((solver.taylor_factor(&scaled).unwrap() - m).abs() < 1e-12);
    // A superposed spherical part (volume change) is ignored.
    let mut with_pressure = plane;
    for k in [0, 4, 8] {
        with_pressure[k] += 5.0;
    }
    assert!((solver.taylor_factor(&with_pressure).unwrap() - m).abs() < 1e-12);
    // diag(1, −½, −½) is the uniaxial tension along the crystal [100] axis.
    let uniaxial = [1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5];
    assert!((solver.taylor_factor(&uniaxial).unwrap() - sqrt6()).abs() < 1e-9);
    // A purely spherical strain rate has no deviatoric part and is rejected.
    assert!(matches!(
        solver.taylor_factor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Err(crate::Error::InvalidInput(_))
    ));
    assert!(solver.taylor_factor(&[0.0; 9]).is_err());
    assert!(
        solver
            .taylor_factor(&[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .is_err()
    );
}

#[test]
fn taylor_solver_rejects_unusable_crystals() {
    let fcc = FccCrystal::new();
    assert!(matches!(
        TaylorSolver::new(&fcc.slip_systems[..4]),
        Err(crate::Error::InvalidInput(_))
    ));
    // Three coplanar systems repeated: no five of them are independent.
    let mut dependent: Vec<SlipSystem> = fcc.slip_systems[..3].to_vec();
    dependent.extend_from_slice(&fcc.slip_systems[..3]);
    assert!(matches!(
        TaylorSolver::new(&dependent),
        Err(crate::Error::SingularSystem)
    ));
    // Too many systems for the exhaustive enumeration.
    let many: Vec<SlipSystem> = (0..40).map(|i| fcc.slip_systems[i % 12].clone()).collect();
    assert!(matches!(
        TaylorSolver::new(&many),
        Err(crate::Error::InvalidInput(_))
    ));
    let solver = TaylorSolver::new(&fcc.slip_systems).unwrap();
    assert!(solver.uniaxial_taylor_factor([0.0; 3]).is_err());
    assert!(solver.uniaxial_taylor_factor([f64::NAN, 1.0, 0.0]).is_err());
}

// ---------------------------------------------------------------------------
// PolycrystalFem::taylor_factor
// ---------------------------------------------------------------------------

fn single_grain(rotation: [f64; 9]) -> PolycrystalFem {
    let mut poly = PolycrystalFem::new(1, 100e6);
    poly.grain_rotations = vec![rotation];
    poly
}

fn identity() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// Rotation whose rows are the sample axes expressed in the crystal frame, with
/// the first row along `direction`.
fn rotation_with_first_row(direction: [f64; 3], second: [f64; 3]) -> [f64; 9] {
    let r0 = {
        let l = dot3(direction, direction).sqrt();
        [direction[0] / l, direction[1] / l, direction[2] / l]
    };
    let r1 = {
        let l = dot3(second, second).sqrt();
        [second[0] / l, second[1] / l, second[2] / l]
    };
    let r2 = cross3(r0, r1);
    [
        r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2],
    ]
}

#[test]
fn polycrystal_taylor_factor_of_single_grains_matches_the_crystal_value() {
    // Identity: sample x ∥ [100].
    let m_100 = single_grain(identity())
        .taylor_factor([1.0, 0.0, 0.0])
        .unwrap();
    assert!((m_100 - sqrt6()).abs() < 1e-9, "{m_100}");
    // The old |l_x·l_y| surrogate was 0 here and about 0.5 at most.
    assert!(m_100 > 2.4);
    // First row of R = [111]/√3: the crystal [111] is along the sample x-axis.
    let r111 = rotation_with_first_row([1.0, 1.0, 1.0], [1.0, -1.0, 0.0]);
    let m_111 = single_grain(r111).taylor_factor([1.0, 0.0, 0.0]).unwrap();
    assert!((m_111 - 1.5 * sqrt6()).abs() < 1e-9, "{m_111}");
    // A loading axis that is not the first sample axis: the second row is [1,-1,0].
    let m_110 = single_grain(r111).taylor_factor([0.0, 1.0, 0.0]).unwrap();
    assert!((m_110 - 1.5 * sqrt6()).abs() < 1e-9, "{m_110}");
}

#[test]
fn polycrystal_taylor_factor_follows_the_crystal_to_sample_convention() {
    // R maps crystal → sample vectors (σ_crystal = Rᵀ σ R in apply_macro_stress), so a
    // sample axis a corresponds to the crystal direction Rᵀ a = a row of R.
    let r = rotation_with_first_row([1.0, 1.0, 2.0], [1.0, -1.0, 0.0]);
    let fcc = FccCrystal::new();
    let m = single_grain(r).taylor_factor([1.0, 0.0, 0.0]).unwrap();
    assert!((m - 1.25 * sqrt6()).abs() < 1e-9, "[112] along x: {m}");
    // The transposed matrix would put the crystal direction (R00, R10, R20) along x.
    let wrong_axis = [r[0], r[3], r[6]];
    let m_wrong = fcc.taylor_factor(wrong_axis).unwrap();
    assert!(
        (m - m_wrong).abs() > 0.05,
        "the test must distinguish R from Rᵀ: {m} vs {m_wrong}"
    );
    // Cross-check with the explicit crystal-frame axes for all three sample axes.
    let poly = single_grain(r);
    for (k, sample_axis) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .iter()
        .enumerate()
    {
        let crystal_axis = [r[3 * k], r[3 * k + 1], r[3 * k + 2]];
        let expected = fcc.taylor_factor(crystal_axis).unwrap();
        let got = poly.taylor_factor(*sample_axis).unwrap();
        assert!(
            (got - expected).abs() < 1e-12,
            "sample axis {k}: {got} vs {expected}"
        );
    }
}

#[test]
fn polycrystal_taylor_factor_is_the_volume_weighted_grain_average() {
    let mut poly = PolycrystalFem::new(2, 100e6);
    poly.grain_rotations = vec![
        identity(),
        rotation_with_first_row([1.0, 1.0, 1.0], [1.0, -1.0, 0.0]),
    ];
    poly.volume_fractions = vec![0.25, 0.75];
    let expected = 0.25 * sqrt6() + 0.75 * 1.5 * sqrt6();
    let m = poly.taylor_factor([1.0, 0.0, 0.0]).unwrap();
    assert!((m - expected).abs() < 1e-9, "{m} vs {expected}");
    // Unnormalised fractions are normalised by their sum.
    poly.volume_fractions = vec![1.0, 3.0];
    let m_scaled = poly.taylor_factor([1.0, 0.0, 0.0]).unwrap();
    assert!((m_scaled - expected).abs() < 1e-9);
    // A zero-weight grain contributes nothing.
    poly.volume_fractions = vec![0.0, 1.0];
    let m_one = poly.taylor_factor([2.0, 0.0, 0.0]).unwrap();
    assert!((m_one - 1.5 * sqrt6()).abs() < 1e-9);
}

#[test]
fn polycrystal_taylor_factor_of_a_random_fcc_texture_is_about_3_06() {
    // Classical result (Taylor 1938, Bishop & Hill 1951): M ≈ 3.06 for a random FCC
    // aggregate in uniaxial tension.  2000 uniformly distributed random rotations.
    let n = 2000;
    let mut rng = SplitMix(2026);
    let mut poly = PolycrystalFem::new(n, 100e6);
    poly.grain_rotations = (0..n).map(|_| rng.rotation()).collect();
    poly.volume_fractions = vec![1.0 / n as f64; n];
    let m = poly.taylor_factor([1.0, 0.0, 0.0]).unwrap();
    assert!((m - 3.06).abs() < 0.05, "random-texture Taylor factor {m}");
    // The aggregate factor does not depend on the loading direction for a random texture.
    let m_z = poly.taylor_factor([0.3, -0.5, 0.8]).unwrap();
    assert!((m_z - 3.06).abs() < 0.05, "{m_z}");
}

#[test]
fn polycrystal_taylor_factor_rejects_invalid_aggregates() {
    let base = PolycrystalFem::new(3, 100e6);
    let invalid_input = |poly: &PolycrystalFem, axis: [f64; 3]| {
        matches!(poly.taylor_factor(axis), Err(crate::Error::InvalidInput(_)))
    };
    assert!(invalid_input(&base, [0.0, 0.0, 0.0]));
    assert!(invalid_input(&base, [f64::NAN, 0.0, 1.0]));
    let mut negative = base.clone();
    negative.volume_fractions[1] = -0.1;
    assert!(invalid_input(&negative, [1.0, 0.0, 0.0]));
    let mut empty_weights = base.clone();
    empty_weights.volume_fractions = vec![0.0; 3];
    assert!(invalid_input(&empty_weights, [1.0, 0.0, 0.0]));
    let mut ragged = base.clone();
    ragged.volume_fractions.pop();
    assert!(invalid_input(&ragged, [1.0, 0.0, 0.0]));
    let mut not_a_rotation = base.clone();
    for entry in not_a_rotation.grain_rotations[0].iter_mut() {
        *entry *= 2.0;
    }
    assert!(invalid_input(&not_a_rotation, [1.0, 0.0, 0.0]));
    let mut mirror = base.clone();
    mirror.grain_rotations[2] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    assert!(invalid_input(&mirror, [1.0, 0.0, 0.0]));
    let mut few_systems = base.clone();
    few_systems.crystal.slip_systems.truncate(4);
    assert!(invalid_input(&few_systems, [1.0, 0.0, 0.0]));
    // The default aggregate (Euler-angle fan) is valid and finite.
    let m = base.taylor_factor([1.0, 0.0, 0.0]).unwrap();
    assert!(m.is_finite() && m > 2.0 && m < 3.7, "{m}");
}
