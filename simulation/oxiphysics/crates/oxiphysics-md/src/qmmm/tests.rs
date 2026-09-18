// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests of the QM/MM interface ([`super`]).

use super::*;

// ---- LinkAtom ----------------------------------------------------------

#[test]
fn test_link_atom_ratio_zero_equals_qm() {
    let mut la = LinkAtom {
        qm_atom_idx: 0,
        mm_atom_idx: 1,
        position: [0.0; 3],
        g_ratio: 0.0,
    };
    let qm_pos = [1.0, 2.0, 3.0];
    let mm_pos = [4.0, 5.0, 6.0];
    la.update_position(qm_pos, mm_pos);
    assert!((la.position[0] - qm_pos[0]).abs() < 1e-12);
    assert!((la.position[1] - qm_pos[1]).abs() < 1e-12);
    assert!((la.position[2] - qm_pos[2]).abs() < 1e-12);
}

#[test]
fn test_link_atom_ratio_one_equals_mm() {
    let mut la = LinkAtom {
        qm_atom_idx: 0,
        mm_atom_idx: 1,
        position: [0.0; 3],
        g_ratio: 1.0,
    };
    let qm_pos = [1.0, 2.0, 3.0];
    let mm_pos = [4.0, 5.0, 6.0];
    la.update_position(qm_pos, mm_pos);
    assert!((la.position[0] - mm_pos[0]).abs() < 1e-12);
    assert!((la.position[1] - mm_pos[1]).abs() < 1e-12);
    assert!((la.position[2] - mm_pos[2]).abs() < 1e-12);
}

// ---- TightBindingHamiltonian -------------------------------------------

#[test]
fn test_hamiltonian_symmetric_two_atoms() {
    let tb = TightBindingHamiltonian {
        atomic_numbers: vec![6, 8],
        positions: vec![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        n_electrons: 7,
    };
    let h = tb.build_hamiltonian();
    let _n = h.len();
    for (i, hi) in h.iter().enumerate() {
        for (j, &val) in hi.iter().enumerate() {
            assert!(
                (val - h[j][i]).abs() < 1e-12,
                "H[{i}][{j}] != H[{j}][{i}]: {} vs {}",
                val,
                h[j][i]
            );
        }
    }
}

#[test]
fn test_slater_koster_decays_with_distance() {
    let r_near = TightBindingHamiltonian::slater_koster_integral(6, 6, 1.0);
    let r_far = TightBindingHamiltonian::slater_koster_integral(6, 6, 5.0);
    assert!(
        r_near > r_far,
        "SK integral should decay: {r_near} > {r_far}"
    );
}

// ---- ElectrostaticEmbedding --------------------------------------------

#[test]
fn test_electrostatic_same_sign_positive_energy() {
    let qm = vec![QmAtom {
        index: 0,
        position: [0.0, 0.0, 0.0],
        atomic_number: 1,
        charge: 1.0,
        gradient: [0.0; 3],
    }];
    let mm = vec![MmAtom {
        index: 0,
        position: [3.0, 0.0, 0.0],
        charge: 1.0,
        epsilon: 0.5,
        sigma: 3.5,
        gradient: [0.0; 3],
    }];
    let e = ElectrostaticEmbedding::qm_mm_interaction(&qm, &mm);
    assert!(
        e > 0.0,
        "Same-sign charges should give positive Coulomb energy: {e}"
    );
}

// ---- QmmmSystem --------------------------------------------------------

/// Standalone QM energy (kcal/mol) of `numbers` at `positions` (neutral singlet
/// unless `charge` says otherwise).
fn standalone_qm_energy(
    method: QmMethod,
    charge: i32,
    numbers: &[u32],
    positions: &[[f64; 3]],
) -> f64 {
    let mut region = QmRegion::new((0..numbers.len()).collect(), method, charge, 1);
    region.atomic_numbers = numbers.to_vec();
    region.set_positions(positions.to_vec());
    region.run().expect("standalone QM run");
    region.energy
}

/// Lennard-Jones energy of a QM atom (default parameters) with an MM atom.
fn qm_mm_lj(eps_mm: f64, sigma_mm: f64, r: f64) -> f64 {
    let eps = (0.5_f64 / 4.184 * eps_mm).sqrt();
    let sigma = 0.5 * (3.5 + sigma_mm);
    let sr6 = (sigma / r).powi(6);
    4.0 * eps * (sr6 * sr6 - sr6)
}

const H2_POSITIONS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]];

#[test]
fn test_qmmm_total_energy_contains_the_qm_electronic_energy() {
    // H2 plus a distant MM point charge: the total must be the standalone
    // SCC-DFTB energy of H2 (the model-repulsion-only energy of the old code
    // was a few kcal/mol and positive).
    let e_qm = standalone_qm_energy(QmMethod::SccDftb, 0, &[1, 1], &H2_POSITIONS);
    assert!(e_qm < -10.0, "H2 SCC-DFTB energy {e_qm} kcal/mol");

    let mut sys = QmmmSystem::new();
    sys.add_qm_atom(H2_POSITIONS[0], 1, 0.0);
    sys.add_qm_atom(H2_POSITIONS[1], 1, 0.0);
    sys.add_mm_atom([20.0, 0.0, 0.0], 1.0, 0.1, 3.0);
    let total = sys.total_energy().expect("QM/MM energy");
    assert!(
        (total - e_qm).abs() < 0.1,
        "total {total} vs standalone {e_qm}"
    );
    // Neutral QM charges and a 20 Å separation leave only a ~1e-5 kcal/mol LJ tail.
    assert!((total - e_qm).abs() < 1e-3, "total {total} vs {e_qm}");
    assert_eq!(
        total,
        sys.total_energy_with(QmMethod::SccDftb, 0, 1).unwrap()
    );
}

#[test]
fn test_qmmm_total_energy_adds_the_classical_terms_in_kcal_per_mol() {
    let e_qm = standalone_qm_energy(QmMethod::SccDftb, 0, &[1, 1], &H2_POSITIONS);
    let mut sys = QmmmSystem::new();
    sys.add_qm_atom(H2_POSITIONS[0], 1, 0.4);
    sys.add_qm_atom(H2_POSITIONS[1], 1, -0.4);
    let mm = [
        ([4.0, 0.5, 0.0], -0.3, 0.2, 3.0),
        ([-3.5, 1.0, 0.5], 0.6, 0.05, 3.8),
    ];
    for (pos, q, eps, sigma) in mm {
        sys.add_mm_atom(pos, q, eps, sigma);
    }
    let dist = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let mut e_coulomb = 0.0;
    let mut e_lj = 0.0;
    for (qm_pos, qm_q) in H2_POSITIONS.iter().zip([0.4, -0.4]) {
        for (pos, q, eps, sigma) in mm {
            let r = dist(*qm_pos, pos);
            e_coulomb += 332.0636 * qm_q * q / r;
            e_lj += qm_mm_lj(eps, sigma, r);
        }
    }
    assert!(
        e_coulomb.abs() > 1.0,
        "Coulomb term must matter: {e_coulomb}"
    );
    let total = sys.total_energy().expect("QM/MM energy");
    assert!(
        (total - (e_qm + e_coulomb + e_lj)).abs() < 1e-9,
        "total {total} vs {}",
        e_qm + e_coulomb + e_lj
    );
}

#[test]
fn test_qm_default_lj_epsilon_is_half_a_kilojoule_in_kcal_per_mol() {
    // MM atom with the QM default parameters (0.5 kJ/mol = 0.5/4.184 kcal/mol,
    // 3.5 Å): at the LJ minimum 2^(1/6) σ the pair energy is exactly −ε.
    let eps = 0.5 / 4.184;
    let r_min = 2.0_f64.powf(1.0 / 6.0) * 3.5;
    let qm = [QmAtom {
        index: 0,
        position: [0.0; 3],
        atomic_number: 6,
        charge: 0.0,
        gradient: [0.0; 3],
    }];
    let mm = [MmAtom {
        index: 0,
        position: [r_min, 0.0, 0.0],
        charge: 0.0,
        epsilon: eps,
        sigma: 3.5,
        gradient: [0.0; 3],
    }];
    let e = VanDerWaalsQmMm::lj_energy(&qm, &mm);
    assert!((e + eps).abs() < 1e-12, "pair energy {e} vs {}", -eps);
    // The gradient vanishes at the minimum.
    assert!(VanDerWaalsQmMm::lj_gradient_qm(&qm, &mm)[0][0].abs() < 1e-12);
    // Excluding the only pair removes the interaction.
    assert_eq!(
        VanDerWaalsQmMm::lj_energy_excluding(&qm, &mm, &[(0, 0)]),
        0.0
    );
    assert_eq!(
        ElectrostaticEmbedding::qm_mm_interaction_excluding(&qm, &mm, &[(0, 0)]),
        0.0
    );
}

/// Methane-like QM fragment: C and three H of an ideal CH4 (C–H 1.09 Å); the
/// fourth H position is where the cap goes, and an MM carbon sits on the same
/// axis at 1.54 Å (a severed C–C bond).
fn capped_methyl_system(with_link_atom: bool) -> (QmmmSystem, Vec<[f64; 3]>) {
    let s = 1.0 / 3.0_f64.sqrt();
    let dirs = [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]];
    let h_positions: Vec<[f64; 3]> = dirs
        .iter()
        .map(|d| [1.09 * d[0], 1.09 * d[1], 1.09 * d[2]])
        .collect();
    let mut sys = QmmmSystem::new();
    let c = sys.add_qm_atom([0.0; 3], 6, 0.0);
    for h in &h_positions[..3] {
        sys.add_qm_atom(*h, 1, 0.0);
    }
    let mm_pos = [1.54 * dirs[3][0], 1.54 * dirs[3][1], 1.54 * dirs[3][2]];
    let mm = sys.add_mm_atom(mm_pos, 0.0, 0.1, 3.4);
    if with_link_atom {
        sys.add_link_atom(c, mm, 1.09 / 1.54);
    }
    (sys, h_positions)
}

#[test]
fn test_qmmm_link_atom_caps_the_qm_region_and_excludes_the_bonded_pair() {
    let (sys, h_positions) = capped_methyl_system(true);
    // The cap lands exactly on the fourth CH4 hydrogen.
    let cap = sys.link_atoms[0].position;
    for d in 0..3 {
        assert!((cap[d] - h_positions[3][d]).abs() < 1e-12);
    }
    let mut numbers = vec![6u32];
    numbers.extend([1, 1, 1, 1]);
    let mut positions = vec![[0.0; 3]];
    positions.extend(h_positions.iter().copied());
    let e_ch4 = standalone_qm_energy(QmMethod::SccDftb, 0, &numbers, &positions);

    // Classical terms: all QM–MM pairs except the bonded C–C(MM) pair.
    let mm_pos = sys.mm_atoms[0].position;
    let dist = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let e_lj_h: f64 = h_positions[..3]
        .iter()
        .map(|&h| qm_mm_lj(0.1, 3.4, dist(h, mm_pos)))
        .sum();
    assert!(e_lj_h > 10.0, "1-3 repulsion {e_lj_h}");
    let naive = VanDerWaalsQmMm::lj_energy(&sys.qm_atoms, &sys.mm_atoms);
    assert!(naive > e_lj_h + 1000.0, "bonded C···C(MM) LJ {naive}");

    let total = sys.total_energy().expect("capped QM/MM energy");
    assert!(
        (total - (e_ch4 + e_lj_h)).abs() < 1e-9,
        "total {total} vs {}",
        e_ch4 + e_lj_h
    );
}

#[test]
fn test_qmmm_uncapped_radical_fragment_is_rejected() {
    // Without the hydrogen cap the CH3 fragment has 7 valence electrons.
    let (sys, _) = capped_methyl_system(false);
    assert_eq!(sys.total_energy(), Err(QmError::OpenShellNotSupported));
}

#[test]
fn test_qmmm_total_energy_errors_instead_of_fabricating_a_value() {
    // Unsupported element.
    let mut sys = QmmmSystem::new();
    sys.add_qm_atom([0.0; 3], 92, 0.0);
    assert_eq!(sys.total_energy(), Err(QmError::UnsupportedElement(92)));

    // Link atom referring to a non-existent MM atom, and to a non-existent QM atom.
    let mut sys = QmmmSystem::new();
    sys.add_qm_atom(H2_POSITIONS[0], 1, 0.0);
    sys.add_qm_atom(H2_POSITIONS[1], 1, 0.0);
    sys.link_atoms.push(LinkAtom {
        qm_atom_idx: 0,
        mm_atom_idx: 3,
        position: [0.0; 3],
        g_ratio: 0.7,
    });
    assert_eq!(
        sys.total_energy(),
        Err(QmError::InvalidLinkAtom {
            region: "MM",
            index: 3,
            len: 0
        })
    );
    sys.add_mm_atom([5.0, 0.0, 0.0], 0.0, 0.1, 3.0);
    sys.link_atoms[0].mm_atom_idx = 0;
    sys.link_atoms[0].qm_atom_idx = 7;
    assert_eq!(
        sys.total_energy(),
        Err(QmError::InvalidLinkAtom {
            region: "QM",
            index: 7,
            len: 2
        })
    );
}

#[test]
fn test_qmmm_empty_qm_region_has_no_qm_mm_energy() {
    assert_eq!(QmmmSystem::new().total_energy(), Ok(0.0));
    let mut sys = QmmmSystem::new();
    sys.add_mm_atom([1.0, 2.0, 3.0], -0.8, 0.1, 3.2);
    assert_eq!(sys.total_energy(), Ok(0.0));
}

#[test]
fn test_qmmm_total_energy_with_forwards_method_charge_and_multiplicity() {
    // H3+ (2 electrons) is closed shell only with charge +1.
    let tri = [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.45, 0.779_422_863, 0.0]];
    let mut sys = QmmmSystem::new();
    for p in tri {
        sys.add_qm_atom(p, 1, 0.0);
    }
    assert_eq!(
        sys.total_energy_with(QmMethod::SccDftb, 0, 1),
        Err(QmError::OpenShellNotSupported)
    );
    let e = sys
        .total_energy_with(QmMethod::SccDftb, 1, 1)
        .expect("H3+ SCC-DFTB");
    let expected = standalone_qm_energy(QmMethod::SccDftb, 1, &[1, 1, 1], &tri);
    assert!((e - expected).abs() < 1e-9, "{e} vs {expected}");
    // A spin multiplicity other than 1 is rejected by the closed-shell engines.
    assert_eq!(
        sys.total_energy_with(QmMethod::SccDftb, 1, 3),
        Err(QmError::OpenShellNotSupported)
    );

    // Another QM method is honoured as well.
    let mut h2 = QmmmSystem::new();
    h2.add_qm_atom(H2_POSITIONS[0], 1, 0.0);
    h2.add_qm_atom(H2_POSITIONS[1], 1, 0.0);
    let pm3 = h2.total_energy_with(QmMethod::Pm3, 0, 1).expect("PM3 H2");
    let pm3_ref = standalone_qm_energy(QmMethod::Pm3, 0, &[1, 1], &H2_POSITIONS);
    assert!((pm3 - pm3_ref).abs() < 1e-9);
    let dftb = h2.total_energy().expect("DFTB H2");
    assert!((pm3 - dftb).abs() > 1.0, "PM3 and SCC-DFTB energies differ");
}

#[test]
fn test_valence_electrons_follow_the_periodic_table() {
    let expected: [(u8, usize); 30] = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 1),
        (6, 4),
        (7, 5),
        (8, 6),
        (9, 7),
        (10, 8),
        (11, 1),
        (16, 6),
        (17, 7),
        (18, 8),
        (19, 1),
        (20, 2),
        (21, 3),
        (30, 12),
        (31, 3),
        (35, 7),
        (36, 8),
        (53, 7),
        (54, 8),
        (72, 4),
        (80, 12),
        (81, 3),
        (86, 8),
        (103, 3),
        (112, 12),
        (118, 8),
        (119, 0),
    ];
    for (z, n) in expected {
        assert_eq!(valence_electrons(z), n, "Z = {z}");
    }
    // Group number = valence electrons for the whole p block of period 3.
    assert_eq!(valence_electrons(14), 4);
    assert_eq!(valence_electrons(15), 5);
}

#[test]
fn test_qmmm_hamiltonian_electron_count_is_a_valence_count() {
    let mut sys = QmmmSystem::new();
    sys.add_qm_atom([0.0; 3], 8, 0.0);
    sys.add_qm_atom([0.96, 0.0, 0.0], 1, 0.0);
    sys.add_qm_atom([-0.24, 0.93, 0.0], 1, 0.0);
    // 6 + 1 + 1 valence electrons (the old Z/2 rule gave 4 + 0 + 0).
    assert_eq!(sys.hamiltonian.n_electrons, 8);
}

// ---- ChargeEquilibration -----------------------------------------------

#[test]
fn test_charge_equilibration_sum_to_total() {
    let ceq = ChargeEquilibration {
        electronegativity: vec![4.53, 6.93, 4.53], // C, O, C
        hardness: vec![5.0, 8.0, 5.0],
    };
    let positions = vec![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]];
    let total_q = 0.0;
    let charges = ceq.compute_charges(&positions, total_q);
    let sum: f64 = charges.iter().sum();
    assert!(
        (sum - total_q).abs() < 1e-8,
        "Charges should sum to {total_q}, got {sum}"
    );
}

// ---- QmRegionSelector --------------------------------------------------

#[test]
fn test_qm_region_selector_includes_core() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
    let selector = QmRegionSelector::new(vec![0], 1.5);
    let qm = selector.select_qm_atoms(&positions);
    assert!(qm.contains(&0), "core atom 0 should be selected");
    assert!(
        qm.contains(&1),
        "atom 1 at 1.0 Å should be within cutoff 1.5"
    );
    assert!(!qm.contains(&2), "atom 2 at 5.0 Å should be outside cutoff");
}

#[test]
fn test_qm_region_selector_boundary_bonds() {
    let qm_atoms = vec![0usize, 1];
    let bonded_pairs = vec![(0, 1), (1, 2), (2, 3)];
    let selector = QmRegionSelector::new(vec![0], 2.0);
    let boundary = selector.boundary_bonds(&qm_atoms, &bonded_pairs);
    // Bond (1, 2): atom 1 is QM, atom 2 is MM → boundary bond
    assert_eq!(
        boundary.len(),
        1,
        "exactly one QM/MM boundary bond, got {}",
        boundary.len()
    );
    assert_eq!(boundary[0], (1, 2));
}

// ---- PointChargeField --------------------------------------------------

#[test]
fn test_point_charge_potential_single_charge() {
    let mut field = PointChargeField::new();
    field.add_charge(1.0, [0.0, 0.0, 0.0]);
    let ke = 332.0636;
    let v = field.potential_at([1.0, 0.0, 0.0]);
    assert!(
        (v - ke).abs() < 1e-6,
        "V at 1 Å from unit charge should be ke={ke}, got {v}"
    );
}

#[test]
fn test_point_charge_field_direction() {
    let mut field = PointChargeField::new();
    field.add_charge(1.0, [0.0, 0.0, 0.0]);
    let e = field.field_at([1.0, 0.0, 0.0]);
    assert!(
        e[0] > 0.0,
        "Electric field x-component should point away from positive charge"
    );
    assert!(e[1].abs() < 1e-10, "Ey should be zero along x axis");
    assert!(e[2].abs() < 1e-10, "Ez should be zero along x axis");
}

#[test]
fn test_point_charge_field_empty() {
    let field = PointChargeField::new();
    assert!(field.is_empty());
    let v = field.potential_at([0.0, 0.0, 0.0]);
    assert_eq!(v, 0.0);
}

// ---- MechanicalEmbedding -----------------------------------------------

#[test]
fn test_mechanical_embedding_energy_finite() {
    let qm_positions = vec![[0.0, 0.0, 0.0]];
    let qm_charges = vec![-0.5];
    let mm_atoms = vec![MmAtom {
        index: 0,
        position: [3.0, 0.0, 0.0],
        charge: 0.5,
        epsilon: 0.3,
        sigma: 3.2,
        gradient: [0.0; 3],
    }];
    let e = MechanicalEmbedding::interaction_energy(&qm_positions, &qm_charges, &mm_atoms);
    assert!(
        e.is_finite(),
        "Mechanical embedding energy should be finite, got {e}"
    );
    // Opposite charges: energy should be negative
    assert!(
        e < 0.0,
        "Opposite charges should give negative energy, got {e}"
    );
}

// ---- redistribute_link_force -------------------------------------------

#[test]
fn test_redistribute_link_force_sum_conserved() {
    let link_grad = [1.0, 0.0, 0.0];
    let g = 0.71;
    let mut grad_qm = [0.0f64; 3];
    let mut grad_mm = [0.0f64; 3];
    redistribute_link_force(link_grad, g, &mut grad_qm, &mut grad_mm);
    // Sum should equal link_grad
    for k in 0..3 {
        let total = grad_qm[k] + grad_mm[k];
        assert!(
            (total - link_grad[k]).abs() < 1e-12,
            "Force sum should be conserved at k={k}: {total} vs {}",
            link_grad[k]
        );
    }
}

#[test]
fn test_redistribute_link_force_ratio() {
    let link_grad = [2.0, 0.0, 0.0];
    let g = 0.5;
    let mut grad_qm = [0.0f64; 3];
    let mut grad_mm = [0.0f64; 3];
    redistribute_link_force(link_grad, g, &mut grad_qm, &mut grad_mm);
    // With g=0.5: each gets 50% = 1.0
    assert!((grad_qm[0] - 1.0).abs() < 1e-12, "QM gets (1-g)*F = 1.0");
    assert!((grad_mm[0] - 1.0).abs() < 1e-12, "MM gets g*F = 1.0");
}

// ---- ONIOM tests -------------------------------------------------------

#[test]
fn test_oniom_energy_additivity() {
    // E_ONIOM = E_QM + E_MM - E_MM_model
    let e = oniom_energy(10.0, 50.0, 8.0);
    assert!(
        (e - 52.0).abs() < 1e-12,
        "ONIOM energy should be 52.0, got {e}"
    );
}

#[test]
fn test_oniom_energy_equal_mm() {
    // If E_MM_real == E_MM_model, only E_QM matters
    let e = oniom_energy(5.0, 20.0, 20.0);
    assert!(
        (e - 5.0).abs() < 1e-12,
        "ONIOM with equal MM energies should give E_QM"
    );
}

#[test]
fn test_oniom3_consistency() {
    // Three-layer ONIOM: verify algebraic consistency
    let e = oniom3_energy(10.0, 20.0, 5.0, 50.0, 25.0);
    // = 10 + 20 - 5 + 50 - 25 = 50
    assert!((e - 50.0).abs() < 1e-12, "ONIOM3 should give 50.0, got {e}");
}

// ---- buffer_weight tests -----------------------------------------------

#[test]
fn test_buffer_weight_inner_boundary() {
    let w = buffer_weight(0.9, 1.0, 2.0);
    assert!(
        (w - 1.0).abs() < 1e-12,
        "Inside inner cutoff, weight should be 1.0"
    );
}

#[test]
fn test_buffer_weight_outer_boundary() {
    let w = buffer_weight(2.1, 1.0, 2.0);
    assert!(
        (w - 0.0).abs() < 1e-12,
        "Outside outer cutoff, weight should be 0.0"
    );
}

#[test]
fn test_buffer_weight_midpoint() {
    let w = buffer_weight(1.5, 1.0, 2.0);
    assert!(
        w > 0.0 && w < 1.0,
        "Buffer weight at midpoint should be in (0,1): {w}"
    );
}

#[test]
fn test_buffer_weight_derivative_numerical() {
    let r_i = 1.0_f64;
    let r_o = 2.0_f64;
    let r = 1.5_f64;
    let h = 1e-6;
    let numerical = (buffer_weight(r + h, r_i, r_o) - buffer_weight(r - h, r_i, r_o)) / (2.0 * h);
    let analytical = buffer_weight_derivative(r, r_i, r_o);
    assert!(
        (numerical - analytical).abs() < 1e-5,
        "Buffer weight derivative mismatch: {analytical} vs {numerical}"
    );
}

// ---- capping_hydrogen tests ---------------------------------------------

#[test]
fn test_capping_hydrogen_position_correct_length() {
    let qm = [0.0, 0.0, 0.0];
    let mm = [4.0, 0.0, 0.0];
    let r_ch = 1.09_f64;
    let cap = capping_hydrogen_position(qm, mm, r_ch);
    let actual_dist =
        ((cap[0] - qm[0]).powi(2) + (cap[1] - qm[1]).powi(2) + (cap[2] - qm[2]).powi(2)).sqrt();
    assert!(
        (actual_dist - r_ch).abs() < 1e-10,
        "Capping H should be at r_ch={r_ch}, got {actual_dist}"
    );
}

#[test]
fn test_capping_hydrogen_along_bond_axis() {
    let qm = [0.0, 0.0, 0.0];
    let mm = [0.0, 3.0, 0.0]; // along y axis
    let cap = capping_hydrogen_position(qm, mm, 1.0);
    assert!(cap[0].abs() < 1e-10, "Cap H should be on y-axis (x≈0)");
    assert!(cap[1] > 0.0, "Cap H should be in +y direction");
    assert!(cap[2].abs() < 1e-10, "Cap H should be on y-axis (z≈0)");
}

// ---- link_bond_constraint tests ----------------------------------------

#[test]
fn test_link_bond_constraint_energy_at_target() {
    let link = [1.09, 0.0, 0.0];
    let qm = [0.0, 0.0, 0.0];
    let e = link_bond_constraint_energy(link, qm, 1.09, 100.0);
    assert!(
        e < 1e-10,
        "Constraint energy at target distance should be 0, got {e}"
    );
}

#[test]
fn test_link_bond_constraint_energy_positive() {
    let link = [2.0, 0.0, 0.0];
    let qm = [0.0, 0.0, 0.0];
    let e = link_bond_constraint_energy(link, qm, 1.09, 100.0);
    assert!(
        e > 0.0,
        "Constraint energy away from target should be positive, got {e}"
    );
}

#[test]
fn test_link_bond_constraint_gradient_direction() {
    // When stretched beyond target, gradient should point away from QM atom
    let link = [2.0, 0.0, 0.0];
    let qm = [0.0, 0.0, 0.0];
    let g = link_bond_constraint_gradient(link, qm, 1.09, 100.0);
    assert!(
        g[0] > 0.0,
        "Gradient should point in +x direction (away from QM), got {}",
        g[0]
    );
    assert!(g[1].abs() < 1e-10, "y-gradient should be 0");
}

// ---- Mulliken tests -----------------------------------------------------

#[test]
fn test_mulliken_populations_identity_overlap() {
    // With S = I (identity), populations equal diagonal of P
    let n = 3;
    let p = vec![
        vec![2.0, 0.1, 0.0],
        vec![0.1, 1.5, 0.2],
        vec![0.0, 0.2, 1.0],
    ];
    let s: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let pop = mulliken_populations(&p, &s);
    // With S=I: q_i = P_ii (contribution from each other orbital is 0)
    assert!((pop[0] - p[0][0]).abs() < 1e-12, "pop[0] should be P[0][0]");
    assert!((pop[1] - p[1][1]).abs() < 1e-12, "pop[1] should be P[1][1]");
}

#[test]
fn test_mulliken_charges_neutral_molecule() {
    // If populations equal nuclear charges, all atomic charges are 0
    let populations = vec![6.0, 1.0, 1.0];
    let nuclear = vec![6.0, 1.0, 1.0];
    let charges = mulliken_charges(&populations, &nuclear);
    for (i, &c) in charges.iter().enumerate() {
        assert!(c.abs() < 1e-12, "Mulliken charge {i} should be 0, got {c}");
    }
}

// ---- Ewald correction tests --------------------------------------------

#[test]
fn test_ewald_self_correction_sign() {
    // Self-correction should always be negative (removes overcounting)
    let charges = vec![0.5, -0.5, 0.3];
    let e = ewald_self_correction(&charges, 0.35);
    assert!(e < 0.0, "Ewald self-correction should be negative, got {e}");
}

#[test]
fn test_ewald_self_correction_proportional_to_alpha() {
    let charges = vec![1.0];
    let e1 = ewald_self_correction(&charges, 0.1);
    let e2 = ewald_self_correction(&charges, 0.2);
    assert!(e2 / e1 > 1.9, "Self-correction should scale with alpha");
}

#[test]
fn test_ewald_g0_neutral_system() {
    // Neutral system (Σ q=0, dipole moment=0) → G=0 correction ≈ 0
    let charges = vec![1.0, -1.0];
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let e = ewald_g0_correction(&charges, &positions, 0.35, 10.0);
    // Dipole is [1*0 + (-1)*1, 0, 0] = [-1, 0, 0]
    assert!(
        e.is_finite(),
        "G=0 correction should be finite for dipole system"
    );
}

#[test]
fn test_adaptive_qm_radius_returns_at_least_initial() {
    let core = [0.0, 0.0, 0.0];
    let positions = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
    let gradients = vec![[0.0f64; 3]; 3]; // all zero grads → converge immediately
    let r = adaptive_qm_radius(core, &positions, &gradients, 1.5, 0.5, 5.0, 0.01);
    assert!(
        r >= 1.5,
        "Adaptive radius should be at least the initial radius, got {r}"
    );
}
