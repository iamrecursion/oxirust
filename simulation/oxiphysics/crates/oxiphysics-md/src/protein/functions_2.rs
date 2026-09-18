//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ProteinChain;

/// Kabsch algorithm: optimal rotation superimposing `coords_mob` onto `coords_ref`.
///
/// Both point sets are centred on their centroids and the proper rotation `R`
/// (3×3, `det R = +1`) that minimises
/// `RMSD(coords_ref − centroid_ref, R · (coords_mob − centroid_mob))` is returned.
/// The rotation is obtained from the singular value decomposition of the
/// cross-covariance matrix by [`crate::structure::kabsch_rotation`]; degenerate
/// (collinear / coplanar / single-point) sets are handled there.
///
/// Only the first `min(coords_ref.len(), coords_mob.len())` points are used; for
/// no points the identity is returned.
pub fn kabsch_rotation(coords_ref: &[[f64; 3]], coords_mob: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let n = coords_ref.len().min(coords_mob.len());
    // `structure::kabsch_rotation(p, q)` returns the rotation taking `p` onto `q`,
    // so the mobile set comes first.
    crate::structure::kabsch_rotation(&coords_mob[..n], &coords_ref[..n])
}
/// Translate chain so its centroid coincides with the origin.
pub fn center_chain(chain: &mut ProteinChain) {
    let n = chain.residues.len();
    if n == 0 {
        return;
    }
    let nf = n as f64;
    let cx: f64 = chain.residues.iter().map(|r| r.ca_position[0]).sum::<f64>() / nf;
    let cy: f64 = chain.residues.iter().map(|r| r.ca_position[1]).sum::<f64>() / nf;
    let cz: f64 = chain.residues.iter().map(|r| r.ca_position[2]).sum::<f64>() / nf;
    for r in chain.residues.iter_mut() {
        r.ca_position[0] -= cx;
        r.ca_position[1] -= cy;
        r.ca_position[2] -= cz;
    }
}
/// Cα RMSD after optimal rigid-body superposition (translation and rotation).
///
/// Both chains are centred on their Cα centroids, `chain_b` is rotated onto
/// `chain_a` with the Kabsch rotation ([`kabsch_rotation`]) and the RMSD of the
/// superimposed Cα positions is returned.  Only proper rotations are allowed (no
/// reflections), so a mirror-image chain does not superimpose.  The first
/// `min(chain_a.length(), chain_b.length())` residues are compared; two empty
/// chains give `0.0`.
pub fn rmsd_kabsch(chain_a: &ProteinChain, chain_b: &ProteinChain) -> f64 {
    let n = chain_a.residues.len().min(chain_b.residues.len());
    if n == 0 {
        return 0.0;
    }
    let reference: Vec<[f64; 3]> = chain_a.residues[..n]
        .iter()
        .map(|r| r.ca_position)
        .collect();
    let mobile: Vec<[f64; 3]> = chain_b.residues[..n]
        .iter()
        .map(|r| r.ca_position)
        .collect();
    crate::structure::rmsd_after_alignment(&mobile, &reference)
}
#[cfg(test)]
mod tests_extended {
    use super::super::functions::*;
    use super::super::types::*;
    use super::*;
    fn make_linear_chain(n: usize) -> ProteinChain {
        let mut chain = ProteinChain::new();
        for i in 0..n {
            chain.add_residue(AminoAcid::Ala, [i as f64 * 3.8, 0.0, 0.0]);
        }
        chain
    }
    /// Ideal α-helix Cα trace: radius 2.3 Å, rise 1.5 Å, 100° per residue.
    fn make_alpha_helix(n: usize) -> ProteinChain {
        let mut chain = ProteinChain::new();
        for k in 0..n {
            let phi = (100.0_f64 * k as f64).to_radians();
            chain.add_residue(
                AminoAcid::Ala,
                [2.3 * phi.cos(), 2.3 * phi.sin(), 1.5 * k as f64],
            );
        }
        chain
    }
    /// Rodrigues rotation matrix about the (not necessarily unit) `axis`.
    fn rotation_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (x, y, z) = (axis[0] / len, axis[1] / len, axis[2] / len);
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    }
    fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }
    fn determinant(r: &[[f64; 3]; 3]) -> f64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }
    fn max_abs_difference(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((a[i][j] - b[i][j]).abs());
            }
        }
        worst
    }
    /// `chain` with every Cα moved by `f`.
    fn map_chain(chain: &ProteinChain, f: impl Fn([f64; 3]) -> [f64; 3]) -> ProteinChain {
        let mut out = ProteinChain::new();
        for r in &chain.residues {
            out.add_residue(r.aa, f(r.ca_position));
        }
        out
    }
    #[test]
    fn test_hbond_energy_negative_at_close_range() {
        let donor = [0.0, 0.0, 0.0];
        let acceptor = [3.0, 0.0, 0.0];
        let e = hydrogen_bond_energy(donor, acceptor, 1.0, 2.0, 10.0);
        assert!(
            e < 0.0,
            "H-bond energy should be negative at close range: {e}"
        );
    }
    #[test]
    fn test_hbond_energy_zero_beyond_cutoff() {
        let donor = [0.0, 0.0, 0.0];
        let acceptor = [100.0, 0.0, 0.0];
        let e = hydrogen_bond_energy(donor, acceptor, 1.0, 2.0, 5.0);
        assert_eq!(e, 0.0, "H-bond should be zero beyond cutoff: {e}");
    }
    #[test]
    fn test_hbond_energy_scales_with_epsilon() {
        let donor = [0.0, 0.0, 0.0];
        let acc = [2.0, 0.0, 0.0];
        let e1 = hydrogen_bond_energy(donor, acc, 1.0, 1.5, 10.0);
        let e2 = hydrogen_bond_energy(donor, acc, 2.0, 1.5, 10.0);
        assert!(
            (e2 / e1 - 2.0).abs() < 1e-10,
            "Energy should scale with epsilon: {e1} {e2}"
        );
    }
    #[test]
    fn test_total_hbond_energy_negative() {
        let chain = make_linear_chain(8);
        let e = total_hbond_energy(&chain, 1.0, 5.0, 20.0);
        assert!(e < 0.0, "Total H-bond energy should be negative: {e}");
    }
    #[test]
    fn test_total_hbond_energy_single_residue_zero() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        let e = total_hbond_energy(&chain, 1.0, 2.0, 10.0);
        assert_eq!(e, 0.0, "Single residue: no H-bonds");
    }
    #[test]
    fn test_ramachandran_statistics_fractions_sum_to_one() {
        let pairs: Vec<(f64, f64)> = vec![
            ((-60.0_f64).to_radians(), (-45.0_f64).to_radians()),
            ((-120.0_f64).to_radians(), (130.0_f64).to_radians()),
            ((60.0_f64).to_radians(), (40.0_f64).to_radians()),
            ((-100.0_f64).to_radians(), (80.0_f64).to_radians()),
            ((150.0_f64).to_radians(), (150.0_f64).to_radians()),
        ];
        let stats = ramachandran_statistics(&pairs);
        let total = stats.helix_fraction
            + stats.beta_fraction
            + stats.left_helix_fraction
            + stats.allowed_fraction
            + stats.disallowed_fraction;
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Fractions must sum to 1: {total}"
        );
    }
    #[test]
    fn test_ramachandran_statistics_empty() {
        let stats = ramachandran_statistics(&[]);
        assert_eq!(stats.n_classified, 0);
        assert_eq!(stats.helix_fraction, 0.0);
    }
    #[test]
    fn test_ramachandran_statistics_all_helix() {
        let pairs: Vec<(f64, f64)> = vec![((-60.0_f64).to_radians(), (-45.0_f64).to_radians()); 5];
        let stats = ramachandran_statistics(&pairs);
        assert!(
            (stats.helix_fraction - 1.0).abs() < 1e-10,
            "All helix: {}",
            stats.helix_fraction
        );
    }
    #[test]
    fn test_ramachandran_statistics_counts_correct() {
        let pairs: Vec<(f64, f64)> = vec![
            ((-60.0_f64).to_radians(), (-45.0_f64).to_radians()),
            ((-60.0_f64).to_radians(), (-45.0_f64).to_radians()),
            ((-120.0_f64).to_radians(), (130.0_f64).to_radians()),
        ];
        let stats = ramachandran_statistics(&pairs);
        assert_eq!(stats.n_classified, 3);
        assert!(
            (stats.helix_fraction - 2.0 / 3.0).abs() < 1e-10,
            "2/3 helix: {}",
            stats.helix_fraction
        );
        assert!(
            (stats.beta_fraction - 1.0 / 3.0).abs() < 1e-10,
            "1/3 beta: {}",
            stats.beta_fraction
        );
    }
    #[test]
    fn test_end_to_end_linear_chain() {
        let chain = make_linear_chain(4);
        let d = end_to_end_distance(&chain);
        assert!(
            (d - 3.0 * 3.8).abs() < 1e-9,
            "End-to-end = 3*3.8 for linear chain: {d}"
        );
    }
    #[test]
    fn test_end_to_end_single_residue_zero() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [1.0, 2.0, 3.0]);
        assert_eq!(end_to_end_distance(&chain), 0.0);
    }
    #[test]
    fn test_end_to_end_two_residues() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Gly, [3.0, 4.0, 0.0]);
        let d = end_to_end_distance(&chain);
        assert!((d - 5.0).abs() < 1e-9, "End-to-end of right triangle: {d}");
    }
    #[test]
    fn test_rg_full_single_residue_zero() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [5.0, 3.0, 2.0]);
        assert!((radius_of_gyration_full(&chain)).abs() < 1e-9);
    }
    #[test]
    fn test_rg_full_positive() {
        let chain = make_linear_chain(5);
        let rg = radius_of_gyration_full(&chain);
        assert!(
            rg > 0.0,
            "Rg must be positive for multi-residue chain: {rg}"
        );
    }
    #[test]
    fn test_rg_full_scales_with_chain_length() {
        let chain5 = make_linear_chain(5);
        let chain10 = make_linear_chain(10);
        let rg5 = radius_of_gyration_full(&chain5);
        let rg10 = radius_of_gyration_full(&chain10);
        assert!(
            rg10 > rg5,
            "Longer chain should have larger Rg: {rg10} vs {rg5}"
        );
    }
    #[test]
    fn test_rg_2d_positive() {
        let chain = make_linear_chain(5);
        let rg = radius_of_gyration_2d(&chain);
        assert!(rg > 0.0, "Rg 2D must be positive: {rg}");
    }
    #[test]
    fn test_rg_2d_zero_for_single_residue() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [1.0, 2.0, 3.0]);
        assert!((radius_of_gyration_2d(&chain)).abs() < 1e-9);
    }
    #[test]
    fn test_contact_map_diagonal_zero() {
        let chain = make_linear_chain(4);
        let map = binary_contact_map(&chain, 5.0);
        for (i, row) in map.iter().enumerate() {
            assert_eq!(row[i], 0, "Diagonal must be 0 (self-contact excluded)");
        }
    }
    #[test]
    fn test_contact_map_symmetry() {
        let chain = make_linear_chain(4);
        let map = binary_contact_map(&chain, 5.0);
        for (i, row) in map.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_eq!(val, map[j][i], "Contact map must be symmetric");
            }
        }
    }
    #[test]
    fn test_contact_map_no_contacts_large_cutoff_small_chain() {
        let chain = make_linear_chain(4);
        let map = binary_contact_map(&chain, 3.0);
        let total: usize = map.iter().flatten().map(|&x| x as usize).sum();
        assert_eq!(total, 0, "No contacts for cutoff < spacing: {total}");
    }
    #[test]
    fn test_contact_map_count_consecutive() {
        let chain = make_linear_chain(4);
        let map = binary_contact_map(&chain, 4.0);
        let cnt = contact_map_count(&map);
        assert_eq!(cnt, 3, "Should find 3 consecutive contacts: {cnt}");
    }
    #[test]
    fn test_contact_density_between_zero_and_one() {
        let chain = make_linear_chain(6);
        let cd = contact_density(&chain, 4.5);
        assert!(
            (0.0..=1.0).contains(&cd),
            "Contact density must be in [0,1]: {cd}"
        );
    }
    #[test]
    fn test_contact_density_increases_with_cutoff() {
        let chain = make_linear_chain(6);
        let cd1 = contact_density(&chain, 4.0);
        let cd2 = contact_density(&chain, 8.0);
        assert!(
            cd2 >= cd1,
            "Contact density increases with cutoff: {cd1} vs {cd2}"
        );
    }
    #[test]
    fn test_mean_contact_distance_positive() {
        let chain = make_linear_chain(5);
        let d = mean_contact_distance(&chain, 4.5);
        assert!(d > 0.0, "Mean contact distance must be positive: {d}");
    }
    #[test]
    fn test_mean_contact_distance_no_contacts() {
        let chain = make_linear_chain(3);
        let d = mean_contact_distance(&chain, 1.0);
        assert_eq!(d, 0.0, "No contacts → zero mean distance: {d}");
    }
    #[test]
    fn test_rmsd_kabsch_identical_chains() {
        let chain = make_linear_chain(5);
        let r = rmsd_kabsch(&chain, &chain);
        assert!(r.abs() < 1e-10, "Kabsch RMSD of identical chains = 0: {r}");
    }
    #[test]
    fn test_rmsd_kabsch_translated_chain() {
        let chain_a = make_linear_chain(5);
        let mut chain_b = ProteinChain::new();
        for r in &chain_a.residues {
            chain_b.add_residue(
                r.aa,
                [r.ca_position[0] + 10.0, r.ca_position[1], r.ca_position[2]],
            );
        }
        let r = rmsd_kabsch(&chain_a, &chain_b);
        assert!(
            r.abs() < 1e-9,
            "Translation-only RMSD should be 0 after centering: {r}"
        );
    }
    #[test]
    fn test_rmsd_kabsch_rotated_about_z_by_90_degrees() {
        let chain_a = make_alpha_helix(12);
        // (x, y, z) -> (-y, x, z) is a +90° rotation about z; add a translation too.
        let chain_b = map_chain(&chain_a, |p| [-p[1] + 3.0, p[0] - 1.0, p[2] + 0.5]);
        // Without superposition the chains differ substantially ...
        assert!(rmsd(&chain_a, &chain_b) > 1.0);
        // ... but they are the same rigid body.
        let r = rmsd_kabsch(&chain_a, &chain_b);
        assert!(r < 1e-10, "Kabsch RMSD of a rigidly moved chain: {r}");
        // The optimal rotation is the inverse z-rotation, not the identity.
        let coords_a: Vec<[f64; 3]> = chain_a.residues.iter().map(|x| x.ca_position).collect();
        let coords_b: Vec<[f64; 3]> = chain_b.residues.iter().map(|x| x.ca_position).collect();
        let rot = kabsch_rotation(&coords_a, &coords_b);
        let inverse_z = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(max_abs_difference(&rot, &inverse_z) < 1e-10, "R = {rot:?}");
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(max_abs_difference(&rot, &identity) > 0.9);
    }
    #[test]
    fn test_rmsd_kabsch_arbitrary_rigid_motion_of_collinear_chain() {
        let chain_a = make_linear_chain(6);
        let r_true = rotation_matrix([2.0, -1.0, 0.7], 2.1);
        let chain_b = map_chain(&chain_a, |p| {
            let q = mat_vec(&r_true, p);
            [q[0] - 5.0, q[1] + 2.0, q[2] + 8.0]
        });
        let r = rmsd_kabsch(&chain_a, &chain_b);
        assert!(r < 1e-10, "collinear chain moved rigidly: {r}");
    }
    #[test]
    fn test_rmsd_kabsch_is_never_worse_than_translation_only() {
        let chain_a = make_alpha_helix(10);
        let r_true = rotation_matrix([0.3, 1.0, -0.4], 0.6);
        // Rigid motion plus a deterministic, non-rigid distortion.
        let mut chain_b = map_chain(&chain_a, |p| {
            let q = mat_vec(&r_true, p);
            [q[0] + 1.0, q[1] - 2.0, q[2] + 0.25]
        });
        for (k, res) in chain_b.residues.iter_mut().enumerate() {
            let w = 0.15 * ((k as f64) * 1.7).sin();
            res.ca_position[0] += w;
            res.ca_position[1] -= 0.5 * w;
            res.ca_position[2] += 0.25 * w;
        }
        let centred_only = {
            let mut a = chain_a.clone();
            let mut b = chain_b.clone();
            center_chain(&mut a);
            center_chain(&mut b);
            rmsd(&a, &b)
        };
        let kabsch = rmsd_kabsch(&chain_a, &chain_b);
        assert!(kabsch > 1e-3, "distortion must remain: {kabsch}");
        assert!(
            kabsch < 0.5 * centred_only,
            "superposition must reduce the RMSD: {kabsch} vs {centred_only}"
        );
    }
    #[test]
    fn test_rmsd_kabsch_mirror_image_does_not_superimpose() {
        // A helix and its mirror image differ by a reflection, which a proper rotation
        // cannot remove.
        let chain_a = make_alpha_helix(12);
        let chain_b = map_chain(&chain_a, |p| [-p[0], p[1], p[2]]);
        let r = rmsd_kabsch(&chain_a, &chain_b);
        assert!(r > 0.5, "mirror image should not superimpose, RMSD = {r}");
    }
    #[test]
    fn test_rmsd_kabsch_is_symmetric() {
        let chain_a = make_alpha_helix(9);
        let r_true = rotation_matrix([1.0, 1.0, 1.0], 1.0);
        let mut chain_b = map_chain(&chain_a, |p| mat_vec(&r_true, p));
        chain_b.residues[4].ca_position[0] += 0.8;
        let ab = rmsd_kabsch(&chain_a, &chain_b);
        let ba = rmsd_kabsch(&chain_b, &chain_a);
        assert!((ab - ba).abs() < 1e-10, "{ab} vs {ba}");
    }
    #[test]
    fn test_rmsd_kabsch_empty_chains() {
        let a = ProteinChain::new();
        let b = ProteinChain::new();
        assert_eq!(rmsd_kabsch(&a, &b), 0.0);
    }
    #[test]
    fn test_center_chain_centroid_at_origin() {
        let mut chain = make_linear_chain(5);
        center_chain(&mut chain);
        let n = chain.residues.len() as f64;
        let cx: f64 = chain.residues.iter().map(|r| r.ca_position[0]).sum::<f64>() / n;
        let cy: f64 = chain.residues.iter().map(|r| r.ca_position[1]).sum::<f64>() / n;
        let cz: f64 = chain.residues.iter().map(|r| r.ca_position[2]).sum::<f64>() / n;
        assert!(cx.abs() < 1e-10, "Centroid x after centering: {cx}");
        assert!(cy.abs() < 1e-10, "Centroid y after centering: {cy}");
        assert!(cz.abs() < 1e-10, "Centroid z after centering: {cz}");
    }
    #[test]
    fn test_kabsch_rotation_identity_for_identical() {
        let coords: Vec<[f64; 3]> = (0..5).map(|i| [i as f64, 0.0, 0.0]).collect();
        let r = kabsch_rotation(&coords, &coords);
        assert!((r[0][0] - 1.0).abs() < 1e-10);
        assert!((r[1][1] - 1.0).abs() < 1e-10);
        assert!((r[2][2] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_kabsch_rotation_determinant_one() {
        let coords_ref = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let coords_mob = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let r = kabsch_rotation(&coords_ref, &coords_mob);
        let det = determinant(&r);
        assert!(
            (det - 1.0).abs() < 1e-12,
            "Rotation det should be +1: {det}"
        );
        // R must be orthogonal: R Rᵀ = I.
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12, "R Rᵀ[{i}][{j}] = {dot}");
            }
        }
        // Two-point (collinear) sets: the centred mobile direction (1,0,-1)/√2 ...
        // ... must be taken onto the centred reference direction (1,-1,0)/√2.
        let mob_dir = [0.5, 0.0, -0.5];
        let ref_dir = [0.5, -0.5, 0.0];
        let rotated = mat_vec(&r, mob_dir);
        for d in 0..3 {
            assert!(
                (rotated[d] - ref_dir[d]).abs() < 1e-12,
                "rotated mobile direction {rotated:?} vs reference {ref_dir:?}"
            );
        }
    }
    #[test]
    fn test_kabsch_rotation_recovers_known_rotation() {
        let chain = make_alpha_helix(12);
        let reference: Vec<[f64; 3]> = chain.residues.iter().map(|r| r.ca_position).collect();
        let r_true = rotation_matrix([1.0, -2.0, 0.5], 0.9);
        let shift = [4.0, -7.5, 2.25];
        let mobile: Vec<[f64; 3]> = reference
            .iter()
            .map(|&p| {
                let q = mat_vec(&r_true, p);
                [q[0] + shift[0], q[1] + shift[1], q[2] + shift[2]]
            })
            .collect();
        let r = kabsch_rotation(&reference, &mobile);
        // The mobile set is R_true · reference + shift, so the optimal rotation that
        // takes it back onto the reference is R_trueᵀ.
        let r_inverse = [
            [r_true[0][0], r_true[1][0], r_true[2][0]],
            [r_true[0][1], r_true[1][1], r_true[2][1]],
            [r_true[0][2], r_true[1][2], r_true[2][2]],
        ];
        assert!(
            max_abs_difference(&r, &r_inverse) < 1e-10,
            "R = {r:?}, expected {r_inverse:?}"
        );
        assert!((determinant(&r) - 1.0).abs() < 1e-12);
    }
    #[test]
    fn test_kabsch_rotation_collinear_sets_in_general_orientation() {
        // Rank-one cross-covariance: the rotation about the common line is free but
        // R must still be a proper rotation that maps the line direction.
        let dir_a = [1.0, 2.0, -2.0];
        let dir_b = [-3.0, 0.5, 1.0];
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let (na, nb) = (norm(dir_a), norm(dir_b));
        let coords_ref: Vec<[f64; 3]> = (0..6)
            .map(|i| {
                let s = i as f64 * 1.3;
                [dir_a[0] / na * s, dir_a[1] / na * s, dir_a[2] / na * s]
            })
            .collect();
        let coords_mob: Vec<[f64; 3]> = (0..6)
            .map(|i| {
                let s = i as f64 * 1.3 + 5.0;
                [dir_b[0] / nb * s, dir_b[1] / nb * s, dir_b[2] / nb * s]
            })
            .collect();
        let r = kabsch_rotation(&coords_ref, &coords_mob);
        assert!(
            (determinant(&r) - 1.0).abs() < 1e-12,
            "det = {}",
            determinant(&r)
        );
        let mapped = mat_vec(&r, [dir_b[0] / nb, dir_b[1] / nb, dir_b[2] / nb]);
        for d in 0..3 {
            assert!(
                (mapped[d] - dir_a[d] / na).abs() < 1e-10,
                "R·b̂ = {mapped:?} should equal â"
            );
        }
    }
    #[test]
    fn test_kabsch_rotation_single_point_is_identity() {
        let r = kabsch_rotation(&[[1.0, 2.0, 3.0]], &[[-4.0, 0.5, 9.0]]);
        assert!(
            max_abs_difference(&r, &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) < 1e-15
        );
    }
    #[test]
    fn test_kabsch_rotation_uses_common_prefix_for_unequal_lengths() {
        let reference: Vec<[f64; 3]> = make_alpha_helix(8)
            .residues
            .iter()
            .map(|r| r.ca_position)
            .collect();
        let r_true = rotation_matrix([0.0, 0.0, 1.0], 1.2);
        let mut mobile: Vec<[f64; 3]> = reference.iter().map(|&p| mat_vec(&r_true, p)).collect();
        mobile.push([100.0, 100.0, 100.0]);
        let r = kabsch_rotation(&reference, &mobile);
        // The unpaired trailing point must be ignored: after centring on the common
        // prefix the first mobile point rotates onto the first reference point.
        let centroid = |pts: &[[f64; 3]]| {
            let n = pts.len() as f64;
            let mut c = [0.0; 3];
            for p in pts {
                for d in 0..3 {
                    c[d] += p[d] / n;
                }
            }
            c
        };
        let c_ref = centroid(&reference);
        let c_mob = centroid(&mobile[..8]);
        let mob0 = [
            mobile[0][0] - c_mob[0],
            mobile[0][1] - c_mob[1],
            mobile[0][2] - c_mob[2],
        ];
        let rotated0 = mat_vec(&r, mob0);
        for d in 0..3 {
            assert!(
                (rotated0[d] - (reference[0][d] - c_ref[d])).abs() < 1e-10,
                "first residue not superimposed: {rotated0:?}"
            );
        }
    }
    #[test]
    fn test_ss_assignment_empty_chain() {
        let chain = ProteinChain::new();
        let ss = assign_secondary_structure(&chain);
        assert!(ss.is_empty());
    }
    #[test]
    fn test_ss_assignment_short_chain() {
        let chain = make_linear_chain(3);
        let ss = assign_secondary_structure(&chain);
        assert_eq!(ss.len(), 3);
        for s in &ss {
            assert_eq!(*s, SecondaryStructure::Coil, "Short chain should be Coil");
        }
    }
    #[test]
    fn test_ss_helix_detection() {
        // Ideal α-helix (radius 2.3 Å, rise 1.5 Å, 100°/residue): residues 1..=n-4
        // are Helix, the termini are Coil.
        let n = 10;
        let ss = assign_secondary_structure(&make_alpha_helix(n));
        assert_eq!(ss.len(), n);
        let n_helix = ss
            .iter()
            .filter(|&&s| s == SecondaryStructure::Helix)
            .count();
        assert_eq!(n_helix, n - 4, "helix residues: {ss:?}");
        assert_eq!(ss[0], SecondaryStructure::Coil);
        assert!(
            ss[1..=n - 4]
                .iter()
                .all(|&s| s == SecondaryStructure::Helix)
        );
        assert!(ss[n - 3..].iter().all(|&s| s == SecondaryStructure::Coil));
    }
    #[test]
    fn test_ss_linear_chain_not_helix() {
        // Bonded Cα–Cα distances are 3.8 Å in a straight chain too; that must not
        // be read as a helix.
        let ss = assign_secondary_structure(&make_linear_chain(8));
        assert!(
            ss.iter().all(|&s| s != SecondaryStructure::Helix),
            "straight chain labelled helix: {ss:?}"
        );
        let n_strand = ss
            .iter()
            .filter(|&&s| s == SecondaryStructure::Strand)
            .count();
        assert_eq!(
            n_strand, 6,
            "interior residues of an extended chain: {ss:?}"
        );
    }
    #[test]
    fn test_rg_consistent_with_chain_rg() {
        let chain = make_linear_chain(6);
        let rg_full = radius_of_gyration_full(&chain);
        let rg_chain = chain.radius_of_gyration();
        assert!(
            (rg_full - rg_chain).abs() < 1e-9,
            "Rg full ({rg_full}) should equal chain Rg ({rg_chain})"
        );
    }
    #[test]
    fn test_end_to_end_consistent_with_chain_method() {
        let chain = make_linear_chain(5);
        let d_free = end_to_end_distance(&chain);
        let d_chain = chain.end_to_end_distance();
        assert!(
            (d_free - d_chain).abs() < 1e-9,
            "end_to_end_distance ({d_free}) should match chain method ({d_chain})"
        );
    }
    #[test]
    fn test_hbond_symmetry_in_distance() {
        let donor = [0.0, 0.0, 0.0];
        let acceptor = [3.0, 4.0, 0.0];
        let e1 = hydrogen_bond_energy(donor, acceptor, 1.0, 3.0, 10.0);
        let e2 = hydrogen_bond_energy(acceptor, donor, 1.0, 3.0, 10.0);
        assert!(
            (e1 - e2).abs() < 1e-10,
            "H-bond energy should be symmetric: {e1} vs {e2}"
        );
    }
    #[test]
    fn test_contact_map_count_symmetric_pairs() {
        let chain = make_linear_chain(3);
        let map = binary_contact_map(&chain, 5.0);
        let cnt = contact_map_count(&map);
        assert_eq!(cnt, 2, "Should find 2 contacts: {cnt}");
    }
    #[test]
    fn test_ramachandran_disallowed_region() {
        let pairs = vec![((150.0_f64).to_radians(), (150.0_f64).to_radians())];
        let stats = ramachandran_statistics(&pairs);
        assert!(
            (stats.disallowed_fraction - 1.0).abs() < 1e-10,
            "Should be 100% disallowed: {}",
            stats.disallowed_fraction
        );
    }
    #[test]
    fn test_mean_contact_distance_within_cutoff() {
        let chain = make_linear_chain(5);
        let cutoff = 4.5;
        let d = mean_contact_distance(&chain, cutoff);
        assert!(
            d <= cutoff,
            "Mean contact distance must be <= cutoff: {d} > {cutoff}"
        );
    }
    #[test]
    fn test_hbond_max_at_zero_distance() {
        let pos = [0.0, 0.0, 0.0];
        let e0 = hydrogen_bond_energy(pos, pos, 1.0, 2.0, 10.0);
        let far = [1.0, 0.0, 0.0];
        let e1 = hydrogen_bond_energy(pos, far, 1.0, 2.0, 10.0);
        assert!(e0 <= e1, "H-bond most attractive at r=0: {e0} vs {e1}");
    }
    #[test]
    fn test_rg_2d_less_equal_rg_full() {
        let chain = make_linear_chain(5);
        let rg2 = radius_of_gyration_2d(&chain);
        let rg3 = radius_of_gyration_full(&chain);
        assert!(
            (rg2 - rg3).abs() < 1e-9,
            "Rg_2D ({rg2}) == Rg_3D ({rg3}) for x-axis chain"
        );
    }
    #[test]
    fn test_ramachandran_left_helix_region_stats() {
        let pairs = vec![
            ((60.0_f64).to_radians(), (40.0_f64).to_radians()),
            ((60.0_f64).to_radians(), (40.0_f64).to_radians()),
        ];
        let stats = ramachandran_statistics(&pairs);
        assert!(
            (stats.left_helix_fraction - 1.0).abs() < 1e-10,
            "All left-helix: {}",
            stats.left_helix_fraction
        );
    }
    #[test]
    fn test_contact_density_zero_cutoff() {
        let chain = make_linear_chain(5);
        let cd = contact_density(&chain, 0.001);
        assert_eq!(cd, 0.0, "Zero cutoff → no contacts: {cd}");
    }
    #[test]
    fn test_kabsch_rotation_empty_returns_identity() {
        let r = kabsch_rotation(&[], &[]);
        assert!((r[0][0] - 1.0).abs() < 1e-10);
    }
}
