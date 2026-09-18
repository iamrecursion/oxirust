//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AminoAcid, ProteinChain, RamachandranRegion, RamachandranStats, Residue, SecondaryStructure,
};

pub(super) fn ca_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
/// Lennard-Jones pair potential: `4ε[(σ/r)¹² − (σ/r)⁶]`.
pub fn lennard_jones_pair(r: f64, epsilon: f64, sigma: f64) -> f64 {
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    let sr12 = sr6 * sr6;
    4.0 * epsilon * (sr12 - sr6)
}
/// Gō model energy over all Cα pairs:
/// `E = Σ ε[(d0/r)¹² − 2(d0/r)⁶]`
/// where `d0` is the native distance from `native_distances`.
/// `native_distances` should be in row-major upper-triangle order as returned
/// by pairs `(i, j)` with `i < j` scanning outer loop `i`, inner loop `j`.
pub fn go_model_energy(chain: &ProteinChain, native_distances: &[f64], epsilon: f64) -> f64 {
    let n = chain.residues.len();
    let mut energy = 0.0;
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d0 = native_distances[idx];
            idx += 1;
            let r = ca_distance(
                &chain.residues[i].ca_position,
                &chain.residues[j].ca_position,
            );
            if r > 1e-10 && d0 > 1e-10 {
                let x = d0 / r;
                let x6 = x.powi(6);
                let x12 = x6 * x6;
                energy += epsilon * (x12 - 2.0 * x6);
            }
        }
    }
    energy
}
/// Harmonic backbone bond energy between consecutive Cα atoms.
/// `E = Σ k(r − r0)²` where `r0 = 3.8 Å`.
pub fn backbone_bond_energy(chain: &ProteinChain, k: f64) -> f64 {
    pub(super) const R0: f64 = 3.8;
    let n = chain.residues.len();
    if n < 2 {
        return 0.0;
    }
    (0..n - 1)
        .map(|i| {
            let r = ca_distance(
                &chain.residues[i].ca_position,
                &chain.residues[i + 1].ca_position,
            );
            let dr = r - R0;
            k * dr * dr
        })
        .sum()
}
/// Contact energy: sum of Lennard-Jones terms `4ε[(σ/r)¹² − (σ/r)⁶]` over all Cα
/// pairs `i < j` closer than `cutoff` (Å).
///
/// `epsilon` is the well depth and `sigma` the LJ diameter (Å); each isolated pair
/// has its minimum `−ε` at `r = 2^(1/6) σ`.  Coincident Cα atoms
/// (`r < 1e-10` Å) are skipped because the 12-6 potential diverges there.
pub fn contact_energy(chain: &ProteinChain, cutoff: f64, epsilon: f64, sigma: f64) -> f64 {
    let n = chain.residues.len();
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = ca_distance(
                &chain.residues[i].ca_position,
                &chain.residues[j].ca_position,
            );
            if r < cutoff && r > 1e-10 {
                energy += lennard_jones_pair(r, epsilon, sigma);
            }
        }
    }
    energy
}
/// Return all Cα pairs `(i, j)` with `i < j` whose distance is within `cutoff`.
pub fn native_contacts(chain: &ProteinChain, cutoff: f64) -> Vec<(usize, usize)> {
    let n = chain.residues.len();
    let mut contacts = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let r = ca_distance(
                &chain.residues[i].ca_position,
                &chain.residues[j].ca_position,
            );
            if r <= cutoff {
                contacts.push((i, j));
            }
        }
    }
    contacts
}
/// Fraction of native contacts (Q value) formed in the current conformation.
pub fn fraction_native_contacts(
    current: &ProteinChain,
    native_contacts_list: &[(usize, usize)],
    cutoff: f64,
) -> f64 {
    if native_contacts_list.is_empty() {
        return 0.0;
    }
    let formed = native_contacts_list
        .iter()
        .filter(|&&(i, j)| {
            if i >= current.residues.len() || j >= current.residues.len() {
                return false;
            }
            let r = ca_distance(
                &current.residues[i].ca_position,
                &current.residues[j].ca_position,
            );
            r <= cutoff
        })
        .count();
    formed as f64 / native_contacts_list.len() as f64
}
/// Contact order: `CO = <|i − j|> / L` averaged over all native contacts.
pub fn contact_order(chain: &ProteinChain, cutoff: f64) -> f64 {
    let contacts = native_contacts(chain, cutoff);
    if contacts.is_empty() {
        return 0.0;
    }
    let l = chain.residues.len() as f64;
    let mean_sep: f64 =
        contacts.iter().map(|&(i, j)| (j - i) as f64).sum::<f64>() / contacts.len() as f64;
    mean_sep / l
}
/// Simplified accessible surface area estimate.
///
/// For each residue, the exposed fraction is `exp(-n_neighbors / scale)`
/// where `n_neighbors` is the count of other Cα atoms within `(vdw_r + probe_r)`.
/// The contribution of residue `i` is `4π(vdw_r + probe_r)² × exposed_fraction`.
pub fn accessible_surface_area_simplified(chain: &ProteinChain, probe_radius: f64) -> f64 {
    pub(super) const VDW_R: f64 = 1.7;
    pub(super) const SCALE: f64 = 3.0;
    let threshold = VDW_R + probe_radius;
    let sphere_area = 4.0 * std::f64::consts::PI * threshold * threshold;
    let n = chain.residues.len();
    let mut total = 0.0;
    for i in 0..n {
        let neighbors = (0..n)
            .filter(|&j| {
                j != i
                    && ca_distance(
                        &chain.residues[i].ca_position,
                        &chain.residues[j].ca_position,
                    ) <= threshold
            })
            .count();
        let exposed = (-(neighbors as f64) / SCALE).exp();
        total += sphere_area * exposed;
    }
    total
}
/// Root-mean-square deviation of Cα positions between two protein chains.
///
/// Both chains must have the same number of residues.
/// Returns 0.0 if chains are empty.
pub fn rmsd(chain_a: &ProteinChain, chain_b: &ProteinChain) -> f64 {
    let n = chain_a.residues.len().min(chain_b.residues.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = (0..n)
        .map(|i| {
            let a = &chain_a.residues[i].ca_position;
            let b = &chain_b.residues[i].ca_position;
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();
    (sum_sq / n as f64).sqrt()
}
/// Classify a (phi, psi) pair into a Ramachandran region (angles in radians).
pub fn ramachandran_region(phi_rad: f64, psi_rad: f64) -> RamachandranRegion {
    let phi = phi_rad.to_degrees();
    let psi = psi_rad.to_degrees();
    if (-90.0..=-30.0).contains(&phi) && (-70.0..=-10.0).contains(&psi) {
        return RamachandranRegion::AlphaHelix;
    }
    if (-160.0..=-60.0).contains(&phi) && (psi >= 100.0 || psi <= -150.0) {
        return RamachandranRegion::BetaSheet;
    }
    if (30.0..=90.0).contains(&phi) && (10.0..=80.0).contains(&psi) {
        return RamachandranRegion::LeftHelix;
    }
    if (-160.0..=0.0).contains(&phi) {
        return RamachandranRegion::AllowedRegion;
    }
    RamachandranRegion::Disallowed
}
/// Compute the all-residue Cα distance matrix.
///
/// Returns an n×n matrix where entry \[i\]\[j\] is the Cα-Cα distance in Angstroms.
pub fn residue_distance_matrix(chain: &ProteinChain) -> Vec<Vec<f64>> {
    let n = chain.residues.len();
    let mut mat = vec![vec![0.0f64; n]; n];
    for (i, row) in mat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i != j {
                *cell = ca_distance(
                    &chain.residues[i].ca_position,
                    &chain.residues[j].ca_position,
                );
            }
        }
    }
    mat
}
/// Virtual bond angle (radians) at Cα atom `b` between the pseudo-bonds `b→a` and
/// `b→c`; `None` when either pseudo-bond has (numerically) zero length.
fn virtual_bond_angle(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> Option<f64> {
    let u = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let v = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let un = norm3(u);
    let vn = norm3(v);
    if un < 1e-10 || vn < 1e-10 {
        return None;
    }
    Some((dot3(u, v) / (un * vn)).clamp(-1.0, 1.0).acos())
}
/// Assign secondary structure from the Cα trace alone.
///
/// Consecutive Cα–Cα distances are ≈ 3.8 Å in every conformation, so they carry no
/// secondary-structure information.  The assignment instead uses the virtual bond
/// angle θ at each interior residue `i` (between Cα(i−1), Cα(i) and Cα(i+1)) and the
/// Cα(i)–Cα(i+3) distance, the same criteria as
/// [`crate::protein_folding::assign_secondary_structure`]:
///
/// - **Helix**: 80° < θ < 105° and 4.5 Å < d(i, i+3) < 6.5 Å
///   (ideal α-helix: θ ≈ 91°, d(i, i+3) ≈ 5.0 Å).
/// - **Strand**: θ > 120° (extended chain).
/// - **Coil**: everything else.
///
/// The first and last residue have no defined angle and a residue without a partner
/// `i + 3` cannot start a helical turn; chains shorter than four residues are too
/// short to define any element and are returned as all [`SecondaryStructure::Coil`].
pub fn assign_secondary_structure(chain: &ProteinChain) -> Vec<SecondaryStructure> {
    let n = chain.residues.len();
    let mut ss = vec![SecondaryStructure::Coil; n];
    if n < 4 {
        return ss;
    }
    let ca = |i: usize| &chain.residues[i].ca_position;
    for (i, label) in ss.iter_mut().enumerate().take(n - 1).skip(1) {
        let Some(theta) = virtual_bond_angle(ca(i - 1), ca(i), ca(i + 1)) else {
            continue;
        };
        let theta_deg = theta.to_degrees();
        if i + 3 < n {
            let d_i3 = ca_distance(ca(i), ca(i + 3));
            if theta_deg > 80.0 && theta_deg < 105.0 && d_i3 > 4.5 && d_i3 < 6.5 {
                *label = SecondaryStructure::Helix;
                continue;
            }
        }
        if theta_deg > 120.0 {
            *label = SecondaryStructure::Strand;
        }
    }
    ss
}
/// Simple statistical potential energy between two residues.
///
/// Uses a Lennard-Jones-like potential with well depth derived from the
/// hydrophobicity product of the two residues.
pub fn residue_pair_energy(res_i: &Residue, res_j: &Residue) -> f64 {
    let r = ca_distance(&res_i.ca_position, &res_j.ca_position);
    if r < 1e-10 {
        return 0.0;
    }
    let hyd_i = res_i.aa.hydrophobicity();
    let hyd_j = res_j.aa.hydrophobicity();
    let epsilon = (hyd_i * hyd_j).max(0.1);
    let sigma = 3.8;
    let sr = sigma / r;
    let sr6 = sr.powi(6);
    4.0 * epsilon * (sr6 * sr6 - sr6)
}
/// Total protein energy from pairwise residue interactions + backbone bonds.
pub fn protein_total_energy(chain: &ProteinChain, bond_k: f64) -> f64 {
    let n = chain.residues.len();
    let mut e = backbone_bond_energy(chain, bond_k);
    for i in 0..n {
        for j in (i + 2)..n {
            e += residue_pair_energy(&chain.residues[i], &chain.residues[j]);
        }
    }
    e
}
/// Simple sequence alignment score using Needleman-Wunsch algorithm.
///
/// Uses a match score of +1, mismatch of -1, and gap penalty of -2.
/// Returns the alignment score.
pub fn needleman_wunsch(seq_a: &[AminoAcid], seq_b: &[AminoAcid]) -> i32 {
    let n = seq_a.len();
    let m = seq_b.len();
    pub(super) const GAP: i32 = -2;
    pub(super) const MATCH: i32 = 1;
    pub(super) const MISMATCH: i32 = -1;
    let mut dp = vec![vec![0i32; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = -(i as i32) * 2;
    }
    for (j, v) in dp[0].iter_mut().enumerate() {
        *v = -(j as i32) * 2;
    }
    for i in 1..=n {
        for j in 1..=m {
            let score = if seq_a[i - 1] == seq_b[j - 1] {
                MATCH
            } else {
                MISMATCH
            };
            dp[i][j] = (dp[i - 1][j - 1] + score)
                .max(dp[i - 1][j] + GAP)
                .max(dp[i][j - 1] + GAP);
        }
    }
    dp[n][m]
}
/// Sequence identity fraction between two chains (using trivial pairwise alignment).
///
/// Returns the fraction of identical positions in the shorter sequence.
pub fn sequence_identity(chain_a: &ProteinChain, chain_b: &ProteinChain) -> f64 {
    let n = chain_a.residues.len().min(chain_b.residues.len());
    if n == 0 {
        return 0.0;
    }
    let identical = (0..n)
        .filter(|&i| chain_a.residues[i].aa == chain_b.residues[i].aa)
        .count();
    identical as f64 / n as f64
}
pub(super) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(super) fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
/// Simplified hydrogen bond energy using DSSP-like electrostatic model.
///
/// DSSP: E_hb = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332 kcal/mol
///
/// Here we use a simplified version based on donor–acceptor distance only.
/// E = -epsilon * exp(-r² / (2 * sigma²)) for r < r_cutoff
///
/// Returns the H-bond energy in arbitrary units (negative = stabilising).
pub fn hydrogen_bond_energy(
    donor_pos: [f64; 3],
    acceptor_pos: [f64; 3],
    epsilon: f64,
    sigma: f64,
    cutoff: f64,
) -> f64 {
    let dx = acceptor_pos[0] - donor_pos[0];
    let dy = acceptor_pos[1] - donor_pos[1];
    let dz = acceptor_pos[2] - donor_pos[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 > cutoff * cutoff {
        return 0.0;
    }
    -epsilon * (-r2 / (2.0 * sigma * sigma)).exp()
}
/// Total hydrogen bond energy in a chain (backbone N–O pairs).
///
/// Considers all residue pairs (i, j) with |i - j| >= 3.
/// Uses Cα positions as a proxy for N and O positions.
pub fn total_hbond_energy(chain: &ProteinChain, epsilon: f64, sigma: f64, cutoff: f64) -> f64 {
    let n = chain.residues.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j || (j as i64 - i as i64).abs() < 3 {
                continue;
            }
            e += hydrogen_bond_energy(
                chain.residues[i].ca_position,
                chain.residues[j].ca_position,
                epsilon,
                sigma,
                cutoff,
            );
        }
    }
    e
}
/// Compute Ramachandran statistics from a list of (phi, psi) pairs.
pub fn ramachandran_statistics(phi_psi_pairs: &[(f64, f64)]) -> RamachandranStats {
    let n = phi_psi_pairs.len();
    if n == 0 {
        return RamachandranStats {
            helix_fraction: 0.0,
            beta_fraction: 0.0,
            left_helix_fraction: 0.0,
            allowed_fraction: 0.0,
            disallowed_fraction: 0.0,
            n_classified: 0,
        };
    }
    let mut counts = [0usize; 5];
    for &(phi, psi) in phi_psi_pairs {
        let region = ramachandran_region(phi, psi);
        match region {
            RamachandranRegion::AlphaHelix => counts[0] += 1,
            RamachandranRegion::BetaSheet => counts[1] += 1,
            RamachandranRegion::LeftHelix => counts[2] += 1,
            RamachandranRegion::AllowedRegion => counts[3] += 1,
            RamachandranRegion::Disallowed => counts[4] += 1,
        }
    }
    let nf = n as f64;
    RamachandranStats {
        helix_fraction: counts[0] as f64 / nf,
        beta_fraction: counts[1] as f64 / nf,
        left_helix_fraction: counts[2] as f64 / nf,
        allowed_fraction: counts[3] as f64 / nf,
        disallowed_fraction: counts[4] as f64 / nf,
        n_classified: n,
    }
}
/// Compute the end-to-end distance for a chain.
///
/// Returns the distance between the first and last Cα atoms.
pub fn end_to_end_distance(chain: &ProteinChain) -> f64 {
    let n = chain.residues.len();
    if n < 2 {
        return 0.0;
    }
    ca_distance(
        &chain.residues[0].ca_position,
        &chain.residues[n - 1].ca_position,
    )
}
/// Compute the mass-weighted radius of gyration.
///
/// R_g = sqrt(Σ m_i |r_i - r_cm|² / Σ m_i)
/// Here we use unit masses for Cα atoms.
pub fn radius_of_gyration_full(chain: &ProteinChain) -> f64 {
    let n = chain.residues.len();
    if n == 0 {
        return 0.0;
    }
    let mut cm = [0.0; 3];
    for r in &chain.residues {
        cm[0] += r.ca_position[0];
        cm[1] += r.ca_position[1];
        cm[2] += r.ca_position[2];
    }
    let nf = n as f64;
    cm[0] /= nf;
    cm[1] /= nf;
    cm[2] /= nf;
    let sum_sq: f64 = chain
        .residues
        .iter()
        .map(|r| {
            let dx = r.ca_position[0] - cm[0];
            let dy = r.ca_position[1] - cm[1];
            let dz = r.ca_position[2] - cm[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();
    (sum_sq / nf).sqrt()
}
/// Radius of gyration in 2D projection (xy-plane).
pub fn radius_of_gyration_2d(chain: &ProteinChain) -> f64 {
    let n = chain.residues.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let cx: f64 = chain.residues.iter().map(|r| r.ca_position[0]).sum::<f64>() / nf;
    let cy: f64 = chain.residues.iter().map(|r| r.ca_position[1]).sum::<f64>() / nf;
    let sum_sq: f64 = chain
        .residues
        .iter()
        .map(|r| {
            let dx = r.ca_position[0] - cx;
            let dy = r.ca_position[1] - cy;
            dx * dx + dy * dy
        })
        .sum();
    (sum_sq / nf).sqrt()
}
/// Binary contact map: entry (i,j) = 1 if Cα-Cα distance < cutoff, else 0.
pub fn binary_contact_map(chain: &ProteinChain, cutoff: f64) -> Vec<Vec<u8>> {
    let n = chain.residues.len();
    let mut map = vec![vec![0u8; n]; n];
    for (i, row) in map.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i != j {
                let r = ca_distance(
                    &chain.residues[i].ca_position,
                    &chain.residues[j].ca_position,
                );
                if r <= cutoff {
                    *cell = 1;
                }
            }
        }
    }
    map
}
/// Number of contacts in the contact map.
pub fn contact_map_count(map: &[Vec<u8>]) -> usize {
    let mut cnt = 0;
    for (i, row) in map.iter().enumerate() {
        for v in row.iter().skip(i + 1) {
            if *v == 1 {
                cnt += 1;
            }
        }
    }
    cnt
}
/// Contact density: contacts per residue pair.
pub fn contact_density(chain: &ProteinChain, cutoff: f64) -> f64 {
    let n = chain.residues.len();
    if n < 2 {
        return 0.0;
    }
    let map = binary_contact_map(chain, cutoff);
    let cnt = contact_map_count(&map) as f64;
    let n_pairs = (n * (n - 1) / 2) as f64;
    cnt / n_pairs
}
/// Average contact distance for pairs in contact.
pub fn mean_contact_distance(chain: &ProteinChain, cutoff: f64) -> f64 {
    let n = chain.residues.len();
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = ca_distance(
                &chain.residues[i].ca_position,
                &chain.residues[j].ca_position,
            );
            if r <= cutoff {
                sum += r;
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_three_letter_known() {
        assert_eq!(AminoAcid::from_three_letter("ALA"), Some(AminoAcid::Ala));
        assert_eq!(AminoAcid::from_three_letter("TRP"), Some(AminoAcid::Trp));
        assert_eq!(AminoAcid::from_three_letter("ARG"), Some(AminoAcid::Arg));
    }
    #[test]
    fn test_from_three_letter_case_insensitive() {
        assert_eq!(AminoAcid::from_three_letter("ala"), Some(AminoAcid::Ala));
        assert_eq!(AminoAcid::from_three_letter("Lys"), Some(AminoAcid::Lys));
    }
    #[test]
    fn test_from_three_letter_unknown() {
        assert_eq!(AminoAcid::from_three_letter("XYZ"), None);
        assert_eq!(AminoAcid::from_three_letter(""), None);
    }
    #[test]
    fn test_from_one_letter_known() {
        assert_eq!(AminoAcid::from_one_letter('A'), Some(AminoAcid::Ala));
        assert_eq!(AminoAcid::from_one_letter('G'), Some(AminoAcid::Gly));
        assert_eq!(AminoAcid::from_one_letter('W'), Some(AminoAcid::Trp));
        assert_eq!(AminoAcid::from_one_letter('R'), Some(AminoAcid::Arg));
    }
    #[test]
    fn test_from_one_letter_lowercase() {
        assert_eq!(AminoAcid::from_one_letter('a'), Some(AminoAcid::Ala));
        assert_eq!(AminoAcid::from_one_letter('k'), Some(AminoAcid::Lys));
    }
    #[test]
    fn test_from_one_letter_unknown() {
        assert_eq!(AminoAcid::from_one_letter('B'), None);
        assert_eq!(AminoAcid::from_one_letter('Z'), None);
    }
    #[test]
    fn test_molecular_weight_range() {
        for aa in [
            AminoAcid::Ala,
            AminoAcid::Gly,
            AminoAcid::Val,
            AminoAcid::Trp,
            AminoAcid::Arg,
        ] {
            let mw = aa.molecular_weight();
            assert!(mw > 50.0 && mw < 300.0, "MW out of range for {:?}", aa);
        }
    }
    #[test]
    fn test_hydrophobicity_values() {
        assert!((AminoAcid::Ile.hydrophobicity() - 4.5).abs() < 1e-9);
        assert!((AminoAcid::Arg.hydrophobicity() - (-4.5)).abs() < 1e-9);
    }
    #[test]
    fn test_is_hydrophobic() {
        assert!(AminoAcid::Ala.is_hydrophobic());
        assert!(AminoAcid::Val.is_hydrophobic());
        assert!(AminoAcid::Ile.is_hydrophobic());
        assert!(!AminoAcid::Arg.is_hydrophobic());
        assert!(!AminoAcid::Asp.is_hydrophobic());
        assert!(!AminoAcid::Gly.is_hydrophobic());
    }
    #[test]
    fn test_charge_at_ph7() {
        assert!((AminoAcid::Asp.charge_at_ph7() - (-1.0)).abs() < 1e-9);
        assert!((AminoAcid::Glu.charge_at_ph7() - (-1.0)).abs() < 1e-9);
        assert!((AminoAcid::Lys.charge_at_ph7() - 1.0).abs() < 1e-9);
        assert!((AminoAcid::Arg.charge_at_ph7() - 1.0).abs() < 1e-9);
        assert!((AminoAcid::Ala.charge_at_ph7() - 0.0).abs() < 1e-9);
    }
    fn make_linear_chain(n: usize) -> ProteinChain {
        let mut chain = ProteinChain::new();
        for i in 0..n {
            chain.add_residue(AminoAcid::Ala, [i as f64 * 3.8, 0.0, 0.0]);
        }
        chain
    }
    #[test]
    fn test_chain_length() {
        let chain = make_linear_chain(5);
        assert_eq!(chain.length(), 5);
    }
    #[test]
    fn test_chain_sequence() {
        let chain = make_linear_chain(3);
        let seq = chain.sequence();
        assert_eq!(seq, vec![AminoAcid::Ala; 3]);
    }
    #[test]
    fn test_chain_molecular_weight() {
        let chain = make_linear_chain(2);
        let expected = 2.0 * 89.094;
        assert!((chain.molecular_weight() - expected).abs() < 1e-3);
    }
    #[test]
    fn test_chain_end_to_end() {
        let chain = make_linear_chain(4);
        assert!((chain.end_to_end_distance() - 3.0 * 3.8).abs() < 1e-9);
    }
    #[test]
    fn test_chain_rg_linear() {
        let chain = make_linear_chain(5);
        let rg = chain.radius_of_gyration();
        assert!(rg > 0.0);
    }
    #[test]
    fn test_chain_rg_single_residue() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Gly, [1.0, 2.0, 3.0]);
        assert!((chain.radius_of_gyration() - 0.0).abs() < 1e-9);
    }
    #[test]
    fn test_chain_hydrophobic_fraction() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Asp, [3.8, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Val, [7.6, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Arg, [11.4, 0.0, 0.0]);
        assert!((chain.hydrophobic_fraction() - 0.5).abs() < 1e-9);
    }
    #[test]
    fn test_lennard_jones_minimum() {
        let sigma = 3.4;
        let epsilon = 1.0;
        let r_min = 2.0_f64.powf(1.0 / 6.0) * sigma;
        let v = lennard_jones_pair(r_min, epsilon, sigma);
        assert!((v - (-epsilon)).abs() < 1e-9);
    }
    #[test]
    fn test_lennard_jones_at_sigma() {
        let sigma = 3.0;
        let v = lennard_jones_pair(sigma, 1.0, sigma);
        assert!(v.abs() < 1e-9);
    }
    #[test]
    fn test_backbone_bond_energy_zero_at_r0() {
        let chain = make_linear_chain(3);
        let e = backbone_bond_energy(&chain, 100.0);
        assert!(e.abs() < 1e-9);
    }
    #[test]
    fn test_backbone_bond_energy_positive() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Ala, [5.0, 0.0, 0.0]);
        let e = backbone_bond_energy(&chain, 1.0);
        let expected = (5.0 - 3.8_f64).powi(2);
        assert!((e - expected).abs() < 1e-9);
    }
    #[test]
    fn test_go_model_energy_native_state() {
        let chain = make_linear_chain(3);
        let n = chain.length();
        let mut native_dist = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let r = ca_distance(
                    &chain.residues[i].ca_position,
                    &chain.residues[j].ca_position,
                );
                native_dist.push(r);
            }
        }
        let epsilon = 1.0;
        let e = go_model_energy(&chain, &native_dist, epsilon);
        let n_pairs = n * (n - 1) / 2;
        assert!((e - (-(n_pairs as f64) * epsilon)).abs() < 1e-9);
    }
    #[test]
    fn test_native_contacts_linear() {
        let chain = make_linear_chain(4);
        let contacts = native_contacts(&chain, 4.0);
        assert_eq!(contacts.len(), 3);
    }
    #[test]
    fn test_fraction_native_contacts_all_formed() {
        let chain = make_linear_chain(4);
        let contacts = native_contacts(&chain, 4.0);
        let q = fraction_native_contacts(&chain, &contacts, 4.0);
        assert!((q - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_fraction_native_contacts_none_formed() {
        let native = make_linear_chain(4);
        let contacts = native_contacts(&native, 4.0);
        let mut current = ProteinChain::new();
        for i in 0..4 {
            current.add_residue(AminoAcid::Ala, [i as f64 * 100.0, 0.0, 0.0]);
        }
        let q = fraction_native_contacts(&current, &contacts, 4.0);
        assert!(q.abs() < 1e-9);
    }
    #[test]
    fn test_contact_order() {
        let chain = make_linear_chain(5);
        let co = contact_order(&chain, 4.0);
        assert!(co > 0.0 && co < 1.0);
    }
    #[test]
    fn test_asa_positive() {
        let chain = make_linear_chain(5);
        let asa = accessible_surface_area_simplified(&chain, 1.4);
        assert!(asa > 0.0);
    }
    #[test]
    fn test_asa_single_residue() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        let probe = 1.4;
        let vdw_r = 1.7;
        let expected = 4.0 * std::f64::consts::PI * (vdw_r + probe) * (vdw_r + probe);
        let asa = accessible_surface_area_simplified(&chain, probe);
        assert!((asa - expected).abs() < 1e-6);
    }
    #[test]
    fn test_rmsd_identical_chains() {
        let chain = make_linear_chain(5);
        let r = rmsd(&chain, &chain);
        assert!(
            r.abs() < 1e-12,
            "RMSD of identical chains should be 0, got {r}"
        );
    }
    #[test]
    fn test_rmsd_displaced_chain() {
        let chain_a = make_linear_chain(4);
        let mut chain_b = ProteinChain::new();
        for r in &chain_a.residues {
            chain_b.add_residue(
                r.aa,
                [r.ca_position[0] + 1.0, r.ca_position[1], r.ca_position[2]],
            );
        }
        let r = rmsd(&chain_a, &chain_b);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "RMSD should be 1.0 after uniform shift of 1 Å, got {r}"
        );
    }
    #[test]
    fn test_rmsd_empty_chains() {
        let a = ProteinChain::new();
        let b = ProteinChain::new();
        assert_eq!(rmsd(&a, &b), 0.0);
    }
    #[test]
    fn test_ramachandran_alpha_helix_region() {
        let phi = (-60.0_f64).to_radians();
        let psi = (-45.0_f64).to_radians();
        let region = ramachandran_region(phi, psi);
        assert_eq!(
            region,
            RamachandranRegion::AlphaHelix,
            "(-60,-45) should be alpha-helix"
        );
    }
    #[test]
    fn test_ramachandran_beta_sheet_region() {
        let phi = (-120.0_f64).to_radians();
        let psi = (130.0_f64).to_radians();
        let region = ramachandran_region(phi, psi);
        assert_eq!(
            region,
            RamachandranRegion::BetaSheet,
            "(-120,130) should be beta-sheet"
        );
    }
    #[test]
    fn test_ramachandran_left_helix() {
        let phi = (60.0_f64).to_radians();
        let psi = (40.0_f64).to_radians();
        let region = ramachandran_region(phi, psi);
        assert_eq!(
            region,
            RamachandranRegion::LeftHelix,
            "(60,40) should be left-helix"
        );
    }
    #[test]
    fn test_ss_assignment_length_matches() {
        let chain = make_linear_chain(10);
        let ss = assign_secondary_structure(&chain);
        assert_eq!(
            ss.len(),
            chain.length(),
            "SS length should match chain length"
        );
    }
    /// Ideal α-helix Cα trace: radius 2.3 Å, rise 1.5 Å, 100° per residue
    /// (consecutive Cα–Cα distance ≈ 3.83 Å, virtual angle ≈ 91°).
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
    /// Planar zig-zag Cα trace with 3.8 Å bonds and the given interior (virtual
    /// bond) angle in degrees at every interior residue.
    fn make_zigzag(n: usize, interior_angle_deg: f64) -> ProteinChain {
        let tilt = ((180.0 - interior_angle_deg) / 2.0).to_radians();
        let mut chain = ProteinChain::new();
        let mut pos = [0.0_f64, 0.0, 0.0];
        for k in 0..n {
            chain.add_residue(AminoAcid::Ala, pos);
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            pos[0] += 3.8 * tilt.cos();
            pos[1] += 3.8 * sign * tilt.sin();
        }
        chain
    }
    fn count_ss(ss: &[SecondaryStructure], kind: SecondaryStructure) -> usize {
        ss.iter().filter(|&&s| s == kind).count()
    }
    #[test]
    fn test_alpha_helix_fixture_geometry() {
        let chain = make_alpha_helix(10);
        for i in 0..9 {
            let d = ca_distance(
                &chain.residues[i].ca_position,
                &chain.residues[i + 1].ca_position,
            );
            assert!((d - 3.83).abs() < 0.02, "bonded Cα distance {d}");
        }
        let theta = virtual_bond_angle(
            &chain.residues[3].ca_position,
            &chain.residues[4].ca_position,
            &chain.residues[5].ca_position,
        )
        .unwrap()
        .to_degrees();
        assert!((theta - 91.0).abs() < 2.0, "virtual angle {theta}");
        let d_i3 = ca_distance(
            &chain.residues[2].ca_position,
            &chain.residues[5].ca_position,
        );
        assert!((d_i3 - 5.05).abs() < 0.1, "d(i,i+3) = {d_i3}");
    }
    #[test]
    fn test_ss_linear_chain_not_helix() {
        // A straight Cα trace (θ = 180°) is fully extended: it must never be
        // labelled Helix, although every bonded Cα–Cα distance is 3.8 Å.
        let chain = make_linear_chain(8);
        let ss = assign_secondary_structure(&chain);
        assert_eq!(ss.len(), 8);
        assert_eq!(count_ss(&ss, SecondaryStructure::Helix), 0);
        assert_eq!(ss[0], SecondaryStructure::Coil);
        assert_eq!(ss[7], SecondaryStructure::Coil);
        assert_eq!(count_ss(&ss, SecondaryStructure::Strand), 6);
    }
    #[test]
    fn test_ss_ideal_alpha_helix_is_helix() {
        let n = 12;
        let ss = assign_secondary_structure(&make_alpha_helix(n));
        assert_eq!(ss.len(), n);
        // Residues 1..=n-4 have both neighbours and a partner i+3.
        assert_eq!(ss[0], SecondaryStructure::Coil);
        for (i, s) in ss.iter().enumerate().take(n - 3).skip(1) {
            assert_eq!(*s, SecondaryStructure::Helix, "residue {i}");
        }
        for (i, s) in ss.iter().enumerate().skip(n - 3) {
            assert_eq!(*s, SecondaryStructure::Coil, "residue {i}");
        }
        assert_eq!(count_ss(&ss, SecondaryStructure::Strand), 0);
    }
    #[test]
    fn test_ss_beta_zigzag_is_strand() {
        let n = 9;
        let ss = assign_secondary_structure(&make_zigzag(n, 125.0));
        assert_eq!(ss[0], SecondaryStructure::Coil);
        assert_eq!(ss[n - 1], SecondaryStructure::Coil);
        for (i, s) in ss.iter().enumerate().take(n - 1).skip(1) {
            assert_eq!(*s, SecondaryStructure::Strand, "residue {i}");
        }
        assert_eq!(count_ss(&ss, SecondaryStructure::Helix), 0);
    }
    #[test]
    fn test_ss_intermediate_angle_is_coil() {
        // θ = 110° lies between the helix (< 105°) and strand (> 120°) windows.
        let ss = assign_secondary_structure(&make_zigzag(9, 110.0));
        assert!(ss.iter().all(|&s| s == SecondaryStructure::Coil), "{ss:?}");
    }
    #[test]
    fn test_ss_helix_needs_compact_i_to_i3_distance() {
        // θ = 91° alone is not enough: in a planar zig-zag the Cα(i)–Cα(i+3)
        // distance is ≈ 8.6 Å, far outside the helical window (4.5, 6.5) Å.
        let ss = assign_secondary_structure(&make_zigzag(10, 91.0));
        assert!(ss.iter().all(|&s| s == SecondaryStructure::Coil), "{ss:?}");
    }
    #[test]
    fn test_ss_coincident_calpha_does_not_panic() {
        let mut chain = make_alpha_helix(8);
        chain.residues[3].ca_position = chain.residues[4].ca_position;
        let ss = assign_secondary_structure(&chain);
        assert_eq!(ss.len(), 8);
        assert_eq!(ss[3], SecondaryStructure::Coil);
        assert_eq!(ss[4], SecondaryStructure::Coil);
    }
    #[test]
    fn test_contact_energy_pair_at_lj_minimum() {
        let (epsilon, sigma) = (0.7, 3.4);
        let r_min = 2.0_f64.powf(1.0 / 6.0) * sigma;
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Gly, [r_min, 0.0, 0.0]);
        let e = contact_energy(&chain, 2.0 * r_min, epsilon, sigma);
        assert!(
            (e + epsilon).abs() < 1e-12,
            "pair at the LJ minimum must give -epsilon, got {e}"
        );
    }
    #[test]
    fn test_contact_energy_zero_beyond_cutoff() {
        let (epsilon, sigma) = (0.7, 3.4);
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        chain.add_residue(AminoAcid::Gly, [6.0, 0.0, 0.0]);
        assert_eq!(contact_energy(&chain, 5.0, epsilon, sigma), 0.0);
        assert!(contact_energy(&chain, 7.0, epsilon, sigma) < 0.0);
    }
    #[test]
    fn test_contact_energy_sums_pairs_and_scales_with_epsilon() {
        let sigma = 3.5;
        let chain = make_linear_chain(4);
        let cutoff = 8.0;
        let e1 = contact_energy(&chain, cutoff, 1.0, sigma);
        let e2 = contact_energy(&chain, cutoff, 2.5, sigma);
        // Pairs at 3.8 Å (x3) and 7.6 Å (x2); 11.4 Å is beyond the cutoff.
        let expected =
            3.0 * lennard_jones_pair(3.8, 1.0, sigma) + 2.0 * lennard_jones_pair(7.6, 1.0, sigma);
        assert!((e1 - expected).abs() < 1e-12, "{e1} vs {expected}");
        assert!((e2 - 2.5 * e1).abs() < 1e-12, "{e2} vs {}", 2.5 * e1);
        assert!(e1.is_finite());
    }
    #[test]
    fn test_contact_energy_skips_coincident_atoms() {
        let mut chain = ProteinChain::new();
        chain.add_residue(AminoAcid::Ala, [1.0, 1.0, 1.0]);
        chain.add_residue(AminoAcid::Ala, [1.0, 1.0, 1.0]);
        let e = contact_energy(&chain, 5.0, 1.0, 3.4);
        assert_eq!(e, 0.0);
    }
    #[test]
    fn test_distance_matrix_diagonal_zero() {
        let chain = make_linear_chain(4);
        let mat = residue_distance_matrix(&chain);
        for (i, row) in mat.iter().enumerate() {
            assert!(row[i].abs() < 1e-12, "Diagonal should be 0, got {}", row[i]);
        }
    }
    #[test]
    fn test_distance_matrix_symmetric() {
        let chain = make_linear_chain(4);
        let mat = residue_distance_matrix(&chain);
        for (i, row) in mat.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!((val - mat[j][i]).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn test_needleman_wunsch_identical() {
        let seq = vec![AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Val];
        let score = needleman_wunsch(&seq, &seq);
        assert_eq!(score, 3, "identical sequences of length 3 should score 3");
    }
    #[test]
    fn test_needleman_wunsch_different() {
        let seq_a = vec![AminoAcid::Ala, AminoAcid::Gly];
        let seq_b = vec![AminoAcid::Lys, AminoAcid::Arg];
        let score = needleman_wunsch(&seq_a, &seq_b);
        assert_eq!(score, -2, "two mismatches should score -2, got {score}");
    }
    #[test]
    fn test_sequence_identity_identical() {
        let chain = make_linear_chain(5);
        let id = sequence_identity(&chain, &chain);
        assert!(
            (id - 1.0).abs() < 1e-12,
            "identity with itself should be 1.0, got {id}"
        );
    }
    #[test]
    fn test_sequence_identity_different() {
        let mut chain_a = ProteinChain::new();
        let mut chain_b = ProteinChain::new();
        for i in 0..4 {
            chain_a.add_residue(AminoAcid::Ala, [i as f64 * 3.8, 0.0, 0.0]);
        }
        chain_b.add_residue(AminoAcid::Ala, [0.0, 0.0, 0.0]);
        chain_b.add_residue(AminoAcid::Gly, [3.8, 0.0, 0.0]);
        chain_b.add_residue(AminoAcid::Ala, [7.6, 0.0, 0.0]);
        chain_b.add_residue(AminoAcid::Gly, [11.4, 0.0, 0.0]);
        let id = sequence_identity(&chain_a, &chain_b);
        assert!(
            (id - 0.5).abs() < 1e-12,
            "2 of 4 match → identity=0.5, got {id}"
        );
    }
    #[test]
    fn test_residue_pair_energy_finite() {
        let chain = make_linear_chain(3);
        let e = residue_pair_energy(&chain.residues[0], &chain.residues[2]);
        assert!(
            e.is_finite(),
            "residue pair energy should be finite, got {e}"
        );
    }
    #[test]
    fn test_protein_total_energy_finite() {
        let chain = make_linear_chain(5);
        let e = protein_total_energy(&chain, 100.0);
        assert!(e.is_finite(), "total energy should be finite, got {e}");
    }
}
