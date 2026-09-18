// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! QM/MM interface for hybrid quantum mechanics / molecular mechanics calculations.
//!
//! Provides structures and algorithms for partitioning a molecular system into
//! a quantum-mechanically treated core region and a classically treated MM region,
//! connected via link atoms at the boundary.
//!
//! ## Units
//!
//! Like [`crate::qm_mm`], every energy in this module is in kcal/mol, every
//! force or gradient in kcal/(mol·Å), distances are in Å and charges in units of e.
//!
//! ## Energy model
//!
//! [`QmmmSystem::total_energy`] adds the converged energy of the QM region (SCC-DFTB
//! by default, hydrogen-capped at the link atoms, computed by the real QM engines of
//! [`crate::qm_mm`]) to the classical QM–MM Coulomb and Lennard-Jones interactions
//! (mechanical embedding with the fixed [`QmAtom::charge`] values).  For a QM region
//! that is polarised by the MM point charges use [`crate::qm_mm::QmMmSystem::compute`].

use crate::qm_mm::{QmError, QmMethod, QmRegion};

/// Classification of atoms in a QM/MM calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionType {
    /// Atom treated quantum mechanically (semi-empirical TB / DFT-B).
    QM,
    /// Atom treated with a classical force field.
    MM,
    /// Link atom or transition-region atom at the QM/MM boundary.
    Buffer,
}

/// An atom residing in the QM region.
#[derive(Debug, Clone)]
pub struct QmAtom {
    /// Global index in the full system.
    pub index: usize,
    /// Cartesian position in Ångström.
    pub position: [f64; 3],
    /// Atomic number (1 = H, 6 = C, …).
    pub atomic_number: u8,
    /// Partial charge obtained from charge fitting (e.g. RESP).
    pub charge: f64,
    /// Energy gradient (force = -gradient), in kcal/(mol·Å).
    pub gradient: [f64; 3],
}

/// An atom residing in the MM region.
#[derive(Debug, Clone)]
pub struct MmAtom {
    /// Global index in the full system.
    pub index: usize,
    /// Cartesian position in Ångström.
    pub position: [f64; 3],
    /// Classical point charge (e).
    pub charge: f64,
    /// Lennard-Jones well depth ε (kcal/mol).
    pub epsilon: f64,
    /// Lennard-Jones diameter σ (Å).
    pub sigma: f64,
    /// Energy gradient, in kcal/(mol·Å).
    pub gradient: [f64; 3],
}

/// A hydrogen link atom that caps a severed QM/MM covalent bond.
///
/// The link atom position is placed along the QM–MM bond vector at a
/// fractional distance `g_ratio` from the QM atom.
#[derive(Debug, Clone)]
pub struct LinkAtom {
    /// Index of the QM atom at the bond origin.
    pub qm_atom_idx: usize,
    /// Index of the MM atom at the bond terminus.
    pub mm_atom_idx: usize,
    /// Current Cartesian position of the link (hydrogen cap) atom.
    pub position: [f64; 3],
    /// Fractional position along the QM→MM bond vector.
    /// Typical value: 0.71 for a C–H replacement.
    pub g_ratio: f64,
}

impl LinkAtom {
    /// Update the link atom position from the current QM and MM atom positions.
    ///
    /// `pos = qm_pos + g_ratio * (mm_pos - qm_pos)`
    pub fn update_position(&mut self, qm_pos: [f64; 3], mm_pos: [f64; 3]) {
        for k in 0..3 {
            self.position[k] = qm_pos[k] + self.g_ratio * (mm_pos[k] - qm_pos[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// Semi-empirical tight-binding Hamiltonian (DFTB-inspired)
// ---------------------------------------------------------------------------

/// A simplified tight-binding (DFTB-like) model Hamiltonian for the QM region.
///
/// Uses exponential Slater-Koster integrals and a Gershgorin estimate for
/// eigenvalue bounds instead of a full diagonaliser.  It is a lightweight
/// diagnostic model and is **not** used for energies: the QM energy of a
/// [`QmmmSystem`] comes from the self-consistent engines of [`crate::qm_mm`].
#[derive(Debug, Clone)]
pub struct TightBindingHamiltonian {
    /// Atomic numbers of the QM atoms.
    pub atomic_numbers: Vec<u8>,
    /// Positions of the QM atoms (Å).
    pub positions: Vec<[f64; 3]>,
    /// Number of valence electrons in the QM region.
    pub n_electrons: usize,
}

/// Pre-factor A used in the Slater-Koster integral.
const SK_A: f64 = 1.0;
/// Decay exponent α used in the Slater-Koster integral (Å⁻¹).
const SK_ALPHA: f64 = 1.5;
/// Pre-factor for the repulsive pairwise energy.
const REP_A: f64 = 2.0;
/// Decay exponent for the repulsive energy (Å⁻¹).
const REP_ALPHA: f64 = 2.0;

impl TightBindingHamiltonian {
    /// Simplified Slater-Koster integral between atoms i and j at distance r (Å).
    ///
    /// Returns `A * exp(-α * r)`.
    pub fn slater_koster_integral(_atom_i: u8, _atom_j: u8, r: f64) -> f64 {
        SK_A * (-SK_ALPHA * r).exp()
    }

    /// On-site energy for an atom given its atomic number.
    fn on_site_energy(z: u8) -> f64 {
        // Very rough: use ionisation-energy-like values in eV (negative binding)
        match z {
            1 => -13.6,  // H
            6 => -11.26, // C
            7 => -14.48, // N
            8 => -17.19, // O
            _ => -10.0,
        }
    }

    /// Euclidean distance between two positions.
    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// Build the dense n_atoms × n_atoms Hamiltonian matrix.
    ///
    /// Diagonal elements are on-site energies; off-diagonal elements are
    /// Slater-Koster integrals.
    pub fn build_hamiltonian(&self) -> Vec<Vec<f64>> {
        let n = self.atomic_numbers.len();
        // Precompute all off-diagonal values to avoid simultaneous mutable borrows.
        let mut h = vec![vec![0.0_f64; n]; n];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = Self::on_site_energy(self.atomic_numbers[i]);
        }
        // Compute off-diagonal SK integrals row by row using split_at_mut.
        for i in 0..n {
            let (top, bot) = h.split_at_mut(i + 1);
            let row_i = &mut top[i];
            for (jj, row_j) in bot.iter_mut().enumerate() {
                let j = i + 1 + jj;
                let r = Self::dist(self.positions[i], self.positions[j]);
                let val =
                    Self::slater_koster_integral(self.atomic_numbers[i], self.atomic_numbers[j], r);
                row_i[j] = val;
                row_j[i] = val;
            }
        }
        h
    }

    /// Build the dense overlap matrix S (identity on diagonal, SK overlap off-diagonal).
    pub fn overlap_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.atomic_numbers.len();
        let mut s = vec![vec![0.0_f64; n]; n];
        for (i, row) in s.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for i in 0..n {
            let (top, bot) = s.split_at_mut(i + 1);
            let row_i = &mut top[i];
            for (jj, row_j) in bot.iter_mut().enumerate() {
                let j = i + 1 + jj;
                let r = Self::dist(self.positions[i], self.positions[j]);
                let val = 0.5
                    * Self::slater_koster_integral(
                        self.atomic_numbers[i],
                        self.atomic_numbers[j],
                        r,
                    );
                row_i[j] = val;
                row_j[i] = val;
            }
        }
        s
    }

    /// Gershgorin circle theorem bounds on the eigenvalues of `h`.
    ///
    /// For each row i the bound is `[h_ii - R_i, h_ii + R_i]` where
    /// `R_i = sum_{j≠i} |h_ij|`.  Returns the lower bound for each row.
    pub fn eigenvalues_gershgorin(&self, h: &[Vec<f64>]) -> Vec<f64> {
        let mut bounds = Vec::with_capacity(h.len());
        for (i, row) in h.iter().enumerate() {
            let radius: f64 = row
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, v)| v.abs())
                .sum();
            bounds.push(row[i] - radius);
        }
        bounds
    }

    /// Compute the model's repulsive pairwise energy (dimensionless model units).
    ///
    /// `E_rep = Σ_{i<j} A * exp(-α * r_ij) / r_ij`
    pub fn repulsive_energy(&self) -> f64 {
        let n = self.atomic_numbers.len();
        let mut e_rep = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let r = Self::dist(self.positions[i], self.positions[j]);
                if r > 1e-10 {
                    e_rep += REP_A * (-REP_ALPHA * r).exp() / r;
                }
            }
        }
        e_rep
    }
}

// ---------------------------------------------------------------------------
// Electrostatic embedding
// ---------------------------------------------------------------------------

/// Electrostatic (Coulomb) QM/MM interaction in the mechanical or electrostatic
/// embedding scheme.
pub struct ElectrostaticEmbedding;

impl ElectrostaticEmbedding {
    /// Compute the QM/MM Coulomb interaction energy in kcal/mol.
    ///
    /// `E = Σ_i Σ_J k_e q_i Q_J / r_iJ` with the fixed [`QmAtom::charge`] values `q_i`
    /// (mechanical embedding), `k_e = 332.0636 kcal·Å/(mol·e²)`, charges in e and
    /// distances in Å.
    pub fn qm_mm_interaction(qm_atoms: &[QmAtom], mm_atoms: &[MmAtom]) -> f64 {
        Self::qm_mm_interaction_excluding(qm_atoms, mm_atoms, &[])
    }

    /// [`qm_mm_interaction`](Self::qm_mm_interaction) without the `(qm index, mm index)`
    /// pairs listed in `excluded` (indices into the two slices), e.g. the covalently
    /// bonded atom pairs across the QM/MM boundary.
    pub fn qm_mm_interaction_excluding(
        qm_atoms: &[QmAtom],
        mm_atoms: &[MmAtom],
        excluded: &[(usize, usize)],
    ) -> f64 {
        let ke = 332.0636; // kcal·Å/(mol·e²) — Coulomb constant
        let mut energy = 0.0;
        for (i, qi) in qm_atoms.iter().enumerate() {
            for (j, qj) in mm_atoms.iter().enumerate() {
                if excluded.contains(&(i, j)) {
                    continue;
                }
                let r = dist3(qi.position, qj.position);
                if r > 1e-10 {
                    energy += ke * qi.charge * qj.charge / r;
                }
            }
        }
        energy
    }

    /// QM gradient of the electrostatic QM/MM interaction.
    ///
    /// Returns `dE/dr_i` for each QM atom in kcal/(mol·Å).
    pub fn qm_mm_gradient_qm(qm_atoms: &[QmAtom], mm_atoms: &[MmAtom]) -> Vec<[f64; 3]> {
        let ke = 332.0636;
        let mut grads = vec![[0.0_f64; 3]; qm_atoms.len()];
        for (i, qi) in qm_atoms.iter().enumerate() {
            for qj in mm_atoms {
                let dr = sub3(qi.position, qj.position);
                let r2 = dot3(dr, dr);
                if r2 > 1e-20 {
                    let r = r2.sqrt();
                    let factor = -ke * qi.charge * qj.charge / (r2 * r);
                    for k in 0..3 {
                        grads[i][k] += factor * dr[k];
                    }
                }
            }
        }
        grads
    }

    /// MM gradient of the electrostatic QM/MM interaction.
    ///
    /// Returns `dE/dr_J` for each MM atom in kcal/(mol·Å).
    pub fn qm_mm_gradient_mm(qm_atoms: &[QmAtom], mm_atoms: &[MmAtom]) -> Vec<[f64; 3]> {
        let ke = 332.0636;
        let mut grads = vec![[0.0_f64; 3]; mm_atoms.len()];
        for qi in qm_atoms {
            for (j, qj) in mm_atoms.iter().enumerate() {
                let dr = sub3(qj.position, qi.position);
                let r2 = dot3(dr, dr);
                if r2 > 1e-20 {
                    let r = r2.sqrt();
                    let factor = -ke * qi.charge * qj.charge / (r2 * r);
                    for k in 0..3 {
                        grads[j][k] += factor * dr[k];
                    }
                }
            }
        }
        grads
    }
}

// ---------------------------------------------------------------------------
// van der Waals QM/MM cross-terms
// ---------------------------------------------------------------------------

/// Lennard-Jones cross-interaction between QM and MM atoms.
///
/// QM atoms are given default LJ parameters (σ = 3.5 Å, ε = 0.5 kJ/mol ≈ 0.1195
/// kcal/mol); MM atoms carry their own [`MmAtom::epsilon`] (kcal/mol) and
/// [`MmAtom::sigma`] (Å).  Energies are in kcal/mol.
pub struct VanDerWaalsQmMm;

/// kJ per kcal.
const KJ_PER_KCAL: f64 = 4.184;
/// Default LJ σ for QM atoms (Å).
const QM_SIGMA_DEFAULT: f64 = 3.5;
/// Default LJ ε for QM atoms (kcal/mol): 0.5 kJ/mol.
const QM_EPSILON_DEFAULT: f64 = 0.5 / KJ_PER_KCAL;

impl VanDerWaalsQmMm {
    /// Lorentz-Berthelot mixing rules for σ and ε.
    fn mix(eps_i: f64, sig_i: f64, eps_j: f64, sig_j: f64) -> (f64, f64) {
        ((eps_i * eps_j).sqrt(), 0.5 * (sig_i + sig_j))
    }

    /// Compute the LJ energy (kcal/mol) between all QM and MM atom pairs.
    pub fn lj_energy(qm_atoms: &[QmAtom], mm_atoms: &[MmAtom]) -> f64 {
        Self::lj_energy_excluding(qm_atoms, mm_atoms, &[])
    }

    /// [`lj_energy`](Self::lj_energy) without the `(qm index, mm index)` pairs listed in
    /// `excluded` (indices into the two slices), e.g. the covalently bonded atom pairs
    /// across the QM/MM boundary.
    pub fn lj_energy_excluding(
        qm_atoms: &[QmAtom],
        mm_atoms: &[MmAtom],
        excluded: &[(usize, usize)],
    ) -> f64 {
        let mut energy = 0.0;
        for (i, qi) in qm_atoms.iter().enumerate() {
            for (j, qj) in mm_atoms.iter().enumerate() {
                if excluded.contains(&(i, j)) {
                    continue;
                }
                let (eps, sig) =
                    Self::mix(QM_EPSILON_DEFAULT, QM_SIGMA_DEFAULT, qj.epsilon, qj.sigma);
                let r = dist3(qi.position, qj.position);
                if r > 1e-10 {
                    let sr6 = (sig / r).powi(6);
                    energy += 4.0 * eps * (sr6 * sr6 - sr6);
                }
            }
        }
        energy
    }

    /// QM gradient (kcal/(mol·Å)) of the LJ QM/MM interaction.
    pub fn lj_gradient_qm(qm_atoms: &[QmAtom], mm_atoms: &[MmAtom]) -> Vec<[f64; 3]> {
        let mut grads = vec![[0.0_f64; 3]; qm_atoms.len()];
        for (i, qi) in qm_atoms.iter().enumerate() {
            for qj in mm_atoms {
                let (eps, sig) =
                    Self::mix(QM_EPSILON_DEFAULT, QM_SIGMA_DEFAULT, qj.epsilon, qj.sigma);
                let dr = sub3(qi.position, qj.position);
                let r2 = dot3(dr, dr);
                if r2 > 1e-20 {
                    let r = r2.sqrt();
                    let sr2 = (sig / r).powi(2);
                    let sr6 = sr2 * sr2 * sr2;
                    // dE/dr = 4ε * (-12 σ^12/r^13 + 6 σ^6/r^7)
                    let factor = 4.0 * eps * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r2;
                    for k in 0..3 {
                        grads[i][k] += factor * dr[k];
                    }
                }
            }
        }
        grads
    }
}

// ---------------------------------------------------------------------------
// Full QM/MM system
// ---------------------------------------------------------------------------

/// Combined QM/MM system container.
#[derive(Debug, Clone)]
pub struct QmmmSystem {
    /// Atoms in the QM region.
    pub qm_atoms: Vec<QmAtom>,
    /// Atoms in the MM region.
    pub mm_atoms: Vec<MmAtom>,
    /// Link (hydrogen cap) atoms at the QM/MM boundary.
    pub link_atoms: Vec<LinkAtom>,
    /// Simplified tight-binding model Hamiltonian of the QM atoms (a diagnostic
    /// model; it is not used for energies, see [`QmmmSystem::total_energy_with`]).
    pub hamiltonian: TightBindingHamiltonian,
}

impl QmmmSystem {
    /// Create an empty QM/MM system.
    pub fn new() -> Self {
        Self {
            qm_atoms: Vec::new(),
            mm_atoms: Vec::new(),
            link_atoms: Vec::new(),
            hamiltonian: TightBindingHamiltonian {
                atomic_numbers: Vec::new(),
                positions: Vec::new(),
                n_electrons: 0,
            },
        }
    }

    /// Add a QM atom. Returns the local QM index.
    ///
    /// The valence electrons of the atom are added to
    /// [`hamiltonian.n_electrons`](TightBindingHamiltonian::n_electrons).
    pub fn add_qm_atom(&mut self, pos: [f64; 3], atomic_number: u8, charge: f64) -> usize {
        let idx = self.qm_atoms.len();
        self.qm_atoms.push(QmAtom {
            index: idx,
            position: pos,
            atomic_number,
            charge,
            gradient: [0.0; 3],
        });
        self.hamiltonian.atomic_numbers.push(atomic_number);
        self.hamiltonian.positions.push(pos);
        self.hamiltonian.n_electrons += valence_electrons(atomic_number);
        idx
    }

    /// Add an MM atom. Returns the local MM index.
    pub fn add_mm_atom(&mut self, pos: [f64; 3], charge: f64, epsilon: f64, sigma: f64) -> usize {
        let idx = self.mm_atoms.len();
        self.mm_atoms.push(MmAtom {
            index: idx,
            position: pos,
            charge,
            epsilon,
            sigma,
            gradient: [0.0; 3],
        });
        idx
    }

    /// Add a link atom between QM atom `qm_idx` and MM atom `mm_idx`.
    pub fn add_link_atom(&mut self, qm_idx: usize, mm_idx: usize, g_ratio: f64) {
        let qm_pos = self.qm_atoms[qm_idx].position;
        let mm_pos = self.mm_atoms[mm_idx].position;
        let mut link = LinkAtom {
            qm_atom_idx: qm_idx,
            mm_atom_idx: mm_idx,
            position: [0.0; 3],
            g_ratio,
        };
        link.update_position(qm_pos, mm_pos);
        self.link_atoms.push(link);
    }

    /// Total QM/MM energy in kcal/mol with an SCC-DFTB description of the neutral,
    /// closed-shell (singlet) QM region.
    ///
    /// Shorthand for [`total_energy_with`](Self::total_energy_with)`(QmMethod::SccDftb, 0, 1)`.
    ///
    /// # Errors
    /// See [`total_energy_with`](Self::total_energy_with).
    pub fn total_energy(&self) -> Result<f64, QmError> {
        self.total_energy_with(QmMethod::SccDftb, 0, 1)
    }

    /// Total QM/MM energy in kcal/mol:
    ///
    /// `E_total = E_QM + E_QM/MM(elec) + E_QM/MM(vdW)`
    ///
    /// - `E_QM` is the converged energy of the QM region computed by
    ///   [`QmRegion::run`] with the chosen `method`, total `charge` and spin
    ///   `multiplicity`.  The region consists of the QM atoms plus one hydrogen cap per
    ///   link atom at `r_Q + g (r_M − r_Q)`, which saturates the severed bonds.
    /// - `E_QM/MM(elec)` is the Coulomb interaction of the fixed [`QmAtom::charge`]
    ///   values with the MM charges ([`ElectrostaticEmbedding`], mechanical embedding).
    /// - `E_QM/MM(vdW)` is the Lennard-Jones QM–MM interaction ([`VanDerWaalsQmMm`]).
    ///
    /// The covalently bonded QM/MM boundary pairs (one per link atom) are excluded from
    /// both classical terms.  A system without QM atoms has no QM/MM energy and gives
    /// `Ok(0.0)`.
    ///
    /// # Errors
    /// [`QmError::InvalidLinkAtom`] if a link atom refers to a non-existent QM or MM
    /// atom, and any failure of the QM engine (unsupported element, open-shell region,
    /// electron count incompatible with `charge`, SCF/SCC non-convergence).
    pub fn total_energy_with(
        &self,
        method: QmMethod,
        charge: i32,
        multiplicity: u32,
    ) -> Result<f64, QmError> {
        let n_qm = self.qm_atoms.len();
        let n_mm = self.mm_atoms.len();

        // Hydrogen caps at the current QM and MM positions.
        let mut cap_positions: Vec<[f64; 3]> = Vec::with_capacity(self.link_atoms.len());
        for la in &self.link_atoms {
            if la.qm_atom_idx >= n_qm {
                return Err(QmError::InvalidLinkAtom {
                    region: "QM",
                    index: la.qm_atom_idx,
                    len: n_qm,
                });
            }
            if la.mm_atom_idx >= n_mm {
                return Err(QmError::InvalidLinkAtom {
                    region: "MM",
                    index: la.mm_atom_idx,
                    len: n_mm,
                });
            }
            let mut cap = la.clone();
            cap.update_position(
                self.qm_atoms[la.qm_atom_idx].position,
                self.mm_atoms[la.mm_atom_idx].position,
            );
            cap_positions.push(cap.position);
        }
        if n_qm == 0 {
            return Ok(0.0);
        }

        let n_cap = cap_positions.len();
        let mut region = QmRegion::new((0..n_qm + n_cap).collect(), method, charge, multiplicity);
        region.atomic_numbers = self
            .qm_atoms
            .iter()
            .map(|a| u32::from(a.atomic_number))
            .chain(std::iter::repeat_n(1, n_cap))
            .collect();
        region.set_positions(
            self.qm_atoms
                .iter()
                .map(|a| a.position)
                .chain(cap_positions)
                .collect(),
        );
        region.run()?;
        // No external charges are embedded, so `energy` is the full QM energy.
        let e_qm = region.energy;

        let excluded: Vec<(usize, usize)> = self
            .link_atoms
            .iter()
            .map(|la| (la.qm_atom_idx, la.mm_atom_idx))
            .collect();
        let e_elec = ElectrostaticEmbedding::qm_mm_interaction_excluding(
            &self.qm_atoms,
            &self.mm_atoms,
            &excluded,
        );
        let e_vdw = VanDerWaalsQmMm::lj_energy_excluding(&self.qm_atoms, &self.mm_atoms, &excluded);
        Ok(e_qm + e_elec + e_vdw)
    }

    /// Update all link atom positions using the current QM/MM atom positions.
    pub fn update_link_positions(&mut self) {
        // Collect positions first to avoid borrow conflict
        let updates: Vec<([f64; 3], [f64; 3])> = self
            .link_atoms
            .iter()
            .map(|la| {
                (
                    self.qm_atoms[la.qm_atom_idx].position,
                    self.mm_atoms[la.mm_atom_idx].position,
                )
            })
            .collect();
        for (la, (qm_pos, mm_pos)) in self.link_atoms.iter_mut().zip(updates.iter()) {
            la.update_position(*qm_pos, *mm_pos);
        }
    }
}

impl Default for QmmmSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Charge equilibration
// ---------------------------------------------------------------------------

/// Simple charge equilibration using the electronegativity equalization method
/// (QEq / EEM).
///
/// Minimises the energy `E = Σ_i χ_i q_i + ½ Σ_{ij} J_ij q_i q_j` subject to
/// `Σ_i q_i = Q_total`.
#[derive(Debug, Clone)]
pub struct ChargeEquilibration {
    /// Atomic electronegativity χ (eV).
    pub electronegativity: Vec<f64>,
    /// Chemical hardness η (eV), appearing on the diagonal of J.
    pub hardness: Vec<f64>,
}

impl ChargeEquilibration {
    /// Compute atomic partial charges via electronegativity equalization.
    ///
    /// Solves the augmented linear system:
    ///
    /// ```text
    /// [ J  1 ] [ q      ]   [ -χ ]
    /// [ 1ᵀ 0 ] [ λ      ] = [ Q  ]
    /// ```
    ///
    /// using Gauss-Jordan elimination.
    pub fn compute_charges(&self, positions: &[[f64; 3]], total_charge: f64) -> Vec<f64> {
        let n = self.electronegativity.len();
        assert_eq!(
            n,
            positions.len(),
            "electronegativity/positions size mismatch"
        );
        let ke = 14.3996; // eV·Å/e²

        // Build the (n+1) × (n+1) augmented matrix for [q; λ]
        let size = n + 1;
        let mut mat = vec![vec![0.0_f64; size + 1]; size];

        // Fill J matrix
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    mat[i][j] = 2.0 * self.hardness[i];
                } else {
                    let r = dist3(positions[i], positions[j]);
                    mat[i][j] = if r > 1e-10 { ke / r } else { 0.0 };
                }
            }
            // Lagrange column and row
            mat[i][n] = 1.0;
            mat[n][i] = 1.0;
        }
        // mat[n][n] = 0 (already)

        // RHS: [-χ_i, …, Q_total]
        for (row, &en) in mat.iter_mut().zip(self.electronegativity.iter()).take(n) {
            row[size] = -en;
        }
        mat[n][size] = total_charge;

        // Gauss-Jordan elimination with partial pivoting
        let mut row_perm: Vec<usize> = (0..size).collect();
        for col in 0..size {
            // Pivot: find row with maximum absolute value
            let pivot_row = (col..size)
                .max_by(|&a, &b| {
                    mat[row_perm[a]][col]
                        .abs()
                        .partial_cmp(&mat[row_perm[b]][col].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(col);
            row_perm.swap(col, pivot_row);
            let pr = row_perm[col];
            let pivot = mat[pr][col];
            if pivot.abs() < 1e-14 {
                continue;
            }
            // Normalise pivot row
            for v in mat[pr].iter_mut() {
                *v /= pivot;
            }
            // Eliminate all other rows
            for r in 0..size {
                if r == pr {
                    continue;
                }
                let factor = mat[r][col];
                // Copy pivot row values before modifying mat[r] to avoid borrow issues.
                let pr_row: Vec<f64> = mat[pr].clone();
                for (mat_rc, &prc) in mat[r].iter_mut().zip(pr_row.iter()) {
                    *mat_rc -= factor * prc;
                }
            }
        }

        // Extract solution: q_i = mat[row_perm[i]][size]
        (0..n).map(|i| mat[row_perm[i]][size]).collect()
    }
}

// ---------------------------------------------------------------------------
// Valence electron count
// ---------------------------------------------------------------------------

/// Number of valence electrons of the neutral atom with atomic number `z`.
///
/// Counts the electrons outside the preceding noble-gas core and, for the p-block
/// elements of periods 4–7 (and Lu–Hg, Lr–Cn), excludes the completely filled
/// (n−1)d and (n−2)f subshells, so that H → 1, C → 4, N → 5, O → 6, F → 7, S → 6,
/// Zn → 12, Ga → 3, Br → 7, Tl → 3.  Lanthanides and actinides count their 4f/5f
/// electrons as valence.  Returns 0 for `z = 0` and for `z > 118`.
fn valence_electrons(z: u8) -> usize {
    // (last atomic number of the period, atomic number of the preceding noble gas)
    const PERIODS: [(u32, u32); 7] = [
        (2, 0),
        (10, 2),
        (18, 10),
        (36, 18),
        (54, 36),
        (86, 54),
        (118, 86),
    ];
    let z = u32::from(z);
    let Some(&(last, core)) = PERIODS.iter().find(|&&(last, _)| z <= last) else {
        return 0;
    };
    let mut valence = z - core;
    match last {
        // Ga–Kr and In–Xe: filled (n−1)d¹⁰.
        36 | 54 if z > last - 6 => valence -= 10,
        // Lu–Hg (filled 4f¹⁴), then Tl–Rn (additionally filled 5d¹⁰).
        86 if z > last - 6 => valence -= 24,
        86 if z >= 71 => valence -= 14,
        // Lr–Cn (filled 5f¹⁴), then Nh–Og (additionally filled 6d¹⁰).
        118 if z > last - 6 => valence -= 24,
        118 if z >= 103 => valence -= 14,
        _ => {}
    }
    valence as usize
}

// ---------------------------------------------------------------------------
// Small vector helpers (avoids nalgebra dependency)
// ---------------------------------------------------------------------------

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ---------------------------------------------------------------------------
// QM region definition helpers
// ---------------------------------------------------------------------------

/// Selection criteria for automatically building a QM region.
#[derive(Debug, Clone)]
pub struct QmRegionSelector {
    /// Indices of atoms that must always be in the QM region.
    pub core_atoms: Vec<usize>,
    /// Cutoff radius in Å from core atoms — all atoms within this sphere are added to QM.
    pub cutoff_radius: f64,
}

impl QmRegionSelector {
    /// Create a new selector with the given core atom indices and radius.
    pub fn new(core_atoms: Vec<usize>, cutoff_radius: f64) -> Self {
        Self {
            core_atoms,
            cutoff_radius,
        }
    }

    /// Select QM atom indices from a full set of positions.
    ///
    /// Returns a sorted, deduplicated list of atom indices within `cutoff_radius`
    /// of any core atom.
    pub fn select_qm_atoms(&self, positions: &[[f64; 3]]) -> Vec<usize> {
        let mut selected: std::collections::BTreeSet<usize> =
            self.core_atoms.iter().cloned().collect();
        for &core in &self.core_atoms {
            if core >= positions.len() {
                continue;
            }
            for (j, &pos) in positions.iter().enumerate() {
                if dist3(positions[core], pos) <= self.cutoff_radius {
                    selected.insert(j);
                }
            }
        }
        selected.into_iter().collect()
    }

    /// Identify QM/MM boundary bonds: pairs (qm_idx, mm_idx) where one atom
    /// is in the QM list and its bonded partner is in the MM list.
    pub fn boundary_bonds(
        &self,
        qm_atoms: &[usize],
        bonded_pairs: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let qm_set: std::collections::HashSet<usize> = qm_atoms.iter().cloned().collect();
        bonded_pairs
            .iter()
            .filter_map(|&(i, j)| {
                let i_qm = qm_set.contains(&i);
                let j_qm = qm_set.contains(&j);
                if i_qm && !j_qm {
                    Some((i, j))
                } else if j_qm && !i_qm {
                    Some((j, i))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// MM point charges
// ---------------------------------------------------------------------------

/// A collection of MM point charges for use in electrostatic embedding.
#[derive(Debug, Clone, Default)]
pub struct PointChargeField {
    /// Charge values (elementary charge units).
    pub charges: Vec<f64>,
    /// Positions of the charges in Å.
    pub positions: Vec<[f64; 3]>,
}

impl PointChargeField {
    /// Create an empty point charge field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a point charge.
    pub fn add_charge(&mut self, charge: f64, position: [f64; 3]) {
        self.charges.push(charge);
        self.positions.push(position);
    }

    /// Compute the electrostatic potential at a point `r` (in Å).
    ///
    /// V(r) = Σ Q_J / |r - R_J|  (Coulomb units: ke = 1 implicit)
    pub fn potential_at(&self, r: [f64; 3]) -> f64 {
        let ke = 332.0636; // kcal·Å/(mol·e²)
        self.charges
            .iter()
            .zip(self.positions.iter())
            .map(|(&q, &pos)| {
                let d = dist3(r, pos);
                if d > 1e-10 { ke * q / d } else { 0.0 }
            })
            .sum()
    }

    /// Compute the electric field vector at a point `r` in Å.
    ///
    /// E(r) = -∇V = Σ Q_J * (r - R_J) / |r - R_J|^3
    pub fn field_at(&self, r: [f64; 3]) -> [f64; 3] {
        let ke = 332.0636;
        let mut field = [0.0f64; 3];
        for (&q, &pos) in self.charges.iter().zip(self.positions.iter()) {
            let dr = sub3(r, pos);
            let r2 = dot3(dr, dr);
            if r2 > 1e-20 {
                let r3 = r2 * r2.sqrt();
                for k in 0..3 {
                    field[k] += ke * q * dr[k] / r3;
                }
            }
        }
        field
    }

    /// Total number of point charges.
    pub fn len(&self) -> usize {
        self.charges.len()
    }

    /// Returns true if there are no charges.
    pub fn is_empty(&self) -> bool {
        self.charges.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Mechanical embedding
// ---------------------------------------------------------------------------

/// Mechanical embedding: compute the QM/MM interaction in the mechanical
/// embedding scheme, where the QM charges are fixed from the MM force field.
pub struct MechanicalEmbedding;

impl MechanicalEmbedding {
    /// Mechanical embedding QM/MM interaction energy.
    ///
    /// Uses the MM charges for QM atoms (not self-consistent QM charges).
    /// `qm_mm_charges` are the MM-force-field charges of the QM atoms.
    pub fn interaction_energy(
        qm_positions: &[[f64; 3]],
        qm_mm_charges: &[f64],
        mm_atoms: &[MmAtom],
    ) -> f64 {
        let ke = 332.0636;
        let mut energy = 0.0;
        for (pos, &charge) in qm_positions.iter().zip(qm_mm_charges.iter()) {
            for mm in mm_atoms {
                let r = dist3(*pos, mm.position);
                if r > 1e-10 {
                    energy += ke * charge * mm.charge / r;
                }
            }
        }
        energy
    }
}

// ---------------------------------------------------------------------------
// QM/MM link atom force redistribution
// ---------------------------------------------------------------------------

/// Redistribute force from link atom to the host QM and MM atoms.
///
/// The link atom gradient is projected back to the QM and MM atoms
/// using the g_ratio factor:
///   grad_qm += (1 - g) * grad_link
///   grad_mm += g * grad_link
///
/// where g = `link_atom.g_ratio`.
pub fn redistribute_link_force(
    link_grad: [f64; 3],
    g_ratio: f64,
    grad_qm: &mut [f64; 3],
    grad_mm: &mut [f64; 3],
) {
    let g = g_ratio;
    for k in 0..3 {
        grad_qm[k] += (1.0 - g) * link_grad[k];
        grad_mm[k] += g * link_grad[k];
    }
}

// ---------------------------------------------------------------------------
// ONIOM extrapolation energy
// ---------------------------------------------------------------------------

/// ONIOM (Our own N-layered Integrated molecular Orbital and molecular Mechanics)
/// energy extrapolation.
///
/// The two-layer ONIOM energy is:
/// ```text
/// E_ONIOM = E_QM(model) + E_MM(real) - E_MM(model)
/// ```
/// where "model" is the QM region treated at both levels.
///
/// # Arguments
/// * `e_qm_model`  – QM energy of the model (inner) region.
/// * `e_mm_real`   – MM energy of the entire (real) system.
/// * `e_mm_model`  – MM energy of the model region with MM parameters.
pub fn oniom_energy(e_qm_model: f64, e_mm_real: f64, e_mm_model: f64) -> f64 {
    e_qm_model + e_mm_real - e_mm_model
}

/// Three-layer ONIOM energy.
///
/// ```text
/// E = E_high(inner) + E_mid(middle) - E_mid(inner)
///   + E_low(outer) - E_low(middle)
/// ```
pub fn oniom3_energy(
    e_high_inner: f64,
    e_mid_middle: f64,
    e_mid_inner: f64,
    e_low_outer: f64,
    e_low_middle: f64,
) -> f64 {
    e_high_inner + e_mid_middle - e_mid_inner + e_low_outer - e_low_middle
}

// ---------------------------------------------------------------------------
// Buffer zone (adaptive QM/MM boundary) smoothing
// ---------------------------------------------------------------------------

/// Smooth transition weight for atoms in the buffer zone.
///
/// Uses a 5th-order polynomial switch that goes from 1 at `r_inner` to 0 at
/// `r_outer`, ensuring C² continuity:
///
/// ```text
/// t = (r - r_inner) / (r_outer - r_inner)
/// S(t) = 1 - 6t^5 + 15t^4 - 10t^3     (0 ≤ t ≤ 1)
/// ```
pub fn buffer_weight(r: f64, r_inner: f64, r_outer: f64) -> f64 {
    if r <= r_inner {
        return 1.0;
    }
    if r >= r_outer {
        return 0.0;
    }
    let t = (r - r_inner) / (r_outer - r_inner);
    1.0 - t * t * t * (10.0 - t * (15.0 - 6.0 * t))
}

/// Derivative of the buffer weight with respect to r.
pub fn buffer_weight_derivative(r: f64, r_inner: f64, r_outer: f64) -> f64 {
    if r <= r_inner || r >= r_outer {
        return 0.0;
    }
    let width = r_outer - r_inner;
    let t = (r - r_inner) / width;
    // dS/dr = (dS/dt) * (dt/dr)
    // dS/dt = -30 t^4 + 60 t^3 - 30 t^2 = -30 t^2 (t^2 - 2t + 1) = -30 t^2 (t-1)^2
    let ds_dt = -30.0 * t * t * (t - 1.0) * (t - 1.0);
    ds_dt / width
}

// ---------------------------------------------------------------------------
// Adaptive QM region expander
// ---------------------------------------------------------------------------

/// Adaptive QM region expansion: grows the QM sphere until the gradient
/// norm on the boundary falls below a threshold.
///
/// Starting from `initial_radius`, increases the QM radius by `step` until
/// either `max_radius` is reached or the criterion is satisfied.
///
/// Returns the chosen QM radius.
pub fn adaptive_qm_radius(
    core_pos: [f64; 3],
    positions: &[[f64; 3]],
    gradients: &[[f64; 3]],
    initial_radius: f64,
    step: f64,
    max_radius: f64,
    threshold: f64,
) -> f64 {
    let mut radius = initial_radius;
    loop {
        // Find atoms at the boundary shell: [radius - step, radius]
        let boundary_grad_norm: f64 = positions
            .iter()
            .zip(gradients.iter())
            .filter_map(|(pos, grad)| {
                let r = dist3(core_pos, *pos);
                if r >= radius - step && r <= radius {
                    Some((grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt())
                } else {
                    None
                }
            })
            .fold(0.0_f64, f64::max);
        if boundary_grad_norm < threshold || radius >= max_radius {
            break;
        }
        radius = (radius + step).min(max_radius);
    }
    radius
}

// ---------------------------------------------------------------------------
// Constraint correction for link atoms
// ---------------------------------------------------------------------------

/// Holonomic constraint correction for a link atom bond.
///
/// Returns the penalty energy for deviating from the target link bond length:
/// `E_constraint = k/2 * (|r_link - r_qm| - d_target)^2`
pub fn link_bond_constraint_energy(
    link_pos: [f64; 3],
    qm_pos: [f64; 3],
    d_target: f64,
    k: f64,
) -> f64 {
    let r = dist3(link_pos, qm_pos);
    let delta = r - d_target;
    0.5 * k * delta * delta
}

/// Gradient of the link bond constraint with respect to the link atom position.
pub fn link_bond_constraint_gradient(
    link_pos: [f64; 3],
    qm_pos: [f64; 3],
    d_target: f64,
    k: f64,
) -> [f64; 3] {
    let dr = sub3(link_pos, qm_pos);
    let r = dist3(link_pos, qm_pos);
    if r < 1e-14 {
        return [0.0; 3];
    }
    let delta = r - d_target;
    let factor = k * delta / r;
    [dr[0] * factor, dr[1] * factor, dr[2] * factor]
}

// ---------------------------------------------------------------------------
// Capping hydrogen geometry
// ---------------------------------------------------------------------------

/// Compute the optimal position of a capping hydrogen atom at the QM/MM bond.
///
/// The capping H is placed along the QM→MM bond at `g_ratio` from the QM atom,
/// scaled to the standard C-H bond length `r_ch` (default ≈ 1.09 Å).
///
/// # Returns
/// Cartesian position of the capping hydrogen.
pub fn capping_hydrogen_position(qm_pos: [f64; 3], mm_pos: [f64; 3], r_ch: f64) -> [f64; 3] {
    let dr = sub3(mm_pos, qm_pos);
    let r = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
    if r < 1e-14 {
        return qm_pos;
    }
    let scale = r_ch / r;
    [
        qm_pos[0] + scale * dr[0],
        qm_pos[1] + scale * dr[1],
        qm_pos[2] + scale * dr[2],
    ]
}

// ---------------------------------------------------------------------------
// Mulliken charge analysis (simplified)
// ---------------------------------------------------------------------------

/// Simplified Mulliken population analysis.
///
/// Given a density matrix `P` (n×n, row-major) and an overlap matrix `S`
/// (n×n, row-major), computes the Mulliken population on each orbital.
///
/// `q_i = Σ_j P_ij * S_ji`  (diagonal of PS).
///
/// Returns the Mulliken population vector of length n.
pub fn mulliken_populations(density_matrix: &[Vec<f64>], overlap_matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = density_matrix.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let p = density_matrix[i].get(j).copied().unwrap_or(0.0);
                    let s = overlap_matrix
                        .get(j)
                        .and_then(|row| row.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    p * s
                })
                .sum()
        })
        .collect()
}

/// Mulliken gross charges given populations, core charges, and number of electrons.
///
/// `q_i = Z_i - N_i`  where Z_i is the nuclear charge and N_i the Mulliken population.
pub fn mulliken_charges(populations: &[f64], nuclear_charges: &[f64]) -> Vec<f64> {
    populations
        .iter()
        .zip(nuclear_charges.iter())
        .map(|(&n, &z)| z - n)
        .collect()
}

// ---------------------------------------------------------------------------
// Ewald correction for QM/MM long-range electrostatics
// ---------------------------------------------------------------------------

/// Ewald correction energy for a QM atom embedded in an MM periodic charge field.
///
/// Computes the short-range correction (removes double-counting from the
/// classical Ewald sum for QM charges):
/// `E_corr = -Σ_i q_i^QM * V_Ewald_self(q_i^QM, α)`
///
/// Self-energy: `E_self = -α/√π * Σ_i (q_i)^2`
pub fn ewald_self_correction(charges: &[f64], alpha: f64) -> f64 {
    let prefactor = alpha / std::f64::consts::PI.sqrt();
    -prefactor * charges.iter().map(|&q| q * q).sum::<f64>()
}

/// Reciprocal-space Ewald correction for QM charges in a cubic box of side L.
///
/// Computes the leading G=0 term only (dipole correction):
/// `E_G0 = -π / (2 α² V) * |Σ_i q_i r_i|² = 0` for neutral systems.
///
/// For a non-neutral system, returns a rough estimate of the G=0 divergence.
pub fn ewald_g0_correction(
    charges: &[f64],
    positions: &[[f64; 3]],
    alpha: f64,
    box_length: f64,
) -> f64 {
    let volume = box_length.powi(3);
    if volume < 1e-30 {
        return 0.0;
    }
    // Dipole moment
    let mut dipole = [0.0_f64; 3];
    for (q, pos) in charges.iter().zip(positions.iter()) {
        for k in 0..3 {
            dipole[k] += q * pos[k];
        }
    }
    let d2 = dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2];
    // Standard dipole correction: E = π/(2 α² V) * |d|² (sign convention varies)
    std::f64::consts::PI / (2.0 * alpha * alpha * volume) * d2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
