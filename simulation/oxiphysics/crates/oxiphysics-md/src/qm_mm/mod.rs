// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! QM/MM hybrid quantum mechanics / molecular mechanics methods.
//!
//! Provides QM/MM hybrid calculations where the QM region uses PM3 (semi-empirical),
//! SCC-DFTB (density-functional tight-binding), HF/STO-3G (Hartree-Fock), or
//! B3LYP/STO-3G (restricted Kohn-Sham DFT).  The MM region uses standard force-field point charges.
//!
//! ## Energy decomposition
//!
//! E_total = E_QM + E_MM + E_QM/MM
//!
//! where E_QM/MM can be computed via:
//! - **Mechanical embedding**: classical Coulomb + Lennard-Jones interaction between
//!   fixed QM and MM charges ([`mechanical_embedding`]).
//! - **Electrostatic embedding**: the MM point charges enter the QM Hamiltonian
//!   ([`QmRegion::external_charges`]) and polarise the QM wavefunction of every
//!   engine (PM3, SCC-DFTB, HF, KS-DFT).  [`QmMmSystem::compute`] fills the external
//!   charges from the MM region, runs the QM engine and evaluates E_QM/MM together
//!   with the QM/MM forces on both regions.
//!
//! ## Boundary treatment
//!
//! Covalent bonds that cross the QM/MM boundary are capped with hydrogen
//! link atoms to saturate the QM region valence.  The charge of the MM boundary
//! atom bonded to a link atom is deleted from the embedding (charge-deletion
//! scheme) and the link-atom force is redistributed onto the QM and MM boundary
//! atoms by the chain rule.
//!
//! References:
//! - Field, M. J., Bash, P. A., & Karplus, M. (1990). *J. Comput. Chem.* 11, 700.
//! - Bakowies, D., & Thiel, W. (1996). *J. Phys. Chem.* 100, 10580.

pub mod dft;
pub(crate) mod embedding;
pub mod error;
pub mod hf;
pub mod pm3;
pub mod scc_dftb;
pub(crate) mod sp_overlap;

pub use error::{MmError, QmError};

/// Coulomb constant in kcal·Å/(mol·e²): E = 332.0637 · q_i q_j / r.
pub const COULOMB_KCAL_PER_MOL_ANG: f64 = 332.063_713;

// ---------------------------------------------------------------------------
// QmMethod — enum for QM level of theory
// ---------------------------------------------------------------------------

/// QM level of theory.
///
/// The enum is `#[non_exhaustive]`: further levels of theory can be added without a
/// breaking change, so a `match` outside this crate needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QmMethod {
    /// PM3 semi-empirical NDDO method (Stewart 1989).
    Pm3,
    /// Self-consistent-charge DFTB (Elstner 1998, mio-0-1 parameters).
    SccDftb,
    /// Restricted Hartree-Fock with STO-3G minimal basis.
    Hf,
    /// Restricted Kohn-Sham DFT with the B3LYP hybrid functional (STO-3G basis):
    /// B88 exchange (weight 0.72) + Slater (0.08) + 20 % exact exchange, with LYP
    /// (0.81) and VWN(RPA) (0.19) correlation, evaluated on a Becke molecular grid
    /// (see [`crate::quantum_chemistry::XcFunctional::B3lyp`]).
    DftB3lyp,
}

impl QmMethod {
    /// Human-readable name for this method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pm3 => "PM3",
            Self::SccDftb => "SCC-DFTB",
            Self::Hf => "HF",
            Self::DftB3lyp => "DFT/B3LYP",
        }
    }

    /// Whether the engine of this method has parameters (PM3, SCC-DFTB) or a basis
    /// set (HF, KS-DFT: STO-3G) for the element with atomic number `atomic_number`.
    pub fn supports_element(&self, atomic_number: u32) -> bool {
        match self {
            Self::Pm3 => pm3::supports_element(atomic_number),
            Self::SccDftb => scc_dftb::supports_element(atomic_number),
            Self::Hf | Self::DftB3lyp => {
                crate::quantum_chemistry::sto3g_supports_element(atomic_number)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QmRegion
// ---------------------------------------------------------------------------

/// The quantum-mechanical region of a QM/MM system.
///
/// Contains the atom indices that are treated quantum mechanically, the
/// chosen QM method, and the total charge of the QM region.
///
/// ## Energy decomposition with external charges
///
/// When [`external_charges`](Self::external_charges) is non-empty (electrostatic
/// embedding) the QM energy is split as
///
/// E = `energy` + `embedding_energy`
///
/// where `embedding_energy` is the electrostatic interaction of the polarised QM
/// electrons and cores with the external charges and `energy` is the remaining
/// (polarised) QM energy.  The reported `forces` and `external_charge_forces`
/// are the negative gradients of the sum.  Without external charges
/// `embedding_energy` is zero and `energy` is the plain QM energy.
pub struct QmRegion {
    /// Indices of atoms in the QM region (into the parent system's atom list).
    pub atom_indices: Vec<usize>,
    /// QM method to use.
    pub method: QmMethod,
    /// Total formal charge of the QM region.
    pub charge: i32,
    /// Spin multiplicity (2S+1).
    pub multiplicity: u32,
    /// Positions of QM atoms (x, y, z) in Å.
    pub positions: Vec<[f64; 3]>,
    /// Atomic numbers of QM atoms.
    pub atomic_numbers: Vec<u32>,
    /// External point charges `(position in Å, charge in e)` that polarise the
    /// QM wavefunction (electrostatic embedding).  Empty for a gas-phase run.
    pub external_charges: Vec<([f64; 3], f64)>,
    /// QM energy (in kcal/mol, filled after a QM calculation), excluding the
    /// electrostatic interaction with `external_charges` (see `embedding_energy`).
    pub energy: f64,
    /// Electrostatic interaction energy (kcal/mol) between the polarised QM region
    /// (electrons and cores) and `external_charges`.
    pub embedding_energy: f64,
    /// QM forces on QM atoms \[\[fx, fy, fz\\]; n_qm] in kcal/(mol·Å):
    /// −∇(energy + embedding_energy).
    pub forces: Vec<[f64; 3]>,
    /// Mulliken net atomic charges (e) of the QM atoms after the last run.
    pub partial_charges: Vec<f64>,
    /// Forces (kcal/(mol·Å)) exerted by the QM region on each external charge.
    pub external_charge_forces: Vec<[f64; 3]>,
    /// Lennard-Jones well depth (kcal/mol) of each QM atom for QM/MM van der Waals
    /// interactions in [`QmMmSystem::compute`]; 0 disables the term for that atom.
    pub lj_epsilon: Vec<f64>,
    /// Lennard-Jones diameter (Å) of each QM atom for QM/MM van der Waals interactions.
    pub lj_sigma: Vec<f64>,
}

impl QmRegion {
    /// Create a new QM region whose per-atom data are **placeholders**.
    ///
    /// `atomic_numbers` is initialised to all hydrogen and `positions` to the
    /// origin; both must be overwritten (`region.atomic_numbers = …`,
    /// [`set_positions`](Self::set_positions)) before [`run`](Self::run).  A region
    /// that was never populated is rejected by every engine with
    /// [`QmError::CoincidentAtoms`] instead of being computed.  Prefer
    /// [`try_new`](Self::try_new), which takes and validates the atoms up front.
    ///
    /// # Arguments
    /// * `atom_indices` — indices into the full system
    /// * `method` — QM level of theory
    /// * `charge` — total charge
    /// * `multiplicity` — spin multiplicity (1 = singlet)
    pub fn new(atom_indices: Vec<usize>, method: QmMethod, charge: i32, multiplicity: u32) -> Self {
        let n = atom_indices.len();
        Self {
            atom_indices,
            method,
            charge,
            multiplicity,
            positions: vec![[0.0; 3]; n],
            atomic_numbers: vec![1; n],
            external_charges: Vec::new(),
            energy: 0.0,
            embedding_energy: 0.0,
            forces: vec![[0.0; 3]; n],
            partial_charges: vec![0.0; n],
            external_charge_forces: Vec::new(),
            lj_epsilon: vec![0.0; n],
            lj_sigma: vec![0.0; n],
        }
    }

    /// Create a QM region from complete, validated atom data.
    ///
    /// # Arguments
    /// * `atom_indices` — indices into the full system
    /// * `method` — QM level of theory
    /// * `charge` — total charge
    /// * `multiplicity` — spin multiplicity (1 = singlet)
    /// * `atomic_numbers` — atomic number of every QM atom
    /// * `positions` — positions of the QM atoms in Å
    ///
    /// # Errors
    /// * [`QmError::EmptyRegion`] if `atom_indices` is empty;
    /// * [`QmError::DimensionMismatch`] if `atomic_numbers` or `positions` does not
    ///   have one entry per atom index;
    /// * [`QmError::UnsupportedElement`] for an element the method has no parameters
    ///   or basis set for (see [`QmMethod::supports_element`]);
    /// * [`QmError::CoincidentAtoms`] if two atoms occupy the same position.
    pub fn try_new(
        atom_indices: Vec<usize>,
        method: QmMethod,
        charge: i32,
        multiplicity: u32,
        atomic_numbers: Vec<u32>,
        positions: Vec<[f64; 3]>,
    ) -> Result<Self, QmError> {
        let n = atom_indices.len();
        if n == 0 {
            return Err(QmError::EmptyRegion);
        }
        if atomic_numbers.len() != n {
            return Err(QmError::DimensionMismatch {
                what: "atomic numbers vs atom indices",
                expected: n,
                found: atomic_numbers.len(),
            });
        }
        if positions.len() != n {
            return Err(QmError::DimensionMismatch {
                what: "positions vs atom indices",
                expected: n,
                found: positions.len(),
            });
        }
        if let Some(&z) = atomic_numbers
            .iter()
            .find(|&&z| !method.supports_element(z))
        {
            return Err(QmError::UnsupportedElement(z));
        }
        embedding::check_no_coincident_atoms(&positions)?;
        let mut region = Self::new(atom_indices, method, charge, multiplicity);
        region.atomic_numbers = atomic_numbers;
        region.positions = positions;
        Ok(region)
    }

    /// Number of QM atoms.
    pub fn n_atoms(&self) -> usize {
        self.atom_indices.len()
    }

    /// Set positions of QM atoms (in Å).
    pub fn set_positions(&mut self, positions: Vec<[f64; 3]>) {
        self.positions = positions;
    }

    /// Total potential energy of the QM region in the field of the external
    /// charges (kcal/mol): `energy + embedding_energy`.
    pub fn total_energy_in_field(&self) -> f64 {
        self.energy + self.embedding_energy
    }

    /// Dipole moment Σ_A q_A R_A (e·Å) of the Mulliken partial charges from the
    /// last run.  Origin independent only for a neutral region.
    pub fn dipole_moment(&self) -> [f64; 3] {
        let mut mu = [0.0_f64; 3];
        for (q, r) in self.partial_charges.iter().zip(self.positions.iter()) {
            for (m, x) in mu.iter_mut().zip(r.iter()) {
                *m += q * x;
            }
        }
        mu
    }

    /// Run the QM calculation using the configured method.
    ///
    /// Populates `energy`, `embedding_energy` (kcal/mol), `forces`,
    /// `external_charge_forces` (kcal/(mol·Å)) and `partial_charges` (e).
    /// All outputs are cleared before the calculation starts, so a failed run
    /// never leaves the results of a previous call behind.
    ///
    /// # Errors
    /// Returns a [`QmError`] if the region is empty or inconsistent, contains an
    /// element the method does not support, is open-shell, has an invalid
    /// electron count or external charge, or if the SCF fails to converge.
    pub fn run(&mut self) -> Result<(), QmError> {
        match self.method {
            QmMethod::Pm3 => pm3::run_pm3(self),
            QmMethod::SccDftb => scc_dftb::run_scc_dftb(self),
            QmMethod::Hf => hf::run_hf(self),
            QmMethod::DftB3lyp => dft::run_dft_b3lyp(self),
        }
    }

    /// Former placeholder QM calculation; now simply forwards to [`run`](QmRegion::run).
    ///
    /// # Deprecated
    /// Use [`run`](QmRegion::run) instead, which dispatches to the real QM engine.
    ///
    /// # Errors
    /// Same as [`run`](QmRegion::run).
    #[deprecated(since = "0.1.2", note = "Use `run()` which calls the real QM solver")]
    pub fn run_placeholder(&mut self) -> Result<(), QmError> {
        self.run()
    }
}

// ---------------------------------------------------------------------------
// MmRegion
// ---------------------------------------------------------------------------

/// The classical molecular mechanics region.
///
/// All per-atom arrays (`positions`, `charges`, `epsilon`, `sigma`) must have one
/// entry per MM atom; [`compute`](Self::compute) checks this and evaluates the
/// non-bonded (Lennard-Jones + Coulomb) energy and forces of the region.
pub struct MmRegion {
    /// Indices of atoms in the MM region.
    pub atom_indices: Vec<usize>,
    /// Positions of MM atoms (x, y, z) in Å.
    pub positions: Vec<[f64; 3]>,
    /// Partial charges of MM atoms in units of elementary charge.
    pub charges: Vec<f64>,
    /// Epsilon (LJ well depth) for each MM atom in kcal/mol.
    pub epsilon: Vec<f64>,
    /// Sigma (LJ diameter) for each MM atom in Å.
    pub sigma: Vec<f64>,
    /// Non-bonded MM energy (kcal/mol), filled by [`compute`](Self::compute).
    pub energy: f64,
    /// Non-bonded MM forces on the MM atoms (kcal/(mol·Å)), filled by
    /// [`compute`](Self::compute).
    pub forces: Vec<[f64; 3]>,
}

impl MmRegion {
    /// Create a new MM region.
    pub fn new(atom_indices: Vec<usize>) -> Self {
        let n = atom_indices.len();
        Self {
            atom_indices,
            positions: vec![[0.0; 3]; n],
            charges: vec![0.0; n],
            epsilon: vec![0.1; n],
            sigma: vec![3.5; n],
            energy: 0.0,
            forces: vec![[0.0; 3]; n],
        }
    }

    /// Number of MM atoms.
    pub fn n_atoms(&self) -> usize {
        self.atom_indices.len()
    }

    /// Evaluate the non-bonded energy and forces of the MM atoms.
    ///
    /// Every pair of atoms interacts through a Lennard-Jones 12-6 potential with
    /// Lorentz–Berthelot mixing and a bare Coulomb potential,
    ///
    /// E = Σ_{i<j} [ 4 ε_ij ((σ_ij/r)¹² − (σ_ij/r)⁶) + k_e q_i q_j / r ],
    ///
    /// with ε_ij = √(ε_i ε_j), σ_ij = (σ_i + σ_j)/2 and
    /// k_e = [`COULOMB_KCAL_PER_MOL_ANG`] (kcal·Å/(mol·e²)).  `energy` (kcal/mol)
    /// and `forces` (F_i = −∇_i E, kcal/(mol·Å)) are filled; both are cleared first,
    /// so a failure never leaves the results of a previous call behind.
    ///
    /// No exclusions of bonded pairs, no cut-off and no periodic boundary conditions
    /// are applied: every pair of listed atoms interacts.  Bonded terms and the
    /// exclusion of 1-2 / 1-3 neighbours are the responsibility of the MM force field.
    ///
    /// # Errors
    /// [`MmError::LengthMismatch`] if `charges`, `epsilon` or `sigma` does not have
    /// one entry per position, [`MmError::InvalidParameter`] for a non-finite
    /// position, charge or Lennard-Jones parameter (or a negative ε or σ), and
    /// [`MmError::CoincidentAtoms`] if two atoms occupy the same position.
    pub fn compute(&mut self) -> Result<(), MmError> {
        /// Squared distance (Å²) below which two atoms count as coincident.
        const MIN_PAIR_DISTANCE_SQ: f64 = 1e-12;

        let n = self.positions.len();
        self.energy = 0.0;
        self.forces = vec![[0.0; 3]; n];
        for (what, found) in [
            ("charges", self.charges.len()),
            ("epsilon", self.epsilon.len()),
            ("sigma", self.sigma.len()),
        ] {
            if found != n {
                return Err(MmError::LengthMismatch {
                    what,
                    expected: n,
                    found,
                });
            }
        }
        for index in 0..n {
            let invalid = |what: &'static str| MmError::InvalidParameter { what, index };
            if self.positions[index].iter().any(|x| !x.is_finite()) {
                return Err(invalid("position"));
            }
            if !self.charges[index].is_finite() {
                return Err(invalid("charge"));
            }
            if !self.epsilon[index].is_finite() || self.epsilon[index] < 0.0 {
                return Err(invalid("Lennard-Jones epsilon"));
            }
            if !self.sigma[index].is_finite() || self.sigma[index] < 0.0 {
                return Err(invalid("Lennard-Jones sigma"));
            }
        }

        let mut energy = 0.0_f64;
        let mut forces = vec![[0.0_f64; 3]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    self.positions[i][0] - self.positions[j][0],
                    self.positions[i][1] - self.positions[j][1],
                    self.positions[i][2] - self.positions[j][2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 < MIN_PAIR_DISTANCE_SQ {
                    return Err(MmError::CoincidentAtoms {
                        first: i,
                        second: j,
                    });
                }
                let eps_ij = (self.epsilon[i] * self.epsilon[j]).sqrt();
                let sigma_ij = 0.5 * (self.sigma[i] + self.sigma[j]);
                let sr2 = sigma_ij * sigma_ij / r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                let e_coulomb =
                    COULOMB_KCAL_PER_MOL_ANG * self.charges[i] * self.charges[j] / r2.sqrt();
                energy += 4.0 * eps_ij * (sr12 - sr6) + e_coulomb;
                // F_i = −dE/dr_i = [24 ε (2 sr¹² − sr⁶) + E_coulomb] / r² · (r_i − r_j).
                let f_scale = (24.0 * eps_ij * (2.0 * sr12 - sr6) + e_coulomb) / r2;
                add3(&mut forces[i], &d, f_scale);
                add3(&mut forces[j], &d, -f_scale);
            }
        }
        self.energy = energy;
        self.forces = forces;
        Ok(())
    }

    /// Former placeholder MM energy (Lennard-Jones only, no forces); now simply
    /// forwards to [`compute`](Self::compute).
    ///
    /// # Deprecated
    /// Use [`compute`](Self::compute), which adds the Coulomb term and the forces.
    ///
    /// # Errors
    /// Same as [`compute`](Self::compute).
    #[deprecated(
        since = "0.1.3",
        note = "Use `compute()`, which also evaluates the Coulomb term and the forces"
    )]
    pub fn run_placeholder(&mut self) -> Result<(), MmError> {
        self.compute()
    }
}

// ---------------------------------------------------------------------------
// QmMmSystem
// ---------------------------------------------------------------------------

/// Combined QM/MM system.
///
/// [`compute`](Self::compute) evaluates the QM region with electrostatic
/// embedding in the MM point charges and fills the QM/MM coupling terms:
///
/// E_total = E_QM + E_MM + E_QM/MM,  E_QM/MM = E_elec + E_vdW
///
/// with E_QM = `qm.energy` (polarised QM energy), E_elec = `qm.embedding_energy`
/// and E_vdW the QM–MM Lennard-Jones energy.  E_MM (`mm.energy`) and the
/// MM–MM forces are the responsibility of the MM force field.
pub struct QmMmSystem {
    /// The QM region.
    pub qm: QmRegion,
    /// The MM region.
    pub mm: MmRegion,
    /// QM/MM interaction energy (kcal/mol): electrostatic embedding + van der Waals.
    pub energy_qm_mm: f64,
    /// QM–MM Lennard-Jones (van der Waals) part of `energy_qm_mm` (kcal/mol).
    pub energy_qm_mm_vdw: f64,
    /// Link atoms inserted at QM/MM boundaries.
    pub link_atoms: Vec<LinkAtom>,
    /// Total forces on the QM atoms (kcal/(mol·Å)): QM (including embedding and
    /// redistributed link-atom forces) plus QM–MM van der Waals.
    pub qm_total_forces: Vec<[f64; 3]>,
    /// QM/MM coupling forces on the MM atoms (kcal/(mol·Å)): electrostatic
    /// embedding, QM–MM van der Waals and redistributed link-atom forces.  Add
    /// these to the MM force-field forces.
    pub mm_coupling_forces: Vec<[f64; 3]>,
}

impl QmMmSystem {
    /// Create a new QM/MM system from QM and MM regions.
    pub fn new(qm: QmRegion, mm: MmRegion) -> Self {
        let n_qm = qm.atomic_numbers.len();
        let n_mm = mm.positions.len();
        Self {
            qm,
            mm,
            energy_qm_mm: 0.0,
            energy_qm_mm_vdw: 0.0,
            link_atoms: Vec::new(),
            qm_total_forces: vec![[0.0; 3]; n_qm],
            mm_coupling_forces: vec![[0.0; 3]; n_mm],
        }
    }

    /// Total energy: E_QM + E_MM + E_QM/MM.
    pub fn total_energy(&self) -> f64 {
        qm_mm_energy(self.qm.energy, self.mm.energy, self.energy_qm_mm)
    }

    /// Add a link atom at the QM/MM boundary.
    pub fn add_link_atom(&mut self, la: LinkAtom) {
        self.link_atoms.push(la);
    }

    /// Evaluate the QM region in the field of the MM point charges and the QM/MM
    /// coupling energy and forces.
    ///
    /// 1. Every link atom is placed at r_Q + g (r_M − r_Q) and added to the QM
    ///    calculation as a hydrogen atom.
    /// 2. The MM charges (except those of MM boundary atoms bonded to a link atom)
    ///    become `qm.external_charges`.
    /// 3. The QM engine is run; `qm.energy`, `qm.embedding_energy`, `qm.forces`,
    ///    `qm.partial_charges` and `qm.external_charge_forces` are filled.  The
    ///    link-atom force F_L is redistributed as (1 − g) F_L onto the QM and
    ///    g F_L onto the MM boundary atom.
    /// 4. The QM–MM Lennard-Jones energy is evaluated with Lorentz–Berthelot
    ///    combination of `qm.lj_epsilon/lj_sigma` and `mm.epsilon/sigma`, excluding
    ///    the bonded boundary pairs.
    /// 5. `energy_qm_mm = qm.embedding_energy + energy_qm_mm_vdw`.
    ///
    /// # Errors
    /// Returns a [`QmError`] if the input arrays are inconsistent, a link atom
    /// references a non-existent atom, or the QM calculation fails.  All outputs
    /// are cleared first, so a failure never leaves stale values behind.
    pub fn compute(&mut self) -> Result<(), QmError> {
        let n_qm = self.qm.atomic_numbers.len();
        let n_mm = self.mm.positions.len();
        self.energy_qm_mm = 0.0;
        self.energy_qm_mm_vdw = 0.0;
        self.qm_total_forces = vec![[0.0; 3]; n_qm];
        self.mm_coupling_forces = vec![[0.0; 3]; n_mm];
        self.validate_inputs(n_qm, n_mm)?;

        // 1. Link atoms.
        let mut boundary_mm = vec![false; n_mm];
        for la in &mut self.link_atoms {
            la.position = link_atom_correction(
                self.qm.positions[la.qm_atom],
                self.mm.positions[la.mm_atom],
                la.scale,
            );
            boundary_mm[la.mm_atom] = true;
        }

        // 2. External charges from the MM region (charge deletion at the boundary).
        let embedded_mm: Vec<usize> = (0..n_mm).filter(|&j| !boundary_mm[j]).collect();
        let external: Vec<([f64; 3], f64)> = embedded_mm
            .iter()
            .map(|&j| (self.mm.positions[j], self.mm.charges[j]))
            .collect();

        // 3. QM calculation on the link-atom-capped region.
        let mut work_indices = self.qm.atom_indices.clone();
        let mut work_numbers = self.qm.atomic_numbers.clone();
        let mut work_positions = self.qm.positions.clone();
        for la in &self.link_atoms {
            work_indices.push(
                self.mm
                    .atom_indices
                    .get(la.mm_atom)
                    .copied()
                    .unwrap_or(la.mm_atom),
            );
            work_numbers.push(1);
            work_positions.push(la.position);
        }
        let mut work = QmRegion::new(
            work_indices,
            self.qm.method,
            self.qm.charge,
            self.qm.multiplicity,
        );
        work.atomic_numbers = work_numbers;
        work.positions = work_positions;
        work.external_charges = external.clone();
        let run_result = work.run();

        self.qm.external_charges = external;
        self.qm.energy = work.energy;
        self.qm.embedding_energy = work.embedding_energy;
        self.qm.forces = work.forces.iter().take(n_qm).copied().collect();
        self.qm.forces.resize(n_qm, [0.0; 3]);
        self.qm.partial_charges = work.partial_charges.iter().take(n_qm).copied().collect();
        self.qm.partial_charges.resize(n_qm, 0.0);
        self.qm.external_charge_forces = work.external_charge_forces.clone();
        run_result?;

        for (k, &j) in embedded_mm.iter().enumerate() {
            if let Some(f) = work.external_charge_forces.get(k) {
                add3(&mut self.mm_coupling_forces[j], f, 1.0);
            }
        }
        for (l, la) in self.link_atoms.iter().enumerate() {
            if let Some(f_link) = work.forces.get(n_qm + l) {
                add3(&mut self.qm.forces[la.qm_atom], f_link, 1.0 - la.scale);
                add3(&mut self.mm_coupling_forces[la.mm_atom], f_link, la.scale);
            }
        }
        self.qm_total_forces = self.qm.forces.clone();

        // 4. QM–MM van der Waals.
        let mut e_vdw = 0.0_f64;
        for a in 0..n_qm {
            for j in 0..n_mm {
                if self
                    .link_atoms
                    .iter()
                    .any(|la| la.qm_atom == a && la.mm_atom == j)
                {
                    continue;
                }
                let eps = (self.qm.lj_epsilon[a] * self.mm.epsilon[j]).sqrt();
                if eps <= 0.0 {
                    continue;
                }
                let sigma = 0.5 * (self.qm.lj_sigma[a] + self.mm.sigma[j]);
                let d = [
                    self.qm.positions[a][0] - self.mm.positions[j][0],
                    self.qm.positions[a][1] - self.mm.positions[j][1],
                    self.qm.positions[a][2] - self.mm.positions[j][2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 < 1e-20 {
                    return Err(QmError::InvalidExternalCharge { index: j });
                }
                let sr2 = sigma * sigma / r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                e_vdw += 4.0 * eps * (sr12 - sr6);
                // F_a = −dE/dr_a = 24 ε (2 sr¹² − sr⁶) / r² · (r_a − r_j)
                let f_scale = 24.0 * eps * (2.0 * sr12 - sr6) / r2;
                add3(&mut self.qm_total_forces[a], &d, f_scale);
                add3(&mut self.mm_coupling_forces[j], &d, -f_scale);
            }
        }
        self.energy_qm_mm_vdw = e_vdw;

        // 5. Interaction energy.
        self.energy_qm_mm = self.qm.embedding_energy + e_vdw;
        Ok(())
    }

    /// Check the consistency of all per-atom arrays and link-atom indices.
    fn validate_inputs(&self, n_qm: usize, n_mm: usize) -> Result<(), QmError> {
        let checks: [(&'static str, usize, usize); 6] = [
            ("QM positions", n_qm, self.qm.positions.len()),
            ("QM lj_epsilon", n_qm, self.qm.lj_epsilon.len()),
            ("QM lj_sigma", n_qm, self.qm.lj_sigma.len()),
            ("MM charges", n_mm, self.mm.charges.len()),
            ("MM epsilon", n_mm, self.mm.epsilon.len()),
            ("MM sigma", n_mm, self.mm.sigma.len()),
        ];
        for (what, expected, found) in checks {
            if expected != found {
                return Err(QmError::DimensionMismatch {
                    what,
                    expected,
                    found,
                });
            }
        }
        for la in &self.link_atoms {
            if la.qm_atom >= n_qm {
                return Err(QmError::InvalidLinkAtom {
                    region: "QM",
                    index: la.qm_atom,
                    len: n_qm,
                });
            }
            if la.mm_atom >= n_mm {
                return Err(QmError::InvalidLinkAtom {
                    region: "MM",
                    index: la.mm_atom,
                    len: n_mm,
                });
            }
        }
        Ok(())
    }
}

/// `target += scale · v`.
fn add3(target: &mut [f64; 3], v: &[f64; 3], scale: f64) {
    for (t, x) in target.iter_mut().zip(v.iter()) {
        *t += scale * x;
    }
}

// ---------------------------------------------------------------------------
// LinkAtom
// ---------------------------------------------------------------------------

/// A hydrogen capping link atom at a QM/MM boundary bond.
///
/// The link atom (always hydrogen) is placed along the bond between a
/// QM atom and a MM atom, at a fixed distance g × r_{QM-MM}.
pub struct LinkAtom {
    /// QM atom index (in QM region) adjacent to the boundary.
    pub qm_atom: usize,
    /// MM atom index (in MM region) that the QM atom is bonded to.
    pub mm_atom: usize,
    /// Scale factor g ∈ (0, 1): link atom placed at g × r_{QM-MM} from QM atom.
    pub scale: f64,
    /// Position of the link atom (Å).
    pub position: [f64; 3],
}

impl LinkAtom {
    /// Create a new link atom.
    pub fn new(qm_atom: usize, mm_atom: usize, scale: f64) -> Self {
        Self {
            qm_atom,
            mm_atom,
            scale,
            position: [0.0; 3],
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Insert/update a hydrogen capping link atom at a QM/MM boundary bond.
///
/// The link atom H is placed along the Q1–M1 bond at:
/// r_H = r_Q1 + g × (r_M1 − r_Q1)
///
/// where g ≈ r_{Q-H} / r_{Q-M} ≈ 0.709 for C–C bonds.
///
/// # Arguments
/// * `pos_qm` — position of the QM boundary atom Q1 (Å)
/// * `pos_mm` — position of the MM boundary atom M1 (Å)
/// * `scale` — scale factor g
///
/// # Returns
/// Position of the link hydrogen atom.
pub fn link_atom_correction(pos_qm: [f64; 3], pos_mm: [f64; 3], scale: f64) -> [f64; 3] {
    [
        pos_qm[0] + scale * (pos_mm[0] - pos_qm[0]),
        pos_qm[1] + scale * (pos_mm[1] - pos_qm[1]),
        pos_qm[2] + scale * (pos_mm[2] - pos_qm[2]),
    ]
}

/// Classical point-charge Coulomb interaction between fixed QM atom charges and
/// MM point charges.
///
/// E_elec = Σ_{i ∈ QM} Σ_{j ∈ MM} q_i q_j / r_ij
///
/// The result is in e²/length of the input (no Coulomb constant or dielectric
/// applied); multiply by [`COULOMB_KCAL_PER_MOL_ANG`] for kcal/mol with Å and e.
/// This does **not** polarise the QM wavefunction — for true electrostatic
/// embedding set [`QmRegion::external_charges`] or use [`QmMmSystem::compute`].
///
/// # Arguments
/// * `qm_positions` — positions of QM atoms
/// * `qm_charges` — partial charges of QM atoms
/// * `mm_positions` — positions of MM atoms
/// * `mm_charges` — partial charges of MM atoms
pub fn electrostatic_embedding(
    qm_positions: &[[f64; 3]],
    qm_charges: &[f64],
    mm_positions: &[[f64; 3]],
    mm_charges: &[f64],
) -> f64 {
    let n_qm = qm_positions.len().min(qm_charges.len());
    let n_mm = mm_positions.len().min(mm_charges.len());
    let mut energy = 0.0;
    for i in 0..n_qm {
        for j in 0..n_mm {
            let dx = qm_positions[i][0] - mm_positions[j][0];
            let dy = qm_positions[i][1] - mm_positions[j][1];
            let dz = qm_positions[i][2] - mm_positions[j][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 {
                continue;
            }
            energy += qm_charges[i] * mm_charges[j] / r;
        }
    }
    energy
}

/// Fixed classical parameters of a set of atoms: the argument type of
/// [`mechanical_embedding`].
///
/// All slices must have one entry per atom, i.e. `positions.len()` entries.
#[derive(Debug, Clone, Copy)]
pub struct ClassicalAtoms<'a> {
    /// Atom positions in Å.
    pub positions: &'a [[f64; 3]],
    /// Fixed partial charges in units of e.
    pub charges: &'a [f64],
    /// Lennard-Jones well depths ε in kcal/mol.
    pub epsilon: &'a [f64],
    /// Lennard-Jones diameters σ in Å.
    pub sigma: &'a [f64],
}

/// Classical QM/MM interaction energy of the mechanical-embedding scheme, in kcal/mol.
///
/// Every QM atom `i` interacts with every MM atom `j` through a bare Coulomb term
/// between the *fixed* charges and a Lennard-Jones 12-6 term,
///
/// E = Σ_i Σ_j [ k_e q_i Q_j / r_ij + 4 ε_ij ((σ_ij/r_ij)¹² − (σ_ij/r_ij)⁶) ],
///
/// with k_e = [`COULOMB_KCAL_PER_MOL_ANG`] (kcal·Å/(mol·e²)) and the
/// Lorentz–Berthelot combination ε_ij = √(ε_i ε_j), σ_ij = (σ_i + σ_j)/2 of the
/// per-atom parameters of both regions.  The QM wavefunction is not polarised by
/// the MM charges (use [`QmMmSystem::compute`] for electrostatic embedding), no
/// pairs are excluded (not even bonded boundary pairs) and only the energy is
/// returned, no forces.
///
/// # Errors
/// [`QmError::DimensionMismatch`] if a charge, ε or σ slice of either region does
/// not have one entry per position, and [`QmError::InvalidExternalCharge`] if an
/// MM atom coincides with a QM atom (the interaction is singular).
pub fn mechanical_embedding(
    qm: ClassicalAtoms<'_>,
    mm: ClassicalAtoms<'_>,
) -> Result<f64, QmError> {
    for (what, expected, found) in [
        ("QM charges", qm.positions.len(), qm.charges.len()),
        ("QM lj_epsilon", qm.positions.len(), qm.epsilon.len()),
        ("QM lj_sigma", qm.positions.len(), qm.sigma.len()),
        ("MM charges", mm.positions.len(), mm.charges.len()),
        ("MM epsilon", mm.positions.len(), mm.epsilon.len()),
        ("MM sigma", mm.positions.len(), mm.sigma.len()),
    ] {
        if expected != found {
            return Err(QmError::DimensionMismatch {
                what,
                expected,
                found,
            });
        }
    }
    let mut energy = 0.0;
    for (i, pos_i) in qm.positions.iter().enumerate() {
        for (j, pos_j) in mm.positions.iter().enumerate() {
            let dx = pos_i[0] - pos_j[0];
            let dy = pos_i[1] - pos_j[1];
            let dz = pos_i[2] - pos_j[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-20 {
                return Err(QmError::InvalidExternalCharge { index: j });
            }
            let eps = (qm.epsilon[i] * mm.epsilon[j]).sqrt();
            let sigma = 0.5 * (qm.sigma[i] + mm.sigma[j]);
            let sr2 = sigma * sigma / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += COULOMB_KCAL_PER_MOL_ANG * qm.charges[i] * mm.charges[j] / r2.sqrt()
                + 4.0 * eps * (sr12 - sr6);
        }
    }
    Ok(energy)
}

/// Total QM/MM energy: E_total = E_QM + E_MM + E_QM/MM.
///
/// # Arguments
/// * `e_qm` — QM energy
/// * `e_mm` — MM energy
/// * `e_qm_mm` — QM/MM coupling energy
pub fn qm_mm_energy(e_qm: f64, e_mm: f64, e_qm_mm: f64) -> f64 {
    e_qm + e_mm + e_qm_mm
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    // ── QmMethod ──────────────────────────────────────────────────────────

    #[test]
    fn test_qm_method_name() {
        assert_eq!(QmMethod::Pm3.name(), "PM3");
        assert_eq!(QmMethod::SccDftb.name(), "SCC-DFTB");
        assert_eq!(QmMethod::Hf.name(), "HF");
        assert_eq!(QmMethod::DftB3lyp.name(), "DFT/B3LYP");
    }

    #[test]
    fn test_qm_method_eq() {
        assert_eq!(QmMethod::Pm3, QmMethod::Pm3);
        assert_ne!(QmMethod::Pm3, QmMethod::Hf);
    }

    #[test]
    fn test_qm_method_supports_element() {
        for z in [1, 6, 7, 8, 9, 16] {
            assert!(QmMethod::Pm3.supports_element(z), "PM3 Z={z}");
        }
        for z in [0, 2, 3, 10, 92] {
            assert!(!QmMethod::Pm3.supports_element(z), "PM3 Z={z}");
        }
        for z in [1, 6, 7, 8, 16] {
            assert!(QmMethod::SccDftb.supports_element(z), "SCC-DFTB Z={z}");
        }
        assert!(!QmMethod::SccDftb.supports_element(9));
        for method in [QmMethod::Hf, QmMethod::DftB3lyp] {
            for z in [1, 6, 8, 18] {
                assert!(method.supports_element(z), "{method:?} Z={z}");
            }
            for z in [0, 36, 92] {
                assert!(!method.supports_element(z), "{method:?} Z={z}");
            }
        }
    }

    // ── QmRegion ──────────────────────────────────────────────────────────

    #[test]
    fn test_qm_region_new() {
        let qm = QmRegion::new(vec![0, 1, 2], QmMethod::Pm3, 0, 1);
        assert_eq!(qm.n_atoms(), 3);
        assert_eq!(qm.charge, 0);
        assert_eq!(qm.multiplicity, 1);
    }

    #[test]
    fn test_qm_region_try_new_accepts_a_valid_region() {
        let region = QmRegion::try_new(
            vec![4, 7],
            QmMethod::Pm3,
            0,
            1,
            vec![1, 1],
            vec![[0.0; 3], [0.74, 0.0, 0.0]],
        )
        .expect("valid region");
        assert_eq!(region.atom_indices, vec![4, 7]);
        assert_eq!(region.atomic_numbers, vec![1, 1]);
        assert_eq!(region.positions[1], [0.74, 0.0, 0.0]);
        assert_eq!(region.forces.len(), 2);
        assert_eq!(region.n_atoms(), 2);
    }

    #[test]
    fn test_qm_region_try_new_rejects_invalid_regions() {
        let two = || vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]];
        let try_new = |indices: Vec<usize>, method, numbers: Vec<u32>, positions| {
            QmRegion::try_new(indices, method, 0, 1, numbers, positions).err()
        };
        assert_eq!(
            try_new(vec![], QmMethod::Pm3, vec![], vec![]),
            Some(QmError::EmptyRegion)
        );
        assert_eq!(
            try_new(vec![0, 1], QmMethod::Pm3, vec![1], two()),
            Some(QmError::DimensionMismatch {
                what: "atomic numbers vs atom indices",
                expected: 2,
                found: 1
            })
        );
        assert_eq!(
            try_new(vec![0, 1], QmMethod::Pm3, vec![1, 1], vec![[0.0; 3]]),
            Some(QmError::DimensionMismatch {
                what: "positions vs atom indices",
                expected: 2,
                found: 1
            })
        );
        // Every method reports its own unsupported elements.
        assert_eq!(
            try_new(vec![0, 1], QmMethod::Pm3, vec![1, 15], two()),
            Some(QmError::UnsupportedElement(15))
        );
        assert_eq!(
            try_new(vec![0, 1], QmMethod::SccDftb, vec![9, 1], two()),
            Some(QmError::UnsupportedElement(9))
        );
        assert_eq!(
            try_new(vec![0, 1], QmMethod::Hf, vec![1, 36], two()),
            Some(QmError::UnsupportedElement(36))
        );
        // The placeholder geometry of `QmRegion::new` is rejected as well.
        assert_eq!(
            try_new(
                vec![0, 1],
                QmMethod::DftB3lyp,
                vec![1, 1],
                vec![[0.0; 3]; 2]
            ),
            Some(QmError::CoincidentAtoms {
                first: 0,
                second: 1
            })
        );
    }

    #[test]
    fn test_unpopulated_region_is_rejected_by_every_engine() {
        // `QmRegion::new` leaves every atom a hydrogen at the origin; no engine may
        // silently compute that.
        for method in [
            QmMethod::Pm3,
            QmMethod::SccDftb,
            QmMethod::Hf,
            QmMethod::DftB3lyp,
        ] {
            let mut qm = QmRegion::new(vec![0, 1], method, 0, 1);
            assert_eq!(
                qm.run(),
                Err(QmError::CoincidentAtoms {
                    first: 0,
                    second: 1
                }),
                "{method:?}"
            );
            assert_eq!(qm.energy, 0.0, "{method:?}");
        }
    }

    #[test]
    fn test_qm_region_run_pm3_h2() {
        // PM3: H2 at 0.74 Å is a bound molecule.
        let mut qm = QmRegion::new(vec![0, 1], QmMethod::Pm3, 0, 1);
        qm.atomic_numbers = vec![1, 1];
        qm.set_positions(vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]);
        qm.run().expect("PM3 H2");
        assert!(
            qm.energy.is_finite() && qm.energy < 0.0,
            "PM3 H2: {}",
            qm.energy
        );
    }

    #[test]
    fn test_qm_region_run_hf_single_h_is_open_shell_error() {
        // A single H atom has one electron: the closed-shell HF engine must
        // report it instead of returning a meaningless energy.
        let mut qm = QmRegion::new(vec![0], QmMethod::Hf, 0, 1);
        qm.atomic_numbers = vec![1];
        qm.set_positions(vec![[0.0, 0.0, 0.0]]);
        assert_eq!(qm.run(), Err(QmError::OpenShellNotSupported));
        assert_eq!(qm.energy, 0.0);
    }

    #[test]
    fn test_qm_region_run_dftb_h2() {
        let mut qm = QmRegion::new(vec![0, 1], QmMethod::SccDftb, 0, 1);
        qm.atomic_numbers = vec![1, 1];
        qm.set_positions(vec![[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]);
        qm.run().expect("DFTB H2");
        assert!(
            qm.energy.is_finite() && qm.energy < 0.0,
            "DFTB H2: {}",
            qm.energy
        );
    }

    #[test]
    fn test_qm_region_run_unsupported_element_is_error_for_every_method() {
        for method in [
            QmMethod::Pm3,
            QmMethod::SccDftb,
            QmMethod::Hf,
            QmMethod::DftB3lyp,
        ] {
            let mut qm = QmRegion::new(vec![0, 1], method, 0, 1);
            qm.atomic_numbers = vec![92, 92];
            qm.set_positions(vec![[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]);
            qm.energy = 42.0;
            qm.forces = vec![[1.0; 3]; 2];
            assert_eq!(qm.run(), Err(QmError::UnsupportedElement(92)), "{method:?}");
            assert_eq!(qm.energy, 0.0, "{method:?} left a stale energy");
            assert!(
                qm.forces.iter().all(|f| *f == [0.0; 3]),
                "{method:?} stale forces"
            );
        }
    }

    #[test]
    fn test_qm_region_empty_and_inconsistent_inputs() {
        let mut empty = QmRegion::new(vec![], QmMethod::Pm3, 0, 1);
        assert_eq!(empty.run(), Err(QmError::EmptyRegion));
        let mut bad = QmRegion::new(vec![0, 1], QmMethod::SccDftb, 0, 1);
        bad.positions = vec![[0.0; 3]];
        assert!(matches!(bad.run(), Err(QmError::DimensionMismatch { .. })));
        let mut on_top = QmRegion::new(vec![0, 1], QmMethod::Pm3, 0, 1);
        on_top.set_positions(vec![[0.0; 3], [0.74, 0.0, 0.0]]);
        on_top.external_charges = vec![([0.74, 0.0, 0.0], 0.5)];
        assert_eq!(
            on_top.run(),
            Err(QmError::InvalidExternalCharge { index: 0 })
        );
    }

    // ── MmRegion ──────────────────────────────────────────────────────────

    #[test]
    fn test_mm_region_new() {
        let mm = MmRegion::new(vec![3, 4, 5]);
        assert_eq!(mm.n_atoms(), 3);
    }

    /// Two-atom MM region with the given parameters.
    fn mm_pair(r: f64, charges: [f64; 2], epsilon: [f64; 2], sigma: [f64; 2]) -> MmRegion {
        let mut mm = MmRegion::new(vec![0, 1]);
        mm.positions = vec![[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
        mm.charges = charges.to_vec();
        mm.epsilon = epsilon.to_vec();
        mm.sigma = sigma.to_vec();
        mm
    }

    #[test]
    fn test_mm_region_compute_lennard_jones_minimum() {
        // The LJ minimum lies at r = 2^(1/6) sigma with depth -epsilon and zero force.
        let r_min = 2.0_f64.powf(1.0 / 6.0) * 3.0;
        let mut mm = mm_pair(r_min, [0.0; 2], [0.2; 2], [3.0; 2]);
        mm.compute().expect("LJ pair");
        assert!((mm.energy + 0.2).abs() < 1e-12, "E = {}", mm.energy);
        for f in &mm.forces {
            assert!(f.iter().all(|x| x.abs() < 1e-10), "force {f:?}");
        }
        // Inside the minimum the pair repels: atom 0 is pushed towards -x.
        let mut close = mm_pair(0.9 * r_min, [0.0; 2], [0.2; 2], [3.0; 2]);
        close.compute().expect("compressed LJ pair");
        assert!(close.energy > -0.2 + 1e-3);
        assert!(close.forces[0][0] < 0.0 && close.forces[1][0] > 0.0);
    }

    #[test]
    fn test_mm_region_compute_coulomb_uses_the_kcal_constant() {
        // Opposite unit charges 2 Å apart, no LJ: E = -k_e / 2, attractive forces.
        let mut mm = mm_pair(2.0, [1.0, -1.0], [0.0; 2], [3.0; 2]);
        mm.compute().expect("Coulomb pair");
        let expected = -COULOMB_KCAL_PER_MOL_ANG / 2.0;
        assert!((mm.energy - expected).abs() < 1e-9, "E = {}", mm.energy);
        // |F| = k_e / r^2, atom 0 is pulled towards +x, atom 1 towards -x.
        let f = COULOMB_KCAL_PER_MOL_ANG / 4.0;
        assert!((mm.forces[0][0] - f).abs() < 1e-9);
        assert!((mm.forces[1][0] + f).abs() < 1e-9);
        assert!(mm.forces[0][1].abs() < 1e-12 && mm.forces[0][2].abs() < 1e-12);
    }

    #[test]
    fn test_mm_region_compute_forces_match_finite_differences() {
        // Two charged LJ atoms (equal and opposite forces matching the central
        // finite difference of the energy to 1e-6), then a three-atom cluster.
        let energy_at = |positions: &[[f64; 3]]| {
            let mut mm = MmRegion::new(vec![0, 1, 2]);
            mm.positions = positions.to_vec();
            mm.charges = vec![0.417, -0.834, 0.417];
            mm.epsilon = vec![0.046, 0.1521, 0.046];
            mm.sigma = vec![0.4, 3.1507, 0.4];
            mm.compute().expect("cluster");
            mm.energy
        };
        let base = [[0.0, 0.1, -0.2], [3.1, 0.4, 0.3], [1.2, 2.9, -0.5]];
        let mut mm = MmRegion::new(vec![0, 1, 2]);
        mm.positions = base.to_vec();
        mm.charges = vec![0.417, -0.834, 0.417];
        mm.epsilon = vec![0.046, 0.1521, 0.046];
        mm.sigma = vec![0.4, 3.1507, 0.4];
        mm.compute().expect("cluster");
        let h = 1e-4;
        for atom in 0..3 {
            for k in 0..3 {
                let mut plus = base;
                plus[atom][k] += h;
                let mut minus = base;
                minus[atom][k] -= h;
                let fd = -(energy_at(&plus) - energy_at(&minus)) / (2.0 * h);
                assert!(
                    (fd - mm.forces[atom][k]).abs() < 1e-6,
                    "atom {atom} component {k}: {} vs FD {fd}",
                    mm.forces[atom][k]
                );
            }
        }
        // Newton's third law.
        for k in 0..3 {
            let total: f64 = mm.forces.iter().map(|f| f[k]).sum();
            assert!(total.abs() < 1e-12, "net force {k}: {total}");
        }
        // Two charged atoms: equal and opposite forces.
        let mut pair = mm_pair(3.4, [0.5, -0.3], [0.1, 0.2], [3.5, 3.0]);
        pair.compute().expect("pair");
        for k in 0..3 {
            assert!((pair.forces[0][k] + pair.forces[1][k]).abs() < 1e-12);
        }
        assert!(pair.forces[0][0].abs() > 1e-3);
    }

    #[test]
    fn test_mm_region_compute_single_atom_is_zero() {
        let mut mm = MmRegion::new(vec![0]);
        mm.positions = vec![[0.0, 0.0, 0.0]];
        mm.compute().expect("single atom");
        assert_eq!(mm.energy, 0.0);
        assert_eq!(mm.forces, vec![[0.0; 3]]);
    }

    #[test]
    fn test_mm_region_compute_rejects_invalid_input_and_clears_outputs() {
        let mut mm = mm_pair(3.5, [0.1, -0.1], [0.1; 2], [3.5; 2]);
        mm.compute().expect("valid pair");
        assert!(mm.energy != 0.0);

        // A length mismatch used to panic in the placeholder.
        mm.sigma.pop();
        assert_eq!(
            mm.compute(),
            Err(MmError::LengthMismatch {
                what: "sigma",
                expected: 2,
                found: 1
            })
        );
        assert_eq!(mm.energy, 0.0, "stale energy");
        assert!(mm.forces.iter().all(|f| *f == [0.0; 3]), "stale forces");

        mm.sigma.push(3.5);
        mm.charges.push(0.0);
        assert_eq!(
            mm.compute(),
            Err(MmError::LengthMismatch {
                what: "charges",
                expected: 2,
                found: 3
            })
        );
        mm.charges.pop();

        mm.epsilon[1] = -0.1;
        assert_eq!(
            mm.compute(),
            Err(MmError::InvalidParameter {
                what: "Lennard-Jones epsilon",
                index: 1
            })
        );
        mm.epsilon[1] = 0.1;
        mm.positions[0][2] = f64::NAN;
        assert_eq!(
            mm.compute(),
            Err(MmError::InvalidParameter {
                what: "position",
                index: 0
            })
        );
        mm.positions[0] = [0.0; 3];
        mm.positions[1] = [0.0; 3];
        assert_eq!(
            mm.compute(),
            Err(MmError::CoincidentAtoms {
                first: 0,
                second: 1
            })
        );
    }

    // ── Link atom correction ──────────────────────────────────────────────

    #[test]
    fn test_link_atom_scale_zero() {
        // scale=0 → link atom at QM atom position
        let pos_qm = [1.0, 2.0, 3.0];
        let pos_mm = [5.0, 6.0, 7.0];
        let la = link_atom_correction(pos_qm, pos_mm, 0.0);
        for k in 0..3 {
            assert!((la[k] - pos_qm[k]).abs() < EPS);
        }
    }

    #[test]
    fn test_link_atom_scale_one() {
        // scale=1 → link atom at MM atom position
        let pos_qm = [1.0, 0.0, 0.0];
        let pos_mm = [3.0, 0.0, 0.0];
        let la = link_atom_correction(pos_qm, pos_mm, 1.0);
        assert!((la[0] - 3.0).abs() < EPS);
    }

    #[test]
    fn test_link_atom_scale_half() {
        // scale=0.5 → midpoint
        let pos_qm = [0.0, 0.0, 0.0];
        let pos_mm = [2.0, 0.0, 0.0];
        let la = link_atom_correction(pos_qm, pos_mm, 0.5);
        assert!((la[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_link_atom_typical_scale() {
        // Typical C-H / C-C scale ≈ 0.709
        let pos_qm = [0.0, 0.0, 0.0];
        let pos_mm = [1.0, 0.0, 0.0];
        let la = link_atom_correction(pos_qm, pos_mm, 0.709);
        assert!((la[0] - 0.709).abs() < EPS);
    }

    // ── Electrostatic embedding ───────────────────────────────────────────

    #[test]
    fn test_electrostatic_embedding_single_pair() {
        // Two unit charges at distance 1.0 → E = 1.0
        let qm_pos = vec![[0.0, 0.0, 0.0]];
        let qm_chg = vec![1.0];
        let mm_pos = vec![[1.0, 0.0, 0.0]];
        let mm_chg = vec![1.0];
        let e = electrostatic_embedding(&qm_pos, &qm_chg, &mm_pos, &mm_chg);
        assert!((e - 1.0).abs() < EPS);
    }

    #[test]
    fn test_electrostatic_embedding_opposite_charges() {
        let qm_pos = vec![[0.0, 0.0, 0.0]];
        let qm_chg = vec![1.0];
        let mm_pos = vec![[1.0, 0.0, 0.0]];
        let mm_chg = vec![-1.0];
        let e = electrostatic_embedding(&qm_pos, &qm_chg, &mm_pos, &mm_chg);
        assert!((e + 1.0).abs() < EPS);
    }

    #[test]
    fn test_electrostatic_embedding_zero_charges() {
        let qm_pos = vec![[0.0, 0.0, 0.0]];
        let qm_chg = vec![0.0];
        let mm_pos = vec![[1.0, 0.0, 0.0]];
        let mm_chg = vec![1.0];
        let e = electrostatic_embedding(&qm_pos, &qm_chg, &mm_pos, &mm_chg);
        assert!(e.abs() < EPS);
    }

    #[test]
    fn test_electrostatic_embedding_inverse_distance() {
        let qm_pos = vec![[0.0, 0.0, 0.0]];
        let qm_chg = vec![1.0];
        let mm_pos1 = vec![[1.0, 0.0, 0.0]];
        let mm_pos2 = vec![[2.0, 0.0, 0.0]];
        let mm_chg = vec![1.0];
        let e1 = electrostatic_embedding(&qm_pos, &qm_chg, &mm_pos1, &mm_chg);
        let e2 = electrostatic_embedding(&qm_pos, &qm_chg, &mm_pos2, &mm_chg);
        assert!((e2 * 2.0 - e1).abs() < EPS);
    }

    // ── Mechanical embedding ──────────────────────────────────────────────

    fn classical<'a>(
        positions: &'a [[f64; 3]],
        charges: &'a [f64],
        epsilon: &'a [f64],
        sigma: &'a [f64],
    ) -> ClassicalAtoms<'a> {
        ClassicalAtoms {
            positions,
            charges,
            epsilon,
            sigma,
        }
    }

    #[test]
    fn test_mechanical_embedding_identical_parameters_reproduce_the_mm_pair_energy() {
        let (q, eps, sig, r) = (0.4, 0.15, 3.2, 3.5);
        let mut mm = mm_pair(r, [q, q], [eps, eps], [sig, sig]);
        mm.compute().expect("MM pair");
        let qm_pos = [[0.0, 0.0, 0.0]];
        let mm_pos = [[r, 0.0, 0.0]];
        let e = mechanical_embedding(
            classical(&qm_pos, &[q], &[eps], &[sig]),
            classical(&mm_pos, &[q], &[eps], &[sig]),
        )
        .expect("mechanical embedding");
        assert!((e - mm.energy).abs() < 1e-12, "{e} vs {}", mm.energy);
        // Both terms are present: Coulomb is repulsive for equal charges and the
        // pair sits near the LJ minimum.
        assert!(e.abs() > 1e-3);
    }

    #[test]
    fn test_mechanical_embedding_coulomb_uses_the_kcal_constant() {
        // No LJ (epsilon = 0 on the QM atom): E = k_e q Q / r.
        let e = mechanical_embedding(
            classical(&[[0.0; 3]], &[0.5], &[0.0], &[3.0]),
            classical(&[[5.0, 0.0, 0.0]], &[-0.5], &[0.1], &[3.5]),
        )
        .expect("mechanical embedding");
        assert!(
            (e + COULOMB_KCAL_PER_MOL_ANG * 0.25 / 5.0).abs() < 1e-12,
            "E = {e}"
        );
    }

    #[test]
    fn test_mechanical_embedding_uses_lorentz_berthelot_mixing_of_both_regions() {
        // eps = sqrt(0.04 * 0.16) = 0.08, sigma = (2 + 4) / 2 = 3: the minimum of the
        // mixed potential at r = 2^(1/6) * 3 has depth -0.08.
        let r_min = 2.0_f64.powf(1.0 / 6.0) * 3.0;
        let e = mechanical_embedding(
            classical(&[[0.0; 3]], &[0.0], &[0.04], &[2.0]),
            classical(&[[0.0, r_min, 0.0]], &[0.0], &[0.16], &[4.0]),
        )
        .expect("mechanical embedding");
        assert!((e + 0.08).abs() < 1e-12, "E = {e}");
    }

    #[test]
    fn test_mechanical_embedding_sums_all_qm_mm_pairs() {
        let qm_pos = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mm_pos = [[4.0, 0.0, 0.0], [0.0, 4.0, 1.0], [-3.0, 2.0, 0.5]];
        let qm_q = [0.3, -0.2];
        let mm_q = [0.1, -0.4, 0.25];
        let qm_eps = [0.05, 0.12];
        let mm_eps = [0.1, 0.2, 0.15];
        let qm_sig = [3.0, 3.4];
        let mm_sig = [3.5, 3.1, 3.3];
        let total = mechanical_embedding(
            classical(&qm_pos, &qm_q, &qm_eps, &qm_sig),
            classical(&mm_pos, &mm_q, &mm_eps, &mm_sig),
        )
        .expect("mechanical embedding");
        let mut expected = 0.0;
        for i in 0..2 {
            for j in 0..3 {
                expected += mechanical_embedding(
                    classical(&qm_pos[i..=i], &qm_q[i..=i], &qm_eps[i..=i], &qm_sig[i..=i]),
                    classical(&mm_pos[j..=j], &mm_q[j..=j], &mm_eps[j..=j], &mm_sig[j..=j]),
                )
                .expect("pair");
            }
        }
        assert!((total - expected).abs() < 1e-12, "{total} vs {expected}");
        assert!(total.is_finite() && total != 0.0);
    }

    #[test]
    fn test_mechanical_embedding_rejects_inconsistent_and_singular_input() {
        let pos = [[0.0; 3], [1.5, 0.0, 0.0]];
        let mm_pos = [[4.0, 0.0, 0.0]];
        // The old implementation silently truncated to the shorter slice.
        assert_eq!(
            mechanical_embedding(
                classical(&pos, &[0.1], &[0.1, 0.1], &[3.0, 3.0]),
                classical(&mm_pos, &[0.1], &[0.1], &[3.5]),
            ),
            Err(QmError::DimensionMismatch {
                what: "QM charges",
                expected: 2,
                found: 1
            })
        );
        assert_eq!(
            mechanical_embedding(
                classical(&pos, &[0.1, 0.1], &[0.1, 0.1], &[3.0, 3.0]),
                classical(&mm_pos, &[0.1], &[], &[3.5]),
            ),
            Err(QmError::DimensionMismatch {
                what: "MM epsilon",
                expected: 1,
                found: 0
            })
        );
        // An MM atom on top of a QM atom is a singular interaction.
        assert_eq!(
            mechanical_embedding(
                classical(&pos, &[0.1, 0.1], &[0.1, 0.1], &[3.0, 3.0]),
                classical(&[[1.5, 0.0, 0.0]], &[0.1], &[0.1], &[3.5]),
            ),
            Err(QmError::InvalidExternalCharge { index: 0 })
        );
    }

    // ── qm_mm_energy ──────────────────────────────────────────────────────

    #[test]
    fn test_qm_mm_energy_additive() {
        let e = qm_mm_energy(10.0, 5.0, 2.0);
        assert!((e - 17.0).abs() < EPS);
    }

    #[test]
    fn test_qm_mm_energy_zero() {
        assert!(qm_mm_energy(0.0, 0.0, 0.0).abs() < EPS);
    }

    #[test]
    fn test_qm_mm_energy_negative_interaction() {
        let e = qm_mm_energy(10.0, 5.0, -3.0);
        assert!((e - 12.0).abs() < EPS);
    }

    // ── QmMmSystem ────────────────────────────────────────────────────────

    #[test]
    fn test_qm_mm_system_total_energy() {
        let mut qm = QmRegion::new(vec![0], QmMethod::Pm3, 0, 1);
        qm.energy = 5.0;
        let mut mm = MmRegion::new(vec![1]);
        mm.energy = 3.0;
        let mut sys = QmMmSystem::new(qm, mm);
        sys.energy_qm_mm = 1.0;
        assert!((sys.total_energy() - 9.0).abs() < EPS);
    }

    /// QM water (SCC-DFTB) next to an MM TIP3P-like water.
    fn water_dimer_system() -> QmMmSystem {
        let mut qm = QmRegion::new(vec![0, 1, 2], QmMethod::SccDftb, 0, 1);
        qm.atomic_numbers = vec![8, 1, 1];
        qm.set_positions(vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ]);
        qm.lj_epsilon = vec![0.1521, 0.0, 0.0];
        qm.lj_sigma = vec![3.1507, 0.0, 0.0];
        let mut mm = MmRegion::new(vec![3, 4, 5]);
        mm.positions = vec![
            [0.2, 0.1, -2.9],
            [0.2, 0.857, -3.486],
            [0.2, -0.657, -3.486],
        ];
        mm.charges = vec![-0.834, 0.417, 0.417];
        mm.epsilon = vec![0.1521, 0.0, 0.0];
        mm.sigma = vec![3.1507, 0.0, 0.0];
        mm.energy = -1.25;
        QmMmSystem::new(qm, mm)
    }

    /// E_QM + E_QM/MM (the MM–MM energy is held fixed by these tests).
    fn coupled_energy(sys: &QmMmSystem) -> f64 {
        sys.qm.energy + sys.energy_qm_mm
    }

    #[test]
    fn test_qm_mm_compute_energy_decomposition() {
        let mut sys = water_dimer_system();
        sys.compute().expect("QM/MM water dimer");
        // MM charges were embedded and polarise the QM region.
        assert_eq!(sys.qm.external_charges.len(), 3);
        assert!(sys.qm.embedding_energy.abs() > 1e-3);
        assert!(sys.energy_qm_mm_vdw.abs() > 1e-6);
        assert!(
            (sys.energy_qm_mm - (sys.qm.embedding_energy + sys.energy_qm_mm_vdw)).abs() < 1e-12
        );
        // No double counting: the embedding energy is not part of qm.energy.
        assert!(
            (sys.total_energy() - (sys.qm.energy + sys.mm.energy + sys.energy_qm_mm)).abs() < 1e-12
        );
        let mut vacuum = water_dimer_system().qm;
        vacuum.run().expect("vacuum");
        // SCC-DFTB is variational, so polarising the QM water costs internal
        // energy, which the embedding interaction more than pays back.
        assert!(sys.qm.energy > vacuum.energy);
        // The dipole of the QM water changes in the field of the MM water.
        let mu_vac = vacuum.dipole_moment();
        let mu_emb = sys.qm.dipole_moment();
        let diff: f64 = (0..3).map(|k| (mu_vac[k] - mu_emb[k]).powi(2)).sum::<f64>();
        assert!(diff.sqrt() > 1e-3);
    }

    #[test]
    fn test_qm_mm_compute_forces_match_finite_differences() {
        let mut sys = water_dimer_system();
        sys.compute().expect("QM/MM water dimer");
        let h = 1e-3;
        // MM oxygen (embedding + van der Waals) and an MM hydrogen (embedding only).
        for (j, k) in [(0usize, 2usize), (0, 0), (1, 1)] {
            let mut e = [0.0; 2];
            for (slot, sign) in e.iter_mut().zip([1.0, -1.0]) {
                let mut d = water_dimer_system();
                d.mm.positions[j][k] += sign * h;
                d.compute().expect("displaced MM atom");
                *slot = coupled_energy(&d);
            }
            let fd = -(e[0] - e[1]) / (2.0 * h);
            let f = sys.mm_coupling_forces[j][k];
            assert!(
                (fd - f).abs() < 5e-3 + 1e-2 * fd.abs(),
                "MM atom {j} component {k}: {f} vs FD {fd}"
            );
        }
        // QM oxygen.
        for k in 0..3 {
            let mut e = [0.0; 2];
            for (slot, sign) in e.iter_mut().zip([1.0, -1.0]) {
                let mut d = water_dimer_system();
                d.qm.positions[0][k] += sign * h;
                d.compute().expect("displaced QM atom");
                *slot = coupled_energy(&d);
            }
            let fd = -(e[0] - e[1]) / (2.0 * h);
            let f = sys.qm_total_forces[0][k];
            assert!(
                (fd - f).abs() < 2e-2 + 1e-2 * fd.abs(),
                "QM O component {k}: {f} vs FD {fd}"
            );
        }
        // Newton's third law over both regions.
        for k in 0..3 {
            let total: f64 = sys.qm_total_forces.iter().map(|f| f[k]).sum::<f64>()
                + sys.mm_coupling_forces.iter().map(|f| f[k]).sum::<f64>();
            assert!(total.abs() < 2e-2, "net force {k}: {total}");
        }
    }

    /// QM methyl group (PM3) bonded to an MM methyl group through a link atom.
    fn ethane_link_system() -> QmMmSystem {
        let b = 1.09;
        let tet = [
            [1.0 / 3.0, 0.942_809, 0.0],
            [1.0 / 3.0, -0.471_405, 0.816_497],
            [1.0 / 3.0, -0.471_405, -0.816_497],
        ];
        let c_mm = [1.53, 0.0, 0.0];
        let mut qm = QmRegion::new(vec![0, 1, 2, 3], QmMethod::Pm3, 0, 1);
        qm.atomic_numbers = vec![6, 1, 1, 1];
        let mut qm_pos = vec![[0.0; 3]];
        for t in &tet {
            qm_pos.push([-b * t[0], b * t[1], b * t[2]]);
        }
        qm.set_positions(qm_pos);
        let mut mm = MmRegion::new(vec![4, 5, 6, 7]);
        let mut mm_pos = vec![c_mm];
        for t in &tet {
            mm_pos.push([c_mm[0] + b * t[0], b * t[1] + 0.05, b * t[2]]);
        }
        mm.positions = mm_pos;
        mm.charges = vec![-0.27, 0.09, 0.09, 0.09];
        mm.epsilon = vec![0.0; 4];
        mm.sigma = vec![0.0; 4];
        let mut sys = QmMmSystem::new(qm, mm);
        sys.add_link_atom(LinkAtom::new(0, 0, 0.709));
        sys
    }

    #[test]
    fn test_qm_mm_compute_with_link_atom() {
        let mut sys = ethane_link_system();
        sys.compute().expect("QM/MM ethane");
        // Link hydrogen placed along the C–C bond; boundary MM charge deleted.
        assert!((sys.link_atoms[0].position[0] - 0.709 * 1.53).abs() < 1e-12);
        assert_eq!(sys.qm.external_charges.len(), 3);
        assert_eq!(sys.qm.forces.len(), 4);
        // The MM boundary carbon only feels the redistributed link-atom force.
        let h = 1e-3;
        for k in 0..3 {
            let mut e = [0.0; 2];
            for (slot, sign) in e.iter_mut().zip([1.0, -1.0]) {
                let mut d = ethane_link_system();
                d.mm.positions[0][k] += sign * h;
                d.compute().expect("displaced boundary atom");
                *slot = coupled_energy(&d);
            }
            let fd = -(e[0] - e[1]) / (2.0 * h);
            let f = sys.mm_coupling_forces[0][k];
            assert!(
                (fd - f).abs() < 1e-2 + 1e-2 * fd.abs(),
                "boundary MM C component {k}: {f} vs FD {fd}"
            );
        }
        // QM boundary carbon: QM force + (1 − g) F_link.
        let mut e = [0.0; 2];
        for (slot, sign) in e.iter_mut().zip([1.0, -1.0]) {
            let mut d = ethane_link_system();
            d.qm.positions[0][0] += sign * h;
            d.compute().expect("displaced QM boundary atom");
            *slot = coupled_energy(&d);
        }
        let fd = -(e[0] - e[1]) / (2.0 * h);
        assert!(
            (fd - sys.qm_total_forces[0][0]).abs() < 2e-2 + 1e-2 * fd.abs(),
            "QM C: {} vs FD {fd}",
            sys.qm_total_forces[0][0]
        );
    }

    #[test]
    fn test_qm_mm_compute_rejects_bad_input_and_clears_outputs() {
        let mut sys = water_dimer_system();
        sys.compute().expect("valid system");
        assert!(sys.energy_qm_mm != 0.0);
        sys.add_link_atom(LinkAtom::new(0, 7, 0.7));
        assert_eq!(
            sys.compute(),
            Err(QmError::InvalidLinkAtom {
                region: "MM",
                index: 7,
                len: 3
            })
        );
        assert_eq!(sys.energy_qm_mm, 0.0);
        assert!(sys.mm_coupling_forces.iter().all(|f| *f == [0.0; 3]));

        let mut sys = water_dimer_system();
        sys.mm.charges.pop();
        assert!(matches!(
            sys.compute(),
            Err(QmError::DimensionMismatch { .. })
        ));

        let mut sys = water_dimer_system();
        sys.qm.atomic_numbers[0] = 54;
        assert_eq!(sys.compute(), Err(QmError::UnsupportedElement(54)));
        assert_eq!(sys.qm.energy, 0.0);
        assert_eq!(sys.energy_qm_mm, 0.0);
    }

    #[test]
    fn test_qm_mm_system_add_link_atom() {
        let qm = QmRegion::new(vec![0], QmMethod::Pm3, 0, 1);
        let mm = MmRegion::new(vec![1]);
        let mut sys = QmMmSystem::new(qm, mm);
        sys.add_link_atom(LinkAtom::new(0, 0, 0.709));
        assert_eq!(sys.link_atoms.len(), 1);
    }

    #[test]
    fn test_link_atom_new() {
        let la = LinkAtom::new(2, 5, 0.72);
        assert_eq!(la.qm_atom, 2);
        assert_eq!(la.mm_atom, 5);
        assert!((la.scale - 0.72).abs() < EPS);
    }

    #[test]
    fn test_qm_region_charged() {
        let qm = QmRegion::new(vec![0, 1], QmMethod::SccDftb, -1, 2);
        assert_eq!(qm.charge, -1);
        assert_eq!(qm.multiplicity, 2);
    }
}
