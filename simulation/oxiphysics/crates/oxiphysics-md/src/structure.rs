// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Protein structure analysis and crystal structure builders.
//!
//! Provides Kabsch alignment, RMSD, secondary structure assignment,
//! crystal structure builders (FCC, BCC, HCP, diamond), lattice parameter
//! optimization, coordination number analysis, Voronoi analysis basics,
//! and static structure factor S(q).

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Basic geometry helpers
// ---------------------------------------------------------------------------

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = norm3(v);
    if n < 1e-15 { v } else { scale3(v, 1.0 / n) }
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm3(sub3(a, b))
}

fn mat3_identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Maximum number of one-sided Jacobi sweeps (convergence is quadratic; a handful
/// of sweeps suffice for a 3×3 matrix).
const SVD_MAX_SWEEPS: usize = 60;
/// Relative orthogonality threshold below which two columns count as orthogonal.
const SVD_ORTHOGONALITY_EPS: f64 = 1e-15;
/// Relative threshold below which a singular value counts as zero.
const SVD_RANK_TOL: f64 = 1e-12;

/// Singular value decomposition `H = U diag(σ) Vᵀ` of a 3×3 matrix by one-sided
/// (Hestenes) Jacobi rotations, which keeps small singular values accurate.
///
/// Returns `(a, sigma, v)`: `a[j]` is the j-th column of `U diag(σ)` (so
/// `σ_j = |a[j]|` and the j-th column of `U` is `a[j] / σ_j` whenever `σ_j > 0`),
/// and `v[j]` is the j-th column of the orthogonal matrix `V`.  The singular
/// values are returned in no particular order.
fn svd3_columns(h: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    let mut a = [
        [h[0][0], h[1][0], h[2][0]],
        [h[0][1], h[1][1], h[2][1]],
        [h[0][2], h[1][2], h[2][2]],
    ];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..SVD_MAX_SWEEPS {
        let mut rotated = false;
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let alpha = dot3(a[p], a[p]);
            let beta = dot3(a[q], a[q]);
            let gamma = dot3(a[p], a[q]);
            if gamma == 0.0 || gamma.abs() <= SVD_ORTHOGONALITY_EPS * (alpha * beta).sqrt() {
                continue;
            }
            // Plane rotation that makes columns p and q orthogonal.
            let zeta = (beta - alpha) / (2.0 * gamma);
            let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = c * t;
            for cols in [&mut a, &mut v] {
                let (col_p, col_q) = (cols[p], cols[q]);
                for k in 0..3 {
                    cols[p][k] = c * col_p[k] - s * col_q[k];
                    cols[q][k] = s * col_p[k] + c * col_q[k];
                }
            }
            rotated = true;
        }
        if !rotated {
            break;
        }
    }
    let sigma = [norm3(a[0]), norm3(a[1]), norm3(a[2])];
    (a, sigma, v)
}

/// Unit vector orthogonal to the unit vector `u`, as close to `hint` as possible.
///
/// If `hint` is (nearly) parallel to `u`, the coordinate axis least aligned with
/// `u` is used instead.
fn unit_orthogonal_to(u: [f64; 3], hint: [f64; 3]) -> [f64; 3] {
    let w = sub3(hint, scale3(u, dot3(hint, u)));
    if norm3(w) > 1e-3 {
        return normalize3(w);
    }
    let mut axis = 0;
    for k in 1..3 {
        if u[k].abs() < u[axis].abs() {
            axis = k;
        }
    }
    let mut e = [0.0; 3];
    e[axis] = 1.0;
    normalize3(sub3(e, scale3(u, dot3(e, u))))
}

// ---------------------------------------------------------------------------
// Center of geometry
// ---------------------------------------------------------------------------

/// Compute the mean (center of geometry) of a set of 3-D positions.
pub fn center_of_geometry(positions: &[[f64; 3]]) -> [f64; 3] {
    if positions.is_empty() {
        return [0.0; 3];
    }
    let mut sum = [0.0f64; 3];
    for p in positions {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    let n = positions.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ---------------------------------------------------------------------------
// Translate / rotate
// ---------------------------------------------------------------------------

/// Translate all positions by vector `t`.
pub fn translate(positions: &mut [[f64; 3]], t: [f64; 3]) {
    for p in positions.iter_mut() {
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];
    }
}

/// Apply 3×3 rotation matrix to all positions.
pub fn rotate(positions: &mut [[f64; 3]], r: &[[f64; 3]; 3]) {
    for p in positions.iter_mut() {
        let x = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2];
        let y = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2];
        let z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2];
        *p = [x, y, z];
    }
}

// ---------------------------------------------------------------------------
// Kabsch rotation
// ---------------------------------------------------------------------------

/// Kabsch algorithm: the proper rotation `R` (3×3, `det R = +1`) that minimises
/// `Σ_k |R (p_k − c_p) − (q_k − c_q)|²`, where `c_p`, `c_q` are the centroids of
/// `p` and `q` (both sets are centred internally).  Apply it to centred
/// coordinates of `p` to superimpose them onto `q`.
///
/// With the cross-covariance `H = Σ_k (p_k − c_p)(q_k − c_q)ᵀ = U Σ Vᵀ` the optimum
/// is `R = V Uᵀ`.  The orthonormal frames are built right-handed from the two
/// dominant singular directions (`u₂ = u₀ × u₁`, `v₂ = v₀ × v₁`), which is exactly
/// the Kabsch reflection correction (it flips the weakest direction) and also
/// handles degenerate input: collinear point sets (rank-one `H`, the rotation about
/// the line is then arbitrary and is completed from `V`), coplanar sets, a single
/// point and coincident points (identity).
///
/// # Panics
/// Panics if `p` and `q` have different lengths.
pub fn kabsch_rotation(p: &[[f64; 3]], q: &[[f64; 3]]) -> [[f64; 3]; 3] {
    assert_eq!(
        p.len(),
        q.len(),
        "kabsch_rotation: point sets must be same length"
    );
    if p.is_empty() {
        return mat3_identity();
    }

    let cp = center_of_geometry(p);
    let cq = center_of_geometry(q);

    // Cross-covariance and the norm bound |H|_2 <= |P|_F |Q|_F used as scale.
    let mut h = [[0.0f64; 3]; 3];
    let mut p_norm_sq = 0.0f64;
    let mut q_norm_sq = 0.0f64;
    for (pk, qk) in p.iter().zip(q.iter()) {
        let pc = sub3(*pk, cp);
        let qc = sub3(*qk, cq);
        p_norm_sq += dot3(pc, pc);
        q_norm_sq += dot3(qc, qc);
        for (i, h_row) in h.iter_mut().enumerate() {
            for (j, h_ij) in h_row.iter_mut().enumerate() {
                *h_ij += pc[i] * qc[j];
            }
        }
    }

    let (a, sigma, v) = svd3_columns(&h);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (i0, i1) = (order[0], order[1]);
    let (sigma0, sigma1) = (sigma[i0], sigma[i1]);
    if sigma0 <= SVD_RANK_TOL * (p_norm_sq * q_norm_sq).sqrt() {
        // One of the sets has no extent: every rotation is optimal.
        return mat3_identity();
    }

    let v0 = v[i0];
    let v1 = v[i1];
    let u0 = scale3(a[i0], 1.0 / sigma0);
    let mut u1 = [0.0f64; 3];
    if sigma1 > SVD_RANK_TOL * sigma0 {
        let candidate = scale3(a[i1], 1.0 / sigma1);
        u1 = sub3(candidate, scale3(u0, dot3(candidate, u0)));
    }
    u1 = if norm3(u1) > 0.5 {
        normalize3(u1)
    } else {
        // Rank-one H (collinear points): complete the frame from V.
        unit_orthogonal_to(u0, v1)
    };
    let u2 = cross3(u0, u1);
    let v2 = cross3(v0, v1);

    let mut r = [[0.0f64; 3]; 3];
    for (row, r_row) in r.iter_mut().enumerate() {
        for (col, r_rc) in r_row.iter_mut().enumerate() {
            *r_rc = v0[row] * u0[col] + v1[row] * u1[col] + v2[row] * u2[col];
        }
    }
    r
}

// ---------------------------------------------------------------------------
// RMSD after alignment
// ---------------------------------------------------------------------------

/// Align `mobile` onto `reference` using Kabsch and return the RMSD.
pub fn rmsd_after_alignment(mobile: &[[f64; 3]], reference: &[[f64; 3]]) -> f64 {
    assert_eq!(mobile.len(), reference.len());
    let n = mobile.len();
    if n == 0 {
        return 0.0;
    }

    let cm = center_of_geometry(mobile);
    let cr = center_of_geometry(reference);

    let mut mob_c: Vec<[f64; 3]> = mobile.iter().map(|&x| sub3(x, cm)).collect();
    let ref_c: Vec<[f64; 3]> = reference.iter().map(|&x| sub3(x, cr)).collect();

    let r = kabsch_rotation(&mob_c, &ref_c);
    rotate(&mut mob_c, &r);

    let mut sum_sq = 0.0f64;
    for i in 0..n {
        let d = sub3(mob_c[i], ref_c[i]);
        sum_sq += dot3(d, d);
    }
    (sum_sq / n as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Dihedral angle
// ---------------------------------------------------------------------------

/// Torsion angle defined by four points a–b–c–d (in radians, range \[-π, π\]).
pub fn dihedral_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let b1 = sub3(b, a);
    let b2 = sub3(c, b);
    let b3 = sub3(d, c);

    let n1 = cross3(b1, b2);
    let n2 = cross3(b2, b3);

    let m1 = cross3(n1, normalize3(b2));

    let x = dot3(n1, n2);
    let y = dot3(m1, n2);

    y.atan2(x)
}

// ---------------------------------------------------------------------------
// phi / psi angles
// ---------------------------------------------------------------------------

/// Given backbone positions `[N, CA, C]` per residue (stride 3), compute φ/ψ
/// dihedral angles.  Returns vectors of length `n_residues`; the first φ and
/// last ψ are set to `999.0` (undefined).
pub fn compute_phi_psi(positions: &[[f64; 3]], n_residues: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(
        positions.len() >= n_residues * 3,
        "positions must contain at least n_residues*3 entries"
    );

    let atom = |res: usize, atom_idx: usize| -> [f64; 3] { positions[res * 3 + atom_idx] };

    let mut phi = vec![999.0f64; n_residues];
    let mut psi = vec![999.0f64; n_residues];

    for i in 0..n_residues {
        if i > 0 {
            phi[i] = dihedral_angle(atom(i - 1, 2), atom(i, 0), atom(i, 1), atom(i, 2));
            phi[i] = phi[i] * 180.0 / PI;
        }

        if i + 1 < n_residues {
            psi[i] = dihedral_angle(atom(i, 0), atom(i, 1), atom(i, 2), atom(i + 1, 0));
            psi[i] = psi[i] * 180.0 / PI;
        }
    }

    (phi, psi)
}

// ---------------------------------------------------------------------------
// Secondary structure
// ---------------------------------------------------------------------------

/// Ramachandran-based secondary structure types.
#[derive(Debug, Clone, PartialEq)]
pub enum SsType {
    /// α-helix region
    Helix,
    /// β-sheet region
    Sheet,
    /// Loop / coil
    Loop,
}

/// Secondary structure assignment for one residue.
#[derive(Debug, Clone)]
pub struct SecondaryStructure {
    /// Residue index (0-based)
    pub residue_idx: usize,
    /// Assigned secondary structure type
    pub ss_type: SsType,
}

/// Assign secondary structure from Ramachandran angles (in degrees).
///
/// - φ ∈ \[-160, -30\] ∧ ψ ∈ \[-70, 50\]  → [`SsType::Helix`]
/// - φ ∈ \[-160, -60\] ∧ ψ ∈ \[90, 180\]  → [`SsType::Sheet`]
/// - otherwise → [`SsType::Loop`]
pub fn assign_secondary_structure_simple(phi: &[f64], psi: &[f64]) -> Vec<SsType> {
    assert_eq!(phi.len(), psi.len());
    phi.iter()
        .zip(psi.iter())
        .map(|(&p, &s)| {
            if (-160.0..=-30.0).contains(&p) && (-70.0..=50.0).contains(&s) {
                SsType::Helix
            } else if (-160.0..=-60.0).contains(&p) && (90.0..=180.0).contains(&s) {
                SsType::Sheet
            } else {
                SsType::Loop
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Crystal structure builders
// ---------------------------------------------------------------------------

/// Build an FCC (face-centered cubic) lattice.
///
/// # Arguments
/// * `a` – lattice parameter (edge length of the conventional cell).
/// * `nx`, `ny`, `nz` – number of unit cells in each direction.
///
/// Returns atom positions in a box of size `[nx*a, ny*a, nz*a]`.
pub fn build_fcc(a: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5 * a, 0.5 * a, 0.0],
        [0.5 * a, 0.0, 0.5 * a],
        [0.0, 0.5 * a, 0.5 * a],
    ];
    let mut positions = Vec::with_capacity(4 * nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let origin = [ix as f64 * a, iy as f64 * a, iz as f64 * a];
                for &b in &basis {
                    positions.push(add3(origin, b));
                }
            }
        }
    }
    positions
}

/// Build a BCC (body-centered cubic) lattice.
///
/// # Arguments
/// * `a` – lattice parameter.
/// * `nx`, `ny`, `nz` – number of unit cells.
pub fn build_bcc(a: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let basis = [[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * a]];
    let mut positions = Vec::with_capacity(2 * nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let origin = [ix as f64 * a, iy as f64 * a, iz as f64 * a];
                for &b in &basis {
                    positions.push(add3(origin, b));
                }
            }
        }
    }
    positions
}

/// Build an HCP (hexagonal close-packed) lattice.
///
/// # Arguments
/// * `a` – lattice parameter (in-plane nearest-neighbor distance).
/// * `nx`, `ny`, `nz` – number of unit cells.
///
/// The c/a ratio is fixed at the ideal value sqrt(8/3).
pub fn build_hcp(a: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let c = a * (8.0_f64 / 3.0).sqrt();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5 * a, a * (3.0_f64).sqrt() / 6.0, 0.5 * c],
    ];
    let ax = a;
    let ay = a * (3.0_f64).sqrt() / 2.0;
    let az = c;
    let mut positions = Vec::with_capacity(2 * nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let origin = [ix as f64 * ax, iy as f64 * ay, iz as f64 * az];
                for &b in &basis {
                    positions.push(add3(origin, b));
                }
            }
        }
    }
    positions
}

/// Build a diamond cubic lattice.
///
/// # Arguments
/// * `a` – lattice parameter.
/// * `nx`, `ny`, `nz` – number of unit cells.
///
/// Diamond = two interpenetrating FCC lattices offset by (a/4, a/4, a/4).
pub fn build_diamond(a: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let fcc_basis = [
        [0.0, 0.0, 0.0],
        [0.5 * a, 0.5 * a, 0.0],
        [0.5 * a, 0.0, 0.5 * a],
        [0.0, 0.5 * a, 0.5 * a],
    ];
    let offset = [0.25 * a, 0.25 * a, 0.25 * a];
    let mut positions = Vec::with_capacity(8 * nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let origin = [ix as f64 * a, iy as f64 * a, iz as f64 * a];
                for &b in &fcc_basis {
                    positions.push(add3(origin, b));
                    positions.push(add3(add3(origin, b), offset));
                }
            }
        }
    }
    positions
}

/// Simple cubic lattice builder.
///
/// # Arguments
/// * `a` – lattice parameter.
/// * `nx`, `ny`, `nz` – number of unit cells.
pub fn build_simple_cubic(a: f64, nx: usize, ny: usize, nz: usize) -> Vec<[f64; 3]> {
    let mut positions = Vec::with_capacity(nx * ny * nz);
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                positions.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
            }
        }
    }
    positions
}

// ---------------------------------------------------------------------------
// Lattice parameter optimization
// ---------------------------------------------------------------------------

/// Optimize lattice parameter to minimize a simple pair energy.
///
/// Uses golden-section search to find the lattice parameter `a` in
/// `[a_min, a_max]` that minimizes the total pair energy computed
/// from all nearest-neighbor distances.
///
/// # Arguments
/// * `build_fn` – function that builds positions from a lattice parameter.
/// * `pair_energy_fn` – function that computes pair energy from distance.
/// * `a_min`, `a_max` – search range.
/// * `tol` – convergence tolerance.
pub fn optimize_lattice_parameter(
    build_fn: &dyn Fn(f64) -> Vec<[f64; 3]>,
    pair_energy_fn: &dyn Fn(f64) -> f64,
    a_min: f64,
    a_max: f64,
    tol: f64,
) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi;

    let total_energy = |a: f64| -> f64 {
        let pos = build_fn(a);
        let n = pos.len();
        let mut e = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let r = dist3(pos[i], pos[j]);
                if r > 0.0 && r < a * 1.5 {
                    e += pair_energy_fn(r);
                }
            }
        }
        e
    };

    let mut lo = a_min;
    let mut hi = a_max;
    let mut x1 = lo + resphi * (hi - lo);
    let mut x2 = hi - resphi * (hi - lo);
    let mut f1 = total_energy(x1);
    let mut f2 = total_energy(x2);

    for _ in 0..100 {
        if (hi - lo).abs() < tol {
            break;
        }
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = lo + resphi * (hi - lo);
            f1 = total_energy(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = hi - resphi * (hi - lo);
            f2 = total_energy(x2);
        }
    }

    (lo + hi) / 2.0
}

// ---------------------------------------------------------------------------
// Coordination number
// ---------------------------------------------------------------------------

/// Compute the coordination number of each atom (number of neighbors
/// within a cutoff distance).
///
/// # Arguments
/// * `positions` – atom positions.
/// * `cutoff` – neighbor cutoff distance.
///
/// Returns a vector of coordination numbers, one per atom.
pub fn coordination_numbers(positions: &[[f64; 3]], cutoff: f64) -> Vec<usize> {
    let n = positions.len();
    let cutoff_sq = cutoff * cutoff;
    let mut counts = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = sub3(positions[i], positions[j]);
            let dsq = dot3(d, d);
            if dsq <= cutoff_sq {
                counts[i] += 1;
                counts[j] += 1;
            }
        }
    }
    counts
}

/// Average coordination number across all atoms.
pub fn average_coordination_number(positions: &[[f64; 3]], cutoff: f64) -> f64 {
    let counts = coordination_numbers(positions, cutoff);
    if counts.is_empty() {
        return 0.0;
    }
    let sum: usize = counts.iter().sum();
    sum as f64 / counts.len() as f64
}

/// Coordination number with periodic boundary conditions.
pub fn coordination_numbers_pbc(
    positions: &[[f64; 3]],
    box_len: [f64; 3],
    cutoff: f64,
) -> Vec<usize> {
    let n = positions.len();
    let cutoff_sq = cutoff * cutoff;
    let mut counts = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = sub3(positions[i], positions[j]);
            for k in 0..3 {
                if d[k] > box_len[k] * 0.5 {
                    d[k] -= box_len[k];
                } else if d[k] < -box_len[k] * 0.5 {
                    d[k] += box_len[k];
                }
            }
            let dsq = dot3(d, d);
            if dsq <= cutoff_sq {
                counts[i] += 1;
                counts[j] += 1;
            }
        }
    }
    counts
}

// ---------------------------------------------------------------------------
// Voronoi analysis basics
// ---------------------------------------------------------------------------

/// Result of a nearest-neighbor Voronoi-like analysis for one atom.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// Index of the central atom.
    pub atom_idx: usize,
    /// Indices of Voronoi neighbors (atoms sharing a Voronoi face).
    pub neighbors: Vec<usize>,
    /// Approximate volume of the Voronoi cell (from nearest-neighbor estimate).
    pub volume_estimate: f64,
}

/// Compute approximate Voronoi neighbors using a cutoff-based approach.
///
/// An atom j is considered a Voronoi neighbor of i if no atom k is closer
/// to both i and j and lies between them (simplified radical plane test).
///
/// # Arguments
/// * `positions` – atom positions.
/// * `cutoff` – maximum distance to consider as a potential neighbor.
pub fn voronoi_neighbors(positions: &[[f64; 3]], cutoff: f64) -> Vec<VoronoiCell> {
    let n = positions.len();
    let cutoff_sq = cutoff * cutoff;
    let mut cells = Vec::with_capacity(n);

    for i in 0..n {
        let mut candidate_neighbors: Vec<(usize, f64)> = Vec::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = sub3(positions[j], positions[i]);
            let dsq = dot3(d, d);
            if dsq <= cutoff_sq {
                candidate_neighbors.push((j, dsq.sqrt()));
            }
        }
        // Sort by distance
        candidate_neighbors
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Filter: j is a Voronoi neighbor if no k is closer to i and
        // the midpoint(i,j) is closer to i than to k
        let mut neighbors = Vec::new();
        for &(j, dij) in &candidate_neighbors {
            let midpoint = scale3(add3(positions[i], positions[j]), 0.5);
            let mut is_voronoi = true;
            for &(k, _dik) in &candidate_neighbors {
                if k == j {
                    continue;
                }
                let dk_mid = dist3(positions[k], midpoint);
                let di_mid = dij * 0.5;
                if dk_mid < di_mid * 0.95 {
                    is_voronoi = false;
                    break;
                }
            }
            if is_voronoi {
                neighbors.push(j);
            }
        }

        // Volume estimate: 4/3 * pi * (nearest_neighbor_distance/2)^3
        let vol = if let Some(&(_, d_nn)) = candidate_neighbors.first() {
            4.0 / 3.0 * PI * (d_nn / 2.0).powi(3)
        } else {
            0.0
        };

        cells.push(VoronoiCell {
            atom_idx: i,
            neighbors,
            volume_estimate: vol,
        });
    }

    cells
}

/// Voronoi index (n3, n4, n5, n6) counting the number of faces with
/// 3, 4, 5, 6 edges respectively.  This is a simplified signature
/// that uses the neighbor count as a proxy.
pub fn voronoi_index(n_neighbors: usize) -> [usize; 4] {
    // Approximate Voronoi index from coordination number.
    // For ideal FCC: (0, 12, 0, 0)
    // For ideal BCC: (0, 6, 0, 8) or (0, 0, 12, 2) depending on convention
    // This is a rough approximation.
    match n_neighbors {
        12 => [0, 12, 0, 0],         // FCC-like
        14 => [0, 6, 0, 8],          // BCC-like
        8 => [0, 0, 0, 8],           // simple cubic
        _ => [0, 0, n_neighbors, 0], // generic
    }
}

// ---------------------------------------------------------------------------
// Structure factor S(q)
// ---------------------------------------------------------------------------

/// Compute the static structure factor S(q) for a set of positions.
///
/// S(q) = (1/N) |Σ_j exp(i q·r_j)|²
///
/// averaged over uniformly distributed q-vectors on a sphere of radius |q|.
///
/// # Arguments
/// * `positions` – atom positions.
/// * `q_values` – magnitudes of wavevectors |q| to evaluate.
/// * `n_q_vectors` – number of random q-directions to average over per |q|.
///
/// Returns a vector of S(q) values, one per q magnitude.
pub fn structure_factor(positions: &[[f64; 3]], q_values: &[f64], n_q_vectors: usize) -> Vec<f64> {
    let n = positions.len();
    if n == 0 {
        return vec![0.0; q_values.len()];
    }
    let n_f = n as f64;

    // Generate uniformly distributed directions on unit sphere using
    // the Fibonacci spiral method
    let directions: Vec<[f64; 3]> = (0..n_q_vectors)
        .map(|k| {
            let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let theta = (1.0 - 2.0 * (k as f64 + 0.5) / n_q_vectors as f64).acos();
            let phi_angle = 2.0 * PI * k as f64 / golden;
            [
                theta.sin() * phi_angle.cos(),
                theta.sin() * phi_angle.sin(),
                theta.cos(),
            ]
        })
        .collect();

    q_values
        .iter()
        .map(|&q_mag| {
            let mut sq_avg = 0.0;
            for dir in &directions {
                let q_vec = scale3(*dir, q_mag);
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for pos in positions {
                    let qr = dot3(q_vec, *pos);
                    re += qr.cos();
                    im += qr.sin();
                }
                sq_avg += (re * re + im * im) / n_f;
            }
            sq_avg / n_q_vectors as f64
        })
        .collect()
}

/// Compute the radial distribution function g(r) from positions.
///
/// # Arguments
/// * `positions` – atom positions.
/// * `box_len` – periodic box lengths \[Lx, Ly, Lz\].
/// * `n_bins` – number of histogram bins.
/// * `r_max` – maximum distance to compute g(r).
///
/// Returns (bin_centers, g_r) where g(r) is normalized.
pub fn radial_distribution_function(
    positions: &[[f64; 3]],
    box_len: [f64; 3],
    n_bins: usize,
    r_max: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = positions.len();
    let dr = r_max / n_bins as f64;
    let mut hist = vec![0usize; n_bins];

    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = sub3(positions[i], positions[j]);
            for k in 0..3 {
                if d[k] > box_len[k] * 0.5 {
                    d[k] -= box_len[k];
                } else if d[k] < -box_len[k] * 0.5 {
                    d[k] += box_len[k];
                }
            }
            let r = norm3(d);
            if r < r_max && r > 1e-15 {
                let bin = (r / dr) as usize;
                if bin < n_bins {
                    hist[bin] += 1;
                }
            }
        }
    }

    let volume = box_len[0] * box_len[1] * box_len[2];
    let _rho = n as f64 / volume;
    let n_pairs = n * (n - 1) / 2;

    let bin_centers: Vec<f64> = (0..n_bins).map(|b| (b as f64 + 0.5) * dr).collect();
    let g_r: Vec<f64> = (0..n_bins)
        .map(|b| {
            let r = (b as f64 + 0.5) * dr;
            let shell_vol = 4.0 / 3.0 * PI * ((r + dr * 0.5).powi(3) - (r - dr * 0.5).powi(3));
            let ideal = n_pairs as f64 * shell_vol / volume;
            if ideal > 1e-30 {
                hist[b] as f64 / ideal
            } else {
                0.0
            }
        })
        .collect();

    (bin_centers, g_r)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_of_geometry_symmetric() {
        let positions: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let cog = center_of_geometry(&positions);
        assert!(cog[0].abs() < 1e-12, "x != 0: {}", cog[0]);
        assert!(cog[1].abs() < 1e-12, "y != 0: {}", cog[1]);
        assert!(cog[2].abs() < 1e-12, "z != 0: {}", cog[2]);
    }

    #[test]
    fn test_rmsd_identical() {
        let pos: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 1.0]];
        let rmsd = rmsd_after_alignment(&pos, &pos);
        assert!(
            rmsd < 1e-10,
            "RMSD of identical sets should be ~0, got {}",
            rmsd
        );
    }

    #[test]
    fn test_rmsd_after_rotation() {
        let original: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let rotated: Vec<[f64; 3]> = original.iter().map(|&[x, y, z]| [-y, x, z]).collect();

        let rmsd = rmsd_after_alignment(&rotated, &original);
        assert!(
            rmsd < 1e-8,
            "RMSD after alignment of rotated config should be ~0, got {}",
            rmsd
        );
    }

    // ---- Kabsch rotation: exact recovery, degenerate input, optimality ----

    /// Deterministic pseudo-random numbers in [-1, 1) (64-bit LCG).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn random_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut state = seed;
        (0..n)
            .map(|_| [lcg(&mut state), lcg(&mut state), lcg(&mut state)])
            .collect()
    }

    /// Rodrigues rotation matrix about `axis` (any non-zero length).
    fn rotation_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let len = norm3(axis);
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

    fn mat_det(r: &[[f64; 3]; 3]) -> f64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    fn assert_proper_rotation(r: &[[f64; 3]; 3]) {
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "R Rᵀ[{i}][{j}] = {dot}, R = {r:?}"
                );
            }
        }
        assert!(
            (mat_det(r) - 1.0).abs() < 1e-12,
            "det R = {}, R = {r:?}",
            mat_det(r)
        );
    }

    fn max_abs_diff(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((a[i][j] - b[i][j]).abs());
            }
        }
        worst
    }

    fn identity() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn centred(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let c = center_of_geometry(points);
        points.iter().map(|&p| sub3(p, c)).collect()
    }

    /// RMSD of `R · p_c` against `q_c` for already-centred sets.
    fn rmsd_for_rotation(r: &[[f64; 3]; 3], p_c: &[[f64; 3]], q_c: &[[f64; 3]]) -> f64 {
        let sum: f64 = p_c
            .iter()
            .zip(q_c.iter())
            .map(|(&p, &q)| {
                let d = sub3(mat_vec(r, p), q);
                dot3(d, d)
            })
            .sum();
        (sum / p_c.len() as f64).sqrt()
    }

    #[test]
    fn test_kabsch_recovers_rotation_of_general_point_set() {
        let p = random_points(25, 7);
        let r_true = rotation_matrix([1.0, 2.0, 3.0], 0.7);
        let q: Vec<[f64; 3]> = p
            .iter()
            .map(|&x| {
                let y = mat_vec(&r_true, x);
                [y[0] + 3.0, y[1] - 1.5, y[2] + 0.25]
            })
            .collect();
        let r = kabsch_rotation(&p, &q);
        assert!(max_abs_diff(&r, &r_true) < 1e-12, "R = {r:?}");
        assert_proper_rotation(&r);
        assert!(rmsd_after_alignment(&p, &q) < 1e-12);
    }

    #[test]
    fn test_kabsch_rotation_is_proper_for_unrelated_and_mirrored_sets() {
        for seed in 0..20u64 {
            let p = random_points(9, 100 + seed);
            let unrelated = random_points(9, 900 + seed);
            assert_proper_rotation(&kabsch_rotation(&p, &unrelated));
            let mirrored: Vec<[f64; 3]> = p.iter().map(|&x| [-x[0], x[1], x[2]]).collect();
            assert_proper_rotation(&kabsch_rotation(&p, &mirrored));
        }
    }

    #[test]
    fn test_kabsch_rotation_is_optimal_when_the_reflection_correction_applies() {
        // q is an axis-scaled mirror image of p plus noise, so det(H) < 0: the optimal
        // PROPER rotation must sacrifice the weakest singular direction, wherever it
        // sits in the diagonalisation order.  No other rotation (global random search
        // and local perturbations) may beat it.
        let scalings = [
            [-1.0, 2.0, 0.5],
            [0.4, 2.0, -1.0],
            [0.3, -1.5, 1.0],
            [2.0, 0.4, -1.0],
        ];
        for (case, scale) in scalings.iter().enumerate() {
            for seed in 0..3u64 {
                let p = random_points(14, 31 + seed + 10 * case as u64);
                let mut state = 555 + seed;
                let q: Vec<[f64; 3]> = p
                    .iter()
                    .map(|&x| {
                        [
                            scale[0] * x[0] + 0.05 * lcg(&mut state),
                            scale[1] * x[1] + 0.05 * lcg(&mut state),
                            scale[2] * x[2] + 0.05 * lcg(&mut state),
                        ]
                    })
                    .collect();
                let p_c = centred(&p);
                let q_c = centred(&q);
                let r = kabsch_rotation(&p, &q);
                assert_proper_rotation(&r);
                let best = rmsd_for_rotation(&r, &p_c, &q_c);
                assert!((best - rmsd_after_alignment(&p, &q)).abs() < 1e-12);

                let mut rng = 4242 + seed;
                for _ in 0..3000 {
                    let axis = [lcg(&mut rng), lcg(&mut rng), lcg(&mut rng) + 1.5];
                    let angle = (lcg(&mut rng) + 1.0) * std::f64::consts::PI;
                    let other = rotation_matrix(axis, angle);
                    assert!(
                        rmsd_for_rotation(&other, &p_c, &q_c) >= best - 1e-12,
                        "a random rotation beats the Kabsch rotation (case {case}, seed {seed})"
                    );
                }
                for _ in 0..3000 {
                    let axis = [lcg(&mut rng), lcg(&mut rng), lcg(&mut rng) + 1.5];
                    let angle = 0.02 * lcg(&mut rng);
                    let nudge = rotation_matrix(axis, angle);
                    let mut perturbed = [[0.0f64; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            perturbed[i][j] = (0..3).map(|k| nudge[i][k] * r[k][j]).sum();
                        }
                    }
                    assert!(
                        rmsd_for_rotation(&perturbed, &p_c, &q_c) >= best - 1e-12,
                        "a nearby rotation beats the Kabsch rotation (case {case}, seed {seed})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_kabsch_collinear_sets() {
        // Identical collinear sets: the rotation is the identity, whatever the line.
        for dir in [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, -2.0],
            [-3.0, 0.5, 1.0],
        ] {
            let n = norm3(dir);
            let line: Vec<[f64; 3]> = (0..7)
                .map(|i| {
                    let s = i as f64 * 0.9 - 1.0;
                    [dir[0] / n * s, dir[1] / n * s, dir[2] / n * s]
                })
                .collect();
            let r = kabsch_rotation(&line, &line);
            assert!(
                max_abs_diff(&r, &identity()) < 1e-12,
                "dir {dir:?}: R = {r:?}"
            );
            assert_proper_rotation(&r);
        }
        // Different lines: R must map one direction onto the other.
        let (da, db) = ([1.0, 2.0, -2.0], [-3.0, 0.5, 1.0]);
        let (na, nb) = (norm3(da), norm3(db));
        let on_line = |d: [f64; 3], n: f64, offset: f64| -> Vec<[f64; 3]> {
            (0..6)
                .map(|i| {
                    let s = i as f64 * 1.3 + offset;
                    [d[0] / n * s, d[1] / n * s, d[2] / n * s]
                })
                .collect()
        };
        let p = on_line(da, na, 0.0);
        let q = on_line(db, nb, 4.0);
        let r = kabsch_rotation(&p, &q);
        assert_proper_rotation(&r);
        let mapped = mat_vec(&r, [da[0] / na, da[1] / na, da[2] / na]);
        for d in 0..3 {
            assert!((mapped[d] - db[d] / nb).abs() < 1e-12, "R·â = {mapped:?}");
        }
        assert!(rmsd_after_alignment(&p, &q) < 1e-12);
        // Two points are always collinear.
        let r2 = kabsch_rotation(
            &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        );
        assert_proper_rotation(&r2);
        let mapped = mat_vec(&r2, [0.5, -0.5, 0.0]);
        for (m, e) in mapped.iter().zip([0.5, 0.0, -0.5]) {
            assert!((m - e).abs() < 1e-12, "mapped {mapped:?}");
        }
    }

    #[test]
    fn test_kabsch_coplanar_sets_in_general_orientation() {
        // Any three points are coplanar; a planar set in a general orientation has a
        // rank-two cross-covariance whose zero singular value need not be the last
        // one in diagonal order.
        let r_plane = rotation_matrix([0.3, -1.0, 0.8], 1.1);
        let r_true = rotation_matrix([2.0, 1.0, -0.5], 2.4);
        for n in [3usize, 4, 7] {
            let planar: Vec<[f64; 3]> = random_points(n, 77 + n as u64)
                .iter()
                .map(|&x| mat_vec(&r_plane, [x[0], x[1], 0.0]))
                .collect();
            let moved: Vec<[f64; 3]> = planar
                .iter()
                .map(|&x| {
                    let y = mat_vec(&r_true, x);
                    [y[0] - 2.0, y[1] + 5.0, y[2] + 1.0]
                })
                .collect();
            let r = kabsch_rotation(&planar, &moved);
            assert_proper_rotation(&r);
            assert!(max_abs_diff(&r, &r_true) < 1e-10, "n = {n}: R = {r:?}");
            assert!(rmsd_after_alignment(&planar, &moved) < 1e-10, "n = {n}");
        }
    }

    #[test]
    fn test_kabsch_degenerate_inputs_give_identity() {
        assert_eq!(kabsch_rotation(&[], &[]), identity());
        // A single point carries no orientation.
        assert_eq!(
            kabsch_rotation(&[[1.0, 2.0, 3.0]], &[[-4.0, 0.5, 9.0]]),
            identity()
        );
        // Coincident points in one set: every rotation is optimal.
        let spread = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let pinned = [[1.0, 1.0, 1.0]; 3];
        assert_eq!(kabsch_rotation(&pinned, &spread), identity());
        assert_eq!(kabsch_rotation(&spread, &pinned), identity());
        assert_eq!(
            rmsd_after_alignment(&[[0.5, 0.5, 0.5]], &[[3.0, 3.0, 3.0]]),
            0.0
        );
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_kabsch_rotation_rejects_unequal_lengths() {
        let _ = kabsch_rotation(&[[0.0; 3], [1.0, 0.0, 0.0]], &[[0.0; 3]]);
    }

    #[test]
    fn test_secondary_structure_helix() {
        let phi = vec![-57.0f64];
        let psi = vec![-47.0f64];
        let ss = assign_secondary_structure_simple(&phi, &psi);
        assert_eq!(ss[0], SsType::Helix, "Expected Helix for phi=-57, psi=-47");
    }

    #[test]
    fn test_secondary_structure_sheet() {
        let phi = vec![-120.0f64];
        let psi = vec![130.0f64];
        let ss = assign_secondary_structure_simple(&phi, &psi);
        assert_eq!(ss[0], SsType::Sheet, "Expected Sheet for phi=-120, psi=130");
    }

    #[test]
    fn test_dihedral_angle_known() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let c = [0.0, 0.0, 1.0];
        let d = [0.0, 1.0, 1.0];
        let angle = dihedral_angle(a, b, c, d);
        assert!(
            (angle.abs() - PI / 2.0).abs() < 1e-10,
            "Expected |pi/2|, got {}",
            angle
        );
    }

    #[test]
    fn test_dihedral_angle_180() {
        let a = [0.0, 1.0, 0.0];
        let b = [0.0, 0.0, 0.0];
        let c = [1.0, 0.0, 0.0];
        let d = [1.0, 0.0, 1.0];
        let angle = dihedral_angle(a, b, c, d);
        assert!(
            (angle.abs() - PI / 2.0).abs() < 1e-10,
            "Expected |pi/2|, got {}",
            angle
        );
    }

    // ---- Crystal structure builder tests ----

    #[test]
    fn test_build_fcc_atom_count() {
        let pos = build_fcc(1.0, 2, 2, 2);
        // 4 atoms per unit cell * 8 cells = 32
        assert_eq!(pos.len(), 32, "FCC 2x2x2 should have 32 atoms");
    }

    #[test]
    fn test_build_bcc_atom_count() {
        let pos = build_bcc(1.0, 3, 3, 3);
        // 2 atoms per unit cell * 27 = 54
        assert_eq!(pos.len(), 54, "BCC 3x3x3 should have 54 atoms");
    }

    #[test]
    fn test_build_hcp_atom_count() {
        let pos = build_hcp(1.0, 2, 2, 2);
        // 2 atoms per cell * 8 = 16
        assert_eq!(pos.len(), 16, "HCP 2x2x2 should have 16 atoms");
    }

    #[test]
    fn test_build_diamond_atom_count() {
        let pos = build_diamond(1.0, 2, 2, 2);
        // 8 atoms per unit cell * 8 = 64
        assert_eq!(pos.len(), 64, "Diamond 2x2x2 should have 64 atoms");
    }

    #[test]
    fn test_build_simple_cubic_atom_count() {
        let pos = build_simple_cubic(1.0, 3, 3, 3);
        assert_eq!(pos.len(), 27, "SC 3x3x3 should have 27 atoms");
    }

    #[test]
    fn test_fcc_nearest_neighbor_distance() {
        let a = 4.08; // Gold lattice parameter in Angstrom
        let pos = build_fcc(a, 1, 1, 1);
        // Nearest neighbor in FCC is a/sqrt(2)
        let expected_nn = a / 2.0_f64.sqrt();
        let mut min_dist = f64::MAX;
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                let d = dist3(pos[i], pos[j]);
                if d < min_dist && d > 1e-10 {
                    min_dist = d;
                }
            }
        }
        assert!(
            (min_dist - expected_nn).abs() < 1e-10,
            "FCC NN distance: got {min_dist}, expected {expected_nn}"
        );
    }

    #[test]
    fn test_bcc_nearest_neighbor_distance() {
        let a = 2.87; // Iron lattice parameter
        let pos = build_bcc(a, 1, 1, 1);
        // BCC NN distance = a*sqrt(3)/2
        let expected_nn = a * 3.0_f64.sqrt() / 2.0;
        let mut min_dist = f64::MAX;
        for i in 0..pos.len() {
            for j in (i + 1)..pos.len() {
                let d = dist3(pos[i], pos[j]);
                if d < min_dist && d > 1e-10 {
                    min_dist = d;
                }
            }
        }
        assert!(
            (min_dist - expected_nn).abs() < 1e-10,
            "BCC NN distance: got {min_dist}, expected {expected_nn}"
        );
    }

    #[test]
    fn test_fcc_positions_in_box() {
        let a = 3.5;
        let nx = 2;
        let ny = 2;
        let nz = 2;
        let pos = build_fcc(a, nx, ny, nz);
        let box_max = [nx as f64 * a, ny as f64 * a, nz as f64 * a];
        for (idx, p) in pos.iter().enumerate() {
            for k in 0..3 {
                assert!(
                    p[k] >= 0.0 && p[k] <= box_max[k],
                    "Atom {idx} coord {k}={} out of box [0, {}]",
                    p[k],
                    box_max[k]
                );
            }
        }
    }

    // ---- Coordination number tests ----

    #[test]
    fn test_coordination_number_fcc() {
        let a = 4.0;
        let pos = build_fcc(a, 3, 3, 3);
        let nn = a / 2.0_f64.sqrt();
        let avg = average_coordination_number(&pos, nn * 1.1);
        // Interior FCC atoms have 12 neighbors; edge atoms have fewer.
        // Average should be reasonably high.
        assert!(avg > 4.0, "FCC avg coord should be >4, got {avg}");
    }

    #[test]
    fn test_coordination_number_simple() {
        // Two atoms within cutoff
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let counts = coordination_numbers(&positions, 1.5);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 1);
    }

    #[test]
    fn test_coordination_number_out_of_range() {
        let positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let counts = coordination_numbers(&positions, 1.5);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 0);
    }

    #[test]
    fn test_coordination_number_pbc() {
        // Two atoms that are far apart in direct distance but close via PBC
        let positions = [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]];
        let box_len = [10.0, 10.0, 10.0];
        let counts = coordination_numbers_pbc(&positions, box_len, 0.5);
        // PBC distance = 0.2, which is within cutoff
        assert_eq!(counts[0], 1, "Should be neighbors via PBC");
        assert_eq!(counts[1], 1);
    }

    // ---- Voronoi analysis tests ----

    #[test]
    fn test_voronoi_neighbors_simple() {
        // Tetrahedron: 4 atoms, each should be neighbor of the other 3
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816],
        ];
        let cells = voronoi_neighbors(&positions, 2.0);
        assert_eq!(cells.len(), 4);
        for cell in &cells {
            assert!(
                cell.neighbors.len() >= 2,
                "atom {} should have at least 2 Voronoi neighbors, got {}",
                cell.atom_idx,
                cell.neighbors.len()
            );
        }
    }

    #[test]
    fn test_voronoi_index_fcc() {
        let idx = voronoi_index(12);
        assert_eq!(idx, [0, 12, 0, 0], "FCC-like Voronoi index");
    }

    #[test]
    fn test_voronoi_index_bcc() {
        let idx = voronoi_index(14);
        assert_eq!(idx, [0, 6, 0, 8], "BCC-like Voronoi index");
    }

    // ---- Structure factor tests ----

    #[test]
    fn test_structure_factor_single_atom() {
        let positions = [[0.0, 0.0, 0.0]];
        let q_values = [1.0, 2.0, 5.0];
        let sq = structure_factor(&positions, &q_values, 20);
        // S(q) = 1/N * |exp(i*q*0)|^2 = 1 for all q
        for (i, &s) in sq.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-10,
                "S(q={}) for single atom should be 1, got {s}",
                q_values[i]
            );
        }
    }

    #[test]
    fn test_structure_factor_empty() {
        let positions: Vec<[f64; 3]> = vec![];
        let q_values = [1.0, 2.0];
        let sq = structure_factor(&positions, &q_values, 10);
        assert_eq!(sq.len(), 2);
        for s in &sq {
            assert!((s - 0.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_structure_factor_two_atoms() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let q_values = [0.01]; // small q => S(q) should be close to N=2
        let sq = structure_factor(&positions, &q_values, 50);
        // At very small q, S(q) ~ N = 2
        assert!(
            sq[0] > 1.0,
            "S(q~0) for 2 atoms should be >1, got {}",
            sq[0]
        );
    }

    // ---- RDF tests ----

    #[test]
    fn test_rdf_simple_pair() {
        let positions = [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]];
        let box_len = [5.0, 5.0, 5.0];
        let (bins, gr) = radial_distribution_function(&positions, box_len, 50, 2.5);
        assert_eq!(bins.len(), 50);
        assert_eq!(gr.len(), 50);
        // The peak should be at r ~= 1.0
        let peak_bin = gr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (bins[peak_bin] - 1.0).abs() < 0.2,
            "RDF peak should be near r=1.0, got r={}",
            bins[peak_bin]
        );
    }

    // ---- Lattice parameter optimization test ----

    #[test]
    fn test_optimize_lattice_parameter() {
        // Simple LJ-like pair energy with minimum at r=1.0
        let pair_energy = |r: f64| -> f64 {
            let s = 1.0;
            let r6 = (s / r).powi(6);
            4.0 * (r6 * r6 - r6)
        };
        // For SC lattice, NN distance = a, so minimum at a~1.12 (2^(1/6))
        let build = |a: f64| -> Vec<[f64; 3]> { build_simple_cubic(a, 2, 2, 2) };
        let a_opt = optimize_lattice_parameter(&build, &pair_energy, 0.8, 2.0, 0.001);
        let r_min = 2.0_f64.powf(1.0 / 6.0);
        assert!(
            (a_opt - r_min).abs() < 0.1,
            "Optimal lattice param should be near {r_min}, got {a_opt}"
        );
    }

    #[test]
    fn test_diamond_has_tetrahedral_coordination() {
        let a = 5.43; // Silicon
        let pos = build_diamond(a, 2, 2, 2);
        // Diamond structure: each atom has 4 nearest neighbors at a*sqrt(3)/4
        let nn_dist = a * 3.0_f64.sqrt() / 4.0;
        let avg = average_coordination_number(&pos, nn_dist * 1.1);
        // Interior atoms have 4 neighbors; edge atoms fewer. Average should be >1.
        assert!(avg > 1.0, "Diamond avg coord should be >1, got {avg}");
    }
}
