//! Geometric deep learning operations.
//!
//! Provides graph Laplacian computation, spectral graph convolution (GCN layer),
//! SO(3) rotation operations, and spherical harmonics basis for graph neural
//! network-style computations within TensorLogic.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from geometric operations.
#[derive(Debug)]
pub enum GeoError {
    /// Matrix or vector dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid graph structure (e.g., out-of-range node index).
    InvalidGraph(String),
    /// Numerical error (e.g., non-unit quaternion).
    NumericalError(String),
    /// Node with given index has invalid (zero) degree where a non-zero degree is required.
    InvalidDegree(usize),
}

impl std::fmt::Display for GeoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            GeoError::InvalidGraph(msg) => write!(f, "Invalid graph: {msg}"),
            GeoError::NumericalError(msg) => write!(f, "Numerical error: {msg}"),
            GeoError::InvalidDegree(node) => {
                write!(
                    f,
                    "Node {node} has zero degree where a non-zero degree is required"
                )
            }
        }
    }
}

impl std::error::Error for GeoError {}

// ---------------------------------------------------------------------------
// AdjacencyMatrix
// ---------------------------------------------------------------------------

/// Dense adjacency matrix for a graph with `n` nodes.
///
/// Stored in row-major order: element `(i, j)` is at index `i * n + j`.
#[derive(Debug, Clone)]
pub struct AdjacencyMatrix {
    n: usize,
    data: Vec<f64>,
}

impl AdjacencyMatrix {
    /// Create an `n × n` adjacency matrix initialised to all zeros.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0_f64; n * n],
        }
    }

    /// Build an unweighted (0/1) adjacency matrix from an edge list.
    ///
    /// Each `(i, j)` edge sets both `A[i,j] = 1` and `A[j,i] = 1` so the
    /// resulting matrix is symmetric (undirected graph).
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj = Self::new(n);
        for &(i, j) in edges {
            adj.data[i * n + j] = 1.0;
            adj.data[j * n + i] = 1.0;
        }
        adj
    }

    /// Build a weighted adjacency matrix from a weighted edge list `(i, j, weight)`.
    ///
    /// Both `A[i,j]` and `A[j,i]` are set to `weight` (undirected).
    pub fn from_edges_weighted(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adj = Self::new(n);
        for &(i, j, w) in edges {
            adj.data[i * n + j] = w;
            adj.data[j * n + i] = w;
        }
        adj
    }

    /// Set element `(i, j)`.
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.n + j] = val;
    }

    /// Get element `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Return number of nodes.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Compute the degree of node `i` as the sum of row `i`.
    pub fn degree(&self, i: usize) -> f64 {
        let start = i * self.n;
        self.data[start..start + self.n].iter().sum()
    }

    /// Return `true` if the matrix is symmetric (`A[i,j] == A[j,i]` for all i,j).
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if (self.data[i * self.n + j] - self.data[j * self.n + i]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    /// Symmetrize in place: `A = (A + A^T) / 2`.
    pub fn symmetrize(&mut self) {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let avg = (self.data[i * self.n + j] + self.data[j * self.n + i]) * 0.5;
                self.data[i * self.n + j] = avg;
                self.data[j * self.n + i] = avg;
            }
        }
    }

    /// Add self-loops: `A[i,i] += 1` for all `i`.
    pub fn add_self_loops(&mut self) {
        for i in 0..self.n {
            self.data[i * self.n + i] += 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// LaplacianType / LaplacianMatrix
// ---------------------------------------------------------------------------

/// Variants of the graph Laplacian.
#[derive(Debug, Clone)]
pub enum LaplacianType {
    /// Combinatorial Laplacian: `L = D - A`.
    Unnormalized,
    /// Symmetric normalised Laplacian: `L_sym = D^{-1/2} (D - A) D^{-1/2}`.
    Symmetric,
    /// Random-walk Laplacian: `L_rw = I - D^{-1} A`.
    RandomWalk,
    /// GCN-style normalised adjacency with self-loops:
    /// `Ã = A + I`, `D̃ = deg(Ã)`, output = `D̃^{-1/2} Ã D̃^{-1/2}`.
    AddSelfLoops,
}

/// Dense Laplacian matrix.
#[derive(Debug, Clone)]
pub struct LaplacianMatrix {
    /// Row-major `n × n` data.
    pub data: Vec<f64>,
    /// Number of nodes.
    pub n: usize,
    /// Which variant was computed.
    pub laplacian_type: LaplacianType,
}

// ---------------------------------------------------------------------------
// graph_laplacian
// ---------------------------------------------------------------------------

/// Compute the graph Laplacian of the given adjacency matrix.
///
/// # Errors
/// Returns [`GeoError::InvalidGraph`] if any node index is out of range.
pub fn graph_laplacian(
    adj: &AdjacencyMatrix,
    lap_type: LaplacianType,
) -> Result<LaplacianMatrix, GeoError> {
    let n = adj.n();
    if n == 0 {
        return Err(GeoError::InvalidGraph(
            "Graph must have at least one node".to_string(),
        ));
    }

    let mut data = vec![0.0_f64; n * n];

    match lap_type {
        LaplacianType::Unnormalized => {
            // L[i,i] = deg(i), L[i,j] = -A[i,j]
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        data[i * n + i] = adj.degree(i);
                    } else {
                        data[i * n + j] = -adj.get(i, j);
                    }
                }
            }
        }
        LaplacianType::Symmetric => {
            // L_sym = D^{-1/2} (D - A) D^{-1/2}
            // L_sym[i,j] = (D-A)[i,j] / (sqrt(d_i) * sqrt(d_j))
            // For isolated nodes (d_i = 0), set L_sym[i,j] = 0
            let degrees: Vec<f64> = (0..n).map(|i| adj.degree(i)).collect();
            for i in 0..n {
                for j in 0..n {
                    let unnorm = if i == j { degrees[i] } else { -adj.get(i, j) };
                    if degrees[i] > 1e-12 && degrees[j] > 1e-12 {
                        data[i * n + j] = unnorm / (degrees[i].sqrt() * degrees[j].sqrt());
                    } else {
                        data[i * n + j] = 0.0;
                    }
                }
            }
        }
        LaplacianType::RandomWalk => {
            // L_rw = I - D^{-1} A
            // L_rw[i,i] = 1 - A[i,i]/d_i
            // L_rw[i,j] = -A[i,j]/d_i  (i ≠ j)
            // For isolated nodes, treat as 0 row (L_rw[i,j] = 0 for all j)
            for i in 0..n {
                let d = adj.degree(i);
                if d < 1e-12 {
                    // isolated node — entire row stays 0
                    continue;
                }
                for j in 0..n {
                    let a_ij = adj.get(i, j);
                    if i == j {
                        data[i * n + j] = 1.0 - a_ij / d;
                    } else {
                        data[i * n + j] = -a_ij / d;
                    }
                }
            }
        }
        LaplacianType::AddSelfLoops => {
            // Ã = A + I, D̃ = diag(Ã * 1)
            // output = D̃^{-1/2} Ã D̃^{-1/2}
            let mut a_tilde = adj.clone();
            a_tilde.add_self_loops();
            let d_tilde: Vec<f64> = (0..n).map(|i| a_tilde.degree(i)).collect();
            for i in 0..n {
                for j in 0..n {
                    let a_ij = a_tilde.get(i, j);
                    if d_tilde[i] > 1e-12 && d_tilde[j] > 1e-12 {
                        data[i * n + j] = a_ij / (d_tilde[i].sqrt() * d_tilde[j].sqrt());
                    } else {
                        data[i * n + j] = 0.0;
                    }
                }
            }
        }
    }

    Ok(LaplacianMatrix {
        data,
        n,
        laplacian_type: lap_type,
    })
}

// ---------------------------------------------------------------------------
// GCN layer
// ---------------------------------------------------------------------------

/// Activation function for a GCN layer.
#[derive(Debug, Clone, Copy)]
pub enum GcnActivation {
    /// Rectified linear unit: `max(0, x)`.
    ReLU,
    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// No activation (identity).
    Linear,
}

impl GcnActivation {
    /// Apply the activation element-wise.
    fn apply(self, x: f64) -> f64 {
        match self {
            GcnActivation::ReLU => x.max(0.0),
            GcnActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            GcnActivation::Tanh => x.tanh(),
            GcnActivation::Linear => x,
        }
    }
}

/// Multiply matrix `a` (m × k) by matrix `b` (k × n), returning a (m × n) matrix.
///
/// Both matrices are represented as `Vec<Vec<f64>>` (outer = rows, inner = columns).
///
/// # Errors
/// Returns [`GeoError::DimensionMismatch`] if the inner dimensions do not match,
/// or [`GeoError::InvalidGraph`] if either matrix is empty / ragged.
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GeoError> {
    if a.is_empty() || b.is_empty() {
        return Err(GeoError::InvalidGraph(
            "Matrix must be non-empty".to_string(),
        ));
    }
    let m = a.len();
    let k = a[0].len();
    let k2 = b.len();
    let n = b[0].len();

    // Validate all rows have consistent width
    for (row_idx, row) in a.iter().enumerate() {
        if row.len() != k {
            return Err(GeoError::InvalidGraph(format!(
                "Matrix A has ragged rows: row 0 has {k} cols but row {row_idx} has {} cols",
                row.len()
            )));
        }
    }
    for (row_idx, row) in b.iter().enumerate() {
        if row.len() != n {
            return Err(GeoError::InvalidGraph(format!(
                "Matrix B has ragged rows: row 0 has {n} cols but row {row_idx} has {} cols",
                row.len()
            )));
        }
    }

    if k != k2 {
        return Err(GeoError::DimensionMismatch {
            expected: k,
            got: k2,
        });
    }

    let mut result = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i][l];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..n {
                result[i][j] += a_il * b[l][j];
            }
        }
    }
    Ok(result)
}

/// Compute a single Graph Convolutional Network (GCN) layer.
///
/// Implements: `H' = σ(L̂ · H · W)` where
/// - `L̂` = `AddSelfLoops` normalised Laplacian (i.e. `D̃^{-1/2} Ã D̃^{-1/2}`),
/// - `H` is the node feature matrix (`n × d_in`),
/// - `W` is the weight matrix (`d_in × d_out`),
/// - `σ` is `activation`.
///
/// # Arguments
/// * `adj`           – unweighted adjacency (`n × n`).
/// * `node_features` – node feature matrix (`n × d_in`).
/// * `weights`       – weight matrix (`d_in × d_out`).
/// * `activation`    – element-wise activation to apply after the linear transform.
///
/// # Errors
/// Returns a [`GeoError`] on dimension mismatch or empty inputs.
pub fn gcn_layer(
    adj: &AdjacencyMatrix,
    node_features: &[Vec<f64>],
    weights: &[Vec<f64>],
    activation: GcnActivation,
) -> Result<Vec<Vec<f64>>, GeoError> {
    let n = adj.n();
    if node_features.len() != n {
        return Err(GeoError::DimensionMismatch {
            expected: n,
            got: node_features.len(),
        });
    }
    if node_features.is_empty() {
        return Err(GeoError::InvalidGraph(
            "node_features must be non-empty".to_string(),
        ));
    }

    let d_in = node_features[0].len();

    // Validate d_in matches weights rows
    if weights.len() != d_in {
        return Err(GeoError::DimensionMismatch {
            expected: d_in,
            got: weights.len(),
        });
    }

    // Step 1: Compute L̂ = D̃^{-1/2} Ã D̃^{-1/2} (AddSelfLoops variant)
    let laplacian = graph_laplacian(adj, LaplacianType::AddSelfLoops)?;

    // Convert laplacian.data (flat n×n) into Vec<Vec<f64>> for mat_mul
    let l_hat: Vec<Vec<f64>> = (0..n)
        .map(|i| laplacian.data[i * n..(i + 1) * n].to_vec())
        .collect();

    // Step 2: L̂ · H  →  shape (n × d_in)
    let lh = mat_mul(&l_hat, node_features)?;

    // Step 3: (L̂ · H) · W  →  shape (n × d_out)
    let lhw = mat_mul(&lh, weights)?;

    // Step 4: Apply activation element-wise
    let output = lhw
        .into_iter()
        .map(|row| row.into_iter().map(|x| activation.apply(x)).collect())
        .collect();

    Ok(output)
}

// ---------------------------------------------------------------------------
// Rotation3 (SO(3))
// ---------------------------------------------------------------------------

/// A 3×3 rotation matrix representing an element of SO(3).
#[derive(Debug, Clone)]
pub struct Rotation3 {
    matrix: [[f64; 3]; 3],
}

impl Rotation3 {
    /// Identity rotation.
    pub fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Construct a rotation from an axis-angle representation using Rodrigues' formula.
    ///
    /// `axis` need not be normalised — it is normalised internally.
    /// `angle_rad` is the rotation angle in radians.
    ///
    /// If `axis` is the zero vector the identity rotation is returned.
    pub fn from_axis_angle(axis: [f64; 3], angle_rad: f64) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-12 {
            return Self::identity();
        }
        let kx = axis[0] / norm;
        let ky = axis[1] / norm;
        let kz = axis[2] / norm;

        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let t = 1.0 - c;

        // Rodrigues' formula: R = I cos θ + (1-cos θ) k k^T + sin θ [k]×
        #[rustfmt::skip]
        let matrix = [
            [t * kx * kx + c,       t * kx * ky - s * kz,  t * kx * kz + s * ky],
            [t * kx * ky + s * kz,  t * ky * ky + c,        t * ky * kz - s * kx],
            [t * kx * kz - s * ky,  t * ky * kz + s * kx,   t * kz * kz + c     ],
        ];

        Self { matrix }
    }

    /// Construct from intrinsic Euler angles (roll = Rx, pitch = Ry, yaw = Rz).
    ///
    /// Convention: `R = Rz(yaw) · Ry(pitch) · Rx(roll)`.
    pub fn from_euler_xyz(roll: f64, pitch: f64, yaw: f64) -> Self {
        let rx = Self::from_axis_angle([1.0, 0.0, 0.0], roll);
        let ry = Self::from_axis_angle([0.0, 1.0, 0.0], pitch);
        let rz = Self::from_axis_angle([0.0, 0.0, 1.0], yaw);
        rz.compose(&ry).compose(&rx)
    }

    /// Construct from a unit quaternion `(w, x, y, z)`.
    ///
    /// # Errors
    /// Returns [`GeoError::NumericalError`] if the quaternion is not (approximately) unit.
    pub fn from_quaternion(w: f64, x: f64, y: f64, z: f64) -> Result<Self, GeoError> {
        let norm_sq = w * w + x * x + y * y + z * z;
        if (norm_sq - 1.0).abs() > 1e-6 {
            return Err(GeoError::NumericalError(format!(
                "Quaternion norm squared is {norm_sq:.6}, expected 1.0"
            )));
        }
        // Normalise defensively
        let n = norm_sq.sqrt();
        let (w, x, y, z) = (w / n, x / n, y / n, z / n);

        #[rustfmt::skip]
        let matrix = [
            [1.0 - 2.0*(y*y + z*z),  2.0*(x*y - w*z),         2.0*(x*z + w*y)        ],
            [2.0*(x*y + w*z),         1.0 - 2.0*(x*x + z*z),   2.0*(y*z - w*x)        ],
            [2.0*(x*z - w*y),         2.0*(y*z + w*x),          1.0 - 2.0*(x*x + y*y) ],
        ];

        Ok(Self { matrix })
    }

    /// Convert to unit quaternion `(w, x, y, z)`.
    ///
    /// Uses Shepperd's method for numerical stability.
    pub fn to_quaternion(&self) -> (f64, f64, f64, f64) {
        let m = &self.matrix;
        let trace = m[0][0] + m[1][1] + m[2][2];

        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0; // s = 4w
            let w = 0.25 * s;
            let x = (m[2][1] - m[1][2]) / s;
            let y = (m[0][2] - m[2][0]) / s;
            let z = (m[1][0] - m[0][1]) / s;
            (w, x, y, z)
        } else if (m[0][0] > m[1][1]) && (m[0][0] > m[2][2]) {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0; // s = 4x
            let w = (m[2][1] - m[1][2]) / s;
            let x = 0.25 * s;
            let y = (m[0][1] + m[1][0]) / s;
            let z = (m[0][2] + m[2][0]) / s;
            (w, x, y, z)
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0; // s = 4y
            let w = (m[0][2] - m[2][0]) / s;
            let x = (m[0][1] + m[1][0]) / s;
            let y = 0.25 * s;
            let z = (m[1][2] + m[2][1]) / s;
            (w, x, y, z)
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0; // s = 4z
            let w = (m[1][0] - m[0][1]) / s;
            let x = (m[0][2] + m[2][0]) / s;
            let y = (m[1][2] + m[2][1]) / s;
            let z = 0.25 * s;
            (w, x, y, z)
        }
    }

    /// Rotate a 3D vector by this rotation: `v' = R · v`.
    pub fn apply(&self, v: [f64; 3]) -> [f64; 3] {
        let m = &self.matrix;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    /// Compose two rotations: `R1.compose(R2)` = `R1 · R2`.
    pub fn compose(&self, other: &Rotation3) -> Rotation3 {
        let a = &self.matrix;
        let b = &other.matrix;
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    m[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        Rotation3 { matrix: m }
    }

    /// Compute the inverse rotation.
    ///
    /// For orthogonal matrices `R^{-1} = R^T`.
    pub fn inverse(&self) -> Rotation3 {
        let m = &self.matrix;
        Rotation3 {
            matrix: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }

    /// Check that this matrix is a valid rotation:
    /// - determinant ≈ 1
    /// - `R^T R ≈ I`
    pub fn is_valid(&self) -> bool {
        // Check R^T R ≈ I
        let rt_r = self.inverse().compose(self);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (rt_r.matrix[i][j] - expected).abs() > 1e-9 {
                    return false;
                }
            }
        }
        // Check determinant ≈ 1
        let det = self.determinant();
        (det - 1.0).abs() < 1e-9
    }

    /// Compute the 3×3 matrix determinant.
    fn determinant(&self) -> f64 {
        let m = &self.matrix;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

// ---------------------------------------------------------------------------
// Spherical harmonics
// ---------------------------------------------------------------------------

/// Evaluate the real spherical harmonic `Y_l^m(θ, φ)`.
///
/// - `l` is the degree (≥ 0); currently supports `l ∈ {0, 1, 2}`.
/// - `m` is the order (`|m| ≤ l`).
/// - `theta ∈ [0, π]` is the polar angle (co-latitude).
/// - `phi ∈ [0, 2π]` is the azimuthal angle.
///
/// Uses real (tesseral) spherical harmonic conventions.  For `l > 2` returns 0.
pub fn sph_harm(l: usize, m: i32, theta: f64, phi: f64) -> f64 {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    match (l, m) {
        (0, 0) => {
            // Y_0^0 = 1 / (2 sqrt(π))
            1.0 / (2.0 * PI.sqrt())
        }
        (1, -1) => {
            // Y_1^{-1} = sqrt(3/(4π)) sin(θ) sin(φ)
            (3.0 / (4.0 * PI)).sqrt() * sin_theta * phi.sin()
        }
        (1, 0) => {
            // Y_1^0 = sqrt(3/(4π)) cos(θ)
            (3.0 / (4.0 * PI)).sqrt() * cos_theta
        }
        (1, 1) => {
            // Y_1^1 = sqrt(3/(4π)) sin(θ) cos(φ)
            (3.0 / (4.0 * PI)).sqrt() * sin_theta * phi.cos()
        }
        (2, -2) => {
            // Y_2^{-2} = (1/2) sqrt(15/(4π)) sin²(θ) sin(2φ)
            0.5 * (15.0 / (4.0 * PI)).sqrt() * sin_theta * sin_theta * (2.0 * phi).sin()
        }
        (2, -1) => {
            // Y_2^{-1} = sqrt(15/(4π)) sin(θ) cos(θ) sin(φ)
            (15.0 / (4.0 * PI)).sqrt() * sin_theta * cos_theta * phi.sin()
        }
        (2, 0) => {
            // Y_2^0 = sqrt(5/(16π)) (3 cos²(θ) - 1)
            (5.0 / (16.0 * PI)).sqrt() * (3.0 * cos_theta * cos_theta - 1.0)
        }
        (2, 1) => {
            // Y_2^1 = sqrt(15/(4π)) sin(θ) cos(θ) cos(φ)
            (15.0 / (4.0 * PI)).sqrt() * sin_theta * cos_theta * phi.cos()
        }
        (2, 2) => {
            // Y_2^2 = sqrt(15/(16π)) sin²(θ) cos(2φ)
            (15.0 / (16.0 * PI)).sqrt() * sin_theta * sin_theta * (2.0 * phi).cos()
        }
        _ => 0.0,
    }
}

/// Evaluate all real spherical harmonics up to degree `max_degree` at the point `(θ, φ)`.
///
/// Returns a `Vec<f64>` with `(max_degree + 1)^2` entries ordered as:
/// `[Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ..., Y_l^l, ...]`
///
/// Currently supports `max_degree ≤ 2`; for higher degrees the values are 0.
pub fn spherical_harmonics(theta: f64, phi: f64, max_degree: usize) -> Vec<f64> {
    let total = (max_degree + 1) * (max_degree + 1);
    let mut values = Vec::with_capacity(total);
    for l in 0..=max_degree {
        let l_i32 = l as i32;
        for m in -l_i32..=l_i32 {
            values.push(sph_harm(l, m, theta, phi));
        }
    }
    values
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ------------------------------------------------------------------
    // AdjacencyMatrix tests
    // ------------------------------------------------------------------

    #[test]
    fn test_adjacency_new_all_zeros() {
        let adj = AdjacencyMatrix::new(4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(adj.get(i, j), 0.0, "Expected 0 at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_from_edges_symmetric() {
        let adj = AdjacencyMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        assert!(
            adj.is_symmetric(),
            "from_edges should produce symmetric matrix"
        );
        assert_eq!(adj.get(0, 1), 1.0);
        assert_eq!(adj.get(1, 0), 1.0);
        assert_eq!(adj.get(0, 2), 0.0);
    }

    #[test]
    fn test_degree_sums_row() {
        // Triangle graph: each node has degree 2
        let adj = AdjacencyMatrix::from_edges(3, &[(0, 1), (1, 2), (0, 2)]);
        for i in 0..3 {
            assert!(
                (adj.degree(i) - 2.0).abs() < EPS,
                "degree of node {i} should be 2"
            );
        }
    }

    #[test]
    fn test_add_self_loops_sets_diagonal() {
        let mut adj = AdjacencyMatrix::new(3);
        adj.add_self_loops();
        for i in 0..3 {
            assert!(
                (adj.get(i, i) - 1.0).abs() < EPS,
                "diagonal {i} should be 1"
            );
        }
        // Off-diagonal stays 0
        assert_eq!(adj.get(0, 1), 0.0);
    }

    // ------------------------------------------------------------------
    // Laplacian tests
    // ------------------------------------------------------------------

    #[test]
    fn test_laplacian_unnormalized_row_sums_zero() {
        // Path graph 0-1-2-3
        let adj = AdjacencyMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let lap = graph_laplacian(&adj, LaplacianType::Unnormalized).unwrap();
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| lap.data[i * 4 + j]).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {i} sum should be 0, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_laplacian_symmetric_trace_equals_sum_of_degrees_div_n() {
        // Complete graph K3: each node has degree 2
        let adj = AdjacencyMatrix::from_edges(3, &[(0, 1), (1, 2), (0, 2)]);
        let lap = graph_laplacian(&adj, LaplacianType::Symmetric).unwrap();
        // Trace of L_sym = sum_i L_sym[i,i] = sum_i (d_i / d_i) = n
        let trace: f64 = (0..3).map(|i| lap.data[i * 3 + i]).sum();
        // Each L_sym[i,i] = d_i / (sqrt(d_i)*sqrt(d_i)) = 1; trace = 3
        assert!(
            (trace - 3.0).abs() < 1e-10,
            "Trace should be 3, got {trace}"
        );
    }

    #[test]
    fn test_laplacian_random_walk_row_sums_zero() {
        let adj = AdjacencyMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
        let lap = graph_laplacian(&adj, LaplacianType::RandomWalk).unwrap();
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| lap.data[i * 4 + j]).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "RandomWalk row {i} sum should be 0, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_isolated_node_symmetric_laplacian_no_nan() {
        // Node 0 is isolated (no edges)
        let adj = AdjacencyMatrix::from_edges(3, &[(1, 2)]);
        let lap = graph_laplacian(&adj, LaplacianType::Symmetric).unwrap();
        for val in &lap.data {
            assert!(
                val.is_finite(),
                "Symmetric Laplacian must not contain NaN/Inf for isolated nodes"
            );
        }
        // Row/col of isolated node should be all zeros
        for j in 0..3 {
            assert_eq!(lap.data[j], 0.0, "Row 0 should be 0 for isolated node");
            assert_eq!(lap.data[j * 3], 0.0, "Col 0 should be 0 for isolated node");
        }
    }

    // ------------------------------------------------------------------
    // GCN layer tests
    // ------------------------------------------------------------------

    #[test]
    fn test_gcn_layer_output_shape() {
        let adj = AdjacencyMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]; // 3×2
        let weights = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]; // 2×3
        let out = gcn_layer(&adj, &features, &weights, GcnActivation::Linear).unwrap();
        assert_eq!(out.len(), 3, "Output should have n=3 rows");
        assert_eq!(out[0].len(), 3, "Output should have d_out=3 cols");
    }

    #[test]
    fn test_gcn_layer_relu_clips_negatives() {
        let adj = AdjacencyMatrix::from_edges(2, &[(0, 1)]);
        // Features that will produce negative values after transform
        let features = vec![vec![-2.0], vec![-3.0]];
        let weights = vec![vec![1.0]]; // d_in=1, d_out=1; identity weight
        let out = gcn_layer(&adj, &features, &weights, GcnActivation::ReLU).unwrap();
        for row in &out {
            for &val in row {
                assert!(val >= 0.0, "ReLU output must be non-negative, got {val}");
            }
        }
    }

    #[test]
    fn test_gcn_layer_linear_no_clipping() {
        let adj = AdjacencyMatrix::from_edges(2, &[(0, 1)]);
        let features = vec![vec![1.0], vec![2.0]];
        // weight = 2: output = 2 * (L̂ H)
        let weights = vec![vec![2.0]];
        let out_lin = gcn_layer(&adj, &features, &weights, GcnActivation::Linear).unwrap();
        let out_relu = gcn_layer(&adj, &features, &weights, GcnActivation::ReLU).unwrap();
        // Since features are positive, ReLU should not change anything
        for i in 0..2 {
            for j in 0..1 {
                assert!(
                    (out_lin[i][j] - out_relu[i][j]).abs() < 1e-10,
                    "Linear and ReLU should agree for positive inputs"
                );
            }
        }
    }

    #[test]
    fn test_gcn_layer_identity_weights_equals_l_hat_h() {
        // With identity weight matrix W = I, output = σ(L̂ · H · I) = σ(L̂ · H)
        let adj = AdjacencyMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        // 2×2 identity weight
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let out = gcn_layer(&adj, &features, &weights, GcnActivation::Linear).unwrap();

        // Manually compute L̂ · H
        let laplacian = graph_laplacian(&adj, LaplacianType::AddSelfLoops).unwrap();
        let l_hat: Vec<Vec<f64>> = (0..3)
            .map(|i| laplacian.data[i * 3..(i + 1) * 3].to_vec())
            .collect();
        let expected = mat_mul(&l_hat, &features).unwrap();

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (out[i][j] - expected[i][j]).abs() < 1e-10,
                    "GCN with identity W should equal L̂·H at ({i},{j})"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // mat_mul tests
    // ------------------------------------------------------------------

    #[test]
    fn test_mat_mul_dimension_mismatch_returns_error() {
        let a = vec![vec![1.0, 2.0, 3.0]]; // 1×3
        let b = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // 2×2  (k mismatch: 3 ≠ 2)
        let result = mat_mul(&a, &b);
        assert!(result.is_err(), "Should return error on dimension mismatch");
    }

    #[test]
    fn test_mat_mul_identity() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let m = vec![vec![3.0, 7.0], vec![2.0, 5.0]];
        let result = mat_mul(&identity, &m).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[i][j] - m[i][j]).abs() < EPS, "I·M should equal M");
            }
        }
    }

    // ------------------------------------------------------------------
    // Rotation3 tests
    // ------------------------------------------------------------------

    #[test]
    fn test_rotation_identity_apply() {
        let r = Rotation3::identity();
        let v = [1.0, 2.0, 3.0];
        let rv = r.apply(v);
        for (a, b) in v.iter().zip(rv.iter()) {
            assert!(
                (a - b).abs() < EPS,
                "Identity rotation should preserve vector"
            );
        }
    }

    #[test]
    fn test_rotation_axis_angle_z_90_rotates_x_to_y() {
        let r = Rotation3::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let x = [1.0, 0.0, 0.0];
        let y = r.apply(x);
        assert!(
            (y[0] - 0.0).abs() < 1e-10,
            "x component should be ≈0, got {}",
            y[0]
        );
        assert!(
            (y[1] - 1.0).abs() < 1e-10,
            "y component should be ≈1, got {}",
            y[1]
        );
        assert!(
            (y[2] - 0.0).abs() < 1e-10,
            "z component should be ≈0, got {}",
            y[2]
        );
    }

    #[test]
    fn test_rotation_euler_zero_is_identity() {
        let r = Rotation3::from_euler_xyz(0.0, 0.0, 0.0);
        let v = [1.0, 2.0, 3.0];
        let rv = r.apply(v);
        for (a, b) in v.iter().zip(rv.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Zero Euler angles should give identity"
            );
        }
    }

    #[test]
    fn test_rotation_quaternion_roundtrip() {
        let original = Rotation3::from_axis_angle([1.0, 1.0, 0.0], PI / 3.0);
        let (w, x, y, z) = original.to_quaternion();
        let reconstructed = Rotation3::from_quaternion(w, x, y, z).unwrap();
        let v = [1.0, 0.0, 0.0];
        let v1 = original.apply(v);
        let v2 = reconstructed.apply(v);
        for i in 0..3 {
            assert!(
                (v1[i] - v2[i]).abs() < 1e-9,
                "Quaternion roundtrip failed at component {i}: {v1:?} vs {v2:?}"
            );
        }
    }

    #[test]
    fn test_rotation_compose_associativity() {
        let r1 = Rotation3::from_axis_angle([1.0, 0.0, 0.0], PI / 4.0);
        let r2 = Rotation3::from_axis_angle([0.0, 1.0, 0.0], PI / 6.0);
        let v = [1.0, 2.0, 3.0];
        // (R1 R2) v = R1 (R2 v)
        let composed = r1.compose(&r2).apply(v);
        let sequential = r1.apply(r2.apply(v));
        for i in 0..3 {
            assert!(
                (composed[i] - sequential[i]).abs() < 1e-10,
                "Compose mismatch at {i}: {composed:?} vs {sequential:?}"
            );
        }
    }

    #[test]
    fn test_rotation_inverse_compose_identity() {
        let r = Rotation3::from_axis_angle([1.0, 2.0, 3.0], 0.7);
        let r_inv = r.inverse();
        let identity_approx = r_inv.compose(&r);
        let id = Rotation3::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (identity_approx.matrix[i][j] - id.matrix[i][j]).abs() < 1e-10,
                    "R^{{-1}} R should be identity at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_rotation_is_valid_true_for_valid_rotation() {
        let r = Rotation3::from_axis_angle([0.0, 1.0, 0.0], PI / 5.0);
        assert!(r.is_valid(), "Valid rotation should pass is_valid()");
    }

    // ------------------------------------------------------------------
    // Spherical harmonics tests
    // ------------------------------------------------------------------

    #[test]
    fn test_sph_harm_l0_m0_is_constant() {
        let expected = 1.0 / (2.0 * PI.sqrt());
        for (theta, phi) in [(0.0, 0.0), (1.0, 2.0), (PI / 2.0, PI), (0.3, 4.5)] {
            let val = sph_harm(0, 0, theta, phi);
            assert!(
                (val - expected).abs() < EPS,
                "Y_0^0 should be constant {expected}, got {val} at (θ={theta}, φ={phi})"
            );
        }
    }

    #[test]
    fn test_spherical_harmonics_returns_correct_count() {
        for max_deg in 0..=2 {
            let vals = spherical_harmonics(0.5, 1.0, max_deg);
            let expected_count = (max_deg + 1) * (max_deg + 1);
            assert_eq!(
                vals.len(),
                expected_count,
                "spherical_harmonics({max_deg}) should return {expected_count} values"
            );
        }
    }

    #[test]
    fn test_sph_harm_l1_m0_at_north_pole() {
        // At θ=0, Y_1^0 = sqrt(3/(4π)) * cos(0) = sqrt(3/(4π))
        let expected = (3.0 / (4.0 * PI)).sqrt();
        let val = sph_harm(1, 0, 0.0, 0.0);
        assert!(
            (val - expected).abs() < EPS,
            "Y_1^0 at north pole should be {expected}, got {val}"
        );
    }

    #[test]
    fn test_sph_harm_unsupported_degree_returns_zero() {
        // l=3 is not implemented, should return 0
        let val = sph_harm(3, 0, 0.5, 1.0);
        assert_eq!(val, 0.0, "Unsupported degree should return 0");
    }

    #[test]
    fn test_sph_harm_l2_m0_equator() {
        // At θ = π/2, cos(θ) = 0, so Y_2^0 = sqrt(5/(16π)) * (0 - 1) = -sqrt(5/(16π))
        let expected = -(5.0_f64 / (16.0 * PI)).sqrt();
        let val = sph_harm(2, 0, PI / 2.0, 0.0);
        assert!(
            (val - expected).abs() < 1e-10,
            "Y_2^0 at equator should be {expected}, got {val}"
        );
    }
}
