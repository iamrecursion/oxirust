//! Vector Base Amplitude Panning (VBAP) — 2D and 3D speaker array panning.
//!
//! VBAP computes amplitude gains for a set of loudspeakers such that the auditory
//! image appears at the desired azimuth (and elevation for 3D).  The algorithm
//! finds the pair (2D) or triplet (3D) of speakers that bracket the target
//! direction, inverts the speaker-vector matrix, and normalises the resulting
//! gains so the total power is preserved.
//!
//! Reference: Pulkki, V. (1997). "Virtual Sound Source Positioning Using Vector
//! Base Amplitude Panning." JAES 45(6).
//!
//! # Coordinate convention
//! - Azimuth: 0 = front, increasing CCW (i.e., 90 = left, 180 = back, 270 = right)
//! - Elevation: 0 = horizontal plane, +90 = above, −90 = below
//! - All angles in degrees unless otherwise noted.

use crate::SpatialError;

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single loudspeaker in the array, identified by its direction and output channel.
#[derive(Debug, Clone)]
pub struct Speaker {
    /// Azimuth in degrees (0 = front, 90 = left, 180 = back, 270 = right).
    pub azimuth_deg: f32,
    /// Elevation in degrees (0 = horizontal, +90 = above, −90 = below).
    pub elevation_deg: f32,
    /// Index into the output channel buffer (0-based).
    pub channel_index: usize,
}

/// An adjacent pair of speakers used for 2D VBAP.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerPair {
    /// Index into the speaker array for the first speaker.
    pub speaker_a: usize,
    /// Index into the speaker array for the second speaker.
    pub speaker_b: usize,
}

/// A 2-D VBAP panner for a ring of loudspeakers at (approximately) the same elevation.
///
/// Build via [`VbapPanner::new`]; then call [`VbapPanner::pan`] to get per-speaker gains.
#[derive(Debug, Clone)]
pub struct VbapPanner {
    /// The full speaker array.
    pub speakers: Vec<Speaker>,
    /// Adjacent speaker pairs in azimuth order.
    pub pairs: Vec<SpeakerPair>,
    /// Inverse 2×2 matrices for each pair (row-major, [[row0col0, row0col1], [row1col0, row1col1]]).
    pub matrices: Vec<[[f32; 2]; 2]>,
}

/// A triplet of speakers for 3D VBAP.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerTriplet {
    /// Indices into the speaker array for the three speakers.
    pub indices: [usize; 3],
}

/// A 3-D VBAP panner for a full-sphere loudspeaker layout.
///
/// Build via [`VbapPanner3d::new`]; then call [`VbapPanner3d::pan`] to get gains.
#[derive(Debug, Clone)]
pub struct VbapPanner3d {
    /// The full speaker array.
    pub speakers: Vec<Speaker>,
    /// Triangulated speaker triplets covering the sphere.
    pub triplets: Vec<SpeakerTriplet>,
    /// Inverse 3×3 matrices for each triplet.
    pub matrices: Vec<[[f32; 3]; 3]>,
}

// ─── Matrix helpers ───────────────────────────────────────────────────────────

/// Compute the inverse of a 2×2 matrix.
///
/// Returns `None` if the determinant is smaller than `1e-6` (degenerate / collinear).
pub fn invert_2x2(m: [[f32; 2]; 2]) -> Option<[[f32; 2]; 2]> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-6 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ])
}

/// Compute the inverse of a 3×3 matrix via cofactor expansion.
///
/// Returns `None` if |det| < `1e-6`.
pub fn invert_3x3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    // Cofactors (transposed = adjugate).
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() < 1e-6 {
        return None;
    }
    let inv_det = 1.0 / det;

    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    // Inverse = (1/det) * adjugate = (1/det) * cofactor matrix transposed.
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

// ─── Angle utilities ─────────────────────────────────────────────────────────

/// Normalise an angle in degrees to [0, 360).
#[allow(dead_code)]
fn normalise_angle_deg(deg: f32) -> f32 {
    let mut a = deg % 360.0;
    if a < 0.0 {
        a += 360.0;
    }
    a
}

/// Angular difference a→b going CCW, result in [0, 360).
#[allow(dead_code)]
fn angular_diff_ccw(a_deg: f32, b_deg: f32) -> f32 {
    normalise_angle_deg(b_deg - a_deg)
}

// ─── Speaker Cartesian helpers ────────────────────────────────────────────────

/// Convert a speaker's azimuth+elevation to a unit Cartesian vector.
fn speaker_to_unit_vec(spk: &Speaker) -> [f32; 3] {
    let az = spk.azimuth_deg.to_radians();
    let el = spk.elevation_deg.to_radians();
    let cos_el = el.cos();
    [cos_el * az.cos(), cos_el * az.sin(), el.sin()]
}

// ─── VbapPanner (2D) ─────────────────────────────────────────────────────────

impl VbapPanner {
    /// Construct a 2D VBAP panner from a set of speakers.
    ///
    /// Speakers are sorted by azimuth and all adjacent pairs (including the
    /// wrap-around pair) are computed.  Each pair's 2×2 speaker-vector matrix
    /// is inverted.  Pairs whose matrix is degenerate (collinear speakers) are
    /// silently skipped.
    ///
    /// Returns an error if fewer than two valid pairs can be formed.
    pub fn new(speakers: Vec<Speaker>) -> Result<Self, SpatialError> {
        if speakers.len() < 2 {
            return Err(SpatialError::InvalidConfig(
                "VBAP requires at least 2 speakers".into(),
            ));
        }

        // Sort speaker indices by azimuth.
        let mut indices: Vec<usize> = (0..speakers.len()).collect();
        indices.sort_by(|&a, &b| {
            speakers[a]
                .azimuth_deg
                .partial_cmp(&speakers[b].azimuth_deg)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pairs = Vec::new();
        let mut matrices = Vec::new();

        let n = indices.len();
        for i in 0..n {
            let ia = indices[i];
            let ib = indices[(i + 1) % n];
            let spk_a = &speakers[ia];
            let spk_b = &speakers[ib];

            // 2×2 matrix columns = unit azimuth vectors of each speaker.
            let az_a = spk_a.azimuth_deg.to_radians();
            let az_b = spk_b.azimuth_deg.to_radians();
            let col_a = [az_a.cos(), az_a.sin()];
            let col_b = [az_b.cos(), az_b.sin()];

            // Matrix M = [col_a | col_b] (each column is a speaker vector).
            let mat: [[f32; 2]; 2] = [[col_a[0], col_b[0]], [col_a[1], col_b[1]]];

            if let Some(inv) = invert_2x2(mat) {
                pairs.push(SpeakerPair {
                    speaker_a: ia,
                    speaker_b: ib,
                });
                matrices.push(inv);
            }
        }

        if pairs.is_empty() {
            return Err(SpatialError::InvalidConfig(
                "All speaker pairs are degenerate (collinear)".into(),
            ));
        }

        Ok(Self {
            speakers,
            pairs,
            matrices,
        })
    }

    /// Compute per-speaker gains for a source at `azimuth_deg` (elevation = 0).
    ///
    /// The returned `Vec` has one entry per speaker in the order supplied to
    /// [`Self::new`].  Gains are non-negative and energy-normalised
    /// (sum of squares = 1).  Speakers outside the active pair get gain 0.
    pub fn pan(&self, azimuth_deg: f32) -> Vec<f32> {
        let n_speakers = self.speakers.len();
        let mut gains = vec![0.0_f32; n_speakers];

        let az = azimuth_deg.to_radians();
        let target = [az.cos(), az.sin()];

        // Find the pair whose angular span contains the target azimuth.
        // We iterate all pairs and choose the one that gives both non-negative
        // gains (the target lies inside the span).
        let mut best_pair_idx: Option<usize> = None;
        let mut best_g = [0.0_f32; 2];

        for (pair_idx, (_pair, inv)) in self.pairs.iter().zip(self.matrices.iter()).enumerate() {
            // g = inv * target
            let g1 = inv[0][0] * target[0] + inv[0][1] * target[1];
            let g2 = inv[1][0] * target[0] + inv[1][1] * target[1];

            if g1 >= -1e-6 && g2 >= -1e-6 {
                // Both gains non-negative → target is inside this pair's span.
                best_pair_idx = Some(pair_idx);
                best_g = [g1.max(0.0), g2.max(0.0)];
                break;
            }

            // Keep the pair with the smallest maximum negative gain as fallback.
            let min_neg = g1.min(g2);
            if let Some(prev_idx) = best_pair_idx {
                let (_, prev_inv) = (&self.pairs[prev_idx], &self.matrices[prev_idx]);
                let pg1 = prev_inv[0][0] * target[0] + prev_inv[0][1] * target[1];
                let pg2 = prev_inv[1][0] * target[0] + prev_inv[1][1] * target[1];
                let prev_min_neg = pg1.min(pg2);
                if min_neg > prev_min_neg {
                    best_pair_idx = Some(pair_idx);
                    best_g = [g1, g2];
                }
            } else {
                best_pair_idx = Some(pair_idx);
                best_g = [g1, g2];
            }
        }

        let Some(pair_idx) = best_pair_idx else {
            return gains;
        };

        let pair = &self.pairs[pair_idx];
        let g1 = best_g[0].max(0.0);
        let g2 = best_g[1].max(0.0);

        // Energy normalisation: divide by vector length so sum-of-squares = 1.
        let len = (g1 * g1 + g2 * g2).sqrt();
        if len > 1e-10 {
            gains[pair.speaker_a] = g1 / len;
            gains[pair.speaker_b] = g2 / len;
        }

        gains
    }

    /// Convenience: sort speakers' azimuths and return the sorted azimuth list.
    pub fn speaker_azimuths(&self) -> Vec<f32> {
        self.speakers.iter().map(|s| s.azimuth_deg).collect()
    }
}

// ─── Delaunay triangulation on the unit sphere ──────────────────────────────

/// A triangle in the Delaunay triangulation, referencing point indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DelaunayTriangle {
    /// Vertex indices (into the points array).
    verts: [usize; 3],
}

/// Cross product of two 3-D vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-D vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalise a 3-D vector to unit length. Returns zero vector if too small.
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Compute the circumcentre of a spherical triangle defined by three unit vectors.
///
/// The circumcentre on the sphere is the normalised cross product of the edge
/// bisector planes, or equivalently, the unit vector equidistant (in arc length)
/// from all three vertices. We use the approach: circumcentre = normalise((B-A) x (C-A))
/// where the sign is chosen so that the circumcentre is on the same hemisphere as the
/// vertices.
fn spherical_circumcentre(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<[f32; 3]> {
    // Normal to the plane through A, B, C.
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = cross3(ab, ac);
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-10 {
        return None; // Degenerate (collinear points)
    }
    let mut cc = [n[0] / len, n[1] / len, n[2] / len];

    // Ensure circumcentre is on the same side as the centroid of the triangle.
    let centroid = normalize3([a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]);
    if dot3(cc, centroid) < 0.0 {
        cc = [-cc[0], -cc[1], -cc[2]];
    }

    Some(cc)
}

/// Check if point `p` lies inside the circumsphere (circumscribed circle on the sphere)
/// of the triangle with vertices `a, b, c`.
///
/// On the sphere, a point is inside the circumcircle if its angular distance to the
/// circumcentre is less than the angular distance from any vertex to the circumcentre.
fn in_circumsphere(a: [f32; 3], b: [f32; 3], c: [f32; 3], p: [f32; 3]) -> bool {
    let Some(cc) = spherical_circumcentre(a, b, c) else {
        return false;
    };
    // Circumradius: arc distance from circumcentre to vertex A (using dot product).
    let cos_radius = dot3(cc, a);
    let cos_dist_p = dot3(cc, p);
    // p is inside if its cosine distance is greater (closer) than the radius.
    cos_dist_p > cos_radius - 1e-6
}

/// Perform Delaunay triangulation on a set of unit-sphere points.
///
/// Uses a Bowyer-Watson-style incremental insertion algorithm adapted for the sphere.
/// Returns a list of triangles as index triples.
///
/// The algorithm:
/// 1. Start with an initial tetrahedron of "super" points that enclose all real points.
/// 2. For each point, find all triangles whose circumsphere contains the point.
/// 3. Remove those triangles, find the boundary polygon (the "hole").
/// 4. Create new triangles connecting the new point to each boundary edge.
/// 5. After all points are inserted, remove triangles that reference super points.
fn spherical_delaunay(points: &[[f32; 3]]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }

    // Super-triangle vertices: 4 points of a tetrahedron on the unit sphere
    // placed far from any realistic speaker position.
    let super_pts: [[f32; 3]; 4] = [
        [0.0, 0.0, 1.0],          // north pole
        [0.0, 0.0, -1.0],         // south pole
        [1.0, 0.0, 0.0],          // +X
        [-0.5, 0.866_025_4, 0.0], // 120 deg in XY plane
    ];

    // Extended points array: real points followed by super points.
    let total = n + 4;
    let mut all_pts: Vec<[f32; 3]> = Vec::with_capacity(total);
    all_pts.extend_from_slice(points);
    all_pts.extend_from_slice(&super_pts);

    // Initial triangulation from the 4 super points (tetrahedron faces).
    let s0 = n;
    let s1 = n + 1;
    let s2 = n + 2;
    let s3 = n + 3;
    let mut triangles: Vec<DelaunayTriangle> = vec![
        DelaunayTriangle {
            verts: [s0, s2, s3],
        },
        DelaunayTriangle {
            verts: [s0, s3, s1],
        },
        DelaunayTriangle {
            verts: [s0, s1, s2],
        },
        DelaunayTriangle {
            verts: [s1, s3, s2],
        },
    ];

    // Insert each real point.
    for pt_idx in 0..n {
        let p = all_pts[pt_idx];

        // Find all triangles whose circumsphere contains p.
        let mut bad_indices: Vec<usize> = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            let va = all_pts[tri.verts[0]];
            let vb = all_pts[tri.verts[1]];
            let vc = all_pts[tri.verts[2]];
            if in_circumsphere(va, vb, vc, p) {
                bad_indices.push(ti);
            }
        }

        // Collect boundary edges of the hole (edges that appear in only one bad triangle).
        let mut edge_count: Vec<([usize; 2], usize)> = Vec::new();
        for &bi in &bad_indices {
            let v = triangles[bi].verts;
            let edges = [[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]];
            for e in &edges {
                let canonical = if e[0] < e[1] {
                    [e[0], e[1]]
                } else {
                    [e[1], e[0]]
                };
                if let Some(entry) = edge_count.iter_mut().find(|(k, _)| *k == canonical) {
                    entry.1 += 1;
                } else {
                    edge_count.push((canonical, 1));
                }
            }
        }

        let boundary_edges: Vec<[usize; 2]> = edge_count
            .iter()
            .filter(|(_, count)| *count == 1)
            .map(|(e, _)| *e)
            .collect();

        // Remove bad triangles (in reverse order to preserve indices).
        bad_indices.sort_unstable();
        for &bi in bad_indices.iter().rev() {
            triangles.swap_remove(bi);
        }

        // Create new triangles from the boundary edges to the new point.
        for edge in &boundary_edges {
            let new_tri = DelaunayTriangle {
                verts: [pt_idx, edge[0], edge[1]],
            };
            triangles.push(new_tri);
        }
    }

    // Remove any triangles that reference super points.
    triangles.retain(|tri| tri.verts.iter().all(|&v| v < n));

    triangles.iter().map(|t| t.verts).collect()
}

// ─── VbapPanner3d (3D) ───────────────────────────────────────────────────────

impl VbapPanner3d {
    /// Construct a 3D VBAP panner using Delaunay triangulation for optimal
    /// speaker triplet selection.
    ///
    /// This is the preferred constructor for irregular speaker layouts. The
    /// Delaunay triangulation on the unit sphere ensures that every direction
    /// is covered by exactly one triplet (no gaps, no overlaps), producing
    /// well-conditioned gain matrices.
    ///
    /// Falls back to the heuristic method if Delaunay produces no triangles
    /// (e.g., all speakers are coplanar).
    ///
    /// Returns an error if fewer than 3 speakers are supplied.
    pub fn new(speakers: Vec<Speaker>) -> Result<Self, SpatialError> {
        if speakers.len() < 3 {
            return Err(SpatialError::InvalidConfig(
                "3D VBAP requires at least 3 speakers".into(),
            ));
        }

        // Convert speakers to unit vectors on the sphere.
        let unit_vecs: Vec<[f32; 3]> = speakers.iter().map(|s| speaker_to_unit_vec(s)).collect();

        // Attempt Delaunay triangulation.
        let delaunay_tris = spherical_delaunay(&unit_vecs);

        let mut triplets = Vec::new();
        let mut matrices = Vec::new();

        // Build triplets from Delaunay output.
        for tri in &delaunay_tris {
            let ia = tri[0];
            let ib = tri[1];
            let ic = tri[2];
            let va = unit_vecs[ia];
            let vb = unit_vecs[ib];
            let vc = unit_vecs[ic];

            let mat: [[f32; 3]; 3] = [
                [va[0], vb[0], vc[0]],
                [va[1], vb[1], vc[1]],
                [va[2], vb[2], vc[2]],
            ];

            if let Some(inv) = invert_3x3(mat) {
                triplets.push(SpeakerTriplet {
                    indices: [ia, ib, ic],
                });
                matrices.push(inv);
            }
        }

        // If Delaunay produced no valid triplets, fall back to heuristic method.
        if triplets.is_empty() {
            return Self::from_heuristic(speakers);
        }

        Ok(Self {
            speakers,
            triplets,
            matrices,
        })
    }

    /// Construct a 3D VBAP panner using the heuristic sequential/stride method.
    ///
    /// This is the fallback when Delaunay triangulation fails (e.g., coplanar speakers).
    pub fn from_heuristic(speakers: Vec<Speaker>) -> Result<Self, SpatialError> {
        if speakers.len() < 3 {
            return Err(SpatialError::InvalidConfig(
                "3D VBAP requires at least 3 speakers".into(),
            ));
        }

        let mut indices: Vec<usize> = (0..speakers.len()).collect();
        indices.sort_by(|&a, &b| {
            let diff_az = speakers[a]
                .azimuth_deg
                .partial_cmp(&speakers[b].azimuth_deg)
                .unwrap_or(std::cmp::Ordering::Equal);
            if diff_az == std::cmp::Ordering::Equal {
                speakers[a]
                    .elevation_deg
                    .partial_cmp(&speakers[b].elevation_deg)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                diff_az
            }
        });

        let n = indices.len();
        let mut triplets = Vec::new();
        let mut matrices = Vec::new();

        let try_triplet = |ia: usize,
                           ib: usize,
                           ic: usize,
                           trips: &mut Vec<SpeakerTriplet>,
                           mats: &mut Vec<[[f32; 3]; 3]>| {
            let va = speaker_to_unit_vec(&speakers[ia]);
            let vb = speaker_to_unit_vec(&speakers[ib]);
            let vc = speaker_to_unit_vec(&speakers[ic]);

            let mat: [[f32; 3]; 3] = [
                [va[0], vb[0], vc[0]],
                [va[1], vb[1], vc[1]],
                [va[2], vb[2], vc[2]],
            ];

            if let Some(inv) = invert_3x3(mat) {
                trips.push(SpeakerTriplet {
                    indices: [ia, ib, ic],
                });
                mats.push(inv);
            }
        };

        for i in 0..n {
            let ia = indices[i];
            let ib = indices[(i + 1) % n];
            let ic = indices[(i + 2) % n];
            try_triplet(ia, ib, ic, &mut triplets, &mut matrices);
        }

        if n >= 4 {
            for i in 0..n {
                let ia = indices[i];
                let ib = indices[(i + 2) % n];
                let ic = indices[(i + n / 2) % n];
                if ia != ib && ib != ic && ia != ic {
                    try_triplet(ia, ib, ic, &mut triplets, &mut matrices);
                }
            }
        }

        if triplets.is_empty() {
            return Err(SpatialError::InvalidConfig(
                "No valid speaker triplets found (all degenerate)".into(),
            ));
        }

        Ok(Self {
            speakers,
            triplets,
            matrices,
        })
    }

    /// Compute per-speaker gains for a 3D source direction.
    ///
    /// Returns a `Vec` with one entry per speaker.  Gains are energy-normalised.
    pub fn pan(&self, azimuth_deg: f32, elevation_deg: f32) -> Vec<f32> {
        let n_speakers = self.speakers.len();
        let mut gains = vec![0.0_f32; n_speakers];

        let az = azimuth_deg.to_radians();
        let el = elevation_deg.to_radians();
        let cos_el = el.cos();
        let target = [cos_el * az.cos(), cos_el * az.sin(), el.sin()];

        let mut best_triplet_idx: Option<usize> = None;
        let mut best_g = [0.0_f32; 3];
        let mut best_min_neg = f32::NEG_INFINITY;

        for (tri_idx, (triplet, inv)) in self.triplets.iter().zip(self.matrices.iter()).enumerate()
        {
            let g0 = inv[0][0] * target[0] + inv[0][1] * target[1] + inv[0][2] * target[2];
            let g1 = inv[1][0] * target[0] + inv[1][1] * target[1] + inv[1][2] * target[2];
            let g2 = inv[2][0] * target[0] + inv[2][1] * target[1] + inv[2][2] * target[2];

            if g0 >= -1e-6 && g1 >= -1e-6 && g2 >= -1e-6 {
                best_triplet_idx = Some(tri_idx);
                best_g = [g0.max(0.0), g1.max(0.0), g2.max(0.0)];
                break;
            }

            let min_neg = g0.min(g1).min(g2);
            if min_neg > best_min_neg {
                best_min_neg = min_neg;
                best_triplet_idx = Some(tri_idx);
                best_g = [g0, g1, g2];
            }
            let _ = triplet;
        }

        let Some(tri_idx) = best_triplet_idx else {
            return gains;
        };

        let triplet = &self.triplets[tri_idx];
        let g0 = best_g[0].max(0.0);
        let g1 = best_g[1].max(0.0);
        let g2 = best_g[2].max(0.0);

        let len = (g0 * g0 + g1 * g1 + g2 * g2).sqrt();
        if len > 1e-10 {
            gains[triplet.indices[0]] = g0 / len;
            gains[triplet.indices[1]] = g1 / len;
            gains[triplet.indices[2]] = g2 / len;
        }

        gains
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a standard stereo pair: L at 30°, R at −30° (= 330°).
    fn stereo_speakers() -> Vec<Speaker> {
        vec![
            Speaker {
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
                channel_index: 0,
            },
            Speaker {
                azimuth_deg: 330.0,
                elevation_deg: 0.0,
                channel_index: 1,
            },
        ]
    }

    /// Build a standard 5-speaker ring (L, C, R, Rs, Ls).
    fn five_speaker_ring() -> Vec<Speaker> {
        vec![
            Speaker {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                channel_index: 0,
            }, // C
            Speaker {
                azimuth_deg: 30.0,
                elevation_deg: 0.0,
                channel_index: 1,
            }, // L
            Speaker {
                azimuth_deg: 110.0,
                elevation_deg: 0.0,
                channel_index: 2,
            }, // Ls
            Speaker {
                azimuth_deg: 250.0,
                elevation_deg: 0.0,
                channel_index: 3,
            }, // Rs
            Speaker {
                azimuth_deg: 330.0,
                elevation_deg: 0.0,
                channel_index: 4,
            }, // R
        ]
    }

    // ── invert_2x2 ──────────────────────────────────────────────────────────

    #[test]
    fn test_invert_2x2_identity() {
        let m = [[1.0_f32, 0.0], [0.0, 1.0]];
        let inv = invert_2x2(m).expect("identity is invertible");
        // inv of I = I
        assert!((inv[0][0] - 1.0).abs() < 1e-5);
        assert!((inv[1][1] - 1.0).abs() < 1e-5);
        assert!((inv[0][1]).abs() < 1e-5);
        assert!((inv[1][0]).abs() < 1e-5);
    }

    #[test]
    fn test_invert_2x2_known_matrix() {
        // [[2, 1], [5, 3]] → inv = [[3, -1], [-5, 2]]
        let m = [[2.0_f32, 1.0], [5.0, 3.0]];
        let inv = invert_2x2(m).expect("invertible");
        assert!((inv[0][0] - 3.0).abs() < 1e-5);
        assert!((inv[0][1] - (-1.0)).abs() < 1e-5);
        assert!((inv[1][0] - (-5.0)).abs() < 1e-5);
        assert!((inv[1][1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_invert_2x2_singular_returns_none() {
        let m = [[1.0_f32, 2.0], [2.0, 4.0]];
        assert!(invert_2x2(m).is_none(), "Singular matrix must return None");
    }

    #[test]
    fn test_invert_2x2_round_trip() {
        let m = [[3.0_f32, 1.0], [2.0, 4.0]];
        let inv = invert_2x2(m).expect("invertible");
        // M * inv ≈ I
        let r00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let r11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];
        assert!((r00 - 1.0).abs() < 1e-5);
        assert!((r11 - 1.0).abs() < 1e-5);
    }

    // ── invert_3x3 ──────────────────────────────────────────────────────────

    #[test]
    fn test_invert_3x3_identity() {
        let m = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(m).expect("identity invertible");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i][j] - expected).abs() < 1e-5,
                    "inv[{i}][{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn test_invert_3x3_singular_returns_none() {
        let m = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert!(invert_3x3(m).is_none(), "Singular 3x3 must return None");
    }

    #[test]
    fn test_invert_3x3_round_trip() {
        let m = [[1.0_f32, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]];
        let inv = invert_3x3(m).expect("invertible");
        // Compute M * inv and check it's ≈ I.
        for i in 0..3 {
            for j in 0..3 {
                let sum: f32 = (0..3).map(|k| m[i][k] * inv[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-4,
                    "M*inv[{i}][{j}] = {sum}, expected {expected}"
                );
            }
        }
    }

    // ── VbapPanner construction ──────────────────────────────────────────────

    #[test]
    fn test_new_too_few_speakers_returns_error() {
        let spks = vec![Speaker {
            azimuth_deg: 0.0,
            elevation_deg: 0.0,
            channel_index: 0,
        }];
        assert!(VbapPanner::new(spks).is_err());
    }

    #[test]
    fn test_new_stereo_succeeds() {
        let panner = VbapPanner::new(stereo_speakers());
        assert!(
            panner.is_ok(),
            "Stereo speaker pair should construct successfully"
        );
    }

    #[test]
    fn test_new_five_speaker_ring_succeeds() {
        let panner = VbapPanner::new(five_speaker_ring());
        assert!(panner.is_ok());
    }

    // ── VbapPanner::pan ──────────────────────────────────────────────────────

    #[test]
    fn test_pan_centre_front_equal_gains_stereo() {
        // For a symmetric stereo pair (L=30°, R=330°), panning at 0° (front)
        // should give equal (or very similar) gains to both speakers.
        let panner = VbapPanner::new(stereo_speakers()).expect("ok");
        let gains = panner.pan(0.0);
        assert_eq!(gains.len(), 2);
        let diff = (gains[0] - gains[1]).abs();
        assert!(
            diff < 0.15,
            "Front pan should give near-equal gains: L={}, R={}",
            gains[0],
            gains[1]
        );
    }

    #[test]
    fn test_pan_energy_normalised() {
        let panner = VbapPanner::new(five_speaker_ring()).expect("ok");
        for az in [0.0_f32, 30.0, 90.0, 180.0, 270.0, 315.0] {
            let gains = panner.pan(az);
            let energy: f32 = gains.iter().map(|g| g * g).sum();
            assert!(
                (energy - 1.0).abs() < 0.05,
                "Energy not normalised at az={az}: energy={energy}"
            );
        }
    }

    #[test]
    fn test_pan_gains_non_negative() {
        let panner = VbapPanner::new(five_speaker_ring()).expect("ok");
        for az in [0.0_f32, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] {
            let gains = panner.pan(az);
            for (i, &g) in gains.iter().enumerate() {
                assert!(g >= 0.0, "Negative gain at az={az}, speaker {i}: g={g}");
            }
        }
    }

    #[test]
    fn test_pan_exact_speaker_direction_full_gain() {
        // Panning exactly at a speaker's azimuth should give that speaker gain ≈ 1.0.
        let panner = VbapPanner::new(five_speaker_ring()).expect("ok");
        let gains = panner.pan(0.0); // Centre speaker is at 0°.
        let max_gain = gains.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // At or near a speaker direction, at least one speaker should have high gain.
        assert!(
            max_gain > 0.9,
            "At speaker direction, max gain should be high, got {max_gain}"
        );
    }

    #[test]
    fn test_pan_returns_correct_length() {
        let panner = VbapPanner::new(five_speaker_ring()).expect("ok");
        let gains = panner.pan(45.0);
        assert_eq!(gains.len(), 5);
    }

    #[test]
    fn test_pan_at_most_two_active_speakers() {
        // VBAP 2D: at most 2 speakers should have non-zero gain.
        let panner = VbapPanner::new(five_speaker_ring()).expect("ok");
        for az in [15.0_f32, 70.0, 190.0, 280.0] {
            let gains = panner.pan(az);
            let active: usize = gains.iter().filter(|&&g| g > 1e-4).count();
            assert!(
                active <= 2,
                "2D VBAP should activate at most 2 speakers at az={az}, but got {active}"
            );
        }
    }

    #[test]
    fn test_pan_left_speaker_louder_at_left_azimuth() {
        // A source at 30° (left) should make the left speaker (index 0) loudest.
        let panner = VbapPanner::new(stereo_speakers()).expect("ok");
        let gains = panner.pan(30.0);
        assert!(
            gains[0] >= gains[1],
            "Left speaker should be >= right at 30° az: L={}, R={}",
            gains[0],
            gains[1]
        );
    }

    // ── VbapPanner3d ────────────────────────────────────────────────────────

    fn build_cube_speakers() -> Vec<Speaker> {
        // 8 corners of a cube projected onto the sphere.
        let angles = [
            (45.0_f32, 35.26_f32),
            (135.0, 35.26),
            (225.0, 35.26),
            (315.0, 35.26),
            (45.0, -35.26),
            (135.0, -35.26),
            (225.0, -35.26),
            (315.0, -35.26),
        ];
        angles
            .iter()
            .enumerate()
            .map(|(i, &(az, el))| Speaker {
                azimuth_deg: az,
                elevation_deg: el,
                channel_index: i,
            })
            .collect()
    }

    #[test]
    fn test_vbap3d_new_succeeds() {
        let spks = build_cube_speakers();
        assert!(VbapPanner3d::new(spks).is_ok());
    }

    #[test]
    fn test_vbap3d_too_few_speakers_error() {
        let spks = vec![
            Speaker {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                channel_index: 0,
            },
            Speaker {
                azimuth_deg: 90.0,
                elevation_deg: 0.0,
                channel_index: 1,
            },
        ];
        assert!(VbapPanner3d::new(spks).is_err());
    }

    #[test]
    fn test_vbap3d_pan_returns_correct_length() {
        let panner = VbapPanner3d::new(build_cube_speakers()).expect("ok");
        let gains = panner.pan(45.0, 30.0);
        assert_eq!(gains.len(), 8);
    }

    #[test]
    fn test_vbap3d_energy_normalised() {
        let panner = VbapPanner3d::new(build_cube_speakers()).expect("ok");
        for (az, el) in [(0.0_f32, 0.0_f32), (45.0, 30.0), (90.0, -20.0)] {
            let gains = panner.pan(az, el);
            let energy: f32 = gains.iter().map(|g| g * g).sum();
            // energy may be 0 if no triplet contains the direction; otherwise ≈ 1.
            if energy > 1e-6 {
                assert!(
                    (energy - 1.0).abs() < 0.05,
                    "3D energy not normalised at ({az},{el}): {energy}"
                );
            }
        }
    }

    #[test]
    fn test_vbap3d_gains_non_negative() {
        let panner = VbapPanner3d::new(build_cube_speakers()).expect("ok");
        for (az, el) in [(45.0_f32, 20.0_f32), (200.0, -10.0)] {
            let gains = panner.pan(az, el);
            for (i, &g) in gains.iter().enumerate() {
                assert!(
                    g >= 0.0,
                    "3D VBAP gain negative at ({az},{el}), spk {i}: {g}"
                );
            }
        }
    }

    #[test]
    fn test_vbap3d_at_most_three_active_speakers() {
        let panner = VbapPanner3d::new(build_cube_speakers()).expect("ok");
        let gains = panner.pan(45.0, 35.26);
        let active: usize = gains.iter().filter(|&&g| g > 1e-4).count();
        assert!(
            active <= 3,
            "3D VBAP should activate at most 3 speakers, got {active}"
        );
    }

    // ── normalise_angle_deg ─────────────────────────────────────────────────

    #[test]
    fn test_normalise_angle_neg() {
        let a = normalise_angle_deg(-30.0);
        assert!((a - 330.0).abs() < 1e-4);
    }

    #[test]
    fn test_normalise_angle_over_360() {
        let a = normalise_angle_deg(400.0);
        assert!((a - 40.0).abs() < 1e-4);
    }

    // ── Delaunay triangulation ─────────────────────────────────────────────

    /// Build an irregular speaker layout (not on a regular grid).
    fn irregular_speakers() -> Vec<Speaker> {
        // Deliberately non-uniform: 7 speakers at random-ish positions including height.
        vec![
            Speaker {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                channel_index: 0,
            },
            Speaker {
                azimuth_deg: 55.0,
                elevation_deg: 20.0,
                channel_index: 1,
            },
            Speaker {
                azimuth_deg: 120.0,
                elevation_deg: -10.0,
                channel_index: 2,
            },
            Speaker {
                azimuth_deg: 190.0,
                elevation_deg: 15.0,
                channel_index: 3,
            },
            Speaker {
                azimuth_deg: 260.0,
                elevation_deg: -25.0,
                channel_index: 4,
            },
            Speaker {
                azimuth_deg: 310.0,
                elevation_deg: 40.0,
                channel_index: 5,
            },
            Speaker {
                azimuth_deg: 80.0,
                elevation_deg: -45.0,
                channel_index: 6,
            },
        ]
    }

    /// Build a 22.2 immersive layout (subset).
    fn immersive_22_speakers() -> Vec<Speaker> {
        let positions = [
            (0.0, 0.0),
            (30.0, 0.0),
            (60.0, 0.0),
            (90.0, 0.0),
            (135.0, 0.0),
            (180.0, 0.0),
            (225.0, 0.0),
            (270.0, 0.0),
            (300.0, 0.0),
            (330.0, 0.0),
            (0.0, 30.0),
            (45.0, 30.0),
            (135.0, 30.0),
            (225.0, 30.0),
            (315.0, 30.0),
            (0.0, -30.0),
            (90.0, -30.0),
            (180.0, -30.0),
            (270.0, -30.0),
            (0.0, 90.0),
            (45.0, -60.0),
            (225.0, -60.0),
        ];
        positions
            .iter()
            .enumerate()
            .map(|(i, &(az, el))| Speaker {
                azimuth_deg: az,
                elevation_deg: el,
                channel_index: i,
            })
            .collect()
    }

    #[test]
    fn test_delaunay_irregular_layout_constructs() {
        let spks = irregular_speakers();
        let result = VbapPanner3d::new(spks);
        assert!(
            result.is_ok(),
            "Delaunay with irregular layout should succeed"
        );
    }

    #[test]
    fn test_delaunay_produces_triplets() {
        let spks = irregular_speakers();
        let panner = VbapPanner3d::new(spks).expect("ok");
        assert!(
            !panner.triplets.is_empty(),
            "Delaunay should produce at least one triplet"
        );
    }

    #[test]
    fn test_delaunay_triplet_indices_valid() {
        let spks = irregular_speakers();
        let n = spks.len();
        let panner = VbapPanner3d::new(spks).expect("ok");
        for (ti, triplet) in panner.triplets.iter().enumerate() {
            for &idx in &triplet.indices {
                assert!(
                    idx < n,
                    "Triplet {ti} has out-of-bounds index {idx} (n={n})"
                );
            }
            // All indices should be distinct.
            assert_ne!(
                triplet.indices[0], triplet.indices[1],
                "Triplet {ti} has duplicate"
            );
            assert_ne!(
                triplet.indices[1], triplet.indices[2],
                "Triplet {ti} has duplicate"
            );
            assert_ne!(
                triplet.indices[0], triplet.indices[2],
                "Triplet {ti} has duplicate"
            );
        }
    }

    #[test]
    fn test_delaunay_irregular_pan_energy_normalised() {
        let panner = VbapPanner3d::new(irregular_speakers()).expect("ok");
        let test_directions = [
            (0.0_f32, 0.0_f32),
            (45.0, 10.0),
            (120.0, -15.0),
            (200.0, 20.0),
            (300.0, -10.0),
            (80.0, -30.0),
        ];
        for (az, el) in test_directions {
            let gains = panner.pan(az, el);
            let energy: f32 = gains.iter().map(|g| g * g).sum();
            if energy > 1e-6 {
                assert!(
                    (energy - 1.0).abs() < 0.1,
                    "Energy not normalised at ({az},{el}): {energy}"
                );
            }
        }
    }

    #[test]
    fn test_delaunay_irregular_pan_non_negative() {
        let panner = VbapPanner3d::new(irregular_speakers()).expect("ok");
        for az in (0..360).step_by(30) {
            for el in [-30, 0, 30] {
                let gains = panner.pan(az as f32, el as f32);
                for (i, &g) in gains.iter().enumerate() {
                    assert!(g >= 0.0, "Negative gain at ({az},{el}), spk {i}: {g}");
                }
            }
        }
    }

    #[test]
    fn test_delaunay_irregular_at_most_three_active() {
        let panner = VbapPanner3d::new(irregular_speakers()).expect("ok");
        for az in (0..360).step_by(45) {
            let gains = panner.pan(az as f32, 0.0);
            let active: usize = gains.iter().filter(|&&g| g > 1e-4).count();
            assert!(
                active <= 3,
                "3D VBAP should activate at most 3 speakers at az={az}, got {active}"
            );
        }
    }

    #[test]
    fn test_delaunay_at_speaker_position_high_gain() {
        let spks = irregular_speakers();
        let panner = VbapPanner3d::new(spks.clone()).expect("ok");
        // Panning at a speaker position should give that speaker high gain.
        for (i, spk) in spks.iter().enumerate() {
            let gains = panner.pan(spk.azimuth_deg, spk.elevation_deg);
            let max_gain = gains.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                max_gain > 0.5,
                "At speaker {i} ({},{}), max gain should be high, got {max_gain}",
                spk.azimuth_deg,
                spk.elevation_deg
            );
        }
    }

    #[test]
    fn test_delaunay_22_speaker_immersive() {
        let spks = immersive_22_speakers();
        let panner = VbapPanner3d::new(spks).expect("ok");
        assert!(
            !panner.triplets.is_empty(),
            "22-speaker layout should triangulate"
        );
        // Test panning at various directions.
        for az in (0..360).step_by(60) {
            for el in [-30, 0, 30] {
                let gains = panner.pan(az as f32, el as f32);
                let energy: f32 = gains.iter().map(|g| g * g).sum();
                assert!(energy > 0.0, "Should have energy at ({az},{el})");
            }
        }
    }

    #[test]
    fn test_delaunay_cube_speakers_same_as_before() {
        // Cube layout should also work with Delaunay.
        let panner = VbapPanner3d::new(build_cube_speakers()).expect("ok");
        assert!(!panner.triplets.is_empty());
        let gains = panner.pan(45.0, 35.26);
        let energy: f32 = gains.iter().map(|g| g * g).sum();
        if energy > 1e-6 {
            assert!((energy - 1.0).abs() < 0.1, "Energy={energy}");
        }
    }

    #[test]
    fn test_heuristic_fallback_works() {
        // from_heuristic should still work.
        let panner = VbapPanner3d::from_heuristic(build_cube_speakers()).expect("ok");
        assert!(!panner.triplets.is_empty());
    }

    // ── VBAP gain matrix cache tests ─────────────────────────────────────────
    //
    // The triangulation (Delaunay) and inverse matrices are computed once at
    // construction time and cached in `VbapPanner3d::triplets` and
    // `VbapPanner3d::matrices`.  The `pan()` method only reads these fields;
    // it never re-triangulates.  The tests below verify:
    //
    //   1. Reproducibility  — two identical `pan()` calls give bitwise-equal output.
    //   2. Correctness       — cached result equals the result from a freshly-built
    //                          panner (i.e. re-triangulating from scratch gives the same gains).

    #[test]
    fn test_vbap_cache_reproducible_2d() {
        // Two calls to `pan()` with the same azimuth on a 2D panner must give
        // identical results (no mutable state in `pan()`).
        let panner = VbapPanner::new(five_speaker_ring()).expect("ok");

        for az in [0.0_f32, 30.0, 75.0, 110.0, 180.0, 260.0, 315.0] {
            let g1 = panner.pan(az);
            let g2 = panner.pan(az);
            assert_eq!(
                g1, g2,
                "2D pan({az}) should be bitwise reproducible: {g1:?} vs {g2:?}"
            );
        }
    }

    #[test]
    fn test_vbap_cache_reproducible_3d() {
        // Two calls to `pan()` with the same (az, el) on a 3D panner must give
        // identical results.
        let panner = VbapPanner3d::new(build_cube_speakers()).expect("ok");

        let test_positions = [
            (0.0_f32, 0.0_f32),
            (45.0, 20.0),
            (135.0, -15.0),
            (270.0, 35.0),
            (310.0, -30.0),
        ];
        for (az, el) in test_positions {
            let g1 = panner.pan(az, el);
            let g2 = panner.pan(az, el);
            assert_eq!(g1, g2, "3D pan({az},{el}) should be bitwise reproducible");
        }
    }

    #[test]
    fn test_vbap_cache_correct_2d_vs_fresh_panner() {
        // The gains from the cached panner must match those from a fresh panner
        // built with identical speaker data.  Verifies that the cached matrices
        // produce the same result as re-computing them.
        let speakers = five_speaker_ring();
        let panner_a = VbapPanner::new(speakers.clone()).expect("ok");
        let panner_b = VbapPanner::new(speakers).expect("ok");

        for az in (0..360).step_by(5).map(|i| i as f32) {
            let ga = panner_a.pan(az);
            let gb = panner_b.pan(az);
            for (i, (&a, &b)) in ga.iter().zip(gb.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "2D cache correct: az={az}, spk={i}: a={a}, b={b}"
                );
            }
        }
    }

    #[test]
    fn test_vbap_cache_correct_3d_vs_fresh_panner() {
        // Same correctness test for the 3D panner.
        let speakers = irregular_speakers();
        let panner_a = VbapPanner3d::new(speakers.clone()).expect("ok");
        let panner_b = VbapPanner3d::new(speakers).expect("ok");

        let test_positions: Vec<(f32, f32)> = (0..100)
            .map(|i| {
                let az = (i as f32 * 3.6) % 360.0;
                let el = ((i as f32 * 1.8).sin() * 30.0) as f32;
                (az, el)
            })
            .collect();

        for (az, el) in &test_positions {
            let ga = panner_a.pan(*az, *el);
            let gb = panner_b.pan(*az, *el);
            for (i, (&a, &b)) in ga.iter().zip(gb.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "3D cache correct: az={az}, el={el}, spk={i}: a={a}, b={b}"
                );
            }
        }
    }

    #[test]
    fn test_vbap_cache_matrices_pre_computed_at_construction() {
        // The panner's matrices field must be non-empty after construction,
        // confirming that pre-computation happened at `new()` time.
        let panner_2d = VbapPanner::new(five_speaker_ring()).expect("ok");
        assert!(
            !panner_2d.matrices.is_empty(),
            "2D panner matrices must be pre-computed at construction"
        );

        let panner_3d = VbapPanner3d::new(build_cube_speakers()).expect("ok");
        assert!(
            !panner_3d.matrices.is_empty(),
            "3D panner matrices must be pre-computed at construction"
        );
        assert_eq!(
            panner_3d.triplets.len(),
            panner_3d.matrices.len(),
            "Number of triplets must equal number of cached inverse matrices"
        );
    }

    #[test]
    fn test_spherical_delaunay_basic() {
        // Four points forming a tetrahedron — should produce some triangles.
        let pts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 1.0],
            [0.943, 0.0, -0.333],
            [-0.471, 0.816, -0.333],
            [-0.471, -0.816, -0.333],
        ];
        let tris = spherical_delaunay(&pts);
        // Should produce at least 1 triangle from 4 non-degenerate points.
        assert!(
            !tris.is_empty(),
            "Tetrahedron should produce at least 1 triangle, got 0"
        );
        // All indices should be in range.
        for tri in &tris {
            for &v in tri {
                assert!(v < 4, "Index {v} out of range for 4 points");
            }
        }
    }

    #[test]
    fn test_spherical_delaunay_all_indices_valid() {
        let pts: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        let tris = spherical_delaunay(&pts);
        for tri in &tris {
            for &v in tri {
                assert!(v < pts.len(), "Index {v} out of range");
            }
        }
    }
}
