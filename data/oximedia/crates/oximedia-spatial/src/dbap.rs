//! Distance-Based Amplitude Panning (DBAP).
//!
//! DBAP is a panning technique for arbitrary, non-regular loudspeaker arrays.
//! Unlike VBAP (which requires explicit triangulation and a convex hull), DBAP
//! distributes energy across **all** speakers using an inverse-power-law of the
//! 3-D Euclidean distance between the virtual source and each loudspeaker.
//!
//! # Algorithm
//!
//! For each speaker *i* at position **p_i** and source at **s**:
//!
//! ```text
//! d_i = ‖ s − p_i ‖           (Euclidean distance)
//! w_i = 1 / d_i ^ rolloff     (raw weight; rolloff typically 6 for 6 dB/oct)
//! g_i = w_i / √(Σ w_j²)      (energy-normalised gain)
//! ```
//!
//! The result is energy-preserving: `Σ g_i² = 1`.
//!
//! # References
//! Lossius, Baltazar & De la Hogue (2009). *DBAP — Distance-Based Amplitude Panning*,
//! Proc. International Computer Music Conference (ICMC).

use crate::SpatialError;

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single loudspeaker in a DBAP layout.
#[derive(Debug, Clone)]
pub struct DbapSpeaker {
    /// 3-D position of the loudspeaker (any consistent unit, e.g. metres).
    pub position: [f32; 3],
    /// Unique identifier for this speaker (used as the key in the gain output).
    pub id: u32,
}

/// DBAP panner for an arbitrary set of loudspeakers.
///
/// Construct with [`DbapPanner::new`] and call [`DbapPanner::get_gains`] to
/// obtain per-speaker amplitude gains for a virtual source position.
#[derive(Debug, Clone)]
pub struct DbapPanner {
    /// Loudspeaker positions and identifiers.
    pub speakers: Vec<DbapSpeaker>,
    /// Distance-attenuation exponent.
    ///
    /// - `rolloff = 1.0` → amplitude falls as 1/d (3 dB/oct in free field)
    /// - `rolloff = 2.0` → amplitude falls as 1/d² (classic inverse-square pressure)
    /// - `rolloff = 6.0` → amplitude falls as 1/d⁶ (very directional)
    ///
    /// The default in the original DBAP paper is **6** (corresponding to
    /// pressure roll-off in a room).
    pub rolloff: f32,
}

// ─── DbapSpeaker ─────────────────────────────────────────────────────────────

impl DbapSpeaker {
    /// Create a speaker at the given 3-D position with the given ID.
    pub fn new(id: u32, position: [f32; 3]) -> Self {
        Self { position, id }
    }
}

// ─── DbapPanner ──────────────────────────────────────────────────────────────

impl DbapPanner {
    /// Create a new DBAP panner.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] when fewer than 2 speakers are
    /// provided (at least 2 are required for any meaningful panning).
    pub fn new(speakers: Vec<DbapSpeaker>, rolloff: f32) -> Result<Self, SpatialError> {
        if speakers.len() < 2 {
            return Err(SpatialError::InvalidConfig(
                "DBAP requires at least 2 speakers".into(),
            ));
        }
        Ok(Self { speakers, rolloff })
    }

    /// Compute per-speaker amplitude gains for a virtual source at `source_pos`.
    ///
    /// Returns a `Vec<(id, gain)>` in the same order as the speaker list.
    /// All gains are ≥ 0 and energy-normalised so that `Σ gain² ≈ 1`.
    ///
    /// If the source coincides exactly with a speaker position, that speaker
    /// receives a gain of 1.0 and all others receive 0.0.
    pub fn get_gains(&self, source_pos: &[f32; 3]) -> Vec<(u32, f32)> {
        // Compute squared distances and check for exact coincidence.
        let dist_sq: Vec<f32> = self
            .speakers
            .iter()
            .map(|sp| euclidean_dist_sq(source_pos, &sp.position))
            .collect();

        // If source coincides with a speaker (distance² < ε), route all energy there.
        if let Some(exact_idx) = dist_sq.iter().position(|&d| d < 1e-12) {
            return self
                .speakers
                .iter()
                .enumerate()
                .map(|(i, sp)| (sp.id, if i == exact_idx { 1.0 } else { 0.0 }))
                .collect();
        }

        // Compute raw DBAP weights: w_i = 1 / d_i ^ rolloff
        // Use sqrt(d²) = d for the distance, then raise to rolloff.
        let rolloff = self.rolloff.max(0.0);
        let raw_weights: Vec<f32> = dist_sq
            .iter()
            .map(|&d2| {
                let d = d2.sqrt();
                // Avoid division by zero with a tiny floor.
                let d_safe = d.max(1e-6);
                1.0 / d_safe.powf(rolloff)
            })
            .collect();

        // Energy normalisation: divide each weight by the root of the sum of squares.
        let sum_sq: f32 = raw_weights.iter().map(|&w| w * w).sum();
        if sum_sq < 1e-30 {
            // Degenerate — return equal gains.
            let equal = (1.0_f32 / self.speakers.len() as f32).sqrt();
            return self.speakers.iter().map(|sp| (sp.id, equal)).collect();
        }

        let norm = sum_sq.sqrt();
        self.speakers
            .iter()
            .zip(raw_weights.iter())
            .map(|(sp, &w)| (sp.id, w / norm))
            .collect()
    }

    /// Compute the total energy of the returned gains.
    ///
    /// For a correctly normalised panner this should always equal 1.0.
    pub fn total_energy(&self, source_pos: &[f32; 3]) -> f32 {
        self.get_gains(source_pos).iter().map(|&(_, g)| g * g).sum()
    }
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────

/// Squared Euclidean distance between two 3-D points.
fn euclidean_dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quad_panner(rolloff: f32) -> DbapPanner {
        // Four speakers at the corners of a 2×2 square in the XY plane.
        let speakers = vec![
            DbapSpeaker::new(0, [-1.0, -1.0, 0.0]),
            DbapSpeaker::new(1, [1.0, -1.0, 0.0]),
            DbapSpeaker::new(2, [1.0, 1.0, 0.0]),
            DbapSpeaker::new(3, [-1.0, 1.0, 0.0]),
        ];
        DbapPanner::new(speakers, rolloff).expect("valid config")
    }

    fn assert_near(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: expected ≈ {b}, got {a} (tol {tol})"
        );
    }

    fn find_gain(gains: &[(u32, f32)], id: u32) -> f32 {
        gains
            .iter()
            .find(|&&(i, _)| i == id)
            .map(|&(_, g)| g)
            .unwrap_or(0.0)
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_dbap_new_valid() {
        let panner = make_quad_panner(6.0);
        assert_eq!(panner.speakers.len(), 4);
        assert_near(panner.rolloff, 6.0, 1e-6, "rolloff");
    }

    #[test]
    fn test_dbap_new_single_speaker_fails() {
        let result = DbapPanner::new(vec![DbapSpeaker::new(0, [0.0, 0.0, 0.0])], 6.0);
        assert!(result.is_err(), "Single speaker should fail");
    }

    // ── Energy normalisation ─────────────────────────────────────────────────

    #[test]
    fn test_dbap_energy_normalised_centre() {
        let panner = make_quad_panner(6.0);
        let energy = panner.total_energy(&[0.0, 0.0, 0.0]);
        assert_near(energy, 1.0, 1e-5, "energy at centre");
    }

    #[test]
    fn test_dbap_energy_normalised_off_centre() {
        let panner = make_quad_panner(2.0);
        for pos in [[0.3_f32, 0.7, 0.0], [-0.5, 0.2, 0.1], [0.9, -0.8, 0.5]] {
            let energy = panner.total_energy(&pos);
            assert_near(energy, 1.0, 1e-4, "energy off-centre");
        }
    }

    #[test]
    fn test_dbap_all_gains_non_negative() {
        let panner = make_quad_panner(6.0);
        let gains = panner.get_gains(&[0.1, 0.2, 0.0]);
        for &(id, g) in &gains {
            assert!(g >= 0.0, "Speaker {id} gain should be ≥ 0, got {g}");
        }
    }

    // ── Directional behaviour ────────────────────────────────────────────────

    #[test]
    fn test_dbap_nearest_speaker_receives_most_gain() {
        let panner = make_quad_panner(6.0);
        // Source very close to speaker 2 (top-right corner).
        let gains = panner.get_gains(&[0.95, 0.95, 0.0]);
        let g2 = find_gain(&gains, 2);
        let g0 = find_gain(&gains, 0);
        assert!(
            g2 > g0,
            "Nearest speaker should receive most gain: g2={g2}, g0={g0}"
        );
    }

    #[test]
    fn test_dbap_centre_gives_equal_gains_to_symmetric_speakers() {
        // For a symmetric quad layout, the centre should give equal gains.
        let panner = make_quad_panner(6.0);
        let gains = panner.get_gains(&[0.0, 0.0, 0.0]);
        let g0 = find_gain(&gains, 0);
        let g1 = find_gain(&gains, 1);
        let g2 = find_gain(&gains, 2);
        let g3 = find_gain(&gains, 3);
        assert_near(g0, g1, 1e-4, "symmetric gains g0=g1");
        assert_near(g1, g2, 1e-4, "symmetric gains g1=g2");
        assert_near(g2, g3, 1e-4, "symmetric gains g2=g3");
    }

    #[test]
    fn test_dbap_exact_coincidence_routes_to_one_speaker() {
        let panner = make_quad_panner(6.0);
        // Source at exactly speaker 0's position.
        let gains = panner.get_gains(&[-1.0, -1.0, 0.0]);
        let g0 = find_gain(&gains, 0);
        assert_near(
            g0,
            1.0,
            1e-5,
            "coincident source should give gain=1 to its speaker",
        );
        for &(id, g) in &gains {
            if id != 0 {
                assert_near(g, 0.0, 1e-5, "other speakers should get 0");
            }
        }
    }

    // ── Rolloff sensitivity ──────────────────────────────────────────────────

    #[test]
    fn test_dbap_higher_rolloff_more_directional() {
        // With high rolloff the nearest speaker should get proportionally more gain.
        let p_low = make_quad_panner(0.5);
        let p_high = make_quad_panner(10.0);
        let src = [0.7_f32, 0.7, 0.0]; // near speaker 2

        let g_low = find_gain(&p_low.get_gains(&src), 2);
        let g_high = find_gain(&p_high.get_gains(&src), 2);

        assert!(
            g_high > g_low,
            "Higher rolloff should focus more gain on nearest: g_low={g_low}, g_high={g_high}"
        );
    }

    #[test]
    fn test_dbap_rolloff_zero_is_equal_spread() {
        // rolloff=0 → all weights = 1 → equal gains regardless of position.
        let panner = make_quad_panner(0.0);
        let gains = panner.get_gains(&[0.5, 0.3, 0.1]);
        let g0 = find_gain(&gains, 0);
        let g1 = find_gain(&gains, 1);
        assert_near(g0, g1, 1e-4, "rolloff=0 should give equal gains");
    }

    // ── 3-D layout ───────────────────────────────────────────────────────────

    #[test]
    fn test_dbap_3d_layout_energy_preserved() {
        let speakers = vec![
            DbapSpeaker::new(0, [0.0, 0.0, 0.0]),
            DbapSpeaker::new(1, [1.0, 0.0, 0.0]),
            DbapSpeaker::new(2, [0.0, 1.0, 0.0]),
            DbapSpeaker::new(3, [0.0, 0.0, 1.0]),
            DbapSpeaker::new(4, [1.0, 1.0, 1.0]),
        ];
        let panner = DbapPanner::new(speakers, 2.0).expect("valid");
        let energy = panner.total_energy(&[0.5, 0.5, 0.5]);
        assert_near(energy, 1.0, 1e-4, "3-D energy preserved");
    }

    #[test]
    fn test_dbap_output_length_matches_speakers() {
        let panner = make_quad_panner(6.0);
        let gains = panner.get_gains(&[0.0, 0.0, 0.0]);
        assert_eq!(gains.len(), panner.speakers.len());
    }
}
