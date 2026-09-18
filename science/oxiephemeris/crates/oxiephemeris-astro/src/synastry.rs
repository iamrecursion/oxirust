//! Cross-aspects between two sets of bodies — the shared engine for
//! synastry (chart A vs chart B), transits (natal vs transiting), and
//! progressions (natal vs progressed).
//!
//! # Model
//!
//! Every body in set `a` is tested against every body in set `b` with
//! [`crate::aspects::find_aspect`], reusing that module's orb policy and
//! applying/separating classification verbatim. The only new element is
//! the `a x b` iteration and the pair of indices identifying which two
//! bodies formed each hit.
//!
//! For a **transit** the caller passes the transiting bodies as `a` and
//! the natal bodies as `b` (or vice versa); for **synastry** the two
//! natal charts; for a **secondary progression** the progressed and the
//! natal positions. The relative speed used for applying/separating is
//! `speed_a - speed_b`, so when one set is treated as fixed the caller
//! passes zero speeds for it (a natal chart does not move) and the
//! motion is attributed entirely to the transiting/progressed side —
//! exactly the conventional reading.
//!
//! # `no_std`
//!
//! The core [`for_each_cross_aspect`] takes a callback and never
//! allocates, so it works in `no_std`. A caller with storage can collect
//! hits into a slice with [`cross_aspects_into`]. Each body is given as
//! a `(lon_rad, speed_rad_per_day)` pair.
//!
//! # Clean-room provenance
//!
//! Pure composition of [`crate::aspects`]; no ephemeris source was
//! consulted.

use crate::aspects::{find_aspect, AspectHit, OrbPolicy};

/// One cross-aspect: which body in each set, and the aspect hit.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossHit {
    /// Index of the body in set `a`.
    pub index_a: usize,
    /// Index of the body in set `b`.
    pub index_b: usize,
    /// The aspect between them (from `a`'s point of view: `find_aspect`
    /// is called with `a` first, so [`AspectHit::applying`] and the
    /// offset sign follow the `a`-then-`b` convention documented on
    /// [`crate::aspects`]).
    pub hit: AspectHit,
}

/// Calls `f` once for every aspecting pair `(a[i], b[j])` within the orb
/// policy.
///
/// Each body is a `(lon_rad, speed_rad_per_day)` pair. The iteration is
/// the full `a.len() * b.len()` grid in row-major order (`a` outer, `b`
/// inner).
pub fn for_each_cross_aspect(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    policy: &OrbPolicy,
    mut f: impl FnMut(CrossHit),
) {
    for (index_a, &(lon_a, speed_a)) in a.iter().enumerate() {
        for (index_b, &(lon_b, speed_b)) in b.iter().enumerate() {
            if let Some(hit) = find_aspect(lon_a, speed_a, lon_b, speed_b, policy) {
                f(CrossHit {
                    index_a,
                    index_b,
                    hit,
                });
            }
        }
    }
}

/// Collects cross-aspects into `out`, returning the filled prefix.
///
/// Stops once `out` is full; the boolean in the return tuple is `true`
/// when every hit fit (no truncation), so a caller can size `out` at
/// `a.len() * b.len()` to be certain of completeness or detect overflow
/// otherwise.
#[must_use]
pub fn cross_aspects_into<'o>(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    policy: &OrbPolicy,
    out: &'o mut [CrossHit],
) -> (&'o [CrossHit], bool) {
    let mut count = 0_usize;
    let mut complete = true;
    for_each_cross_aspect(a, b, policy, |cross| {
        if count < out.len() {
            out[count] = cross;
            count += 1;
        } else {
            complete = false;
        }
    });
    (&out[..count], complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aspects::AspectKind;
    use core::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    #[test]
    fn finds_a_known_cross_conjunction() {
        // a[0] at 100 deg, b[1] at 101 deg -> conjunction.
        let a = [(deg(100.0), 1.0)];
        let b = [(deg(200.0), 0.5), (deg(101.0), 0.0)];
        let mut hits = 0;
        let mut kind = None;
        for_each_cross_aspect(&a, &b, &OrbPolicy::default(), |cross| {
            hits += 1;
            if cross.index_a == 0 && cross.index_b == 1 {
                kind = Some(cross.hit.aspect.kind);
            }
        });
        assert_eq!(kind, Some(AspectKind::Conjunction));
        assert!(hits >= 1);
    }

    /// A valid `CrossHit` for seeding fixed-size output buffers in tests.
    fn seed_hit() -> CrossHit {
        let Some(hit) = find_aspect(0.0, 0.0, 0.0, 0.0, &OrbPolicy::default()) else {
            panic!("0 vs 0 must be a conjunction");
        };
        CrossHit {
            index_a: 0,
            index_b: 0,
            hit,
        }
    }

    #[test]
    fn collect_into_slice_and_completeness() {
        let a = [(deg(0.0), 0.0), (deg(90.0), 0.0)];
        let b = [(deg(0.0), 0.0), (deg(180.0), 0.0)];
        // a0-b0 conj, a0-b1 opp, a1-b0 square, a1-b1 square => 4 hits.
        let mut out = [seed_hit(); 8];
        let (found, complete) = cross_aspects_into(&a, &b, &OrbPolicy::default(), &mut out);
        assert_eq!(found.len(), 4);
        assert!(complete);
    }

    #[test]
    fn truncation_flag_when_buffer_too_small() {
        let a = [(deg(0.0), 0.0), (deg(90.0), 0.0)];
        let b = [(deg(0.0), 0.0), (deg(180.0), 0.0)];
        let mut out = [seed_hit(); 2];
        let (found, complete) = cross_aspects_into(&a, &b, &OrbPolicy::default(), &mut out);
        assert_eq!(found.len(), 2);
        assert!(!complete);
    }
}
