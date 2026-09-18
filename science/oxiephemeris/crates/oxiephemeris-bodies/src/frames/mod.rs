//! Reference-frame transformations: IAU 2006 frame bias + precession,
//! IAU `2000A_R06` nutation (the full series and a truncation of IAU
//! 2000B accuracy class), IAU 2006 mean obliquity, ecliptic rotations.
//!
//! All functions are pure functions of `t`, Julian centuries TT since
//! J2000.0 (see the crate docs for the formal TDB vs. TT distinction —
//! negligible here per IERS TN36 §5.6.4 and §5.7.2).
//!
//! The frame chain implemented in this module (vectors transform left to
//! right; every step is a proper rotation):
//!
//! ```text
//! GCRS --PB(t)--> mean equator & equinox of date --N(t)--> true of date
//!                       |                                     |
//!                  R1(eps_A)                            R1(eps_A + deps)
//!                       v                                     v
//!            mean ecliptic of date                 true ecliptic of date
//! ```
//!
//! where `PB` ([`precession_bias_matrix`]) already contains the GCRS frame
//! bias, and `N` is the classical nutation matrix
//! ([`nutation_matrix`]).

mod nutation;
mod nutation_2000a_eps_ls;
mod nutation_2000a_eps_pl;
mod nutation_2000a_psi_ls;
mod nutation_2000a_psi_pl;
mod nutation_truncated_table;
mod obliquity;
mod precession;

pub use nutation::{
    gcrs_to_true_of_date, gcrs_to_true_of_date_with, mean_of_date_to_ecliptic, nutation_iau2000a,
    nutation_iau2000a_truncated, nutation_matrix, nutation_matrix_with, true_of_date_to_ecliptic,
    true_of_date_to_ecliptic_with, Nutation, NutationModel, NUTATION_IAU2000A_TERM_COUNTS,
};
pub use obliquity::mean_obliquity_iau2006;
pub use precession::{fw_angles_iau2006, precession_bias_matrix, FwAngles};

/// Arcseconds to radians: `pi / (180 * 3600)`.
pub(crate) const ARCSEC_TO_RAD: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// One full turn in arcseconds (`360 * 3600`), for argument reduction.
pub(crate) const TURN_ARCSEC: f64 = 1_296_000.0;
