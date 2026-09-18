//! OxiEphemeris `bodies` crate: reference-frame mathematics and the
//! apparent-place pipeline for solar-system bodies and fixed stars.
//!
//! `no_std` by design (enable the `std` feature for `std`-only
//! conveniences). All float math in library paths goes through [`libm`].
//!
//! # Modules
//!
//! - [`math`] — 3-vectors, 3×3 matrices, the IERS `R1`/`R2`/`R3` rotation
//!   convention, and spherical/Cartesian helpers.
//! - [`frames`] — IAU 2006 frame bias + precession (Fukushima–Williams
//!   parameterization), IAU `2000A_R06` nutation (the full series,
//!   [`frames::nutation_iau2000a`], the default; and a truncation of
//!   IAU 2000B accuracy class — not the IAU 2000B series itself, see
//!   [`frames::nutation_iau2000a_truncated`] — selected through
//!   [`frames::NutationModel`]), IAU 2006 mean obliquity, and
//!   equatorial ↔ ecliptic transformations.
//! - [`apparent`] — the apparent-place pipeline on a JPL DE ephemeris:
//!   light-time iteration, solar gravitational deflection, relativistic
//!   annual aberration, frame rotation, spherical output with optional
//!   daily speeds (USNO Circular 179 §1.3; Kaplan et al. 1989, AJ 97,
//!   1197).
//!
//! # Example (requires a DE file and the `std` feature of
//! `oxiephemeris-de` for the loader)
//!
//! ```no_run
//! use oxiephemeris_bodies::apparent::{apparent, Frame, Options, Target};
//! use oxiephemeris_core::time::JulianDate;
//! use oxiephemeris_de::DeFile;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let bytes = std::fs::read("data/de440/linux_p1550p2650.440")?;
//! let de = DeFile::parse(&bytes)?;
//! let mut opts = Options::default();
//! opts.frame = Frame::EclipticTrueOfDate; // SE-style apparent longitude
//! opts.with_speed = true;
//! let pos = apparent(&de, Target::Mars, JulianDate::from_f64(2_460_000.5), opts)?;
//! println!("lon = {} rad, dist = {} AU", pos.lon_rad, pos.r_au);
//! # Ok(())
//! # }
//! ```
//!
//! # Clean-room provenance
//!
//! Everything here is implemented from published standards and papers —
//! never from SOFA, ERFA, or Swiss Ephemeris source code:
//!
//! - IERS Conventions (2010), IERS Technical Note 36 ("TN36"), chapter 5
//!   (transformation between ITRS and GCRS), including the electronic
//!   tables `tab5.3a.txt` / `tab5.3b.txt` hosted by the IERS Conventions
//!   Centre, <https://iers-conventions.obspm.fr/>.
//! - Capitaine, Wallace & Chapront (2003), A&A 412, 567 — the P03
//!   precession solution adopted as IAU 2006 precession.
//! - Hilton et al. (2006), Celest. Mech. Dyn. Astron. 94, 351 — report of
//!   the IAU WG on precession and the ecliptic that recommended P03.
//! - McCarthy & Luzum (2003), Celest. Mech. Dyn. Astron. 85, 37 — the
//!   IAU 2000B abridged-nutation concept and its accuracy criterion.
//! - Simon et al. (1994), A&A 282, 663 — fundamental (Delaunay) arguments.
//!
//! # Time argument
//!
//! All frame functions take `t`, Julian centuries since J2000.0 TT, i.e.
//! `t = (TT - 2000 January 1d 12h TT) in days / 36525` (TN36 eq. 5.2).
//! Formally several of the underlying developments are functions of TDB,
//! but TT is used in practice: the resulting error in the precession
//! quantity `psi_A` is periodic with an annual period and an amplitude of
//! 2.7e-9 arcsec (TN36 §5.6.4), and the CIP location error from using TT
//! in the nutation arguments is below 0.01 microarcsec (TN36 §5.7.2) —
//! both negligible at this crate's accuracy targets.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod apparent;
pub mod frames;
pub mod math;
pub mod sidereal;
pub mod star;
pub mod topocentric;

pub use apparent::{
    apparent_topocentric, BodiesError, BodyPosition, Center, Frame, Options, SphericalRates, Target,
};
pub use frames::NutationModel;
pub use sidereal::{equation_of_equinoxes, era, gast_iau2006, gmst_iau2006};
pub use topocentric::{Eop, Observer, TopocentricObserver};
