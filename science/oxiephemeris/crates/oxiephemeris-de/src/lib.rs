//! JPL Development Ephemeris (classic binary export format) parser and
//! Chebyshev interpolation. `no_std`; the caller supplies `&[u8]`.
//!
//! # Format provenance (clean room)
//!
//! The classic binary layout is implemented from the public-domain JPL
//! Fortran export utilities and papers only:
//!
//! * `asc2eph.f` (JPL, last modified 2013-08-15) — defines the exact byte
//!   layout of the two header records and the data records
//!   (`WRITE(12,REC=1) TTL,CNAM,SS,NCON,AU,EMRAT,IPT,NUMDE,LPT,[CNAM…],RPT,TPT`).
//! * `testeph.f` (JPL, 2013-03-25) — defines interpolation (`INTERP`),
//!   record/granule lookup (`STATE`), and the `testpo` comparison semantics
//!   (`PLEPH`).
//! * E. M. Standish, "JPL Planetary and Lunar Ephemerides, DE405/LE405",
//!   JPL IOM 312.F-98-048 (1998) — export-format description.
//! * R. S. Park, W. M. Folkner, J. G. Williams, D. H. Boggs, "The JPL
//!   Planetary and Lunar Ephemerides DE440 and DE441", AJ 161, 105 (2021).
//!
//! Verification data: the public-domain JPL `testpo.XXX` golden test files.
//!
//! # Overview
//!
//! * [`DeFile::parse`] parses a classic binary DE file (for example
//!   `linux_p1550p2650.440`) from a byte slice, auto-detecting endianness.
//! * [`DeFile::series_state`] evaluates one raw Chebyshev series
//!   ([`Series`]) at a two-part Julian date, returning components and
//!   derivatives (km and km/day for bodies, radians for nutation/libration,
//!   seconds for TT-TDB; rates are per day).
//! * [`DeFile::state_km`] returns barycentric (SSB-centered) states for a
//!   [`Body`], deriving Earth and Moon from the Earth-Moon-barycenter and
//!   geocentric-Moon series.
//! * [`DeFile::testpo_value`] reproduces the `PLEPH` target/center/coordinate
//!   semantics used by the JPL `testpo` golden test files.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod cheby;
mod eval;
mod parse;
pub mod spk;

pub use parse::DeFile;
pub use spk::{SpkError, SpkFile, SpkSegment, SpkState};

/// Errors while parsing or evaluating a DE file.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeError {
    /// Header did not validate (endianness / `NCON` sanity check failed,
    /// inconsistent pointers, or unsupported coefficient counts).
    InvalidHeader,
    /// The byte slice is shorter than the header and record layout require.
    Truncated,
    /// Requested epoch is outside the file's span.
    EpochOutOfRange,
    /// The requested series has no coefficients in this file
    /// (for example TT-TDB on a plain DE440 file).
    SeriesUnavailable,
    /// Invalid caller argument (unknown `testpo` target or coordinate).
    BadArgument,
    /// A data record failed its internal epoch-bracket consistency check.
    Corrupt,
}

impl core::fmt::Display for DeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::InvalidHeader => "invalid DE header",
            Self::Truncated => "byte slice is truncated",
            Self::EpochOutOfRange => "epoch outside ephemeris span",
            Self::SeriesUnavailable => "series not present in this file",
            Self::BadArgument => "invalid argument",
            Self::Corrupt => "data record failed consistency check",
        };
        f.write_str(msg)
    }
}

impl core::error::Error for DeError {}

/// Byte order of a parsed DE file.
///
/// Classic binary DE files carry no magic number; the order is detected by
/// sanity-checking `NCON`, `SS` and `AU` under both interpretations
/// (`asc2eph.f` writes native-endian Fortran unformatted records).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Least-significant byte first (e.g. JPL `linux_*` exports).
    Little,
    /// Most-significant byte first (e.g. JPL `unxp*` SunOS-era exports).
    Big,
}

/// One raw Chebyshev-fitted series of the file, in file order.
///
/// Slots 1–13 follow the `IPT`/`LPT` pointer table of `asc2eph.f`; slots 14
/// (lunar Euler-angle rates, `RPT`) and 15 (TT-TDB, `TPT`) were added in the
/// 2013 format revision and are optional.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Series {
    /// Slot 1: Mercury, SSB-centered, km and km/day.
    Mercury,
    /// Slot 2: Venus, SSB-centered, km and km/day.
    Venus,
    /// Slot 3: Earth-Moon barycenter, SSB-centered, km and km/day.
    EarthMoonBarycenter,
    /// Slot 4: Mars, SSB-centered, km and km/day.
    Mars,
    /// Slot 5: Jupiter, SSB-centered, km and km/day.
    Jupiter,
    /// Slot 6: Saturn, SSB-centered, km and km/day.
    Saturn,
    /// Slot 7: Uranus, SSB-centered, km and km/day.
    Uranus,
    /// Slot 8: Neptune, SSB-centered, km and km/day.
    Neptune,
    /// Slot 9: Pluto, SSB-centered, km and km/day.
    Pluto,
    /// Slot 10: Moon, geocentric, km and km/day.
    Moon,
    /// Slot 11: Sun, SSB-centered, km and km/day.
    Sun,
    /// Slot 12: nutation angles (dpsi, deps), 2 components, radians.
    Nutation,
    /// Slot 13: lunar libration Euler angles (phi, theta, psi), radians.
    Libration,
    /// Slot 14: lunar Euler-angle rates (`RPT`), radians/day, if present.
    LunarEulerRates,
    /// Slot 15: TT-TDB at the geocenter (`TPT`), seconds, if present.
    TtTdb,
}

impl Series {
    /// All series in file (pointer-table) order.
    pub const ALL: &'static [Self] = &[
        Self::Mercury,
        Self::Venus,
        Self::EarthMoonBarycenter,
        Self::Mars,
        Self::Jupiter,
        Self::Saturn,
        Self::Uranus,
        Self::Neptune,
        Self::Pluto,
        Self::Moon,
        Self::Sun,
        Self::Nutation,
        Self::Libration,
        Self::LunarEulerRates,
        Self::TtTdb,
    ];

    /// Zero-based pointer-table slot of this series.
    #[must_use]
    pub const fn slot(self) -> usize {
        match self {
            Self::Mercury => 0,
            Self::Venus => 1,
            Self::EarthMoonBarycenter => 2,
            Self::Mars => 3,
            Self::Jupiter => 4,
            Self::Saturn => 5,
            Self::Uranus => 6,
            Self::Neptune => 7,
            Self::Pluto => 8,
            Self::Moon => 9,
            Self::Sun => 10,
            Self::Nutation => 11,
            Self::Libration => 12,
            Self::LunarEulerRates => 13,
            Self::TtTdb => 14,
        }
    }

    /// Number of interpolated components of this series
    /// (3 for bodies/angle triplets, 2 for nutation, 1 for TT-TDB), per the
    /// `INTERP` calls in `testeph.f`.
    #[must_use]
    pub const fn ncomp(self) -> usize {
        match self {
            Self::Nutation => 2,
            Self::TtTdb => 1,
            _ => 3,
        }
    }
}

/// Result of evaluating one series at an epoch.
///
/// Only the first [`ncomp`](Self::ncomp) entries of `value`/`rate` are
/// meaningful; the rest are zero.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesState {
    /// Interpolated components (km, radians, or seconds).
    pub value: [f64; 3],
    /// Time derivatives of `value`, per day.
    pub rate: [f64; 3],
    /// Number of meaningful components (1, 2, or 3).
    pub ncomp: usize,
}

/// Solar-system points for barycentric convenience states.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Body {
    /// Mercury.
    Mercury,
    /// Venus.
    Venus,
    /// Earth, derived: `EMB - Moon_geocentric / (1 + EMRAT)`.
    Earth,
    /// Mars.
    Mars,
    /// Jupiter.
    Jupiter,
    /// Saturn.
    Saturn,
    /// Uranus.
    Uranus,
    /// Neptune.
    Neptune,
    /// Pluto.
    Pluto,
    /// Sun.
    Sun,
    /// Moon, derived: `Earth + Moon_geocentric`.
    Moon,
    /// Earth-Moon barycenter (stored directly in the file).
    Emb,
    /// Solar-system barycenter (the origin; state is all zeros).
    Ssb,
}

/// Reads a whole DE file into memory (convenience loader).
///
/// # Errors
/// Propagates any [`std::io::Error`] from reading the file.
#[cfg(feature = "std")]
pub fn read_de_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}
