//! `OxiEphemeris` `astro` crate: the astrology layer — chart angles, house
//! systems, aspects, lunar nodes/apogee, and the sidereal zodiac
//! (ayanamshas).
//!
//! Pure spherical trigonometry and published mean-element polynomials on
//! top of [`oxiephemeris_bodies`]; no data files. `no_std` by design
//! (enable the `std` feature for `std`-only conveniences). All float math
//! in library paths goes through [`libm`].
//!
//! # Modules
//!
//! - [`angles`] — Ascendant, Midheaven (MC), Vertex, East Point from
//!   local apparent sidereal time, geographic latitude, and the true
//!   obliquity of date.
//! - [`houses`] — house-cusp computation: Placidus, Koch, Whole Sign,
//!   Equal, Porphyry, Regiomontanus, Campanus.
//! - [`aspects`] — aspect detection with configurable orbs and
//!   applying/separating classification via daily speeds.
//! - [`nodes`] — mean and true (osculating) lunar node and apogee.
//! - [`ayanamsha`] — sidereal-zodiac offsets (Fagan/Bradley, Lahiri, …)
//!   from documented definitions.
//! - [`zodiac`] — the twelve tropical signs, their element/modality, and
//!   longitude → sign/degree/DMS decomposition.
//! - [`motion`] — direct/retrograde/stationary classification from a
//!   daily longitude speed.
//! - [`dignities`] — essential dignities (domicile/exaltation/triplicity/
//!   term/face, detriment/fall/peregrine) and element/modality balance.
//! - [`parts`] — chart sect (day/night) and the Arabic Parts (Lots).
//! - [`midpoints`] — near/far ecliptic midpoints and composite-chart
//!   midpoint mapping.
//! - [`declination`] — ecliptic→equatorial declination, antiscia, and the
//!   parallel/contraparallel/out-of-bounds declination aspects.
//! - [`synastry`] — cross-aspects between two body sets (synastry,
//!   transits, progressions).
//! - [`stars`] — a committed Hipparcos bright-star subset feeding the
//!   apparent-star pipeline of `oxiephemeris_bodies::star`.
//!
//! # Clean-room provenance
//!
//! Everything here is implemented from published definitions and papers —
//! never from Swiss Ephemeris or SOFA/ERFA source code. Behavioral
//! compatibility with SE is limited to API shape and documented output
//! conventions.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod angles;
pub mod aspects;
pub mod ayanamsha;
pub mod declination;
pub mod dignities;
pub mod houses;
pub mod midpoints;
pub mod motion;
pub mod nodes;
pub mod parts;
pub mod stars;
mod stars_table;
pub mod synastry;
pub mod zodiac;

pub use angles::{ascendant, east_point, local_sidereal_time, mc, ramc, vertex, AnglesError};
pub use declination::DeclinationAspect;
pub use dignities::{EssentialDignity, Planet, RulershipScheme};
pub use houses::{cusps, HouseSystem, HousesError};
pub use midpoints::midpoint;
pub use motion::MotionState;
pub use parts::Sect;
pub use synastry::{cross_aspects_into, for_each_cross_aspect, CrossHit};
pub use zodiac::{Element, Modality, Sign, SignPosition};
