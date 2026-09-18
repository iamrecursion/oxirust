//! Generated VSOP87E tables, one module per body — regenerate with
//! `cargo run -p xtask -- gen-vsop87` (see the file docs for the
//! truncation bookkeeping). Each module exposes
//! `TABLES: [[&[(f64, f64, f64)]; 6]; 3]`, indexed `[coordinate][alpha]`
//! with `(A, B, C)` terms of `A·cos(B + C·T)·T^alpha`.

pub(crate) mod earth;
pub(crate) mod jupiter;
pub(crate) mod mars;
pub(crate) mod mercury;
pub(crate) mod neptune;
pub(crate) mod saturn;
pub(crate) mod sun;
pub(crate) mod uranus;
pub(crate) mod venus;
