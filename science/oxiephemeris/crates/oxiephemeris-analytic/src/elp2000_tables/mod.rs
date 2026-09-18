//! Generated ELP2000-82B tables — regenerate with
//! `cargo run -p xtask -- gen-elp` (see the file docs for the record
//! provenance, baked-in fit corrections and truncation bookkeeping).

mod main_problem;
mod planetary1;
mod planetary2;
mod zeta;

pub(crate) use main_problem::{MAIN_DIST, MAIN_LAT, MAIN_LON};
pub(crate) use planetary1::{
    PLAN1_DIST_T0, PLAN1_DIST_T1, PLAN1_LAT_T0, PLAN1_LAT_T1, PLAN1_LON_T0, PLAN1_LON_T1,
};
pub(crate) use planetary2::{
    PLAN2_DIST_T0, PLAN2_DIST_T1, PLAN2_LAT_T0, PLAN2_LAT_T1, PLAN2_LON_T0, PLAN2_LON_T1,
};
pub(crate) use zeta::{
    ZETA_DIST_T0, ZETA_DIST_T1, ZETA_DIST_T2, ZETA_LAT_T0, ZETA_LAT_T1, ZETA_LAT_T2, ZETA_LON_T0,
    ZETA_LON_T1, ZETA_LON_T2,
};
