//! Emulation support for future preservation

pub mod package;
pub mod prepare;

pub use package::{EmulationPackage, EmulationPackager};
pub use prepare::{EmulationPreparation, EmulationPreparer};
