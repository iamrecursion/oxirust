//! Electronic Program Guide (EPG) generation and export.

pub mod generate;
pub mod xmltv;

pub use generate::{EpgGenerator, ProgramEntry};
pub use xmltv::XmltvExporter;
