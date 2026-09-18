//! DC offset removal.

pub mod remover;

pub use remover::{detect_dc_offset, remove_dc_simple, DcRemover};
