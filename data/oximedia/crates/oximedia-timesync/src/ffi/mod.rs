//! FFI for system clock adjustment.

pub mod clock_adjust;

pub use clock_adjust::{adjust_system_clock, get_system_time};
