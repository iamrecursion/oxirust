//! Clock synchronization for frame-accurate playout.

pub mod offset;
pub mod sync;

pub use offset::{OffsetManager, TimeOffset};
pub use sync::{ClockSource, ClockSync};
