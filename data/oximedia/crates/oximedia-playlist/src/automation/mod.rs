//! Automated playout engine and triggers.

pub mod playout;
pub mod preroll;
pub mod triggers;

pub use playout::PlayoutEngine;
pub use preroll::{PostRoll, PreRoll, RollManager};
pub use triggers::{Trigger, TriggerManager, TriggerType};
