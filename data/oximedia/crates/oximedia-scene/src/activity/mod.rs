//! Activity recognition from motion patterns.
//!
//! This module provides activity recognition algorithms:
//!
//! - **General activity recognition**: Walking, running, sitting, etc.
//! - **Sports activity recognition**: Sport-specific actions

pub mod recognize;
pub mod sports;

pub use recognize::{ActivityRecognizer, ActivityType, RecognizedActivity};
pub use sports::{SportsActivity, SportsActivityRecognizer, SportsType};
