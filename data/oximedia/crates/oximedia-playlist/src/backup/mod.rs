//! Backup playlist failover and filler management.

pub mod failover;
pub mod filler;

pub use failover::{FailoverManager, FailoverStrategy};
pub use filler::{FillerContent, FillerManager};
