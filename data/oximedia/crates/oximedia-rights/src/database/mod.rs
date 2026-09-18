//! Database layer for rights management

pub mod query;
pub mod storage;

pub use storage::{RightsDatabase, RightsPool};
