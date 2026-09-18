//! Proxy link management module.

pub mod database;
pub mod manager;
pub mod statistics;
pub mod verify;

pub use database::{LinkDatabase, ProxyLinkRecord};
pub use manager::{ProxyLink, ProxyLinkManager};
pub use statistics::{CodecStatistics, LinkStatistics, Statistics};
pub use verify::{ProxyVerifier, VerificationReport};
