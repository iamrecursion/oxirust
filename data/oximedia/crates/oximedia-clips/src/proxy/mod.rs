//! Proxy media management.

pub mod link;
pub mod manager;

pub use link::{ProxyLink, ProxyLinkId, ProxyQuality};
pub use manager::ProxyManager;
