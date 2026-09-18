//! Multi-channel management and routing.

pub mod manager;
pub mod router;

pub use manager::{Channel, ChannelManager};
pub use router::{ChannelRouter, RouteConfig};
