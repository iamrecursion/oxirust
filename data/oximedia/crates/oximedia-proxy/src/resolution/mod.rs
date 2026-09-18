//! Resolution management module.

pub mod manager;
pub mod switcher;

pub use manager::{ProxyResolution, ProxyVariant, ResolutionManager};
pub use switcher::ResolutionSwitcher;
