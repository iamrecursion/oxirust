//! REST API endpoints.

pub mod auth;
pub mod collections;
pub mod media;
pub mod search;
pub mod stats;
pub mod streaming;
pub mod transcode;
pub mod users;

pub use auth::*;
pub use collections::*;
pub use media::*;
pub use search::*;
pub use stats::*;
pub use transcode::*;
pub use users::*;
