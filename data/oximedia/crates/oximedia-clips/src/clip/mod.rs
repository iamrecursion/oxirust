//! Core clip types and management.

pub mod core;
pub mod metadata;
pub mod subclip;
pub mod version;

pub use self::core::{Clip, ClipId};
pub use metadata::ClipMetadata;
pub use subclip::SubClip;
pub use version::{ClipVersion, VersionId};
