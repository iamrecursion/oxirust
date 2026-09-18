//! Audio embedding and de-embedding module for SDI.

pub mod audio;
pub mod deembed;

pub use audio::{AudioEmbedder, EmbedChannel, EmbedError};
pub use deembed::{AudioDeembedder, DeembedChannel, DeembedError};
