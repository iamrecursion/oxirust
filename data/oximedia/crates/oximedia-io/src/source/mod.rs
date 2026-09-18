//! Media source abstractions for reading from various inputs.
//!
//! This module provides the [`MediaSource`] trait and implementations for
//! reading media data from files, memory buffers, and other sources.

#[cfg(not(target_arch = "wasm32"))]
mod file;
mod memory;
mod traits;

#[cfg(not(target_arch = "wasm32"))]
pub use file::FileSource;
pub use memory::MemorySource;
pub use traits::MediaSource;
