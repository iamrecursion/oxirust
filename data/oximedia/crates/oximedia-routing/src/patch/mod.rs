//! Virtual patch bay module for input/output management.

pub mod bay;
pub mod input;
pub mod output;

pub use bay::{Patch, PatchBay, PatchError};
pub use input::{InputId, InputManager, PatchInput, SourceType};
pub use output::{DestinationType, OutputId, OutputManager, PatchOutput};
