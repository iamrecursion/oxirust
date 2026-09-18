//! Edit list and composition time support.
//!
//! Provides MP4 edit lists and composition time offsets for frame-accurate editing.

#![forbid(unsafe_code)]

pub mod composition;
pub mod list;

pub use composition::{
    CompositionTimeBuilder, CompositionTimeOffset, CompositionTimeTable, CompositionTimeUtils,
    FrameReorderer,
};
pub use list::{EditEntry, EditList, EditListBuilder, EditListPresets};
