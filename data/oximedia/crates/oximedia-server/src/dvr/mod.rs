//! DVR (Digital Video Recording) and time-shifting module.

mod buffer;
mod manager;
mod storage;

pub use buffer::{DvrBuffer, DvrConfig};
pub use manager::{DvrManager, TimeShiftRequest};
pub use storage::{DvrSegment, DvrStorage};
