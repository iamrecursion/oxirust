//! Stream recording module.

mod format;
mod recorder;
mod storage;

pub use format::{FormatWriter, RecordingFormat};
pub use recorder::{RecordingConfig, RecordingInfo, StreamRecorder};
pub use storage::{RecordingStorage, StorageBackend};
