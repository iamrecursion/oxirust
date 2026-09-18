//! Annotations and notes system.

pub mod annotation;
pub mod thread;

pub use annotation::{Annotation, Note, NoteId};
pub use thread::{NoteThread, ThreadId};
