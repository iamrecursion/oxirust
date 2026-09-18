//! Embedded file attachments.
//!
//! Provides attachment support for Matroska and MP4 containers.

#![forbid(unsafe_code)]

pub mod matroska;
pub mod mp4;

pub use matroska::{MatroskaAttachment, MatroskaAttachments, MimeTypes};
pub use mp4::{Mp4Attachment, Mp4AttachmentTypes};
