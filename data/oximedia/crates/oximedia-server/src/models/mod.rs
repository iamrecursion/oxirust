//! Data models for the media server.

pub mod collection;
pub mod media;
pub mod transcode;
pub mod upload;
pub mod user;

pub use collection::{Collection, CollectionItem};
pub use media::{Media, MediaMetadata, MediaStatus};
pub use transcode::{TranscodeJob, TranscodeStatus};
pub use upload::{MultipartUpload, UploadChunk, UploadStatus};
pub use user::{ApiKey, User, UserRole};
