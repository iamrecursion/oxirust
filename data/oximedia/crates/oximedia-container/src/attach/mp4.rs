//! MP4 attachments.
//!
//! Provides attachment handling for MP4 containers.

#![forbid(unsafe_code)]

use bytes::Bytes;

/// An attached file in MP4 (stored in udta).
#[derive(Debug, Clone)]
pub struct Mp4Attachment {
    /// Attachment type (four-character code).
    pub attachment_type: [u8; 4],
    /// File data.
    pub data: Bytes,
    /// Filename (if available).
    pub filename: Option<String>,
}

impl Mp4Attachment {
    /// Creates a new MP4 attachment.
    #[must_use]
    pub const fn new(attachment_type: [u8; 4], data: Bytes) -> Self {
        Self {
            attachment_type,
            data,
            filename: None,
        }
    }

    /// Sets the filename.
    #[must_use]
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the type as a string.
    #[must_use]
    pub fn type_string(&self) -> String {
        String::from_utf8_lossy(&self.attachment_type).into_owned()
    }
}

/// Common MP4 attachment types.
pub struct Mp4AttachmentTypes;

impl Mp4AttachmentTypes {
    /// Cover art.
    pub const COVER: [u8; 4] = *b"covr";
    /// Copyright.
    pub const COPYRIGHT: [u8; 4] = *b"cprt";
    /// Description.
    pub const DESCRIPTION: [u8; 4] = *b"desc";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp4_attachment() {
        let data = Bytes::from_static(b"cover image");
        let attachment =
            Mp4Attachment::new(Mp4AttachmentTypes::COVER, data).with_filename("cover.jpg");

        assert_eq!(attachment.attachment_type, Mp4AttachmentTypes::COVER);
        assert_eq!(attachment.size(), 11);
        assert_eq!(attachment.filename, Some("cover.jpg".into()));
        assert_eq!(attachment.type_string(), "covr");
    }
}
