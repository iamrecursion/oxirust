//! Caption embedding into container formats

use crate::error::{CaptionError, Result};
use crate::types::CaptionTrack;
use crate::CaptionFormat;

/// Caption embedder for container formats
pub struct Embedder;

impl Embedder {
    /// Embed captions into MP4 container
    pub fn embed_mp4(
        _track: &CaptionTrack,
        _video_data: &[u8],
        _format: CaptionFormat,
    ) -> Result<Vec<u8>> {
        // Full implementation would integrate with oximedia-container
        Err(CaptionError::Embedding(
            "MP4 embedding requires oximedia-container integration".to_string(),
        ))
    }

    /// Embed captions into Matroska/WebM container
    pub fn embed_matroska(
        _track: &CaptionTrack,
        _video_data: &[u8],
        _format: CaptionFormat,
    ) -> Result<Vec<u8>> {
        Err(CaptionError::Embedding(
            "Matroska embedding requires oximedia-container integration".to_string(),
        ))
    }

    /// Embed captions into MPEG-TS
    pub fn embed_mpeg_ts(
        _track: &CaptionTrack,
        _video_data: &[u8],
        _format: CaptionFormat,
    ) -> Result<Vec<u8>> {
        Err(CaptionError::Embedding(
            "MPEG-TS embedding requires oximedia-container integration".to_string(),
        ))
    }

    /// Generate sidecar file (separate caption file)
    pub fn generate_sidecar(track: &CaptionTrack, format: CaptionFormat) -> Result<Vec<u8>> {
        crate::export::Exporter::export(track, format)
    }
}

/// Container format for embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// MP4/M4V
    Mp4,
    /// Matroska (MKV)
    Matroska,
    /// `WebM`
    WebM,
    /// MPEG-TS
    MpegTs,
    /// MOV (`QuickTime`)
    Mov,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_formats() {
        assert_eq!(ContainerFormat::Mp4, ContainerFormat::Mp4);
        assert_ne!(ContainerFormat::Mp4, ContainerFormat::Matroska);
    }
}
