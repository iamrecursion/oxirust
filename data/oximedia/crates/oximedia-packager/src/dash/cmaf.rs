//! CMAF (Common Media Application Format) support.

use crate::error::PackagerResult;
use bytes::{BufMut, BytesMut};
use std::time::Duration;
use tracing::debug;

/// CMAF track type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle/text track.
    Text,
}

/// CMAF track configuration.
#[derive(Debug, Clone)]
pub struct CmafTrack {
    /// Track ID.
    pub track_id: u32,
    /// Track type.
    pub track_type: TrackType,
    /// Timescale.
    pub timescale: u32,
    /// Codec string.
    pub codec: String,
    /// Duration.
    pub duration: Duration,
}

impl CmafTrack {
    /// Create a new CMAF track.
    #[must_use]
    pub fn new(track_id: u32, track_type: TrackType, timescale: u32, codec: String) -> Self {
        Self {
            track_id,
            track_type,
            timescale,
            codec,
            duration: Duration::ZERO,
        }
    }

    /// Set duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// CMAF header generator.
pub struct CmafHeader {
    track: CmafTrack,
}

impl CmafHeader {
    /// Create a new CMAF header generator.
    #[must_use]
    pub fn new(track: CmafTrack) -> Self {
        Self { track }
    }

    /// Generate CMAF initialization segment.
    pub fn generate_init_segment(&self) -> PackagerResult<Vec<u8>> {
        let mut output = BytesMut::new();

        // Write ftyp box
        self.write_ftyp(&mut output)?;

        // Write moov box
        self.write_moov(&mut output)?;

        Ok(output.to_vec())
    }

    /// Write ftyp (File Type) box.
    fn write_ftyp(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let ftyp_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"ftyp");

        // Major brand - cmfc (CMAF chunk)
        output.put_slice(b"cmfc");

        // Minor version
        output.put_u32(0);

        // Compatible brands
        output.put_slice(b"iso6"); // ISO Base Media File Format
        output.put_slice(b"cmfc"); // CMAF chunk
        output.put_slice(b"dash"); // DASH

        // Update size
        let size = output.len() - ftyp_start;
        let size_bytes = (size as u32).to_be_bytes();
        output[ftyp_start..ftyp_start + 4].copy_from_slice(&size_bytes);

        debug!("Wrote ftyp box: {} bytes", size);

        Ok(())
    }

    /// Write moov (Movie) box.
    fn write_moov(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let moov_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"moov");

        // mvhd (Movie Header)
        self.write_mvhd(output)?;

        // trak (Track)
        self.write_trak(output)?;

        // mvex (Movie Extends) for fragmented MP4
        self.write_mvex(output)?;

        // Update size
        let size = output.len() - moov_start;
        let size_bytes = (size as u32).to_be_bytes();
        output[moov_start..moov_start + 4].copy_from_slice(&size_bytes);

        debug!("Wrote moov box: {} bytes", size);

        Ok(())
    }

    /// Write mvhd (Movie Header) box.
    fn write_mvhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(108); // Box size (version 0)
        output.put_slice(b"mvhd");

        // Version and flags
        output.put_u32(0);

        // Creation and modification time
        output.put_u32(0);
        output.put_u32(0);

        // Timescale
        output.put_u32(self.track.timescale);

        // Duration
        #[allow(clippy::cast_possible_truncation)]
        let duration = (self.track.duration.as_secs_f64() * f64::from(self.track.timescale)) as u32;
        output.put_u32(duration);

        // Rate (1.0)
        output.put_u32(0x00010000);

        // Volume (1.0)
        output.put_u16(0x0100);

        // Reserved
        output.put_u16(0);

        // Reserved
        output.put_u64(0);

        // Matrix (identity)
        output.put_u32(0x00010000);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0x00010000);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0x40000000);

        // Pre-defined
        output.put_bytes(0, 24);

        // Next track ID
        output.put_u32(2);

        Ok(())
    }

    /// Write trak (Track) box.
    fn write_trak(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let trak_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"trak");

        // tkhd (Track Header)
        self.write_tkhd(output)?;

        // mdia (Media)
        self.write_mdia(output)?;

        // Update size
        let size = output.len() - trak_start;
        let size_bytes = (size as u32).to_be_bytes();
        output[trak_start..trak_start + 4].copy_from_slice(&size_bytes);

        Ok(())
    }

    /// Write tkhd (Track Header) box.
    fn write_tkhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(92); // Box size
        output.put_slice(b"tkhd");

        // Version and flags (track enabled)
        output.put_u32(0x000007);

        // Creation and modification time
        output.put_u32(0);
        output.put_u32(0);

        // Track ID
        output.put_u32(self.track.track_id);

        // Reserved
        output.put_u32(0);

        // Duration
        #[allow(clippy::cast_possible_truncation)]
        let duration = (self.track.duration.as_secs_f64() * f64::from(self.track.timescale)) as u32;
        output.put_u32(duration);

        // Reserved
        output.put_u64(0);

        // Layer and alternate group
        output.put_u16(0);
        output.put_u16(0);

        // Volume (1.0 for audio, 0.0 for video)
        let volume = match self.track.track_type {
            TrackType::Audio => 0x0100u16,
            _ => 0u16,
        };
        output.put_u16(volume);

        // Reserved
        output.put_u16(0);

        // Matrix (identity)
        output.put_u32(0x00010000);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0x00010000);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0x40000000);

        // Width and height (0 for non-visual tracks)
        output.put_u32(0);
        output.put_u32(0);

        Ok(())
    }

    /// Write mdia (Media) box.
    fn write_mdia(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let mdia_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"mdia");

        // mdhd (Media Header)
        self.write_mdhd(output)?;

        // hdlr (Handler Reference)
        self.write_hdlr(output)?;

        // minf (Media Information)
        self.write_minf(output)?;

        // Update size
        let size = output.len() - mdia_start;
        let size_bytes = (size as u32).to_be_bytes();
        output[mdia_start..mdia_start + 4].copy_from_slice(&size_bytes);

        Ok(())
    }

    /// Write mdhd (Media Header) box.
    fn write_mdhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(32); // Box size
        output.put_slice(b"mdhd");

        // Version and flags
        output.put_u32(0);

        // Creation and modification time
        output.put_u32(0);
        output.put_u32(0);

        // Timescale
        output.put_u32(self.track.timescale);

        // Duration
        #[allow(clippy::cast_possible_truncation)]
        let duration = (self.track.duration.as_secs_f64() * f64::from(self.track.timescale)) as u32;
        output.put_u32(duration);

        // Language (und = undetermined)
        output.put_u16(0x55C4);

        // Pre-defined
        output.put_u16(0);

        Ok(())
    }

    /// Write hdlr (Handler Reference) box.
    fn write_hdlr(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let handler_type = match self.track.track_type {
            TrackType::Video => b"vide",
            TrackType::Audio => b"soun",
            TrackType::Text => b"text",
        };

        let name = match self.track.track_type {
            TrackType::Video => "VideoHandler\0",
            TrackType::Audio => "SoundHandler\0",
            TrackType::Text => "TextHandler\0",
        };

        let box_size = 32 + name.len();
        output.put_u32(box_size as u32);
        output.put_slice(b"hdlr");

        // Version and flags
        output.put_u32(0);

        // Pre-defined
        output.put_u32(0);

        // Handler type
        output.put_slice(handler_type);

        // Reserved
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0);

        // Name
        output.put_slice(name.as_bytes());

        Ok(())
    }

    /// Write minf (Media Information) box.
    fn write_minf(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let minf_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"minf");

        // Media header (vmhd, smhd, or nmhd)
        match self.track.track_type {
            TrackType::Video => self.write_vmhd(output)?,
            TrackType::Audio => self.write_smhd(output)?,
            TrackType::Text => self.write_nmhd(output)?,
        }

        // dinf (Data Information)
        self.write_dinf(output)?;

        // stbl (Sample Table)
        self.write_stbl(output)?;

        // Update size
        let size = output.len() - minf_start;
        let size_bytes = (size as u32).to_be_bytes();
        output[minf_start..minf_start + 4].copy_from_slice(&size_bytes);

        Ok(())
    }

    /// Write vmhd (Video Media Header) box.
    fn write_vmhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(20); // Box size
        output.put_slice(b"vmhd");

        // Version and flags
        output.put_u32(1);

        // Graphics mode
        output.put_u16(0);

        // Op color
        output.put_u16(0);
        output.put_u16(0);
        output.put_u16(0);

        Ok(())
    }

    /// Write smhd (Sound Media Header) box.
    fn write_smhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(16); // Box size
        output.put_slice(b"smhd");

        // Version and flags
        output.put_u32(0);

        // Balance
        output.put_u16(0);

        // Reserved
        output.put_u16(0);

        Ok(())
    }

    /// Write nmhd (Null Media Header) box.
    fn write_nmhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(12); // Box size
        output.put_slice(b"nmhd");

        // Version and flags
        output.put_u32(0);

        Ok(())
    }

    /// Write dinf (Data Information) box.
    fn write_dinf(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(36); // Box size
        output.put_slice(b"dinf");

        // dref (Data Reference)
        output.put_u32(28); // Box size
        output.put_slice(b"dref");

        // Version and flags
        output.put_u32(0);

        // Entry count
        output.put_u32(1);

        // url entry
        output.put_u32(12);
        output.put_slice(b"url ");
        output.put_u32(1); // Self-contained

        Ok(())
    }

    /// Write stbl (Sample Table) box.
    fn write_stbl(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let stbl_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"stbl");

        // Minimal sample table for fragmented MP4
        // stsd (Sample Description)
        output.put_u32(16);
        output.put_slice(b"stsd");
        output.put_u32(0); // Version and flags
        output.put_u32(0); // Entry count (none for fragmented)

        // stts (Decoding Time to Sample)
        output.put_u32(16);
        output.put_slice(b"stts");
        output.put_u32(0);
        output.put_u32(0);

        // stsc (Sample to Chunk)
        output.put_u32(16);
        output.put_slice(b"stsc");
        output.put_u32(0);
        output.put_u32(0);

        // stsz (Sample Size)
        output.put_u32(20);
        output.put_slice(b"stsz");
        output.put_u32(0);
        output.put_u32(0);
        output.put_u32(0);

        // stco (Chunk Offset)
        output.put_u32(16);
        output.put_slice(b"stco");
        output.put_u32(0);
        output.put_u32(0);

        // Update size
        let size = output.len() - stbl_start;
        let size_bytes = (size as u32).to_be_bytes();
        output[stbl_start..stbl_start + 4].copy_from_slice(&size_bytes);

        Ok(())
    }

    /// Write mvex (Movie Extends) box.
    fn write_mvex(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(40); // Box size
        output.put_slice(b"mvex");

        // trex (Track Extends)
        output.put_u32(32);
        output.put_slice(b"trex");

        // Version and flags
        output.put_u32(0);

        // Track ID
        output.put_u32(self.track.track_id);

        // Default sample description index
        output.put_u32(1);

        // Default sample duration
        output.put_u32(0);

        // Default sample size
        output.put_u32(0);

        // Default sample flags
        output.put_u32(0);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmaf_track_creation() {
        let track = CmafTrack::new(1, TrackType::Video, 90000, "av01.0.04M.08".to_string());

        assert_eq!(track.track_id, 1);
        assert_eq!(track.track_type, TrackType::Video);
        assert_eq!(track.timescale, 90000);
    }

    #[test]
    fn test_cmaf_header_generation() {
        let track = CmafTrack::new(1, TrackType::Video, 90000, "av01.0.04M.08".to_string())
            .with_duration(Duration::from_secs(60));

        let header = CmafHeader::new(track);
        let init_segment = header
            .generate_init_segment()
            .expect("should succeed in test");

        assert!(!init_segment.is_empty());
        // Verify it starts with a valid box (size followed by type)
        assert!(init_segment.len() > 8); // At least one box
    }
}
