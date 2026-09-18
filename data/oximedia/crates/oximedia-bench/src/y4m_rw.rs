//! YUV4MPEG2 stream reader and writer.
//!
//! Provides [`Y4mReader`], [`Y4mWriter`], [`BenchFrame`], and
//! [`Y4mColorFormat`] for reading and writing YUV4MPEG2 (`.y4m`) streams.
//!
//! The stream format follows the Xiph.org specification:
//!   - Header: `YUV4MPEG2 W<w> H<h> F<num>:<den> I<interlace> A<par> C<chroma>\n`
//!   - Frame:  `FRAME\n` followed by raw planar pixel data (Y then U then V).

use std::io::{BufRead, Read};

use crate::BenchError;

/// YUV4MPEG2 chroma subsampling format, as declared in the `C` header field.
#[derive(Debug, Clone, PartialEq)]
pub enum Y4mColorFormat {
    /// 4:2:0 JPEG (full-range)
    C420Jpeg,
    /// 4:2:0 MPEG-2 (limited range)
    C420Mpeg2,
    /// 4:2:0 PAL-DV
    C420Paldv,
    /// 4:2:0 (generic / unspecified variant)
    C420,
    /// 4:2:2
    C422,
    /// 4:4:4
    C444,
    /// Monochrome / greyscale (no chroma planes)
    CMono,
}

impl Y4mColorFormat {
    /// Return the Y4M header string for this format (the part after `C`).
    #[must_use]
    pub fn as_header_str(&self) -> &'static str {
        match self {
            Self::C420Jpeg => "420jpeg",
            Self::C420Mpeg2 => "420mpeg2",
            Self::C420Paldv => "420paldv",
            Self::C420 => "420",
            Self::C422 => "422",
            Self::C444 => "444",
            Self::CMono => "mono",
        }
    }

    /// Parse the chroma descriptor from a Y4M `C` field value.
    pub(crate) fn parse(s: &str) -> Self {
        // Match longest prefix first to avoid classifying "420jpeg" as "420".
        if s.eq_ignore_ascii_case("420jpeg") {
            Self::C420Jpeg
        } else if s.eq_ignore_ascii_case("420mpeg2") {
            Self::C420Mpeg2
        } else if s.eq_ignore_ascii_case("420paldv") {
            Self::C420Paldv
        } else if s.eq_ignore_ascii_case("422") {
            Self::C422
        } else if s.eq_ignore_ascii_case("444") {
            Self::C444
        } else if s.eq_ignore_ascii_case("mono") || s.eq_ignore_ascii_case("400") {
            Self::CMono
        } else {
            // "420" and any unrecognised 4:2:0 variant
            Self::C420
        }
    }

    /// Return `(u_bytes, v_bytes)` plane sizes for a frame of this format.
    #[must_use]
    pub(crate) fn chroma_plane_size(&self, width: usize, height: usize) -> (usize, usize) {
        match self {
            Self::C420Jpeg | Self::C420Mpeg2 | Self::C420Paldv | Self::C420 => {
                let sz = (width / 2) * (height / 2);
                (sz, sz)
            }
            Self::C422 => {
                let sz = (width / 2) * height;
                (sz, sz)
            }
            Self::C444 => {
                let sz = width * height;
                (sz, sz)
            }
            Self::CMono => (0, 0),
        }
    }
}

/// A single decoded Y4M frame with separate planar data.
#[derive(Debug, Clone)]
pub struct BenchFrame {
    /// Y (luma) plane: `width × height` bytes.
    pub y_plane: Vec<u8>,
    /// U (Cb) chroma plane.
    pub u_plane: Vec<u8>,
    /// V (Cr) chroma plane.
    pub v_plane: Vec<u8>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Chroma subsampling of this frame.
    pub color_format: Y4mColorFormat,
}

impl BenchFrame {
    /// Create a solid-colour synthetic frame (4:2:0 JPEG subsampling).
    ///
    /// Useful for unit tests that need a valid frame without reading a file.
    #[must_use]
    pub fn synthetic(width: usize, height: usize, y: u8, u: u8, v: u8) -> Self {
        let (u_sz, v_sz) = Y4mColorFormat::C420Jpeg.chroma_plane_size(width, height);
        Self {
            y_plane: vec![y; width * height],
            u_plane: vec![u; u_sz],
            v_plane: vec![v; v_sz],
            width,
            height,
            color_format: Y4mColorFormat::C420Jpeg,
        }
    }
}

/// Streaming iterator over frames in a YUV4MPEG2 stream.
///
/// Parses the stream header on construction then yields frames one at a time
/// via [`Y4mReader::next_frame`].  The underlying reader is wrapped in a
/// `BufReader` for performance.
pub struct Y4mReader<R: std::io::Read> {
    reader: std::io::BufReader<R>,
    /// Frame width declared in the stream header.
    pub width: usize,
    /// Frame height declared in the stream header.
    pub height: usize,
    /// Numerator of the frame-rate rational (e.g. 30 for 30/1).
    pub fps_num: u32,
    /// Denominator of the frame-rate rational.
    pub fps_den: u32,
    /// Chroma subsampling format declared in the stream header.
    pub color_format: Y4mColorFormat,
}

impl<R: std::io::Read> Y4mReader<R> {
    /// Parse the YUV4MPEG2 stream header and return a reader ready to yield
    /// frames.
    ///
    /// # Errors
    ///
    /// Returns an error if the header is missing, malformed, or does not
    /// contain valid width and height fields.
    pub fn new(reader: R) -> Result<Self, BenchError> {
        let mut buf_reader = std::io::BufReader::new(reader);

        // Read the header line.
        let mut header = String::new();
        buf_reader.read_line(&mut header)?;
        let header = header.trim_end_matches('\n').trim_end_matches('\r');

        if !header.starts_with("YUV4MPEG2") {
            return Err(BenchError::ExecutionFailed(
                "Y4mReader: missing YUV4MPEG2 magic in stream header".to_string(),
            ));
        }

        let mut width: Option<usize> = None;
        let mut height: Option<usize> = None;
        let mut fps_num: u32 = 25;
        let mut fps_den: u32 = 1;
        let mut color_format = Y4mColorFormat::C420;

        for token in header["YUV4MPEG2".len()..].split_whitespace() {
            if token.is_empty() {
                continue;
            }
            let tag = &token[..1];
            let val = &token[1..];
            match tag {
                "W" => {
                    width = Some(val.parse::<usize>().map_err(|e| {
                        BenchError::ExecutionFailed(format!(
                            "Y4mReader: invalid width '{val}': {e}"
                        ))
                    })?);
                }
                "H" => {
                    height = Some(val.parse::<usize>().map_err(|e| {
                        BenchError::ExecutionFailed(format!(
                            "Y4mReader: invalid height '{val}': {e}"
                        ))
                    })?);
                }
                "F" => {
                    // Format: "{num}:{den}"
                    let mut parts = val.splitn(2, ':');
                    let n = parts.next().unwrap_or("25");
                    let d = parts.next().unwrap_or("1");
                    fps_num = n.parse::<u32>().unwrap_or(25);
                    fps_den = d.parse::<u32>().unwrap_or(1);
                }
                "C" => {
                    color_format = Y4mColorFormat::parse(val);
                }
                _ => {}
            }
        }

        let width = width.ok_or_else(|| {
            BenchError::ExecutionFailed("Y4mReader: missing width (W) in header".to_string())
        })?;
        let height = height.ok_or_else(|| {
            BenchError::ExecutionFailed("Y4mReader: missing height (H) in header".to_string())
        })?;

        if width == 0 || height == 0 {
            return Err(BenchError::ExecutionFailed(
                "Y4mReader: width and height must be non-zero".to_string(),
            ));
        }

        Ok(Self {
            reader: buf_reader,
            width,
            height,
            fps_num,
            fps_den,
            color_format,
        })
    }

    /// Read the next frame from the stream.
    ///
    /// Returns `Ok(None)` at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the `FRAME` separator is missing, data is
    /// truncated, or an I/O error occurs.
    pub fn next_frame(&mut self) -> Result<Option<BenchFrame>, BenchError> {
        // Read the FRAME marker line.
        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        if line.is_empty() {
            return Ok(None);
        }
        if !line.starts_with("FRAME") {
            return Err(BenchError::ExecutionFailed(format!(
                "Y4mReader: expected 'FRAME' separator, got '{line}'"
            )));
        }

        let y_size = self.width * self.height;
        let (u_size, v_size) = self.color_format.chroma_plane_size(self.width, self.height);
        let total = y_size + u_size + v_size;

        let mut buf = vec![0u8; total];
        let mut filled = 0;
        while filled < total {
            match self.reader.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) => return Err(BenchError::Io(e)),
            }
        }
        if filled < total {
            if filled == 0 {
                return Ok(None);
            }
            return Err(BenchError::ExecutionFailed(format!(
                "Y4mReader: frame payload truncated: expected {total} bytes, got {filled}"
            )));
        }

        let y_plane = buf[..y_size].to_vec();
        let u_plane = buf[y_size..y_size + u_size].to_vec();
        let v_plane = buf[y_size + u_size..].to_vec();

        Ok(Some(BenchFrame {
            y_plane,
            u_plane,
            v_plane,
            width: self.width,
            height: self.height,
            color_format: self.color_format.clone(),
        }))
    }
}

/// Writer for YUV4MPEG2 streams.
///
/// Emits a stream header on the first call to [`Y4mWriter::write_frame`] and
/// writes each frame as `FRAME\n` followed by raw planar bytes.
pub struct Y4mWriter<W: std::io::Write> {
    writer: W,
    width: usize,
    height: usize,
    fps_num: u32,
    fps_den: u32,
    color_format: Y4mColorFormat,
    header_written: bool,
}

impl<W: std::io::Write> Y4mWriter<W> {
    /// Create a new Y4M writer.
    ///
    /// The stream header is written lazily on the first call to
    /// [`Y4mWriter::write_frame`].
    #[must_use]
    pub fn new(
        writer: W,
        width: usize,
        height: usize,
        fps_num: u32,
        fps_den: u32,
        color_format: Y4mColorFormat,
    ) -> Self {
        Self {
            writer,
            width,
            height,
            fps_num,
            fps_den,
            color_format,
            header_written: false,
        }
    }

    /// Write a single frame to the stream.
    ///
    /// The YUV4MPEG2 stream header is written automatically before the first
    /// frame.
    ///
    /// # Errors
    ///
    /// Returns an error if an I/O error occurs.
    pub fn write_frame(&mut self, frame: &BenchFrame) -> Result<(), BenchError> {
        if !self.header_written {
            writeln!(
                self.writer,
                "YUV4MPEG2 W{} H{} F{}:{} Ip A0:0 C{}",
                self.width,
                self.height,
                self.fps_num,
                self.fps_den,
                self.color_format.as_header_str(),
            )?;
            self.header_written = true;
        }

        // Write FRAME separator.
        self.writer.write_all(b"FRAME\n")?;

        // Write Y plane.
        self.writer.write_all(&frame.y_plane)?;

        // Write chroma planes (CMono has zero-length planes so this is a no-op).
        self.writer.write_all(&frame.u_plane)?;
        self.writer.write_all(&frame.v_plane)?;

        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y4m_roundtrip_5frames() {
        let width = 64usize;
        let height = 48usize;
        let frames: Vec<BenchFrame> = (0..5)
            .map(|i| BenchFrame::synthetic(width, height, (i * 40) as u8, 128, 128))
            .collect();

        // Write to temp buffer.
        let mut buf = Vec::new();
        let mut writer = Y4mWriter::new(&mut buf, width, height, 30, 1, Y4mColorFormat::C420Jpeg);
        for f in &frames {
            writer.write_frame(f).expect("write_frame should succeed");
        }

        // Read back.
        let mut reader = Y4mReader::new(buf.as_slice()).expect("Y4mReader::new should succeed");
        assert_eq!(reader.width, width);
        assert_eq!(reader.height, height);
        assert_eq!(reader.fps_num, 30);
        assert_eq!(reader.fps_den, 1);

        let mut read_frames = Vec::new();
        while let Some(f) = reader.next_frame().expect("next_frame should succeed") {
            read_frames.push(f);
        }
        assert_eq!(read_frames.len(), 5);

        // Byte-identical planes.
        for (orig, read) in frames.iter().zip(read_frames.iter()) {
            assert_eq!(orig.y_plane, read.y_plane, "Y plane mismatch");
            assert_eq!(orig.u_plane, read.u_plane, "U plane mismatch");
            assert_eq!(orig.v_plane, read.v_plane, "V plane mismatch");
        }
    }

    #[test]
    fn y4m_reader_parses_color_format_fields() {
        let width = 32usize;
        let height = 32usize;

        for fmt in &[
            Y4mColorFormat::C420Jpeg,
            Y4mColorFormat::C420Mpeg2,
            Y4mColorFormat::C420Paldv,
            Y4mColorFormat::C420,
            Y4mColorFormat::C422,
            Y4mColorFormat::C444,
        ] {
            let mut buf = Vec::new();
            let (u_sz, v_sz) = fmt.chroma_plane_size(width, height);
            let f = BenchFrame {
                y_plane: vec![100u8; width * height],
                u_plane: vec![128u8; u_sz],
                v_plane: vec![128u8; v_sz],
                width,
                height,
                color_format: fmt.clone(),
            };
            let mut writer = Y4mWriter::new(&mut buf, width, height, 25, 1, fmt.clone());
            writer.write_frame(&f).expect("write_frame");

            let reader = Y4mReader::new(buf.as_slice()).expect("Y4mReader");
            assert_eq!(reader.color_format, *fmt);
        }
    }

    #[test]
    fn y4m_mono_roundtrip() {
        let width = 16usize;
        let height = 16usize;
        let frame = BenchFrame {
            y_plane: vec![200u8; width * height],
            u_plane: Vec::new(),
            v_plane: Vec::new(),
            width,
            height,
            color_format: Y4mColorFormat::CMono,
        };

        let mut buf = Vec::new();
        let mut writer = Y4mWriter::new(&mut buf, width, height, 30, 1, Y4mColorFormat::CMono);
        writer.write_frame(&frame).expect("write mono frame");

        let mut reader = Y4mReader::new(buf.as_slice()).expect("read mono header");
        assert_eq!(reader.color_format, Y4mColorFormat::CMono);
        let f = reader.next_frame().expect("next_frame").expect("frame");
        assert_eq!(f.y_plane, frame.y_plane);
        assert!(f.u_plane.is_empty());
        assert!(f.v_plane.is_empty());
    }
}
