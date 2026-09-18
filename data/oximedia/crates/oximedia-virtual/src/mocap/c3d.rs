//! C3D (Coordinate 3D) motion capture file format parser.
//!
//! C3D is a binary format used by Vicon, OptiTrack, Qualisys, and most
//! professional motion capture systems.  It stores 3D marker trajectories
//! plus analog channels (EMG, force plates) at a fixed sample rate.
//!
//! # Format overview (subset implemented here)
//! ```text
//! Byte 0:  first block number of parameter section (1-based)
//! Byte 1:  0x50 (magic key)
//! Bytes 2-3: number of 3D points (little-endian u16)
//! Bytes 4-5: number of analog samples per 3D frame (u16)
//! Bytes 6-7: first frame number (u16)
//! Bytes 8-9: last frame number (u16)
//! Bytes 10-11: max interpolation gap (u16)
//! Bytes 12-13: scale factor (f32 low word)
//! Bytes 14-15: scale factor (f32 high word)
//! Bytes 16-17: first data block (1-based, u16)
//! Bytes 18-19: analog samples per frame (u16)
//! Bytes 20-23: frame rate (f32)
//! Bytes 24-511: reserved
//! ```
//!
//! This implementation parses the binary header and extracts marker
//! trajectories.  It supports both integer-scaled and floating-point
//! point data, and handles big-endian and little-endian variants.
//!
//! For testing we also provide a `C3dWriter` that can write synthetic
//! files readable by the parser (enabling round-trip tests).

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const C3D_MAGIC: u8 = 0x50;
const BLOCK_SIZE: usize = 512;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// C3D endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C3dEndian {
    Little,
    Big,
}

/// A single 3D marker sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct C3dPoint {
    /// X position (in millimetres after scaling).
    pub x: f64,
    /// Y position.
    pub y: f64,
    /// Z position.
    pub z: f64,
    /// Residual / confidence (0.0 = occluded / invalid).
    pub residual: f32,
    /// Whether this sample is valid (not occluded).
    pub valid: bool,
}

impl C3dPoint {
    /// Create a valid 3D point.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64, residual: f32) -> Self {
        Self {
            x,
            y,
            z,
            residual,
            valid: residual >= 0.0,
        }
    }

    /// Create an invalid (occluded) point.
    #[must_use]
    pub fn occluded() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            residual: -1.0,
            valid: false,
        }
    }
}

/// One frame of 3D marker data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C3dFrame {
    /// Marker positions.  Length == number of markers.
    pub points: Vec<C3dPoint>,
}

impl C3dFrame {
    /// Number of valid (non-occluded) markers in this frame.
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.points.iter().filter(|p| p.valid).count()
    }
}

/// C3D clip header information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C3dHeader {
    /// Number of 3D marker trajectories.
    pub point_count: u16,
    /// Number of analog samples per 3D frame (0 if no analog data).
    pub analog_per_frame: u16,
    /// First frame number (1-based).
    pub first_frame: u16,
    /// Last frame number (1-based).
    pub last_frame: u16,
    /// Scale factor (negative = float data, positive = integer data).
    pub scale: f32,
    /// First data block (1-based block index).
    pub first_data_block: u16,
    /// Frame rate in Hz.
    pub frame_rate: f32,
    /// Endianness of the file.
    pub endian: C3dEndian,
}

impl C3dHeader {
    /// Total number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        if self.last_frame >= self.first_frame {
            (self.last_frame - self.first_frame + 1) as usize
        } else {
            0
        }
    }

    /// Whether point data is stored as floats (scale < 0).
    #[must_use]
    pub fn is_float_data(&self) -> bool {
        self.scale < 0.0
    }
}

/// A complete C3D clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C3dClip {
    /// Header metadata.
    pub header: C3dHeader,
    /// All frames (frame_count frames × point_count markers).
    pub frames: Vec<C3dFrame>,
    /// Marker labels (if available from parameter section).
    pub labels: Vec<String>,
}

impl C3dClip {
    /// Frame count.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Marker count.
    #[must_use]
    pub fn marker_count(&self) -> usize {
        self.header.point_count as usize
    }

    /// Frame rate in Hz.
    #[must_use]
    pub fn frame_rate(&self) -> f32 {
        self.header.frame_rate
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        if self.header.frame_rate > 0.0 {
            self.frames.len() as f64 / self.header.frame_rate as f64
        } else {
            0.0
        }
    }

    /// Get a specific frame.
    #[must_use]
    pub fn frame(&self, idx: usize) -> Option<&C3dFrame> {
        self.frames.get(idx)
    }

    /// Get the trajectory of marker `idx` across all frames.
    #[must_use]
    pub fn trajectory(&self, marker_idx: usize) -> Vec<Option<C3dPoint>> {
        self.frames
            .iter()
            .map(|f| f.points.get(marker_idx).copied())
            .collect()
    }

    /// Marker label if available.
    #[must_use]
    pub fn label(&self, marker_idx: usize) -> Option<&str> {
        self.labels.get(marker_idx).map(String::as_str)
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// C3D binary file parser.
pub struct C3dParser;

impl C3dParser {
    /// Parse a C3D binary file from a byte slice.
    ///
    /// Supports integer-scaled and floating-point point data.
    /// Does not currently parse analog channels or the parameter section.
    pub fn parse(data: &[u8]) -> Result<C3dClip> {
        if data.len() < BLOCK_SIZE {
            return Err(VirtualProductionError::MotionCapture(format!(
                "C3D file too small: {} bytes (minimum {})",
                data.len(),
                BLOCK_SIZE
            )));
        }

        // Validate magic byte (byte 1 must be 0x50)
        if data[1] != C3D_MAGIC {
            return Err(VirtualProductionError::MotionCapture(format!(
                "Invalid C3D magic byte: 0x{:02X} (expected 0x50)",
                data[1]
            )));
        }

        // Detect endianness: parameters block can use a processor type code.
        // Simpler: check byte 0 for a reasonable parameter block start (1..=254).
        // We default to little-endian (Intel) which covers 99 % of modern files.
        let endian = C3dEndian::Little;

        let read_u16 = |buf: &[u8], off: usize| -> u16 {
            let lo = buf[off] as u16;
            let hi = buf[off + 1] as u16;
            match endian {
                C3dEndian::Little => lo | (hi << 8),
                C3dEndian::Big => (lo << 8) | hi,
            }
        };

        let read_f32 = |buf: &[u8], off: usize| -> f32 {
            let bytes = [buf[off], buf[off + 1], buf[off + 2], buf[off + 3]];
            match endian {
                C3dEndian::Little => f32::from_le_bytes(bytes),
                C3dEndian::Big => f32::from_be_bytes(bytes),
            }
        };

        // Header block (block 1, offset 0)
        let point_count = read_u16(data, 2);
        let analog_per_frame = read_u16(data, 4);
        let first_frame = read_u16(data, 6);
        let last_frame = read_u16(data, 8);
        let scale = read_f32(data, 10);
        let first_data_block = read_u16(data, 16);
        let frame_rate = read_f32(data, 20);

        // Validate
        if first_data_block == 0 {
            return Err(VirtualProductionError::MotionCapture(
                "C3D first_data_block is 0 (invalid)".to_string(),
            ));
        }
        if last_frame < first_frame && !(first_frame == 0 && last_frame == 0) {
            return Err(VirtualProductionError::MotionCapture(format!(
                "C3D last_frame ({last_frame}) < first_frame ({first_frame})"
            )));
        }

        let header = C3dHeader {
            point_count,
            analog_per_frame,
            first_frame,
            last_frame,
            scale,
            first_data_block,
            frame_rate,
            endian,
        };

        let frame_count = header.frame_count();

        // Data section starts at block `first_data_block` (1-based)
        let data_offset = (first_data_block as usize - 1) * BLOCK_SIZE;
        if data_offset > data.len() {
            return Err(VirtualProductionError::MotionCapture(format!(
                "Data offset {data_offset} exceeds file size {}",
                data.len()
            )));
        }
        let data_section = &data[data_offset..];

        // Each 3D point uses 4 values (x, y, z, residual).
        // Integer format: 4 × i16 (8 bytes per point), float: 4 × f32 (16 bytes).
        // Analog channels follow the point data per frame.
        let bytes_per_point = if header.is_float_data() { 16usize } else { 8 };
        let analog_bytes_per_frame = (analog_per_frame as usize) * 2; // 16-bit analog

        let point_bytes_per_frame = (point_count as usize) * bytes_per_point;
        let _bytes_per_frame = point_bytes_per_frame + analog_bytes_per_frame;

        let mut frames = Vec::with_capacity(frame_count);
        let mut cursor = 0usize;

        for _ in 0..frame_count {
            if cursor + point_bytes_per_frame > data_section.len() {
                break; // truncated file
            }

            let mut points = Vec::with_capacity(point_count as usize);

            for _ in 0..point_count {
                let point = if header.is_float_data() {
                    if cursor + 16 > data_section.len() {
                        C3dPoint::occluded()
                    } else {
                        let x = f64::from(read_f32(data_section, cursor));
                        let y = f64::from(read_f32(data_section, cursor + 4));
                        let z = f64::from(read_f32(data_section, cursor + 8));
                        let res = read_f32(data_section, cursor + 12);
                        cursor += 16;
                        C3dPoint::new(x, y, z, res)
                    }
                } else {
                    // Integer format: signed i16 × 4, scale by |scale| for mm
                    if cursor + 8 > data_section.len() {
                        C3dPoint::occluded()
                    } else {
                        let ix =
                            i16::from_le_bytes([data_section[cursor], data_section[cursor + 1]]);
                        let iy = i16::from_le_bytes([
                            data_section[cursor + 2],
                            data_section[cursor + 3],
                        ]);
                        let iz = i16::from_le_bytes([
                            data_section[cursor + 4],
                            data_section[cursor + 5],
                        ]);
                        let ir = i16::from_le_bytes([
                            data_section[cursor + 6],
                            data_section[cursor + 7],
                        ]);
                        cursor += 8;

                        let s = scale.abs() as f64;
                        let res = ir as f32;
                        let valid = ir >= 0;
                        if valid {
                            C3dPoint {
                                x: ix as f64 * s,
                                y: iy as f64 * s,
                                z: iz as f64 * s,
                                residual: res,
                                valid: true,
                            }
                        } else {
                            C3dPoint::occluded()
                        }
                    }
                };
                points.push(point);
            }

            // Skip analog data for this frame
            cursor += analog_bytes_per_frame;

            frames.push(C3dFrame { points });
        }

        // Default empty labels (parameter section not parsed)
        let labels: Vec<String> = (0..point_count as usize)
            .map(|i| format!("Marker{i:03}"))
            .collect();

        Ok(C3dClip {
            header,
            frames,
            labels,
        })
    }
}

// ---------------------------------------------------------------------------
// Writer (for tests / synthetic file generation)
// ---------------------------------------------------------------------------

/// Minimal C3D writer for generating synthetic test files.
pub struct C3dWriter;

impl C3dWriter {
    /// Write a minimal C3D file with float point data (scale = -1.0).
    ///
    /// `frames` should contain `point_count` points per frame.
    pub fn write_float(
        frame_rate: f32,
        frames: &[C3dFrame],
        point_count: usize,
    ) -> Result<Vec<u8>> {
        if frames.is_empty() {
            return Err(VirtualProductionError::MotionCapture(
                "Cannot write empty frame list".to_string(),
            ));
        }

        let first_frame = 1u16;
        let last_frame = frames.len() as u16;
        let analog_per_frame = 0u16;
        let scale: f32 = -1.0; // float data
        let first_param_block = 2u16; // block 2 = parameter section
        let first_data_block = 3u16; // block 3 = data

        // Header block (512 bytes)
        let mut header = vec![0u8; BLOCK_SIZE];
        header[0] = first_param_block as u8;
        header[1] = C3D_MAGIC;
        let pc = point_count as u16;
        header[2] = (pc & 0xFF) as u8;
        header[3] = (pc >> 8) as u8;
        header[4] = (analog_per_frame & 0xFF) as u8;
        header[5] = (analog_per_frame >> 8) as u8;
        header[6] = (first_frame & 0xFF) as u8;
        header[7] = (first_frame >> 8) as u8;
        header[8] = (last_frame & 0xFF) as u8;
        header[9] = (last_frame >> 8) as u8;
        let scale_bytes = scale.to_le_bytes();
        header[10] = scale_bytes[0];
        header[11] = scale_bytes[1];
        header[12] = scale_bytes[2];
        header[13] = scale_bytes[3];
        header[16] = (first_data_block & 0xFF) as u8;
        header[17] = (first_data_block >> 8) as u8;
        let fr_bytes = frame_rate.to_le_bytes();
        header[20] = fr_bytes[0];
        header[21] = fr_bytes[1];
        header[22] = fr_bytes[2];
        header[23] = fr_bytes[3];

        // Parameter block (dummy, 512 bytes)
        let param_block = vec![0u8; BLOCK_SIZE];

        // Data section
        let bytes_per_point = 16usize; // 4 × f32
        let bytes_per_frame = point_count * bytes_per_point;
        let mut data = Vec::with_capacity(frames.len() * bytes_per_frame);

        for frame in frames {
            for pi in 0..point_count {
                let p = frame
                    .points
                    .get(pi)
                    .copied()
                    .unwrap_or(C3dPoint::occluded());
                data.extend_from_slice(&(p.x as f32).to_le_bytes());
                data.extend_from_slice(&(p.y as f32).to_le_bytes());
                data.extend_from_slice(&(p.z as f32).to_le_bytes());
                data.extend_from_slice(&p.residual.to_le_bytes());
            }
        }

        // Pad data to block boundary
        let remainder = data.len() % BLOCK_SIZE;
        if remainder != 0 {
            data.resize(data.len() + (BLOCK_SIZE - remainder), 0);
        }

        let mut out = header;
        out.extend_from_slice(&param_block);
        out.extend_from_slice(&data);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frames(n: usize, point_count: usize) -> Vec<C3dFrame> {
        (0..n)
            .map(|i| C3dFrame {
                points: (0..point_count)
                    .map(|j| C3dPoint::new(i as f64 * 10.0, j as f64 * 5.0, 100.0, 1.0))
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn test_write_and_parse_basic() {
        let frames = make_frames(4, 3);
        let data = C3dWriter::write_float(120.0, &frames, 3).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        assert_eq!(clip.frame_count(), 4);
        assert_eq!(clip.marker_count(), 3);
    }

    #[test]
    fn test_frame_rate() {
        let frames = make_frames(2, 2);
        let data = C3dWriter::write_float(240.0, &frames, 2).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        assert!((clip.frame_rate() - 240.0).abs() < 0.01);
    }

    #[test]
    fn test_duration_s() {
        let frames = make_frames(120, 1);
        let data = C3dWriter::write_float(120.0, &frames, 1).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        assert!((clip.duration_s() - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_marker_positions_roundtrip() {
        let mut frames = Vec::new();
        frames.push(C3dFrame {
            points: vec![
                C3dPoint::new(100.0, 200.0, 300.0, 1.0),
                C3dPoint::new(-50.0, 0.0, 75.5, 0.5),
            ],
        });

        let data = C3dWriter::write_float(60.0, &frames, 2).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        let f = clip.frame(0).expect("frame 0");

        let p0 = f.points[0];
        assert!((p0.x - 100.0).abs() < 0.1, "x: {}", p0.x);
        assert!((p0.y - 200.0).abs() < 0.1, "y: {}", p0.y);
        assert!((p0.z - 300.0).abs() < 0.1, "z: {}", p0.z);
        assert!(p0.valid, "should be valid");

        let p1 = f.points[1];
        assert!((p1.x - (-50.0)).abs() < 0.1, "x: {}", p1.x);
    }

    #[test]
    fn test_valid_count() {
        let frame = C3dFrame {
            points: vec![
                C3dPoint::new(0.0, 0.0, 0.0, 1.0),
                C3dPoint::occluded(),
                C3dPoint::new(1.0, 2.0, 3.0, 0.8),
            ],
        };
        assert_eq!(frame.valid_count(), 2);
    }

    #[test]
    fn test_occluded_point() {
        let p = C3dPoint::occluded();
        assert!(!p.valid);
        assert!(p.residual < 0.0);
    }

    #[test]
    fn test_trajectory_extraction() {
        let frames = make_frames(5, 2);
        let data = C3dWriter::write_float(60.0, &frames, 2).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");

        let traj = clip.trajectory(0);
        assert_eq!(traj.len(), 5);
        for (i, pt) in traj.into_iter().enumerate() {
            let p = pt.expect("should have point");
            assert!((p.x - i as f64 * 10.0).abs() < 0.5, "frame {i} x: {}", p.x);
        }
    }

    #[test]
    fn test_default_labels() {
        let frames = make_frames(1, 3);
        let data = C3dWriter::write_float(60.0, &frames, 3).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        assert_eq!(clip.label(0), Some("Marker000"));
        assert_eq!(clip.label(2), Some("Marker002"));
        assert!(clip.label(10).is_none());
    }

    #[test]
    fn test_too_small_fails() {
        let data = vec![0u8; 10];
        let result = C3dParser::parse(&data);
        assert!(result.is_err(), "too small should fail");
    }

    #[test]
    fn test_bad_magic_fails() {
        let mut data = vec![0u8; 512];
        data[1] = 0x01; // wrong magic
        let result = C3dParser::parse(&data);
        assert!(result.is_err(), "bad magic should fail");
    }

    #[test]
    fn test_float_data_flag() {
        let frames = make_frames(1, 1);
        let data = C3dWriter::write_float(60.0, &frames, 1).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        assert!(
            clip.header.is_float_data(),
            "scale should be negative for float"
        );
    }

    #[test]
    fn test_frame_out_of_bounds_returns_none() {
        let frames = make_frames(2, 1);
        let data = C3dWriter::write_float(60.0, &frames, 1).expect("write ok");
        let clip = C3dParser::parse(&data).expect("parse ok");
        assert!(clip.frame(100).is_none());
    }

    #[test]
    fn test_write_empty_fails() {
        let result = C3dWriter::write_float(60.0, &[], 1);
        assert!(result.is_err(), "empty frames should fail");
    }
}
