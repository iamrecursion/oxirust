//! PGS (Presentation Graphic Stream) subtitle decoder.
//!
//! PGS is the subtitle format used in Blu-ray discs. It is a bitmap-based
//! format defined by the Blu-ray Disc Association.

use crate::{SubtitleError, SubtitleResult};
use std::collections::HashMap;

/// PGS subtitle decoder.
pub struct PgsDecoder {
    /// Palette definitions.
    palettes: HashMap<u8, Palette>,
    /// Object definitions.
    objects: HashMap<u16, PgsObject>,
    /// Window definitions.
    windows: HashMap<u8, Window>,
    /// Composition states.
    compositions: Vec<CompositionState>,
    /// Current presentation timestamp.
    current_pts: i64,
    /// Decoded subtitles.
    subtitles: Vec<PgsSubtitle>,
}

/// A PGS subtitle (bitmap-based).
#[derive(Clone, Debug)]
pub struct PgsSubtitle {
    /// Presentation timestamp in milliseconds.
    pub pts: i64,
    /// Decode timestamp in milliseconds (if different from PTS).
    pub dts: Option<i64>,
    /// Width of the subtitle.
    pub width: u16,
    /// Height of the subtitle.
    pub height: u16,
    /// Frame rate code.
    pub framerate: u8,
    /// Composition number.
    pub composition_number: u16,
    /// Composition state.
    pub composition_state: u8,
    /// Composition objects.
    pub objects: Vec<CompositionObject>,
}

/// Composition object.
#[derive(Clone, Debug)]
pub struct CompositionObject {
    /// Object ID.
    pub object_id: u16,
    /// Window ID.
    pub window_id: u8,
    /// Cropped flag.
    pub cropped: bool,
    /// Horizontal position.
    pub x: u16,
    /// Vertical position.
    pub y: u16,
    /// Crop rectangle (if cropped).
    pub crop: Option<CropRect>,
    /// Bitmap data (indexed color).
    pub bitmap: Vec<u8>,
    /// Palette.
    pub palette: Vec<PaletteEntry>,
}

/// Crop rectangle.
#[derive(Clone, Copy, Debug)]
pub struct CropRect {
    /// Crop X.
    pub x: u16,
    /// Crop Y.
    pub y: u16,
    /// Crop width.
    pub width: u16,
    /// Crop height.
    pub height: u16,
}

/// Palette definition.
#[derive(Clone, Debug)]
struct Palette {
    palette_id: u8,
    palette_version: u8,
    entries: Vec<PaletteEntry>,
}

/// Palette entry (RGBA).
#[derive(Clone, Copy, Debug)]
pub struct PaletteEntry {
    /// Palette entry ID.
    pub id: u8,
    /// Red component.
    pub r: u8,
    /// Green component.
    pub g: u8,
    /// Blue component.
    pub b: u8,
    /// Alpha component.
    pub a: u8,
}

/// Object definition.
#[derive(Clone, Debug)]
struct PgsObject {
    object_id: u16,
    object_version: u8,
    width: u16,
    height: u16,
    data: Vec<u8>,
}

/// Window definition.
#[derive(Clone, Debug)]
struct Window {
    window_id: u8,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
}

/// Composition state.
#[derive(Clone, Debug)]
struct CompositionState {
    composition_number: u16,
    composition_state: u8,
    palette_update_flag: bool,
    palette_id: u8,
    objects: Vec<CompositionObjectRef>,
}

/// Reference to a composition object.
#[derive(Clone, Debug)]
struct CompositionObjectRef {
    object_id: u16,
    window_id: u8,
    cropped: bool,
    x: u16,
    y: u16,
    crop: Option<CropRect>,
}

// Segment types
const PALETTE_SEGMENT: u8 = 0x14;
const OBJECT_SEGMENT: u8 = 0x15;
const PRESENTATION_SEGMENT: u8 = 0x16;
const WINDOW_SEGMENT: u8 = 0x17;
const END_SEGMENT: u8 = 0x80;

impl PgsDecoder {
    /// Create a new PGS subtitle decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            palettes: HashMap::new(),
            objects: HashMap::new(),
            windows: HashMap::new(),
            compositions: Vec::new(),
            current_pts: 0,
            subtitles: Vec::new(),
        }
    }

    /// Decode a PGS segment.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode_segment(
        &mut self,
        data: &[u8],
        pts: i64,
        dts: Option<i64>,
    ) -> SubtitleResult<()> {
        self.current_pts = pts;

        if data.len() < 13 {
            return Err(SubtitleError::ParseError(
                "PGS segment too short".to_string(),
            ));
        }

        // Parse segment header
        if &data[0..2] != b"PG" {
            return Err(SubtitleError::ParseError("Invalid PGS magic".to_string()));
        }

        let pts_u32 = u32::from_be_bytes([data[2], data[3], data[4], data[5]]);
        let dts_u32 = u32::from_be_bytes([data[6], data[7], data[8], data[9]]);

        let segment_type = data[10];
        let segment_size = u16::from_be_bytes([data[11], data[12]]) as usize;

        if data.len() < 13 + segment_size {
            return Err(SubtitleError::ParseError(
                "PGS segment data too short".to_string(),
            ));
        }

        let segment_data = &data[13..13 + segment_size];

        match segment_type {
            PALETTE_SEGMENT => self.decode_palette_segment(segment_data)?,
            OBJECT_SEGMENT => self.decode_object_segment(segment_data)?,
            PRESENTATION_SEGMENT => self.decode_presentation_segment(segment_data, pts)?,
            WINDOW_SEGMENT => self.decode_window_segment(segment_data)?,
            END_SEGMENT => {
                // End of display set - render subtitle
                self.render_subtitle(pts, dts)?;
            }
            _ => {
                // Unknown segment - skip
            }
        }

        Ok(())
    }

    /// Decode palette segment.
    fn decode_palette_segment(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 2 {
            return Ok(());
        }

        let palette_id = data[0];
        let palette_version = data[1];

        let mut entries = Vec::new();
        let mut pos = 2;

        while pos + 5 <= data.len() {
            let entry_id = data[pos];
            let y = data[pos + 1];
            let cr = data[pos + 2];
            let cb = data[pos + 3];
            let alpha = data[pos + 4];

            // Convert YCrCb to RGB
            let (r, g, b) = Self::ycrcb_to_rgb(y, cr, cb);

            entries.push(PaletteEntry {
                id: entry_id,
                r,
                g,
                b,
                a: alpha,
            });

            pos += 5;
        }

        self.palettes.insert(
            palette_id,
            Palette {
                palette_id,
                palette_version,
                entries,
            },
        );

        Ok(())
    }

    /// Decode object segment.
    fn decode_object_segment(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 11 {
            return Ok(());
        }

        let object_id = u16::from_be_bytes([data[0], data[1]]);
        let object_version = data[2];
        let sequence_flag = data[3];

        let data_len = (u32::from_be_bytes([0, data[4], data[5], data[6]])) as usize;
        let width = u16::from_be_bytes([data[7], data[8]]);
        let height = u16::from_be_bytes([data[9], data[10]]);

        let object_data = if data.len() > 11 {
            data[11..].to_vec()
        } else {
            Vec::new()
        };

        // Store or append object data
        if sequence_flag == 0x40 || sequence_flag == 0xC0 {
            // First or only segment
            self.objects.insert(
                object_id,
                PgsObject {
                    object_id,
                    object_version,
                    width,
                    height,
                    data: object_data,
                },
            );
        } else if sequence_flag == 0x80 {
            // Last segment - append
            if let Some(obj) = self.objects.get_mut(&object_id) {
                obj.data.extend_from_slice(&object_data);
            }
        }

        Ok(())
    }

    /// Decode presentation segment.
    fn decode_presentation_segment(&mut self, data: &[u8], pts: i64) -> SubtitleResult<()> {
        if data.len() < 11 {
            return Ok(());
        }

        let _width = u16::from_be_bytes([data[0], data[1]]);
        let _height = u16::from_be_bytes([data[2], data[3]]);
        let _framerate = data[4];
        let composition_number = u16::from_be_bytes([data[5], data[6]]);
        let composition_state = data[7];
        let palette_update_flag = data[8] != 0;
        let palette_id = data[9];
        let object_count = data[10];

        let mut objects = Vec::new();
        let mut pos = 11;

        for _ in 0..object_count {
            if pos + 8 > data.len() {
                break;
            }

            let object_id = u16::from_be_bytes([data[pos], data[pos + 1]]);
            let window_id = data[pos + 2];
            let object_cropped = data[pos + 3] != 0;
            let x = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
            let y = u16::from_be_bytes([data[pos + 6], data[pos + 7]]);

            pos += 8;

            let crop = if object_cropped && pos + 8 <= data.len() {
                let crop_x = u16::from_be_bytes([data[pos], data[pos + 1]]);
                let crop_y = u16::from_be_bytes([data[pos + 2], data[pos + 3]]);
                let crop_width = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);
                let crop_height = u16::from_be_bytes([data[pos + 6], data[pos + 7]]);
                pos += 8;
                Some(CropRect {
                    x: crop_x,
                    y: crop_y,
                    width: crop_width,
                    height: crop_height,
                })
            } else {
                None
            };

            objects.push(CompositionObjectRef {
                object_id,
                window_id,
                cropped: object_cropped,
                x,
                y,
                crop,
            });
        }

        self.compositions.push(CompositionState {
            composition_number,
            composition_state,
            palette_update_flag,
            palette_id,
            objects,
        });

        Ok(())
    }

    /// Decode window segment.
    fn decode_window_segment(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.is_empty() {
            return Ok(());
        }

        let window_count = data[0];
        let mut pos = 1;

        for _ in 0..window_count {
            if pos + 9 > data.len() {
                break;
            }

            let window_id = data[pos];
            let x = u16::from_be_bytes([data[pos + 1], data[pos + 2]]);
            let y = u16::from_be_bytes([data[pos + 3], data[pos + 4]]);
            let width = u16::from_be_bytes([data[pos + 5], data[pos + 6]]);
            let height = u16::from_be_bytes([data[pos + 7], data[pos + 8]]);

            self.windows.insert(
                window_id,
                Window {
                    window_id,
                    x,
                    y,
                    width,
                    height,
                },
            );

            pos += 9;
        }

        Ok(())
    }

    /// Render subtitle from current state.
    fn render_subtitle(&mut self, pts: i64, dts: Option<i64>) -> SubtitleResult<()> {
        if let Some(comp) = self.compositions.last() {
            let mut objects = Vec::new();

            for obj_ref in &comp.objects {
                if let Some(obj) = self.objects.get(&obj_ref.object_id) {
                    let palette = self
                        .palettes
                        .get(&comp.palette_id)
                        .map(|p| p.entries.clone())
                        .unwrap_or_default();

                    objects.push(CompositionObject {
                        object_id: obj.object_id,
                        window_id: obj_ref.window_id,
                        cropped: obj_ref.cropped,
                        x: obj_ref.x,
                        y: obj_ref.y,
                        crop: obj_ref.crop,
                        bitmap: obj.data.clone(),
                        palette,
                    });
                }
            }

            if !objects.is_empty() {
                self.subtitles.push(PgsSubtitle {
                    pts,
                    dts,
                    width: 1920, // Default Blu-ray resolution
                    height: 1080,
                    framerate: 0x10, // 24 fps
                    composition_number: comp.composition_number,
                    composition_state: comp.composition_state,
                    objects,
                });
            }
        }

        Ok(())
    }

    /// Convert YCrCb to RGB.
    fn ycrcb_to_rgb(y: u8, cr: u8, cb: u8) -> (u8, u8, u8) {
        let y_f = f32::from(y);
        let cr_f = f32::from(cr) - 128.0;
        let cb_f = f32::from(cb) - 128.0;

        let r = (y_f + 1.402 * cr_f).clamp(0.0, 255.0) as u8;
        let g = (y_f - 0.344136 * cb_f - 0.714136 * cr_f).clamp(0.0, 255.0) as u8;
        let b = (y_f + 1.772 * cb_f).clamp(0.0, 255.0) as u8;

        (r, g, b)
    }

    /// Get all decoded subtitles.
    #[must_use]
    pub fn take_subtitles(&mut self) -> Vec<PgsSubtitle> {
        std::mem::take(&mut self.subtitles)
    }

    /// Finalize decoding and return all subtitles.
    #[must_use]
    pub fn finalize(self) -> Vec<PgsSubtitle> {
        self.subtitles
    }
}

impl Default for PgsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ycrcb_conversion() {
        let (r, g, b) = PgsDecoder::ycrcb_to_rgb(255, 128, 128);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);

        let (r, g, b) = PgsDecoder::ycrcb_to_rgb(0, 128, 128);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = PgsDecoder::new();
        assert_eq!(decoder.palettes.len(), 0);
        assert_eq!(decoder.objects.len(), 0);
    }
}
