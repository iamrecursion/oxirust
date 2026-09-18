//! Image frame data structures.

use crate::{ColorSpace, PixelType};
use bytes::Bytes;
use std::collections::HashMap;

/// Image frame with pixel data and metadata.
#[derive(Clone, Debug)]
pub struct ImageFrame {
    /// Frame number in sequence.
    pub frame_number: u32,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,

    /// Pixel data type.
    pub pixel_type: PixelType,

    /// Number of color components (e.g., 3 for RGB, 4 for RGBA).
    pub components: u8,

    /// Color space.
    pub color_space: ColorSpace,

    /// Pixel data (interleaved or planar).
    pub data: ImageData,

    /// Frame metadata.
    pub metadata: HashMap<String, String>,
}

impl ImageFrame {
    /// Creates a new image frame.
    #[must_use]
    pub fn new(
        frame_number: u32,
        width: u32,
        height: u32,
        pixel_type: PixelType,
        components: u8,
        color_space: ColorSpace,
        data: ImageData,
    ) -> Self {
        Self {
            frame_number,
            width,
            height,
            pixel_type,
            components,
            color_space,
            data,
            metadata: HashMap::new(),
        }
    }

    /// Returns the total number of pixels in this frame.
    #[must_use]
    pub const fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Returns the stride (bytes per row) for this frame.
    #[must_use]
    pub fn stride(&self) -> usize {
        (self.width as usize) * (self.components as usize) * self.pixel_type.bytes_per_component()
    }

    /// Adds metadata to this frame.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Image pixel data storage.
#[derive(Clone, Debug)]
pub enum ImageData {
    /// Interleaved pixel data (e.g., RGBRGBRGB...).
    Interleaved(Bytes),

    /// Planar pixel data (e.g., RRR...GGG...BBB...).
    Planar(Vec<Bytes>),
}

impl ImageData {
    /// Creates interleaved image data.
    #[must_use]
    pub fn interleaved(data: Vec<u8>) -> Self {
        Self::Interleaved(Bytes::from(data))
    }

    /// Creates planar image data.
    #[must_use]
    pub fn planar(planes: Vec<Vec<u8>>) -> Self {
        Self::Planar(planes.into_iter().map(Bytes::from).collect())
    }

    /// Returns true if data is interleaved.
    #[must_use]
    pub const fn is_interleaved(&self) -> bool {
        matches!(self, Self::Interleaved(_))
    }

    /// Returns true if data is planar.
    #[must_use]
    pub const fn is_planar(&self) -> bool {
        matches!(self, Self::Planar(_))
    }

    /// Returns the number of planes (1 for interleaved).
    #[must_use]
    pub fn plane_count(&self) -> usize {
        match self {
            Self::Interleaved(_) => 1,
            Self::Planar(planes) => planes.len(),
        }
    }

    /// Returns a reference to the data as a slice (for interleaved data).
    #[must_use]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Self::Interleaved(data) => Some(data),
            Self::Planar(_) => None,
        }
    }

    /// Returns a reference to the planes (for planar data).
    #[must_use]
    pub fn as_planes(&self) -> Option<&[Bytes]> {
        match self {
            Self::Interleaved(_) => None,
            Self::Planar(planes) => Some(planes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_frame_creation() {
        let data = ImageData::interleaved(vec![0u8; 100]);
        let frame = ImageFrame::new(1, 10, 10, PixelType::U8, 3, ColorSpace::Srgb, data);

        assert_eq!(frame.frame_number, 1);
        assert_eq!(frame.width, 10);
        assert_eq!(frame.height, 10);
        assert_eq!(frame.pixel_count(), 100);
    }

    #[test]
    fn test_image_data_interleaved() {
        let data = ImageData::interleaved(vec![1, 2, 3, 4]);
        assert!(data.is_interleaved());
        assert!(!data.is_planar());
        assert_eq!(data.plane_count(), 1);
        assert_eq!(data.as_slice(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn test_image_data_planar() {
        let data = ImageData::planar(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        assert!(!data.is_interleaved());
        assert!(data.is_planar());
        assert_eq!(data.plane_count(), 3);
        assert!(data.as_slice().is_none());
    }

    #[test]
    fn test_frame_metadata() {
        let data = ImageData::interleaved(vec![0u8; 100]);
        let mut frame = ImageFrame::new(1, 10, 10, PixelType::U8, 3, ColorSpace::Srgb, data);

        frame.add_metadata("camera".to_string(), "ARRI ALEXA".to_string());
        assert_eq!(frame.get_metadata("camera"), Some("ARRI ALEXA"));
        assert_eq!(frame.get_metadata("lens"), None);
    }
}
