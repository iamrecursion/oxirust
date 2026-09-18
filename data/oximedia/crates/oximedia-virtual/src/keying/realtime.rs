//! Real-time chroma key processor

use super::KeyingParams;
use crate::Result;

/// Real-time keyer
pub struct RealtimeKeyer {
    #[allow(dead_code)]
    params: KeyingParams,
}

impl RealtimeKeyer {
    /// Create new real-time keyer
    #[must_use]
    pub fn new(params: KeyingParams) -> Self {
        Self { params }
    }

    /// Apply keying to frame
    pub fn key(&mut self, frame: &[u8], _width: usize, _height: usize) -> Result<Vec<u8>> {
        Ok(frame.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_keyer() {
        let params = KeyingParams::default();
        let _keyer = RealtimeKeyer::new(params);
    }
}
