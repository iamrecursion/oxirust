//! VP9 bitstream parsing utilities.

#![allow(dead_code)]

/// Boolean decoder for VP9 entropy coding.
#[derive(Debug, Default)]
pub struct BoolDecoder {
    pub(crate) range: u32,
    pub(crate) value: u32,
    pub(crate) count: i32,
}

impl BoolDecoder {
    /// Creates a new boolean decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            range: 255,
            value: 0,
            count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_decoder_new() {
        let decoder = BoolDecoder::new();
        assert_eq!(decoder.range, 255);
    }
}
