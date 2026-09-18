//! Encoder implementation

use super::{Encoder, Writer};
use crate::{config::Config, utils::Sealed};

/// An Encoder that writes bytes into a given writer `W`.
///
/// This struct should rarely be used directly.
/// In most cases, prefer the `encode_*` functions in the crate root.
///
/// # Example
///
/// ```rust,ignore
/// use oxicode::enc::{EncoderImpl, Encode, SliceWriter};
///
/// let slice: &mut [u8] = &mut [0, 0, 0, 0];
/// let config = oxicode::config::legacy().with_big_endian();
///
/// let mut encoder = EncoderImpl::new(SliceWriter::new(slice), config);
/// 5u32.encode(&mut encoder).unwrap();
/// assert_eq!(encoder.into_writer().bytes_written(), 4);
/// ```
pub struct EncoderImpl<W: Writer, C: Config> {
    writer: W,
    config: C,
}

impl<W: Writer, C: Config> EncoderImpl<W, C> {
    /// Create a new Encoder
    pub const fn new(writer: W, config: C) -> Self {
        Self { writer, config }
    }

    /// Return the underlying writer
    #[inline]
    pub fn into_writer(self) -> W {
        self.writer
    }
}

impl<W: Writer, C: Config> Encoder for EncoderImpl<W, C> {
    type W = W;
    type C = C;

    #[inline]
    fn writer(&mut self) -> &mut Self::W {
        &mut self.writer
    }

    #[inline]
    fn config(&self) -> &Self::C {
        &self.config
    }
}

impl<W: Writer, C: Config> Sealed for EncoderImpl<W, C> {}
