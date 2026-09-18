//! Media source trait definition.

use async_trait::async_trait;
use oximedia_core::OxiResult;
use std::io::SeekFrom;

/// Unified media source abstraction for reading from and writing to various inputs.
///
/// This trait provides an async interface for reading and writing media data from
/// files, network streams, memory buffers, and other sources.
///
/// # Implementors
///
/// - [`FileSource`](super::FileSource) - Local file access
/// - [`MemorySource`](super::MemorySource) - In-memory buffer access
///
/// # Example
///
/// ```no_run
/// use oximedia_io::source::{FileSource, MediaSource};
/// use std::io::SeekFrom;
///
/// #[tokio::main]
/// async fn main() -> oximedia_core::OxiResult<()> {
///     let mut source = FileSource::open("video.webm").await?;
///
///     // Read some bytes
///     let mut buffer = [0u8; 1024];
///     let n = source.read(&mut buffer).await?;
///
///     // Seek to a position
///     source.seek(SeekFrom::Start(0)).await?;
///
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait MediaSource: Send + Sync {
    /// Reads bytes into the provided buffer.
    ///
    /// Returns the number of bytes read. A return value of 0 indicates
    /// end of stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    async fn read(&mut self, buf: &mut [u8]) -> OxiResult<usize>;

    /// Writes all bytes from the buffer to the source.
    ///
    /// This method will continuously write bytes until the entire buffer
    /// has been written or an error occurs.
    ///
    /// # Errors
    ///
    /// Returns an error if the write operation fails or if writing
    /// is not supported by this source.
    async fn write_all(&mut self, buf: &[u8]) -> OxiResult<()>;

    /// Seeks to a position in the stream.
    ///
    /// Returns the new position from the start of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek operation fails or if the source
    /// is not seekable.
    async fn seek(&mut self, pos: SeekFrom) -> OxiResult<u64>;

    /// Returns the total length of the source in bytes, if known.
    ///
    /// Returns `None` for live streams or sources where the length
    /// cannot be determined.
    fn len(&self) -> Option<u64>;

    /// Returns `true` if the source is empty (zero length).
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Returns `true` if the source supports seeking.
    fn is_seekable(&self) -> bool;

    /// Returns the current position in the stream.
    fn position(&self) -> u64;

    /// Returns `true` if this source supports writing.
    fn is_writable(&self) -> bool {
        false
    }
}
