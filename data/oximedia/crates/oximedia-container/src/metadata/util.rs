//! Utility functions for metadata operations.

use async_trait::async_trait;
use oximedia_core::{OxiError, OxiResult};
use oximedia_io::MediaSource;

/// Extension trait for reading exact number of bytes.
#[async_trait]
pub(crate) trait MediaSourceExt {
    /// Reads exactly `buf.len()` bytes.
    async fn read_exact(&mut self, buf: &mut [u8]) -> OxiResult<()>;
}

#[async_trait]
impl<T: MediaSource> MediaSourceExt for T {
    async fn read_exact(&mut self, buf: &mut [u8]) -> OxiResult<()> {
        let mut offset = 0;
        while offset < buf.len() {
            let n = self.read(&mut buf[offset..]).await?;
            if n == 0 {
                return Err(OxiError::UnexpectedEof);
            }
            offset += n;
        }
        Ok(())
    }
}
