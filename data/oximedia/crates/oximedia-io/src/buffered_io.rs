//! Buffered I/O: read-ahead buffering, write coalescing, and buffer pools.
//!
//! Provides a lightweight synchronous buffered I/O layer that sits on top of
//! any byte slice or cursor, offering configurable buffer sizes, read-ahead
//! semantics, and write coalescing to reduce syscall overhead.

use std::collections::VecDeque;
use std::io::{self, Read, Write};

// ---------------------------------------------------------------------------
// Buffer pool
// ---------------------------------------------------------------------------

/// A fixed-size pool of reusable byte buffers.
#[derive(Debug)]
pub struct BufferPool {
    buf_size: usize,
    free: VecDeque<Vec<u8>>,
    capacity: usize,
}

impl BufferPool {
    /// Create a new pool with `capacity` buffers of `buf_size` bytes each.
    #[must_use]
    pub fn new(buf_size: usize, capacity: usize) -> Self {
        let mut free = VecDeque::with_capacity(capacity);
        for _ in 0..capacity {
            free.push_back(vec![0u8; buf_size]);
        }
        Self {
            buf_size,
            free,
            capacity,
        }
    }

    /// Acquire a buffer from the pool. Returns `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Option<Vec<u8>> {
        if let Some(mut buf) = self.free.pop_front() {
            // Zero the buffer for safety before handing it out.
            buf.fill(0);
            Some(buf)
        } else {
            None
        }
    }

    /// Return a buffer to the pool.  If the pool is full the buffer is dropped.
    pub fn release(&mut self, buf: Vec<u8>) {
        if self.free.len() < self.capacity {
            self.free.push_back(buf);
        }
    }

    #[must_use]
    pub fn available(&self) -> usize {
        self.free.len()
    }

    #[must_use]
    pub fn buf_size(&self) -> usize {
        self.buf_size
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Read-ahead buffer
// ---------------------------------------------------------------------------

/// Wraps a `Read` source with a configurable read-ahead buffer.
pub struct ReadAheadBuffer<R: Read> {
    inner: R,
    buf: Vec<u8>,
    pos: usize,
    filled: usize,
}

impl<R: Read> ReadAheadBuffer<R> {
    /// Create a new read-ahead buffer with `buf_size` bytes of capacity.
    pub fn new(inner: R, buf_size: usize) -> Self {
        Self {
            inner,
            buf: vec![0u8; buf_size],
            pos: 0,
            filled: 0,
        }
    }

    /// Fill the internal buffer from the underlying reader if it is empty.
    fn fill_buf(&mut self) -> io::Result<usize> {
        if self.pos >= self.filled {
            self.pos = 0;
            self.filled = self.inner.read(&mut self.buf)?;
        }
        Ok(self.filled - self.pos)
    }

    /// Read up to `dst.len()` bytes, filling from the buffer first.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the underlying reader fails.
    pub fn read_bytes(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        if available == 0 {
            return Ok(0);
        }
        let n = dst.len().min(available);
        dst[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }

    /// Bytes currently available without hitting the underlying reader.
    pub fn buffered(&self) -> usize {
        self.filled.saturating_sub(self.pos)
    }

    /// Consume the wrapper and return the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Write coalescing buffer
// ---------------------------------------------------------------------------

/// Coalesces small writes into larger chunks before flushing to the sink.
pub struct CoalescingWriter<W: Write> {
    inner: W,
    buf: Vec<u8>,
    threshold: usize,
    total_written: u64,
    flush_count: u64,
}

impl<W: Write> CoalescingWriter<W> {
    /// Create a new coalescing writer with a flush `threshold` in bytes.
    pub fn new(inner: W, threshold: usize) -> Self {
        Self {
            inner,
            buf: Vec::with_capacity(threshold),
            threshold,
            total_written: 0,
            flush_count: 0,
        }
    }

    /// Buffer `data`, flushing to the sink when the threshold is reached.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if flushing to the underlying writer fails.
    pub fn write_bytes(&mut self, data: &[u8]) -> io::Result<()> {
        self.buf.extend_from_slice(data);
        self.total_written += data.len() as u64;
        if self.buf.len() >= self.threshold {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush any buffered data to the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing to the underlying writer fails.
    pub fn flush(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            self.inner.write_all(&self.buf)?;
            self.buf.clear();
            self.flush_count += 1;
        }
        Ok(())
    }

    pub fn buffered_bytes(&self) -> usize {
        self.buf.len()
    }

    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    pub fn flush_count(&self) -> u64 {
        self.flush_count
    }

    /// Flush and return the inner writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the final flush fails.
    pub fn into_inner(mut self) -> io::Result<W> {
        self.flush()?;
        Ok(self.inner)
    }
}

// ---------------------------------------------------------------------------
// I/O cursor helper for tests
// ---------------------------------------------------------------------------

/// A simple in-memory cursor that implements `Read` and `Write`.
#[derive(Debug, Default)]
pub struct MemCursor {
    data: Vec<u8>,
    read_pos: usize,
}

impl MemCursor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data, read_pos: 0 }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Read for MemCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.data.len().saturating_sub(self.read_pos);
        if remaining == 0 {
            return Ok(0);
        }
        let n = buf.len().min(remaining);
        buf[..n].copy_from_slice(&self.data[self.read_pos..self.read_pos + n]);
        self.read_pos += n;
        Ok(n)
    }
}

impl Write for MemCursor {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.data.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Double-buffered reader
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Shared state between the caller and the background fill thread.
struct SharedState<R: Read + Send + 'static> {
    /// The underlying reader, moved here so the background thread can own it.
    reader: Mutex<Option<R>>,
    /// The back buffer being filled by the background thread.
    back_buf: Mutex<BackBuf>,
    /// Condvar signalled by the background thread when `back_buf` is ready,
    /// and by the caller when the back buffer should be refilled.
    condvar: Condvar,
    /// Set to `true` by `Drop` to ask the background thread to stop.
    cancelled: AtomicBool,
}

/// Contents of the back buffer.
struct BackBuf {
    data: Vec<u8>,
    len: usize,
    /// `true` once the background thread has finished a fill pass.
    ready: bool,
    /// `true` if the underlying reader returned 0 bytes (EOF).
    eof: bool,
    /// `true` if the underlying reader returned an error.
    error: bool,
}

/// A reader that uses two alternating buffers so a background thread can fill
/// the next buffer while the caller drains the current one, hiding I/O latency.
///
/// # Concurrency model
///
/// * The **background thread** owns the reader (via `SharedState::reader`).
///   It loops: fill the back buffer → lock `back_buf`, set `ready=true`,
///   signal the condvar → wait until `ready` is cleared again before refilling.
/// * The **foreground (caller)** drains the front buffer via [`Read::read`].
///   When the front is exhausted it waits on the condvar for `ready=true`,
///   swaps front and back, clears `ready`, signals the condvar (let background
///   refill), then continues serving bytes from the new front.
pub struct DoubleBufferedReader<R: Read + Send + 'static> {
    /// Buffer being consumed by the caller.
    front: Vec<u8>,
    /// How many valid bytes are in `front`.
    front_len: usize,
    /// Next unread offset within `front`.
    front_pos: usize,
    /// Shared state for coordination with the background thread.
    shared: Arc<SharedState<R>>,
    /// Handle to the background fill thread.
    fill_thread: Option<std::thread::JoinHandle<()>>,
    /// Set once we have drained the very last filled buffer.
    eof: bool,
}

impl<R: Read + Send + 'static> DoubleBufferedReader<R> {
    /// Create a new `DoubleBufferedReader` wrapping `reader` with two buffers
    /// of `buf_size` bytes each, and spawn the background fill thread.
    pub fn new(reader: R, buf_size: usize) -> Self {
        let buf_size = buf_size.max(1);

        let shared = Arc::new(SharedState {
            reader: Mutex::new(Some(reader)),
            back_buf: Mutex::new(BackBuf {
                data: vec![0u8; buf_size],
                len: 0,
                ready: false,
                eof: false,
                error: false,
            }),
            condvar: Condvar::new(),
            cancelled: AtomicBool::new(false),
        });

        let shared_clone = Arc::clone(&shared);
        let fill_thread = std::thread::spawn(move || {
            Self::fill_loop(shared_clone, buf_size);
        });

        // The front buffer starts empty; the first `read()` call will block
        // until the background thread completes its first fill.
        Self {
            front: vec![0u8; buf_size],
            front_len: 0,
            front_pos: 0,
            shared,
            fill_thread: Some(fill_thread),
            eof: false,
        }
    }

    /// Background thread loop: fill the back buffer then wait to be asked again.
    fn fill_loop(shared: Arc<SharedState<R>>, buf_size: usize) {
        loop {
            if shared.cancelled.load(Ordering::Acquire) {
                break;
            }

            // Read into a temporary local buffer to avoid holding the lock
            // during I/O.
            let mut tmp = vec![0u8; buf_size];
            let (n, eof, error) = {
                let mut guard = shared.reader.lock().unwrap_or_else(|e| e.into_inner());
                match guard.as_mut() {
                    None => (0, true, false),
                    Some(r) => match r.read(&mut tmp) {
                        Ok(0) => (0, true, false),
                        Ok(n) => (n, false, false),
                        Err(_) => (0, false, true),
                    },
                }
            };

            // Publish results into the back buffer and signal the caller.
            {
                let mut back = shared.back_buf.lock().unwrap_or_else(|e| e.into_inner());
                back.data[..n].copy_from_slice(&tmp[..n]);
                back.len = n;
                back.ready = true;
                back.eof = eof;
                back.error = error;
            }
            shared.condvar.notify_one();

            if eof || error || shared.cancelled.load(Ordering::Acquire) {
                break;
            }

            // Wait until the caller swaps and clears `ready` before we
            // overwrite the back buffer.
            let mut back = shared.back_buf.lock().unwrap_or_else(|e| e.into_inner());
            back = shared
                .condvar
                .wait_while(back, |b| {
                    b.ready && !shared.cancelled.load(Ordering::Acquire)
                })
                .unwrap_or_else(|e| e.into_inner());
            drop(back);
        }
    }

    /// Wait for the back buffer to be ready, then swap it with the front.
    /// Returns `false` if the back buffer signals EOF or error.
    fn swap_buffers(&mut self) -> io::Result<bool> {
        let mut back = self
            .shared
            .back_buf
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Wait until the background thread marks back as ready.
        back = self
            .shared
            .condvar
            .wait_while(back, |b| !b.ready)
            .unwrap_or_else(|e| e.into_inner());

        if back.error {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "background fill thread encountered a read error",
            ));
        }

        let filled = back.len;
        let eof = back.eof;

        // Swap front and back buffers.
        std::mem::swap(&mut self.front, &mut back.data);
        self.front_len = filled;
        self.front_pos = 0;

        // Signal the background thread that it may refill the back buffer.
        back.ready = false;
        drop(back);
        self.shared.condvar.notify_one();

        if eof && filled == 0 {
            self.eof = true;
            return Ok(false);
        }

        Ok(true)
    }
}

impl<R: Read + Send + 'static> Read for DoubleBufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.eof {
            return Ok(0);
        }

        // Refill front if exhausted (including the very first call).
        while self.front_pos >= self.front_len {
            if !self.swap_buffers()? {
                return Ok(0); // EOF
            }
            if self.front_len > 0 {
                break;
            }
        }

        let available = self.front_len - self.front_pos;
        let n = buf.len().min(available);
        buf[..n].copy_from_slice(&self.front[self.front_pos..self.front_pos + n]);
        self.front_pos += n;
        Ok(n)
    }
}

impl<R: Read + Send + 'static> Drop for DoubleBufferedReader<R> {
    fn drop(&mut self) {
        // Signal the background thread to stop.
        self.shared.cancelled.store(true, Ordering::Release);

        // Drop the inner reader so any blocked `read()` in the background
        // thread will get an error / return 0 (for Cursor-like sources it
        // returns immediately).
        if let Ok(mut guard) = self.shared.reader.lock() {
            *guard = None;
        }

        // Wake the background thread in case it is waiting on the condvar.
        {
            if let Ok(mut back) = self.shared.back_buf.lock() {
                back.ready = false; // allow the thread to exit its wait_while
            }
        }
        self.shared.condvar.notify_all();

        // Join the fill thread.
        if let Some(handle) = self.fill_thread.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_acquire_and_release() {
        let mut pool = BufferPool::new(512, 4);
        assert_eq!(pool.available(), 4);
        let buf = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available(), 3);
        assert_eq!(buf.len(), 512);
        pool.release(buf);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_buffer_pool_exhaustion() {
        let mut pool = BufferPool::new(64, 2);
        let _b1 = pool.acquire().expect("acquire should succeed");
        let _b2 = pool.acquire().expect("acquire should succeed");
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn test_buffer_pool_capacity_and_buf_size() {
        let pool = BufferPool::new(1024, 8);
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.buf_size(), 1024);
    }

    #[test]
    fn test_buffer_pool_release_beyond_capacity_drops() {
        let mut pool = BufferPool::new(16, 1);
        // Already full
        pool.release(vec![0u8; 16]);
        assert_eq!(pool.available(), 1); // still 1, extra was dropped
    }

    #[test]
    fn test_read_ahead_buffer_reads_all_data() {
        let src = MemCursor::from_bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let mut rab = ReadAheadBuffer::new(src, 4);
        let mut out = [0u8; 8];
        let mut total = 0;
        while total < 8 {
            let n = rab
                .read_bytes(&mut out[total..])
                .expect("read_bytes should succeed");
            if n == 0 {
                break;
            }
            total += n;
        }
        assert_eq!(total, 8);
        assert_eq!(&out, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_read_ahead_buffer_returns_zero_on_eof() {
        let src = MemCursor::from_bytes(vec![]);
        let mut rab = ReadAheadBuffer::new(src, 16);
        let mut buf = [0u8; 4];
        let n = rab.read_bytes(&mut buf).expect("read_bytes should succeed");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_read_ahead_buffer_buffered_count() {
        let src = MemCursor::from_bytes(vec![10, 20, 30, 40]);
        let mut rab = ReadAheadBuffer::new(src, 4);
        // Prime the buffer
        let mut buf = [0u8; 2];
        rab.read_bytes(&mut buf).expect("read_bytes should succeed");
        assert_eq!(rab.buffered(), 2);
    }

    #[test]
    fn test_coalescing_writer_does_not_flush_below_threshold() {
        let sink = MemCursor::new();
        let mut cw = CoalescingWriter::new(sink, 16);
        cw.write_bytes(&[1, 2, 3])
            .expect("write_bytes should succeed");
        assert_eq!(cw.buffered_bytes(), 3);
        assert_eq!(cw.flush_count(), 0);
    }

    #[test]
    fn test_coalescing_writer_flushes_at_threshold() {
        let sink = MemCursor::new();
        let mut cw = CoalescingWriter::new(sink, 4);
        cw.write_bytes(&[1, 2, 3, 4])
            .expect("write_bytes should succeed");
        assert_eq!(cw.flush_count(), 1);
        assert_eq!(cw.buffered_bytes(), 0);
    }

    #[test]
    fn test_coalescing_writer_total_written() {
        let sink = MemCursor::new();
        let mut cw = CoalescingWriter::new(sink, 256);
        cw.write_bytes(&[0u8; 100])
            .expect("write_bytes should succeed");
        cw.write_bytes(&[0u8; 50])
            .expect("write_bytes should succeed");
        assert_eq!(cw.total_written(), 150);
    }

    #[test]
    fn test_coalescing_writer_into_inner_flushes() {
        let sink = MemCursor::new();
        let mut cw = CoalescingWriter::new(sink, 256);
        cw.write_bytes(&[9, 8, 7])
            .expect("write_bytes should succeed");
        let out = cw.into_inner().expect("into_inner should succeed");
        assert_eq!(out.as_slice(), &[9, 8, 7]);
    }

    #[test]
    fn test_mem_cursor_read_write() {
        let mut cur = MemCursor::new();
        cur.write_all(&[1, 2, 3]).expect("failed to write");
        assert_eq!(cur.len(), 3);
        assert!(!cur.is_empty());
    }

    #[test]
    fn test_mem_cursor_from_bytes() {
        let cur = MemCursor::from_bytes(vec![5, 6]);
        assert_eq!(cur.as_slice(), &[5, 6]);
    }

    // -----------------------------------------------------------------------
    // DoubleBufferedReader tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_double_buffered_same_as_plain_read() {
        let original: Vec<u8> = (0u8..=255).collect();
        let cursor = std::io::Cursor::new(original.clone());
        let mut dbr = DoubleBufferedReader::new(cursor, 64);
        let mut out = Vec::new();
        io::Read::read_to_end(&mut dbr, &mut out).expect("read_to_end should succeed");
        assert_eq!(out, original);
    }

    #[test]
    fn test_double_buffered_empty() {
        let cursor = std::io::Cursor::new(Vec::<u8>::new());
        let mut dbr = DoubleBufferedReader::new(cursor, 64);
        let mut out = Vec::new();
        io::Read::read_to_end(&mut dbr, &mut out).expect("read_to_end should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn test_double_buffered_larger_than_buf() {
        // Source is 1 KiB, buffer size is 32 bytes — many swap cycles.
        let original: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let cursor = std::io::Cursor::new(original.clone());
        let mut dbr = DoubleBufferedReader::new(cursor, 32);
        let mut out = Vec::new();
        io::Read::read_to_end(&mut dbr, &mut out).expect("read_to_end should succeed");
        assert_eq!(out, original);
    }
}
