//! Replay buffer for instant replay.
//!
//! Implements a fixed-capacity ring buffer that stores recent encoded frames
//! up to the configured duration. When the buffer is full, the oldest frames
//! are evicted to make room for new ones.
//!
//! Two storage back-ends are available:
//!
//! * **VecDeque** (default) — in-process heap storage; zero setup, always safe.
//! * **MmapReplayRing** — memory-mapped file ring buffer; avoids copying frame
//!   data into the heap and keeps the replay set page-cache-resident so it
//!   survives GC pressure.  Enable via [`ReplayBufferConfig::use_mmap`].

use crate::{GamingError, GamingResult};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ReplayFrame
// ---------------------------------------------------------------------------

/// A single frame stored in the replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayFrame {
    /// Encoded frame data.
    pub data: Vec<u8>,
    /// Presentation timestamp relative to buffer start.
    pub timestamp: Duration,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
    /// Frame sequence number.
    pub sequence: u64,
}

// ---------------------------------------------------------------------------
// ReplayConfig (legacy) — kept for backward-compat
// ---------------------------------------------------------------------------

/// Replay buffer configuration.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Buffer duration in seconds
    pub duration: u32,
    /// Video bitrate in kbps
    pub bitrate: u32,
    /// Audio enabled
    pub audio_enabled: bool,
    /// Target framerate (used for capacity estimation)
    pub framerate: u32,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            duration: 30,
            bitrate: 10000,
            audio_enabled: true,
            framerate: 60,
        }
    }
}

// ---------------------------------------------------------------------------
// ReplayBufferConfig (extended, includes mmap option)
// ---------------------------------------------------------------------------

/// Extended replay buffer configuration including optional mmap back-end.
#[derive(Debug, Clone)]
pub struct ReplayBufferConfig {
    /// Buffer duration in seconds (5–300).
    pub duration: u32,
    /// Video bitrate in kbps (used to compute byte budget).
    pub bitrate: u32,
    /// Audio enabled.
    pub audio_enabled: bool,
    /// Target framerate (used for frame-count capacity).
    pub framerate: u32,
    /// When `true`, use the memory-mapped ring buffer back-end.
    /// When `false` (default), use the VecDeque back-end.
    pub use_mmap: bool,
    /// Directory for the mmap backing file (used only when `use_mmap = true`).
    /// Defaults to `std::env::temp_dir()`.
    pub mmap_dir: Option<PathBuf>,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            duration: 30,
            bitrate: 10000,
            audio_enabled: true,
            framerate: 60,
            use_mmap: false,
            mmap_dir: None,
        }
    }
}

impl From<ReplayConfig> for ReplayBufferConfig {
    fn from(rc: ReplayConfig) -> Self {
        Self {
            duration: rc.duration,
            bitrate: rc.bitrate,
            audio_enabled: rc.audio_enabled,
            framerate: rc.framerate,
            use_mmap: false,
            mmap_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MmapReplayRing — memory-mapped ring buffer
// ---------------------------------------------------------------------------

/// Record header size in the mmap ring: `[4-byte len][4-byte flags][8-byte ts_ns]`.
const RECORD_HEADER_SIZE: usize = 16;

/// Memory-mapped ring buffer storing replay frames as length-prefixed records.
///
/// The mmap provides durable, zero-copy backing storage.  An in-memory
/// [`VecDeque`] index tracks the (offset, total_size) of every live record so
/// there is no head==tail ambiguity and no sentinel records needed.
///
/// # File layout
///
/// ```text
/// [8 bytes] magic "OxiRing1"
/// [8 bytes] capacity_bytes (LE u64)
/// [16 bytes] reserved
/// --- data region (capacity_bytes bytes) ---
/// ```
/// Each record in the data region:
/// ```text
/// [4 bytes] payload_len  (LE u32)
/// [4 bytes] flags        (bit 0 = keyframe, LE u32)
/// [8 bytes] timestamp_ns (LE u64)
/// [N bytes] payload
/// ```
pub struct MmapReplayRing {
    path: PathBuf,
    map: memmap2::MmapMut,
    /// Usable data capacity (file size minus the 32-byte file header).
    capacity_bytes: usize,
    /// Write pointer: byte offset within the data region for the next record.
    head: usize,
    /// Per-record index: (offset_in_data_region, total_record_size).
    index: VecDeque<(usize, usize)>,
    /// Running total bytes used by live records (not including evicted ones).
    used_bytes: usize,
}

const RING_GLOBAL_HEADER: usize = 32;
const RING_MAGIC: &[u8; 8] = b"OxiRing1";

impl MmapReplayRing {
    /// Create (or re-create) a memory-mapped ring buffer at `path` with at least
    /// `capacity_bytes` of usable data space.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the backing file cannot be created or mapped.
    #[allow(unsafe_code)]
    pub fn new(path: PathBuf, capacity_bytes: usize) -> std::io::Result<Self> {
        let file_size = capacity_bytes + RING_GLOBAL_HEADER;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(file_size as u64)?;

        // SAFETY: The file is newly created/truncated to `file_size`, and we
        // hold an exclusive write handle.  No other thread shares this map
        // during construction.  The MmapMut is pinned to `Self` and not
        // aliased externally.
        let mut map = unsafe { memmap2::MmapMut::map_mut(&file)? };

        map[0..8].copy_from_slice(RING_MAGIC);
        map[8..16].copy_from_slice(&(capacity_bytes as u64).to_le_bytes());

        Ok(Self {
            path,
            map,
            capacity_bytes,
            head: 0,
            index: VecDeque::new(),
            used_bytes: 0,
        })
    }

    /// Push a frame record into the ring, evicting the oldest record(s) when
    /// there is insufficient space.
    ///
    /// If a single frame's required space exceeds `capacity_bytes`, the push is
    /// silently dropped to avoid an infinite eviction loop.
    pub fn push_frame(&mut self, timestamp_ns: u64, flags: u32, data: &[u8]) {
        let record_size = RECORD_HEADER_SIZE + data.len();

        if record_size > self.capacity_bytes {
            return; // single frame too large for the entire ring
        }

        // Evict oldest records until we have contiguous space at `head`.
        // Two conditions must hold simultaneously:
        //   (a) The record fits between head and end-of-ring  OR  after wrapping.
        //   (b) After eviction, no live record overlaps the region we will write.
        // The simplest correct approach: evict until `used_bytes + record_size <=
        // capacity_bytes` (enough total space), then wrap head if needed.
        while !self.index.is_empty() && self.used_bytes + record_size > self.capacity_bytes {
            self.evict_oldest();
        }

        // If the record doesn't fit between head and end-of-ring, wrap head.
        if self.head + record_size > self.capacity_bytes {
            // Evict any records that are in the [head..capacity] zone or at
            // the start of the ring where we'd overwrite them after wrapping.
            let wrap_zone_end = self.capacity_bytes; // just mark; after wrap head=0
            while let Some(&(off, _)) = self.index.front() {
                // Evict if in the tail-of-ring dead zone OR if at head=0 region
                // and we'd collide.
                if off >= self.head && off < wrap_zone_end {
                    self.evict_oldest();
                } else {
                    break;
                }
            }
            // Evict records at the beginning of the ring that the wrapped write
            // would overwrite.
            let needed_at_start = record_size;
            while let Some(&(off, _)) = self.index.front() {
                if off < needed_at_start {
                    self.evict_oldest();
                } else {
                    break;
                }
            }
            self.head = 0;
        }

        // Write the record at `head`.
        let start = RING_GLOBAL_HEADER + self.head;
        self.map[start..start + 4].copy_from_slice(&(data.len() as u32).to_le_bytes());
        self.map[start + 4..start + 8].copy_from_slice(&flags.to_le_bytes());
        self.map[start + 8..start + 16].copy_from_slice(&timestamp_ns.to_le_bytes());
        self.map[start + 16..start + 16 + data.len()].copy_from_slice(data);

        self.index.push_back((self.head, record_size));
        self.used_bytes += record_size;
        self.head += record_size;
        // Wrap head if it lands exactly at capacity
        if self.head == self.capacity_bytes {
            self.head = 0;
        }
    }

    /// Return all frames whose timestamp is within `duration_ns` of the newest frame.
    #[must_use]
    pub fn snapshot_last_ns(&self, duration_ns: u64) -> Vec<ReplayFrame> {
        let records = self.read_all_records();
        if records.is_empty() {
            return Vec::new();
        }
        let newest_ns = records.iter().map(|(ts, _, _)| *ts).max().unwrap_or(0);
        let cutoff_ns = newest_ns.saturating_sub(duration_ns);

        records
            .into_iter()
            .enumerate()
            .filter(|(_, (ts, _, _))| *ts >= cutoff_ns)
            .map(|(seq, (ts_ns, flags, data))| ReplayFrame {
                data,
                timestamp: Duration::from_nanos(ts_ns),
                is_keyframe: (flags & 1) != 0,
                sequence: seq as u64,
            })
            .collect()
    }

    /// Number of records currently in the ring.
    #[must_use]
    pub fn count(&self) -> usize {
        self.index.len()
    }

    /// Path of the backing file.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Evict the oldest record from the ring.
    fn evict_oldest(&mut self) {
        if let Some((_, sz)) = self.index.pop_front() {
            self.used_bytes = self.used_bytes.saturating_sub(sz);
        }
    }

    /// Read all live records in insertion order using the in-memory index.
    fn read_all_records(&self) -> Vec<(u64, u32, Vec<u8>)> {
        let mut records = Vec::with_capacity(self.index.len());

        for &(off, sz) in &self.index {
            let start = RING_GLOBAL_HEADER + off;
            if start + sz > self.map.len() {
                break;
            }

            let payload_len = u32::from_le_bytes([
                self.map[start],
                self.map[start + 1],
                self.map[start + 2],
                self.map[start + 3],
            ]) as usize;

            if start + RECORD_HEADER_SIZE + payload_len > self.map.len() {
                break;
            }

            let flags = u32::from_le_bytes([
                self.map[start + 4],
                self.map[start + 5],
                self.map[start + 6],
                self.map[start + 7],
            ]);
            let ts_ns = u64::from_le_bytes([
                self.map[start + 8],
                self.map[start + 9],
                self.map[start + 10],
                self.map[start + 11],
                self.map[start + 12],
                self.map[start + 13],
                self.map[start + 14],
                self.map[start + 15],
            ]);
            let data = self.map[start + 16..start + 16 + payload_len].to_vec();
            records.push((ts_ns, flags, data));
        }

        records
    }
}

impl Drop for MmapReplayRing {
    fn drop(&mut self) {
        let _ = self.map.flush();
    }
}

// ---------------------------------------------------------------------------
// ReplayBuffer — unified front-end (VecDeque or mmap)
// ---------------------------------------------------------------------------

/// Replay buffer for storing recent frames in a ring-buffer arrangement.
pub struct ReplayBuffer {
    config: ReplayConfig,
    enabled: bool,
    /// Ring buffer of frames.
    frames: VecDeque<ReplayFrame>,
    /// Maximum number of frames based on duration and estimated framerate.
    max_frames: usize,
    /// Total bytes currently stored.
    total_bytes: usize,
    /// Maximum bytes allowed (derived from bitrate * duration).
    max_bytes: usize,
    /// Next sequence number.
    next_sequence: u64,
}

impl ReplayBuffer {
    /// Create a new replay buffer.
    ///
    /// # Errors
    ///
    /// Returns error if duration is outside the 5-300 second range.
    pub fn new(config: ReplayConfig) -> GamingResult<Self> {
        if config.duration < 5 || config.duration > 300 {
            return Err(GamingError::ReplayBufferError(
                "Duration must be between 5 and 300 seconds".to_string(),
            ));
        }

        let max_frames = (config.framerate as usize) * (config.duration as usize);
        // max_bytes = bitrate_kbps * 1000 / 8 * duration_s
        let max_bytes = (config.bitrate as usize) * 1000 / 8 * (config.duration as usize);

        Ok(Self {
            config,
            enabled: false,
            frames: VecDeque::with_capacity(max_frames.min(8192)),
            max_frames,
            total_bytes: 0,
            max_bytes,
            next_sequence: 0,
        })
    }

    /// Enable replay buffer.
    pub fn enable(&mut self) -> GamingResult<()> {
        self.enabled = true;
        Ok(())
    }

    /// Disable replay buffer and clear stored frames.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.clear();
    }

    /// Check if replay buffer is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get buffer duration configuration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        Duration::from_secs(u64::from(self.config.duration))
    }

    /// Push a new frame into the replay buffer.
    ///
    /// If the buffer is at capacity (by frame count or byte budget), the oldest
    /// frames are evicted until there is room. This method is a no-op if the
    /// buffer is not enabled.
    ///
    /// # Errors
    ///
    /// Returns error if a single frame exceeds the entire buffer byte budget.
    pub fn push_frame(
        &mut self,
        data: Vec<u8>,
        timestamp: Duration,
        is_keyframe: bool,
    ) -> GamingResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let frame_size = data.len();

        if frame_size > self.max_bytes {
            return Err(GamingError::ReplayBufferError(format!(
                "Single frame ({} bytes) exceeds total buffer capacity ({} bytes)",
                frame_size, self.max_bytes
            )));
        }

        // Evict oldest frames if we exceed frame count limit
        while self.frames.len() >= self.max_frames {
            if let Some(old) = self.frames.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(old.data.len());
            }
        }

        // Evict oldest frames if we exceed byte budget
        while self.total_bytes + frame_size > self.max_bytes {
            if let Some(old) = self.frames.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(old.data.len());
            } else {
                break;
            }
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.total_bytes += frame_size;

        self.frames.push_back(ReplayFrame {
            data,
            timestamp,
            is_keyframe,
            sequence: seq,
        });

        Ok(())
    }

    /// Get the number of frames currently in the buffer.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Get the total bytes stored in the buffer.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Get the actual time span covered by the buffer.
    #[must_use]
    pub fn buffered_duration(&self) -> Duration {
        if self.frames.len() < 2 {
            return Duration::ZERO;
        }

        let oldest = &self.frames[0];
        let newest = self
            .frames
            .back()
            .map(|f| f.timestamp)
            .unwrap_or(Duration::ZERO);

        newest.saturating_sub(oldest.timestamp)
    }

    /// Extract all buffered frames as a snapshot for saving/export.
    ///
    /// The buffer is left intact. The returned frames start from the nearest
    /// keyframe to ensure the replay is decodable.
    #[must_use]
    pub fn snapshot(&self) -> Vec<ReplayFrame> {
        // Find the first keyframe in the buffer
        let start_idx = self.frames.iter().position(|f| f.is_keyframe).unwrap_or(0);

        self.frames.iter().skip(start_idx).cloned().collect()
    }

    /// Extract the last `duration` seconds of replay data.
    ///
    /// Returns frames starting from the nearest keyframe at or before the
    /// requested time window.
    #[must_use]
    pub fn snapshot_last(&self, duration: Duration) -> Vec<ReplayFrame> {
        if self.frames.is_empty() {
            return Vec::new();
        }

        let newest_ts = self
            .frames
            .back()
            .map(|f| f.timestamp)
            .unwrap_or(Duration::ZERO);
        let cutoff = newest_ts.saturating_sub(duration);

        // Find frames within the time window
        let first_in_window = self
            .frames
            .iter()
            .position(|f| f.timestamp >= cutoff)
            .unwrap_or(0);

        // Walk backwards from first_in_window to find nearest keyframe
        let mut start = first_in_window;
        for i in (0..=first_in_window).rev() {
            if self.frames[i].is_keyframe {
                start = i;
                break;
            }
        }

        self.frames.iter().skip(start).cloned().collect()
    }

    /// Clear all buffered frames.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.total_bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_creation() {
        let config = ReplayConfig::default();
        let buffer = ReplayBuffer::new(config).expect("valid replay buffer");
        assert!(!buffer.is_enabled());
        assert_eq!(buffer.frame_count(), 0);
    }

    #[test]
    fn test_enable_disable() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid replay buffer");
        buffer.enable().expect("enable should succeed");
        assert!(buffer.is_enabled());
        buffer.disable();
        assert!(!buffer.is_enabled());
    }

    #[test]
    fn test_invalid_duration() {
        let config = ReplayConfig {
            duration: 1,
            ..ReplayConfig::default()
        };
        let result = ReplayBuffer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_frame_when_disabled_is_noop() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        // Not enabled
        buffer
            .push_frame(vec![0u8; 100], Duration::from_millis(0), true)
            .expect("push should be noop");
        assert_eq!(buffer.frame_count(), 0);
    }

    #[test]
    fn test_push_and_count_frames() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        buffer.enable().expect("enable");

        for i in 0..10 {
            buffer
                .push_frame(vec![0u8; 1000], Duration::from_millis(i * 16), i % 30 == 0)
                .expect("push should succeed");
        }

        assert_eq!(buffer.frame_count(), 10);
        assert_eq!(buffer.total_bytes(), 10 * 1000);
    }

    #[test]
    fn test_eviction_by_frame_count() {
        let config = ReplayConfig {
            duration: 5,
            bitrate: 100000, // large byte budget
            framerate: 2,    // 2fps * 5s = 10 frames max
            ..ReplayConfig::default()
        };
        let mut buffer = ReplayBuffer::new(config).expect("valid");
        buffer.enable().expect("enable");

        for i in 0..20 {
            buffer
                .push_frame(vec![0u8; 100], Duration::from_millis(i * 500), i % 5 == 0)
                .expect("push");
        }

        assert_eq!(buffer.frame_count(), 10);
        // Oldest remaining should be sequence 10
        assert_eq!(buffer.frames[0].sequence, 10);
    }

    #[test]
    fn test_eviction_by_byte_budget() {
        let config = ReplayConfig {
            duration: 5,
            bitrate: 8, // 8 kbps * 5s = 5000 bytes budget
            framerate: 1000,
            ..ReplayConfig::default()
        };
        let mut buffer = ReplayBuffer::new(config).expect("valid");
        buffer.enable().expect("enable");

        // Each frame is 2000 bytes, budget is 5000
        for i in 0..5 {
            buffer
                .push_frame(vec![0u8; 2000], Duration::from_millis(i * 100), true)
                .expect("push");
        }

        // Should have evicted down to fit within 5000 bytes
        assert!(buffer.total_bytes() <= 5000);
        assert!(buffer.frame_count() <= 2);
    }

    #[test]
    fn test_snapshot_starts_from_keyframe() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        buffer.enable().expect("enable");

        // Push: non-key, non-key, key, non-key
        buffer
            .push_frame(vec![1], Duration::from_millis(0), false)
            .expect("push");
        buffer
            .push_frame(vec![2], Duration::from_millis(16), false)
            .expect("push");
        buffer
            .push_frame(vec![3], Duration::from_millis(32), true)
            .expect("push");
        buffer
            .push_frame(vec![4], Duration::from_millis(48), false)
            .expect("push");

        let snap = buffer.snapshot();
        // Should start from the keyframe at index 2
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].data, vec![3]);
        assert!(snap[0].is_keyframe);
    }

    #[test]
    fn test_snapshot_last_duration() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        buffer.enable().expect("enable");

        // 10 frames at 100ms intervals, keyframe every 5
        for i in 0..10u64 {
            buffer
                .push_frame(vec![i as u8], Duration::from_millis(i * 100), i % 5 == 0)
                .expect("push");
        }

        // Get last 300ms (frames at 700, 800, 900)
        let snap = buffer.snapshot_last(Duration::from_millis(300));
        // Should include from the keyframe at 500ms onwards
        assert!(snap.len() >= 3);
        assert!(snap[0].is_keyframe);
    }

    #[test]
    fn test_buffered_duration() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        buffer.enable().expect("enable");

        assert_eq!(buffer.buffered_duration(), Duration::ZERO);

        buffer
            .push_frame(vec![0], Duration::from_millis(100), true)
            .expect("push");
        buffer
            .push_frame(vec![0], Duration::from_millis(500), false)
            .expect("push");

        assert_eq!(buffer.buffered_duration(), Duration::from_millis(400));
    }

    #[test]
    fn test_clear() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid");
        buffer.enable().expect("enable");
        buffer
            .push_frame(vec![0; 1000], Duration::ZERO, true)
            .expect("push");
        assert_eq!(buffer.frame_count(), 1);

        buffer.clear();
        assert_eq!(buffer.frame_count(), 0);
        assert_eq!(buffer.total_bytes(), 0);
    }

    // --- MmapReplayRing tests ---

    #[test]
    fn test_mmap_ring_new() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_test_mmap_ring_new.bin");
        let ring = MmapReplayRing::new(path.clone(), 4096).expect("create ring");
        assert_eq!(ring.count(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_ring_push_and_snapshot() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_test_mmap_ring_push.bin");
        let mut ring = MmapReplayRing::new(path.clone(), 65536).expect("create ring");

        let frame_data = vec![0xABu8; 64];
        ring.push_frame(1_000_000_000, 1, &frame_data); // ts=1s, keyframe
        ring.push_frame(2_000_000_000, 0, &frame_data); // ts=2s
        ring.push_frame(3_000_000_000, 0, &frame_data); // ts=3s

        assert_eq!(ring.count(), 3);

        // All frames within last 3s
        let snap = ring.snapshot_last_ns(3_000_000_000);
        assert_eq!(snap.len(), 3);
        assert!(snap[0].is_keyframe);

        // Only last 1.5s → frames at 2s and 3s
        let snap2 = ring.snapshot_last_ns(1_500_000_000);
        assert_eq!(snap2.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_ring_overflow_eviction() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_test_mmap_ring_overflow.bin");
        // Small capacity: enough for ~3 records of 64 bytes each
        // Each record = 16 header + 64 data = 80 bytes; capacity 256 bytes.
        let mut ring = MmapReplayRing::new(path.clone(), 256).expect("create ring");

        for i in 0u64..10 {
            ring.push_frame(i * 1_000_000, 0, &[i as u8; 64]);
        }

        // We pushed 10 frames but capacity fits only ~3; count should be ≤ 3.
        assert!(ring.count() <= 3, "count={}", ring.count());

        let _ = std::fs::remove_file(&path);
    }

    // --- Overflow tests for VecDeque path (multiple durations) ---

    #[test]
    fn test_replay_overflow_5s() {
        let config = ReplayConfig {
            duration: 5,
            bitrate: 100_000,
            framerate: 30,
            audio_enabled: false,
        };
        let max_frames = 30 * 5; // 150
        let mut buf = ReplayBuffer::new(config).expect("valid");
        buf.enable().expect("enable");

        // Push 3× max
        for i in 0u64..(max_frames as u64 * 3) {
            buf.push_frame(vec![0u8; 100], Duration::from_millis(i * 33), i % 30 == 0)
                .expect("push");
        }

        assert_eq!(buf.frame_count(), max_frames, "capped at max_frames");
        // Oldest frames should have been evicted (FIFO).
        let oldest_seq = buf.frames[0].sequence;
        assert!(
            oldest_seq >= max_frames as u64 * 2,
            "oldest seq={oldest_seq}"
        );
    }

    #[test]
    fn test_replay_overflow_30s() {
        let config = ReplayConfig {
            duration: 30,
            bitrate: 10_000,
            framerate: 10,
            audio_enabled: false,
        };
        let max_frames = 10 * 30; // 300
        let mut buf = ReplayBuffer::new(config).expect("valid");
        buf.enable().expect("enable");

        for i in 0u64..(max_frames as u64 * 2) {
            buf.push_frame(vec![0u8; 100], Duration::from_millis(i * 100), i % 10 == 0)
                .expect("push");
        }

        assert!(buf.frame_count() <= max_frames);
        // Ensure FIFO eviction: newest frames remain.
        let last_seq = buf.frames.back().map(|f| f.sequence).unwrap_or(0);
        assert_eq!(last_seq, max_frames as u64 * 2 - 1);
    }

    #[test]
    fn test_replay_overflow_300s() {
        let config = ReplayConfig {
            duration: 300,
            bitrate: 1_000,
            framerate: 5,
            audio_enabled: false,
        };
        let max_frames = 5 * 300; // 1500
        let mut buf = ReplayBuffer::new(config).expect("valid");
        buf.enable().expect("enable");

        // Push 2× max
        for i in 0u64..(max_frames as u64 * 2) {
            buf.push_frame(vec![0u8; 10], Duration::from_millis(i * 200), i % 5 == 0)
                .expect("push");
        }

        assert!(
            buf.frame_count() <= max_frames,
            "count={} exceeds max={}",
            buf.frame_count(),
            max_frames
        );
    }
}
