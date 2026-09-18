//! Ogg logical stream writer.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use super::writer::OggPage;

// ============================================================================
// Page Flags
// ============================================================================

/// Continuation flag (packet continues from previous page).
const FLAG_CONTINUATION: u8 = 0x01;

/// Beginning of stream flag.
const FLAG_BOS: u8 = 0x02;

/// End of stream flag.
const FLAG_EOS: u8 = 0x04;

// ============================================================================
// Ogg Stream Writer
// ============================================================================

/// Writer for a single logical Ogg stream.
///
/// Manages page sequencing, segmentation, and granule position tracking
/// for a single stream within an Ogg container.
#[derive(Debug)]
pub struct OggStreamWriter {
    /// Stream serial number.
    serial: u32,

    /// Current page sequence number.
    sequence: u32,

    /// Last granule position written.
    last_granule: u64,

    /// Buffer for incomplete packets.
    buffer: Vec<u8>,
}

impl OggStreamWriter {
    /// Creates a new Ogg stream writer.
    ///
    /// # Arguments
    ///
    /// * `serial` - Stream serial number (should be unique within container)
    #[must_use]
    pub const fn new(serial: u32) -> Self {
        Self {
            serial,
            sequence: 0,
            last_granule: 0,
            buffer: Vec::new(),
        }
    }

    /// Returns the stream serial number.
    #[must_use]
    pub const fn serial(&self) -> u32 {
        self.serial
    }

    /// Returns the current page sequence number.
    #[must_use]
    pub const fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Returns the last granule position.
    #[must_use]
    pub const fn last_granule(&self) -> u64 {
        self.last_granule
    }

    /// Builds an Ogg page from packet data.
    ///
    /// # Arguments
    ///
    /// * `data` - Packet data to include in the page
    /// * `continuation` - Whether this continues a packet from previous page
    /// * `eos` - Whether this is the last page of the stream
    /// * `complete` - Whether the packet ends on this page
    ///
    /// # Returns
    ///
    /// The constructed `OggPage`.
    #[must_use]
    pub fn build_page(
        &mut self,
        data: &[u8],
        continuation: bool,
        eos: bool,
        complete: bool,
    ) -> OggPage {
        self.build_page_with_granule(data, continuation, eos, complete, 0)
    }

    /// Builds an Ogg page with a specific granule position.
    ///
    /// # Arguments
    ///
    /// * `data` - Packet data to include in the page
    /// * `continuation` - Whether this continues a packet from previous page
    /// * `eos` - Whether this is the last page of the stream
    /// * `complete` - Whether the packet ends on this page
    /// * `granule` - Granule position for this page
    ///
    /// # Returns
    ///
    /// The constructed `OggPage`.
    #[must_use]
    pub fn build_page_with_granule(
        &mut self,
        data: &[u8],
        continuation: bool,
        eos: bool,
        complete: bool,
        granule: u64,
    ) -> OggPage {
        let mut page = OggPage::new(self.serial, self.sequence);

        // Set flags
        let mut flags = 0u8;
        if continuation {
            flags |= FLAG_CONTINUATION;
        }
        if self.sequence == 0 && !continuation {
            flags |= FLAG_BOS;
        }
        if eos {
            flags |= FLAG_EOS;
        }
        page.flags = flags;

        // Set granule position
        // Only set granule if packet is complete (otherwise use -1)
        page.granule_position = if complete && granule != u64::MAX {
            granule
        } else {
            u64::MAX
        };

        // Build segment table
        page.segments = build_segment_table(data.len(), complete);

        // Set data
        page.data = data.to_vec();

        // Update state
        self.sequence += 1;
        if complete && granule != u64::MAX {
            self.last_granule = granule;
        }

        page
    }

    /// Builds a BOS (beginning of stream) page.
    ///
    /// # Arguments
    ///
    /// * `data` - Packet data for the BOS page
    ///
    /// # Returns
    ///
    /// The constructed BOS `OggPage`.
    #[must_use]
    pub fn build_bos_page(&mut self, data: &[u8]) -> OggPage {
        let mut page = OggPage::new(self.serial, self.sequence);
        page.flags = FLAG_BOS;
        page.granule_position = 0;
        page.segments = build_segment_table(data.len(), true);
        page.data = data.to_vec();

        self.sequence += 1;
        page
    }

    /// Builds an EOS (end of stream) page.
    ///
    /// # Arguments
    ///
    /// * `granule` - Final granule position
    ///
    /// # Returns
    ///
    /// The constructed EOS `OggPage`.
    #[must_use]
    pub fn build_eos_page(&mut self, granule: u64) -> OggPage {
        let mut page = OggPage::new(self.serial, self.sequence);
        page.flags = FLAG_EOS;
        page.granule_position = granule;
        page.segments = vec![0]; // Empty packet
        page.data = Vec::new();

        self.sequence += 1;
        self.last_granule = granule;
        page
    }

    /// Appends data to the internal buffer.
    ///
    /// Used when building multi-page packets.
    pub fn append_to_buffer(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Takes and clears the internal buffer.
    #[must_use]
    pub fn take_buffer(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buffer)
    }

    /// Returns true if the buffer has data.
    #[must_use]
    pub fn has_buffered_data(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Resets the stream state.
    pub fn reset(&mut self) {
        self.sequence = 0;
        self.last_granule = 0;
        self.buffer.clear();
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Builds a segment table for packet data.
///
/// In Ogg, packets are divided into segments of up to 255 bytes.
/// A segment of size < 255 indicates the end of a packet.
/// A segment of size 255 indicates continuation.
///
/// # Arguments
///
/// * `data_size` - Total size of the packet data
/// * `complete` - Whether this is a complete packet
///
/// # Returns
///
/// The segment table as a vector of segment sizes.
#[must_use]
fn build_segment_table(data_size: usize, complete: bool) -> Vec<u8> {
    let mut segments = Vec::new();
    let mut remaining = data_size;

    while remaining >= 255 {
        segments.push(255);
        remaining -= 255;
    }

    // Add final segment
    if complete {
        // Complete packet: add remaining bytes (even if 0)
        segments.push(remaining as u8);
    } else if remaining > 0 {
        // Incomplete packet: add remaining bytes without terminator
        segments.push(remaining as u8);
    }
    // If incomplete and remaining == 0, we've hit a page boundary exactly

    segments
}

/// Calculates the number of segments needed for a packet.
///
/// # Arguments
///
/// * `data_size` - Size of the packet data
///
/// # Returns
///
/// Number of segments required.
#[must_use]
#[allow(dead_code)]
pub fn calculate_segment_count(data_size: usize) -> usize {
    if data_size == 0 {
        1 // Empty packet still needs one zero-length segment
    } else {
        data_size.div_ceil(255)
    }
}

/// Splits packet data into page-sized chunks.
///
/// Each page can hold up to 255 segments of 255 bytes each.
///
/// # Arguments
///
/// * `data` - The packet data to split
///
/// # Returns
///
/// Vector of data chunks, each suitable for one page.
#[must_use]
#[allow(dead_code)]
pub fn split_packet_for_pages(data: &[u8]) -> Vec<Vec<u8>> {
    const MAX_PAGE_DATA: usize = 255 * 255;
    let mut chunks = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        let chunk_size = (data.len() - offset).min(MAX_PAGE_DATA);
        chunks.push(data[offset..offset + chunk_size].to_vec());
        offset += chunk_size;
    }

    if chunks.is_empty() {
        chunks.push(Vec::new());
    }

    chunks
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_writer_new() {
        let writer = OggStreamWriter::new(0x1234);
        assert_eq!(writer.serial(), 0x1234);
        assert_eq!(writer.sequence(), 0);
        assert_eq!(writer.last_granule(), 0);
    }

    #[test]
    fn test_build_segment_table_small() {
        let segments = build_segment_table(100, true);
        assert_eq!(segments, vec![100]);
    }

    #[test]
    fn test_build_segment_table_exact() {
        let segments = build_segment_table(255, true);
        assert_eq!(segments, vec![255, 0]);
    }

    #[test]
    fn test_build_segment_table_large() {
        let segments = build_segment_table(600, true);
        assert_eq!(segments, vec![255, 255, 90]);
    }

    #[test]
    fn test_build_segment_table_empty() {
        let segments = build_segment_table(0, true);
        assert_eq!(segments, vec![0]);
    }

    #[test]
    fn test_build_page() {
        let mut writer = OggStreamWriter::new(1);
        let page = writer.build_page(&[1, 2, 3, 4, 5], false, false, true);

        assert_eq!(page.serial_number, 1);
        assert_eq!(page.page_sequence, 0);
        assert_eq!(page.flags & FLAG_BOS, FLAG_BOS);
        assert_eq!(page.segments, vec![5]);
        assert_eq!(page.data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_build_page_sequence() {
        let mut writer = OggStreamWriter::new(1);

        let page1 = writer.build_page(&[1, 2], false, false, true);
        assert_eq!(page1.page_sequence, 0);

        let page2 = writer.build_page(&[3, 4], false, false, true);
        assert_eq!(page2.page_sequence, 1);

        let page3 = writer.build_page(&[5, 6], false, false, true);
        assert_eq!(page3.page_sequence, 2);
    }

    #[test]
    fn test_build_page_with_granule() {
        let mut writer = OggStreamWriter::new(1);
        let page = writer.build_page_with_granule(&[1, 2, 3], false, false, true, 48000);

        assert_eq!(page.granule_position, 48000);
        assert_eq!(writer.last_granule(), 48000);
    }

    #[test]
    fn test_build_bos_page() {
        let mut writer = OggStreamWriter::new(1);
        let page = writer.build_bos_page(&[1, 2, 3]);

        assert_eq!(page.flags, FLAG_BOS);
        assert_eq!(page.granule_position, 0);
    }

    #[test]
    fn test_build_eos_page() {
        let mut writer = OggStreamWriter::new(1);
        writer.sequence = 5;
        let page = writer.build_eos_page(100000);

        assert_eq!(page.flags, FLAG_EOS);
        assert_eq!(page.granule_position, 100000);
        assert_eq!(page.page_sequence, 5);
    }

    #[test]
    fn test_buffer_operations() {
        let mut writer = OggStreamWriter::new(1);

        assert!(!writer.has_buffered_data());

        writer.append_to_buffer(&[1, 2, 3]);
        assert!(writer.has_buffered_data());

        let data = writer.take_buffer();
        assert_eq!(data, vec![1, 2, 3]);
        assert!(!writer.has_buffered_data());
    }

    #[test]
    fn test_reset() {
        let mut writer = OggStreamWriter::new(1);
        writer.sequence = 10;
        writer.last_granule = 48000;
        writer.append_to_buffer(&[1, 2, 3]);

        writer.reset();

        assert_eq!(writer.sequence(), 0);
        assert_eq!(writer.last_granule(), 0);
        assert!(!writer.has_buffered_data());
    }

    #[test]
    fn test_calculate_segment_count() {
        assert_eq!(calculate_segment_count(0), 1);
        assert_eq!(calculate_segment_count(100), 1);
        assert_eq!(calculate_segment_count(255), 1);
        assert_eq!(calculate_segment_count(256), 2);
        assert_eq!(calculate_segment_count(510), 2);
        assert_eq!(calculate_segment_count(511), 3);
    }

    #[test]
    fn test_split_packet_for_pages() {
        // Small packet
        let chunks = split_packet_for_pages(&[1, 2, 3]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], vec![1, 2, 3]);

        // Empty packet
        let chunks = split_packet_for_pages(&[]);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_empty());
    }

    #[test]
    fn test_continuation_flag() {
        let mut writer = OggStreamWriter::new(1);

        // First page (BOS, no continuation)
        let page1 = writer.build_page(&[1, 2], false, false, true);
        assert_eq!(page1.flags & FLAG_CONTINUATION, 0);

        // Continuation page
        let page2 = writer.build_page(&[3, 4], true, false, true);
        assert_eq!(page2.flags & FLAG_CONTINUATION, FLAG_CONTINUATION);
    }

    #[test]
    fn test_granule_not_set_for_incomplete() {
        let mut writer = OggStreamWriter::new(1);
        let page = writer.build_page_with_granule(&[1, 2, 3], false, false, false, 48000);

        assert_eq!(page.granule_position, u64::MAX);
    }
}
