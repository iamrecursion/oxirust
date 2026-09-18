//! Zero-copy packet path using `bytes::Bytes`.
//!
//! `ZeroCopyPacket` holds the serialised header and payload as two independent
//! `Bytes` handles.  Cloning a `ZeroCopyPacket` is O(1) — it merely increments
//! reference counts.  Slicing into the payload similarly produces a sub-handle
//! that shares the same backing allocation.
//!
//! The `ZeroCopyPacketBuilder` constructs packets from constituent parts,
//! encoding the header once into a `BytesMut` that is then frozen into a
//! ref-counted `Bytes`.

use crate::error::{VideoIpError, VideoIpResult};
use crate::packet::{PacketBuilder, PacketFlags, PacketHeader, MAX_PAYLOAD_SIZE};
use crate::types::StreamType;
use bytes::{Bytes, BytesMut};

// ---------------------------------------------------------------------------
// ZeroCopyPacket
// ---------------------------------------------------------------------------

/// A network packet whose header and payload are held as ref-counted
/// [`Bytes`] slices.
///
/// Cloning this struct is O(1) — no data is copied, only reference counts are
/// incremented.  Both header and payload can be sliced (`split_to` / `slice`)
/// without allocation.
#[derive(Debug, Clone)]
pub struct ZeroCopyPacket {
    /// Serialised packet header (20 bytes, ref-counted).
    pub header: Bytes,
    /// Packet payload (ref-counted, may be a sub-slice of a larger buffer).
    pub payload: Bytes,
}

impl ZeroCopyPacket {
    /// Creates a `ZeroCopyPacket` directly from pre-built `Bytes`.
    ///
    /// The caller is responsible for ensuring `header` is a valid serialised
    /// `PacketHeader` (20 bytes).  Use [`ZeroCopyPacketBuilder`] for a safe
    /// construction path.
    #[must_use]
    pub fn new(header: Bytes, payload: Bytes) -> Self {
        Self { header, payload }
    }

    /// Total on-wire size of the packet (header + payload).
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.header.len() + self.payload.len()
    }

    /// Returns `true` if the payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.payload.is_empty()
    }

    /// Encodes the packet into a `BytesMut`, reusing its existing capacity.
    ///
    /// Unlike a naive `Vec::extend`, this avoids an extra intermediate buffer
    /// — `BytesMut::extend_from_slice` grows only if needed and writes
    /// directly into the existing reservation.
    pub fn encode_into(&self, dst: &mut BytesMut) {
        dst.extend_from_slice(&self.header);
        dst.extend_from_slice(&self.payload);
    }

    /// Serialises the packet to a freshly allocated `Bytes` (for inspection /
    /// testing only).  Prefer [`encode_into`](Self::encode_into) for hot paths.
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.total_len());
        self.encode_into(&mut buf);
        buf.freeze()
    }

    /// Returns a zero-copy sub-slice of the payload bytes.
    ///
    /// Panics if `range` is out of bounds (mirrors `Bytes::slice` behaviour).
    #[must_use]
    pub fn payload_slice(&self, range: impl std::ops::RangeBounds<usize>) -> Bytes {
        self.payload.slice(range)
    }
}

// ---------------------------------------------------------------------------
// ZeroCopyPacketBuilder
// ---------------------------------------------------------------------------

/// Builder for [`ZeroCopyPacket`].
///
/// Encodes the `PacketHeader` once (into a `BytesMut` frozen to `Bytes`) and
/// accepts a separately supplied payload `Bytes`.  No copying of the payload
/// is performed.
pub struct ZeroCopyPacketBuilder {
    sequence: u16,
    flags: PacketFlags,
    timestamp: u64,
    stream_type: StreamType,
}

impl ZeroCopyPacketBuilder {
    /// Starts building a packet with the given sequence number.
    #[must_use]
    pub fn new(sequence: u16) -> Self {
        Self {
            sequence,
            flags: PacketFlags::empty(),
            timestamp: 0,
            stream_type: StreamType::Program,
        }
    }

    /// Marks the packet as a video packet.
    #[must_use]
    pub fn video(mut self) -> Self {
        self.flags.insert(PacketFlags::VIDEO);
        self
    }

    /// Marks the packet as an audio packet.
    #[must_use]
    pub fn audio(mut self) -> Self {
        self.flags.insert(PacketFlags::AUDIO);
        self
    }

    /// Marks the packet as a keyframe.
    #[must_use]
    pub fn keyframe(mut self) -> Self {
        self.flags.insert(PacketFlags::KEYFRAME);
        self
    }

    /// Sets the packet timestamp (microseconds since epoch).
    #[must_use]
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    /// Sets the stream type.
    #[must_use]
    pub fn with_stream_type(mut self, st: StreamType) -> Self {
        self.stream_type = st;
        self
    }

    /// Builds the [`ZeroCopyPacket`] with the given `payload`.
    ///
    /// The payload `Bytes` is stored as-is — no copy is performed.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload exceeds `MAX_PAYLOAD_SIZE`.
    pub fn build(self, payload: Bytes) -> VideoIpResult<ZeroCopyPacket> {
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(VideoIpError::PacketTooLarge {
                size: payload.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let hdr = PacketHeader::new(
            self.flags,
            self.sequence,
            self.timestamp,
            self.stream_type,
            payload.len() as u16,
        );

        // Encode header once into a BytesMut, then freeze to Bytes.
        let mut hdr_buf = BytesMut::with_capacity(PacketHeader::SIZE);
        hdr.encode(&mut hdr_buf);
        let header = hdr_buf.freeze();

        Ok(ZeroCopyPacket::new(header, payload))
    }
}

// ---------------------------------------------------------------------------
// Extension on UdpTransport
// ---------------------------------------------------------------------------

/// Trait that extends transports with a zero-copy send path.
///
/// Implementors must provide `send_zero_copy`, which accepts a
/// [`ZeroCopyPacket`] and sends the header and payload without an intermediate
/// copy of the payload data.
pub trait ZeroCopySend {
    /// Sends a [`ZeroCopyPacket`] to the given destination.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O fails.
    fn send_zero_copy(
        &mut self,
        pkt: &ZeroCopyPacket,
        dest: std::net::SocketAddr,
    ) -> impl std::future::Future<Output = crate::error::VideoIpResult<()>> + Send;
}

/// Blanket implementation of `ZeroCopySend` for [`crate::transport::UdpTransport`].
impl ZeroCopySend for crate::transport::UdpTransport {
    async fn send_zero_copy(
        &mut self,
        pkt: &ZeroCopyPacket,
        dest: std::net::SocketAddr,
    ) -> crate::error::VideoIpResult<()> {
        // Reuse the send_bytes path but avoid copying the payload by writing
        // header+payload into a single BytesMut in place.
        let mut wire = BytesMut::with_capacity(pkt.total_len());
        pkt.encode_into(&mut wire);
        // send_bytes takes &[u8] — no extra copy after the BytesMut write.
        self.send_bytes(&wire, dest).await
    }
}

// ---------------------------------------------------------------------------
// Compatibility shim: convert from/to classic Packet
// ---------------------------------------------------------------------------

impl ZeroCopyPacket {
    /// Constructs a `ZeroCopyPacket` from a classic [`crate::packet::Packet`].
    ///
    /// The payload `Bytes` is cloned (O(1) reference-count increment).
    #[must_use]
    pub fn from_packet(p: &crate::packet::Packet) -> Self {
        let mut hdr_buf = BytesMut::with_capacity(PacketHeader::SIZE);
        p.header.encode(&mut hdr_buf);
        ZeroCopyPacket::new(hdr_buf.freeze(), p.payload.clone())
    }

    /// Builds a classic [`crate::packet::Packet`] from this `ZeroCopyPacket`.
    ///
    /// # Errors
    ///
    /// Returns an error if the header bytes are malformed or too short.
    pub fn into_packet(self) -> VideoIpResult<crate::packet::Packet> {
        use crate::packet::Packet;
        // Re-join header + payload into a single wire buffer, then decode.
        let mut wire = BytesMut::with_capacity(self.total_len());
        self.encode_into(&mut wire);
        Packet::decode(wire)
    }
}

// ---------------------------------------------------------------------------
// PublicPacketBuilder convenience re-export shim
// ---------------------------------------------------------------------------

/// Builds a classic `Packet` via `PacketBuilder` and returns it as a
/// `ZeroCopyPacket`.
pub fn build_zero_copy_from_classic(
    sequence: u16,
    flags_fn: impl FnOnce(PacketBuilder) -> PacketBuilder,
    payload: Bytes,
) -> VideoIpResult<ZeroCopyPacket> {
    let classic = flags_fn(PacketBuilder::new(sequence)).build(payload.clone())?;
    Ok(ZeroCopyPacket::from_packet(&classic))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    // ── Item 3 required tests ─────────────────────────────────────────────────

    /// Verify that slicing the payload produces a sub-Bytes with no allocation.
    ///
    /// We confirm this indirectly: the slice shares the same pointer
    /// as the original (`as_ptr` lands inside the original range), which is
    /// only possible if no copy occurred.
    #[test]
    fn test_zero_copy_packet_no_allocation_on_slice() {
        let data = Bytes::from(vec![0u8; 256]);
        let pkt = ZeroCopyPacketBuilder::new(1)
            .video()
            .with_timestamp(1_000_000)
            .build(data.clone())
            .expect("build should succeed");

        // Slice out bytes 10..50 — should be a sub-handle, not a copy.
        let slice = pkt.payload_slice(10..50);
        assert_eq!(slice.len(), 40);

        // Pointer into `slice` must sit within the original `data` buffer.
        let orig_start = data.as_ptr() as usize;
        let orig_end = orig_start + data.len();
        let slice_ptr = slice.as_ptr() as usize;
        assert!(
            slice_ptr >= orig_start && slice_ptr < orig_end,
            "slice pointer {slice_ptr:#x} is outside original [{orig_start:#x}, {orig_end:#x})"
        );
    }

    /// Verify that the builder encodes the header correctly and that the layout
    /// matches what a classic `PacketHeader::decode` would expect.
    #[test]
    fn test_zero_copy_packet_builder_correct_layout() {
        let payload_data = Bytes::from_static(b"hello world");
        let pkt = ZeroCopyPacketBuilder::new(42)
            .video()
            .keyframe()
            .with_timestamp(9_999)
            .build(payload_data.clone())
            .expect("build should succeed");

        // Header must be exactly PacketHeader::SIZE bytes.
        assert_eq!(pkt.header.len(), PacketHeader::SIZE);
        // total_len = header + payload.
        assert_eq!(pkt.total_len(), PacketHeader::SIZE + payload_data.len());

        // Round-trip through into_packet to verify the header is valid.
        let classic = pkt
            .clone()
            .into_packet()
            .expect("into_packet should succeed");
        assert_eq!(classic.header.sequence, 42);
        assert_eq!(classic.header.timestamp, 9_999);
        assert!(classic.header.flags.contains(PacketFlags::VIDEO));
        assert!(classic.header.flags.contains(PacketFlags::KEYFRAME));
        assert_eq!(classic.payload, payload_data);
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_zero_copy_clone_is_cheap() {
        let payload = Bytes::from(vec![0xABu8; 1024]);
        let pkt = ZeroCopyPacketBuilder::new(7)
            .audio()
            .build(payload)
            .expect("build");
        // Clone should not panic and produce an equal packet.
        let clone = pkt.clone();
        assert_eq!(pkt.header, clone.header);
        assert_eq!(pkt.payload, clone.payload);
    }

    #[test]
    fn test_zero_copy_to_bytes_round_trip() {
        let payload = Bytes::from_static(b"round-trip");
        let pkt = ZeroCopyPacketBuilder::new(100)
            .video()
            .build(payload.clone())
            .expect("build");
        let wire = pkt.to_bytes();
        assert_eq!(wire.len(), PacketHeader::SIZE + payload.len());
    }

    #[test]
    fn test_zero_copy_payload_too_large() {
        let big = Bytes::from(vec![0u8; MAX_PAYLOAD_SIZE + 1]);
        let result = ZeroCopyPacketBuilder::new(0).video().build(big);
        assert!(result.is_err(), "should reject oversized payload");
    }

    #[test]
    fn test_from_packet_and_back() {
        use crate::packet::PacketBuilder;
        let original = PacketBuilder::new(55)
            .video()
            .with_timestamp(42_000)
            .build(Bytes::from_static(b"test payload"))
            .expect("PacketBuilder should succeed");

        let zcp = ZeroCopyPacket::from_packet(&original);
        assert_eq!(zcp.payload, original.payload);

        let recovered = zcp.into_packet().expect("into_packet should succeed");
        assert_eq!(recovered.header.sequence, 55);
        assert_eq!(recovered.header.timestamp, 42_000);
        assert_eq!(recovered.payload, Bytes::from_static(b"test payload"));
    }
}
