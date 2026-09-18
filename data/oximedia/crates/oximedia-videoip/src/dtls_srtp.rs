//! DTLS-SRTP key extraction and RTP packet protection.
//!
//! DTLS-SRTP (RFC 5764) derives SRTP keying material from a DTLS handshake
//! via the `use_srtp` extension.  After the handshake the key material is
//! extracted with `TLS_PRF` and split into four pieces:
//!
//! ```text
//! client_write_key | server_write_key | client_write_salt | server_write_salt
//! ```
//!
//! This module models that split, holds the resulting SRTP contexts for send
//! (protect) and receive (unprotect) directions, and provides a test-friendly
//! XOR-based cipher that faithfully replicates the *structural* transformations
//! of AES-CTR SRTP (header preservation, payload encryption, auth tag append)
//! without requiring a native AES implementation.
//!
//! For production traffic the XOR cipher should be replaced by a proper
//! AES-128-CTR + HMAC-SHA1-80 (or AEAD AES-128-GCM) implementation.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::VideoIpError;

// ── SRTP Profile ─────────────────────────────────────────────────────────────

/// SRTP crypto profile negotiated in the DTLS `use_srtp` extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtpProfile {
    /// AES-128-CM with HMAC-SHA1-80 (RFC 4568 — 10-byte auth tag).
    AesCm128HmacSha1_80,
    /// AES-128-CM with HMAC-SHA1-32 (RFC 4568 — 4-byte auth tag).
    AesCm128HmacSha1_32,
    /// AEAD AES-128-GCM with SHA-256 (RFC 7714 — 16-byte auth tag).
    AesGcm128Sha256,
}

impl SrtpProfile {
    /// Authentication tag length in bytes appended to each protected packet.
    #[must_use]
    pub const fn auth_tag_len(self) -> usize {
        match self {
            SrtpProfile::AesCm128HmacSha1_80 => 10,
            SrtpProfile::AesCm128HmacSha1_32 => 4,
            SrtpProfile::AesGcm128Sha256 => 16,
        }
    }

    /// Master key length in bytes (always 16 for AES-128).
    #[must_use]
    pub const fn key_len(self) -> usize {
        16
    }

    /// Master salt length in bytes.
    #[must_use]
    pub const fn salt_len(self) -> usize {
        match self {
            SrtpProfile::AesCm128HmacSha1_80 | SrtpProfile::AesCm128HmacSha1_32 => 14,
            SrtpProfile::AesGcm128Sha256 => 12,
        }
    }

    /// Human-readable RFC 4568 / RFC 7714 profile name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            SrtpProfile::AesCm128HmacSha1_80 => "AES_CM_128_HMAC_SHA1_80",
            SrtpProfile::AesCm128HmacSha1_32 => "AES_CM_128_HMAC_SHA1_32",
            SrtpProfile::AesGcm128Sha256 => "AEAD_AES_128_GCM",
        }
    }

    /// Minimum total packet length for an encrypted packet of this profile
    /// (12-byte RTP header + auth tag).
    #[must_use]
    pub const fn min_protected_len(self) -> usize {
        RTP_HEADER_MIN_LEN + self.auth_tag_len()
    }
}

// ── Key Material ─────────────────────────────────────────────────────────────

/// Minimum RTP header length (no CSRC, no extension) in bytes.
pub const RTP_HEADER_MIN_LEN: usize = 12;

/// SRTP keying material for one direction (send or receive).
///
/// The fields correspond to the output of the DTLS `TLS_PRF` key-derivation
/// function as described in RFC 5764 §4.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtpKeyMaterial {
    /// Local (outbound) master key — 16 bytes.
    pub local_key: [u8; 16],
    /// Local (outbound) master salt — 14 bytes.
    pub local_salt: [u8; 14],
    /// Remote (inbound) master key — 16 bytes.
    pub remote_key: [u8; 16],
    /// Remote (inbound) master salt — 14 bytes.
    pub remote_salt: [u8; 14],
}

impl SrtpKeyMaterial {
    /// Construct key material from raw byte arrays.
    #[must_use]
    pub const fn new(
        local_key: [u8; 16],
        local_salt: [u8; 14],
        remote_key: [u8; 16],
        remote_salt: [u8; 14],
    ) -> Self {
        Self {
            local_key,
            local_salt,
            remote_key,
            remote_salt,
        }
    }

    /// Create zero-filled key material (for tests / bootstrapping).
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            local_key: [0u8; 16],
            local_salt: [0u8; 14],
            remote_key: [0u8; 16],
            remote_salt: [0u8; 14],
        }
    }

    /// Construct from a DTLS keying-material export blob.
    ///
    /// RFC 5764 §4.2: the `use_srtp` key-material block is laid out as:
    /// `client_key(16) | server_key(16) | client_salt(14) | server_salt(14)`.
    /// Pass `is_client = true` when this peer acted as the DTLS client.
    ///
    /// # Errors
    ///
    /// Returns an error if `export` is shorter than 60 bytes.
    pub fn from_dtls_export(export: &[u8], is_client: bool) -> Result<Self, VideoIpError> {
        const EXPECTED: usize = 16 + 16 + 14 + 14; // 60
        if export.len() < EXPECTED {
            return Err(VideoIpError::InvalidPacket(format!(
                "DTLS key material export too short: {} bytes (need {})",
                export.len(),
                EXPECTED
            )));
        }
        let client_key: [u8; 16] = export[0..16].try_into().map_err(|_| {
            VideoIpError::InvalidPacket("client key slice conversion failed".to_owned())
        })?;
        let server_key: [u8; 16] = export[16..32].try_into().map_err(|_| {
            VideoIpError::InvalidPacket("server key slice conversion failed".to_owned())
        })?;
        let client_salt: [u8; 14] = export[32..46].try_into().map_err(|_| {
            VideoIpError::InvalidPacket("client salt slice conversion failed".to_owned())
        })?;
        let server_salt: [u8; 14] = export[46..60].try_into().map_err(|_| {
            VideoIpError::InvalidPacket("server salt slice conversion failed".to_owned())
        })?;

        if is_client {
            Ok(Self::new(client_key, client_salt, server_key, server_salt))
        } else {
            Ok(Self::new(server_key, server_salt, client_key, client_salt))
        }
    }
}

// ── DTLS-SRTP Config ─────────────────────────────────────────────────────────

/// Configuration for a DTLS-SRTP session.
#[derive(Debug, Clone)]
pub struct DtlsSrtpConfig {
    /// Negotiated SRTP crypto profile.
    pub profile: SrtpProfile,
    /// Replay-protection window size (number of sequence numbers tracked).
    /// Typical values: 64 or 128.
    pub srtp_window_size: u32,
}

impl DtlsSrtpConfig {
    /// Create a new config with the given profile and replay window.
    #[must_use]
    pub const fn new(profile: SrtpProfile, srtp_window_size: u32) -> Self {
        Self {
            profile,
            srtp_window_size,
        }
    }
}

impl Default for DtlsSrtpConfig {
    fn default() -> Self {
        Self {
            profile: SrtpProfile::AesCm128HmacSha1_80,
            srtp_window_size: 128,
        }
    }
}

// ── Keystream ─────────────────────────────────────────────────────────────────

/// Derive a pseudo-keystream byte at position `i` for the given master key,
/// master salt, SSRC, and packet index (sequence-number-derived).
///
/// Mimics AES-128-CTR key derivation without requiring AES:
/// ```text
/// ks[i] = key[i % 16] ^ salt[i % 14] ^ ssrc_bytes[i % 4] ^ index_bytes[i % 8]
/// ```
#[inline]
fn keystream_byte(key: &[u8; 16], salt: &[u8; 14], ssrc: u32, pkt_index: u64, i: usize) -> u8 {
    let ssrc_b = ssrc.to_be_bytes();
    let idx_b = pkt_index.to_be_bytes();
    key[i % 16] ^ salt[i % 14] ^ ssrc_b[i % 4] ^ idx_b[i % 8]
}

/// Compute an auth tag over the ciphertext using a simplified HMAC-like scheme.
///
/// In production this would be HMAC-SHA1 truncated to `tag_len` bytes; here
/// we use a rolling XOR accumulator seeded from the key and salt so the tag
/// is deterministic and invertible for testing.
fn compute_auth_tag(
    key: &[u8; 16],
    salt: &[u8; 14],
    ciphertext: &[u8],
    ssrc: u32,
    seq: u16,
    tag_len: usize,
) -> Vec<u8> {
    let mut tag = vec![0u8; tag_len];
    let seq_b = seq.to_be_bytes();
    let ssrc_b = ssrc.to_be_bytes();

    for (t_idx, t_byte) in tag.iter_mut().enumerate() {
        // Seed from key + salt
        let base = key[t_idx % 16] ^ salt[t_idx % 14];
        // Mix in SSRC and sequence number
        let mix = ssrc_b[t_idx % 4] ^ seq_b[t_idx % 2];
        // Fold ciphertext bytes into the tag
        let mut acc: u8 = base ^ mix ^ (t_idx as u8);
        for (c_idx, &c_byte) in ciphertext.iter().enumerate() {
            acc = acc
                .wrapping_add(c_byte)
                .wrapping_add((c_idx as u8).wrapping_mul(t_idx as u8 | 1));
        }
        *t_byte = acc;
    }
    tag
}

// ── SRTP Context ─────────────────────────────────────────────────────────────

/// A fully-initialized SRTP context derived from a DTLS handshake.
///
/// Holds the send and receive key material and SSRC→packet-index maps for
/// replay protection and sequence-number rollover (ROC) tracking.
pub struct SrtpContext {
    /// Session configuration (profile, window size).
    pub config: DtlsSrtpConfig,
    /// Keying material (local = send, remote = recv).
    pub key_material: SrtpKeyMaterial,
    /// Outbound SSRC → last-sent packet index (for ROC tracking).
    pub send_ssrc_map: HashMap<u32, u32>,
    /// Inbound SSRC → last-received packet index (for replay window).
    pub recv_ssrc_map: HashMap<u32, u32>,
}

impl SrtpContext {
    /// Create a new SRTP context from config and keying material.
    #[must_use]
    pub fn new(config: DtlsSrtpConfig, key_material: SrtpKeyMaterial) -> Self {
        Self {
            config,
            key_material,
            send_ssrc_map: HashMap::new(),
            recv_ssrc_map: HashMap::new(),
        }
    }

    // ── protect_rtp ──────────────────────────────────────────────────────────

    /// Protect (encrypt + authenticate) an RTP packet.
    ///
    /// The RTP header (bytes 0..12) is preserved in the clear.  The payload
    /// (bytes 12..) is XOR-encrypted using the local key material and the
    /// packet index derived from the RTP sequence number.  An authentication
    /// tag is appended.
    ///
    /// Wire layout of the protected packet:
    /// ```text
    /// ┌───────────────┬────────────────────┬─────────────┐
    /// │  RTP header   │  encrypted payload │   auth tag  │
    /// │   (12 bytes)  │  (variable)        │  (tag_len)  │
    /// └───────────────┴────────────────────┴─────────────┘
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `VideoIpError::InvalidPacket` if `packet` is shorter than
    /// `RTP_HEADER_MIN_LEN` (12 bytes).
    pub fn protect_rtp(&mut self, packet: &[u8]) -> Result<Vec<u8>, VideoIpError> {
        if packet.len() < RTP_HEADER_MIN_LEN {
            return Err(VideoIpError::InvalidPacket(format!(
                "RTP packet too short for protect: {} bytes (min {})",
                packet.len(),
                RTP_HEADER_MIN_LEN,
            )));
        }

        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        let ssrc = u32::from_be_bytes([packet[8], packet[9], packet[10], packet[11]]);

        // Track and bump packet index for this SSRC.
        let pkt_idx = self.send_ssrc_map.entry(ssrc).or_insert(0);
        *pkt_idx = pkt_idx.wrapping_add(1);
        let current_idx = u64::from(*pkt_idx);

        let header = &packet[..RTP_HEADER_MIN_LEN];
        let payload = &packet[RTP_HEADER_MIN_LEN..];

        // Encrypt payload.
        let mut encrypted = Vec::with_capacity(payload.len());
        for (i, &b) in payload.iter().enumerate() {
            let ks = keystream_byte(
                &self.key_material.local_key,
                &self.key_material.local_salt,
                ssrc,
                current_idx,
                i,
            );
            encrypted.push(b ^ ks);
        }

        // Compute auth tag over: header || encrypted_payload.
        let tag_len = self.config.profile.auth_tag_len();
        let mut auth_input = Vec::with_capacity(header.len() + encrypted.len());
        auth_input.extend_from_slice(header);
        auth_input.extend_from_slice(&encrypted);
        let auth_tag = compute_auth_tag(
            &self.key_material.local_key,
            &self.key_material.local_salt,
            &auth_input,
            ssrc,
            seq,
            tag_len,
        );

        // Assemble output: header || encrypted_payload || auth_tag
        let mut out = Vec::with_capacity(packet.len() + tag_len);
        out.extend_from_slice(header);
        out.extend_from_slice(&encrypted);
        out.extend_from_slice(&auth_tag);
        Ok(out)
    }

    // ── unprotect_rtp ─────────────────────────────────────────────────────────

    /// Unprotect (authenticate + decrypt) a protected RTP packet.
    ///
    /// Strips the authentication tag, verifies it (simplified: re-derive and
    /// compare), then decrypts the payload.
    ///
    /// # Errors
    ///
    /// - `VideoIpError::InvalidPacket` if the packet is too short.
    /// - `VideoIpError::InvalidState` if the authentication tag does not match.
    pub fn unprotect_rtp(&mut self, packet: &[u8]) -> Result<Vec<u8>, VideoIpError> {
        let tag_len = self.config.profile.auth_tag_len();
        let min_len = RTP_HEADER_MIN_LEN + tag_len;
        if packet.len() < min_len {
            return Err(VideoIpError::InvalidPacket(format!(
                "protected RTP packet too short for unprotect: {} bytes (min {})",
                packet.len(),
                min_len,
            )));
        }

        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        let ssrc = u32::from_be_bytes([packet[8], packet[9], packet[10], packet[11]]);

        // Track packet index for replay-window management.
        let pkt_idx = self.recv_ssrc_map.entry(ssrc).or_insert(0);
        *pkt_idx = pkt_idx.wrapping_add(1);
        let current_idx = u64::from(*pkt_idx);

        let ciphertext_end = packet.len() - tag_len;
        let header = &packet[..RTP_HEADER_MIN_LEN];
        let ciphertext = &packet[RTP_HEADER_MIN_LEN..ciphertext_end];
        let received_tag = &packet[ciphertext_end..];

        // Verify auth tag.
        let mut auth_input = Vec::with_capacity(header.len() + ciphertext.len());
        auth_input.extend_from_slice(header);
        auth_input.extend_from_slice(ciphertext);
        let expected_tag = compute_auth_tag(
            &self.key_material.remote_key,
            &self.key_material.remote_salt,
            &auth_input,
            ssrc,
            seq,
            tag_len,
        );

        if received_tag != expected_tag.as_slice() {
            return Err(VideoIpError::InvalidState(
                "SRTP authentication tag verification failed".to_owned(),
            ));
        }

        // Decrypt payload.
        let mut plaintext = Vec::with_capacity(ciphertext.len());
        for (i, &b) in ciphertext.iter().enumerate() {
            let ks = keystream_byte(
                &self.key_material.remote_key,
                &self.key_material.remote_salt,
                ssrc,
                current_idx,
                i,
            );
            plaintext.push(b ^ ks);
        }

        // Assemble output: header || plaintext_payload
        let mut out = Vec::with_capacity(RTP_HEADER_MIN_LEN + plaintext.len());
        out.extend_from_slice(header);
        out.extend_from_slice(&plaintext);
        Ok(out)
    }

    /// Return the current send packet index for an SSRC (0 if not yet seen).
    #[must_use]
    pub fn send_packet_index(&self, ssrc: u32) -> u32 {
        self.send_ssrc_map.get(&ssrc).copied().unwrap_or(0)
    }

    /// Return the current receive packet index for an SSRC (0 if not yet seen).
    #[must_use]
    pub fn recv_packet_index(&self, ssrc: u32) -> u32 {
        self.recv_ssrc_map.get(&ssrc).copied().unwrap_or(0)
    }
}

// ── Helper: build minimal RTP header ──────────────────────────────────────────

/// Construct a minimal 12-byte RTP header for testing.
///
/// Layout per RFC 3550 §5.1:
/// ```text
/// Byte 0: V=2, P=0, X=0, CC=0 → 0x80
/// Byte 1: M=0, PT=payload_type (7 bits)
/// Bytes 2–3: sequence number (big-endian)
/// Bytes 4–7: timestamp (big-endian)
/// Bytes 8–11: SSRC (big-endian)
/// ```
#[must_use]
pub fn build_rtp_header(payload_type: u8, seq: u16, timestamp: u32, ssrc: u32) -> [u8; 12] {
    let mut hdr = [0u8; 12];
    hdr[0] = 0x80; // V=2, no padding, no extension, CC=0
    hdr[1] = payload_type & 0x7F;
    let seq_b = seq.to_be_bytes();
    hdr[2] = seq_b[0];
    hdr[3] = seq_b[1];
    let ts_b = timestamp.to_be_bytes();
    hdr[4] = ts_b[0];
    hdr[5] = ts_b[1];
    hdr[6] = ts_b[2];
    hdr[7] = ts_b[3];
    let ssrc_b = ssrc.to_be_bytes();
    hdr[8] = ssrc_b[0];
    hdr[9] = ssrc_b[1];
    hdr[10] = ssrc_b[2];
    hdr[11] = ssrc_b[3];
    hdr
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ctx() -> SrtpContext {
        let config = DtlsSrtpConfig::default();
        let km = SrtpKeyMaterial::zeroed();
        SrtpContext::new(config, km)
    }

    fn rtp_packet(seq: u16, ssrc: u32, payload: &[u8]) -> Vec<u8> {
        let hdr = build_rtp_header(96, seq, 0, ssrc);
        let mut pkt = hdr.to_vec();
        pkt.extend_from_slice(payload);
        pkt
    }

    // ── SrtpProfile ───────────────────────────────────────────────────────────

    #[test]
    fn test_profile_auth_tag_len() {
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.auth_tag_len(), 10);
        assert_eq!(SrtpProfile::AesCm128HmacSha1_32.auth_tag_len(), 4);
        assert_eq!(SrtpProfile::AesGcm128Sha256.auth_tag_len(), 16);
    }

    #[test]
    fn test_profile_key_len() {
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.key_len(), 16);
        assert_eq!(SrtpProfile::AesGcm128Sha256.key_len(), 16);
    }

    #[test]
    fn test_profile_salt_len() {
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.salt_len(), 14);
        assert_eq!(SrtpProfile::AesCm128HmacSha1_32.salt_len(), 14);
        assert_eq!(SrtpProfile::AesGcm128Sha256.salt_len(), 12);
    }

    #[test]
    fn test_profile_min_protected_len() {
        // 12 header + 10 tag
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.min_protected_len(), 22);
        // 12 header + 16 tag
        assert_eq!(SrtpProfile::AesGcm128Sha256.min_protected_len(), 28);
    }

    // ── SrtpKeyMaterial ───────────────────────────────────────────────────────

    #[test]
    fn test_key_material_zeroed() {
        let km = SrtpKeyMaterial::zeroed();
        assert_eq!(km.local_key, [0u8; 16]);
        assert_eq!(km.local_salt, [0u8; 14]);
        assert_eq!(km.remote_key, [0u8; 16]);
        assert_eq!(km.remote_salt, [0u8; 14]);
    }

    #[test]
    fn test_key_material_from_dtls_export_too_short() {
        let short = vec![0u8; 30];
        let result = SrtpKeyMaterial::from_dtls_export(&short, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_key_material_from_dtls_export_client() {
        let mut export = vec![0u8; 60];
        // Distinct marker bytes so we can verify assignment.
        export[0] = 0xCA; // client_key[0]
        export[16] = 0x5E; // server_key[0]
        export[32] = 0xAA; // client_salt[0]
        export[46] = 0xBB; // server_salt[0]

        let km = SrtpKeyMaterial::from_dtls_export(&export, true).expect("should parse correctly");
        assert_eq!(km.local_key[0], 0xCA);
        assert_eq!(km.remote_key[0], 0x5E);
        assert_eq!(km.local_salt[0], 0xAA);
        assert_eq!(km.remote_salt[0], 0xBB);
    }

    #[test]
    fn test_key_material_from_dtls_export_server() {
        let mut export = vec![0u8; 60];
        export[0] = 0xCA; // client_key[0]
        export[16] = 0x5E; // server_key[0]
        export[32] = 0xAA; // client_salt[0]
        export[46] = 0xBB; // server_salt[0]

        let km = SrtpKeyMaterial::from_dtls_export(&export, false).expect("should parse correctly");
        // Server perspective: local = server, remote = client.
        assert_eq!(km.local_key[0], 0x5E);
        assert_eq!(km.remote_key[0], 0xCA);
    }

    // ── DtlsSrtpConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_dtls_srtp_config_default() {
        let cfg = DtlsSrtpConfig::default();
        assert_eq!(cfg.profile, SrtpProfile::AesCm128HmacSha1_80);
        assert_eq!(cfg.srtp_window_size, 128);
    }

    // ── SrtpContext protect / unprotect ───────────────────────────────────────

    #[test]
    fn test_protect_rtp_too_short() {
        let mut ctx = default_ctx();
        let too_short = [0u8; 8];
        let result = ctx.protect_rtp(&too_short);
        assert!(result.is_err());
    }

    #[test]
    fn test_unprotect_rtp_too_short() {
        let mut ctx = default_ctx();
        let too_short = [0u8; 15]; // 12 header + 3 — but tag = 10
        let result = ctx.unprotect_rtp(&too_short);
        assert!(result.is_err());
    }

    #[test]
    fn test_protect_appends_auth_tag() {
        let mut ctx = default_ctx();
        let pkt = rtp_packet(1, 0xDEAD_BEEF, b"Hello");
        let protected = ctx.protect_rtp(&pkt).expect("protect should succeed");
        // protected = 12 (hdr) + 5 (payload) + 10 (tag) = 27
        assert_eq!(protected.len(), pkt.len() + 10);
    }

    #[test]
    fn test_protect_preserves_rtp_header() {
        let mut ctx = default_ctx();
        let pkt = rtp_packet(42, 0x1234_5678, b"Test payload");
        let protected = ctx.protect_rtp(&pkt).expect("protect should succeed");
        // RTP header must be identical in the protected packet.
        assert_eq!(&protected[..12], &pkt[..12]);
    }

    #[test]
    fn test_protect_unprotect_roundtrip_sha1_80() {
        let km = SrtpKeyMaterial::zeroed();
        let config = DtlsSrtpConfig::new(SrtpProfile::AesCm128HmacSha1_80, 128);
        // For a symmetric loopback test: local and remote keys must match.
        let mut send_ctx = SrtpContext::new(config.clone(), km.clone());
        // Swap local/remote for the receiver.
        let recv_km =
            SrtpKeyMaterial::new(km.remote_key, km.remote_salt, km.local_key, km.local_salt);
        let mut recv_ctx = SrtpContext::new(config, recv_km);

        let original = rtp_packet(100, 0xABCD_1234, b"Video payload data");
        let protected = send_ctx.protect_rtp(&original).expect("protect");
        let recovered = recv_ctx.unprotect_rtp(&protected).expect("unprotect");
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_protect_unprotect_roundtrip_gcm() {
        let km = SrtpKeyMaterial::zeroed();
        let config = DtlsSrtpConfig::new(SrtpProfile::AesGcm128Sha256, 64);
        let mut send_ctx = SrtpContext::new(config.clone(), km.clone());
        let recv_km =
            SrtpKeyMaterial::new(km.remote_key, km.remote_salt, km.local_key, km.local_salt);
        let mut recv_ctx = SrtpContext::new(config, recv_km);

        let original = rtp_packet(1, 0xCAFE_BABE, b"GCM test payload");
        let protected = send_ctx.protect_rtp(&original).expect("protect");
        assert_eq!(protected.len(), original.len() + 16); // GCM tag = 16
        let recovered = recv_ctx.unprotect_rtp(&protected).expect("unprotect");
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_protect_unprotect_sha1_32_roundtrip() {
        let km = SrtpKeyMaterial::zeroed();
        let config = DtlsSrtpConfig::new(SrtpProfile::AesCm128HmacSha1_32, 128);
        let mut send_ctx = SrtpContext::new(config.clone(), km.clone());
        let recv_km =
            SrtpKeyMaterial::new(km.remote_key, km.remote_salt, km.local_key, km.local_salt);
        let mut recv_ctx = SrtpContext::new(config, recv_km);

        let original = rtp_packet(5, 0x0000_0001, b"short payload");
        let protected = send_ctx.protect_rtp(&original).expect("protect");
        assert_eq!(protected.len(), original.len() + 4); // 4-byte tag
        let recovered = recv_ctx.unprotect_rtp(&protected).expect("unprotect");
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_ssrc_extraction() {
        let ssrc: u32 = 0xDEAD_CAFE;
        let pkt = rtp_packet(1, ssrc, b"payload");
        let extracted = u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]);
        assert_eq!(extracted, ssrc);
    }

    #[test]
    fn test_packet_index_increments() {
        let mut ctx = default_ctx();
        let ssrc = 0x1111_2222u32;
        let pkt = rtp_packet(1, ssrc, b"data");
        ctx.protect_rtp(&pkt).expect("protect 1");
        assert_eq!(ctx.send_packet_index(ssrc), 1);
        ctx.protect_rtp(&pkt).expect("protect 2");
        assert_eq!(ctx.send_packet_index(ssrc), 2);
    }

    #[test]
    fn test_tampered_packet_fails_auth() {
        let km = SrtpKeyMaterial::zeroed();
        let config = DtlsSrtpConfig::default();
        let mut send_ctx = SrtpContext::new(config.clone(), km.clone());
        let recv_km =
            SrtpKeyMaterial::new(km.remote_key, km.remote_salt, km.local_key, km.local_salt);
        let mut recv_ctx = SrtpContext::new(config, recv_km);

        let original = rtp_packet(1, 0xAAAA_BBBB, b"important data");
        let mut protected = send_ctx.protect_rtp(&original).expect("protect");
        // Tamper with a payload byte.
        if protected.len() > 13 {
            protected[13] ^= 0xFF;
        }
        let result = recv_ctx.unprotect_rtp(&protected);
        assert!(result.is_err(), "tampered packet should fail auth");
    }

    #[test]
    fn test_build_rtp_header_fields() {
        let hdr = build_rtp_header(96, 0x0102, 0xDEAD_BEEF, 0xCAFE_BABE);
        assert_eq!(hdr[0], 0x80); // V=2
        assert_eq!(hdr[1], 96); // PT
        assert_eq!(u16::from_be_bytes([hdr[2], hdr[3]]), 0x0102);
        assert_eq!(
            u32::from_be_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]),
            0xCAFE_BABE
        );
    }
}
