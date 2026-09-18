//! SRTP (Secure Real-time Transport Protocol) packet protection.
//!
//! This module provides a simulated SRTP/SRTCP implementation for WebRTC.
//! In production, real AES-CM or AES-GCM encryption would be used.
//! Here we simulate with an XOR-based scheme and a checksum auth tag.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

/// SRTP protection profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtpProfile {
    /// AES-128-CM with HMAC-SHA1-80.
    AesCm128HmacSha1_80,
    /// AES-128-CM with HMAC-SHA1-32.
    AesCm128HmacSha1_32,
    /// AEAD AES-128-GCM.
    AeadAes128Gcm,
    /// AEAD AES-256-GCM.
    AeadAes256Gcm,
}

impl SrtpProfile {
    /// Returns the master key length in bytes.
    #[must_use]
    pub const fn key_length(self) -> usize {
        match self {
            Self::AesCm128HmacSha1_80 | Self::AesCm128HmacSha1_32 | Self::AeadAes128Gcm => 16,
            Self::AeadAes256Gcm => 32,
        }
    }

    /// Returns the master salt length in bytes.
    #[must_use]
    pub const fn salt_length(self) -> usize {
        match self {
            Self::AesCm128HmacSha1_80 | Self::AesCm128HmacSha1_32 => 14,
            Self::AeadAes128Gcm | Self::AeadAes256Gcm => 12,
        }
    }

    /// Returns the authentication tag length appended per-packet.
    #[must_use]
    pub const fn tag_length(self) -> usize {
        match self {
            Self::AesCm128HmacSha1_80 => 10,
            Self::AesCm128HmacSha1_32 => 4,
            Self::AeadAes128Gcm | Self::AeadAes256Gcm => 16,
        }
    }
}

/// SRTP master key + salt.
#[derive(Debug, Clone)]
pub struct SrtpKey {
    /// Master key bytes.
    pub key: Vec<u8>,
    /// Master salt bytes.
    pub salt: Vec<u8>,
}

impl SrtpKey {
    /// Creates a new `SrtpKey`.
    #[must_use]
    pub fn new(key: Vec<u8>, salt: Vec<u8>) -> Self {
        Self { key, salt }
    }

    /// Creates a zeroed key suitable for testing.
    #[must_use]
    pub fn zeroed(profile: SrtpProfile) -> Self {
        Self {
            key: vec![0u8; profile.key_length()],
            salt: vec![0u8; profile.salt_length()],
        }
    }
}

/// SRTP context for protecting/unprotecting RTP packets.
#[derive(Debug)]
pub struct SrtpContext {
    /// Protection profile.
    pub profile: SrtpProfile,
    /// Master key material.
    pub master_key: SrtpKey,
    /// Packet index (sequence counter).
    pub index: u32,
    /// Rollover counter (ROC).
    pub rollover_counter: u32,
}

impl SrtpContext {
    /// Creates a new SRTP context.
    #[must_use]
    pub fn new(profile: SrtpProfile, master_key: SrtpKey) -> Self {
        Self {
            profile,
            master_key,
            index: 0,
            rollover_counter: 0,
        }
    }

    /// Protects an RTP packet by appending a simulated HMAC-SHA1 auth tag.
    ///
    /// Simulation: the tag is `tag_length` bytes where the first byte is an
    /// XOR checksum of the packet and the rest are zero-padded.
    #[must_use]
    pub fn protect_rtp(&mut self, packet: &[u8]) -> Vec<u8> {
        let tag_len = self.profile.tag_length();
        let mut out = packet.to_vec();

        // Simulated auth tag: XOR checksum of all input bytes in first byte.
        let checksum: u8 = packet.iter().fold(0u8, |acc, &b| acc ^ b);
        let mut tag = vec![0u8; tag_len];
        if tag_len > 0 {
            tag[0] = checksum;
        }
        out.extend_from_slice(&tag);
        self.index = self.index.wrapping_add(1);
        out
    }

    /// Unprotects an RTP packet by verifying and stripping the auth tag.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the packet is too short or the auth tag is invalid.
    pub fn unprotect_rtp<'a>(&self, packet: &'a [u8]) -> Result<Vec<u8>, &'static str> {
        let tag_len = self.profile.tag_length();
        if packet.len() < tag_len {
            return Err("packet too short to contain auth tag");
        }

        let (payload, tag) = packet.split_at(packet.len() - tag_len);
        let checksum: u8 = payload.iter().fold(0u8, |acc, &b| acc ^ b);

        // Verify the simulated tag (first byte must match checksum).
        if tag_len > 0 && tag[0] != checksum {
            return Err("SRTP auth tag mismatch");
        }

        Ok(payload.to_vec())
    }
}

/// SRTCP context for protecting/unprotecting RTCP packets.
#[derive(Debug)]
pub struct SrtcpContext {
    /// Protection profile.
    pub profile: SrtpProfile,
    /// Master key material.
    pub master_key: SrtpKey,
    /// SRTCP packet index.
    pub srtcp_index: u32,
}

impl SrtcpContext {
    /// Creates a new SRTCP context.
    #[must_use]
    pub fn new(profile: SrtpProfile, master_key: SrtpKey) -> Self {
        Self {
            profile,
            master_key,
            srtcp_index: 0,
        }
    }

    /// Protects an RTCP packet.
    ///
    /// Appends the SRTCP index (4 bytes, big-endian, E-bit set) and a
    /// simulated auth tag.
    #[must_use]
    pub fn protect_rtcp(&mut self, packet: &[u8]) -> Vec<u8> {
        let tag_len = self.profile.tag_length();

        let mut out = packet.to_vec();

        // SRTCP index: E-bit (bit 31) set to 1 plus the 31-bit index.
        let srtcp_index_field = 0x8000_0000u32 | (self.srtcp_index & 0x7FFF_FFFF);
        out.extend_from_slice(&srtcp_index_field.to_be_bytes());

        // Simulated auth tag.
        let checksum: u8 = out.iter().fold(0u8, |acc, &b| acc ^ b);
        let mut tag = vec![0u8; tag_len];
        if tag_len > 0 {
            tag[0] = checksum;
        }
        out.extend_from_slice(&tag);

        self.srtcp_index = self.srtcp_index.wrapping_add(1);
        out
    }

    /// Unprotects an RTCP packet by verifying and stripping the index + auth tag.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the packet is too short or the auth tag is invalid.
    pub fn unprotect_rtcp(&self, packet: &[u8]) -> Result<Vec<u8>, &'static str> {
        let tag_len = self.profile.tag_length();
        // Must have at least 4 bytes for SRTCP index + tag_len bytes.
        let overhead = 4 + tag_len;
        if packet.len() < overhead {
            return Err("RTCP packet too short");
        }

        // Strip tag first.
        let without_tag = &packet[..packet.len() - tag_len];
        let tag = &packet[packet.len() - tag_len..];

        let checksum: u8 = without_tag.iter().fold(0u8, |acc, &b| acc ^ b);
        if tag_len > 0 && tag[0] != checksum {
            return Err("SRTCP auth tag mismatch");
        }

        // Strip SRTCP index (4 bytes before tag).
        let payload = &without_tag[..without_tag.len() - 4];
        Ok(payload.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_key_length() {
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.key_length(), 16);
        assert_eq!(SrtpProfile::AeadAes256Gcm.key_length(), 32);
    }

    #[test]
    fn test_profile_salt_length() {
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.salt_length(), 14);
        assert_eq!(SrtpProfile::AeadAes128Gcm.salt_length(), 12);
    }

    #[test]
    fn test_profile_tag_length() {
        assert_eq!(SrtpProfile::AesCm128HmacSha1_80.tag_length(), 10);
        assert_eq!(SrtpProfile::AesCm128HmacSha1_32.tag_length(), 4);
        assert_eq!(SrtpProfile::AeadAes128Gcm.tag_length(), 16);
        assert_eq!(SrtpProfile::AeadAes256Gcm.tag_length(), 16);
    }

    #[test]
    fn test_srtp_key_zeroed() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        assert_eq!(key.key.len(), 16);
        assert_eq!(key.salt.len(), 14);
    }

    #[test]
    fn test_srtp_protect_rtp() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        let packet = vec![0x80u8, 0x60, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0xAA, 0xBB];
        let protected = ctx.protect_rtp(&packet);

        // Protected packet should be packet + 10-byte tag.
        assert_eq!(protected.len(), packet.len() + 10);
    }

    #[test]
    fn test_srtp_protect_unprotect_roundtrip() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key.clone());
        let ctx2 = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        let packet = b"Hello SRTP packet!";
        let protected = ctx.protect_rtp(packet);
        let recovered = ctx2
            .unprotect_rtp(&protected)
            .expect("should succeed in test");

        assert_eq!(recovered, packet.to_vec());
    }

    #[test]
    fn test_srtp_unprotect_tampered() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key.clone());
        let ctx2 = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        let packet = b"test packet data";
        let mut protected = ctx.protect_rtp(packet);

        // Tamper with the auth tag.
        let len = protected.len();
        protected[len - 1] ^= 0xFF;

        // Should still parse (we only check byte[0] of the tag).
        // Tamper the first byte of the tag instead.
        let tag_start = len - 10;
        protected[tag_start] ^= 0xFF;

        let result = ctx2.unprotect_rtp(&protected);
        assert!(result.is_err());
    }

    #[test]
    fn test_srtcp_protect_unprotect_roundtrip() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtcpContext::new(SrtpProfile::AesCm128HmacSha1_80, key.clone());
        let ctx2 = SrtcpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        let rtcp = b"RTCP sender report payload";
        let protected = ctx.protect_rtcp(rtcp);
        let recovered = ctx2
            .unprotect_rtcp(&protected)
            .expect("should succeed in test");

        assert_eq!(recovered, rtcp.to_vec());
    }

    #[test]
    fn test_srtcp_packet_overhead() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtcpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        let rtcp = b"RTCP";
        let protected = ctx.protect_rtcp(rtcp);

        // Should add 4 bytes (SRTCP index) + 10 bytes (tag).
        assert_eq!(protected.len(), rtcp.len() + 4 + 10);
    }

    #[test]
    fn test_srtp_index_increments() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        assert_eq!(ctx.index, 0);
        let _ = ctx.protect_rtp(b"p1");
        assert_eq!(ctx.index, 1);
        let _ = ctx.protect_rtp(b"p2");
        assert_eq!(ctx.index, 2);
    }

    #[test]
    fn test_srtcp_index_increments() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let mut ctx = SrtcpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        assert_eq!(ctx.srtcp_index, 0);
        let _ = ctx.protect_rtcp(b"r1");
        assert_eq!(ctx.srtcp_index, 1);
    }

    #[test]
    fn test_srtp_aead_gcm_profiles() {
        let key = SrtpKey::zeroed(SrtpProfile::AeadAes128Gcm);
        let mut ctx = SrtpContext::new(SrtpProfile::AeadAes128Gcm, key.clone());
        let ctx2 = SrtpContext::new(SrtpProfile::AeadAes128Gcm, key);

        let packet = b"GCM test packet";
        let protected = ctx.protect_rtp(packet);
        assert_eq!(protected.len(), packet.len() + 16);

        let recovered = ctx2
            .unprotect_rtp(&protected)
            .expect("should succeed in test");
        assert_eq!(recovered, packet.to_vec());
    }

    #[test]
    fn test_srtp_unprotect_too_short() {
        let key = SrtpKey::zeroed(SrtpProfile::AesCm128HmacSha1_80);
        let ctx = SrtpContext::new(SrtpProfile::AesCm128HmacSha1_80, key);

        // 5 bytes < 10-byte tag.
        let result = ctx.unprotect_rtp(b"short");
        assert!(result.is_err());
    }
}
