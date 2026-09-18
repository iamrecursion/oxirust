//! Forward Error Correction (FEC) for RTP streams.
//!
//! This module provides XOR-based FEC (RFC 5109 style) for protecting RTP
//! media streams against packet loss.  For every group of `k` source packets
//! the encoder produces one (1-D) or more (2-D interleaved) repair packets.
//! The decoder can recover exactly one lost source packet per group.
//!
//! # Example — 1-D FEC
//!
//! ```rust
//! use oximedia_net::fec::{FecConfig, FecEncoder, FecDecoder};
//!
//! let cfg = FecConfig::one_dimensional(4);
//! let mut encoder = FecEncoder::new(cfg.clone());
//! let mut decoder = FecDecoder::new(cfg);
//!
//! let payloads: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 12]).collect();
//! let mut fec_pkt = None;
//! for (seq, payload) in payloads.iter().enumerate() {
//!     let result = encoder.feed_packet(seq as u16, payload);
//!     if result.is_some() {
//!         fec_pkt = result;
//!     }
//! }
//!
//! // "Lose" packet 2, feed all others + FEC to decoder.
//! for (seq, payload) in payloads.iter().enumerate() {
//!     if seq != 2 {
//!         decoder.feed_source(seq as u16, payload.clone());
//!     }
//! }
//! decoder.feed_fec(fec_pkt.expect("FEC packet produced"));
//! decoder.register_group_base(0);
//!
//! let recovered = decoder.try_recover();
//! assert_eq!(recovered.len(), 1);
//! assert_eq!(recovered[0].0, 2);
//! ```

pub mod xor_fec;
pub use xor_fec::{FecConfig, FecDecoder, FecEncoder, FecPacket};
