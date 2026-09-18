//! Variable-length integer encoding and decoding
//!
//! This module implements bincode-compatible varint encoding:
//! - Values 0-250: encoded as a single byte
//! - Larger values: encoded with a tag byte followed by the value
//!   - 251 (U16_BYTE): 2-byte value follows
//!   - 252 (U32_BYTE): 4-byte value follows
//!   - 253 (U64_BYTE): 8-byte value follows
//!   - 254 (U128_BYTE): 16-byte value follows

mod decode_signed;
mod decode_unsigned;
mod encode_signed;
mod encode_unsigned;

pub use self::{
    decode_signed::{
        varint_decode_i128, varint_decode_i16, varint_decode_i32, varint_decode_i64,
        varint_decode_isize,
    },
    decode_unsigned::{
        varint_decode_u128, varint_decode_u16, varint_decode_u32, varint_decode_u64,
        varint_decode_usize,
    },
    encode_signed::{
        varint_encode_i128, varint_encode_i16, varint_encode_i32, varint_encode_i64,
        varint_encode_isize,
    },
    encode_unsigned::{
        varint_encode_u128, varint_encode_u16, varint_encode_u32, varint_encode_u64,
        varint_encode_usize,
    },
};

/// Maximum value that can be encoded in a single byte
pub(crate) const SINGLE_BYTE_MAX: u8 = 250;

/// Tag byte indicating a u16 follows
pub(crate) const U16_BYTE: u8 = 251;

/// Tag byte indicating a u32 follows
pub(crate) const U32_BYTE: u8 = 252;

/// Tag byte indicating a u64 follows
pub(crate) const U64_BYTE: u8 = 253;

/// Tag byte indicating a u128 follows
pub(crate) const U128_BYTE: u8 = 254;
