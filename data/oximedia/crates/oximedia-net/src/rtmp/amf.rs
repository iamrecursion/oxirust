//! AMF0 (Action Message Format) encoding/decoding.
//!
//! AMF0 is used for serializing ActionScript objects in RTMP.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::collections::HashMap;

/// AMF0 type markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AmfMarker {
    /// Number (64-bit double).
    Number = 0x00,
    /// Boolean.
    Boolean = 0x01,
    /// String (16-bit length).
    String = 0x02,
    /// Object.
    Object = 0x03,
    /// MovieClip (reserved).
    MovieClip = 0x04,
    /// Null.
    Null = 0x05,
    /// Undefined.
    Undefined = 0x06,
    /// Reference.
    Reference = 0x07,
    /// ECMA Array.
    EcmaArray = 0x08,
    /// Object End marker.
    ObjectEnd = 0x09,
    /// Strict Array.
    StrictArray = 0x0A,
    /// Date.
    Date = 0x0B,
    /// Long String (32-bit length).
    LongString = 0x0C,
    /// Unsupported type.
    Unsupported = 0x0D,
    /// RecordSet (reserved).
    RecordSet = 0x0E,
    /// XML Document.
    XmlDocument = 0x0F,
    /// Typed Object.
    TypedObject = 0x10,
    /// Switch to AMF3.
    AvmPlus = 0x11,
}

impl AmfMarker {
    /// Creates from byte value.
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Number),
            0x01 => Some(Self::Boolean),
            0x02 => Some(Self::String),
            0x03 => Some(Self::Object),
            0x04 => Some(Self::MovieClip),
            0x05 => Some(Self::Null),
            0x06 => Some(Self::Undefined),
            0x07 => Some(Self::Reference),
            0x08 => Some(Self::EcmaArray),
            0x09 => Some(Self::ObjectEnd),
            0x0A => Some(Self::StrictArray),
            0x0B => Some(Self::Date),
            0x0C => Some(Self::LongString),
            0x0D => Some(Self::Unsupported),
            0x0E => Some(Self::RecordSet),
            0x0F => Some(Self::XmlDocument),
            0x10 => Some(Self::TypedObject),
            0x11 => Some(Self::AvmPlus),
            _ => None,
        }
    }
}

/// AMF0 value type.
#[derive(Debug, Clone, PartialEq)]
pub enum AmfValue {
    /// Number (64-bit IEEE 754).
    Number(f64),
    /// Boolean.
    Boolean(bool),
    /// String.
    String(String),
    /// Object (key-value pairs).
    Object(HashMap<String, AmfValue>),
    /// Null.
    Null,
    /// Undefined.
    Undefined,
    /// ECMA Array.
    EcmaArray(HashMap<String, AmfValue>),
    /// Strict Array.
    StrictArray(Vec<AmfValue>),
    /// Date (milliseconds since epoch + timezone offset).
    Date {
        /// Milliseconds since Unix epoch.
        timestamp: f64,
        /// Timezone offset in minutes.
        timezone: i16,
    },
    /// Long String.
    LongString(String),
    /// XML Document.
    XmlDocument(String),
}

impl AmfValue {
    /// Creates a new number value.
    #[must_use]
    pub const fn number(n: f64) -> Self {
        Self::Number(n)
    }

    /// Creates a new boolean value.
    #[must_use]
    pub const fn boolean(b: bool) -> Self {
        Self::Boolean(b)
    }

    /// Creates a new string value.
    #[must_use]
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Creates a new null value.
    #[must_use]
    pub const fn null() -> Self {
        Self::Null
    }

    /// Creates a new object value.
    #[must_use]
    pub fn object(props: HashMap<String, AmfValue>) -> Self {
        Self::Object(props)
    }

    /// Creates an empty object.
    #[must_use]
    pub fn empty_object() -> Self {
        Self::Object(HashMap::new())
    }

    /// Returns true if this is a number.
    #[must_use]
    pub const fn is_number(&self) -> bool {
        matches!(self, Self::Number(_))
    }

    /// Returns true if this is a boolean.
    #[must_use]
    pub const fn is_boolean(&self) -> bool {
        matches!(self, Self::Boolean(_))
    }

    /// Returns true if this is a string.
    #[must_use]
    pub const fn is_string(&self) -> bool {
        matches!(self, Self::String(_) | Self::LongString(_))
    }

    /// Returns true if this is an object.
    #[must_use]
    pub const fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    /// Returns true if this is null.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Tries to get as number.
    #[must_use]
    pub const fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Tries to get as boolean.
    #[must_use]
    pub const fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Tries to get as string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) | Self::LongString(s) => Some(s),
            _ => None,
        }
    }

    /// Tries to get as object.
    #[must_use]
    pub const fn as_object(&self) -> Option<&HashMap<String, AmfValue>> {
        match self {
            Self::Object(o) | Self::EcmaArray(o) => Some(o),
            _ => None,
        }
    }

    /// Tries to get as array.
    #[must_use]
    pub const fn as_array(&self) -> Option<&Vec<AmfValue>> {
        match self {
            Self::StrictArray(a) => Some(a),
            _ => None,
        }
    }

    /// Gets a property from an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&AmfValue> {
        match self {
            Self::Object(o) | Self::EcmaArray(o) => o.get(key),
            _ => None,
        }
    }
}

impl Default for AmfValue {
    fn default() -> Self {
        Self::Null
    }
}

/// AMF0 encoder.
#[derive(Debug, Default)]
pub struct AmfEncoder {
    buffer: BytesMut,
}

impl AmfEncoder {
    /// Creates a new encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::new(),
        }
    }

    /// Creates an encoder with initial capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(capacity),
        }
    }

    /// Encodes a value.
    pub fn encode(&mut self, value: &AmfValue) {
        match value {
            AmfValue::Number(n) => self.encode_number(*n),
            AmfValue::Boolean(b) => self.encode_boolean(*b),
            AmfValue::String(s) => self.encode_string(s),
            AmfValue::Object(o) => self.encode_object(o),
            AmfValue::Null => self.encode_null(),
            AmfValue::Undefined => self.encode_undefined(),
            AmfValue::EcmaArray(a) => self.encode_ecma_array(a),
            AmfValue::StrictArray(a) => self.encode_strict_array(a),
            AmfValue::Date {
                timestamp,
                timezone,
            } => self.encode_date(*timestamp, *timezone),
            AmfValue::LongString(s) => self.encode_long_string(s),
            AmfValue::XmlDocument(s) => self.encode_xml_document(s),
        }
    }

    /// Encodes a number.
    pub fn encode_number(&mut self, n: f64) {
        self.buffer.put_u8(AmfMarker::Number as u8);
        self.buffer.put_f64(n);
    }

    /// Encodes a boolean.
    pub fn encode_boolean(&mut self, b: bool) {
        self.buffer.put_u8(AmfMarker::Boolean as u8);
        self.buffer.put_u8(u8::from(b));
    }

    /// Encodes a string.
    pub fn encode_string(&mut self, s: &str) {
        if s.len() > 0xFFFF {
            self.encode_long_string(s);
        } else {
            self.buffer.put_u8(AmfMarker::String as u8);
            self.write_utf8(s);
        }
    }

    /// Encodes a long string.
    pub fn encode_long_string(&mut self, s: &str) {
        self.buffer.put_u8(AmfMarker::LongString as u8);
        self.write_utf8_long(s);
    }

    /// Encodes null.
    pub fn encode_null(&mut self) {
        self.buffer.put_u8(AmfMarker::Null as u8);
    }

    /// Encodes undefined.
    pub fn encode_undefined(&mut self) {
        self.buffer.put_u8(AmfMarker::Undefined as u8);
    }

    /// Encodes an object.
    pub fn encode_object(&mut self, obj: &HashMap<String, AmfValue>) {
        self.buffer.put_u8(AmfMarker::Object as u8);
        self.write_object_properties(obj);
    }

    /// Encodes an ECMA array.
    pub fn encode_ecma_array(&mut self, arr: &HashMap<String, AmfValue>) {
        self.buffer.put_u8(AmfMarker::EcmaArray as u8);
        self.buffer.put_u32(arr.len() as u32);
        self.write_object_properties(arr);
    }

    /// Encodes a strict array.
    pub fn encode_strict_array(&mut self, arr: &[AmfValue]) {
        self.buffer.put_u8(AmfMarker::StrictArray as u8);
        self.buffer.put_u32(arr.len() as u32);
        for value in arr {
            self.encode(value);
        }
    }

    /// Encodes a date.
    pub fn encode_date(&mut self, timestamp: f64, timezone: i16) {
        self.buffer.put_u8(AmfMarker::Date as u8);
        self.buffer.put_f64(timestamp);
        self.buffer.put_i16(timezone);
    }

    /// Encodes an XML document.
    pub fn encode_xml_document(&mut self, s: &str) {
        self.buffer.put_u8(AmfMarker::XmlDocument as u8);
        self.write_utf8_long(s);
    }

    fn write_utf8(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.buffer.put_u16(bytes.len() as u16);
        self.buffer.put_slice(bytes);
    }

    fn write_utf8_long(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.buffer.put_u32(bytes.len() as u32);
        self.buffer.put_slice(bytes);
    }

    fn write_object_properties(&mut self, obj: &HashMap<String, AmfValue>) {
        for (key, value) in obj {
            self.write_utf8(key);
            self.encode(value);
        }
        // Object end marker
        self.buffer.put_u16(0); // Empty string
        self.buffer.put_u8(AmfMarker::ObjectEnd as u8);
    }

    /// Finishes encoding and returns the buffer.
    #[must_use]
    pub fn finish(self) -> Bytes {
        self.buffer.freeze()
    }

    /// Returns the current buffer contents.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Maximum AMF object/array nesting depth accepted by the decoder.
///
/// AMF0 allows objects and arrays to contain further objects and arrays, and
/// each level recurses through [`AmfDecoder::decode`]. A hostile RTMP peer can
/// send deeply nested containers to exhaust the stack; capping the depth turns
/// that into a clean error. 32 is far deeper than any real RTMP command.
const MAX_AMF_DEPTH: usize = 32;

/// AMF0 decoder.
#[derive(Debug)]
pub struct AmfDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// Current nesting depth, used to bound recursion (see `MAX_AMF_DEPTH`).
    depth: usize,
}

impl<'a> AmfDecoder<'a> {
    /// Creates a new decoder.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            depth: 0,
        }
    }

    /// Returns remaining bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Returns true if there's more data.
    #[must_use]
    pub fn has_remaining(&self) -> bool {
        self.pos < self.data.len()
    }

    /// Decodes the next value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is malformed.
    pub fn decode(&mut self) -> NetResult<AmfValue> {
        // Bound recursion: a peer nesting objects/arrays past MAX_AMF_DEPTH
        // would otherwise recurse until the thread stack overflows (abort).
        self.depth += 1;
        if self.depth > MAX_AMF_DEPTH {
            self.depth -= 1;
            return Err(NetError::encoding("AMF nesting exceeds maximum depth"));
        }
        let result = self.decode_value();
        self.depth -= 1;
        result
    }

    /// Decodes the next value without touching the recursion guard.
    fn decode_value(&mut self) -> NetResult<AmfValue> {
        let marker = self.read_u8()?;
        let marker = AmfMarker::from_byte(marker)
            .ok_or_else(|| NetError::encoding(format!("Unknown AMF marker: {marker:02x}")))?;

        match marker {
            AmfMarker::Number => self.decode_number(),
            AmfMarker::Boolean => self.decode_boolean(),
            AmfMarker::String => self.decode_string(),
            AmfMarker::Object => self.decode_object(),
            AmfMarker::Null => Ok(AmfValue::Null),
            AmfMarker::Undefined => Ok(AmfValue::Undefined),
            AmfMarker::EcmaArray => self.decode_ecma_array(),
            AmfMarker::StrictArray => self.decode_strict_array(),
            AmfMarker::Date => self.decode_date(),
            AmfMarker::LongString => self.decode_long_string(),
            AmfMarker::XmlDocument => self.decode_xml_document(),
            _ => Err(NetError::encoding(format!(
                "Unsupported AMF type: {:?}",
                marker
            ))),
        }
    }

    fn decode_number(&mut self) -> NetResult<AmfValue> {
        let n = self.read_f64()?;
        Ok(AmfValue::Number(n))
    }

    fn decode_boolean(&mut self) -> NetResult<AmfValue> {
        let b = self.read_u8()?;
        Ok(AmfValue::Boolean(b != 0))
    }

    fn decode_string(&mut self) -> NetResult<AmfValue> {
        let s = self.read_utf8()?;
        Ok(AmfValue::String(s))
    }

    fn decode_long_string(&mut self) -> NetResult<AmfValue> {
        let s = self.read_utf8_long()?;
        Ok(AmfValue::LongString(s))
    }

    fn decode_object(&mut self) -> NetResult<AmfValue> {
        let props = self.read_object_properties()?;
        Ok(AmfValue::Object(props))
    }

    fn decode_ecma_array(&mut self) -> NetResult<AmfValue> {
        let _count = self.read_u32()?; // Hint, not necessarily accurate
        let props = self.read_object_properties()?;
        Ok(AmfValue::EcmaArray(props))
    }

    fn decode_strict_array(&mut self) -> NetResult<AmfValue> {
        let count = self.read_u32()? as usize;
        // Never pre-allocate from the unchecked on-wire count: a peer can claim
        // count = u32::MAX (~4 billion) to force a multi-GB reservation before
        // a single element is read. Every element needs at least one marker
        // byte, so the reservation cannot legitimately exceed the bytes left;
        // if `count` really is larger the loop below errors on end-of-data.
        let cap = count.min(self.remaining());
        let mut arr = Vec::with_capacity(cap);
        for _ in 0..count {
            arr.push(self.decode()?);
        }
        Ok(AmfValue::StrictArray(arr))
    }

    fn decode_date(&mut self) -> NetResult<AmfValue> {
        let timestamp = self.read_f64()?;
        let timezone = self.read_i16()?;
        Ok(AmfValue::Date {
            timestamp,
            timezone,
        })
    }

    fn decode_xml_document(&mut self) -> NetResult<AmfValue> {
        let s = self.read_utf8_long()?;
        Ok(AmfValue::XmlDocument(s))
    }

    fn read_object_properties(&mut self) -> NetResult<HashMap<String, AmfValue>> {
        let mut props = HashMap::new();

        loop {
            let key = self.read_utf8()?;
            if key.is_empty() {
                // Check for object end marker
                let marker = self.read_u8()?;
                if marker == AmfMarker::ObjectEnd as u8 {
                    break;
                }
                return Err(NetError::encoding("Expected object end marker"));
            }
            let value = self.decode()?;
            props.insert(key, value);
        }

        Ok(props)
    }

    fn read_u8(&mut self) -> NetResult<u8> {
        if self.pos >= self.data.len() {
            return Err(NetError::encoding("Unexpected end of AMF data"));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u16(&mut self) -> NetResult<u16> {
        if self.pos + 2 > self.data.len() {
            return Err(NetError::encoding("Unexpected end of AMF data"));
        }
        let mut buf = &self.data[self.pos..self.pos + 2];
        self.pos += 2;
        Ok(buf.get_u16())
    }

    fn read_u32(&mut self) -> NetResult<u32> {
        if self.pos + 4 > self.data.len() {
            return Err(NetError::encoding("Unexpected end of AMF data"));
        }
        let mut buf = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(buf.get_u32())
    }

    fn read_i16(&mut self) -> NetResult<i16> {
        if self.pos + 2 > self.data.len() {
            return Err(NetError::encoding("Unexpected end of AMF data"));
        }
        let mut buf = &self.data[self.pos..self.pos + 2];
        self.pos += 2;
        Ok(buf.get_i16())
    }

    fn read_f64(&mut self) -> NetResult<f64> {
        if self.pos + 8 > self.data.len() {
            return Err(NetError::encoding("Unexpected end of AMF data"));
        }
        let mut buf = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        Ok(buf.get_f64())
    }

    fn read_utf8(&mut self) -> NetResult<String> {
        let len = self.read_u16()? as usize;
        if self.pos + len > self.data.len() {
            return Err(NetError::encoding("String length exceeds data"));
        }
        let s = String::from_utf8(self.data[self.pos..self.pos + len].to_vec())
            .map_err(|_| NetError::encoding("Invalid UTF-8 string"))?;
        self.pos += len;
        Ok(s)
    }

    fn read_utf8_long(&mut self) -> NetResult<String> {
        let len = self.read_u32()? as usize;
        if self.pos + len > self.data.len() {
            return Err(NetError::encoding("Long string length exceeds data"));
        }
        let s = String::from_utf8(self.data[self.pos..self.pos + len].to_vec())
            .map_err(|_| NetError::encoding("Invalid UTF-8 long string"))?;
        self.pos += len;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_number() {
        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::Number(42.5));
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        assert_eq!(value, AmfValue::Number(42.5));
    }

    #[test]
    fn test_encode_decode_boolean() {
        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::Boolean(true));
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        assert_eq!(value, AmfValue::Boolean(true));
    }

    #[test]
    fn test_encode_decode_string() {
        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::String("hello world".to_string()));
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        assert_eq!(value.as_str(), Some("hello world"));
    }

    #[test]
    fn test_encode_decode_null() {
        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::Null);
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        assert!(value.is_null());
    }

    #[test]
    fn test_encode_decode_object() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), AmfValue::String("test".to_string()));
        props.insert("value".to_string(), AmfValue::Number(123.0));

        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::Object(props));
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        assert!(value.is_object());

        let obj = value.as_object().expect("should succeed in test");
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("test"));
        assert_eq!(obj.get("value").and_then(|v| v.as_number()), Some(123.0));
    }

    #[test]
    fn test_encode_decode_array() {
        let arr = vec![
            AmfValue::Number(1.0),
            AmfValue::Number(2.0),
            AmfValue::String("three".to_string()),
        ];

        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::StrictArray(arr));
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        let arr = value.as_array().expect("should succeed in test");

        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_number(), Some(1.0));
        assert_eq!(arr[2].as_str(), Some("three"));
    }

    #[test]
    fn test_encode_decode_date() {
        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::Date {
            timestamp: 1_640_000_000_000.0,
            timezone: 0,
        });
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        let value = dec.decode().expect("should succeed in test");
        if let AmfValue::Date {
            timestamp,
            timezone,
        } = value
        {
            assert!((timestamp - 1_640_000_000_000.0).abs() < 0.001);
            assert_eq!(timezone, 0);
        } else {
            panic!("Expected date");
        }
    }

    #[test]
    fn test_amf_value_helpers() {
        let num = AmfValue::number(42.0);
        assert!(num.is_number());
        assert_eq!(num.as_number(), Some(42.0));

        let b = AmfValue::boolean(false);
        assert!(b.is_boolean());
        assert_eq!(b.as_boolean(), Some(false));

        let s = AmfValue::string("test");
        assert!(s.is_string());
        assert_eq!(s.as_str(), Some("test"));

        let null = AmfValue::null();
        assert!(null.is_null());
    }

    #[test]
    fn test_multiple_values() {
        let mut enc = AmfEncoder::new();
        enc.encode(&AmfValue::String("connect".to_string()));
        enc.encode(&AmfValue::Number(1.0));
        enc.encode(&AmfValue::Null);
        let data = enc.finish();

        let mut dec = AmfDecoder::new(&data);
        assert_eq!(
            dec.decode().expect("should succeed in test").as_str(),
            Some("connect")
        );
        assert_eq!(
            dec.decode().expect("should succeed in test").as_number(),
            Some(1.0)
        );
        assert!(dec.decode().expect("should succeed in test").is_null());
        assert!(!dec.has_remaining());
    }

    #[test]
    fn strict_array_huge_count_does_not_oom() {
        // StrictArray marker + count = u32::MAX with no element bytes. The
        // decoder must NOT pre-allocate ~4 billion slots; it should error out
        // reading the (absent) first element instead.
        let mut data = vec![AmfMarker::StrictArray as u8];
        data.extend_from_slice(&u32::MAX.to_be_bytes());
        let mut dec = AmfDecoder::new(&data);
        assert!(dec.decode().is_err());
    }

    #[test]
    fn deeply_nested_amf_is_rejected_not_stack_overflow() {
        // 40 nested single-element strict arrays exceed MAX_AMF_DEPTH (32).
        // Reaching this assertion at all proves the stack did not overflow.
        let mut data = Vec::new();
        for _ in 0..40 {
            data.push(AmfMarker::StrictArray as u8);
            data.extend_from_slice(&1u32.to_be_bytes());
        }
        data.push(AmfMarker::Null as u8);
        let mut dec = AmfDecoder::new(&data);
        assert!(dec.decode().is_err());
    }

    #[test]
    fn nesting_at_limit_still_decodes() {
        // A handful of nested arrays (well under the cap) must still succeed,
        // proving the guard does not reject legitimate nested structures.
        let mut data = Vec::new();
        for _ in 0..4 {
            data.push(AmfMarker::StrictArray as u8);
            data.extend_from_slice(&1u32.to_be_bytes());
        }
        data.push(AmfMarker::Number as u8);
        data.extend_from_slice(&1.5f64.to_be_bytes());
        let mut dec = AmfDecoder::new(&data);
        assert!(dec.decode().is_ok());
    }
}
