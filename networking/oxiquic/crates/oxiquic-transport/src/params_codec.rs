//! TLS wire encoding of QUIC transport parameters (RFC 9000 Section 18).
//!
//! Transport parameters are carried inside the TLS `quic_transport_parameters`
//! extension as a sequence of `(varint id, varint length, value)` triples. This
//! module lowers [`oxiquic_core::TransportParams`] to those bytes (which are
//! handed to `rustls::quic::{ClientConnection, ServerConnection}::new`) and
//! parses the peer's parameters back out. A malformed encoding fails the
//! handshake with `TRANSPORT_PARAMETER_ERROR`, so decoding is strict.

use crate::coding::{put_varint, varint_size, Buf};
use oxiquic_core::{OxiQuicError, TransportErrorCode, TransportParams};

// RFC 9000 Section 18.2 parameter identifiers.
const ID_MAX_IDLE_TIMEOUT: u64 = 0x01;
const ID_MAX_UDP_PAYLOAD_SIZE: u64 = 0x03;
const ID_INITIAL_MAX_DATA: u64 = 0x04;
const ID_INITIAL_MAX_STREAM_DATA_BIDI_LOCAL: u64 = 0x05;
const ID_INITIAL_MAX_STREAM_DATA_BIDI_REMOTE: u64 = 0x06;
const ID_INITIAL_MAX_STREAM_DATA_UNI: u64 = 0x07;
const ID_INITIAL_MAX_STREAMS_BIDI: u64 = 0x08;
const ID_INITIAL_MAX_STREAMS_UNI: u64 = 0x09;
const ID_ACK_DELAY_EXPONENT: u64 = 0x0a;
const ID_MAX_ACK_DELAY: u64 = 0x0b;
const ID_DISABLE_ACTIVE_MIGRATION: u64 = 0x0c;
const ID_ACTIVE_CONNECTION_ID_LIMIT: u64 = 0x0e;
/// RFC 9221 §3: `max_datagram_frame_size` transport parameter.
const ID_MAX_DATAGRAM_FRAME_SIZE: u64 = 0x20;
/// RFC 9000 §18.2: `original_destination_connection_id` (server-only).
const ID_ORIGINAL_DESTINATION_CONNECTION_ID: u64 = 0x00;
/// RFC 9000 §18.2: `initial_source_connection_id` (both endpoints).
const ID_INITIAL_SOURCE_CONNECTION_ID: u64 = 0x0f;
/// RFC 9000 §18.2: `retry_source_connection_id` (server-only, after a Retry).
const ID_RETRY_SOURCE_CONNECTION_ID: u64 = 0x10;

/// RFC 9000 §17.2: a connection ID is at most 20 bytes on the wire.
const MAX_CONNECTION_ID_LEN: usize = 20;

fn put_bytes_param(out: &mut Vec<u8>, id: u64, value: &[u8]) {
    put_varint(out, id);
    put_varint(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_varint_param(out: &mut Vec<u8>, id: u64, value: u64) {
    put_varint(out, id);
    put_varint(out, varint_size(value) as u64);
    put_varint(out, value);
}

/// Encode transport parameters into the TLS extension byte string
/// (RFC 9000 Section 18).
#[must_use]
pub fn encode_transport_params(params: &TransportParams) -> Vec<u8> {
    let mut out = Vec::new();
    if params.max_idle_timeout_ms != 0 {
        put_varint_param(&mut out, ID_MAX_IDLE_TIMEOUT, params.max_idle_timeout_ms);
    }
    put_varint_param(
        &mut out,
        ID_MAX_UDP_PAYLOAD_SIZE,
        params.max_udp_payload_size,
    );
    if params.initial_max_data != 0 {
        put_varint_param(&mut out, ID_INITIAL_MAX_DATA, params.initial_max_data);
    }
    if params.initial_max_stream_data_bidi_local != 0 {
        put_varint_param(
            &mut out,
            ID_INITIAL_MAX_STREAM_DATA_BIDI_LOCAL,
            params.initial_max_stream_data_bidi_local,
        );
    }
    if params.initial_max_stream_data_bidi_remote != 0 {
        put_varint_param(
            &mut out,
            ID_INITIAL_MAX_STREAM_DATA_BIDI_REMOTE,
            params.initial_max_stream_data_bidi_remote,
        );
    }
    if params.initial_max_stream_data_uni != 0 {
        put_varint_param(
            &mut out,
            ID_INITIAL_MAX_STREAM_DATA_UNI,
            params.initial_max_stream_data_uni,
        );
    }
    if params.initial_max_streams_bidi != 0 {
        put_varint_param(
            &mut out,
            ID_INITIAL_MAX_STREAMS_BIDI,
            params.initial_max_streams_bidi,
        );
    }
    if params.initial_max_streams_uni != 0 {
        put_varint_param(
            &mut out,
            ID_INITIAL_MAX_STREAMS_UNI,
            params.initial_max_streams_uni,
        );
    }
    put_varint_param(
        &mut out,
        ID_ACK_DELAY_EXPONENT,
        u64::from(params.ack_delay_exponent),
    );
    put_varint_param(&mut out, ID_MAX_ACK_DELAY, params.max_ack_delay_ms);
    put_varint_param(
        &mut out,
        ID_ACTIVE_CONNECTION_ID_LIMIT,
        params.active_connection_id_limit,
    );
    if params.disable_active_migration {
        // Zero-length value parameter (a flag).
        put_varint(&mut out, ID_DISABLE_ACTIVE_MIGRATION);
        put_varint(&mut out, 0);
    }
    if params.max_datagram_frame_size != 0 {
        put_varint_param(
            &mut out,
            ID_MAX_DATAGRAM_FRAME_SIZE,
            params.max_datagram_frame_size,
        );
    }
    // RFC 9000 §7.3 connection-ID transcript. Each is emitted only when set, so
    // a client never sends the two server-only parameters and a server only
    // sends `retry_source_connection_id` when it actually issued a Retry.
    if let Some(ref cid) = params.original_destination_connection_id {
        put_bytes_param(&mut out, ID_ORIGINAL_DESTINATION_CONNECTION_ID, cid);
    }
    if let Some(ref cid) = params.initial_source_connection_id {
        put_bytes_param(&mut out, ID_INITIAL_SOURCE_CONNECTION_ID, cid);
    }
    if let Some(ref cid) = params.retry_source_connection_id {
        put_bytes_param(&mut out, ID_RETRY_SOURCE_CONNECTION_ID, cid);
    }
    out
}

/// Validate a connection-ID-valued transport parameter (RFC 9000 §18.2: at most
/// 20 bytes, matching the wire limit on connection IDs).
fn connection_id(value: &[u8]) -> Result<Vec<u8>, OxiQuicError> {
    if value.len() > MAX_CONNECTION_ID_LEN {
        return Err(param_error(format!(
            "connection-ID transport parameter is {} bytes (max {MAX_CONNECTION_ID_LEN})",
            value.len()
        )));
    }
    Ok(value.to_vec())
}

fn param_error(reason: impl Into<String>) -> OxiQuicError {
    OxiQuicError::TransportError {
        code: TransportErrorCode::TransportParameterError,
        frame_type: None,
        reason: reason.into(),
    }
}

/// Decode the peer's transport parameters from the TLS extension byte string.
///
/// Unknown parameter IDs are ignored (RFC 9000 Section 18.1). Parameters absent
/// from the encoding retain their [`TransportParams::default`] values.
///
/// # Errors
/// Returns [`OxiQuicError::TransportError`] with `TRANSPORT_PARAMETER_ERROR`
/// for truncated input, a declared length that overruns the buffer, or a
/// scalar parameter whose value does not fill its declared length.
pub fn decode_transport_params(bytes: &[u8]) -> Result<TransportParams, OxiQuicError> {
    let mut params = TransportParams::default();
    let mut buf = Buf::new(bytes);
    while !buf.is_empty() {
        let id = buf
            .get_varint()
            .map_err(|_| param_error("truncated transport parameter id"))?;
        let len = buf
            .get_varint()
            .map_err(|_| param_error("truncated transport parameter length"))?;
        let value = buf
            .get_bytes(len as usize)
            .map_err(|_| param_error("transport parameter length overruns buffer"))?;

        let scalar = |value: &[u8]| -> Result<u64, OxiQuicError> {
            let mut inner = Buf::new(value);
            let v = inner
                .get_varint()
                .map_err(|_| param_error("transport parameter value is not a varint"))?;
            if !inner.is_empty() {
                return Err(param_error("trailing bytes in transport parameter value"));
            }
            Ok(v)
        };

        match id {
            ID_MAX_IDLE_TIMEOUT => params.max_idle_timeout_ms = scalar(value)?,
            ID_MAX_UDP_PAYLOAD_SIZE => params.max_udp_payload_size = scalar(value)?,
            ID_INITIAL_MAX_DATA => params.initial_max_data = scalar(value)?,
            ID_INITIAL_MAX_STREAM_DATA_BIDI_LOCAL => {
                params.initial_max_stream_data_bidi_local = scalar(value)?;
            }
            ID_INITIAL_MAX_STREAM_DATA_BIDI_REMOTE => {
                params.initial_max_stream_data_bidi_remote = scalar(value)?;
            }
            ID_INITIAL_MAX_STREAM_DATA_UNI => params.initial_max_stream_data_uni = scalar(value)?,
            ID_INITIAL_MAX_STREAMS_BIDI => params.initial_max_streams_bidi = scalar(value)?,
            ID_INITIAL_MAX_STREAMS_UNI => params.initial_max_streams_uni = scalar(value)?,
            ID_ACK_DELAY_EXPONENT => {
                params.ack_delay_exponent = scalar(value)?.min(u64::from(u8::MAX)) as u8;
            }
            ID_MAX_ACK_DELAY => params.max_ack_delay_ms = scalar(value)?,
            ID_ACTIVE_CONNECTION_ID_LIMIT => params.active_connection_id_limit = scalar(value)?,
            ID_DISABLE_ACTIVE_MIGRATION => {
                if !value.is_empty() {
                    return Err(param_error("disable_active_migration must be zero-length"));
                }
                params.disable_active_migration = true;
            }
            ID_MAX_DATAGRAM_FRAME_SIZE => {
                params.max_datagram_frame_size = scalar(value)?;
            }
            ID_ORIGINAL_DESTINATION_CONNECTION_ID => {
                params.original_destination_connection_id = Some(connection_id(value)?);
            }
            ID_INITIAL_SOURCE_CONNECTION_ID => {
                params.initial_source_connection_id = Some(connection_id(value)?);
            }
            ID_RETRY_SOURCE_CONNECTION_ID => {
                params.retry_source_connection_id = Some(connection_id(value)?);
            }
            // Unknown / reserved parameters (including GREASE) are ignored.
            _ => {}
        }
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_default() {
        let params = TransportParams::default();
        let bytes = encode_transport_params(&params);
        let decoded = decode_transport_params(&bytes).expect("decode");
        assert_eq!(decoded, params);
    }

    #[test]
    fn roundtrip_populated() {
        let params = TransportParams {
            max_idle_timeout_ms: 30_000,
            max_udp_payload_size: 1452,
            initial_max_data: 1 << 20,
            initial_max_stream_data_bidi_local: 256 * 1024,
            initial_max_stream_data_bidi_remote: 256 * 1024,
            initial_max_stream_data_uni: 256 * 1024,
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            ack_delay_exponent: 3,
            max_ack_delay_ms: 25,
            active_connection_id_limit: 4,
            disable_active_migration: true,
            max_datagram_frame_size: 65535,
            original_destination_connection_id: Some(vec![0xaa; 8]),
            initial_source_connection_id: Some(vec![0xbb; 20]),
            retry_source_connection_id: Some(vec![0xcc; 4]),
        };
        let bytes = encode_transport_params(&params);
        let decoded = decode_transport_params(&bytes).expect("decode");
        assert_eq!(decoded, params);
    }

    /// RFC 9000 §7.3: the connection-ID parameters are omitted entirely when
    /// unset, so a client never emits the server-only ones.
    #[test]
    fn cid_params_are_absent_when_unset() {
        let params = TransportParams {
            initial_source_connection_id: Some(vec![1, 2, 3, 4]),
            ..TransportParams::default()
        };
        let decoded = decode_transport_params(&encode_transport_params(&params)).expect("decode");
        assert_eq!(decoded.initial_source_connection_id, Some(vec![1, 2, 3, 4]));
        assert_eq!(decoded.original_destination_connection_id, None);
        assert_eq!(decoded.retry_source_connection_id, None);
    }

    /// A zero-length connection ID is legal (RFC 9000 §5.1) and must round-trip
    /// as `Some(vec![])`, distinct from an absent parameter.
    #[test]
    fn zero_length_cid_param_roundtrips() {
        let params = TransportParams {
            initial_source_connection_id: Some(Vec::new()),
            ..TransportParams::default()
        };
        let decoded = decode_transport_params(&encode_transport_params(&params)).expect("decode");
        assert_eq!(decoded.initial_source_connection_id, Some(Vec::new()));
    }

    /// A connection-ID parameter longer than the 20-byte wire limit is a
    /// TRANSPORT_PARAMETER_ERROR rather than a silently truncated value.
    #[test]
    fn oversized_cid_param_is_rejected() {
        let mut bytes = Vec::new();
        put_varint(&mut bytes, ID_INITIAL_SOURCE_CONNECTION_ID);
        put_varint(&mut bytes, 21);
        bytes.extend_from_slice(&[0u8; 21]);
        let err = decode_transport_params(&bytes).expect_err("21-byte CID must be rejected");
        match err {
            OxiQuicError::TransportError { code, .. } => {
                assert_eq!(code, TransportErrorCode::TransportParameterError);
            }
            other => panic!("expected a transport error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_params_ignored() {
        let mut bytes = encode_transport_params(&TransportParams::default());
        // Append a GREASE-style unknown parameter id with a 3-byte value.
        put_varint(&mut bytes, 0x1234_5678);
        put_varint(&mut bytes, 3);
        bytes.extend_from_slice(&[1, 2, 3]);
        let decoded = decode_transport_params(&bytes).expect("decode tolerates unknown");
        assert_eq!(decoded, TransportParams::default());
    }

    #[test]
    fn truncated_value_rejected() {
        let mut bytes = Vec::new();
        put_varint(&mut bytes, ID_INITIAL_MAX_DATA);
        put_varint(&mut bytes, 4); // claims 4 bytes
        bytes.extend_from_slice(&[0x80, 0x00]); // only 2 present
        assert!(decode_transport_params(&bytes).is_err());
    }
}
