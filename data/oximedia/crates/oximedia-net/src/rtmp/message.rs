//! RTMP message types.
//!
//! This module defines the various message types used in RTMP.

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

/// RTMP message type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Set Chunk Size (1).
    SetChunkSize = 1,
    /// Abort Message (2).
    Abort = 2,
    /// Acknowledgement (3).
    Acknowledgement = 3,
    /// User Control Message (4).
    UserControl = 4,
    /// Window Acknowledgement Size (5).
    WindowAckSize = 5,
    /// Set Peer Bandwidth (6).
    SetPeerBandwidth = 6,
    /// Audio Message (8).
    Audio = 8,
    /// Video Message (9).
    Video = 9,
    /// Data Message AMF3 (15).
    DataAmf3 = 15,
    /// Shared Object AMF3 (16).
    SharedObjectAmf3 = 16,
    /// Command Message AMF3 (17).
    CommandAmf3 = 17,
    /// Data Message AMF0 (18).
    DataAmf0 = 18,
    /// Shared Object AMF0 (19).
    SharedObjectAmf0 = 19,
    /// Command Message AMF0 (20).
    CommandAmf0 = 20,
    /// Aggregate Message (22).
    Aggregate = 22,
}

impl MessageType {
    /// Creates from raw type ID.
    #[must_use]
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(Self::SetChunkSize),
            2 => Some(Self::Abort),
            3 => Some(Self::Acknowledgement),
            4 => Some(Self::UserControl),
            5 => Some(Self::WindowAckSize),
            6 => Some(Self::SetPeerBandwidth),
            8 => Some(Self::Audio),
            9 => Some(Self::Video),
            15 => Some(Self::DataAmf3),
            16 => Some(Self::SharedObjectAmf3),
            17 => Some(Self::CommandAmf3),
            18 => Some(Self::DataAmf0),
            19 => Some(Self::SharedObjectAmf0),
            20 => Some(Self::CommandAmf0),
            22 => Some(Self::Aggregate),
            _ => None,
        }
    }

    /// Returns true if this is a control message.
    #[must_use]
    pub const fn is_control(&self) -> bool {
        matches!(
            self,
            Self::SetChunkSize
                | Self::Abort
                | Self::Acknowledgement
                | Self::UserControl
                | Self::WindowAckSize
                | Self::SetPeerBandwidth
        )
    }

    /// Returns true if this is a command message.
    #[must_use]
    pub const fn is_command(&self) -> bool {
        matches!(self, Self::CommandAmf0 | Self::CommandAmf3)
    }

    /// Returns true if this is a media message.
    #[must_use]
    pub const fn is_media(&self) -> bool {
        matches!(self, Self::Audio | Self::Video)
    }
}

/// User control event types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum UserControlEvent {
    /// Stream Begin.
    StreamBegin = 0,
    /// Stream EOF.
    StreamEof = 1,
    /// Stream Dry.
    StreamDry = 2,
    /// Set Buffer Length.
    SetBufferLength = 3,
    /// Stream Is Recorded.
    StreamIsRecorded = 4,
    /// Ping Request.
    PingRequest = 6,
    /// Ping Response.
    PingResponse = 7,
}

impl UserControlEvent {
    /// Creates from raw event type.
    #[must_use]
    pub const fn from_id(id: u16) -> Option<Self> {
        match id {
            0 => Some(Self::StreamBegin),
            1 => Some(Self::StreamEof),
            2 => Some(Self::StreamDry),
            3 => Some(Self::SetBufferLength),
            4 => Some(Self::StreamIsRecorded),
            6 => Some(Self::PingRequest),
            7 => Some(Self::PingResponse),
            _ => None,
        }
    }
}

/// Control message payload.
#[derive(Debug, Clone)]
pub enum ControlMessage {
    /// Set chunk size.
    SetChunkSize(u32),
    /// Abort chunk stream.
    Abort(u32),
    /// Acknowledgement.
    Acknowledgement(u32),
    /// Window acknowledgement size.
    WindowAckSize(u32),
    /// Set peer bandwidth.
    SetPeerBandwidth {
        /// Window size.
        size: u32,
        /// Limit type (0=hard, 1=soft, 2=dynamic).
        limit_type: u8,
    },
    /// User control event.
    UserControl {
        /// Event type.
        event: UserControlEvent,
        /// Event data (stream ID, timestamp, etc.).
        data: u32,
        /// Extra data for buffer length.
        extra: Option<u32>,
    },
}

impl ControlMessage {
    /// Encodes the control message.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        match self {
            Self::SetChunkSize(size) => {
                buf.put_u32(*size & 0x7FFF_FFFF); // Bit 0 must be 0
            }
            Self::Abort(csid) => {
                buf.put_u32(*csid);
            }
            Self::Acknowledgement(seq) => {
                buf.put_u32(*seq);
            }
            Self::WindowAckSize(size) => {
                buf.put_u32(*size);
            }
            Self::SetPeerBandwidth { size, limit_type } => {
                buf.put_u32(*size);
                buf.put_u8(*limit_type);
            }
            Self::UserControl { event, data, extra } => {
                buf.put_u16(*event as u16);
                buf.put_u32(*data);
                if let Some(ex) = extra {
                    buf.put_u32(*ex);
                }
            }
        }

        buf.freeze()
    }

    /// Decodes a control message.
    ///
    /// # Errors
    ///
    /// Returns an error if the message is malformed.
    pub fn decode(message_type: MessageType, data: &[u8]) -> NetResult<Self> {
        let mut buf = data;

        match message_type {
            MessageType::SetChunkSize => {
                if buf.len() < 4 {
                    return Err(NetError::parse(0, "SetChunkSize too short"));
                }
                Ok(Self::SetChunkSize(buf.get_u32() & 0x7FFF_FFFF))
            }
            MessageType::Abort => {
                if buf.len() < 4 {
                    return Err(NetError::parse(0, "Abort too short"));
                }
                Ok(Self::Abort(buf.get_u32()))
            }
            MessageType::Acknowledgement => {
                if buf.len() < 4 {
                    return Err(NetError::parse(0, "Ack too short"));
                }
                Ok(Self::Acknowledgement(buf.get_u32()))
            }
            MessageType::WindowAckSize => {
                if buf.len() < 4 {
                    return Err(NetError::parse(0, "WindowAckSize too short"));
                }
                Ok(Self::WindowAckSize(buf.get_u32()))
            }
            MessageType::SetPeerBandwidth => {
                if buf.len() < 5 {
                    return Err(NetError::parse(0, "SetPeerBandwidth too short"));
                }
                let size = buf.get_u32();
                let limit_type = buf.get_u8();
                Ok(Self::SetPeerBandwidth { size, limit_type })
            }
            MessageType::UserControl => {
                if buf.len() < 6 {
                    return Err(NetError::parse(0, "UserControl too short"));
                }
                let event_id = buf.get_u16();
                let event = UserControlEvent::from_id(event_id).ok_or_else(|| {
                    NetError::parse(0, format!("Unknown user control event: {event_id}"))
                })?;
                let data = buf.get_u32();
                let extra = if buf.remaining() >= 4 {
                    Some(buf.get_u32())
                } else {
                    None
                };
                Ok(Self::UserControl { event, data, extra })
            }
            _ => Err(NetError::parse(
                0,
                format!("Not a control message: {:?}", message_type),
            )),
        }
    }

    /// Returns the message type for this control message.
    #[must_use]
    pub const fn message_type(&self) -> MessageType {
        match self {
            Self::SetChunkSize(_) => MessageType::SetChunkSize,
            Self::Abort(_) => MessageType::Abort,
            Self::Acknowledgement(_) => MessageType::Acknowledgement,
            Self::WindowAckSize(_) => MessageType::WindowAckSize,
            Self::SetPeerBandwidth { .. } => MessageType::SetPeerBandwidth,
            Self::UserControl { .. } => MessageType::UserControl,
        }
    }
}

/// Command message (AMF-encoded).
#[derive(Debug, Clone)]
pub struct CommandMessage {
    /// Command name (e.g., "connect", "play").
    pub name: String,
    /// Transaction ID.
    pub transaction_id: f64,
    /// Command object (optional).
    pub command_object: Option<super::amf::AmfValue>,
    /// Additional arguments.
    pub args: Vec<super::amf::AmfValue>,
}

impl CommandMessage {
    /// Creates a new command message.
    #[must_use]
    pub fn new(name: impl Into<String>, transaction_id: f64) -> Self {
        Self {
            name: name.into(),
            transaction_id,
            command_object: None,
            args: Vec::new(),
        }
    }

    /// Sets the command object.
    #[must_use]
    pub fn with_command_object(mut self, obj: super::amf::AmfValue) -> Self {
        self.command_object = Some(obj);
        self
    }

    /// Adds an argument.
    #[must_use]
    pub fn with_arg(mut self, arg: super::amf::AmfValue) -> Self {
        self.args.push(arg);
        self
    }

    /// Common command: connect.
    #[must_use]
    pub fn connect(app: &str, tc_url: &str) -> Self {
        use super::amf::AmfValue;

        let mut props = std::collections::HashMap::new();
        props.insert("app".to_string(), AmfValue::String(app.to_string()));
        props.insert("tcUrl".to_string(), AmfValue::String(tc_url.to_string()));
        props.insert("fpad".to_string(), AmfValue::Boolean(false));
        props.insert("capabilities".to_string(), AmfValue::Number(239.0));
        props.insert("audioCodecs".to_string(), AmfValue::Number(3575.0));
        props.insert("videoCodecs".to_string(), AmfValue::Number(252.0));
        props.insert("videoFunction".to_string(), AmfValue::Number(1.0));

        Self::new("connect", 1.0).with_command_object(AmfValue::Object(props))
    }

    /// Common command: createStream.
    #[must_use]
    pub fn create_stream(transaction_id: f64) -> Self {
        use super::amf::AmfValue;
        Self::new("createStream", transaction_id).with_command_object(AmfValue::Null)
    }

    /// Common command: play.
    #[must_use]
    pub fn play(stream_name: &str, transaction_id: f64) -> Self {
        use super::amf::AmfValue;
        Self::new("play", transaction_id)
            .with_command_object(AmfValue::Null)
            .with_arg(AmfValue::String(stream_name.to_string()))
    }

    /// Common command: publish.
    #[must_use]
    pub fn publish(stream_name: &str, publish_type: &str, transaction_id: f64) -> Self {
        use super::amf::AmfValue;
        Self::new("publish", transaction_id)
            .with_command_object(AmfValue::Null)
            .with_arg(AmfValue::String(stream_name.to_string()))
            .with_arg(AmfValue::String(publish_type.to_string()))
    }

    /// Common command: _result (success response).
    #[must_use]
    pub fn result(transaction_id: f64, result: super::amf::AmfValue) -> Self {
        Self::new("_result", transaction_id)
            .with_command_object(super::amf::AmfValue::Null)
            .with_arg(result)
    }

    /// Common command: _error (error response).
    #[must_use]
    pub fn error(transaction_id: f64, error: super::amf::AmfValue) -> Self {
        Self::new("_error", transaction_id)
            .with_command_object(super::amf::AmfValue::Null)
            .with_arg(error)
    }
}

/// Data message (AMF-encoded metadata).
#[derive(Debug, Clone)]
pub struct DataMessage {
    /// Handler name (e.g., "@setDataFrame", "onMetaData").
    pub handler: String,
    /// Data values.
    pub values: Vec<super::amf::AmfValue>,
}

impl DataMessage {
    /// Creates a new data message.
    #[must_use]
    pub fn new(handler: impl Into<String>) -> Self {
        Self {
            handler: handler.into(),
            values: Vec::new(),
        }
    }

    /// Adds a value.
    #[must_use]
    pub fn with_value(mut self, value: super::amf::AmfValue) -> Self {
        self.values.push(value);
        self
    }

    /// Creates an onMetaData message.
    #[must_use]
    pub fn on_metadata(metadata: super::amf::AmfValue) -> Self {
        Self::new("onMetaData").with_value(metadata)
    }
}

/// Complete RTMP message.
#[derive(Debug, Clone)]
pub enum RtmpMessage {
    /// Control protocol message.
    Control(ControlMessage),
    /// Command message.
    Command(CommandMessage),
    /// Data/metadata message.
    Data(DataMessage),
    /// Audio data.
    Audio(Bytes),
    /// Video data.
    Video(Bytes),
    /// Unknown/unsupported message.
    Unknown {
        /// Message type ID.
        type_id: u8,
        /// Raw payload.
        payload: Bytes,
    },
}

impl RtmpMessage {
    /// Returns the message type ID.
    #[must_use]
    pub fn type_id(&self) -> u8 {
        match self {
            Self::Control(ctrl) => ctrl.message_type() as u8,
            Self::Command(_) => MessageType::CommandAmf0 as u8,
            Self::Data(_) => MessageType::DataAmf0 as u8,
            Self::Audio(_) => MessageType::Audio as u8,
            Self::Video(_) => MessageType::Video as u8,
            Self::Unknown { type_id, .. } => *type_id,
        }
    }

    /// Returns true if this is a control message.
    #[must_use]
    pub const fn is_control(&self) -> bool {
        matches!(self, Self::Control(_))
    }

    /// Returns true if this is a media message.
    #[must_use]
    pub const fn is_media(&self) -> bool {
        matches!(self, Self::Audio(_) | Self::Video(_))
    }

    /// Returns true if this is a command message.
    #[must_use]
    pub const fn is_command(&self) -> bool {
        matches!(self, Self::Command(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_from_id() {
        assert_eq!(MessageType::from_id(1), Some(MessageType::SetChunkSize));
        assert_eq!(MessageType::from_id(20), Some(MessageType::CommandAmf0));
        assert_eq!(MessageType::from_id(99), None);
    }

    #[test]
    fn test_message_type_classification() {
        assert!(MessageType::SetChunkSize.is_control());
        assert!(!MessageType::SetChunkSize.is_command());
        assert!(MessageType::CommandAmf0.is_command());
        assert!(MessageType::Video.is_media());
    }

    #[test]
    fn test_control_message_encode_decode() {
        let msg = ControlMessage::SetChunkSize(4096);
        let encoded = msg.encode();
        let decoded = ControlMessage::decode(MessageType::SetChunkSize, &encoded)
            .expect("should succeed in test");

        if let ControlMessage::SetChunkSize(size) = decoded {
            assert_eq!(size, 4096);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_window_ack_size() {
        let msg = ControlMessage::WindowAckSize(2_500_000);
        let encoded = msg.encode();
        let decoded = ControlMessage::decode(MessageType::WindowAckSize, &encoded)
            .expect("should succeed in test");

        if let ControlMessage::WindowAckSize(size) = decoded {
            assert_eq!(size, 2_500_000);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_peer_bandwidth() {
        let msg = ControlMessage::SetPeerBandwidth {
            size: 5_000_000,
            limit_type: 2,
        };
        let encoded = msg.encode();
        let decoded = ControlMessage::decode(MessageType::SetPeerBandwidth, &encoded)
            .expect("should succeed in test");

        if let ControlMessage::SetPeerBandwidth { size, limit_type } = decoded {
            assert_eq!(size, 5_000_000);
            assert_eq!(limit_type, 2);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_user_control_event() {
        let msg = ControlMessage::UserControl {
            event: UserControlEvent::StreamBegin,
            data: 1,
            extra: None,
        };
        let encoded = msg.encode();
        let decoded = ControlMessage::decode(MessageType::UserControl, &encoded)
            .expect("should succeed in test");

        if let ControlMessage::UserControl { event, data, .. } = decoded {
            assert_eq!(event, UserControlEvent::StreamBegin);
            assert_eq!(data, 1);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_command_message_connect() {
        let cmd = CommandMessage::connect("live", "rtmp://localhost/live");
        assert_eq!(cmd.name, "connect");
        assert_eq!(cmd.transaction_id, 1.0);
        assert!(cmd.command_object.is_some());
    }

    #[test]
    fn test_command_message_play() {
        let cmd = CommandMessage::play("stream1", 5.0);
        assert_eq!(cmd.name, "play");
        assert_eq!(cmd.transaction_id, 5.0);
        assert_eq!(cmd.args.len(), 1);
    }

    #[test]
    fn test_rtmp_message_type_id() {
        let ctrl = RtmpMessage::Control(ControlMessage::SetChunkSize(128));
        assert_eq!(ctrl.type_id(), 1);
        assert!(ctrl.is_control());

        let audio = RtmpMessage::Audio(Bytes::new());
        assert_eq!(audio.type_id(), 8);
        assert!(audio.is_media());
    }
}
