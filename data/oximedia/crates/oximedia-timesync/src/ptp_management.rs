//! PTP management messages per IEEE 1588-2019 §15.
//!
//! Management messages allow a management node to read or set the state of
//! PTP clocks and ports at run time.  Three actions are supported:
//!
//! * `GET`     — read the current value of a management data set.
//! * `SET`     — write a new value to a management data set.
//! * `COMMAND` — trigger an action (e.g. disable a port, force a BMCA run).
//!
//! # Message structure (simplified)
//! ```text
//! Common Header  (34 bytes, message type = 0xD)
//! ├── targetPortIdentity  (10 bytes)
//! ├── startingBoundaryHops (1 byte)
//! ├── boundaryHops         (1 byte)
//! ├── actionField          (1 nibble, upper nibble reserved)
//! ├── reserved             (1 byte)
//! └── TLV block
//!     ├── tlvType          (2 bytes, 0x0001 for MANAGEMENT)
//!     ├── lengthField      (2 bytes)
//!     ├── managementId     (2 bytes)
//!     └── dataField        (variable)
//! ```

use crate::error::{TimeSyncError, TimeSyncResult};
use crate::ptp::{ClockIdentity, Domain, PortIdentity};
use bytes::{Buf, BufMut, BytesMut};

// ---------------------------------------------------------------------------
// Action field
// ---------------------------------------------------------------------------

/// PTP management message action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ManagementAction {
    /// Read a dataset field.
    Get = 0x0,
    /// Write a dataset field.
    Set = 0x1,
    /// Execute an action.
    Command = 0x2,
    /// Response to a GET (from the clock).
    Response = 0x3,
    /// Acknowledge a SET or COMMAND.
    Acknowledge = 0x4,
}

impl ManagementAction {
    /// Converts from the lower nibble of the action byte.
    pub fn from_u8(value: u8) -> TimeSyncResult<Self> {
        match value & 0x0F {
            0x0 => Ok(Self::Get),
            0x1 => Ok(Self::Set),
            0x2 => Ok(Self::Command),
            0x3 => Ok(Self::Response),
            0x4 => Ok(Self::Acknowledge),
            v => Err(TimeSyncError::InvalidPacket(format!(
                "Unknown management action: {v:#x}"
            ))),
        }
    }

    /// Returns the numeric value (lower nibble).
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ---------------------------------------------------------------------------
// Management ID
// ---------------------------------------------------------------------------

/// Management data set identifiers (IEEE 1588-2019 Table 56).
///
/// Only a representative subset is included; the full table has ~50 entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ManagementId {
    /// NULL_PTP_MANAGEMENT (always valid, returns empty data).
    NullPtpManagement = 0x0000,
    /// CLOCK_DESCRIPTION — clock hardware and protocol information.
    ClockDescription = 0x0001,
    /// USER_DESCRIPTION — human-readable description string.
    UserDescription = 0x0002,
    /// SAVE_IN_NON_VOLATILE_STORAGE.
    SaveInNonVolatileStorage = 0x0003,
    /// RESET_NON_VOLATILE_STORAGE.
    ResetNonVolatileStorage = 0x0004,
    /// DEFAULT_DATA_SET — the clock's default dataset.
    DefaultDataSet = 0x2000,
    /// CURRENT_DATA_SET — the clock's current dataset.
    CurrentDataSet = 0x2001,
    /// PARENT_DATA_SET — the clock's parent dataset.
    ParentDataSet = 0x2002,
    /// TIME_PROPERTIES_DATA_SET.
    TimePropertiesDataSet = 0x2003,
    /// PORT_DATA_SET — a single port's dataset.
    PortDataSet = 0x2004,
    /// PRIORITY1 — priority 1 value.
    Priority1 = 0x2005,
    /// PRIORITY2 — priority 2 value.
    Priority2 = 0x2006,
    /// DOMAIN — domain number.
    Domain = 0x2007,
    /// SLAVE_ONLY — slave-only flag.
    SlaveOnly = 0x2008,
    /// PORT_ENABLE — enable/disable a port.
    PortEnable = 0x2011,
    /// ANNOUNCE_RECEIPT_TIMEOUT.
    AnnounceReceiptTimeout = 0x2014,
    /// LOG_ANNOUNCE_INTERVAL.
    LogAnnounceInterval = 0x2015,
    /// LOG_SYNC_INTERVAL.
    LogSyncInterval = 0x2016,
    /// VERSION_NUMBER — PTP version.
    VersionNumber = 0x2020,
    /// ENABLE_PORT (COMMAND only).
    EnablePort = 0x2030,
    /// DISABLE_PORT (COMMAND only).
    DisablePort = 0x2031,
}

impl ManagementId {
    /// Converts from a raw u16.
    pub fn from_u16(value: u16) -> TimeSyncResult<Self> {
        match value {
            0x0000 => Ok(Self::NullPtpManagement),
            0x0001 => Ok(Self::ClockDescription),
            0x0002 => Ok(Self::UserDescription),
            0x0003 => Ok(Self::SaveInNonVolatileStorage),
            0x0004 => Ok(Self::ResetNonVolatileStorage),
            0x2000 => Ok(Self::DefaultDataSet),
            0x2001 => Ok(Self::CurrentDataSet),
            0x2002 => Ok(Self::ParentDataSet),
            0x2003 => Ok(Self::TimePropertiesDataSet),
            0x2004 => Ok(Self::PortDataSet),
            0x2005 => Ok(Self::Priority1),
            0x2006 => Ok(Self::Priority2),
            0x2007 => Ok(Self::Domain),
            0x2008 => Ok(Self::SlaveOnly),
            0x2011 => Ok(Self::PortEnable),
            0x2014 => Ok(Self::AnnounceReceiptTimeout),
            0x2015 => Ok(Self::LogAnnounceInterval),
            0x2016 => Ok(Self::LogSyncInterval),
            0x2020 => Ok(Self::VersionNumber),
            0x2030 => Ok(Self::EnablePort),
            0x2031 => Ok(Self::DisablePort),
            v => Err(TimeSyncError::InvalidPacket(format!(
                "Unknown management ID: {v:#06x}"
            ))),
        }
    }

    /// Returns the raw u16 value.
    #[must_use]
    pub fn as_u16(self) -> u16 {
        self as u16
    }
}

// ---------------------------------------------------------------------------
// TLV
// ---------------------------------------------------------------------------

/// A management TLV (Type-Length-Value) block.
///
/// IEEE 1588-2019 §14.1.  For management messages, `tlv_type` is always
/// `0x0001`.
#[derive(Debug, Clone)]
pub struct ManagementTlv {
    /// Management ID for this TLV.
    pub management_id: ManagementId,
    /// Raw data field bytes (may be empty for GET requests).
    pub data: Vec<u8>,
}

impl ManagementTlv {
    /// TLV type for MANAGEMENT TLVs (IEEE 1588-2019 Table 52).
    pub const TLV_TYPE_MANAGEMENT: u16 = 0x0001;

    /// Creates a new TLV.
    #[must_use]
    pub fn new(management_id: ManagementId, data: Vec<u8>) -> Self {
        Self {
            management_id,
            data,
        }
    }

    /// Creates an empty TLV (used for GET requests).
    #[must_use]
    pub fn empty(management_id: ManagementId) -> Self {
        Self {
            management_id,
            data: Vec::new(),
        }
    }

    /// Serialises the TLV into `buf`.
    ///
    /// Layout: `[tlvType(2)][lengthField(2)][managementId(2)][data]`
    /// The `lengthField` covers `managementId` (2 bytes) + `data`.
    pub fn serialize(&self, buf: &mut BytesMut) {
        let length = 2 + self.data.len() as u16; // managementId + data
        buf.put_u16(Self::TLV_TYPE_MANAGEMENT);
        buf.put_u16(length);
        buf.put_u16(self.management_id.as_u16());
        buf.put_slice(&self.data);
    }

    /// Deserialises a TLV from `buf`.
    pub fn deserialize(buf: &mut impl Buf) -> TimeSyncResult<Self> {
        if buf.remaining() < 6 {
            return Err(TimeSyncError::InvalidPacket("TLV too short".to_string()));
        }
        let tlv_type = buf.get_u16();
        if tlv_type != Self::TLV_TYPE_MANAGEMENT {
            return Err(TimeSyncError::InvalidPacket(format!(
                "Expected MANAGEMENT TLV type 0x0001, got {tlv_type:#06x}"
            )));
        }
        let length = buf.get_u16() as usize;
        if length < 2 || buf.remaining() < length {
            return Err(TimeSyncError::InvalidPacket(format!(
                "TLV length {length} exceeds buffer"
            )));
        }
        let mgmt_id_raw = buf.get_u16();
        let management_id = ManagementId::from_u16(mgmt_id_raw)?;
        let data_len = length - 2;
        let mut data = vec![0u8; data_len];
        buf.copy_to_slice(&mut data);
        Ok(Self {
            management_id,
            data,
        })
    }

    /// Returns the total wire size of this TLV in bytes.
    #[must_use]
    pub fn wire_size(&self) -> usize {
        6 + self.data.len() // tlvType(2) + length(2) + managementId(2) + data
    }
}

// ---------------------------------------------------------------------------
// Management message header
// ---------------------------------------------------------------------------

/// PTP management message (IEEE 1588-2019 §15.3.1).
///
/// The common PTP header is not duplicated here; callers are responsible for
/// prepending it.  This struct models only the management-specific body.
#[derive(Debug, Clone)]
pub struct ManagementMessage {
    /// Source port identity (from the common header — carried here for
    /// convenience when building replies).
    pub source_port_identity: PortIdentity,
    /// Port identity this message targets (`all-ports` if `port_number == 0`).
    pub target_port_identity: PortIdentity,
    /// Starting boundary hops count.
    pub starting_boundary_hops: u8,
    /// Remaining boundary hops.
    pub boundary_hops: u8,
    /// GET / SET / COMMAND / RESPONSE / ACKNOWLEDGE.
    pub action: ManagementAction,
    /// Management TLV.
    pub tlv: ManagementTlv,
}

impl ManagementMessage {
    /// Creates a GET message for the specified management ID.
    #[must_use]
    pub fn get(source: PortIdentity, target: PortIdentity, management_id: ManagementId) -> Self {
        Self {
            source_port_identity: source,
            target_port_identity: target,
            starting_boundary_hops: 0,
            boundary_hops: 0,
            action: ManagementAction::Get,
            tlv: ManagementTlv::empty(management_id),
        }
    }

    /// Creates a SET message with the given TLV data.
    #[must_use]
    pub fn set(
        source: PortIdentity,
        target: PortIdentity,
        management_id: ManagementId,
        data: Vec<u8>,
    ) -> Self {
        Self {
            source_port_identity: source,
            target_port_identity: target,
            starting_boundary_hops: 0,
            boundary_hops: 0,
            action: ManagementAction::Set,
            tlv: ManagementTlv::new(management_id, data),
        }
    }

    /// Creates a COMMAND message.
    #[must_use]
    pub fn command(
        source: PortIdentity,
        target: PortIdentity,
        management_id: ManagementId,
    ) -> Self {
        Self {
            source_port_identity: source,
            target_port_identity: target,
            starting_boundary_hops: 0,
            boundary_hops: 0,
            action: ManagementAction::Command,
            tlv: ManagementTlv::empty(management_id),
        }
    }

    /// Creates a RESPONSE message with TLV data (reply to GET).
    #[must_use]
    pub fn response(
        source: PortIdentity,
        target: PortIdentity,
        management_id: ManagementId,
        data: Vec<u8>,
    ) -> Self {
        Self {
            source_port_identity: source,
            target_port_identity: target,
            starting_boundary_hops: 0,
            boundary_hops: 0,
            action: ManagementAction::Response,
            tlv: ManagementTlv::new(management_id, data),
        }
    }

    /// Serialises the management message body (excluding the common header)
    /// into a new [`BytesMut`].
    ///
    /// Layout:
    /// ```text
    /// targetPortIdentity.clockIdentity  (8 bytes)
    /// targetPortIdentity.portNumber     (2 bytes)
    /// startingBoundaryHops              (1 byte)
    /// boundaryHops                      (1 byte)
    /// actionField                       (1 byte, lower nibble)
    /// reserved                          (1 byte)
    /// TLV block                         (variable)
    /// ```
    pub fn serialize(&self) -> BytesMut {
        let tlv_size = self.tlv.wire_size();
        let mut buf = BytesMut::with_capacity(14 + tlv_size);

        buf.put_slice(&self.target_port_identity.clock_identity.0);
        buf.put_u16(self.target_port_identity.port_number);
        buf.put_u8(self.starting_boundary_hops);
        buf.put_u8(self.boundary_hops);
        buf.put_u8(self.action.as_u8() & 0x0F);
        buf.put_u8(0); // reserved

        self.tlv.serialize(&mut buf);
        buf
    }

    /// Deserialises a management message body from `buf`, given the
    /// `source_port_identity` from the already-parsed common header.
    pub fn deserialize(buf: &mut impl Buf, source: PortIdentity) -> TimeSyncResult<Self> {
        if buf.remaining() < 14 {
            return Err(TimeSyncError::InvalidPacket(
                "Management message body too short".to_string(),
            ));
        }

        let mut clock_id_bytes = [0u8; 8];
        buf.copy_to_slice(&mut clock_id_bytes);
        let target_clock_id = ClockIdentity(clock_id_bytes);
        let target_port_number = buf.get_u16();
        let target_port_identity = PortIdentity::new(target_clock_id, target_port_number);

        let starting_boundary_hops = buf.get_u8();
        let boundary_hops = buf.get_u8();
        let action_byte = buf.get_u8();
        let action = ManagementAction::from_u8(action_byte)?;
        buf.get_u8(); // reserved

        let tlv = ManagementTlv::deserialize(buf)?;

        Ok(Self {
            source_port_identity: source,
            target_port_identity,
            starting_boundary_hops,
            boundary_hops,
            action,
            tlv,
        })
    }

    /// Returns the wire size of the entire management body (excluding common
    /// header) in bytes.
    #[must_use]
    pub fn body_wire_size(&self) -> usize {
        14 + self.tlv.wire_size()
    }
}

// ---------------------------------------------------------------------------
// Simple clock state snapshot (used in GET responses)
// ---------------------------------------------------------------------------

/// Snapshot of priority fields for a GET PRIORITY1 / PRIORITY2 response.
#[derive(Debug, Clone, Copy)]
pub struct PrioritySnapshot {
    /// Priority 1 value (lower = better master).
    pub priority1: u8,
    /// Priority 2 value (lower = better master).
    pub priority2: u8,
}

impl PrioritySnapshot {
    /// Serialises to 2 bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 2] {
        [self.priority1, self.priority2]
    }

    /// Deserialises from 2 bytes.
    #[must_use]
    pub fn from_bytes(bytes: [u8; 2]) -> Self {
        Self {
            priority1: bytes[0],
            priority2: bytes[1],
        }
    }
}

/// A simple in-memory management responder for use in unit tests and stubs.
///
/// Holds a small set of writable fields and produces appropriate RESPONSE /
/// ACKNOWLEDGE messages for incoming GET/SET/COMMAND management messages.
#[derive(Debug)]
pub struct SimpleManagementResponder {
    /// Local port identity.
    pub local_port: PortIdentity,
    /// Domain number.
    pub domain: Domain,
    /// Priority 1.
    pub priority1: u8,
    /// Priority 2.
    pub priority2: u8,
    /// Whether the port is enabled.
    pub port_enabled: bool,
    /// PTP version (should be 2).
    pub version: u8,
}

impl SimpleManagementResponder {
    /// Creates a responder with default values.
    #[must_use]
    pub fn new(local_port: PortIdentity) -> Self {
        Self {
            local_port,
            domain: Domain::DEFAULT,
            priority1: 128,
            priority2: 128,
            port_enabled: true,
            version: 2,
        }
    }

    /// Handles a management message and returns an optional reply.
    ///
    /// Returns `None` for messages that require no reply (e.g. ACKNOWLEDGE).
    pub fn handle(&mut self, msg: &ManagementMessage) -> Option<ManagementMessage> {
        match msg.action {
            ManagementAction::Get => self.handle_get(msg),
            ManagementAction::Set => self.handle_set(msg),
            ManagementAction::Command => self.handle_command(msg),
            ManagementAction::Response | ManagementAction::Acknowledge => None,
        }
    }

    fn handle_get(&self, msg: &ManagementMessage) -> Option<ManagementMessage> {
        let data = match msg.tlv.management_id {
            ManagementId::Priority1 => vec![self.priority1],
            ManagementId::Priority2 => vec![self.priority2],
            ManagementId::Domain => vec![self.domain.0],
            ManagementId::VersionNumber => vec![self.version],
            ManagementId::PortEnable => vec![u8::from(self.port_enabled)],
            ManagementId::NullPtpManagement => vec![],
            _ => return None, // unsupported
        };
        Some(ManagementMessage::response(
            self.local_port,
            msg.source_port_identity,
            msg.tlv.management_id,
            data,
        ))
    }

    fn handle_set(&mut self, msg: &ManagementMessage) -> Option<ManagementMessage> {
        let data = &msg.tlv.data;
        match msg.tlv.management_id {
            ManagementId::Priority1 if data.len() >= 1 => {
                self.priority1 = data[0];
            }
            ManagementId::Priority2 if data.len() >= 1 => {
                self.priority2 = data[0];
            }
            ManagementId::Domain if data.len() >= 1 => {
                self.domain = Domain(data[0]);
            }
            ManagementId::PortEnable if data.len() >= 1 => {
                self.port_enabled = data[0] != 0;
            }
            _ => return None,
        }
        // Acknowledge the SET.
        Some(ManagementMessage {
            source_port_identity: self.local_port,
            target_port_identity: msg.source_port_identity,
            starting_boundary_hops: 0,
            boundary_hops: 0,
            action: ManagementAction::Acknowledge,
            tlv: ManagementTlv::empty(msg.tlv.management_id),
        })
    }

    fn handle_command(&mut self, msg: &ManagementMessage) -> Option<ManagementMessage> {
        match msg.tlv.management_id {
            ManagementId::EnablePort => {
                self.port_enabled = true;
            }
            ManagementId::DisablePort => {
                self.port_enabled = false;
            }
            _ => return None,
        }
        Some(ManagementMessage {
            source_port_identity: self.local_port,
            target_port_identity: msg.source_port_identity,
            starting_boundary_hops: 0,
            boundary_hops: 0,
            action: ManagementAction::Acknowledge,
            tlv: ManagementTlv::empty(msg.tlv.management_id),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_port(byte: u8) -> PortIdentity {
        PortIdentity::new(ClockIdentity([byte; 8]), 1)
    }

    // -----------------------------------------------------------------------
    // ManagementAction
    // -----------------------------------------------------------------------

    #[test]
    fn test_action_round_trip() {
        for (val, expected) in [
            (0u8, ManagementAction::Get),
            (1, ManagementAction::Set),
            (2, ManagementAction::Command),
            (3, ManagementAction::Response),
            (4, ManagementAction::Acknowledge),
        ] {
            let decoded = ManagementAction::from_u8(val).expect("known action");
            assert_eq!(decoded, expected);
            assert_eq!(decoded.as_u8(), val);
        }
    }

    #[test]
    fn test_action_unknown_errors() {
        assert!(ManagementAction::from_u8(5).is_err());
        assert!(ManagementAction::from_u8(0xFF).is_err());
    }

    // -----------------------------------------------------------------------
    // ManagementId
    // -----------------------------------------------------------------------

    #[test]
    fn test_management_id_round_trip() {
        for id in [
            ManagementId::NullPtpManagement,
            ManagementId::Priority1,
            ManagementId::Priority2,
            ManagementId::Domain,
            ManagementId::PortEnable,
            ManagementId::EnablePort,
            ManagementId::DisablePort,
        ] {
            let raw = id.as_u16();
            let back = ManagementId::from_u16(raw).expect("round trip");
            assert_eq!(back, id, "round trip failed for {id:?}");
        }
    }

    #[test]
    fn test_management_id_unknown_errors() {
        assert!(ManagementId::from_u16(0xFFFF).is_err());
    }

    // -----------------------------------------------------------------------
    // ManagementTlv serialisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_tlv_serialize_deserialize_empty() {
        let tlv = ManagementTlv::empty(ManagementId::NullPtpManagement);
        let mut buf = BytesMut::new();
        tlv.serialize(&mut buf);
        // Wire: tlvType(2) + length(2) + managementId(2) = 6 bytes.
        assert_eq!(buf.len(), 6);

        let back = ManagementTlv::deserialize(&mut buf.as_ref()).expect("deserialize");
        assert_eq!(back.management_id, ManagementId::NullPtpManagement);
        assert!(back.data.is_empty());
    }

    #[test]
    fn test_tlv_serialize_deserialize_with_data() {
        let data = vec![0xAB, 0xCD, 0xEF];
        let tlv = ManagementTlv::new(ManagementId::Priority1, data.clone());
        let mut buf = BytesMut::new();
        tlv.serialize(&mut buf);

        let back = ManagementTlv::deserialize(&mut buf.as_ref()).expect("deserialize");
        assert_eq!(back.management_id, ManagementId::Priority1);
        assert_eq!(back.data, data);
    }

    // -----------------------------------------------------------------------
    // ManagementMessage serialisation
    // -----------------------------------------------------------------------

    #[test]
    fn test_management_message_get_roundtrip() {
        let src = make_port(0x01);
        let dst = make_port(0x02);
        let msg = ManagementMessage::get(src, dst, ManagementId::Priority1);
        let serialised = msg.serialize();

        let back =
            ManagementMessage::deserialize(&mut serialised.as_ref(), src).expect("deserialize");
        assert_eq!(back.action, ManagementAction::Get);
        assert_eq!(back.tlv.management_id, ManagementId::Priority1);
        assert!(back.tlv.data.is_empty());
    }

    #[test]
    fn test_management_message_set_roundtrip() {
        let src = make_port(0x10);
        let dst = make_port(0x20);
        let data = vec![100u8];
        let msg = ManagementMessage::set(src, dst, ManagementId::Priority1, data.clone());
        let serialised = msg.serialize();

        let back =
            ManagementMessage::deserialize(&mut serialised.as_ref(), src).expect("deserialize");
        assert_eq!(back.action, ManagementAction::Set);
        assert_eq!(back.tlv.management_id, ManagementId::Priority1);
        assert_eq!(back.tlv.data, data);
    }

    #[test]
    fn test_management_message_command_roundtrip() {
        let src = make_port(0x05);
        let dst = make_port(0x06);
        let msg = ManagementMessage::command(src, dst, ManagementId::DisablePort);
        let serialised = msg.serialize();

        let back =
            ManagementMessage::deserialize(&mut serialised.as_ref(), src).expect("deserialize");
        assert_eq!(back.action, ManagementAction::Command);
        assert_eq!(back.tlv.management_id, ManagementId::DisablePort);
    }

    // -----------------------------------------------------------------------
    // SimpleManagementResponder
    // -----------------------------------------------------------------------

    #[test]
    fn test_responder_get_priority1() {
        let port = make_port(0xAA);
        let mut resp = SimpleManagementResponder::new(port);
        resp.priority1 = 200;

        let get_msg = ManagementMessage::get(make_port(0x01), port, ManagementId::Priority1);
        let reply = resp.handle(&get_msg).expect("should reply");
        assert_eq!(reply.action, ManagementAction::Response);
        assert_eq!(reply.tlv.data, vec![200u8]);
    }

    #[test]
    fn test_responder_set_priority1() {
        let port = make_port(0xBB);
        let mut resp = SimpleManagementResponder::new(port);

        let set_msg =
            ManagementMessage::set(make_port(0x01), port, ManagementId::Priority1, vec![50]);
        let ack = resp.handle(&set_msg).expect("should acknowledge");
        assert_eq!(ack.action, ManagementAction::Acknowledge);
        assert_eq!(resp.priority1, 50, "priority1 should be updated");
    }

    #[test]
    fn test_responder_command_disable_port() {
        let port = make_port(0xCC);
        let mut resp = SimpleManagementResponder::new(port);
        assert!(resp.port_enabled);

        let cmd = ManagementMessage::command(make_port(0x01), port, ManagementId::DisablePort);
        let ack = resp.handle(&cmd).expect("should acknowledge");
        assert_eq!(ack.action, ManagementAction::Acknowledge);
        assert!(!resp.port_enabled, "port should be disabled");
    }

    #[test]
    fn test_responder_command_enable_port() {
        let port = make_port(0xDD);
        let mut resp = SimpleManagementResponder::new(port);
        resp.port_enabled = false;

        let cmd = ManagementMessage::command(make_port(0x01), port, ManagementId::EnablePort);
        resp.handle(&cmd).expect("should acknowledge");
        assert!(resp.port_enabled, "port should be enabled");
    }

    #[test]
    fn test_responder_response_returns_none() {
        let port = make_port(0xEE);
        let mut resp = SimpleManagementResponder::new(port);

        // RESPONSE messages should produce no further reply.
        let rsp_msg = ManagementMessage::response(
            make_port(0x01),
            port,
            ManagementId::NullPtpManagement,
            vec![],
        );
        let reply = resp.handle(&rsp_msg);
        assert!(reply.is_none(), "RESPONSE should not generate a reply");
    }

    #[test]
    fn test_priority_snapshot_round_trip() {
        let snap = PrioritySnapshot {
            priority1: 100,
            priority2: 200,
        };
        let bytes = snap.to_bytes();
        let back = PrioritySnapshot::from_bytes(bytes);
        assert_eq!(back.priority1, 100);
        assert_eq!(back.priority2, 200);
    }
}
