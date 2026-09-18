//! mDNS / DNS-SD service discovery (RFC 6762 / RFC 6763).
//!
//! Implements pure-Rust packet construction and parsing for mDNS queries,
//! announcements, and record parsing without any external dependencies beyond
//! `std`.

use crate::error::{NetError, NetResult};
use std::net::{Ipv4Addr, Ipv6Addr};

// ─── mDNS multicast constants ─────────────────────────────────────────────────

/// mDNS IPv4 multicast address (224.0.0.251).
pub const MDNS_IPV4_MULTICAST: Ipv4Addr = Ipv4Addr::new(224, 0, 0, 251);
/// mDNS IPv6 multicast address (FF02::FB).
pub const MDNS_IPV6_MULTICAST: Ipv6Addr = Ipv6Addr::new(0xFF02, 0, 0, 0, 0, 0, 0, 0x00FB);
/// mDNS UDP port.
pub const MDNS_PORT: u16 = 5353;

// ─── DNS record type codes ────────────────────────────────────────────────────

const DNS_TYPE_A: u16 = 1;
const DNS_TYPE_AAAA: u16 = 28;
const DNS_TYPE_PTR: u16 = 12;
const DNS_TYPE_SRV: u16 = 33;
const DNS_TYPE_TXT: u16 = 16;

const DNS_CLASS_IN: u16 = 1;
/// QU (Unicast-response) bit set on the class field in a query.
const DNS_QU_BIT: u16 = 0x8000;

// ─── MdnsType ────────────────────────────────────────────────────────────────

/// DNS record type used in mDNS records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdnsType {
    /// Address record — IPv4 host address.
    A,
    /// Quad-A record — IPv6 host address.
    Aaaa,
    /// Pointer record — domain name pointer (used for service discovery).
    Ptr,
    /// Service location record.
    Srv,
    /// Text record — key/value metadata.
    Txt,
    /// Unknown record type.
    Unknown(u16),
}

impl MdnsType {
    fn from_u16(v: u16) -> Self {
        match v {
            DNS_TYPE_A => Self::A,
            DNS_TYPE_AAAA => Self::Aaaa,
            DNS_TYPE_PTR => Self::Ptr,
            DNS_TYPE_SRV => Self::Srv,
            DNS_TYPE_TXT => Self::Txt,
            other => Self::Unknown(other),
        }
    }

    fn as_u16(self) -> u16 {
        match self {
            Self::A => DNS_TYPE_A,
            Self::Aaaa => DNS_TYPE_AAAA,
            Self::Ptr => DNS_TYPE_PTR,
            Self::Srv => DNS_TYPE_SRV,
            Self::Txt => DNS_TYPE_TXT,
            Self::Unknown(v) => v,
        }
    }
}

// ─── MdnsData ────────────────────────────────────────────────────────────────

/// Typed record data for an mDNS record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MdnsData {
    /// IPv4 address (A record).
    A(Ipv4Addr),
    /// IPv6 address (AAAA record).
    Aaaa(Ipv6Addr),
    /// Domain name pointer (PTR record).
    Ptr(String),
    /// Service location (SRV record).
    Srv {
        /// Relative priority (lower = higher priority).
        priority: u16,
        /// Relative weight for load balancing.
        weight: u16,
        /// TCP/UDP port.
        port: u16,
        /// Target host name.
        target: String,
    },
    /// Text strings (TXT record) — each entry is a `key=value` or lone `key`.
    Txt(Vec<String>),
}

// ─── MdnsRecord ──────────────────────────────────────────────────────────────

/// A single mDNS resource record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MdnsRecord {
    /// Owner name (fully-qualified domain name).
    pub name: String,
    /// Record type.
    pub record_type: MdnsType,
    /// Time-to-live in seconds.
    pub ttl: u32,
    /// Record-type-specific data.
    pub data: MdnsData,
}

// ─── ServiceInfo ─────────────────────────────────────────────────────────────

/// Description of a DNS-SD service to announce or browse.
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    /// Service type (e.g. `_http._tcp.local.`).
    pub service_type: String,
    /// Unique instance name (e.g. `My Web Server._http._tcp.local.`).
    pub instance_name: String,
    /// TCP/UDP port the service listens on.
    pub port: u16,
    /// Hostname of the machine offering the service (e.g. `myhost.local.`).
    pub host: String,
    /// TXT record key/value pairs.
    pub txt_records: Vec<String>,
    /// Optional IPv4 address to include in an A record.
    pub ipv4: Option<Ipv4Addr>,
}

// ─── MdnsAnnouncer ───────────────────────────────────────────────────────────

/// Builds mDNS announcement (DNS response) packets.
pub struct MdnsAnnouncer;

impl MdnsAnnouncer {
    /// Build a multicast DNS response packet that announces `service`.
    ///
    /// The packet contains:
    /// - PTR record: service_type → instance_name
    /// - SRV record: instance_name → host:port
    /// - TXT record: instance_name → txt_records
    /// - A record (if `service.ipv4` is `Some`): host → IPv4
    pub fn announce(service: &ServiceInfo) -> Vec<u8> {
        let mut packet = DnsPacketBuilder::new();

        // DNS response header: QR=1 (response), AA=1 (authoritative), ANCOUNT=3/4
        let answer_count: u16 = 3 + u16::from(service.ipv4.is_some());
        packet.write_header(0, 0x8400, 0, answer_count, 0, 0);

        // PTR record
        let ttl_ptr: u32 = 4500; // 75 minutes, typical for DNS-SD
        let ptr_name = ensure_fqdn(&service.service_type);
        let instance_fqdn = ensure_fqdn(&service.instance_name);
        let ptr_rdata = encode_dns_name(&instance_fqdn);
        packet.write_rr(&ptr_name, DNS_TYPE_PTR, DNS_CLASS_IN, ttl_ptr, &ptr_rdata);

        // SRV record
        let ttl_srv: u32 = 120;
        let host_fqdn = ensure_fqdn(&service.host);
        let mut srv_rdata = Vec::with_capacity(6 + host_fqdn.len() + 2);
        srv_rdata.extend_from_slice(&0u16.to_be_bytes()); // priority
        srv_rdata.extend_from_slice(&0u16.to_be_bytes()); // weight
        srv_rdata.extend_from_slice(&service.port.to_be_bytes());
        srv_rdata.extend_from_slice(&encode_dns_name(&host_fqdn));
        packet.write_rr(
            &instance_fqdn,
            DNS_TYPE_SRV,
            DNS_CLASS_IN,
            ttl_srv,
            &srv_rdata,
        );

        // TXT record
        let ttl_txt: u32 = 4500;
        let txt_rdata = encode_txt_records(&service.txt_records);
        packet.write_rr(
            &instance_fqdn,
            DNS_TYPE_TXT,
            DNS_CLASS_IN,
            ttl_txt,
            &txt_rdata,
        );

        // A record (optional)
        if let Some(ipv4) = service.ipv4 {
            let ttl_a: u32 = 120;
            packet.write_rr(&host_fqdn, DNS_TYPE_A, DNS_CLASS_IN, ttl_a, &ipv4.octets());
        }

        packet.into_bytes()
    }
}

// ─── MdnsQuery ───────────────────────────────────────────────────────────────

/// Builds mDNS query packets for DNS-SD browse requests.
pub struct MdnsQuery;

impl MdnsQuery {
    /// Build an mDNS PTR query for `service_type` with the QU (unicast-response)
    /// bit set (RFC 6762 §5.4).
    ///
    /// `service_type` should be a DNS-SD service type such as `_http._tcp.local.`
    pub fn build_query(service_type: &str) -> Vec<u8> {
        let mut packet = DnsPacketBuilder::new();
        // Header: QR=0 (query), QDCOUNT=1
        packet.write_header(0, 0x0000, 1, 0, 0, 0);

        let fqdn = ensure_fqdn(service_type);
        let name_bytes = encode_dns_name(&fqdn);

        // Question section
        packet.extend(&name_bytes);
        packet.write_u16(DNS_TYPE_PTR);
        // class IN with QU bit
        packet.write_u16(DNS_CLASS_IN | DNS_QU_BIT);

        packet.into_bytes()
    }
}

// ─── MdnsParser ──────────────────────────────────────────────────────────────

/// Parses raw DNS/mDNS packets into `MdnsRecord` lists.
pub struct MdnsParser;

impl MdnsParser {
    /// Parse a raw DNS message, returning all resource records found in the
    /// Answer, Authority, and Additional sections.
    pub fn parse(data: &[u8]) -> NetResult<Vec<MdnsRecord>> {
        if data.len() < 12 {
            return Err(NetError::parse(0, "DNS message too short (need ≥12 bytes)"));
        }

        let _id = u16::from_be_bytes([data[0], data[1]]);
        let flags = u16::from_be_bytes([data[2], data[3]]);
        let qd_count = u16::from_be_bytes([data[4], data[5]]) as usize;
        let an_count = u16::from_be_bytes([data[6], data[7]]) as usize;
        let ns_count = u16::from_be_bytes([data[8], data[9]]) as usize;
        let ar_count = u16::from_be_bytes([data[10], data[11]]) as usize;

        let _ = flags; // used for QR bit checking if needed

        let mut offset = 12usize;

        // Skip question section
        for _ in 0..qd_count {
            offset = skip_name(data, offset)?;
            if offset + 4 > data.len() {
                return Err(NetError::parse(
                    offset as u64,
                    "Truncated DNS question section",
                ));
            }
            offset += 4; // QTYPE + QCLASS
        }

        // Parse answer, authority, additional records
        let total_rrs = an_count + ns_count + ar_count;
        let mut records = Vec::with_capacity(total_rrs);

        for _ in 0..total_rrs {
            if offset >= data.len() {
                break;
            }
            match parse_rr(data, offset) {
                Ok((record, next_offset)) => {
                    records.push(record);
                    offset = next_offset;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(records)
    }
}

// ─── Internal packet builder ──────────────────────────────────────────────────

struct DnsPacketBuilder {
    buf: Vec<u8>,
}

impl DnsPacketBuilder {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(512),
        }
    }

    fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn extend(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn write_header(
        &mut self,
        id: u16,
        flags: u16,
        qd_count: u16,
        an_count: u16,
        ns_count: u16,
        ar_count: u16,
    ) {
        self.write_u16(id);
        self.write_u16(flags);
        self.write_u16(qd_count);
        self.write_u16(an_count);
        self.write_u16(ns_count);
        self.write_u16(ar_count);
    }

    fn write_rr(&mut self, name: &str, rtype: u16, rclass: u16, ttl: u32, rdata: &[u8]) {
        let name_bytes = encode_dns_name(name);
        self.extend(&name_bytes);
        self.write_u16(rtype);
        self.write_u16(rclass);
        self.write_u32(ttl);
        self.write_u16(rdata.len() as u16);
        self.extend(rdata);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

// ─── DNS name encoding ────────────────────────────────────────────────────────

/// Encode a domain name as label-encoded bytes (RFC 1035 §3.1).
/// Trailing dot (FQDN) is handled; result ends with a zero-length label.
fn encode_dns_name(name: &str) -> Vec<u8> {
    let name = name.trim_end_matches('.');
    let mut out = Vec::new();
    if name.is_empty() {
        out.push(0u8); // root
        return out;
    }
    for label in name.split('.') {
        let bytes = label.as_bytes();
        out.push(bytes.len() as u8);
        out.extend_from_slice(bytes);
    }
    out.push(0u8); // terminator
    out
}

/// Ensure a DNS name ends with a trailing dot (FQDN).
fn ensure_fqdn(name: &str) -> String {
    if name.ends_with('.') {
        name.to_string()
    } else {
        format!("{name}.")
    }
}

/// Encode TXT record RDATA from a list of strings.
fn encode_txt_records(records: &[String]) -> Vec<u8> {
    if records.is_empty() {
        // Empty TXT — single zero-length string per RFC
        return vec![0x00];
    }
    let mut out = Vec::new();
    for record in records {
        let bytes = record.as_bytes();
        out.push(bytes.len() as u8);
        out.extend_from_slice(bytes);
    }
    out
}

// ─── DNS name decoding ────────────────────────────────────────────────────────

/// Decode a domain name starting at `offset` in `data`, following compression
/// pointers (RFC 1035 §4.1.4).
///
/// Returns (decoded_name, offset_after_name_in_original_buffer).
fn decode_dns_name(data: &[u8], offset: usize) -> NetResult<(String, usize)> {
    let mut labels = Vec::new();
    let mut pos = offset;
    let mut followed_pointer = false;
    let mut end_offset = 0usize;
    let mut hops = 0u32;

    loop {
        if pos >= data.len() {
            return Err(NetError::parse(pos as u64, "DNS name truncated"));
        }

        let byte = data[pos];

        if byte == 0 {
            // End of name
            if !followed_pointer {
                end_offset = pos + 1;
            }
            break;
        } else if byte & 0xC0 == 0xC0 {
            // Compression pointer
            if pos + 1 >= data.len() {
                return Err(NetError::parse(pos as u64, "DNS pointer truncated"));
            }
            if !followed_pointer {
                end_offset = pos + 2;
                followed_pointer = true;
            }
            let ptr = (u16::from(byte & 0x3F) << 8 | u16::from(data[pos + 1])) as usize;
            if ptr >= data.len() {
                return Err(NetError::parse(
                    pos as u64,
                    format!("DNS pointer {ptr} out of bounds (data len {})", data.len()),
                ));
            }
            hops += 1;
            if hops > 128 {
                return Err(NetError::parse(
                    pos as u64,
                    "DNS name pointer loop detected",
                ));
            }
            pos = ptr;
        } else if byte & 0xC0 == 0 {
            // Regular label
            let label_len = byte as usize;
            pos += 1;
            if pos + label_len > data.len() {
                return Err(NetError::parse(pos as u64, "DNS label truncated"));
            }
            let label = std::str::from_utf8(&data[pos..pos + label_len])
                .map_err(|_| NetError::parse(pos as u64, "DNS label is not valid UTF-8"))?;
            labels.push(label.to_string());
            pos += label_len;
        } else {
            return Err(NetError::parse(
                pos as u64,
                format!("Invalid DNS label length byte: {byte:#04x}"),
            ));
        }
    }

    if !followed_pointer {
        end_offset = pos + 1; // skip null terminator
    }

    Ok((labels.join("."), end_offset))
}

/// Skip over a DNS name (following pointers) and return the offset after it.
fn skip_name(data: &[u8], offset: usize) -> NetResult<usize> {
    let (_, end) = decode_dns_name(data, offset)?;
    Ok(end)
}

// ─── RR parsing ──────────────────────────────────────────────────────────────

/// Parse a single resource record at `offset`.
///
/// Returns `(record, offset_after_rr)`.
fn parse_rr(data: &[u8], offset: usize) -> NetResult<(MdnsRecord, usize)> {
    let (name, mut pos) = decode_dns_name(data, offset)?;

    if pos + 10 > data.len() {
        return Err(NetError::parse(pos as u64, "RR header truncated"));
    }

    let rtype_raw = u16::from_be_bytes([data[pos], data[pos + 1]]);
    pos += 2;
    let _class = u16::from_be_bytes([data[pos], data[pos + 1]]) & 0x7FFF;
    pos += 2;
    let ttl = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    pos += 4;
    let rdlength = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
    pos += 2;

    if pos + rdlength > data.len() {
        return Err(NetError::parse(
            pos as u64,
            format!("RDATA truncated: need {rdlength} bytes at offset {pos}"),
        ));
    }

    let rdata = &data[pos..pos + rdlength];
    let rdata_start = pos;
    pos += rdlength;

    let record_type = MdnsType::from_u16(rtype_raw);
    let record_data = match record_type {
        MdnsType::A => {
            if rdata.len() < 4 {
                return Err(NetError::parse(
                    rdata_start as u64,
                    "A record RDATA must be 4 bytes",
                ));
            }
            MdnsData::A(Ipv4Addr::new(rdata[0], rdata[1], rdata[2], rdata[3]))
        }
        MdnsType::Aaaa => {
            if rdata.len() < 16 {
                return Err(NetError::parse(
                    rdata_start as u64,
                    "AAAA record RDATA must be 16 bytes",
                ));
            }
            let bytes: [u8; 16] = rdata[..16]
                .try_into()
                .map_err(|_| NetError::parse(rdata_start as u64, "AAAA RDATA too short"))?;
            MdnsData::Aaaa(Ipv6Addr::from(bytes))
        }
        MdnsType::Ptr => {
            let (ptr_name, _) = decode_dns_name(data, rdata_start)?;
            MdnsData::Ptr(ptr_name)
        }
        MdnsType::Srv => {
            if rdata.len() < 6 {
                return Err(NetError::parse(
                    rdata_start as u64,
                    "SRV record RDATA must be ≥6 bytes",
                ));
            }
            let priority = u16::from_be_bytes([rdata[0], rdata[1]]);
            let weight = u16::from_be_bytes([rdata[2], rdata[3]]);
            let port = u16::from_be_bytes([rdata[4], rdata[5]]);
            let (target, _) = decode_dns_name(data, rdata_start + 6)?;
            MdnsData::Srv {
                priority,
                weight,
                port,
                target,
            }
        }
        MdnsType::Txt => {
            let strings = parse_txt_rdata(rdata)?;
            MdnsData::Txt(strings)
        }
        MdnsType::Unknown(_) => {
            // Store raw bytes as a single opaque TXT-like entry
            MdnsData::Txt(vec![format!("raw:{}", hex_encode(rdata))])
        }
    };

    Ok((
        MdnsRecord {
            name,
            record_type,
            ttl,
            data: record_data,
        },
        pos,
    ))
}

/// Parse TXT RDATA into individual strings.
fn parse_txt_rdata(rdata: &[u8]) -> NetResult<Vec<String>> {
    let mut strings = Vec::new();
    let mut pos = 0;
    while pos < rdata.len() {
        let len = rdata[pos] as usize;
        pos += 1;
        if pos + len > rdata.len() {
            return Err(NetError::parse(pos as u64, "TXT string truncated"));
        }
        let s = std::str::from_utf8(&rdata[pos..pos + len])
            .map_err(|_| NetError::parse(pos as u64, "TXT entry is not valid UTF-8"))?;
        if !s.is_empty() {
            strings.push(s.to_string());
        }
        pos += len;
    }
    Ok(strings)
}

/// Minimal hex encoder for diagnostic purposes.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── encode_dns_name ──────────────────────────────────────────────────────

    #[test]
    fn test_encode_dns_name_simple() {
        let encoded = encode_dns_name("example.com");
        // \x07example\x03com\x00
        assert_eq!(encoded, b"\x07example\x03com\x00");
    }

    #[test]
    fn test_encode_dns_name_fqdn() {
        let a = encode_dns_name("example.com");
        let b = encode_dns_name("example.com.");
        assert_eq!(a, b, "trailing dot should not affect encoding");
    }

    #[test]
    fn test_encode_dns_name_root() {
        let encoded = encode_dns_name(".");
        assert_eq!(encoded, b"\x00");
    }

    // ── decode_dns_name ──────────────────────────────────────────────────────

    #[test]
    fn test_decode_dns_name_simple() {
        let data = b"\x07example\x03com\x00";
        let (name, end) = decode_dns_name(data, 0).expect("ok");
        assert_eq!(name, "example.com");
        assert_eq!(end, data.len());
    }

    #[test]
    fn test_decode_dns_name_with_pointer() {
        // Build: "\x07example\x03com\x00" at offset 0
        //        "\x03www\xC0\x00"        at offset 13
        let mut data = Vec::new();
        data.extend_from_slice(b"\x07example\x03com\x00"); // offset 0..13
        data.extend_from_slice(b"\x03www\xC0\x00"); // offset 13..20, pointer → 0
        let (name, end) = decode_dns_name(&data, 13).expect("ok");
        assert_eq!(name, "www.example.com");
        // \x03www is 4 bytes, \xC0\x00 is 2 bytes → 6 bytes consumed before pointer target
        assert_eq!(end, 13 + 6);
    }

    #[test]
    fn test_decode_dns_name_empty_label() {
        // Single null byte = root/empty
        let data = b"\x00";
        let (name, end) = decode_dns_name(data, 0).expect("ok");
        assert_eq!(name, "");
        assert_eq!(end, 1);
    }

    #[test]
    fn test_decode_dns_name_pointer_loop_error() {
        // Pointer at offset 0 pointing to itself
        let data = [0xC0u8, 0x00];
        let result = decode_dns_name(&data, 0);
        assert!(result.is_err());
    }

    // ── MdnsQuery ────────────────────────────────────────────────────────────

    #[test]
    fn test_build_query_parses_back() {
        let query = MdnsQuery::build_query("_http._tcp.local");
        // Header (12) + name + 2(type) + 2(class)
        assert!(query.len() > 12);

        // Verify header: QR=0, QDCOUNT=1
        let flags = u16::from_be_bytes([query[2], query[3]]);
        assert_eq!(flags & 0x8000, 0, "QR must be 0 for query");
        let qd_count = u16::from_be_bytes([query[4], query[5]]);
        assert_eq!(qd_count, 1);
    }

    #[test]
    fn test_build_query_qu_bit_set() {
        let query = MdnsQuery::build_query("_http._tcp.local.");
        // Find the QCLASS bytes after the question name
        // Parse the name to find where QTYPE+QCLASS are
        let (_, name_end) = decode_dns_name(&query, 12).expect("ok");
        let qtype = u16::from_be_bytes([query[name_end], query[name_end + 1]]);
        let qclass = u16::from_be_bytes([query[name_end + 2], query[name_end + 3]]);
        assert_eq!(qtype, DNS_TYPE_PTR, "PTR query type expected");
        assert_eq!(qclass & DNS_QU_BIT, DNS_QU_BIT, "QU bit must be set");
    }

    // ── MdnsAnnouncer ────────────────────────────────────────────────────────

    #[test]
    fn test_announce_produces_response_flag() {
        let service = ServiceInfo {
            service_type: "_http._tcp.local.".to_string(),
            instance_name: "Test Server._http._tcp.local.".to_string(),
            port: 8080,
            host: "testhost.local.".to_string(),
            txt_records: vec!["path=/".to_string()],
            ipv4: Some(Ipv4Addr::new(192, 168, 1, 1)),
        };
        let packet = MdnsAnnouncer::announce(&service);
        assert!(packet.len() > 12);

        // QR=1 (response), AA=1 (authoritative)
        let flags = u16::from_be_bytes([packet[2], packet[3]]);
        assert_eq!(flags & 0x8000, 0x8000, "QR bit must be set (response)");
        assert_eq!(flags & 0x0400, 0x0400, "AA bit must be set");
    }

    #[test]
    fn test_announce_answer_count_with_ipv4() {
        let service = ServiceInfo {
            service_type: "_http._tcp.local.".to_string(),
            instance_name: "My._http._tcp.local.".to_string(),
            port: 80,
            host: "myhost.local.".to_string(),
            txt_records: vec![],
            ipv4: Some(Ipv4Addr::new(10, 0, 0, 1)),
        };
        let packet = MdnsAnnouncer::announce(&service);
        let an_count = u16::from_be_bytes([packet[6], packet[7]]);
        assert_eq!(an_count, 4, "PTR + SRV + TXT + A = 4");
    }

    #[test]
    fn test_announce_answer_count_without_ipv4() {
        let service = ServiceInfo {
            service_type: "_http._tcp.local.".to_string(),
            instance_name: "My._http._tcp.local.".to_string(),
            port: 80,
            host: "myhost.local.".to_string(),
            txt_records: vec![],
            ipv4: None,
        };
        let packet = MdnsAnnouncer::announce(&service);
        let an_count = u16::from_be_bytes([packet[6], packet[7]]);
        assert_eq!(an_count, 3, "PTR + SRV + TXT = 3");
    }

    // ── MdnsParser ───────────────────────────────────────────────────────────

    #[test]
    fn test_parse_announced_packet() {
        let service = ServiceInfo {
            service_type: "_http._tcp.local.".to_string(),
            instance_name: "WebApp._http._tcp.local.".to_string(),
            port: 3000,
            host: "webapp.local.".to_string(),
            txt_records: vec!["version=1.0".to_string(), "secure=no".to_string()],
            ipv4: Some(Ipv4Addr::new(192, 168, 100, 5)),
        };
        let packet = MdnsAnnouncer::announce(&service);
        let records = MdnsParser::parse(&packet).expect("parse ok");

        assert_eq!(records.len(), 4, "expect PTR + SRV + TXT + A");

        // PTR record
        let ptr = records.iter().find(|r| r.record_type == MdnsType::Ptr);
        assert!(ptr.is_some(), "PTR record expected");
        if let Some(rec) = ptr {
            assert_eq!(rec.name, "_http._tcp.local");
        }

        // A record
        let a_rec = records.iter().find(|r| r.record_type == MdnsType::A);
        assert!(a_rec.is_some(), "A record expected");
        if let Some(rec) = a_rec {
            assert_eq!(rec.data, MdnsData::A(Ipv4Addr::new(192, 168, 100, 5)));
        }
    }

    #[test]
    fn test_parse_srv_record_fields() {
        let service = ServiceInfo {
            service_type: "_myapp._tcp.local.".to_string(),
            instance_name: "MyApp._myapp._tcp.local.".to_string(),
            port: 9000,
            host: "serverbox.local.".to_string(),
            txt_records: vec![],
            ipv4: None,
        };
        let packet = MdnsAnnouncer::announce(&service);
        let records = MdnsParser::parse(&packet).expect("parse ok");

        let srv = records.iter().find(|r| r.record_type == MdnsType::Srv);
        assert!(srv.is_some());
        if let Some(rec) = srv {
            if let MdnsData::Srv { port, .. } = rec.data {
                assert_eq!(port, 9000);
            } else {
                panic!("Expected SRV data");
            }
        }
    }

    #[test]
    fn test_parse_txt_record_values() {
        let service = ServiceInfo {
            service_type: "_chat._tcp.local.".to_string(),
            instance_name: "ChatApp._chat._tcp.local.".to_string(),
            port: 5222,
            host: "chatserver.local.".to_string(),
            txt_records: vec!["user=alice".to_string(), "room=main".to_string()],
            ipv4: None,
        };
        let packet = MdnsAnnouncer::announce(&service);
        let records = MdnsParser::parse(&packet).expect("parse ok");

        let txt = records.iter().find(|r| r.record_type == MdnsType::Txt);
        assert!(txt.is_some());
        if let Some(rec) = txt {
            if let MdnsData::Txt(ref strings) = rec.data {
                assert!(strings.contains(&"user=alice".to_string()));
                assert!(strings.contains(&"room=main".to_string()));
            } else {
                panic!("Expected TXT data");
            }
        }
    }

    #[test]
    fn test_parse_error_too_short() {
        let result = MdnsParser::parse(&[0u8; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_minimal_response_no_records() {
        // Valid DNS header with zero counts
        let header = [0u8, 1, 0x84, 0x00, 0, 0, 0, 0, 0, 0, 0, 0];
        let records = MdnsParser::parse(&header).expect("ok");
        assert!(records.is_empty());
    }

    #[test]
    fn test_ensure_fqdn() {
        assert_eq!(ensure_fqdn("example.com"), "example.com.");
        assert_eq!(ensure_fqdn("example.com."), "example.com.");
    }

    #[test]
    fn test_encode_txt_empty() {
        let encoded = encode_txt_records(&[]);
        assert_eq!(encoded, vec![0x00]);
    }

    #[test]
    fn test_encode_decode_txt_roundtrip() {
        let strings = vec!["key=value".to_string(), "flag".to_string()];
        let encoded = encode_txt_records(&strings);
        let decoded = parse_txt_rdata(&encoded).expect("ok");
        assert_eq!(decoded, strings);
    }
}
