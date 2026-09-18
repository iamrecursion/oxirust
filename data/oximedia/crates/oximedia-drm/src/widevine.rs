//! Widevine DRM implementation
//!
//! Supports Widevine L3 (software-based) DRM integration.
//! Note: This is a partial implementation for educational purposes.
//! Full Widevine integration requires licensed CDM libraries.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Widevine System ID constant (edef8ba9-79d6-4ace-a3c8-27dcd51d21ed)
// ──────────────────────────────────────────────────────────────────────────────

/// Widevine DRM System ID as a 16-byte array (MPEG CENC / ISOBMFF PSSH).
///
/// UUID: `edef8ba9-79d6-4ace-a3c8-27dcd51d21ed`
pub const WIDEVINE_SYSTEM_ID: [u8; 16] = [
    0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21, 0xed,
];

// ──────────────────────────────────────────────────────────────────────────────
// WidevineConfig — high-level configuration for a Widevine-protected asset
// ──────────────────────────────────────────────────────────────────────────────

/// Top-level Widevine configuration that ties together a license server,
/// a content provider, and the content identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineConfig {
    /// Full URL of the Widevine license server endpoint.
    pub license_server_url: String,
    /// Provider string registered with Google's Widevine infrastructure.
    pub provider: String,
    /// Opaque content identifier supplied to the license server.
    pub content_id: Vec<u8>,
}

impl WidevineConfig {
    /// Create a new `WidevineConfig`.
    pub fn new(
        license_server_url: impl Into<String>,
        provider: impl Into<String>,
        content_id: Vec<u8>,
    ) -> Self {
        Self {
            license_server_url: license_server_url.into(),
            provider: provider.into(),
            content_id,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WidevinePsshBox — binary PSSH box with Widevine-specific payload
// ──────────────────────────────────────────────────────────────────────────────

/// Protection scheme fourcc values used in the PSSH payload.
pub mod protection_scheme {
    /// Common Encryption CTR mode (`cenc`).
    pub const CENC: u32 = 0x63656e63;
    /// Common Encryption CBC mode (`cbcs`).
    pub const CBCS: u32 = 0x63626373;
}

/// Widevine-specific PSSH payload as defined by the Widevine DASH/HLS
/// content protection specification.
///
/// This struct provides a **binary-serialisable** view complementing the
/// existing JSON-oriented [`WidevinePsshData`] type.  The serialisation
/// format matches the PSSH v0/v1 box payload expected by Widevine CDMs:
///
/// ```text
/// 4 bytes  : algorithm  (0 = UNENCRYPTED, 1 = AESCTR)
/// 4 bytes  : key_id_count
/// N × 16 bytes : key IDs
/// 4 bytes  : provider length
/// N bytes  : provider UTF-8
/// 4 bytes  : content_id length
/// N bytes  : content_id bytes
/// 4 bytes  : protection_scheme fourcc
/// ```
///
/// All multi-byte integers are **big-endian** to match the ISO-BMFF box
/// convention.
#[derive(Debug, Clone)]
pub struct WidevinePsshBox {
    /// 16-byte key IDs (each key ID must be exactly 16 bytes).
    pub key_ids: Vec<Vec<u8>>,
    /// Opaque content identifier.
    pub content_id: Vec<u8>,
    /// Provider name registered with Widevine.
    pub provider: String,
    /// Protection scheme fourcc — use [`protection_scheme::CENC`] or
    /// [`protection_scheme::CBCS`].
    pub protection_scheme: u32,
}

impl WidevinePsshBox {
    /// Construct a new `WidevinePsshBox` with CENC as the default scheme.
    ///
    /// Returns an error if any key ID is not exactly 16 bytes long.
    pub fn new(key_ids: Vec<Vec<u8>>, provider: &str) -> Result<Self> {
        for (i, kid) in key_ids.iter().enumerate() {
            if kid.len() != 16 {
                return Err(DrmError::InvalidKey(format!(
                    "key_id[{}] must be 16 bytes, got {}",
                    i,
                    kid.len()
                )));
            }
        }
        Ok(Self {
            key_ids,
            content_id: Vec::new(),
            provider: provider.to_string(),
            protection_scheme: protection_scheme::CENC,
        })
    }

    /// Set the content identifier.
    pub fn with_content_id(mut self, content_id: Vec<u8>) -> Self {
        self.content_id = content_id;
        self
    }

    /// Set the protection scheme (e.g. [`protection_scheme::CBCS`]).
    pub fn with_protection_scheme(mut self, scheme: u32) -> Self {
        self.protection_scheme = scheme;
        self
    }

    /// Serialise the Widevine PSSH payload to bytes (big-endian, ISO-BMFF
    /// PSSH box data field).
    ///
    /// The returned bytes do **not** include the outer 8-byte box header
    /// (`size` + `'pssh'`); call [`Self::to_pssh_box`] for the complete
    /// ISOBMFF box.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // algorithm = 1 (AESCTR)
        out.extend_from_slice(&1u32.to_be_bytes());

        // key_id_count
        out.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());

        // key IDs (each 16 bytes)
        for kid in &self.key_ids {
            out.extend_from_slice(kid);
        }

        // provider (length-prefixed UTF-8)
        let provider_bytes = self.provider.as_bytes();
        out.extend_from_slice(&(provider_bytes.len() as u32).to_be_bytes());
        out.extend_from_slice(provider_bytes);

        // content_id (length-prefixed)
        out.extend_from_slice(&(self.content_id.len() as u32).to_be_bytes());
        out.extend_from_slice(&self.content_id);

        // protection_scheme fourcc
        out.extend_from_slice(&self.protection_scheme.to_be_bytes());

        out
    }

    /// Parse a Widevine PSSH payload byte buffer (the `data` field of a
    /// PSSH box after the system-ID has been verified).
    ///
    /// Returns [`DrmError::PsshError`] on any structural mismatch.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut pos = 0usize;

        let read_u32 = |buf: &[u8], p: &mut usize| -> Result<u32> {
            if *p + 4 > buf.len() {
                return Err(DrmError::PsshError(
                    "unexpected end of PSSH data".to_string(),
                ));
            }
            let v = u32::from_be_bytes([buf[*p], buf[*p + 1], buf[*p + 2], buf[*p + 3]]);
            *p += 4;
            Ok(v)
        };

        let read_bytes = |buf: &[u8], p: &mut usize, len: usize| -> Result<Vec<u8>> {
            if *p + len > buf.len() {
                return Err(DrmError::PsshError(format!(
                    "not enough bytes: need {}, have {}",
                    len,
                    buf.len() - *p
                )));
            }
            let v = buf[*p..*p + len].to_vec();
            *p += len;
            Ok(v)
        };

        // algorithm (ignored — must be present)
        let _algorithm = read_u32(data, &mut pos)?;

        // key IDs
        let key_id_count = read_u32(data, &mut pos)? as usize;
        let mut key_ids = Vec::with_capacity(key_id_count);
        for _ in 0..key_id_count {
            let kid = read_bytes(data, &mut pos, 16)?;
            key_ids.push(kid);
        }

        // provider
        let provider_len = read_u32(data, &mut pos)? as usize;
        let provider_bytes = read_bytes(data, &mut pos, provider_len)?;
        let provider = String::from_utf8(provider_bytes)
            .map_err(|e| DrmError::PsshError(format!("provider is not valid UTF-8: {e}")))?;

        // content_id
        let content_id_len = read_u32(data, &mut pos)? as usize;
        let content_id = read_bytes(data, &mut pos, content_id_len)?;

        // protection_scheme (optional; default to CENC if truncated)
        let protection_scheme = if pos + 4 <= data.len() {
            read_u32(data, &mut pos)?
        } else {
            protection_scheme::CENC
        };

        Ok(Self {
            key_ids,
            content_id,
            provider,
            protection_scheme,
        })
    }

    /// Wrap the serialised payload in a complete ISO-BMFF PSSH box (version 0).
    ///
    /// Layout: `[size:4][pssh:4][version:1][flags:3][system_id:16][data_len:4][data:N]`
    pub fn to_pssh_box(&self) -> Vec<u8> {
        let payload = self.serialize();
        let total_size: u32 = (4 + 4 + 1 + 3 + 16 + 4 + payload.len()) as u32;

        let mut box_bytes = Vec::with_capacity(total_size as usize);
        box_bytes.extend_from_slice(&total_size.to_be_bytes()); // box size
        box_bytes.extend_from_slice(b"pssh"); // box type
        box_bytes.push(0u8); // version = 0
        box_bytes.extend_from_slice(&[0u8; 3]); // flags
        box_bytes.extend_from_slice(&WIDEVINE_SYSTEM_ID); // system_id
        box_bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        box_bytes.extend_from_slice(&payload);
        box_bytes
    }
}

/// Widevine PSSH data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevinePsshData {
    /// Algorithm (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    /// Key IDs
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub key_ids: Vec<Vec<u8>>,
    /// Provider (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Content ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_id: Option<Vec<u8>>,
    /// Track type (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_type: Option<String>,
    /// Policy (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    /// Crypto period index (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crypto_period_index: Option<u32>,
    /// Protection scheme (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protection_scheme: Option<String>,
}

impl WidevinePsshData {
    /// Create new Widevine PSSH data
    pub fn new() -> Self {
        Self {
            algorithm: None,
            key_ids: Vec::new(),
            provider: None,
            content_id: None,
            track_type: None,
            policy: None,
            crypto_period_index: None,
            protection_scheme: Some("cenc".to_string()),
        }
    }

    /// Add a key ID
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.key_ids.push(key_id);
    }

    /// Set provider
    pub fn with_provider(mut self, provider: String) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set content ID
    pub fn with_content_id(mut self, content_id: Vec<u8>) -> Self {
        self.content_id = Some(content_id);
        self
    }

    /// Set track type
    pub fn with_track_type(mut self, track_type: String) -> Self {
        self.track_type = Some(track_type);
        self
    }

    /// Set protection scheme
    pub fn with_protection_scheme(mut self, scheme: String) -> Self {
        self.protection_scheme = Some(scheme);
        self
    }

    /// Serialize to bytes (simplified - would use protobuf in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from bytes (simplified - would use protobuf in production)
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

impl Default for WidevinePsshData {
    fn default() -> Self {
        Self::new()
    }
}

/// Widevine license request type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseType {
    /// Streaming license (temporary)
    Streaming,
    /// Offline license (persistent)
    Offline,
    /// License renewal
    Renewal,
    /// License release
    Release,
}

impl LicenseType {
    /// Get license type as string
    pub fn as_str(&self) -> &'static str {
        match self {
            LicenseType::Streaming => "STREAMING",
            LicenseType::Offline => "OFFLINE",
            LicenseType::Renewal => "RENEWAL",
            LicenseType::Release => "RELEASE",
        }
    }
}

/// Widevine license request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineLicenseRequest {
    /// License type
    pub license_type: LicenseType,
    /// Content ID
    pub content_id: Vec<u8>,
    /// Key IDs being requested
    pub key_ids: Vec<Vec<u8>>,
    /// Client ID (optional)
    pub client_id: Option<Vec<u8>>,
    /// Request ID (optional)
    pub request_id: Option<Vec<u8>>,
    /// Session ID (optional)
    pub session_id: Option<Vec<u8>>,
}

impl WidevineLicenseRequest {
    /// Create a new Widevine license request
    pub fn new(license_type: LicenseType, content_id: Vec<u8>, key_ids: Vec<Vec<u8>>) -> Self {
        Self {
            license_type,
            content_id,
            key_ids,
            client_id: None,
            request_id: None,
            session_id: None,
        }
    }

    /// Set client ID
    pub fn with_client_id(mut self, client_id: Vec<u8>) -> Self {
        self.client_id = Some(client_id);
        self
    }

    /// Serialize to bytes (simplified - would use protobuf in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Serialize to base64
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(STANDARD.encode(&bytes))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

/// Widevine key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineKey {
    /// Key ID
    pub key_id: Vec<u8>,
    /// Key value (encrypted)
    pub key: Vec<u8>,
    /// Key type (optional)
    pub key_type: Option<String>,
}

impl WidevineKey {
    /// Create a new Widevine key
    pub fn new(key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            key_id,
            key,
            key_type: Some("CONTENT".to_string()),
        }
    }

    /// Set key type
    pub fn with_key_type(mut self, key_type: String) -> Self {
        self.key_type = Some(key_type);
        self
    }
}

/// Widevine license response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineLicenseResponse {
    /// Status (e.g., "OK")
    pub status: String,
    /// License duration in seconds (optional)
    pub license_duration: Option<u64>,
    /// Playback duration in seconds (optional)
    pub playback_duration: Option<u64>,
    /// Renewal server URL (optional)
    pub renewal_server_url: Option<String>,
    /// Keys
    pub keys: Vec<WidevineKey>,
}

impl WidevineLicenseResponse {
    /// Create a new Widevine license response
    pub fn new(keys: Vec<WidevineKey>) -> Self {
        Self {
            status: "OK".to_string(),
            license_duration: None,
            playback_duration: None,
            renewal_server_url: None,
            keys,
        }
    }

    /// Set license duration
    pub fn with_license_duration(mut self, duration: u64) -> Self {
        self.license_duration = Some(duration);
        self
    }

    /// Set playback duration
    pub fn with_playback_duration(mut self, duration: u64) -> Self {
        self.playback_duration = Some(duration);
        self
    }

    /// Serialize to bytes (simplified - would use protobuf in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Serialize to base64
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(STANDARD.encode(&bytes))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }

    /// Get keys as a map
    pub fn get_keys_map(&self) -> HashMap<Vec<u8>, Vec<u8>> {
        self.keys
            .iter()
            .map(|k| (k.key_id.clone(), k.key.clone()))
            .collect()
    }
}

/// Widevine key hierarchy levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLevel {
    /// Root key
    Root,
    /// Intermediate key
    Intermediate,
    /// Content key
    Content,
}

/// Widevine CDM (Content Decryption Module) client (simplified)
pub struct WidevineCdm {
    /// Client ID
    client_id: Vec<u8>,
    /// Session keys
    sessions: HashMap<Vec<u8>, HashMap<Vec<u8>, Vec<u8>>>,
}

impl WidevineCdm {
    /// Create a new Widevine CDM
    pub fn new(client_id: Vec<u8>) -> Self {
        Self {
            client_id,
            sessions: HashMap::new(),
        }
    }

    /// Generate a license request
    pub fn generate_request(
        &self,
        license_type: LicenseType,
        content_id: Vec<u8>,
        key_ids: Vec<Vec<u8>>,
    ) -> Result<WidevineLicenseRequest> {
        let request = WidevineLicenseRequest::new(license_type, content_id, key_ids)
            .with_client_id(self.client_id.clone());

        Ok(request)
    }

    /// Process a license response
    pub fn process_response(
        &mut self,
        session_id: Vec<u8>,
        response: &WidevineLicenseResponse,
    ) -> Result<()> {
        if response.status != "OK" {
            return Err(DrmError::LicenseError(format!(
                "License error: {}",
                response.status
            )));
        }

        // Store keys for this session
        let session_keys: HashMap<Vec<u8>, Vec<u8>> = response.get_keys_map();
        self.sessions.insert(session_id, session_keys);

        Ok(())
    }

    /// Get a content key for a session
    pub fn get_key(&self, session_id: &[u8], key_id: &[u8]) -> Option<Vec<u8>> {
        self.sessions
            .get(session_id)
            .and_then(|keys| keys.get(key_id).cloned())
    }

    /// Close a session and remove its keys
    pub fn close_session(&mut self, session_id: &[u8]) {
        self.sessions.remove(session_id);
    }

    /// Get number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Return a clone of the keys stored for a particular session, or an
    /// empty map if the session does not exist.
    pub fn session_keys(&self, session_id: &[u8]) -> HashMap<Vec<u8>, Vec<u8>> {
        self.sessions.get(session_id).cloned().unwrap_or_default()
    }

    /// Drive a complete license acquisition round-trip against a remote
    /// Widevine license server.
    ///
    /// The flow is:
    ///
    /// 1. Build a [`WidevineLicenseRequest`] via
    ///    [`Self::generate_request`].
    /// 2. Set the session id on the request so the server can correlate the
    ///    response.
    /// 3. Serialise the request to bytes.
    /// 4. POST those bytes to `server_url` via the supplied
    ///    [`crate::widevine_rpc::LicenseClient`].
    /// 5. Parse the response and register the returned keys against the
    ///    given `session_id` via [`Self::process_response`].
    /// 6. Return the resulting key map (`key_id → key`) for the session.
    ///
    /// The caller chooses the [`LicenseType`] explicitly because streaming,
    /// offline, renewal, and release flows differ in policy: callers should
    /// not silently default this.
    ///
    /// # Errors
    ///
    /// * [`crate::DrmError::NetworkError`] if the transport fails (DNS, TCP,
    ///   TLS, HTTP parsing).
    /// * [`crate::DrmError::LicenseDenied`] if the server returns a non-2xx
    ///   HTTP status.
    /// * [`crate::DrmError::LicenseError`] / [`crate::DrmError::JsonError`]
    ///   if the response body is structurally valid HTTP but contains a
    ///   negative license status or malformed license payload.
    pub async fn acquire_license<C>(
        &mut self,
        session_id: Vec<u8>,
        license_type: LicenseType,
        content_id: Vec<u8>,
        key_ids: Vec<Vec<u8>>,
        server_url: &str,
        client: &C,
        extra_headers: &[(String, String)],
    ) -> Result<HashMap<Vec<u8>, Vec<u8>>>
    where
        C: crate::widevine_rpc::LicenseClient + ?Sized,
    {
        let mut request = self.generate_request(license_type, content_id, key_ids)?;
        request.session_id = Some(session_id.clone());

        let body = request.to_bytes()?;
        let response_bytes = client
            .fetch_license(server_url, &body, extra_headers)
            .await?;

        let response = WidevineLicenseResponse::from_bytes(&response_bytes)?;
        self.process_response(session_id.clone(), &response)?;

        Ok(self.session_keys(&session_id))
    }
}

/// Widevine license server (for testing/mocking)
pub struct WidevineLicenseServer {
    keys: HashMap<Vec<u8>, HashMap<Vec<u8>, Vec<u8>>>, // content_id -> (key_id -> key)
    license_duration: u64,
    playback_duration: u64,
}

impl WidevineLicenseServer {
    /// Create a new Widevine license server
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            license_duration: 86400, // 24 hours
            playback_duration: 7200, // 2 hours
        }
    }

    /// Add keys for a content ID
    pub fn add_content_keys(&mut self, content_id: Vec<u8>, keys: HashMap<Vec<u8>, Vec<u8>>) {
        self.keys.insert(content_id, keys);
    }

    /// Add a single key for a content ID
    pub fn add_key(&mut self, content_id: Vec<u8>, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys
            .entry(content_id)
            .or_insert_with(HashMap::new)
            .insert(key_id, key);
    }

    /// Set license duration
    pub fn set_license_duration(&mut self, duration: u64) {
        self.license_duration = duration;
    }

    /// Set playback duration
    pub fn set_playback_duration(&mut self, duration: u64) {
        self.playback_duration = duration;
    }

    /// Process a license request
    pub fn process_request(
        &self,
        request: &WidevineLicenseRequest,
    ) -> Result<WidevineLicenseResponse> {
        // Get keys for this content
        let content_keys = self.keys.get(&request.content_id).ok_or_else(|| {
            DrmError::LicenseError(format!(
                "Content not found: {}",
                hex::encode(&request.content_id)
            ))
        })?;

        // Collect requested keys
        let mut response_keys = Vec::new();
        for key_id in &request.key_ids {
            let key = content_keys.get(key_id).ok_or_else(|| {
                DrmError::LicenseError(format!("Key not found: {}", hex::encode(key_id)))
            })?;

            response_keys.push(WidevineKey::new(key_id.clone(), key.clone()));
        }

        Ok(WidevineLicenseResponse::new(response_keys)
            .with_license_duration(self.license_duration)
            .with_playback_duration(self.playback_duration))
    }

    /// Get number of content entries
    pub fn content_count(&self) -> usize {
        self.keys.len()
    }
}

impl Default for WidevineLicenseServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a Widevine PSSH box
pub fn create_widevine_pssh(key_ids: Vec<Vec<u8>>, content_id: Vec<u8>) -> Result<Vec<u8>> {
    use crate::cenc::PsshBox;

    let mut pssh_data = WidevinePsshData::new();
    for key_id in &key_ids {
        pssh_data.add_key_id(key_id.clone());
    }
    pssh_data = pssh_data.with_content_id(content_id);

    let data = pssh_data.to_bytes()?;
    let pssh = PsshBox::new_v1(DrmSystem::Widevine.system_id(), key_ids, data);
    pssh.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widevine_pssh_data() {
        let mut pssh_data = WidevinePsshData::new();
        pssh_data.add_key_id(vec![1, 2, 3, 4]);
        pssh_data.add_key_id(vec![5, 6, 7, 8]);

        let bytes = pssh_data.to_bytes().expect("operation should succeed");
        let parsed = WidevinePsshData::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(parsed.key_ids.len(), 2);
        assert_eq!(parsed.key_ids[0], vec![1, 2, 3, 4]);
        assert_eq!(parsed.key_ids[1], vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_license_type() {
        assert_eq!(LicenseType::Streaming.as_str(), "STREAMING");
        assert_eq!(LicenseType::Offline.as_str(), "OFFLINE");
        assert_eq!(LicenseType::Renewal.as_str(), "RENEWAL");
        assert_eq!(LicenseType::Release.as_str(), "RELEASE");
    }

    #[test]
    fn test_license_request() {
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];

        let request = WidevineLicenseRequest::new(
            LicenseType::Streaming,
            content_id.clone(),
            vec![key_id.clone()],
        );

        assert_eq!(request.license_type, LicenseType::Streaming);
        assert_eq!(request.content_id, content_id);
        assert_eq!(request.key_ids[0], key_id);
    }

    #[test]
    fn test_license_request_serialization() {
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];

        let request = WidevineLicenseRequest::new(LicenseType::Streaming, content_id, vec![key_id]);

        let bytes = request.to_bytes().expect("operation should succeed");
        let parsed = WidevineLicenseRequest::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(parsed.license_type, request.license_type);
        assert_eq!(parsed.content_id, request.content_id);
    }

    #[test]
    fn test_license_response() {
        let key_id = vec![1, 2, 3, 4];
        let key = vec![5, 6, 7, 8];

        let widevine_key = WidevineKey::new(key_id.clone(), key.clone());
        let response = WidevineLicenseResponse::new(vec![widevine_key])
            .with_license_duration(3600)
            .with_playback_duration(1800);

        assert_eq!(response.status, "OK");
        assert_eq!(response.license_duration, Some(3600));
        assert_eq!(response.playback_duration, Some(1800));
        assert_eq!(response.keys.len(), 1);

        let keys_map = response.get_keys_map();
        assert_eq!(keys_map.get(&key_id), Some(&key));
    }

    #[test]
    fn test_widevine_cdm() {
        let client_id = vec![1, 2, 3, 4];
        let mut cdm = WidevineCdm::new(client_id.clone());

        let content_id = vec![5, 6, 7, 8];
        let key_id = vec![9, 10, 11, 12];
        let key = vec![13, 14, 15, 16];

        let request = cdm
            .generate_request(LicenseType::Streaming, content_id, vec![key_id.clone()])
            .expect("operation should succeed");

        assert_eq!(request.client_id, Some(client_id));

        let widevine_key = WidevineKey::new(key_id.clone(), key.clone());
        let response = WidevineLicenseResponse::new(vec![widevine_key]);

        let session_id = vec![20, 21, 22, 23];
        cdm.process_response(session_id.clone(), &response)
            .expect("operation should succeed");

        assert_eq!(cdm.get_key(&session_id, &key_id), Some(key));
        assert_eq!(cdm.session_count(), 1);

        cdm.close_session(&session_id);
        assert_eq!(cdm.session_count(), 0);
    }

    #[test]
    fn test_license_server() {
        let mut server = WidevineLicenseServer::new();
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];
        let key = vec![9, 10, 11, 12];

        server.add_key(content_id.clone(), key_id.clone(), key.clone());

        let request =
            WidevineLicenseRequest::new(LicenseType::Streaming, content_id, vec![key_id.clone()]);

        let response = server
            .process_request(&request)
            .expect("operation should succeed");

        assert_eq!(response.status, "OK");
        assert_eq!(response.keys.len(), 1);
        assert_eq!(response.keys[0].key_id, key_id);
        assert_eq!(response.keys[0].key, key);
    }

    #[test]
    fn test_license_server_missing_content() {
        let server = WidevineLicenseServer::new();
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];

        let request = WidevineLicenseRequest::new(LicenseType::Streaming, content_id, vec![key_id]);

        let result = server.process_request(&request);
        assert!(result.is_err());
    }

    // ── Tests for new WidevineConfig / WidevinePsshBox / WIDEVINE_SYSTEM_ID ──

    #[test]
    fn test_widevine_system_id_bytes() {
        assert_eq!(WIDEVINE_SYSTEM_ID[0], 0xed);
        assert_eq!(WIDEVINE_SYSTEM_ID[15], 0xed);
        assert_eq!(WIDEVINE_SYSTEM_ID.len(), 16);
    }

    #[test]
    fn test_widevine_config_construction() {
        let cfg = WidevineConfig::new(
            "https://license.example.com/widevine",
            "example_provider",
            b"content-abc".to_vec(),
        );
        assert_eq!(cfg.provider, "example_provider");
        assert_eq!(cfg.content_id, b"content-abc".to_vec());
        assert!(cfg.license_server_url.contains("widevine"));
    }

    #[test]
    fn test_widevine_pssh_box_serialize_parse_roundtrip() {
        let key_ids: Vec<Vec<u8>> = vec![(0u8..16).collect(), (16u8..32).collect()];
        let pssh_box = WidevinePsshBox::new(key_ids.clone(), "test_provider")
            .expect("valid key_ids")
            .with_content_id(b"my-content".to_vec())
            .with_protection_scheme(protection_scheme::CENC);

        let serialized = pssh_box.serialize();
        let parsed = WidevinePsshBox::parse(&serialized).expect("parse should succeed");

        assert_eq!(parsed.key_ids.len(), 2);
        assert_eq!(parsed.key_ids[0], key_ids[0]);
        assert_eq!(parsed.key_ids[1], key_ids[1]);
        assert_eq!(parsed.provider, "test_provider");
        assert_eq!(parsed.content_id, b"my-content".to_vec());
        assert_eq!(parsed.protection_scheme, protection_scheme::CENC);
    }

    #[test]
    fn test_widevine_pssh_box_rejects_bad_key_id() {
        let bad_key_ids = vec![vec![0u8; 8]]; // only 8 bytes
        let result = WidevinePsshBox::new(bad_key_ids, "provider");
        assert!(result.is_err(), "should reject 8-byte key_id");
    }

    #[test]
    fn test_widevine_pssh_box_to_pssh_box_header() {
        let key_ids: Vec<Vec<u8>> = vec![(0u8..16).collect()];
        let pssh_box = WidevinePsshBox::new(key_ids, "prov")
            .expect("valid key_id")
            .to_pssh_box();

        assert!(pssh_box.len() >= 28, "PSSH box must be at least 28 bytes");
        assert_eq!(&pssh_box[4..8], b"pssh", "box type must be 'pssh'");
        assert_eq!(pssh_box[8], 0, "PSSH version field must be 0");
        assert_eq!(&pssh_box[12..28], &WIDEVINE_SYSTEM_ID, "system_id mismatch");

        // Verify declared box size matches actual buffer length
        let declared_size =
            u32::from_be_bytes([pssh_box[0], pssh_box[1], pssh_box[2], pssh_box[3]]) as usize;
        assert_eq!(
            declared_size,
            pssh_box.len(),
            "declared box size must match buffer length"
        );
    }

    #[test]
    fn test_widevine_pssh_box_cbcs_scheme() {
        let key_ids: Vec<Vec<u8>> = vec![(0u8..16).collect()];
        let pssh_box = WidevinePsshBox::new(key_ids, "prov")
            .expect("valid key_id")
            .with_protection_scheme(protection_scheme::CBCS);

        let serialized = pssh_box.serialize();
        let parsed = WidevinePsshBox::parse(&serialized).expect("parse should succeed");
        assert_eq!(parsed.protection_scheme, protection_scheme::CBCS);
    }
}
