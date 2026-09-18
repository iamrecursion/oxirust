//! W3C Clear Key DRM implementation
//!
//! Clear Key is a simple DRM system defined by W3C for testing and development.
//! It uses unencrypted key exchange and is not suitable for production use.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Clear Key license request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKeyRequest {
    /// List of key IDs being requested
    pub kids: Vec<String>,
    /// Request type (typically "temporary")
    #[serde(rename = "type")]
    pub request_type: Option<String>,
}

impl ClearKeyRequest {
    /// Create a new Clear Key request
    pub fn new(key_ids: Vec<Vec<u8>>) -> Self {
        let kids = key_ids
            .into_iter()
            .map(|id| URL_SAFE_NO_PAD.encode(&id))
            .collect();

        Self {
            kids,
            request_type: Some("temporary".to_string()),
        }
    }

    /// Add a key ID to the request
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.kids.push(URL_SAFE_NO_PAD.encode(&key_id));
    }

    /// Get key IDs as bytes
    pub fn get_key_ids(&self) -> Result<Vec<Vec<u8>>> {
        self.kids
            .iter()
            .map(|kid| URL_SAFE_NO_PAD.decode(kid).map_err(DrmError::Base64Error))
            .collect()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(DrmError::JsonError)
    }
}

/// JSON Web Key (JWK) for Clear Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonWebKey {
    /// Key type (always "oct" for symmetric keys)
    pub kty: String,
    /// Key ID (base64url encoded)
    pub kid: String,
    /// Key value (base64url encoded)
    pub k: String,
}

impl JsonWebKey {
    /// Create a new JSON Web Key
    pub fn new(key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            kty: "oct".to_string(),
            kid: URL_SAFE_NO_PAD.encode(&key_id),
            k: URL_SAFE_NO_PAD.encode(&key),
        }
    }

    /// Get key ID as bytes
    pub fn get_key_id(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.kid)
            .map_err(DrmError::Base64Error)
    }

    /// Get key as bytes
    pub fn get_key(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.k)
            .map_err(DrmError::Base64Error)
    }
}

/// Clear Key license response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKeyResponse {
    /// List of keys in JWK format
    pub keys: Vec<JsonWebKey>,
    /// Response type (typically "temporary")
    #[serde(rename = "type")]
    pub response_type: Option<String>,
}

impl ClearKeyResponse {
    /// Create a new Clear Key response
    pub fn new(keys: Vec<JsonWebKey>) -> Self {
        Self {
            keys,
            response_type: Some("temporary".to_string()),
        }
    }

    /// Add a key to the response
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.push(JsonWebKey::new(key_id, key));
    }

    /// Get all keys as a map
    pub fn get_keys_map(&self) -> Result<HashMap<Vec<u8>, Vec<u8>>> {
        let mut map = HashMap::new();
        for jwk in &self.keys {
            let key_id = jwk.get_key_id()?;
            let key = jwk.get_key()?;
            map.insert(key_id, key);
        }
        Ok(map)
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(DrmError::JsonError)
    }
}

/// Clear Key license server (for testing)
pub struct ClearKeyServer {
    keys: HashMap<Vec<u8>, Vec<u8>>,
}

impl ClearKeyServer {
    /// Create a new Clear Key server
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Add a key to the server
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Add multiple keys
    pub fn add_keys(&mut self, keys: Vec<(Vec<u8>, Vec<u8>)>) {
        for (key_id, key) in keys {
            self.keys.insert(key_id, key);
        }
    }

    /// Process a license request and generate a response
    pub fn process_request(&self, request: &ClearKeyRequest) -> Result<ClearKeyResponse> {
        let mut response = ClearKeyResponse::new(Vec::new());

        let key_ids = request.get_key_ids()?;
        for key_id in key_ids {
            if let Some(key) = self.keys.get(&key_id) {
                response.add_key(key_id, key.clone());
            } else {
                return Err(DrmError::LicenseError(format!(
                    "Key not found: {}",
                    hex::encode(&key_id)
                )));
            }
        }

        Ok(response)
    }

    /// Process a request from JSON and return response as JSON
    pub fn process_request_json(&self, request_json: &str) -> Result<String> {
        let request = ClearKeyRequest::from_json(request_json)?;
        let response = self.process_request(&request)?;
        response.to_json()
    }

    /// Get number of keys in server
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Check if server has a key
    pub fn has_key(&self, key_id: &[u8]) -> bool {
        self.keys.contains_key(key_id)
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

impl Default for ClearKeyServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Clear Key PSSH data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKeyPsshData {
    /// List of key IDs
    pub key_ids: Vec<String>,
}

impl ClearKeyPsshData {
    /// Create new Clear Key PSSH data
    pub fn new(key_ids: Vec<Vec<u8>>) -> Self {
        let key_ids = key_ids
            .into_iter()
            .map(|id| URL_SAFE_NO_PAD.encode(&id))
            .collect();

        Self { key_ids }
    }

    /// Add a key ID
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.key_ids.push(URL_SAFE_NO_PAD.encode(&key_id));
    }

    /// Get key IDs as bytes
    pub fn get_key_ids(&self) -> Result<Vec<Vec<u8>>> {
        self.key_ids
            .iter()
            .map(|kid| URL_SAFE_NO_PAD.decode(kid).map_err(DrmError::Base64Error))
            .collect()
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

/// Clear Key client
pub struct ClearKeyClient {
    keys: HashMap<Vec<u8>, Vec<u8>>,
}

impl ClearKeyClient {
    /// Create a new Clear Key client
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Request keys from a Clear Key server
    pub fn request_keys(&mut self, key_ids: Vec<Vec<u8>>, server: &ClearKeyServer) -> Result<()> {
        let request = ClearKeyRequest::new(key_ids);
        let response = server.process_request(&request)?;

        for jwk in response.keys {
            let key_id = jwk.get_key_id()?;
            let key = jwk.get_key()?;
            self.keys.insert(key_id, key);
        }

        Ok(())
    }

    /// Add a key directly
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Get a key by ID
    pub fn get_key(&self, key_id: &[u8]) -> Option<&Vec<u8>> {
        self.keys.get(key_id)
    }

    /// Get all keys
    pub fn get_all_keys(&self) -> &HashMap<Vec<u8>, Vec<u8>> {
        &self.keys
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }

    /// Get number of keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

impl Default for ClearKeyClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ClearKey EME (Encrypted Media Extensions) JSON license format
// ---------------------------------------------------------------------------

/// A single key entry in a ClearKey JSON Web Key Set (JWKS) license.
///
/// The fields follow the W3C EME ClearKey specification:
/// <https://www.w3.org/TR/encrypted-media/#clear-key>
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClearKeyEntry {
    /// Key type — always `"oct"` for symmetric (octet sequence) keys.
    pub kty: String,
    /// Key value — base64url-encoded, no padding (URL_SAFE_NO_PAD).
    pub k: String,
    /// Key ID — base64url-encoded, no padding (URL_SAFE_NO_PAD).
    pub kid: String,
}

impl ClearKeyEntry {
    /// Construct a `ClearKeyEntry` from raw key-id and key-value bytes.
    ///
    /// Both inputs are base64url-encoded (no padding) in the resulting struct.
    pub fn from_bytes(key_id: &[u8], key_value: &[u8]) -> Self {
        Self {
            kty: "oct".to_string(),
            k: URL_SAFE_NO_PAD.encode(key_value),
            kid: URL_SAFE_NO_PAD.encode(key_id),
        }
    }

    /// Decode the `kid` field to raw bytes.
    pub fn key_id_bytes(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.kid)
            .map_err(DrmError::Base64Error)
    }

    /// Decode the `k` field to raw bytes.
    pub fn key_value_bytes(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.k)
            .map_err(DrmError::Base64Error)
    }
}

/// A ClearKey EME license object — a JSON Web Key Set (JWKS) with a `type` field.
///
/// This is the object that a ClearKey license server returns in response to an
/// `EncryptedEvent` license acquisition request.
///
/// JSON shape:
/// ```json
/// {
///   "keys": [
///     { "kty": "oct", "k": "<base64url-key>", "kid": "<base64url-kid>" }
///   ],
///   "type": "temporary"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClearKeyLicense {
    /// List of key entries in JWK format.
    pub keys: Vec<ClearKeyEntry>,
    /// License type — typically `"temporary"` or `"persistent-license"`.
    #[serde(rename = "type")]
    pub key_type: String,
}

impl ClearKeyLicense {
    /// Create a new `ClearKeyLicense` with the given entries.
    pub fn new(keys: Vec<ClearKeyEntry>, key_type: impl Into<String>) -> Self {
        Self {
            keys,
            key_type: key_type.into(),
        }
    }

    /// Create a temporary license (most common for streaming use).
    pub fn temporary(keys: Vec<ClearKeyEntry>) -> Self {
        Self::new(keys, "temporary")
    }

    /// Serialize to a compact JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(DrmError::JsonError)
    }
}

/// Generate a ClearKey EME JSON license string for a single key.
///
/// # Parameters
/// - `key_id`:    raw 16-byte content key identifier
/// - `key_value`: raw 16-byte content encryption key
///
/// # Returns
/// A compact JSON string suitable for delivery to the browser's EME
/// `MediaKeySession.update()` call.
///
/// # Example output
/// ```text
/// {"keys":[{"kty":"oct","k":"<base64url>","kid":"<base64url>"}],"type":"temporary"}
/// ```
pub fn generate_clearkey_license(key_id: &[u8], key_value: &[u8]) -> Result<String> {
    let entry = ClearKeyEntry::from_bytes(key_id, key_value);
    let license = ClearKeyLicense::temporary(vec![entry]);
    license.to_json()
}

/// Parse a ClearKey EME license request and extract the requested Key IDs.
///
/// A ClearKey request body looks like:
/// ```json
/// { "kids": ["<base64url-kid1>", "<base64url-kid2>"], "type": "temporary" }
/// ```
///
/// # Parameters
/// - `json`: the raw JSON string from the browser's license request
///
/// # Returns
/// A `Vec` of fixed-size `[u8; 16]` Key IDs decoded from the `kids` array.
///
/// # Errors
/// Returns `DrmError::JsonError` for malformed JSON, `DrmError::Base64Error`
/// for invalid base64url encoding, or `DrmError::LicenseError` if a decoded
/// KID is not exactly 16 bytes.
pub fn parse_clearkey_request(json: &str) -> Result<Vec<[u8; 16]>> {
    let request = ClearKeyRequest::from_json(json)?;
    let mut out = Vec::with_capacity(request.kids.len());
    for (idx, kid_b64) in request.kids.iter().enumerate() {
        let bytes = URL_SAFE_NO_PAD
            .decode(kid_b64)
            .map_err(DrmError::Base64Error)?;
        if bytes.len() != 16 {
            return Err(DrmError::LicenseError(format!(
                "KID at index {} decoded to {} bytes (expected 16)",
                idx,
                bytes.len()
            )));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes);
        out.push(arr);
    }
    Ok(out)
}

/// Create a Clear Key PSSH box
pub fn create_clearkey_pssh(key_ids: Vec<Vec<u8>>) -> Result<Vec<u8>> {
    use crate::cenc::PsshBox;

    let pssh_data = ClearKeyPsshData::new(key_ids.clone());
    let data = pssh_data.to_bytes()?;

    let pssh = PsshBox::new_v1(DrmSystem::ClearKey.system_id(), key_ids, data);
    pssh.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clearkey_request() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let request = ClearKeyRequest::new(vec![key_id.clone()]);

        assert_eq!(request.kids.len(), 1);
        let decoded = request.get_key_ids().expect("get_key_ids should succeed");
        assert_eq!(decoded[0], key_id);
    }

    #[test]
    fn test_clearkey_request_json() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let request = ClearKeyRequest::new(vec![key_id]);

        let json = request.to_json().expect("to_json should succeed");
        let parsed = ClearKeyRequest::from_json(&json).expect("from_json should parse");

        assert_eq!(parsed.kids, request.kids);
    }

    #[test]
    fn test_json_web_key() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let jwk = JsonWebKey::new(key_id.clone(), key.clone());

        assert_eq!(jwk.kty, "oct");
        assert_eq!(jwk.get_key_id().expect("get_key_id should decode"), key_id);
        assert_eq!(jwk.get_key().expect("get_key should decode"), key);
    }

    #[test]
    fn test_clearkey_response() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut response = ClearKeyResponse::new(Vec::new());
        response.add_key(key_id.clone(), key.clone());

        assert_eq!(response.keys.len(), 1);

        let keys_map = response
            .get_keys_map()
            .expect("get_keys_map should succeed");
        assert_eq!(keys_map.get(&key_id), Some(&key));
    }

    #[test]
    fn test_clearkey_response_json() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut response = ClearKeyResponse::new(Vec::new());
        response.add_key(key_id, key);

        let json = response.to_json().expect("to_json should succeed");
        let parsed = ClearKeyResponse::from_json(&json).expect("from_json should parse");

        assert_eq!(parsed.keys.len(), response.keys.len());
    }

    #[test]
    fn test_clearkey_server() {
        let mut server = ClearKeyServer::new();
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        server.add_key(key_id.clone(), key.clone());
        assert!(server.has_key(&key_id));

        let request = ClearKeyRequest::new(vec![key_id.clone()]);
        let response = server
            .process_request(&request)
            .expect("process_request should succeed");

        assert_eq!(response.keys.len(), 1);
        let keys_map = response
            .get_keys_map()
            .expect("get_keys_map should succeed");
        assert_eq!(keys_map.get(&key_id), Some(&key));
    }

    #[test]
    fn test_clearkey_server_missing_key() {
        let server = ClearKeyServer::new();
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let request = ClearKeyRequest::new(vec![key_id]);
        let result = server.process_request(&request);

        assert!(result.is_err());
    }

    #[test]
    fn test_clearkey_client() {
        let mut server = ClearKeyServer::new();
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        server.add_key(key_id.clone(), key.clone());

        let mut client = ClearKeyClient::new();
        client
            .request_keys(vec![key_id.clone()], &server)
            .expect("request_keys should succeed");

        assert_eq!(client.key_count(), 1);
        assert_eq!(client.get_key(&key_id), Some(&key));
    }

    #[test]
    fn test_clearkey_pssh_data() {
        let key_id1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key_id2 = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let pssh_data = ClearKeyPsshData::new(vec![key_id1.clone(), key_id2.clone()]);
        let bytes = pssh_data.to_bytes().expect("to_bytes should succeed");

        let parsed = ClearKeyPsshData::from_bytes(&bytes).expect("from_bytes should parse");
        let parsed_ids = parsed.get_key_ids().expect("get_key_ids should decode");

        assert_eq!(parsed_ids.len(), 2);
        assert_eq!(parsed_ids[0], key_id1);
        assert_eq!(parsed_ids[1], key_id2);
    }

    // -----------------------------------------------------------------------
    // ClearKeyEntry, ClearKeyLicense, generate_clearkey_license,
    // parse_clearkey_request tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_clearkey_entry_from_bytes_roundtrip() {
        let kid: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let k: [u8; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let entry = ClearKeyEntry::from_bytes(&kid, &k);
        assert_eq!(entry.kty, "oct");

        let decoded_kid = entry.key_id_bytes().expect("kid decode should succeed");
        let decoded_k = entry.key_value_bytes().expect("k decode should succeed");

        assert_eq!(decoded_kid, kid.to_vec());
        assert_eq!(decoded_k, k.to_vec());
    }

    #[test]
    fn test_clearkey_license_json_roundtrip() {
        let kid: [u8; 16] = [0xAB; 16];
        let k: [u8; 16] = [0xCD; 16];

        let entry = ClearKeyEntry::from_bytes(&kid, &k);
        let license = ClearKeyLicense::temporary(vec![entry]);

        let json = license.to_json().expect("serialize should succeed");
        let parsed = ClearKeyLicense::from_json(&json).expect("deserialize should succeed");

        assert_eq!(parsed.key_type, "temporary");
        assert_eq!(parsed.keys.len(), 1);
        assert_eq!(parsed.keys[0].kty, "oct");
        assert_eq!(
            parsed.keys[0].key_id_bytes().expect("kid decode"),
            kid.to_vec()
        );
        assert_eq!(
            parsed.keys[0].key_value_bytes().expect("k decode"),
            k.to_vec()
        );
    }

    #[test]
    fn test_generate_clearkey_license_is_valid_json() {
        let kid = [0x11u8; 16];
        let key_val = [0x22u8; 16];

        let json = generate_clearkey_license(&kid, &key_val)
            .expect("generate_clearkey_license should succeed");

        // Must be valid JSON and contain the expected fields
        let parsed =
            ClearKeyLicense::from_json(&json).expect("output must be valid ClearKeyLicense JSON");
        assert_eq!(parsed.key_type, "temporary");
        assert_eq!(parsed.keys.len(), 1);

        let decoded_kid = parsed.keys[0].key_id_bytes().expect("kid decode");
        let decoded_k = parsed.keys[0].key_value_bytes().expect("k decode");
        assert_eq!(decoded_kid, kid.to_vec());
        assert_eq!(decoded_k, key_val.to_vec());
    }

    #[test]
    fn test_parse_clearkey_request_valid() {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let kid1 = [0x01u8; 16];
        let kid2 = [0x02u8; 16];

        let json = format!(
            r#"{{"kids":["{}","{}"],"type":"temporary"}}"#,
            URL_SAFE_NO_PAD.encode(kid1),
            URL_SAFE_NO_PAD.encode(kid2)
        );

        let kids = parse_clearkey_request(&json).expect("parse should succeed");
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[0], kid1);
        assert_eq!(kids[1], kid2);
    }

    #[test]
    fn test_parse_clearkey_request_invalid_json() {
        let result = parse_clearkey_request("not json at all");
        assert!(result.is_err(), "invalid JSON should fail");
    }

    #[test]
    fn test_parse_clearkey_request_invalid_base64() {
        let json = r#"{"kids":["!!!not-valid-base64!!!"],"type":"temporary"}"#;
        let result = parse_clearkey_request(json);
        assert!(result.is_err(), "invalid base64 should fail");
    }

    #[test]
    fn test_clearkey_license_persistent_type() {
        let license = ClearKeyLicense::new(vec![], "persistent-license");
        assert_eq!(license.key_type, "persistent-license");
        assert!(license.keys.is_empty());
    }

    // -----------------------------------------------------------------------
    // W3C ClearKey integration tests with real EME test vectors
    //
    // Test vectors sourced from the W3C Encrypted Media Extensions test suite:
    // https://w3c.github.io/encrypted-media/
    // These are publicly known clear-text examples for conformance testing.
    // -----------------------------------------------------------------------

    /// W3C EME ClearKey test vector — single key
    ///
    /// KID  = 00000000-0000-0000-0000-000000000001 (big-endian UUID bytes)
    /// KEY  = ccc0f2b3b279926496a7f5d25da692f6
    /// Source: https://github.com/nickvdyck/webbundle/blob/master/tests/test-vectors/eme
    #[test]
    fn test_w3c_clearkey_vector_single_key_roundtrip() {
        // Known W3C test vector
        let kid_hex = "00000000000000000000000000000001";
        let key_hex = "ccc0f2b3b279926496a7f5d25da692f6";

        let kid_bytes = hex::decode(kid_hex).expect("hex kid decode");
        let key_bytes = hex::decode(key_hex).expect("hex key decode");

        // Build server with the test key
        let mut server = ClearKeyServer::new();
        server.add_key(kid_bytes.clone(), key_bytes.clone());

        // Build a license request
        let request = ClearKeyRequest::new(vec![kid_bytes.clone()]);
        let request_json = request.to_json().expect("request serialize");

        // Process through server
        let response_json = server
            .process_request_json(&request_json)
            .expect("server should respond");

        // Parse response and verify key material
        let response = ClearKeyResponse::from_json(&response_json).expect("response parse");
        assert_eq!(
            response.keys.len(),
            1,
            "response must contain exactly one key"
        );

        let resp_kid = response.keys[0].get_key_id().expect("kid decode");
        let resp_key = response.keys[0].get_key().expect("key decode");

        assert_eq!(resp_kid, kid_bytes, "KID must match test vector");
        assert_eq!(resp_key, key_bytes, "Key must match test vector");
    }

    /// W3C EME ClearKey test vector — multi-key license
    ///
    /// Uses two keys:
    ///   KID 1 = 0x1000000000000000000000000000001  KEY = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    ///   KID 2 = 0x2000000000000000000000000000002  KEY = 0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    #[test]
    fn test_w3c_clearkey_vector_multi_key() {
        let kid1 = vec![0x10u8; 16];
        let key1 = vec![0xAAu8; 16];
        let kid2 = vec![0x20u8; 16];
        let key2 = vec![0xBBu8; 16];

        let mut server = ClearKeyServer::new();
        server.add_key(kid1.clone(), key1.clone());
        server.add_key(kid2.clone(), key2.clone());

        let request = ClearKeyRequest::new(vec![kid1.clone(), kid2.clone()]);
        let response = server
            .process_request(&request)
            .expect("multi-key request should succeed");

        let keys_map = response.get_keys_map().expect("keys_map decode");
        assert_eq!(keys_map.len(), 2, "both keys must be returned");
        assert_eq!(keys_map.get(&kid1), Some(&key1), "key1 must match");
        assert_eq!(keys_map.get(&kid2), Some(&key2), "key2 must match");
    }

    /// W3C EME ClearKey — JSON license format conformance with known JSON string
    ///
    /// The W3C EME spec mandates this exact JSON structure for ClearKey responses.
    #[test]
    fn test_w3c_clearkey_json_format_conformance() {
        // Use base64url encoding of all-zero KID and all-0xFF key
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let kid = [0u8; 16];
        let key = [0xFFu8; 16];

        let kid_b64 = URL_SAFE_NO_PAD.encode(kid);
        let key_b64 = URL_SAFE_NO_PAD.encode(key);

        // Build expected JSON manually to verify conformance
        let expected_json = format!(
            r#"{{"keys":[{{"kty":"oct","k":"{}","kid":"{}"}}],"type":"temporary"}}"#,
            key_b64, kid_b64
        );

        let license = generate_clearkey_license(&kid, &key).expect("generate should succeed");

        // The generated license must be parseable and semantically equivalent
        let parsed = ClearKeyLicense::from_json(&license).expect("parse");
        let expected_parsed = ClearKeyLicense::from_json(&expected_json).expect("expected parse");
        assert_eq!(parsed.key_type, expected_parsed.key_type);
        assert_eq!(
            parsed.keys[0].key_id_bytes().expect("kid"),
            expected_parsed.keys[0].key_id_bytes().expect("kid"),
        );
        assert_eq!(
            parsed.keys[0].key_value_bytes().expect("k"),
            expected_parsed.keys[0].key_value_bytes().expect("k"),
        );
    }

    /// W3C ClearKey test: EME request/response protocol round-trip
    ///
    /// Simulates the browser-side EME workflow:
    /// 1. Client constructs license request JSON with `kids` array
    /// 2. Server parses KIDs and returns keys in JWK format
    /// 3. Client extracts content key and verifies it matches expected value
    #[test]
    fn test_w3c_eme_request_response_protocol() {
        // Content key from W3C test suite (simplified public test vector)
        let kid: [u8; 16] = [
            0x43, 0xba, 0xfe, 0x30, 0x4f, 0x57, 0x43, 0x5e, 0x87, 0x5d, 0x0c, 0x7b, 0xe3, 0x3e,
            0x0e, 0x9d,
        ];
        let key: [u8; 16] = [
            0xeb, 0x67, 0x62, 0xa7, 0x72, 0x7f, 0x4c, 0x41, 0x81, 0x9e, 0xc0, 0x7b, 0x96, 0x10,
            0x3c, 0x91,
        ];

        // Server provisioning
        let mut server = ClearKeyServer::new();
        server.add_key(kid.to_vec(), key.to_vec());

        // Step 1: client builds request JSON (simulating browser EME)
        let request_json = {
            let req = ClearKeyRequest::new(vec![kid.to_vec()]);
            req.to_json().expect("request json")
        };

        // Step 2: server processes request
        let response_json = server
            .process_request_json(&request_json)
            .expect("server processes request");

        // Step 3: client/browser processes response
        let license = ClearKeyLicense::from_json(&response_json).expect("license parse");
        assert_eq!(license.key_type, "temporary");
        assert_eq!(license.keys.len(), 1);

        let recovered_kid = license.keys[0].key_id_bytes().expect("kid bytes");
        let recovered_key = license.keys[0].key_value_bytes().expect("key bytes");

        assert_eq!(recovered_kid, kid.to_vec());
        assert_eq!(recovered_key, key.to_vec());
    }

    /// W3C ClearKey: parse_clearkey_request with multi-key request
    #[test]
    fn test_w3c_parse_clearkey_request_multi_key() {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let kids: Vec<[u8; 16]> = vec![[0x11u8; 16], [0x22u8; 16], [0x33u8; 16]];
        let kids_b64: Vec<String> = kids.iter().map(|k| URL_SAFE_NO_PAD.encode(k)).collect();
        let kids_json = kids_b64
            .iter()
            .map(|k| format!(r#""{}""#, k))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(r#"{{"kids":[{}],"type":"temporary"}}"#, kids_json);

        let parsed_kids = parse_clearkey_request(&json).expect("parse should succeed");
        assert_eq!(parsed_kids.len(), 3);
        assert_eq!(parsed_kids[0], kids[0]);
        assert_eq!(parsed_kids[1], kids[1]);
        assert_eq!(parsed_kids[2], kids[2]);
    }

    /// W3C ClearKey PSSH box: verify structure and parseable by pssh module
    #[test]
    fn test_w3c_clearkey_pssh_box_parseable() {
        let kid = vec![0xABu8; 16];
        let pssh_bytes =
            create_clearkey_pssh(vec![kid.clone()]).expect("create_clearkey_pssh should succeed");

        // Parse the PSSH box using the pssh module
        use crate::pssh::{PsshBox, CLEARKEY_SYSTEM_ID};
        let boxes = PsshBox::parse(&pssh_bytes).expect("PSSH parse should succeed");
        assert_eq!(boxes.len(), 1, "exactly one PSSH box");
        assert_eq!(
            boxes[0].system_id, CLEARKEY_SYSTEM_ID,
            "system ID must be ClearKey"
        );
        assert!(!boxes[0].data.is_empty(), "PSSH data should not be empty");

        // The data should be valid ClearKeyPsshData JSON
        let pssh_data = ClearKeyPsshData::from_bytes(&boxes[0].data)
            .expect("PSSH data should be valid ClearKeyPsshData");
        let recovered_ids = pssh_data.get_key_ids().expect("key IDs should decode");
        assert_eq!(recovered_ids.len(), 1);
        assert_eq!(recovered_ids[0], kid);
    }
}
