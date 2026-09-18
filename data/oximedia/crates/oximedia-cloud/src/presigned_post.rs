//! Pre-signed POST policy generation for browser-based direct uploads.
//!
//! Pre-signed POST allows a web browser to upload objects directly to cloud
//! storage without proxying through an application server.  This reduces
//! bandwidth costs and server load while keeping access control on the server
//! side (the server generates and signs the policy; the browser submits it).
//!
//! ## Algorithm (AWS S3)
//!
//! 1. Build a JSON policy document containing conditions (bucket, key prefix,
//!    content-length range, expiry, etc.).
//! 2. Base64-encode the policy JSON → `base64_policy`.
//! 3. Compute the AWS Signature V4 signing key:
//!    - `k_date    = HMAC-SHA256("AWS4" + secret_key, YYYYMMDD)`
//!    - `k_region  = HMAC-SHA256(k_date, region)`
//!    - `k_service = HMAC-SHA256(k_region, "s3")`
//!    - `k_signing = HMAC-SHA256(k_service, "aws4_request")`
//! 4. Compute `signature = hex(HMAC-SHA256(k_signing, base64_policy))`.
//! 5. Return the form URL and fields map that the browser POSTs as a
//!    multipart/form-data body.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_cloud::presigned_post::{PresignedPostPolicy, generate_presigned_post};
//! use std::time::{SystemTime, Duration};
//!
//! let policy = PresignedPostPolicy {
//!     bucket: "my-bucket".to_string(),
//!     key_prefix: "uploads/".to_string(),
//!     expires_at: SystemTime::now() + Duration::from_secs(3600),
//!     max_size_bytes: 100 * 1024 * 1024,
//!     allowed_content_types: vec!["video/mp4".to_string(), "video/webm".to_string()],
//!     region: "us-east-1".to_string(),
//!     access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
//!     secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
//! };
//!
//! let fields = generate_presigned_post(&policy).expect("valid policy");
//! assert!(fields.fields.contains_key("policy"));
//! assert!(fields.fields.contains_key("x-amz-signature"));
//! ```

use crate::error::{CloudError, Result};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::Utc;
use hmac::{Hmac, KeyInit, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

// ── Policy types ──────────────────────────────────────────────────────────────

/// Configuration for a pre-signed POST policy.
#[derive(Debug, Clone)]
pub struct PresignedPostPolicy {
    /// Target bucket name.
    pub bucket: String,
    /// Allowed key prefix for uploaded objects (e.g. `"uploads/user123/"`).
    pub key_prefix: String,
    /// Absolute expiry time for the policy.
    pub expires_at: SystemTime,
    /// Maximum allowed object size in bytes.
    pub max_size_bytes: u64,
    /// MIME types the browser may upload (empty list = any type allowed).
    pub allowed_content_types: Vec<String>,
    /// AWS region (e.g. `"us-east-1"`).
    pub region: String,
    /// AWS access key ID used to sign the policy.
    pub access_key_id: String,
    /// AWS secret access key used to derive the signing key.
    pub secret_access_key: String,
}

impl PresignedPostPolicy {
    /// Validate that the policy is well-formed.
    ///
    /// # Errors
    ///
    /// Returns [`CloudError::InvalidParameter`] when:
    /// - `bucket`, `region`, `access_key_id`, or `secret_access_key` is empty.
    /// - `max_size_bytes` is zero.
    /// - `expires_at` is in the past.
    pub fn validate(&self) -> Result<()> {
        if self.bucket.is_empty() {
            return Err(CloudError::InvalidParameter(
                "bucket must not be empty".to_string(),
            ));
        }
        if self.region.is_empty() {
            return Err(CloudError::InvalidParameter(
                "region must not be empty".to_string(),
            ));
        }
        if self.access_key_id.is_empty() {
            return Err(CloudError::InvalidParameter(
                "access_key_id must not be empty".to_string(),
            ));
        }
        if self.secret_access_key.is_empty() {
            return Err(CloudError::InvalidParameter(
                "secret_access_key must not be empty".to_string(),
            ));
        }
        if self.max_size_bytes == 0 {
            return Err(CloudError::InvalidParameter(
                "max_size_bytes must be greater than zero".to_string(),
            ));
        }
        let now = SystemTime::now();
        if self.expires_at <= now {
            return Err(CloudError::InvalidParameter(
                "expires_at must be in the future".to_string(),
            ));
        }
        Ok(())
    }
}

/// Fields returned to the caller — these become the multipart form fields
/// that the browser sends along with the file.
#[derive(Debug, Clone)]
pub struct PresignedPostFields {
    /// The POST URL (e.g. `https://<bucket>.s3.<region>.amazonaws.com/`).
    pub url: String,
    /// Form fields that must accompany the file in the multipart POST body.
    /// The browser must include ALL of these as form fields before the `file`
    /// field.
    pub fields: HashMap<String, String>,
}

// ── Core function ─────────────────────────────────────────────────────────────

/// Generate a pre-signed POST policy for AWS S3.
///
/// Returns a [`PresignedPostFields`] containing the `url` and all required
/// form fields.  The caller passes these to the browser (e.g. as JSON in an
/// API response), and the browser constructs a `multipart/form-data` POST.
///
/// # Errors
///
/// Returns an error if the policy fails validation.
pub fn generate_presigned_post(policy: &PresignedPostPolicy) -> Result<PresignedPostFields> {
    policy.validate()?;

    // ── Date strings ──────────────────────────────────────────────────────────

    let now = Utc::now();
    let date_ymd = now.format("%Y%m%d").to_string(); // YYYYMMDD
    let date_iso = now.format("%Y%m%dT%H%M%SZ").to_string(); // ISO 8601 compact

    // ── Credential scope ──────────────────────────────────────────────────────

    let credential = format!(
        "{}/{}/{}/s3/aws4_request",
        policy.access_key_id, date_ymd, policy.region
    );

    // ── Expiry string (ISO 8601) ───────────────────────────────────────────────

    let expires_secs = policy
        .expires_at
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    let expires_dt = chrono::DateTime::<Utc>::from_timestamp(expires_secs as i64, 0)
        .unwrap_or_else(|| Utc::now() + chrono::Duration::hours(1));
    let expires_str = expires_dt.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    // ── Conditions ────────────────────────────────────────────────────────────

    let mut conditions: Vec<Value> = vec![
        json!({"bucket": &policy.bucket}),
        json!(["starts-with", "$key", &policy.key_prefix]),
        json!({"x-amz-credential": &credential}),
        json!({"x-amz-algorithm": "AWS4-HMAC-SHA256"}),
        json!({"x-amz-date": &date_iso}),
        json!(["content-length-range", 1, policy.max_size_bytes]),
    ];

    if !policy.allowed_content_types.is_empty() {
        if policy.allowed_content_types.len() == 1 {
            conditions.push(json!({"Content-Type": &policy.allowed_content_types[0]}));
        } else {
            // Allow any of the listed content types via starts-with on the
            // major type (e.g. "video/") — use the first type's prefix.
            let first = &policy.allowed_content_types[0];
            let prefix = first.split('/').next().unwrap_or("application");
            conditions.push(json!([
                "starts-with",
                "$Content-Type",
                format!("{prefix}/")
            ]));
        }
    }

    // ── Policy document ───────────────────────────────────────────────────────

    let policy_doc = json!({
        "expiration": expires_str,
        "conditions": conditions,
    });

    let policy_json = serde_json::to_string(&policy_doc)
        .map_err(|e| CloudError::Serialization(format!("Policy serialization failed: {e}")))?;

    let base64_policy = BASE64.encode(policy_json.as_bytes());

    // ── Signing key derivation (AWS Signature V4) ─────────────────────────────

    let signing_key = derive_signing_key(&policy.secret_access_key, &date_ymd, &policy.region);

    // ── Signature over base64 policy ──────────────────────────────────────────

    let mut mac = HmacSha256::new_from_slice(&signing_key)
        .map_err(|e| CloudError::Encryption(format!("HMAC key error: {e}")))?;
    mac.update(base64_policy.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    // ── Fields map ────────────────────────────────────────────────────────────

    let mut fields = HashMap::new();
    fields.insert(
        "key".to_string(),
        format!("{}${{filename}}", policy.key_prefix),
    );
    fields.insert(
        "x-amz-algorithm".to_string(),
        "AWS4-HMAC-SHA256".to_string(),
    );
    fields.insert("x-amz-credential".to_string(), credential);
    fields.insert("x-amz-date".to_string(), date_iso);
    fields.insert("policy".to_string(), base64_policy);
    fields.insert("x-amz-signature".to_string(), signature);

    if !policy.allowed_content_types.is_empty() {
        fields.insert(
            "Content-Type".to_string(),
            policy.allowed_content_types[0].clone(),
        );
    }

    // ── POST URL ──────────────────────────────────────────────────────────────

    let url = format!(
        "https://{}.s3.{}.amazonaws.com/",
        policy.bucket, policy.region
    );

    Ok(PresignedPostFields { url, fields })
}

// ── Signing key derivation ────────────────────────────────────────────────────

/// Derive an AWS Signature V4 signing key.
///
/// The derivation chain is:
/// ```text
/// k_date    = HMAC-SHA256("AWS4" + secret,   YYYYMMDD)
/// k_region  = HMAC-SHA256(k_date,            region)
/// k_service = HMAC-SHA256(k_region,          "s3")
/// k_signing = HMAC-SHA256(k_service,         "aws4_request")
/// ```
fn derive_signing_key(secret: &str, date_ymd: &str, region: &str) -> Vec<u8> {
    let k_secret = format!("AWS4{secret}");
    let k_date = hmac_sha256(k_secret.as_bytes(), date_ymd.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    hmac_sha256(&k_service, b"aws4_request")
}

/// Compute `HMAC-SHA256(key, data)` and return the raw bytes.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn make_valid_policy() -> PresignedPostPolicy {
        PresignedPostPolicy {
            bucket: "test-bucket".to_string(),
            key_prefix: "uploads/".to_string(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            max_size_bytes: 100 * 1024 * 1024,
            allowed_content_types: vec!["video/mp4".to_string()],
            region: "us-east-1".to_string(),
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        }
    }

    // ── Policy validation ─────────────────────────────────────────────────────

    #[test]
    fn test_presigned_post_policy_valid() {
        let policy = make_valid_policy();
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_presigned_post_policy_empty_bucket() {
        let mut p = make_valid_policy();
        p.bucket = "".to_string();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_presigned_post_policy_empty_region() {
        let mut p = make_valid_policy();
        p.region = "".to_string();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_presigned_post_policy_empty_access_key() {
        let mut p = make_valid_policy();
        p.access_key_id = "".to_string();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_presigned_post_policy_empty_secret_key() {
        let mut p = make_valid_policy();
        p.secret_access_key = "".to_string();
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_presigned_post_policy_zero_size() {
        let mut p = make_valid_policy();
        p.max_size_bytes = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_presigned_post_policy_expired() {
        let mut p = make_valid_policy();
        // Set expiry 1 second in the past
        p.expires_at = SystemTime::now() - Duration::from_secs(1);
        assert!(p.validate().is_err());
    }

    // ── Field generation ──────────────────────────────────────────────────────

    #[test]
    fn test_presigned_post_fields_contain_required_keys() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy must succeed");

        // Required AWS S3 POST policy fields
        assert!(
            fields.fields.contains_key("policy"),
            "fields must contain 'policy'"
        );
        assert!(
            fields.fields.contains_key("x-amz-signature"),
            "fields must contain 'x-amz-signature'"
        );
        assert!(
            fields.fields.contains_key("x-amz-credential"),
            "fields must contain 'x-amz-credential'"
        );
        assert!(
            fields.fields.contains_key("x-amz-algorithm"),
            "fields must contain 'x-amz-algorithm'"
        );
        assert!(
            fields.fields.contains_key("x-amz-date"),
            "fields must contain 'x-amz-date'"
        );
        assert!(
            fields.fields.contains_key("key"),
            "fields must contain 'key'"
        );
    }

    #[test]
    fn test_presigned_post_url_format() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy");
        assert!(
            fields.url.contains("test-bucket"),
            "URL must contain bucket name"
        );
        assert!(fields.url.contains("us-east-1"), "URL must contain region");
        assert!(fields.url.starts_with("https://"), "URL must use HTTPS");
    }

    #[test]
    fn test_presigned_post_algorithm_is_aws4_hmac_sha256() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy");
        assert_eq!(
            fields.fields.get("x-amz-algorithm").map(String::as_str),
            Some("AWS4-HMAC-SHA256")
        );
    }

    #[test]
    fn test_presigned_post_credential_contains_scope() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy");
        let cred = fields
            .fields
            .get("x-amz-credential")
            .expect("credential must be present");
        assert!(
            cred.contains("AKIAIOSFODNN7EXAMPLE"),
            "Credential must contain access key ID"
        );
        assert!(cred.contains("us-east-1"), "Credential must contain region");
        assert!(cred.contains("/s3/"), "Credential must contain service");
        assert!(
            cred.contains("aws4_request"),
            "Credential must contain request type"
        );
    }

    #[test]
    fn test_presigned_post_policy_serialization() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy");

        // The policy field is base64-encoded JSON
        let policy_b64 = fields.fields.get("policy").expect("policy field");
        let decoded = BASE64
            .decode(policy_b64)
            .expect("policy must be valid base64");
        let doc: serde_json::Value =
            serde_json::from_slice(&decoded).expect("policy must be valid JSON");

        assert!(
            doc.get("expiration").is_some(),
            "Policy JSON must have expiration"
        );
        assert!(
            doc.get("conditions").is_some(),
            "Policy JSON must have conditions"
        );
    }

    #[test]
    fn test_presigned_post_key_prefix_in_key_field() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy");
        let key_field = fields.fields.get("key").expect("key field must exist");
        assert!(
            key_field.starts_with("uploads/"),
            "Key field must start with key_prefix"
        );
    }

    #[test]
    fn test_presigned_post_content_type_included_when_specified() {
        let policy = make_valid_policy();
        let fields = generate_presigned_post(&policy).expect("valid policy");
        assert_eq!(
            fields.fields.get("Content-Type").map(String::as_str),
            Some("video/mp4")
        );
    }

    #[test]
    fn test_presigned_post_no_content_type_when_empty() {
        let mut policy = make_valid_policy();
        policy.allowed_content_types.clear();
        let fields = generate_presigned_post(&policy).expect("valid policy");
        // Content-Type should not be forced when the list is empty
        assert!(
            !fields.fields.contains_key("Content-Type"),
            "Content-Type must not be set when allowed list is empty"
        );
    }

    #[test]
    fn test_presigned_post_different_secrets_different_signatures() {
        let mut p1 = make_valid_policy();
        let mut p2 = make_valid_policy();
        p1.secret_access_key = "secret1".to_string();
        p2.secret_access_key = "secret2".to_string();

        let f1 = generate_presigned_post(&p1).expect("p1");
        let f2 = generate_presigned_post(&p2).expect("p2");

        assert_ne!(
            f1.fields.get("x-amz-signature"),
            f2.fields.get("x-amz-signature"),
            "Different secrets must produce different signatures"
        );
    }

    // ── Signing key derivation ────────────────────────────────────────────────

    #[test]
    fn test_derive_signing_key_deterministic() {
        let k1 = derive_signing_key("secret", "20260515", "us-east-1");
        let k2 = derive_signing_key("secret", "20260515", "us-east-1");
        assert_eq!(k1, k2, "Signing key derivation must be deterministic");
    }

    #[test]
    fn test_derive_signing_key_differs_by_region() {
        let k1 = derive_signing_key("secret", "20260515", "us-east-1");
        let k2 = derive_signing_key("secret", "20260515", "eu-west-1");
        assert_ne!(k1, k2, "Signing keys for different regions must differ");
    }

    #[test]
    fn test_hmac_sha256_known_output() {
        // Sanity check: HMAC-SHA256("key", "data") is deterministic
        let out1 = hmac_sha256(b"key", b"data");
        let out2 = hmac_sha256(b"key", b"data");
        assert_eq!(out1, out2);
        assert_eq!(out1.len(), 32, "HMAC-SHA256 output must be 32 bytes");
    }
}
