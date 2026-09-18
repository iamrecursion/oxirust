//! Server-side Digest authentication (RFC 2617).
//!
//! Servers issue a `WWW-Authenticate: Digest realm=..., nonce=...` challenge
//! on the first unauthenticated request, then verify the client's
//! `Authorization: Digest ...` response on the retry.

use std::time::{SystemTime, UNIX_EPOCH};

/// A server-side Digest authentication challenge.
///
/// Generated once per unauthenticated request; shared state for verifying
/// the client's subsequent retry.
///
/// # Example
///
/// ```
/// use oximedia_net::rtsp::server::ServerChallenge;
/// let ch = ServerChallenge::issue("secure-realm");
/// let hdr = ch.www_authenticate_header();
/// assert!(hdr.starts_with("Digest realm="));
/// assert!(hdr.contains("nonce="));
/// ```
#[derive(Debug, Clone)]
pub struct ServerChallenge {
    /// Authentication realm — typically the server's hostname or a fixed string.
    pub realm: String,
    /// Server-supplied nonce — unique per challenge; clients echo it back.
    pub nonce: String,
}

impl ServerChallenge {
    /// Issue a new challenge for the given `realm`.
    ///
    /// The nonce is derived from the current system time (nanoseconds since
    /// UNIX epoch) formatted as a 16-character hex string, which provides
    /// sufficient uniqueness for RTSP sessions without pulling in a PRNG crate.
    #[must_use]
    pub fn issue(realm: &str) -> Self {
        let nonce = generate_nonce();
        Self {
            realm: realm.to_string(),
            nonce,
        }
    }

    /// Build the `WWW-Authenticate:` header value to send to the client.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::server::ServerChallenge;
    /// let ch = ServerChallenge::issue("cam");
    /// let v = ch.www_authenticate_header();
    /// assert!(v.contains("realm=\"cam\""));
    /// assert!(v.contains("algorithm=MD5"));
    /// ```
    #[must_use]
    pub fn www_authenticate_header(&self) -> String {
        format!(
            "Digest realm=\"{}\", nonce=\"{}\", algorithm=MD5",
            self.realm, self.nonce
        )
    }

    /// Verify a client's `Authorization: Digest ...` header.
    ///
    /// Returns `true` if the client's computed response matches what the server
    /// would compute for `expected_password`.
    ///
    /// `method` is the RTSP method string (e.g. `"DESCRIBE"`).
    /// `uri` is the request URI.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::server::ServerChallenge;
    /// use oximedia_net::rtsp::{Challenge, Credentials};
    ///
    /// let server = ServerChallenge { realm: "r".into(), nonce: "testnonce".into() };
    ///
    /// // Build the auth header the client would send
    /// let client_challenge = Challenge::Digest {
    ///     realm: "r".into(),
    ///     nonce: "testnonce".into(),
    ///     opaque: None,
    ///     qop: None,
    ///     algorithm: None,
    /// };
    /// let creds = Credentials { username: "admin".into(), password: "secret".into() };
    /// let auth = client_challenge.build_authorization(&creds, "DESCRIBE", "rtsp://x/y", 1, "c");
    ///
    /// assert!(server.verify(&auth, "DESCRIBE", "rtsp://x/y", "secret"));
    /// assert!(!server.verify(&auth, "DESCRIBE", "rtsp://x/y", "wrong"));
    /// ```
    #[must_use]
    pub fn verify(
        &self,
        authorization_header: &str,
        method: &str,
        uri: &str,
        expected_password: &str,
    ) -> bool {
        let _ = uri; // URI not used in server-side verification (client echoes it back)
                     // Parse the Authorization: Digest ... header into key=value pairs.
        let trimmed = authorization_header.trim_start();
        let params_str = match trimmed
            .strip_prefix("Digest ")
            .or_else(|| trimmed.strip_prefix("digest "))
        {
            Some(s) => s,
            None => return false,
        };

        let params = parse_auth_params(params_str);

        let get = |key: &str| -> Option<&str> {
            params
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map(|(_, v)| v.as_str())
        };

        let username = match get("username") {
            Some(u) => u,
            None => return false,
        };
        let realm = match get("realm") {
            Some(r) => r,
            None => return false,
        };
        let nonce = match get("nonce") {
            Some(n) => n,
            None => return false,
        };
        let client_uri = match get("uri") {
            Some(u) => u,
            None => return false,
        };
        let response = match get("response") {
            Some(r) => r,
            None => return false,
        };

        // Reject stale nonces (the nonce in the header must match ours).
        if nonce != self.nonce {
            return false;
        }

        // Compute expected response.
        let ha1 = md5_hex(&format!("{username}:{realm}:{expected_password}"));
        let ha2 = md5_hex(&format!("{method}:{client_uri}"));

        let qop = get("qop");
        let expected_response = if qop.map(|q| q.contains("auth")).unwrap_or(false) {
            let nc = get("nc").unwrap_or("00000001");
            let cnonce = get("cnonce").unwrap_or("");
            md5_hex(&format!("{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}"))
        } else {
            // RFC 2069 / no-qop
            md5_hex(&format!("{ha1}:{nonce}:{ha2}"))
        };

        // Constant-time comparison to resist timing attacks.
        constant_time_eq(expected_response.as_bytes(), response.as_bytes())
    }
}

/// Constant-time equality comparison for byte slices.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Parse `key="value", key2=value2` into a vec. Shared logic with client auth.
fn parse_auth_params(input: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b',' || bytes[i] == b'\t') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' {
            i += 1;
        }
        let key = std::str::from_utf8(&bytes[key_start..i])
            .unwrap_or("")
            .trim()
            .to_string();
        if i >= bytes.len() {
            break;
        }
        i += 1; // skip '='

        if i < bytes.len() && bytes[i] == b'"' {
            i += 1;
            let val_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                    continue;
                }
                i += 1;
            }
            let value = std::str::from_utf8(&bytes[val_start..i])
                .unwrap_or("")
                .to_string();
            out.push((key, value));
            if i < bytes.len() {
                i += 1; // skip closing quote
            }
        } else {
            let val_start = i;
            while i < bytes.len() && bytes[i] != b',' {
                i += 1;
            }
            let value = std::str::from_utf8(&bytes[val_start..i])
                .unwrap_or("")
                .trim()
                .to_string();
            out.push((key, value));
        }
    }
    out
}

/// Generate a unique nonce from the current system time (nanoseconds since epoch).
fn generate_nonce() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // XOR with a static salt so consecutive calls in the same nanosecond differ.
    static COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0x9e3779b97f4a7c15);
    let salt = COUNTER.fetch_add(0x517cc1b727220a95, std::sync::atomic::Ordering::Relaxed);
    format!("{:016x}", ts ^ salt)
}

/// MD5 hex digest (lowercase) — uses the same inline implementation as auth.rs.
fn md5_hex(input: &str) -> String {
    let digest = md5(input.as_bytes());
    let mut out = String::with_capacity(32);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// MD5 (RFC 1321) — identical to the impl in auth.rs.
fn md5(input: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    let orig_len = input.len() as u64;
    let bit_len = orig_len.wrapping_mul(8);

    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_le_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            m[i] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        }

        let mut a = a0;
        let mut b = b0;
        let mut c = c0;
        let mut d = d0;
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtsp::{Challenge, Credentials};

    #[test]
    fn www_authenticate_header_format() {
        let ch = ServerChallenge {
            realm: "test-realm".into(),
            nonce: "abc123".into(),
        };
        let hdr = ch.www_authenticate_header();
        assert!(hdr.starts_with("Digest "));
        assert!(hdr.contains("realm=\"test-realm\""));
        assert!(hdr.contains("nonce=\"abc123\""));
        assert!(hdr.contains("algorithm=MD5"));
    }

    #[test]
    fn verify_valid_no_qop() {
        let server = ServerChallenge {
            realm: "r".into(),
            nonce: "testnonce".into(),
        };
        let client_challenge = Challenge::Digest {
            realm: "r".into(),
            nonce: "testnonce".into(),
            opaque: None,
            qop: None,
            algorithm: None,
        };
        let creds = Credentials {
            username: "admin".into(),
            password: "secret".into(),
        };
        let auth = client_challenge.build_authorization(&creds, "DESCRIBE", "rtsp://x/y", 1, "c");
        assert!(server.verify(&auth, "DESCRIBE", "rtsp://x/y", "secret"));
    }

    #[test]
    fn verify_wrong_password_rejected() {
        let server = ServerChallenge {
            realm: "r".into(),
            nonce: "nonce1".into(),
        };
        let client_challenge = Challenge::Digest {
            realm: "r".into(),
            nonce: "nonce1".into(),
            opaque: None,
            qop: None,
            algorithm: None,
        };
        let creds = Credentials {
            username: "u".into(),
            password: "right".into(),
        };
        let auth = client_challenge.build_authorization(&creds, "PLAY", "rtsp://h/s", 1, "c");
        assert!(!server.verify(&auth, "PLAY", "rtsp://h/s", "wrong"));
    }

    #[test]
    fn verify_stale_nonce_rejected() {
        let server = ServerChallenge {
            realm: "r".into(),
            nonce: "current".into(),
        };
        let client_challenge = Challenge::Digest {
            realm: "r".into(),
            nonce: "stale".into(),
            opaque: None,
            qop: None,
            algorithm: None,
        };
        let creds = Credentials {
            username: "u".into(),
            password: "p".into(),
        };
        let auth = client_challenge.build_authorization(&creds, "PLAY", "rtsp://h/s", 1, "c");
        // Even if password is right, nonce mismatch must reject
        assert!(!server.verify(&auth, "PLAY", "rtsp://h/s", "p"));
    }

    #[test]
    fn verify_valid_with_qop() {
        let server = ServerChallenge {
            realm: "qrealm".into(),
            nonce: "qnonce".into(),
        };
        let client_challenge = Challenge::Digest {
            realm: "qrealm".into(),
            nonce: "qnonce".into(),
            opaque: None,
            qop: Some("auth".into()),
            algorithm: Some("MD5".into()),
        };
        let creds = Credentials {
            username: "user".into(),
            password: "pass".into(),
        };
        let auth =
            client_challenge.build_authorization(&creds, "SETUP", "rtsp://h/s/t1", 1, "cnonce123");
        assert!(server.verify(&auth, "SETUP", "rtsp://h/s/t1", "pass"));
    }

    #[test]
    fn nonce_uniqueness() {
        // Two consecutive calls should produce different nonces.
        let n1 = generate_nonce();
        let n2 = generate_nonce();
        assert_ne!(n1, n2);
    }
}
