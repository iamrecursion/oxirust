//! DTLS (Datagram Transport Layer Security) wrapper.
//!
//! # ⚠️ Experimental — DTLS-SRTP handshake is NOT implemented
//!
//! This module can generate a **real** self-signed Ed25519 certificate and the
//! corresponding SDP fingerprint (used during signaling), but it does **not**
//! implement a DTLS 1.2 handshake or the RFC 5764 DTLS-SRTP key export.
//!
//! Because no handshake exists, no SRTP keying material can be derived. To avoid
//! silently transmitting media in plaintext under a "DTLS-protected" label:
//!
//! * [`DtlsEndpoint::handshake`] returns an error instead of fabricating a
//!   connection with all-zero SRTP keys.
//! * [`DtlsConnection::send`] / [`DtlsConnection::recv`] refuse to operate.
//!
//! Implementing a correct, interoperable DTLS-SRTP stack in pure Rust (record
//! layer, `HelloVerifyRequest` cookie, ECDHE key exchange, `Finished`
//! verification, the `use_srtp` extension and the RFC 5705 exporter) is a large
//! undertaking that cannot be completed without either a full hand-rolled DTLS
//! implementation or a C/FFI-backed crate (e.g. `ring`), and is therefore left
//! unimplemented rather than faked. **Do not use WebRTC media transport for
//! confidential media until this is implemented.**
//!
//! The [`DtlsSession`] type below is a non-cryptographic state-machine
//! simulation used only by unit tests; it must never be used to protect real
//! media (see its own documentation).

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::error::{NetError, NetResult};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::net::UdpSocket;

/// DTLS role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlsRole {
    /// Client role (active).
    Client,
    /// Server role (passive).
    Server,
}

impl DtlsRole {
    /// Parses from SDP setup attribute.
    #[must_use]
    pub fn from_setup(setup: &str) -> Option<Self> {
        match setup {
            "active" => Some(Self::Client),
            "passive" => Some(Self::Server),
            "actpass" => Some(Self::Server), // Default to server
            _ => None,
        }
    }

    /// Returns the SDP setup attribute.
    #[must_use]
    pub const fn to_setup(&self) -> &'static str {
        match self {
            Self::Client => "active",
            Self::Server => "passive",
        }
    }
}

/// DTLS fingerprint.
#[derive(Debug, Clone)]
pub struct DtlsFingerprint {
    /// Hash algorithm.
    pub algorithm: String,
    /// Fingerprint value (hex).
    pub value: String,
}

impl DtlsFingerprint {
    /// Creates a SHA-256 fingerprint from certificate.
    #[must_use]
    pub fn from_certificate(cert: &CertificateDer) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(cert.as_ref());
        let hash = hasher.finalize();

        let value = hash
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(":");

        Self {
            algorithm: "sha-256".to_string(),
            value,
        }
    }

    /// Formats for SDP.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        format!("{} {}", self.algorithm, self.value)
    }
}

/// DTLS configuration.
pub struct DtlsConfig {
    /// Certificate chain.
    pub certificates: Vec<CertificateDer<'static>>,
    /// Private key.
    pub private_key: PrivateKeyDer<'static>,
    /// DTLS role.
    pub role: DtlsRole,
}

impl DtlsConfig {
    /// Creates a new configuration with self-signed certificate.
    pub fn new_self_signed(role: DtlsRole) -> NetResult<Self> {
        let (cert, key) = generate_self_signed_cert()?;

        Ok(Self {
            certificates: vec![cert],
            private_key: key,
            role,
        })
    }

    /// Gets the fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> DtlsFingerprint {
        DtlsFingerprint::from_certificate(&self.certificates[0])
    }
}

/// DTLS endpoint.
pub struct DtlsEndpoint {
    /// Configuration.
    config: DtlsConfig,
    /// UDP socket.
    socket: Arc<UdpSocket>,
}

impl DtlsEndpoint {
    /// Creates a new DTLS endpoint.
    #[must_use]
    pub fn new(config: DtlsConfig, socket: Arc<UdpSocket>) -> Self {
        Self { config, socket }
    }

    /// Attempts a DTLS handshake.
    ///
    /// # ⚠️ Not implemented
    ///
    /// A real DTLS 1.2 handshake and RFC 5764 DTLS-SRTP key export are **not
    /// implemented**. Rather than returning a fabricated connection whose SRTP
    /// keys are all zero — which would cause media to be sent in plaintext while
    /// labeled DTLS-SRTP protected — this method returns an error so callers
    /// fail loudly instead of silently establishing an insecure channel.
    ///
    /// # Errors
    ///
    /// Always returns [`NetError::Handshake`]; see the module documentation for
    /// the rationale and what a real implementation would require.
    pub async fn handshake(&self) -> NetResult<DtlsConnection> {
        // NOTE: `self.socket` / `self.config` are intentionally unused here.
        // A real implementation would drive the DTLS state machine over the
        // socket, verify the remote fingerprint against the negotiated SDP,
        // and export SRTP keying material (RFC 5764 / RFC 5705). None of that
        // exists yet, so we refuse instead of returning zeroed keys.
        Err(NetError::handshake(
            "WebRTC DTLS-SRTP handshake is not implemented (experimental). \
             Refusing to establish a connection with all-zero SRTP keys that \
             would transmit media in plaintext under a DTLS-protected label. \
             See the oximedia_net::webrtc::dtls module documentation.",
        ))
    }

    /// Gets the local fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> DtlsFingerprint {
        self.config.fingerprint()
    }
}

/// A DTLS connection.
///
/// # ⚠️ Not usable for media transport
///
/// Because [`DtlsEndpoint::handshake`] is not implemented, this type can only be
/// constructed inside unit tests. Its [`send`](Self::send) and
/// [`recv`](Self::recv) methods deliberately refuse to operate so that no media
/// can be transmitted in plaintext under a "DTLS-protected" label. It no longer
/// carries the previous all-zero SRTP master key/salt.
pub struct DtlsConnection {
    /// UDP socket.
    socket: Arc<UdpSocket>,
}

impl DtlsConnection {
    /// Refuses to send: DTLS-SRTP protection is not implemented.
    ///
    /// A previous version transmitted `data` directly over the raw UDP socket
    /// with **no encryption** while presenting itself as a DTLS-protected
    /// channel. That silent plaintext transmission has been removed.
    ///
    /// # Errors
    ///
    /// Always returns [`NetError::Protocol`].
    pub async fn send(&self, _data: &[u8]) -> NetResult<()> {
        Err(NetError::protocol(
            "WebRTC DTLS-SRTP is not implemented: refusing to send media that \
             would be transmitted in plaintext under a DTLS-protected label. \
             See the oximedia_net::webrtc::dtls module documentation.",
        ))
    }

    /// Refuses to receive: DTLS-SRTP protection is not implemented.
    ///
    /// # Errors
    ///
    /// Always returns [`NetError::Protocol`].
    pub async fn recv(&self, _buf: &mut [u8]) -> NetResult<usize> {
        Err(NetError::protocol(
            "WebRTC DTLS-SRTP is not implemented: refusing to receive because \
             no DTLS decryption is available. \
             See the oximedia_net::webrtc::dtls module documentation.",
        ))
    }

    /// Gets the underlying socket.
    #[must_use]
    pub fn socket(&self) -> &Arc<UdpSocket> {
        &self.socket
    }
}

/// Generates a self-signed Ed25519 certificate and its PKCS#8 private key.
///
/// The certificate is a genuine, self-consistent X.509 v3 certificate signed
/// with Ed25519 (RFC 8410), so the SHA-256 fingerprint placed in SDP is real.
/// This is used during signaling only; it is **not** presented in a DTLS
/// handshake (which is unimplemented — see the module documentation).
fn generate_self_signed_cert() -> NetResult<(CertificateDer<'static>, PrivateKeyDer<'static>)> {
    use ed25519_dalek::pkcs8::EncodePrivateKey;
    use ed25519_dalek::SigningKey;
    use rand::Rng;

    let mut secret = [0u8; 32];
    rand::rng().fill_bytes(&mut secret);
    let signing_key = SigningKey::from_bytes(&secret);
    let pkcs8_doc = signing_key
        .to_pkcs8_der()
        .map_err(|e| NetError::protocol(format!("Failed to encode key as PKCS8: {e}")))?;

    let cert_der = build_ed25519_self_signed_cert(&signing_key);
    let key_der = PrivateKeyDer::Pkcs8(pkcs8_doc.as_bytes().to_vec().into());

    Ok((cert_der, key_der))
}

/// DER-encoded `AlgorithmIdentifier` for Ed25519 (RFC 8410): `SEQUENCE { OID
/// 1.3.101.112 }` with no parameters.
const ED25519_ALG_ID: [u8; 7] = [0x30, 0x05, 0x06, 0x03, 0x2B, 0x65, 0x70];

/// Encodes a DER definite-form length.
fn der_len(len: usize) -> Vec<u8> {
    if len < 0x80 {
        vec![len as u8]
    } else {
        let mut bytes = Vec::new();
        let mut n = len;
        while n > 0 {
            bytes.push((n & 0xFF) as u8);
            n >>= 8;
        }
        bytes.reverse();
        let mut out = Vec::with_capacity(bytes.len() + 1);
        out.push(0x80 | (bytes.len() as u8));
        out.extend_from_slice(&bytes);
        out
    }
}

/// Wraps `contents` in a DER TLV with the given `tag`.
fn der_tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + contents.len());
    out.push(tag);
    out.extend_from_slice(&der_len(contents.len()));
    out.extend_from_slice(contents);
    out
}

/// Builds an X.509 `Name` containing a single `commonName` attribute.
fn der_name_cn(cn: &str) -> Vec<u8> {
    // OID id-at-commonName 2.5.4.3.
    let oid: [u8; 5] = [0x06, 0x03, 0x55, 0x04, 0x03];
    let value = der_tlv(0x0C, cn.as_bytes()); // UTF8String
    let mut atv = Vec::new();
    atv.extend_from_slice(&oid);
    atv.extend_from_slice(&value);
    let atv_seq = der_tlv(0x30, &atv); // AttributeTypeAndValue
    let rdn = der_tlv(0x31, &atv_seq); // RelativeDistinguishedName (SET)
    der_tlv(0x30, &rdn) // Name (SEQUENCE OF RDN)
}

/// Builds the `SubjectPublicKeyInfo` for an Ed25519 public key.
fn der_spki(public_key: &[u8; 32]) -> Vec<u8> {
    let mut bit = Vec::with_capacity(33);
    bit.push(0x00); // unused bits
    bit.extend_from_slice(public_key);
    let bit_string = der_tlv(0x03, &bit);
    let mut contents = Vec::new();
    contents.extend_from_slice(&ED25519_ALG_ID);
    contents.extend_from_slice(&bit_string);
    der_tlv(0x30, &contents)
}

/// Builds the `Validity` field with a fixed wide window (UTCTime).
///
/// Self-signed WebRTC certificates are fingerprint-pinned, not CA-validated, so
/// a fixed validity window is standard and avoids time-dependent output.
fn der_validity() -> Vec<u8> {
    let not_before = der_tlv(0x17, b"000101000000Z"); // 2000-01-01T00:00:00Z
    let not_after = der_tlv(0x17, b"491231235959Z"); // 2049-12-31T23:59:59Z
    let mut contents = Vec::new();
    contents.extend_from_slice(&not_before);
    contents.extend_from_slice(&not_after);
    der_tlv(0x30, &contents)
}

/// Builds a genuine Ed25519-signed, self-signed X.509 v3 certificate.
fn build_ed25519_self_signed_cert(
    signing_key: &ed25519_dalek::SigningKey,
) -> CertificateDer<'static> {
    use ed25519_dalek::Signer;
    use rand::Rng;

    let public_key = signing_key.verifying_key().to_bytes();

    // version [0] EXPLICIT INTEGER 2 (v3).
    let version = der_tlv(0xA0, &der_tlv(0x02, &[0x02]));

    // serialNumber: positive INTEGER (mask the high bit to guarantee positive).
    let mut serial_bytes = [0u8; 8];
    rand::rng().fill_bytes(&mut serial_bytes);
    serial_bytes[0] &= 0x7F;
    if serial_bytes[0] == 0 {
        serial_bytes[0] = 0x01;
    }
    let serial = der_tlv(0x02, &serial_bytes);

    let name = der_name_cn("OxiMedia WebRTC (experimental, not for production)");

    // TBSCertificate.
    let mut tbs_contents = Vec::new();
    tbs_contents.extend_from_slice(&version);
    tbs_contents.extend_from_slice(&serial);
    tbs_contents.extend_from_slice(&ED25519_ALG_ID); // signature algorithm
    tbs_contents.extend_from_slice(&name); // issuer
    tbs_contents.extend_from_slice(&der_validity());
    tbs_contents.extend_from_slice(&name); // subject (self-signed)
    tbs_contents.extend_from_slice(&der_spki(&public_key));
    let tbs = der_tlv(0x30, &tbs_contents);

    // signatureValue: Ed25519 signature over the DER-encoded TBSCertificate.
    let signature = signing_key.sign(&tbs).to_bytes();
    let mut sig_bit = Vec::with_capacity(65);
    sig_bit.push(0x00); // unused bits
    sig_bit.extend_from_slice(&signature);
    let signature_value = der_tlv(0x03, &sig_bit);

    // Certificate ::= SEQUENCE { tbsCertificate, signatureAlgorithm, signature }.
    let mut cert_contents = Vec::new();
    cert_contents.extend_from_slice(&tbs);
    cert_contents.extend_from_slice(&ED25519_ALG_ID);
    cert_contents.extend_from_slice(&signature_value);
    let cert_der = der_tlv(0x30, &cert_contents);

    CertificateDer::from(cert_der)
}

// =============================================
// Extended DTLS simulation types
// =============================================

/// DTLS cipher suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlsCipherSuite {
    /// TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256 (16-byte key).
    TlsEcdhEcdsaWithAes128GcmSha256,
    /// TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384 (32-byte key).
    TlsEcdhEcdsaWithAes256GcmSha384,
    /// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 (16-byte key).
    TlsEcdheRsaWithAes128GcmSha256,
}

impl DtlsCipherSuite {
    /// Returns the key length in bytes for this cipher suite.
    #[must_use]
    pub const fn key_length_bytes(self) -> usize {
        match self {
            Self::TlsEcdhEcdsaWithAes128GcmSha256 => 16,
            Self::TlsEcdhEcdsaWithAes256GcmSha384 => 32,
            Self::TlsEcdheRsaWithAes128GcmSha256 => 16,
        }
    }
}

/// DTLS handshake state for the simulated session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlsHandshakeState {
    /// New, not yet started.
    New,
    /// Handshake in progress.
    Connecting,
    /// Handshake complete and connection established.
    Connected,
    /// Connection closed.
    Closed,
    /// Handshake failed.
    Failed,
}

/// Fingerprint hash algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerprintAlgorithm {
    /// SHA-256.
    Sha256,
    /// SHA-384.
    Sha384,
    /// SHA-512.
    Sha512,
}

impl FingerprintAlgorithm {
    /// Returns the algorithm name as used in SDP fingerprint attributes.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sha256 => "sha-256",
            Self::Sha384 => "sha-384",
            Self::Sha512 => "sha-512",
        }
    }
}

/// A DTLS fingerprint with algorithm and hex value.
#[derive(Debug, Clone)]
pub struct DtlsFingerprintTyped {
    /// Hash algorithm.
    pub algorithm: FingerprintAlgorithm,
    /// Fingerprint hex value (colon-separated octets).
    pub value: String,
}

impl DtlsFingerprintTyped {
    /// Formats the fingerprint for an SDP attribute.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        format!("{} {}", self.algorithm.name(), self.value)
    }
}

/// Simulated DTLS session (state-machine only, **no real crypto**).
///
/// # ⚠️ Non-cryptographic — test/simulation use only
///
/// This type models the DTLS handshake *state machine* for unit tests. Its
/// [`protect`](Self::protect) / [`unprotect`](Self::unprotect) methods use a
/// trivial XOR and provide **no confidentiality or integrity**. It must never
/// be used to protect real media. It is not part of the (unimplemented)
/// production DTLS path and is not re-exported from the crate.
#[derive(Debug)]
pub struct DtlsSession {
    /// Role (client or server).
    pub role: DtlsRole,
    /// Current handshake state.
    pub state: DtlsHandshakeState,
    /// Local fingerprint.
    pub local_fingerprint: DtlsFingerprintTyped,
    /// Remote fingerprint (set after receiving remote's Hello).
    pub remote_fingerprint: Option<DtlsFingerprintTyped>,
    /// Negotiated cipher suite.
    pub cipher_suite: Option<DtlsCipherSuite>,
    /// Simple session key derived at connect time (for simulation XOR).
    session_key: Vec<u8>,
}

impl DtlsSession {
    /// Creates a new DTLS session with a simulated fingerprint.
    #[must_use]
    pub fn new(role: DtlsRole) -> Self {
        // Generate a deterministic simulated fingerprint from role + timestamp.
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0xDEAD_BEEF);

        // Build a fake 32-byte fingerprint.
        let mut bytes = Vec::with_capacity(32);
        for i in 0..32u8 {
            bytes.push(
                ((ts >> (i % 8)) as u8)
                    .wrapping_add(i)
                    .wrapping_add(if role == DtlsRole::Client { 0xAA } else { 0x55 }),
            );
        }
        let value = bytes
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(":");

        Self {
            role,
            state: DtlsHandshakeState::New,
            local_fingerprint: DtlsFingerprintTyped {
                algorithm: FingerprintAlgorithm::Sha256,
                value,
            },
            remote_fingerprint: None,
            cipher_suite: None,
            session_key: vec![0u8; 16],
        }
    }

    /// Performs the simulated DTLS handshake, transitioning to Connected.
    ///
    /// Returns `true` if the handshake succeeds.
    pub fn connect(&mut self) -> bool {
        self.state = DtlsHandshakeState::Connecting;

        // Select cipher suite.
        self.cipher_suite = Some(DtlsCipherSuite::TlsEcdhEcdsaWithAes128GcmSha256);

        // Derive a simple session key (deterministic for simulation).
        let key_len = self
            .cipher_suite
            .map(|cs| cs.key_length_bytes())
            .unwrap_or(16);
        self.session_key = (0..key_len as u8).collect();

        self.state = DtlsHandshakeState::Connected;
        true
    }

    /// Obfuscates data with a simple XOR (simulation only — **not** encryption).
    ///
    /// Provides no confidentiality or integrity; see the type-level warning.
    #[must_use]
    pub fn protect(&self, data: &[u8]) -> Vec<u8> {
        if self.session_key.is_empty() {
            return data.to_vec();
        }
        data.iter()
            .enumerate()
            .map(|(i, &b)| b ^ self.session_key[i % self.session_key.len()])
            .collect()
    }

    /// Unprotects (decrypts) data with the same XOR (symmetric).
    #[must_use]
    pub fn unprotect(&self, data: &[u8]) -> Vec<u8> {
        // XOR is its own inverse.
        self.protect(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtls_role() {
        assert_eq!(DtlsRole::from_setup("active"), Some(DtlsRole::Client));
        assert_eq!(DtlsRole::from_setup("passive"), Some(DtlsRole::Server));
        assert_eq!(DtlsRole::Client.to_setup(), "active");
    }

    #[test]
    fn test_fingerprint_from_cert() {
        let (cert, _) = generate_self_signed_cert().expect("should succeed in test");
        let fp = DtlsFingerprint::from_certificate(&cert);
        assert_eq!(fp.algorithm, "sha-256");
        assert!(fp.value.contains(':'));
    }

    /// Reads a single DER TLV, returning `(tag, contents, rest)`.
    fn der_read_tlv(buf: &[u8]) -> Option<(u8, &[u8], &[u8])> {
        if buf.len() < 2 {
            return None;
        }
        let tag = buf[0];
        let first = buf[1];
        let (len, header) = if first < 0x80 {
            (first as usize, 2usize)
        } else {
            let n = (first & 0x7F) as usize;
            if n == 0 || n > 4 || buf.len() < 2 + n {
                return None;
            }
            let mut l = 0usize;
            for &b in &buf[2..2 + n] {
                l = (l << 8) | b as usize;
            }
            (l, 2 + n)
        };
        if buf.len() < header + len {
            return None;
        }
        Some((tag, &buf[header..header + len], &buf[header + len..]))
    }

    /// The generated certificate must be a genuine, self-consistent Ed25519
    /// self-signed X.509 certificate: it parses cleanly and its embedded
    /// signature verifies against its embedded public key. This proves we no
    /// longer emit the old 4-byte non-parseable dummy certificate.
    #[test]
    fn test_self_signed_cert_is_valid_ed25519() {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let (cert, _key) = generate_self_signed_cert().expect("cert generation");
        let der = cert.as_ref();

        // Outer Certificate SEQUENCE with no trailing bytes.
        let (tag, cert_contents, rest) = der_read_tlv(der).expect("outer TLV");
        assert_eq!(tag, 0x30, "certificate must be a SEQUENCE");
        assert!(rest.is_empty(), "no trailing bytes after certificate");

        // tbsCertificate is the first element; keep its raw (signed) bytes.
        let (tbs_tag, tbs_contents, after_tbs) = der_read_tlv(cert_contents).expect("tbs");
        assert_eq!(tbs_tag, 0x30, "tbsCertificate must be a SEQUENCE");
        let tbs_raw = &cert_contents[..cert_contents.len() - after_tbs.len()];

        // signatureAlgorithm, then signatureValue BIT STRING.
        let (_alg_tag, _alg, after_alg) = der_read_tlv(after_tbs).expect("sigAlgorithm");
        let (sig_tag, sig_bits, sig_rest) = der_read_tlv(after_alg).expect("signatureValue");
        assert_eq!(sig_tag, 0x03, "signatureValue must be a BIT STRING");
        assert!(sig_rest.is_empty(), "no trailing bytes after signature");
        assert_eq!(sig_bits[0], 0x00, "BIT STRING unused-bits octet");
        let sig_bytes = &sig_bits[1..];
        assert_eq!(sig_bytes.len(), 64, "Ed25519 signature is 64 bytes");

        // Walk the tbsCertificate fields; the SPKI is the last one.
        let mut cursor = tbs_contents;
        let mut last_field: Option<&[u8]> = None;
        while let Some((_t, contents, next)) = der_read_tlv(cursor) {
            last_field = Some(contents);
            if next.is_empty() {
                break;
            }
            cursor = next;
        }
        let spki = last_field.expect("SPKI present");

        // SPKI = SEQUENCE { AlgorithmIdentifier, BIT STRING publicKey }.
        let (_spki_alg_tag, _spki_alg, after_spki_alg) = der_read_tlv(spki).expect("spki alg");
        let (pk_tag, pk_bits, _) = der_read_tlv(after_spki_alg).expect("spki bitstring");
        assert_eq!(pk_tag, 0x03, "public key must be a BIT STRING");
        assert_eq!(pk_bits[0], 0x00, "BIT STRING unused-bits octet");
        let pk = &pk_bits[1..];
        assert_eq!(pk.len(), 32, "Ed25519 public key is 32 bytes");

        let mut pk_arr = [0u8; 32];
        pk_arr.copy_from_slice(pk);
        let vk = VerifyingKey::from_bytes(&pk_arr).expect("valid Ed25519 public key");
        let signature = Signature::from_slice(sig_bytes).expect("valid signature encoding");
        vk.verify(tbs_raw, &signature)
            .expect("self-signature must verify over the tbsCertificate");
    }

    /// The handshake must fail loudly instead of fabricating a connection with
    /// all-zero SRTP keys (which would transmit media in plaintext).
    #[tokio::test]
    async fn test_handshake_refuses_fake_connection() {
        let config = DtlsConfig::new_self_signed(DtlsRole::Client).expect("config");
        let socket = Arc::new(
            tokio::net::UdpSocket::bind("127.0.0.1:0")
                .await
                .expect("bind"),
        );
        let endpoint = DtlsEndpoint::new(config, socket.clone());

        let Err(err) = endpoint.handshake().await else {
            panic!("handshake must not fabricate a DTLS connection");
        };
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("not implemented") && msg.contains("dtls"),
            "handshake error should explain DTLS is unimplemented: {msg}"
        );

        // A DtlsConnection (only constructible in tests) must refuse to send or
        // receive rather than leak plaintext under a DTLS-protected label.
        let conn = DtlsConnection { socket };
        let Err(send_err) = conn.send(b"confidential media payload").await else {
            panic!("send must refuse when DTLS-SRTP is unimplemented");
        };
        assert!(
            send_err.to_string().to_lowercase().contains("plaintext"),
            "send refusal should mention plaintext: {send_err}"
        );

        let mut buf = [0u8; 32];
        let Err(recv_err) = conn.recv(&mut buf).await else {
            panic!("recv must refuse when DTLS-SRTP is unimplemented");
        };
        assert!(recv_err
            .to_string()
            .to_lowercase()
            .contains("not implemented"));
    }

    #[test]
    fn test_fingerprint_sdp() {
        let fp = DtlsFingerprint {
            algorithm: "sha-256".to_string(),
            value: "AA:BB:CC:DD".to_string(),
        };
        assert_eq!(fp.to_sdp(), "sha-256 AA:BB:CC:DD");
    }

    // ---- Extended DTLS simulation tests ----

    #[test]
    fn test_cipher_suite_key_length() {
        assert_eq!(
            DtlsCipherSuite::TlsEcdhEcdsaWithAes128GcmSha256.key_length_bytes(),
            16
        );
        assert_eq!(
            DtlsCipherSuite::TlsEcdhEcdsaWithAes256GcmSha384.key_length_bytes(),
            32
        );
        assert_eq!(
            DtlsCipherSuite::TlsEcdheRsaWithAes128GcmSha256.key_length_bytes(),
            16
        );
    }

    #[test]
    fn test_fingerprint_algorithm_name() {
        assert_eq!(FingerprintAlgorithm::Sha256.name(), "sha-256");
        assert_eq!(FingerprintAlgorithm::Sha384.name(), "sha-384");
        assert_eq!(FingerprintAlgorithm::Sha512.name(), "sha-512");
    }

    #[test]
    fn test_dtls_fingerprint_typed_sdp() {
        let fp = DtlsFingerprintTyped {
            algorithm: FingerprintAlgorithm::Sha256,
            value: "AA:BB:CC".to_string(),
        };
        assert_eq!(fp.to_sdp(), "sha-256 AA:BB:CC");
    }

    #[test]
    fn test_dtls_session_new_client() {
        let session = DtlsSession::new(DtlsRole::Client);
        assert_eq!(session.role, DtlsRole::Client);
        assert_eq!(session.state, DtlsHandshakeState::New);
        assert!(session.cipher_suite.is_none());
        assert!(session.remote_fingerprint.is_none());
        assert!(!session.local_fingerprint.value.is_empty());
    }

    #[test]
    fn test_dtls_session_new_server() {
        let session = DtlsSession::new(DtlsRole::Server);
        assert_eq!(session.role, DtlsRole::Server);
        assert_eq!(session.state, DtlsHandshakeState::New);
    }

    #[test]
    fn test_dtls_session_connect() {
        let mut session = DtlsSession::new(DtlsRole::Client);
        let ok = session.connect();
        assert!(ok);
        assert_eq!(session.state, DtlsHandshakeState::Connected);
        assert!(session.cipher_suite.is_some());
    }

    #[test]
    fn test_dtls_session_protect_unprotect() {
        let mut session = DtlsSession::new(DtlsRole::Client);
        session.connect();

        let original = b"Hello DTLS world!";
        let protected = session.protect(original);
        assert_ne!(protected, original.to_vec());

        let unprotected = session.unprotect(&protected);
        assert_eq!(unprotected, original.to_vec());
    }

    #[test]
    fn test_dtls_session_protect_empty() {
        let mut session = DtlsSession::new(DtlsRole::Server);
        session.connect();

        let protected = session.protect(b"");
        assert!(protected.is_empty());
    }

    #[test]
    fn test_dtls_handshake_state_variants() {
        let states = [
            DtlsHandshakeState::New,
            DtlsHandshakeState::Connecting,
            DtlsHandshakeState::Connected,
            DtlsHandshakeState::Closed,
            DtlsHandshakeState::Failed,
        ];
        for &s in &states {
            assert_eq!(s, s);
        }
    }
}
