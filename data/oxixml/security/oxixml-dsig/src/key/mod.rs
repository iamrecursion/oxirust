//! Keys, and the policy that chooses one.
//!
//! Verifying an XML signature needs a key, and the signature itself is not
//! allowed to decide which one. `<ds:KeyInfo>` is, in the words of
//! [XML Signature §4.5](https://www.w3.org/TR/xmldsig-core1/#sec-KeyInfo), "an
//! optional element that enables the recipient(s) to obtain the key" — a hint
//! from the signer, carried inside the very document whose authenticity is in
//! question. Believing it uncritically is the single most common way to build
//! a verifier that validates every forgery handed to it.
//!
//! This crate therefore never picks a key on its own. Every verification runs
//! through a [`KeySelector`], which is where an application puts its trust
//! decision: a certificate store, a pinned public key, a shared HMAC secret, a
//! lookup by `<ds:KeyName>`. Four implementations are provided:
//!
//! * [`FixedKey`] — always the same key, ignoring `KeyInfo`. The safe default
//!   for a pinned peer.
//! * [`SharedSecret`] — an HMAC secret; refuses public-key methods.
//! * [`TrustedCertificates`] — accepts an embedded certificate only if its DER
//!   is byte-identical to one the caller listed, then reads out the RSA,
//!   elliptic-curve or DSA key it certifies, whichever the method needs. Given
//!   a [`PathValidator`](crate::PathValidator) it instead treats the listed
//!   certificates as trust anchors and validates the whole certification path
//!   the `<ds:X509Data>` carries.
//! * [`UntrustedKeyInfo`] — takes whatever public key the signature carries.
//!   Deliberately awkward to name, because it proves only that *whoever wrote
//!   the document* had the private key, which is what "self-signed" means.
//!
//! Any `Fn(&KeyContext<'_>) -> DsigResult<VerificationKey>` is also a
//! `KeySelector`, so a closure works where none of the four fit.
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{KeyContext, KeySelector, SharedSecret, VerificationKey};
//! use oxixml_dsig::SignatureAlgorithm;
//!
//! let selector = SharedSecret::new(b"secret".to_vec());
//! let context = KeyContext::new(SignatureAlgorithm::HmacSha256, None);
//! assert!(matches!(
//!     selector.select(&context)?,
//!     VerificationKey::Hmac(secret) if secret == b"secret",
//! ));
//!
//! // …but it will not pretend to have an RSA key.
//! let rsa = KeyContext::new(SignatureAlgorithm::RsaSha256, None);
//! assert!(selector.select(&rsa).is_err());
//! # Ok::<(), oxixml_dsig::DsigError>(())
//! ```

mod dsa;
mod ec;
mod keyinfo;
mod rsa;

pub use dsa::{
    DsaPrivateKey, DsaPublicKey, MAXIMUM_DSA_PRIME_BITS, MAXIMUM_DSA_SUBGROUP_BITS,
    MINIMUM_DSA_PRIME_BITS, MINIMUM_DSA_SUBGROUP_BITS,
};
pub use ec::{
    EcPrivateKey, EcPublicKey, NamedCurve, P256_NAMED_CURVE_URI, P384_NAMED_CURVE_URI,
    named_curve_from_uri, named_curve_uri,
};
pub use keyinfo::{IssuerSerial, KeyInfo, KeyInfoItem, X509Data};
pub use rsa::{
    MAXIMUM_RSA_MODULUS_BITS, MAXIMUM_RSA_PUBLIC_EXPONENT_BITS, MINIMUM_RSA_MODULUS_BITS,
    RsaPrivateKey, RsaPublicKey,
};

use crate::algorithm::{SignatureAlgorithm, SignatureFamily};
use crate::error::{DsigError, DsigResult};
use crate::x509::{Certificate, CertificateError, PathValidator};

/// A key that can *check* a signature.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerificationKey {
    /// A shared secret for an HMAC method. Whoever can verify with it can also
    /// forge with it.
    Hmac(Vec<u8>),
    /// An RSA public key, for both RSASSA-PKCS1-v1_5 and RSASSA-PSS.
    Rsa(RsaPublicKey),
    /// An elliptic-curve public key, for ECDSA.
    Ec(EcPublicKey),
    /// A DSA public key.
    Dsa(DsaPublicKey),
}

impl VerificationKey {
    /// Whether this key can be used with `algorithm`.
    ///
    /// Matching the *family* is necessary but not sufficient: a P-256 key cannot
    /// check a signature made on P-384 either, and that is caught when the
    /// signature value turns out to be the wrong width.
    #[must_use]
    pub const fn matches(&self, algorithm: SignatureAlgorithm) -> bool {
        match self {
            Self::Hmac(_) => matches!(algorithm.family(), SignatureFamily::Hmac),
            Self::Rsa(_) => matches!(
                algorithm.family(),
                SignatureFamily::RsaPkcs1V15 | SignatureFamily::RsaPss
            ),
            Self::Ec(_) => matches!(algorithm.family(), SignatureFamily::Ecdsa),
            Self::Dsa(_) => matches!(algorithm.family(), SignatureFamily::Dsa),
        }
    }
}

/// A key that can *produce* a signature.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SigningKey {
    /// A shared secret for an HMAC method.
    Hmac(Vec<u8>),
    /// An RSA private key, for both RSASSA-PKCS1-v1_5 and RSASSA-PSS.
    Rsa(Box<RsaPrivateKey>),
    /// An elliptic-curve private key, for ECDSA.
    Ec(Box<EcPrivateKey>),
    /// A DSA private key.
    Dsa(Box<DsaPrivateKey>),
}

impl SigningKey {
    /// Wraps an RSA private key.
    #[must_use]
    pub fn rsa(key: RsaPrivateKey) -> Self {
        Self::Rsa(Box::new(key))
    }

    /// Wraps an elliptic-curve private key.
    #[must_use]
    pub fn ec(key: EcPrivateKey) -> Self {
        Self::Ec(Box::new(key))
    }

    /// Wraps a DSA private key.
    #[must_use]
    pub fn dsa(key: DsaPrivateKey) -> Self {
        Self::Dsa(Box::new(key))
    }

    /// Wraps an HMAC secret.
    #[must_use]
    pub fn hmac(secret: impl Into<Vec<u8>>) -> Self {
        Self::Hmac(secret.into())
    }

    /// Whether this key can be used with `algorithm`.
    #[must_use]
    pub const fn matches(&self, algorithm: SignatureAlgorithm) -> bool {
        match self {
            Self::Hmac(_) => matches!(algorithm.family(), SignatureFamily::Hmac),
            Self::Rsa(_) => matches!(
                algorithm.family(),
                SignatureFamily::RsaPkcs1V15 | SignatureFamily::RsaPss
            ),
            Self::Ec(_) => matches!(algorithm.family(), SignatureFamily::Ecdsa),
            Self::Dsa(_) => matches!(algorithm.family(), SignatureFamily::Dsa),
        }
    }

    /// The verification key that matches this signing key: the secret itself for
    /// HMAC, and the public half for every public-key algorithm.
    #[must_use]
    pub fn verification_key(&self) -> VerificationKey {
        match self {
            Self::Hmac(secret) => VerificationKey::Hmac(secret.clone()),
            Self::Rsa(key) => VerificationKey::Rsa(key.public_key()),
            Self::Ec(key) => VerificationKey::Ec(key.public_key()),
            Self::Dsa(key) => VerificationKey::Dsa(key.public_key()),
        }
    }
}

/// Everything a [`KeySelector`] gets to look at.
#[derive(Debug, Clone, Copy)]
pub struct KeyContext<'a> {
    algorithm: SignatureAlgorithm,
    key_info: Option<&'a KeyInfo>,
}

impl<'a> KeyContext<'a> {
    /// Builds a context.
    #[must_use]
    pub const fn new(algorithm: SignatureAlgorithm, key_info: Option<&'a KeyInfo>) -> Self {
        Self {
            algorithm,
            key_info,
        }
    }

    /// The `<ds:SignatureMethod>` the signature declares.
    #[must_use]
    pub const fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm
    }

    /// The signature's `<ds:KeyInfo>`, if it has one.
    ///
    /// Any same-document `<dsig11:KeyInfoReference>` has already been followed,
    /// so this is the *effective* `KeyInfo`.
    #[must_use]
    pub const fn key_info(&self) -> Option<&'a KeyInfo> {
        self.key_info
    }
}

/// Chooses the key a signature is checked against.
///
/// This is a trait, rather than something the crate decides for you, because
/// choosing a key *is* the trust decision — see
/// [the security notes](crate#security).
pub trait KeySelector {
    /// Returns the key to verify with.
    ///
    /// # Errors
    ///
    /// [`DsigError::NoKey`] when the selector declines — which is what a
    /// rejected `KeyInfo` looks like, and is *not* the same as a signature that
    /// fails to verify.
    fn select(&self, context: &KeyContext<'_>) -> DsigResult<VerificationKey>;
}

impl<F> KeySelector for F
where
    F: Fn(&KeyContext<'_>) -> DsigResult<VerificationKey>,
{
    fn select(&self, context: &KeyContext<'_>) -> DsigResult<VerificationKey> {
        self(context)
    }
}

/// Always returns the same key, whatever the `KeyInfo` claims.
///
/// This is the right selector whenever the peer's key is already known — which
/// is most of the time, and is the only configuration in which a successful
/// verification says anything about *who* signed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedKey {
    key: VerificationKey,
}

impl FixedKey {
    /// Pins `key`.
    #[must_use]
    pub fn new(key: VerificationKey) -> Self {
        Self { key }
    }

    /// Pins an RSA public key.
    #[must_use]
    pub fn rsa(key: RsaPublicKey) -> Self {
        Self {
            key: VerificationKey::Rsa(key),
        }
    }

    /// Pins an elliptic-curve public key.
    #[must_use]
    pub fn ec(key: EcPublicKey) -> Self {
        Self {
            key: VerificationKey::Ec(key),
        }
    }

    /// Pins a DSA public key.
    #[must_use]
    pub fn dsa(key: DsaPublicKey) -> Self {
        Self {
            key: VerificationKey::Dsa(key),
        }
    }
}

impl KeySelector for FixedKey {
    fn select(&self, context: &KeyContext<'_>) -> DsigResult<VerificationKey> {
        if self.key.matches(context.algorithm()) {
            Ok(self.key.clone())
        } else {
            Err(DsigError::NoKey(format!(
                "the pinned key cannot be used with {}",
                context.algorithm()
            )))
        }
    }
}

/// A shared secret for the HMAC methods.
///
/// Declines every public-key method, because an HMAC secret is not a
/// verification key for one and silently succeeding would be worse than
/// failing.
#[derive(Clone, PartialEq, Eq)]
pub struct SharedSecret {
    secret: Vec<u8>,
}

impl SharedSecret {
    /// Wraps a secret.
    #[must_use]
    pub fn new(secret: Vec<u8>) -> Self {
        Self { secret }
    }
}

impl std::fmt::Debug for SharedSecret {
    /// Prints the length and nothing else.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedSecret")
            .field("len", &self.secret.len())
            .finish()
    }
}

impl KeySelector for SharedSecret {
    fn select(&self, context: &KeyContext<'_>) -> DsigResult<VerificationKey> {
        if context.algorithm().is_hmac() {
            Ok(VerificationKey::Hmac(self.secret.clone()))
        } else {
            Err(DsigError::NoKey(format!(
                "a shared secret cannot verify {}",
                context.algorithm()
            )))
        }
    }
}

/// A set of certificates the caller trusts, used in either of two ways.
///
/// **Pinned, by default.** An embedded `<ds:X509Certificate>` is accepted only
/// when its DER encoding is byte-for-byte one of the certificates supplied to
/// [`with_certificate`](TrustedCertificates::with_certificate). This is the
/// cheapest honest way to use `KeyInfo`: the signature may say *which* of
/// several known certificates it used, but it cannot introduce a new one.
///
/// **As trust anchors, given a [`PathValidator`].** With
/// [`with_path_validation`](TrustedCertificates::with_path_validation) the
/// stored certificates become anchors, and any certificate that chains to one
/// of them is accepted — after full
/// [RFC 5280 §6.1](https://www.rfc-editor.org/rfc/rfc5280#section-6.1) path
/// validation of every certificate in between. This is what makes an ordinary
/// certificate authority usable: the signer's own certificate need not be known
/// in advance, only its issuer's issuer.
///
/// A chain that fails validation is an error naming the reason. Neither mode
/// ever falls back to "read the key out of the leaf anyway" — that is
/// [`UntrustedKeyInfo`], and it is spelled differently on purpose.
///
/// # Example
///
/// ```
/// use oxixml_dsig::{PathValidator, Timestamp, TrustedCertificates};
///
/// # let root_der: Vec<u8> = vec![];
/// let trust = TrustedCertificates::new()
///     .with_certificate(root_der)
///     .with_path_validation(PathValidator::new(Timestamp::from_unix_seconds(1_800_000_000)));
/// assert!(trust.path_validator().is_some());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrustedCertificates {
    certificates: Vec<Vec<u8>>,
    validator: Option<PathValidator>,
}

impl TrustedCertificates {
    /// An empty store, which accepts nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            certificates: Vec::new(),
            validator: None,
        }
    }

    /// Adds a DER certificate to the trusted set.
    #[must_use]
    pub fn with_certificate(mut self, der: impl Into<Vec<u8>>) -> Self {
        self.certificates.push(der.into());
        self
    }

    /// Treats the stored certificates as trust anchors and validates the
    /// `<ds:X509Data>` chain against them with `validator`.
    ///
    /// The validator carries the time the validity periods are checked at,
    /// which this crate never takes from a clock — see
    /// [`Timestamp`](crate::Timestamp).
    #[must_use]
    pub const fn with_path_validation(mut self, validator: PathValidator) -> Self {
        self.validator = Some(validator);
        self
    }

    /// The path validator, if one was configured.
    #[must_use]
    pub const fn path_validator(&self) -> Option<&PathValidator> {
        self.validator.as_ref()
    }

    /// Builds and validates a path from the `KeyInfo`'s certificates to one of
    /// the anchors, and returns the validated leaf's key.
    fn validated_key(
        &self,
        validator: &PathValidator,
        key_info: &KeyInfo,
        algorithm: SignatureAlgorithm,
    ) -> DsigResult<VerificationKey> {
        // Counted before anything is parsed: the certificates come out of the
        // document being checked, and refusing an absurd number of them costs
        // nothing while parsing them would.
        let offered = key_info.certificates().count();
        if offered > validator.certificate_limit() {
            return Err(CertificateError::TooManyCertificates {
                certificates: offered,
                limit: validator.certificate_limit(),
            }
            .into());
        }
        let chain = key_info
            .certificates()
            .map(Certificate::parse)
            .collect::<Result<Vec<_>, _>>()
            .map_err(DsigError::from)?;
        let anchors = self
            .certificates
            .iter()
            .map(|der| Certificate::parse(der))
            .collect::<Result<Vec<_>, _>>()
            .map_err(DsigError::from)?;
        // Which certificate is the signer's is not recorded anywhere in the
        // Recommendation, so it is looked for: a certificate that certifies a
        // key of the kind this signature method needs and that issued nothing
        // else in the chain. Failing that, the algorithm alone decides, and
        // failing that the structural rule does.
        let leaf = chain
            .iter()
            .find(|certificate| {
                certifies_key_for(certificate, algorithm)
                    && !chain.iter().any(|other| {
                        !std::ptr::eq(other, *certificate)
                            && other.issuer().matches(certificate.subject())
                    })
            })
            .or_else(|| {
                chain
                    .iter()
                    .find(|certificate| certifies_key_for(certificate, algorithm))
            })
            .or_else(|| Certificate::select_end_entity(&chain))
            .ok_or(CertificateError::NoCertificates)?;
        let path = validator.validate_leaf(leaf, &chain, &anchors)?;
        Ok(path.public_key().clone())
    }
}

impl KeySelector for TrustedCertificates {
    fn select(&self, context: &KeyContext<'_>) -> DsigResult<VerificationKey> {
        let algorithm = context.algorithm();
        if algorithm.is_hmac() {
            return Err(DsigError::NoKey(format!(
                "a certificate store cannot verify {algorithm}, whose key is a shared secret"
            )));
        }
        let key_info = context
            .key_info()
            .ok_or_else(|| DsigError::NoKey("the signature carries no KeyInfo".to_owned()))?;
        if let Some(validator) = &self.validator {
            let key = self.validated_key(validator, key_info, algorithm)?;
            if !key.matches(algorithm) {
                return Err(DsigError::NoKey(format!(
                    "the validated certificate does not certify a key {algorithm} can use"
                )));
            }
            return Ok(key);
        }
        for der in key_info.certificates() {
            if self.certificates.iter().any(|trusted| trusted == der) {
                return certified_key(der, algorithm);
            }
        }
        Err(DsigError::NoKey(
            "no certificate in the KeyInfo is in the trusted set".to_owned(),
        ))
    }
}

/// Whether a certificate certifies a key `algorithm` can be verified with.
fn certifies_key_for(certificate: &Certificate, algorithm: SignatureAlgorithm) -> bool {
    certificate
        .public_key()
        .is_some_and(|key| key.matches(algorithm))
}

/// Reads the public key a trusted certificate certifies, as the kind
/// `algorithm` needs.
fn certified_key(der: &[u8], algorithm: SignatureAlgorithm) -> DsigResult<VerificationKey> {
    match algorithm.family() {
        SignatureFamily::RsaPkcs1V15 | SignatureFamily::RsaPss => {
            RsaPublicKey::from_x509_certificate(der).map(VerificationKey::Rsa)
        }
        SignatureFamily::Ecdsa => EcPublicKey::from_x509_certificate(der).map(VerificationKey::Ec),
        SignatureFamily::Dsa => DsaPublicKey::from_x509_certificate(der).map(VerificationKey::Dsa),
        SignatureFamily::Hmac => Err(DsigError::NoKey(format!(
            "a certificate cannot carry the key {algorithm} needs"
        ))),
    }
}

/// Takes whatever public key the signature carries, with no trust decision at
/// all.
///
/// A signature verified this way proves only *internal consistency*: the
/// document was signed by whoever holds the private half of the key printed
/// inside the same document. An attacker who can rewrite the document can
/// replace the key and the signature together, and this selector will happily
/// say "valid".
///
/// It is exactly right for two jobs — testing against published interop
/// vectors, and checking a signature whose key you then compare against a
/// trusted one yourself — and wrong for everything else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UntrustedKeyInfo;

impl KeySelector for UntrustedKeyInfo {
    fn select(&self, context: &KeyContext<'_>) -> DsigResult<VerificationKey> {
        let algorithm = context.algorithm();
        let key_info = context.key_info();
        let (kind, key) = match algorithm.family() {
            SignatureFamily::Hmac => {
                return Err(DsigError::NoKey(format!(
                    "{algorithm} needs a key that KeyInfo cannot carry in the clear"
                )));
            }
            SignatureFamily::RsaPkcs1V15 | SignatureFamily::RsaPss => (
                "RSA",
                key_info
                    .and_then(KeyInfo::rsa_public_key)
                    .map(VerificationKey::Rsa),
            ),
            SignatureFamily::Ecdsa => (
                "elliptic-curve",
                key_info
                    .and_then(KeyInfo::ec_public_key)
                    .map(VerificationKey::Ec),
            ),
            SignatureFamily::Dsa => (
                "DSA",
                key_info
                    .and_then(KeyInfo::dsa_public_key)
                    .map(VerificationKey::Dsa),
            ),
        };
        key.ok_or_else(|| {
            DsigError::NoKey(format!("the KeyInfo carries no usable {kind} public key"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey;

    fn public_key() -> DsigResult<RsaPublicKey> {
        RsaPublicKey::from_components(
            &testkey::bytes(testkey::MODULUS_BASE64),
            &testkey::bytes(testkey::PUBLIC_EXPONENT_BASE64),
        )
    }

    #[test]
    fn a_shared_secret_refuses_public_key_methods() -> DsigResult<()> {
        let selector = SharedSecret::new(b"secret".to_vec());
        let hmac = KeyContext::new(SignatureAlgorithm::HmacSha1, None);
        assert_eq!(
            selector.select(&hmac)?,
            VerificationKey::Hmac(b"secret".to_vec())
        );
        let rsa = KeyContext::new(SignatureAlgorithm::RsaSha1, None);
        assert!(matches!(selector.select(&rsa), Err(DsigError::NoKey(_))));
        Ok(())
    }

    #[test]
    fn a_pinned_key_refuses_the_wrong_family() -> DsigResult<()> {
        let selector = FixedKey::rsa(public_key()?);
        assert!(
            selector
                .select(&KeyContext::new(SignatureAlgorithm::RsaSha256, None))
                .is_ok()
        );
        assert!(
            selector
                .select(&KeyContext::new(SignatureAlgorithm::HmacSha256, None))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn an_untrusted_selector_reads_the_embedded_key() -> DsigResult<()> {
        let key_info = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::RsaKeyValue {
                modulus: testkey::bytes(testkey::MODULUS_BASE64),
                exponent: testkey::bytes(testkey::PUBLIC_EXPONENT_BASE64),
            }],
        };
        let context = KeyContext::new(SignatureAlgorithm::RsaSha256, Some(&key_info));
        assert_eq!(
            UntrustedKeyInfo.select(&context)?,
            VerificationKey::Rsa(public_key()?)
        );
        Ok(())
    }

    #[test]
    fn a_certificate_store_rejects_an_unknown_certificate() -> DsigResult<()> {
        let key_info = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::X509Data(X509Data {
                certificates: vec![testkey::bytes(testkey::CERTIFICATE_BASE64)],
                ..X509Data::default()
            })],
        };
        let context = KeyContext::new(SignatureAlgorithm::RsaSha256, Some(&key_info));

        let empty = TrustedCertificates::new();
        assert!(matches!(empty.select(&context), Err(DsigError::NoKey(_))));

        let trusting = TrustedCertificates::new()
            .with_certificate(testkey::bytes(testkey::CERTIFICATE_BASE64));
        assert_eq!(
            trusting.select(&context)?,
            VerificationKey::Rsa(public_key()?)
        );
        Ok(())
    }

    /// With a validator configured, the store stops being a list of acceptable
    /// certificates and becomes a list of trust anchors: a leaf nobody has ever
    /// seen before is accepted because of what issued it.
    #[test]
    fn a_certificate_store_validates_a_chain_to_an_anchor() -> DsigResult<()> {
        use crate::x509::{Timestamp, testcert};

        let chain = testcert::chain().map_err(DsigError::from)?;
        let key_info = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::X509Data(X509Data {
                certificates: vec![
                    chain.leaf.encoding().to_vec(),
                    chain.intermediate.encoding().to_vec(),
                ],
                ..X509Data::default()
            })],
        };
        let context = KeyContext::new(SignatureAlgorithm::EcdsaSha256, Some(&key_info));

        // Neither the leaf nor the intermediate is in the store, so without a
        // validator there is nothing to accept.
        let pinned = TrustedCertificates::new().with_certificate(chain.root.encoding().to_vec());
        assert!(matches!(pinned.select(&context), Err(DsigError::NoKey(_))));

        // 2030-01-01, inside the validity of every fixture certificate.
        let at = Timestamp::from_unix_seconds(1_893_456_000);
        let trusting = pinned.clone().with_path_validation(PathValidator::new(at));
        assert_eq!(
            Some(&trusting.select(&context)?),
            chain.leaf.public_key(),
            "the key must be the validated leaf's, never the anchor's"
        );

        // The same chain against a validation time outside the leaf's validity.
        let expired = pinned
            .clone()
            .with_path_validation(PathValidator::new(Timestamp::from_unix_seconds(0)));
        assert!(matches!(
            expired.select(&context),
            Err(DsigError::Certificate(_))
        ));

        // …and against a store whose certificates the chain never reaches.
        let elsewhere = TrustedCertificates::new()
            .with_certificate(chain.leaf.encoding().to_vec())
            .with_path_validation(PathValidator::new(at));
        let error = elsewhere
            .select(&context)
            .expect_err("the chain reaches no anchor in this store");
        assert!(matches!(error, DsigError::Certificate(_)), "{error}");
        Ok(())
    }

    /// A chain whose intermediate has been tampered with must be reported as a
    /// broken signature, not quietly fall back to the leaf's own key.
    #[test]
    fn a_tampered_chain_never_falls_back_to_the_leaf() -> DsigResult<()> {
        use crate::x509::{Timestamp, testcert};

        let chain = testcert::chain().map_err(DsigError::from)?;
        let key_info = KeyInfo {
            id: None,
            items: vec![KeyInfoItem::X509Data(X509Data {
                certificates: vec![
                    chain.leaf.encoding().to_vec(),
                    testcert::break_signature(chain.intermediate.encoding()),
                ],
                ..X509Data::default()
            })],
        };
        let context = KeyContext::new(SignatureAlgorithm::EcdsaSha256, Some(&key_info));
        let trusting = TrustedCertificates::new()
            .with_certificate(chain.root.encoding().to_vec())
            .with_path_validation(PathValidator::new(Timestamp::from_unix_seconds(
                1_893_456_000,
            )));
        let error = trusting
            .select(&context)
            .expect_err("the intermediate is not signed by the root");
        assert!(matches!(error, DsigError::Certificate(_)), "{error}");
        assert!(error.to_string().contains("not signed by"), "{error}");
        Ok(())
    }

    #[test]
    fn a_closure_is_a_selector() -> DsigResult<()> {
        let selector = |context: &KeyContext<'_>| {
            if context.algorithm().is_hmac() {
                Ok(VerificationKey::Hmac(b"k".to_vec()))
            } else {
                Err(DsigError::NoKey("only HMAC here".to_owned()))
            }
        };
        let context = KeyContext::new(SignatureAlgorithm::HmacSha512, None);
        assert_eq!(
            KeySelector::select(&selector, &context)?,
            VerificationKey::Hmac(b"k".to_vec())
        );
        Ok(())
    }

    #[test]
    fn a_signing_key_hands_back_its_verification_half() -> DsigResult<()> {
        let key = RsaPrivateKey::from_der(&testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64))?;
        let signing = SigningKey::rsa(key);
        assert_eq!(
            signing.verification_key(),
            VerificationKey::Rsa(public_key()?)
        );
        assert!(signing.matches(SignatureAlgorithm::RsaSha384));
        assert!(!signing.matches(SignatureAlgorithm::HmacSha384));
        Ok(())
    }

    #[test]
    fn a_shared_secret_does_not_print_itself() {
        let rendered = format!("{:?}", SharedSecret::new(b"hunter2".to_vec()));
        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(rendered.contains('7'), "{rendered}");
    }
}
