//! The `xenc:EncryptionMethod` element and the parameters it may carry.

use crate::algorithm::{BlockEncryption, KeyTransport, KeyWrap};
use crate::crypto::digest::Digest;
use crate::crypto::mgf1;
use crate::crypto::rsa::OaepParams;
use crate::model::cipher::XmlFragment;

/// The `xenc:EncryptionMethod` element of
/// [XMLEnc §3.2](https://www.w3.org/TR/xmlenc-core1/#sec-EncryptionMethod).
///
/// The element is deliberately open: it has one required `Algorithm` attribute and a content
/// model of `KeySize?`, `OAEPparams?` and then anything at all from another namespace. XML
/// Encryption 1.1 adds two children the encryption crate itself must understand — `ds:DigestMethod`
/// and `xenc11:MGF`, both of which parameterize RSA-OAEP — so those get their own fields and
/// everything else is kept verbatim in [`other`](Self::other).
///
/// # `KeySize` is advisory
///
/// XMLEnc §3.2 notes that `KeySize` is redundant for every algorithm this crate implements,
/// because the AES identifiers already fix the key length. It is parsed and re-emitted for
/// fidelity but never used to select a key length; a `KeySize` that disagrees with the
/// algorithm is reported by [`Self::key_size_disagrees_with_algorithm`] rather than silently
/// obeyed, because obeying it would let an attacker who can edit the plaintext XML downgrade a
/// 256-bit key to 128 bits.
///
/// # Example
///
/// ```
/// use oxixml_enc::algorithm::BlockEncryption;
/// use oxixml_enc::crypto::digest::Digest;
/// use oxixml_enc::model::EncryptionMethod;
///
/// let method = EncryptionMethod::block(BlockEncryption::Aes128Cbc);
/// assert_eq!(method.block_encryption(), Some(BlockEncryption::Aes128Cbc));
/// assert_eq!(method.key_size, Some(128));
/// assert!(!method.key_size_disagrees_with_algorithm());
///
/// // RSA-OAEP parameters come from three separate places in the element.
/// let method = EncryptionMethod::new("http://www.w3.org/2009/xmlenc11#rsa-oaep")
///     .with_digest_method(Digest::Sha256.uri())
///     .with_oaep_params(vec![0xf6, 0x55]);
/// let params = method.oaep_parameters()?;
/// assert_eq!(params.digest, Digest::Sha256);
/// assert_eq!(params.mgf_digest, Digest::Sha1); // the RFC 8017 default, not the digest
/// assert_eq!(params.label, vec![0xf6, 0x55]);
/// # Ok::<(), oxixml_enc::EncError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptionMethod {
    /// The required `Algorithm` attribute, kept as the URI so that an unrecognized algorithm
    /// survives a parse/serialize round trip.
    pub algorithm: String,
    /// The optional `xenc:KeySize` child, in **bits**.
    pub key_size: Option<u32>,
    /// The optional `xenc:OAEPparams` child, base64-decoded. This is the OAEP label `L`.
    pub oaep_params: Option<Vec<u8>>,
    /// The optional `ds:DigestMethod` child's `Algorithm` attribute.
    pub digest_method: Option<String>,
    /// The optional `xenc11:MGF` child's `Algorithm` attribute.
    pub mgf: Option<String>,
    /// Any further children, kept verbatim.
    pub other: Vec<XmlFragment>,
}

impl EncryptionMethod {
    /// Creates a method that names `algorithm` and carries no parameters.
    #[must_use]
    pub fn new(algorithm: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            key_size: None,
            oaep_params: None,
            digest_method: None,
            mgf: None,
            other: Vec::new(),
        }
    }

    /// Creates a method for a block-encryption algorithm, including the matching `KeySize`.
    #[must_use]
    pub fn block(algorithm: BlockEncryption) -> Self {
        let mut method = Self::new(algorithm.uri());
        method.key_size = Some(algorithm.key_bits());
        method
    }

    /// Creates a method for a symmetric key-wrap algorithm.
    #[must_use]
    pub fn key_wrap(algorithm: KeyWrap) -> Self {
        Self::new(algorithm.uri())
    }

    /// Creates a method for a public-key key-transport algorithm.
    #[must_use]
    pub fn key_transport(algorithm: KeyTransport) -> Self {
        Self::new(algorithm.uri())
    }

    /// Sets the `ds:DigestMethod` algorithm URI.
    #[must_use]
    pub fn with_digest_method(mut self, uri: impl Into<String>) -> Self {
        self.digest_method = Some(uri.into());
        self
    }

    /// Sets the `xenc11:MGF` algorithm URI.
    #[must_use]
    pub fn with_mgf(mut self, uri: impl Into<String>) -> Self {
        self.mgf = Some(uri.into());
        self
    }

    /// Sets the `xenc:OAEPparams` content, i.e. the OAEP label.
    #[must_use]
    pub fn with_oaep_params(mut self, label: impl Into<Vec<u8>>) -> Self {
        self.oaep_params = Some(label.into());
        self
    }

    /// Returns the block-encryption algorithm this method names, or `None` when it names
    /// something else or something unknown.
    #[must_use]
    pub fn block_encryption(&self) -> Option<BlockEncryption> {
        BlockEncryption::from_uri(&self.algorithm)
    }

    /// Returns the key-wrap algorithm this method names, or `None`.
    #[must_use]
    pub fn key_wrap_algorithm(&self) -> Option<KeyWrap> {
        KeyWrap::from_uri(&self.algorithm)
    }

    /// Returns the key-transport algorithm this method names, or `None`.
    #[must_use]
    pub fn key_transport_algorithm(&self) -> Option<KeyTransport> {
        KeyTransport::from_uri(&self.algorithm)
    }

    /// Returns `true` when a `KeySize` is present and contradicts the algorithm's own key
    /// length.
    ///
    /// A caller that accepts documents from untrusted senders should treat this as a rejection
    /// condition; see the note on the type.
    #[must_use]
    pub fn key_size_disagrees_with_algorithm(&self) -> bool {
        match (self.key_size, self.block_encryption()) {
            (Some(bits), Some(algorithm)) => bits != algorithm.key_bits(),
            _ => false,
        }
    }

    /// Resolves the RSA-OAEP parameters this method describes, applying the defaults of
    /// RFC 8017 and XMLEnc 1.1 §5.5.2.
    ///
    /// The rules, in order:
    ///
    /// * The message digest is `ds:DigestMethod` if present, otherwise SHA-1.
    /// * For `rsa-oaep-mgf1p` the mask generation function is fixed to MGF1 with SHA-1 and an
    ///   `xenc11:MGF` element is a structural error.
    /// * For `xenc11:rsa-oaep` it is what `xenc11:MGF` selects, defaulting to MGF1 with SHA-1 —
    ///   **not** to the message digest, which is a common misreading.
    /// * The label is the decoded `OAEPparams`, defaulting to empty.
    ///
    /// # Errors
    ///
    /// Returns [`EncError::UnsupportedAlgorithm`](crate::EncError::UnsupportedAlgorithm) when a
    /// digest or mask-generation URI is not recognized, and
    /// [`EncError::InvalidStructure`](crate::EncError::InvalidStructure) when an `xenc11:MGF`
    /// element appears with an algorithm that forbids it.
    pub fn oaep_parameters(&self) -> crate::Result<OaepParams> {
        let digest = match self.digest_method.as_deref() {
            None => Digest::Sha1,
            Some(uri) => Digest::from_uri(uri)
                .ok_or_else(|| crate::EncError::UnsupportedAlgorithm(uri.to_owned()))?,
        };

        let transport = self.key_transport_algorithm();
        let mgf_digest = match self.mgf.as_deref() {
            None => Digest::Sha1,
            Some(uri) => {
                if transport.is_some_and(|algorithm| !algorithm.allows_mgf_element()) {
                    return Err(crate::EncError::InvalidStructure(format!(
                        "an xenc11:MGF element is not permitted with `{}`",
                        self.algorithm
                    )));
                }
                mgf1::mgf1_digest_from_uri(uri)
                    .ok_or_else(|| crate::EncError::UnsupportedAlgorithm(uri.to_owned()))?
            }
        };

        Ok(OaepParams {
            digest,
            mgf_digest,
            label: self.oaep_params.clone().unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_methods_carry_the_matching_key_size() {
        for algorithm in BlockEncryption::ALL {
            let method = EncryptionMethod::block(algorithm);
            assert_eq!(method.key_size, Some(algorithm.key_bits()));
            assert_eq!(method.block_encryption(), Some(algorithm));
            assert!(!method.key_size_disagrees_with_algorithm());
        }
    }

    #[test]
    fn a_contradictory_key_size_is_flagged() {
        let mut method = EncryptionMethod::block(BlockEncryption::Aes256Cbc);
        method.key_size = Some(128);
        assert!(method.key_size_disagrees_with_algorithm());
        // Nothing to contradict when the algorithm is unknown.
        let mut unknown = EncryptionMethod::new("urn:example:cipher");
        unknown.key_size = Some(999);
        assert!(!unknown.key_size_disagrees_with_algorithm());
    }

    #[test]
    fn oaep_defaults_follow_rfc_8017() -> crate::Result<()> {
        let method = EncryptionMethod::key_transport(KeyTransport::RsaOaepMgf1p);
        let params = method.oaep_parameters()?;
        assert_eq!(params.digest, Digest::Sha1);
        assert_eq!(params.mgf_digest, Digest::Sha1);
        assert!(params.label.is_empty());
        Ok(())
    }

    #[test]
    fn the_mgf_digest_does_not_follow_the_message_digest() -> crate::Result<()> {
        let method = EncryptionMethod::key_transport(KeyTransport::RsaOaep)
            .with_digest_method(Digest::Sha512.uri());
        let params = method.oaep_parameters()?;
        assert_eq!(params.digest, Digest::Sha512);
        assert_eq!(params.mgf_digest, Digest::Sha1);
        Ok(())
    }

    #[test]
    fn an_mgf_element_is_rejected_for_mgf1p() {
        let method = EncryptionMethod::key_transport(KeyTransport::RsaOaepMgf1p)
            .with_mgf(mgf1::MGF1_SHA256_URI);
        assert!(matches!(
            method.oaep_parameters(),
            Err(crate::EncError::InvalidStructure(_))
        ));
    }

    #[test]
    fn unknown_parameter_uris_are_reported() {
        let method = EncryptionMethod::key_transport(KeyTransport::RsaOaep)
            .with_digest_method("urn:example:md5");
        assert!(matches!(
            method.oaep_parameters(),
            Err(crate::EncError::UnsupportedAlgorithm(_))
        ));
        let method =
            EncryptionMethod::key_transport(KeyTransport::RsaOaep).with_mgf("urn:example:mgf");
        assert!(matches!(
            method.oaep_parameters(),
            Err(crate::EncError::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn the_algorithm_families_are_reported_separately() {
        let method = EncryptionMethod::key_wrap(KeyWrap::Aes192);
        assert_eq!(method.key_wrap_algorithm(), Some(KeyWrap::Aes192));
        assert_eq!(method.block_encryption(), None);
        assert_eq!(method.key_transport_algorithm(), None);
    }
}
