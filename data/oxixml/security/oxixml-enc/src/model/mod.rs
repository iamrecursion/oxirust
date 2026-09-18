//! The XML Encryption structure model: a faithful, round-trippable representation of
//! `xenc:EncryptedData` and `xenc:EncryptedKey`.
//!
//! The model is deliberately *lossless for everything it does not understand*. XML Encryption
//! is an extension point — `EncryptionMethod`, `ds:KeyInfo`, `EncryptionProperty` and
//! `ds:Transform` all admit arbitrary foreign-namespace children — and a decryptor that dropped
//! them would corrupt documents it merely passes through. Anything the crate does not model
//! structurally is kept as verbatim, self-contained XML in an
//! [`XmlFragment`], so parsing and re-serializing reproduces it.
//!
//! # The shape
//!
//! ```text
//! EncryptedData ─┐
//! EncryptedKey  ─┴─ EncryptedType
//!                    ├── Id, Type, MimeType, Encoding   (attributes)
//!                    ├── EncryptionMethod?              (Algorithm, KeySize?, OAEPparams?,
//!                    │                                   ds:DigestMethod?, xenc11:MGF?, …)
//!                    ├── ds:KeyInfo?                    (KeyName | RetrievalMethod |
//!                    │                                   EncryptedKey | AgreementMethod |
//!                    │                                   EcKeyValue | DhKeyValue | …)
//!                    ├── CipherData                     (CipherValue | CipherReference)
//!                    └── EncryptionProperties?
//!
//! EncryptedKey adds: Recipient (attribute), ReferenceList?, CarriedKeyName?
//! ```
//!
//! # Example
//!
//! ```
//! use oxixml_enc::algorithm::{BlockEncryption, KeyWrap};
//! use oxixml_enc::model::{
//!     ContentType, EncryptedData, EncryptedKey, EncryptionMethod, KeyInfo, Reference,
//!     ReferenceList,
//! };
//!
//! // An EncryptedData whose key travels with it, wrapped under a named key-encryption key.
//! let mut wrapped_key = EncryptedKey::new(vec![0xaa; 40]);
//! wrapped_key.common.encryption_method = Some(EncryptionMethod::key_wrap(KeyWrap::Aes256));
//! wrapped_key.common.key_info = Some(KeyInfo::with_key_name("kek-2026"));
//! wrapped_key.reference_list = Some(ReferenceList {
//!     references: vec![Reference::data("#ed1")],
//! });
//!
//! let mut data = EncryptedData::new(vec![0xbb; 48]);
//! data.common.id = Some("ed1".to_owned());
//! data.common.content_type = Some(ContentType::Element);
//! data.common.encryption_method = Some(EncryptionMethod::block(BlockEncryption::Aes256Gcm));
//! data.common.key_info = Some(KeyInfo::with_encrypted_key(wrapped_key));
//!
//! // Serializing and re-parsing is lossless.
//! let xml = data.to_xml()?;
//! assert_eq!(EncryptedData::from_xml(&xml)?, data);
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

mod agreement;
mod cipher;
mod encrypted;
mod keyinfo;
mod method;
mod parse;
mod properties;
mod serialize;

pub use agreement::{AgreementMethod, DhKeyValue, EcKeyValue, KeyDerivationMethod};
pub use cipher::{CipherData, CipherReference, Transform, XmlFragment};
pub use encrypted::{ContentType, EncryptedData, EncryptedKey, EncryptedType};
pub use keyinfo::{KeyInfo, KeyInfoItem, Reference, ReferenceKind, ReferenceList};
pub use method::EncryptionMethod;
pub use properties::{EncryptionProperties, EncryptionProperty};
pub use serialize::{DS_PREFIX, DSIG11_PREFIX, XENC_PREFIX, XENC11_PREFIX};
