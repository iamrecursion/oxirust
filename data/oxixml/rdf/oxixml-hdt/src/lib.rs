//! `oxixml-hdt` — HDT (Header-Dictionary-Triples) binary RDF: read, write, and iterate.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate implements the
//! [HDT binary format](http://www.rdfhdt.org/hdt-binary-format/) (a W3C member submission,
//! superseded in its binary details by the linked specification): a compact, self-indexed
//! binary encoding for large, static RDF datasets, built around three components:
//!
//! * The [`control_info`] module — the small `$HDT`-cookie-prefixed preamble that introduces
//!   every section, guarded by a CRC16 (see [`crc`]).
//! * The [`dictionary`] module — [`dictionary::FourSectionDictionary`], mapping every distinct
//!   RDF term string to a compact integer ID and back, using block-compressed
//!   ("plain front-coded") string sections (see [`bitarray`] for the block-pointer packing).
//! * The [`triples`] module — [`triples::BitmapTriples`], the RDF graph itself, encoded purely
//!   in dictionary IDs as a bitmap-indexed adjacency structure (see [`bitmap`] for the
//!   `rank`/`select` bit sequence it is built on).
//!
//! [`Hdt`] ties these together into a single, queryable, fully in-memory dataset; [`HdtWriter`]
//! serializes RDF triples the other way.
//!
//! # Example
//!
//! ```
//! use oxixml_hdt::{Hdt, HdtWriter};
//! use oxixml_model::{NamedNode, Triple};
//!
//! let ex = |s: &str| NamedNode::new(format!("http://example.com/{s}")).expect("valid iri");
//! let triples = vec![
//!     Triple::new(ex("alice"), ex("knows"), ex("bob")),
//!     Triple::new(ex("bob"), ex("knows"), ex("alice")),
//! ];
//!
//! // Write.
//! let mut bytes = Vec::new();
//! HdtWriter::new().write(&mut bytes, triples.clone())?;
//!
//! // Read back and query.
//! let hdt = Hdt::from_bytes(&bytes)?;
//! assert_eq!(hdt.len(), 2);
//!
//! let alice = ex("alice");
//! let matches: Vec<Triple> = hdt
//!     .triples_matching(Some(alice.as_ref().into()), None, None)
//!     .collect::<Result<_, _>>()?;
//! assert_eq!(matches, vec![Triple::new(ex("alice"), ex("knows"), ex("bob"))]);
//! # Result::<(), Box<dyn std::error::Error>>::Ok(())
//! ```
//!
//! # Scope
//!
//! This crate implements the default, single-order (SPO) `BitmapTriples` layout and the
//! default `FourSectionDictionary` with "mapping 1" IDs — the combination every mainstream HDT
//! tool writes and reads. It does not implement the alternative `TriplesList` /
//! `DictionarySectionPlain` fallbacks, or the auxiliary `.hdt.index` companion file other HDT
//! tools persist to disk; instead, [`triples::BitmapTriples`] builds its own PSO and OSP
//! secondary indexes purely in memory, lazily, the first time a query needs one (see
//! [`triples`] for the complexity characteristics and memory cost this implies).

#![deny(missing_docs)]

pub mod bitarray;
pub mod bitmap;
pub mod control_info;
pub mod crc;
pub mod dictionary;
pub mod error;
mod framing;
mod hdt;
pub mod header;
pub mod triples;
pub mod vbyte;
mod writer;

pub use crate::error::{HdtError, Result};
pub use crate::hdt::{Hdt, HdtTriplesIter};
pub use crate::writer::HdtWriter;
