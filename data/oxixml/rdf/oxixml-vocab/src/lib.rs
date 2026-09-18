//! `oxixml-vocab` — typed RDF vocabulary constants for the COOLJAPAN ecosystem.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! Every well-known RDF/OWL/SHACL/etc. term used across the ecosystem is exposed here as a
//! `'static` [`NamedNodeRef`](oxixml_model::NamedNodeRef) constant, built with
//! [`NamedNodeRef::new_unchecked`](oxixml_model::NamedNodeRef::new_unchecked) so that using a
//! term allocates nothing.
//!
//! # Vocabularies
//!
//! | Module | Vocabulary | Namespace |
//! |---|---|---|
//! | [`rdf`] | [RDF 1.1/1.2 Concepts](https://www.w3.org/TR/rdf12-concepts/) | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` |
//! | [`rdfs`] | [RDF Schema](https://www.w3.org/TR/rdf-schema/) | `http://www.w3.org/2000/01/rdf-schema#` |
//! | [`xsd`] | [XML Schema Datatypes](https://www.w3.org/TR/xmlschema11-2/) | `http://www.w3.org/2001/XMLSchema#` |
//! | [`owl`] | [OWL 2](https://www.w3.org/TR/owl2-overview/) | `http://www.w3.org/2002/07/owl#` |
//! | [`skos`] | [SKOS](https://www.w3.org/TR/skos-reference/) | `http://www.w3.org/2004/02/skos/core#` |
//! | [`dcterms`] | [DCMI Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) | `http://purl.org/dc/terms/` |
//! | [`dcmitype`] | [DCMI Type Vocabulary](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/#section-7) | `http://purl.org/dc/dcmitype/` |
//! | [`dcat`] | [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/) | `http://www.w3.org/ns/dcat#` |
//! | [`foaf`] | [FOAF](http://xmlns.com/foaf/spec/) | `http://xmlns.com/foaf/0.1/` |
//! | [`prov`] | [PROV-O](https://www.w3.org/TR/prov-o/) | `http://www.w3.org/ns/prov#` |
//! | [`sh`] | [SHACL](https://www.w3.org/TR/shacl/) | `http://www.w3.org/ns/shacl#` |
//! | [`geo`] | [OGC GeoSPARQL](http://www.opengis.net/doc/IS/geosparql/1.0) | `http://www.opengis.net/ont/geosparql#` |
//! | [`wgs84`] | [W3C Basic Geo (WGS84)](https://www.w3.org/2003/01/geo/) | `http://www.w3.org/2003/01/geo/wgs84_pos#` |
//! | [`void`] | [VoID](https://www.w3.org/TR/void/) | `http://rdfs.org/ns/void#` |
//! | [`schema`] | [schema.org](https://schema.org/) (curated subset) | `http://schema.org/` |
//! | [`time`] | [OWL-Time](https://www.w3.org/TR/owl-time/) | `http://www.w3.org/2006/time#` |
//!
//! Each module exposes a `NAMESPACE` string constant, one `NamedNodeRef<'static>` constant per
//! term, and an `ALL` slice listing every term it defines (useful for introspection, and used by
//! this crate's own tests to check for internal consistency).
//!
//! # Example
//!
//! ```
//! use oxixml_model::Literal;
//! use oxixml_vocab::{rdf, xsd};
//!
//! // Build an `xsd:integer`-typed literal using a vocabulary constant.
//! let one = Literal::new_typed_literal("1", xsd::INTEGER);
//! assert_eq!(one.datatype().as_str(), xsd::INTEGER.as_str());
//!
//! // Namespaces are always absolute IRIs ending in `#` or `/`.
//! assert!(rdf::NAMESPACE.ends_with('#'));
//! assert_eq!(rdf::TYPE.as_str(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod dcat;
pub mod dcmitype;
pub mod dcterms;
pub mod foaf;
pub mod geo;
pub mod owl;
pub mod prov;
pub mod rdf;
pub mod rdfs;
pub mod schema;
pub mod sh;
pub mod skos;
pub mod time;
pub mod void;
pub mod wgs84;
pub mod xsd;
