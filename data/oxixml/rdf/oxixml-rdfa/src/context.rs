//! The RDFa 1.1 [initial contexts](https://www.w3.org/TR/rdfa-core/#s_initialcontexts).
//!
//! An initial context seeds the evaluation context with prefix mappings, term
//! mappings and, potentially, a default vocabulary before any element of the
//! document is looked at. RDFa Core 1.1 defines one at
//! `http://www.w3.org/2011/rdfa-context/rdfa-1.1`, and each host language may
//! layer further ones on top of it — XHTML+RDFa 1.1 adds
//! `http://www.w3.org/2011/rdfa-context/xhtml-rdfa-1.1`, and HTML5+RDFa 1.1
//! adds `http://www.w3.org/2011/rdfa-context/html-rdfa-1.1`, which the W3C
//! publishes as an empty context.
//!
//! The tables below are transcribed verbatim from those three published
//! documents. They are sorted so that a binary search can find an entry, and
//! the lookups in `crate::curie` (a private module) rely on that ordering.

/// The IRI of the RDFa Core 1.1 initial context.
pub const RDFA_CORE_CONTEXT: &str = "http://www.w3.org/2011/rdfa-context/rdfa-1.1";

/// The IRI of the XHTML+RDFa 1.1 initial context.
pub const XHTML_CONTEXT: &str = "http://www.w3.org/2011/rdfa-context/xhtml-rdfa-1.1";

/// The IRI of the HTML5+RDFa 1.1 initial context.
pub const HTML_CONTEXT: &str = "http://www.w3.org/2011/rdfa-context/html-rdfa-1.1";

/// The `http://www.w3.org/1999/xhtml/vocab#` vocabulary, which is the term
/// vocabulary of the `@role` attribute and the target of most XHTML terms.
pub const XHTML_VOCABULARY: &str = "http://www.w3.org/1999/xhtml/vocab#";

/// The prefix mappings of the RDFa Core 1.1 initial context.
pub static RDFA_CORE_PREFIXES: [(&str, &str); 46] = [
    ("as", "https://www.w3.org/ns/activitystreams#"),
    ("cc", "http://creativecommons.org/ns#"),
    ("csvw", "http://www.w3.org/ns/csvw#"),
    ("ctag", "http://commontag.org/ns#"),
    ("dc", "http://purl.org/dc/terms/"),
    ("dc11", "http://purl.org/dc/elements/1.1/"),
    ("dcat", "http://www.w3.org/ns/dcat#"),
    ("dcterms", "http://purl.org/dc/terms/"),
    ("dqv", "http://www.w3.org/ns/dqv#"),
    ("duv", "https://www.w3.org/ns/duv#"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("gr", "http://purl.org/goodrelations/v1#"),
    ("grddl", "http://www.w3.org/2003/g/data-view#"),
    ("ical", "http://www.w3.org/2002/12/cal/icaltzd#"),
    ("jsonld", "http://www.w3.org/ns/json-ld#"),
    ("ldp", "http://www.w3.org/ns/ldp#"),
    ("ma", "http://www.w3.org/ns/ma-ont#"),
    ("oa", "http://www.w3.org/ns/oa#"),
    ("odrl", "http://www.w3.org/ns/odrl/2/"),
    ("og", "http://ogp.me/ns#"),
    ("org", "http://www.w3.org/ns/org#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("prov", "http://www.w3.org/ns/prov#"),
    ("qb", "http://purl.org/linked-data/cube#"),
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfa", "http://www.w3.org/ns/rdfa#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("rev", "http://purl.org/stuff/rev#"),
    ("rif", "http://www.w3.org/2007/rif#"),
    ("rr", "http://www.w3.org/ns/r2rml#"),
    ("schema", "http://schema.org/"),
    ("sd", "http://www.w3.org/ns/sparql-service-description#"),
    ("sioc", "http://rdfs.org/sioc/ns#"),
    ("skos", "http://www.w3.org/2004/02/skos/core#"),
    ("skosxl", "http://www.w3.org/2008/05/skos-xl#"),
    ("sosa", "http://www.w3.org/ns/sosa/"),
    ("ssn", "http://www.w3.org/ns/ssn/"),
    ("time", "http://www.w3.org/2006/time#"),
    ("v", "http://rdf.data-vocabulary.org/#"),
    ("vcard", "http://www.w3.org/2006/vcard/ns#"),
    ("void", "http://rdfs.org/ns/void#"),
    ("wdr", "http://www.w3.org/2007/05/powder#"),
    ("wdrs", "http://www.w3.org/2007/05/powder-s#"),
    ("xhv", "http://www.w3.org/1999/xhtml/vocab#"),
    ("xml", "http://www.w3.org/XML/1998/namespace"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
];

/// The term mappings of the RDFa Core 1.1 initial context.
pub static RDFA_CORE_TERMS: [(&str, &str); 3] = [
    (
        "describedby",
        "http://www.w3.org/2007/05/powder-s#describedby",
    ),
    ("license", "http://www.w3.org/1999/xhtml/vocab#license"),
    ("role", "http://www.w3.org/1999/xhtml/vocab#role"),
];

/// The term mappings of the XHTML+RDFa 1.1 initial context.
pub static XHTML_TERMS: [(&str, &str); 25] = [
    ("alternate", "http://www.w3.org/1999/xhtml/vocab#alternate"),
    ("appendix", "http://www.w3.org/1999/xhtml/vocab#appendix"),
    ("bookmark", "http://www.w3.org/1999/xhtml/vocab#bookmark"),
    ("chapter", "http://www.w3.org/1999/xhtml/vocab#chapter"),
    ("cite", "http://www.w3.org/1999/xhtml/vocab#cite"),
    ("contents", "http://www.w3.org/1999/xhtml/vocab#contents"),
    ("copyright", "http://www.w3.org/1999/xhtml/vocab#copyright"),
    ("first", "http://www.w3.org/1999/xhtml/vocab#first"),
    ("glossary", "http://www.w3.org/1999/xhtml/vocab#glossary"),
    ("help", "http://www.w3.org/1999/xhtml/vocab#help"),
    ("icon", "http://www.w3.org/1999/xhtml/vocab#icon"),
    ("index", "http://www.w3.org/1999/xhtml/vocab#index"),
    ("last", "http://www.w3.org/1999/xhtml/vocab#last"),
    ("license", "http://www.w3.org/1999/xhtml/vocab#license"),
    ("meta", "http://www.w3.org/1999/xhtml/vocab#meta"),
    ("next", "http://www.w3.org/1999/xhtml/vocab#next"),
    ("p3pv1", "http://www.w3.org/1999/xhtml/vocab#p3pv1"),
    ("prev", "http://www.w3.org/1999/xhtml/vocab#prev"),
    ("previous", "http://www.w3.org/1999/xhtml/vocab#previous"),
    ("section", "http://www.w3.org/1999/xhtml/vocab#section"),
    ("start", "http://www.w3.org/1999/xhtml/vocab#start"),
    (
        "stylesheet",
        "http://www.w3.org/1999/xhtml/vocab#stylesheet",
    ),
    (
        "subsection",
        "http://www.w3.org/1999/xhtml/vocab#subsection",
    ),
    ("top", "http://www.w3.org/1999/xhtml/vocab#top"),
    ("up", "http://www.w3.org/1999/xhtml/vocab#up"),
];

/// The prefix mappings of the XHTML+RDFa 1.1 initial context.
pub static XHTML_PREFIXES: [(&str, &str); 0] = [];

/// The term mappings of the HTML5+RDFa 1.1 initial context, which is empty.
pub static HTML_TERMS: [(&str, &str); 0] = [];

/// The prefix mappings of the HTML5+RDFa 1.1 initial context, which is empty.
pub static HTML_PREFIXES: [(&str, &str); 0] = [];
