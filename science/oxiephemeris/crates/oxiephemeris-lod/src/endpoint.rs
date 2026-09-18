//! The SPARQL 1.1 Protocol request handler.
//!
//! # A pure function, on purpose
//!
//! The core of this module is [`handle`], a *pure* function from
//! `(&FileBackedStore, read_only, &mut Request)` to a `Response`. It never
//! touches a socket. That is a deliberate testability decision: binding a
//! TCP port in unit tests invites flakiness (port already in use, race on
//! shutdown, firewall prompts), so instead the tests construct a
//! [`Request`] in memory and assert on the returned [`Response`]. The
//! `oxieph-sparqld` binary is the only place that binds a socket, and all it
//! does is hand each request straight to [`handle`].
//!
//! # Routes
//!
//! | Method & path | Behaviour |
//! |---|---|
//! | `GET /` | Plain-text service description. |
//! | `GET /sparql?query=…` | Run a SPARQL query. |
//! | `POST /sparql` | Run a SPARQL query (raw body or `query=` form). |
//! | `POST /update` | Run a SPARQL update (`403` if read-only). |
//! | `GET /ns/oxiephemeris/astro` | Ontology as Turtle. |
//! | `GET /ns/oxiephemeris/{concept,scheme}` | SKOS vocabulary as Turtle. |
//! | anything else | `404`. |
//!
//! Content negotiation honours the `Accept` header: SELECT/ASK results
//! default to SPARQL-Results JSON (also XML, CSV, TSV), CONSTRUCT/DESCRIBE
//! graphs default to Turtle (also N-Triples). Serving the `/ns/…` graphs is
//! what makes the published `oxa:`/`oxc:`/`oxs:` IRIs dereferenceable when
//! this endpoint is hosted at `cooljapan.tech`.

use std::io::Read;

use oxhttp::model::{header, Body, Request, Response, StatusCode};
use oxiephemeris_rdf::{concept_scheme_graph, ontology_graph, to_turtle_string};
use oxigraph::io::{RdfFormat, RdfSerializer};
use oxigraph::sparql::results::{QueryResultsFormat, QueryResultsSerializer};
use oxigraph::sparql::{QueryResults, QuerySolutionIter, QueryTripleIter};
// The `/ns/…` routes serve graphs built by `oxiephemeris_rdf`, so this is
// the document model, not `oxigraph::model`.
use oxiephemeris_rdf::oxrdf::Graph;

use crate::error::LodError;
use crate::store::FileBackedStore;

/// Handles one HTTP request against `store`.
///
/// This is the whole protocol surface as a pure function — see the module
/// docs for why. If `read_only` is set, `POST /update` is refused with a
/// `403` and the store is never mutated.
///
/// It never panics and never returns `Err`: every failure is turned into an
/// HTTP status (`400` for malformed SPARQL, `403` for a rejected update,
/// `404` for an unknown route, `500` for an internal error).
#[must_use]
pub fn handle(
    store: &FileBackedStore,
    read_only: bool,
    request: &mut Request<Body>,
) -> Response<Body> {
    // Snapshot everything we need as owned values *before* borrowing the
    // body mutably, so the request-line/header reads do not conflict with
    // the body read in the POST arms.
    let method = request.method().as_str().to_owned();
    let path = request.uri().path().to_owned();
    let query_string = request.uri().query().map(str::to_owned);
    let accept = header_value(request, &header::ACCEPT);
    let content_type = header_value(request, &header::CONTENT_TYPE);

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => service_description(),
        ("GET", "/sparql") => {
            let Some(sparql) = query_string
                .as_deref()
                .and_then(|qs| form_param(qs, "query"))
            else {
                return respond(StatusCode::BAD_REQUEST, TEXT, "missing 'query' parameter");
            };
            run_query(store, &sparql, accept.as_deref())
        }
        ("POST", "/sparql") => {
            let body = match read_body(request) {
                Ok(body) => body,
                Err(e) => return internal_error(&e),
            };
            let sparql = if is_form(content_type.as_deref()) {
                match form_param(&body, "query") {
                    Some(q) => q,
                    None => {
                        return respond(
                            StatusCode::BAD_REQUEST,
                            TEXT,
                            "missing 'query' form parameter",
                        )
                    }
                }
            } else {
                body
            };
            run_query(store, &sparql, accept.as_deref())
        }
        ("POST", "/update") => {
            if read_only {
                return respond(StatusCode::FORBIDDEN, TEXT, "endpoint is read-only");
            }
            let body = match read_body(request) {
                Ok(body) => body,
                Err(e) => return internal_error(&e),
            };
            let update = if is_form(content_type.as_deref()) {
                match form_param(&body, "update") {
                    Some(u) => u,
                    None => {
                        return respond(
                            StatusCode::BAD_REQUEST,
                            TEXT,
                            "missing 'update' form parameter",
                        )
                    }
                }
            } else {
                body
            };
            run_update(store, &update)
        }
        ("GET", "/ns/oxiephemeris/astro") => serve_turtle(&ontology_graph()),
        ("GET", "/ns/oxiephemeris/concept" | "/ns/oxiephemeris/scheme") => {
            serve_turtle(&concept_scheme_graph())
        }
        _ => respond(StatusCode::NOT_FOUND, TEXT, "not found"),
    }
}

/// The `text/plain` content type used for status and error bodies.
const TEXT: &str = "text/plain; charset=utf-8";

/// Reads a header as an owned `String`, dropping non-ASCII values.
fn header_value(request: &Request<Body>, name: &header::HeaderName) -> Option<String> {
    request
        .headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// Reads the full request body into a `String`.
fn read_body(request: &mut Request<Body>) -> Result<String, LodError> {
    let mut buf = String::new();
    request.body_mut().read_to_string(&mut buf)?;
    Ok(buf)
}

/// Whether a `Content-Type` denotes an HTML form submission.
fn is_form(content_type: Option<&str>) -> bool {
    content_type.is_some_and(|ct| ct.contains("application/x-www-form-urlencoded"))
}

/// Runs a query and serializes the results per the `Accept` header.
fn run_query(store: &FileBackedStore, sparql: &str, accept: Option<&str>) -> Response<Body> {
    match store.query(sparql) {
        Ok(QueryResults::Solutions(iter)) => {
            let format = results_format(accept);
            match write_solutions(iter, format) {
                Ok(body) => respond(StatusCode::OK, format.media_type(), body),
                Err(e) => internal_error(&e),
            }
        }
        Ok(QueryResults::Boolean(value)) => {
            let format = results_format(accept);
            match QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_writer(Vec::new(), value)
            {
                Ok(body) => respond(StatusCode::OK, format.media_type(), body),
                Err(e) => internal_error(&LodError::Io(e)),
            }
        }
        Ok(QueryResults::Graph(iter)) => {
            let format = rdf_format(accept);
            match write_graph(iter, format) {
                Ok(body) => respond(StatusCode::OK, format.media_type(), body),
                Err(e) => internal_error(&e),
            }
        }
        Err(LodError::Syntax(e)) => respond(
            StatusCode::BAD_REQUEST,
            TEXT,
            format!("malformed query: {e}"),
        ),
        Err(e) => internal_error(&e),
    }
}

/// Runs an update, then persists the store (a no-op when in-memory).
fn run_update(store: &FileBackedStore, sparql: &str) -> Response<Body> {
    match store.update(sparql) {
        Ok(()) => match store.save() {
            Ok(()) => respond(StatusCode::OK, TEXT, "update applied"),
            Err(e) => internal_error(&e),
        },
        Err(LodError::Syntax(e)) => respond(
            StatusCode::BAD_REQUEST,
            TEXT,
            format!("malformed update: {e}"),
        ),
        Err(e) => internal_error(&e),
    }
}

/// Serializes SELECT solutions to bytes in the chosen results format.
fn write_solutions(
    iter: QuerySolutionIter,
    format: QueryResultsFormat,
) -> Result<Vec<u8>, LodError> {
    let variables = iter.variables().to_vec();
    let mut serializer = QueryResultsSerializer::from_format(format)
        .serialize_solutions_to_writer(Vec::new(), variables)?;
    for solution in iter {
        serializer.serialize(&solution?)?;
    }
    Ok(serializer.finish()?)
}

/// Serializes CONSTRUCT/DESCRIBE triples to bytes in the chosen RDF format.
fn write_graph(iter: QueryTripleIter, format: RdfFormat) -> Result<Vec<u8>, LodError> {
    let mut serializer = RdfSerializer::from_format(format).for_writer(Vec::new());
    for triple in iter {
        serializer.serialize_triple(triple?.as_ref())?;
    }
    Ok(serializer.finish()?)
}

/// Serves an in-memory graph as Turtle (the `/ns/…` dereference routes).
fn serve_turtle(graph: &Graph) -> Response<Body> {
    match to_turtle_string(graph) {
        Ok(text) => respond(StatusCode::OK, "text/turtle; charset=utf-8", text),
        Err(e) => internal_error(&LodError::Rdf(e)),
    }
}

/// The plain-text service description at `GET /`.
fn service_description() -> Response<Body> {
    let text = "OxiEphemeris SPARQL endpoint\n\n\
        GET  /sparql?query=...                 SPARQL 1.1 query\n\
        POST /sparql                           SPARQL 1.1 query (raw body or query= form)\n\
        POST /update                           SPARQL 1.1 update\n\
        GET  /ns/oxiephemeris/astro            ontology (Turtle)\n\
        GET  /ns/oxiephemeris/concept          SKOS concepts (Turtle)\n\
        GET  /ns/oxiephemeris/scheme           SKOS concept schemes (Turtle)\n";
    respond(StatusCode::OK, TEXT, text)
}

/// Picks a SPARQL-results format from `Accept`, defaulting to JSON.
fn results_format(accept: Option<&str>) -> QueryResultsFormat {
    accept
        .into_iter()
        .flat_map(|a| a.split(','))
        .find_map(|m| QueryResultsFormat::from_media_type(m.trim()))
        .unwrap_or(QueryResultsFormat::Json)
}

/// Picks an RDF graph format from `Accept`, restricted to Turtle or
/// N-Triples (the only two the spec asks CONSTRUCT/DESCRIBE to emit), and
/// defaulting to Turtle.
fn rdf_format(accept: Option<&str>) -> RdfFormat {
    accept
        .into_iter()
        .flat_map(|a| a.split(','))
        .find_map(|m| {
            RdfFormat::from_media_type(m.trim())
                .filter(|f| matches!(f, RdfFormat::Turtle | RdfFormat::NTriples))
        })
        .unwrap_or(RdfFormat::Turtle)
}

/// Extracts a `key=value` parameter from an `x-www-form-urlencoded` string.
fn form_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                return Some(percent_decode(v));
            }
        }
    }
    None
}

/// Decodes `application/x-www-form-urlencoded` text: `+` to space and
/// `%XX` escapes. Invalid escapes are passed through literally, and invalid
/// UTF-8 is replaced rather than rejected.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                if let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                    out.push((hi << 4) | lo);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parses a single ASCII hex digit.
fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Builds a `500` response from an internal error.
fn internal_error(error: &LodError) -> Response<Body> {
    respond(StatusCode::INTERNAL_SERVER_ERROR, TEXT, format!("{error}"))
}

/// Builds a response, falling back to a bare `500` if the builder errors.
///
/// [`Response::builder`] can only fail on an invalid status or header, none
/// of which this crate ever supplies, but the constraint is that we must
/// not `unwrap`; the fallback path keeps that guarantee without panicking.
fn respond(status: StatusCode, content_type: &str, body: impl Into<Body>) -> Response<Body> {
    if let Ok(response) = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .body(body.into())
    {
        response
    } else {
        let mut response = Response::new(Body::from("error building response"));
        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_vocab() -> FileBackedStore {
        let Ok(store) = FileBackedStore::open(None) else {
            panic!("open must succeed");
        };
        let Ok(()) = store.load_graph(&ontology_graph(), None) else {
            panic!("load ontology");
        };
        let Ok(()) = store.load_graph(&concept_scheme_graph(), None) else {
            panic!("load concept scheme");
        };
        store
    }

    fn get(uri: &str, accept: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder().method("GET").uri(uri);
        if let Some(accept) = accept {
            builder = builder.header(header::ACCEPT, accept);
        }
        match builder.body(Body::empty()) {
            Ok(request) => request,
            Err(e) => panic!("build request: {e}"),
        }
    }

    fn post(uri: &str, content_type: &str, body: &'static str) -> Request<Body> {
        match Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
        {
            Ok(request) => request,
            Err(e) => panic!("build request: {e}"),
        }
    }

    fn body_string(response: Response<Body>) -> String {
        match response.into_body().to_string() {
            Ok(text) => text,
            Err(e) => panic!("read body: {e}"),
        }
    }

    #[test]
    fn select_returns_json() {
        let store = store_with_vocab();
        let mut request = get(
            "/sparql?query=SELECT%20%3Fs%20WHERE%20%7B%20%3Fs%20a%20%3Fo%20%7D%20LIMIT%201",
            None,
        );
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response.headers().get(header::CONTENT_TYPE);
        let Some(ct) = ct.and_then(|v| v.to_str().ok()) else {
            panic!("content type missing");
        };
        assert!(ct.contains("application/sparql-results+json"), "got {ct}");
        let text = body_string(response);
        assert!(text.contains("\"bindings\""), "json body: {text}");
    }

    #[test]
    fn select_negotiates_csv() {
        let store = store_with_vocab();
        let mut request = get(
            "/sparql?query=SELECT%20%3Fs%20WHERE%20%7B%20%3Fs%20a%20%3Fo%20%7D%20LIMIT%201",
            Some("text/csv"),
        );
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let Some(ct) = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
        else {
            panic!("content type missing");
        };
        assert!(ct.contains("text/csv"), "got {ct}");
    }

    #[test]
    fn ask_over_vocabulary_is_true() {
        let store = store_with_vocab();
        // Scorpio must be a SKOS concept in the loaded vocabulary.
        let query = "ASK%20%7B%20%3Chttps%3A%2F%2Fcooljapan.tech%2Fns%2Foxiephemeris%2Fconcept%2Fsign%2FScorpio%3E%20a%20%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23Concept%3E%20%7D";
        let mut request = get(
            &format!("/sparql?query={query}"),
            Some("application/sparql-results+json"),
        );
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_string(response);
        assert!(text.contains("true"), "ASK must be true, got {text}");
    }

    #[test]
    fn construct_returns_turtle() {
        let store = store_with_vocab();
        let mut request = post(
            "/sparql",
            "application/sparql-query",
            "CONSTRUCT { ?s a ?o } WHERE { ?s a ?o } LIMIT 1",
        );
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let Some(ct) = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
        else {
            panic!("content type missing");
        };
        assert!(ct.contains("text/turtle"), "got {ct}");
    }

    #[test]
    fn ns_astro_serves_ontology_turtle() {
        let store = store_with_vocab();
        let mut request = get("/ns/oxiephemeris/astro", None);
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_string(response);
        assert!(text.contains("@prefix oxa:"), "turtle body: {text}");
    }

    #[test]
    fn ns_concept_serves_vocabulary_turtle() {
        let store = store_with_vocab();
        let mut request = get("/ns/oxiephemeris/concept", None);
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_string(response);
        assert!(
            text.contains("skos:Concept"),
            "turtle body missing concepts"
        );
    }

    #[test]
    fn update_rejected_when_read_only() {
        let store = store_with_vocab();
        let Ok(before) = store.len() else {
            panic!("len");
        };
        let mut request = post(
            "/update",
            "application/sparql-update",
            "INSERT DATA { <http://example.com/s> <http://example.com/p> <http://example.com/o> }",
        );
        let response = handle(&store, true, &mut request);
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let Ok(after) = store.len() else {
            panic!("len");
        };
        assert_eq!(before, after, "read-only update must not mutate the store");
    }

    #[test]
    fn update_applied_when_writable() {
        let store = store_with_vocab();
        let Ok(before) = store.len() else {
            panic!("len");
        };
        let mut request = post(
            "/update",
            "application/sparql-update",
            "INSERT DATA { <http://example.com/s> <http://example.com/p> <http://example.com/o> }",
        );
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let Ok(after) = store.len() else {
            panic!("len");
        };
        assert_eq!(after, before + 1, "update must add one triple");
    }

    #[test]
    fn malformed_query_is_bad_request() {
        let store = store_with_vocab();
        let mut request = get("/sparql?query=SELECT%20nonsense%20%7B", None);
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn unknown_path_is_not_found() {
        let store = store_with_vocab();
        let mut request = get("/nope", None);
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn root_is_service_description() {
        let store = store_with_vocab();
        let mut request = get("/", None);
        let response = handle(&store, false, &mut request);
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_string(response);
        assert!(
            text.contains("/sparql"),
            "service description names /sparql"
        );
    }
}
