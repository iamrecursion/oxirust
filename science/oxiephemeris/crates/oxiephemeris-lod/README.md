# oxiephemeris-lod

A SPARQL store and HTTP endpoint for the astrological Linked Open Data
layer: an in-memory, pure-Rust triple store with whole-file persistence,
a SPARQL 1.1 Protocol handler, and the `oxieph-sparqld` binary that serves
both.

This crate is the serving layer directly above
[`oxiephemeris-rdf`](../oxiephemeris-rdf): it takes the `oxa:`/`oxc:`/
`oxs:` ontology and chart graphs that crate builds and makes them
queryable and dereferenceable over HTTP. It has three pieces:

- [`FileBackedStore`] — an [`oxigraph`](https://docs.rs/oxigraph) `Store`
  built with `default-features = false`, so it never pulls in RocksDB (a
  C++ dependency, ruled out by the COOLJAPAN pure-Rust policy). Since that
  leaves only the in-memory backend, this crate adds the thinnest
  persistence that still works: `save` serializes the *entire* dataset to
  N-Quads and atomically renames it over the target file (a sibling
  `.tmp` file, then `rename`), so a crash never leaves a half-written
  store on disk — see the `store` module docs for the honest trade-offs
  (no write-ahead log, `O(triples)` per save).
- [`endpoint::handle`] — the SPARQL 1.1 Protocol request handler,
  implemented as a *pure function* from `(&FileBackedStore, read_only,
  &mut Request)` to a `Response`. It never binds a socket, which is
  deliberate: the whole HTTP surface is testable by constructing a
  `Request` in memory and asserting on the `Response`, with no port
  conflicts or shutdown races in the test suite.
- `oxieph-sparqld` (`src/bin/sparqld.rs`) — a thin binary that parses
  arguments, opens a `FileBackedStore`, optionally preloads the
  vocabulary, and hands every request straight to `handle`.

```rust
use oxiephemeris_lod::FileBackedStore;
use oxiephemeris_rdf::{concept_scheme_graph, ontology_graph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `None` keeps everything in memory (`save` becomes a no-op); pass
    // `Some(path)` instead to persist as a whole-file N-Quads dump on
    // every `save` (atomic rename, no partial writes).
    let store = FileBackedStore::open(None)?;
    store.load_graph(&ontology_graph(), None)?;
    store.load_graph(&concept_scheme_graph(), None)?;
    println!("{} triples loaded", store.len()?);

    store.update(
        "INSERT DATA { <https://example.org/s> <https://example.org/p> <https://example.org/o> }",
    )?;
    store.save()?; // no-op here, since `store` was opened with `None`
    Ok(())
}
```

`store.query(sparql)` runs a read-only SPARQL query and returns
`oxigraph`'s own `QueryResults` (solutions, a boolean, or a graph); a
malformed query surfaces as `LodError::Syntax` rather than being
conflated with a genuine evaluation failure. Serving those results over
HTTP with content negotiation is exactly what `endpoint::handle` does —
`oxieph-sparqld`'s own `main` is a small `oxhttp` server wired straight to
it:

```rust
use std::net::SocketAddr;
use std::sync::Arc;

use oxhttp::model::{Body, Request, Response};
use oxhttp::Server;
use oxiephemeris_lod::{handle, FileBackedStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(FileBackedStore::open(None)?);
    let handler_store = Arc::clone(&store);
    let addr: SocketAddr = "127.0.0.1:7878".parse()?;
    let server = Server::new(move |request: &mut Request<Body>| -> Response<Body> {
        handle(handler_store.as_ref(), false, request)
    })
    .bind(addr);
    server.spawn()?.join()?;
    Ok(())
}
```

## The `oxieph-sparqld` binary

```sh
cargo run -p oxiephemeris-lod --bin oxieph-sparqld -- --bind 127.0.0.1:7878 --vocab
```

```
oxieph-sparqld: listening on http://127.0.0.1:7878 (1378 triples, read-write)
```

Flags: `--bind <ADDR>` (default `127.0.0.1:7878`), `--store <PATH>`
(N-Quads file to persist to; omit for in-memory only), `--read-only`
(`POST /update` returns `403` and the store is never mutated), `--load
<PATH>` (repeatable; loads an RDF file at startup, format inferred from
its extension), and `--vocab` (preloads the `oxa:` ontology and `oxc:`/
`oxs:` SKOS concept schemes so the `/ns/oxiephemeris/…` routes below have
something to serve).

With the server above running:

```sh
curl 'http://127.0.0.1:7878/ns/oxiephemeris/astro'   # ontology, as Turtle
curl -G 'http://127.0.0.1:7878/sparql' \
  --data-urlencode 'query=ASK { <https://cooljapan.tech/ns/oxiephemeris/concept/sign/Scorpio> a <http://www.w3.org/2004/02/skos/core#Concept> }' \
  -H 'Accept: application/sparql-results+json'
# {"head":{},"boolean":true}
```

Content negotiation honors `Accept`: SELECT/ASK results default to
SPARQL-Results JSON (also XML, CSV, TSV via oxigraph's own negotiation),
and CONSTRUCT/DESCRIBE graphs default to Turtle (also N-Triples). Serving
`GET /ns/oxiephemeris/astro` and `GET /ns/oxiephemeris/{concept,scheme}`
is what makes the published `oxa:`/`oxc:`/`oxs:` IRIs dereferenceable.

A public instance of the OxiEphemeris vocabulary is live at
<https://sparql.cooljapan.tech/> — SPARQL 1.1 queries at its `/sparql`
path, and the IRIs under <https://cooljapan.tech/ns/oxiephemeris/>
dereference there as Turtle / N-Triples / HTML. That public deployment is
served by OxiRS behind CloudFlare; the `oxieph-sparqld` documented here is
the self-hostable, oxigraph-based alternative — see the security note below
before exposing it to untrusted networks.

## Known issue (security)

This crate's SPARQL/RDF-XML support (via `oxigraph`) transitively depends
on `quick-xml 0.37.5`, which carries 2 HIGH-severity (7.5) RUSTSEC
advisories — [RUSTSEC-2026-0194](https://rustsec.org/advisories/RUSTSEC-2026-0194.html)
and [RUSTSEC-2026-0195](https://rustsec.org/advisories/RUSTSEC-2026-0195.html) —
both denial-of-service via crafted XML input. No upstream fix is available
yet. **Do not expose `oxieph-sparqld` to untrusted networks, and do not
load untrusted RDF/XML via `--load`, until this is resolved upstream.**

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview. Licensed under Apache-2.0 (see workspace
[LICENSE](../../LICENSE)).
