//! `oxieph-sparqld`: SPARQL 1.1 endpoint over the `OxiEphemeris` LOD store.
//!
//! A thin `main` around [`handle`]: it parses arguments, opens a
//! [`FileBackedStore`],
//! optionally preloads the astrology vocabulary and any `--load` files, then
//! serves every request by handing it straight to the pure request handler.
//! All the interesting logic (and all the tests) live in the library; this
//! binary only binds the socket.
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::BufReader;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use oxhttp::model::{Body, Request, Response};
use oxhttp::Server;
use oxiephemeris_lod::error::LodError;
use oxiephemeris_lod::{handle, FileBackedStore};
use oxiephemeris_rdf::{concept_scheme_graph, ontology_graph};
use oxigraph::io::RdfFormat;

/// Command-line arguments for `oxieph-sparqld`.
#[derive(Debug, Parser)]
#[command(
    name = "oxieph-sparqld",
    version,
    about = "SPARQL 1.1 endpoint over the OxiEphemeris Linked Open Data store"
)]
struct Args {
    /// Socket address to bind, e.g. `127.0.0.1:7878`.
    #[arg(long, default_value = "127.0.0.1:7878")]
    bind: String,

    /// N-Quads file to persist the store to (omit for in-memory only).
    #[arg(long)]
    store: Option<PathBuf>,

    /// Refuse SPARQL updates (`POST /update` returns 403).
    #[arg(long)]
    read_only: bool,

    /// RDF file(s) to load at startup (format inferred from extension).
    #[arg(long)]
    load: Vec<PathBuf>,

    /// Preload the astrology ontology and SKOS concept schemes.
    #[arg(long)]
    vocab: bool,
}

/// Loads one RDF file into the store, inferring the format from its
/// extension (defaulting to Turtle).
fn load_file(store: &FileBackedStore, path: &Path) -> Result<(), LodError> {
    let format = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(RdfFormat::from_extension)
        .unwrap_or(RdfFormat::Turtle);
    let reader = BufReader::new(File::open(path)?);
    store.load_from_reader(format, reader)
}

/// Builds and runs the server.
fn run() -> Result<(), LodError> {
    let args = Args::parse();

    let addr: SocketAddr = args
        .bind
        .parse()
        .map_err(|e| LodError::Config(format!("invalid --bind address {:?}: {e}", args.bind)))?;

    let store = FileBackedStore::open(args.store.as_deref())?;

    if args.vocab {
        store.load_graph(&ontology_graph(), None)?;
        store.load_graph(&concept_scheme_graph(), None)?;
    }
    for path in &args.load {
        load_file(&store, path)?;
    }
    // Persist whatever we preloaded so an on-disk store reflects it.
    if args.store.is_some() {
        store.save()?;
    }

    let count = store.len()?;
    let read_only = args.read_only;
    eprintln!(
        "oxieph-sparqld: listening on http://{addr} ({count} triples, {})",
        if read_only { "read-only" } else { "read-write" }
    );

    let store = Arc::new(store);
    let handler_store = Arc::clone(&store);
    let server = Server::new(move |request: &mut Request<Body>| -> Response<Body> {
        handle(handler_store.as_ref(), read_only, request)
    })
    .bind(addr);

    server.spawn()?.join()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
