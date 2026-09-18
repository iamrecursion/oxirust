# OxiEphemeris

A pure-Rust, clean-room astronomical ephemeris engine: JPL DE/SPK binary
kernel reading and Chebyshev interpolation, IAU 2006/2000A precession-nutation
and apparent-place computation, an astrology layer (houses, aspects, lunar
nodes, sidereal zodiac/ayanamshas, essential dignities, declinations, Arabic
Parts, midpoints, synastry, fixed stars), an analytic (data-file-free)
fallback, and a Swiss-Ephemeris-*shaped* compatibility API — plus a CLI
(`oxieph`), a Linked Open Data layer (a custom RDF/SKOS/PROV-O astrology
vocabulary with a SPARQL store and HTTP endpoint), and Python and
WebAssembly bindings sharing one chart-computation facade with the CLI.

This is a **0.1.1** release: 490 tests green across the workspace
(`cargo nextest run --workspace --all-features`; see the per-crate counts
in the table below). The API surface is functional and verified against
independent oracles (see below), but is not yet declared stable; expect
possible breaking changes before 1.0.

## Clean-room provenance

OxiEphemeris is implemented from published papers and standards only — the
Swiss Ephemeris and SOFA source trees have never been read or consulted by
anyone working on this project. The `oxiephemeris-compat` crate matches Swiss
Ephemeris' API *shape* (function names, flags, house-system identifiers)
using only its public programmer's documentation, and its numerical behavior
is verified against committed Swiss Ephemeris *output* fixtures — never its
source. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full clean-room
provenance statement and the list of source standards every algorithm is
implemented from.

## Crates

| Crate | Purpose | Tests |
|---|---|---:|
| [`oxiephemeris-core`](crates/oxiephemeris-core) | Two-part Julian Date/calendar arithmetic (proleptic Gregorian *and* Julian, BCE-sign-correct), TAI/TT/TDB time scales, IERS leap-second table. | 36 |
| [`oxiephemeris-de`](crates/oxiephemeris-de) | JPL DE/SPK binary ephemeris kernel reader — classic DE440/DE441 binary format *and* the SPK/DAF `.bsp` format (types 2/3/13) — plus Chebyshev interpolation (Clenshaw algorithm), position and velocity. | 29 |
| [`oxiephemeris-bodies`](crates/oxiephemeris-bodies) | IAU 2006 precession + frame bias, full IAU 2000A nutation (not truncated) plus a 2000B-class truncation option, IAU 2006 obliquity, the apparent-place pipeline (light-time iteration, relativistic annual aberration, optional gravitational light deflection), and fixed-star apparent positions (Kaplan 1989 method). | 61 |
| [`oxiephemeris-astro`](crates/oxiephemeris-astro) | The astrology layer: chart angles (Ascendant/MC/Vertex/East Point), 7 house systems (Placidus, Koch, Whole Sign, and others), aspects with configurable orbs, lunar nodes/apogee (mean + true/osculating), sidereal zodiac / ayanamshas (5 today, e.g. Lahiri), essential dignities, declinations (incl. antiscia), Arabic Parts, midpoints, synastry cross-aspects, motion classification, and a 177-star Hipparcos (V≤3) catalog. | 108 |
| [`oxiephemeris-analytic`](crates/oxiephemeris-analytic) | A data-file-free analytic fallback: truncated VSOP87E (planets, Bretagnon & Francou 1988) + ELP2000-82B (Moon, Chapront-Touzé & Chapront 1983/1988), fit to JPL DE200. No Pluto. Useful when DE/SPK kernel files aren't available. | 2 |
| [`oxiephemeris-compat`](crates/oxiephemeris-compat) | A Swiss-Ephemeris-*shaped* API surface (`Context`, body/flag constants, house systems, sidereal modes) built entirely from the SE public programmer's documentation, verified against 960 real `pyswisseph` output fixtures. | 50 |
| [`oxiephemeris-rdf`](crates/oxiephemeris-rdf) | The Linked Open Data layer: a custom `oxa:`/`oxc:`/`oxs:` astrology vocabulary (RDFS/OWL classes and SKOS concept schemes) generated from the engine's own enums and dignity tables, PROV-O provenance, and canonical byte-stable Turtle/N-Triples serialization. | 84 |
| [`oxiephemeris-lod`](crates/oxiephemeris-lod) | A SPARQL 1.1 store and HTTP endpoint: an in-memory, pure-Rust `oxigraph` store (`FileBackedStore`) with whole-file N-Quads persistence (no RocksDB), a pure-function SPARQL Protocol handler, and the `oxieph-sparqld` binary serving the vocabulary as dereferenceable Turtle. | 14 |
| [`oxiephemeris-chart`](crates/oxiephemeris-chart) | The UI-free chart-computation facade: `natal_chart`, `synastry`/`transit`/`progression`/`composite`, and their JSON/RDF serializers — the single compute path shared bit-for-bit by the CLI and the Python/WASM bindings. | 27 |
| [`oxiephemeris-py`](crates/oxiephemeris-py) | PyO3 (`abi3-py38`) Python bindings: `natal`, `synastry`, `transit`, `progress`, `composite`, and their RDF variants — DE ephemeris bytes in, a Python `dict` (JSON view) or `str` (RDF) out. | 0¹ |
| [`oxiephemeris-wasm`](crates/oxiephemeris-wasm) | `wasm-bindgen` WebAssembly bindings: the same natal/synastry/transit/progress/composite charts (JSON and Turtle) computed entirely offline in the browser from a DE ephemeris buffer. | 0¹ |
| [`oxiephemeris-cli`](crates/oxiephemeris-cli) | The `oxieph` binary: `convert` (JD ↔ calendar), `pos` (single or `--all` body positions), `houses`, and the chart family — `chart`, `synastry`, `transit`, `progress`, `composite` (thin renderers over `oxiephemeris-chart`) — plus `vocab` (emit the RDF vocabulary). Hand-rolled ISO 8601 parsing, no `chrono` dependency. | 79 |

*(Tests: `cargo nextest run -p <crate> --all-features`; 490 total across the
workspace, all green. ¹`oxiephemeris-py`/`oxiephemeris-wasm` are PyO3/
wasm-bindgen `cdylib` bindings with no `#[test]` functions of their own —
not a coverage gap; the logic they call into is exercised by
`oxiephemeris-chart`'s own test suite.)*

An internal `xtask` crate (dev-only, not published) generates the nutation,
VSOP87, and ELP2000-82B coefficient tables committed as Rust source.

## Live Linked Open Data

The OxiEphemeris astrology vocabulary is published as dereferenceable
Linked Open Data, with a live public SPARQL endpoint:

- **SPARQL endpoint:** <https://sparql.cooljapan.tech/> — a live query
  service (OxiRS, behind CloudFlare) preloaded with the OxiEphemeris `oxa:`
  ontology and `oxc:`/`oxs:` SKOS concept schemes. Its `/sparql` path speaks
  the SPARQL 1.1 Protocol:

  ```sh
  curl -G 'https://sparql.cooljapan.tech/sparql' \
    --data-urlencode 'query=ASK { <https://cooljapan.tech/ns/oxiephemeris/concept/sign/Scorpio> a <http://www.w3.org/2004/02/skos/core#Concept> }' \
    -H 'Accept: application/sparql-results+json'
  # {"head":{},"boolean":true}
  ```

- **Vocabulary namespace:** <https://cooljapan.tech/ns/oxiephemeris/> — the
  base IRI baked into `oxiephemeris-rdf`. Every `oxa:` term and every
  `oxc:`/`oxs:` SKOS concept dereferences with content negotiation (Turtle,
  N-Triples, or HTML) — e.g.
  <https://cooljapan.tech/ns/oxiephemeris/concept/sign/Scorpio>.

To generate this data yourself or self-host an equivalent endpoint, see
[`oxiephemeris-rdf`](crates/oxiephemeris-rdf) (the vocabulary and
serializers) and [`oxiephemeris-lod`](crates/oxiephemeris-lod) (the
self-hostable `oxieph-sparqld` binary).

## Verified accuracy

All figures below are measured against independent oracles (JPL test data,
published constants, or JPL Horizons/`pyswisseph` outputs) — not
self-consistency checks:

- JPL `testpo.440`: max error **1.421e-14 AU**. `testpo.441` (360,001 cases
  spanning a long BCE-to-future span): max error **2.842e-14 AU**.
- TT↔TDB analytical series vs published values: **2.4e-7 s**.
- Full IAU 2000A nutation vs oracle: **1.7e-14 arcsec**.
- JPL Horizons geocentric astrometric spot-checks: **0.000358 arcsec**;
  apparent (tied): **0.00219 arcsec**.
- Topocentric (Tokyo test site) vs Horizons: **≤0.0025 arcsec**.
- SPK (`.bsp`) reader vs the classic binary reader on the same ephemeris
  (DE440): **9.5e-7 km**; SPK type-13 Chebyshev-node reproduction:
  **8e-16 relative**.
- Lahiri ayanamsha at epoch 1999: **23.851°**, matching the published
  23°51′.
- Fixed stars / VSOP87 / ELP2000-82B: VSOP87 `vsop87.chk` reference (90
  cases): max **3.64e-7 AU**.
- 960-case `pyswisseph` (Swiss Ephemeris, binary wheel only, driven
  black-box — no SE source ever read) fixture comparison, all angles:
  position calc **5.5e-7°**, houses **6.1e-6°**, sidereal time
  **21 microseconds**, topocentric **4.6e-5°**.
- Sidereal-longitude convention, cross-stack: `oxiephemeris-astro`/CLI
  (`houses`/`chart --sidereal`) and `oxiephemeris-compat`
  (`SEFLG_SIDEREAL`) both subtract the ayanamsha referred to the *true*
  equinox of date (mean ayanamsha + nutation-in-longitude `Δψ`) —
  matching Swiss Ephemeris' measured output convention. `oxieph houses
  --sidereal` cross-checked against `oxiephemeris-compat`'s
  independently-implemented `Context::houses_ex(SEFLG_SIDEREAL)` at the
  same epoch/site/ayanamsha: agreement **< 1e-6°** (~0.0036″) on the
  Ascendant, MC and cusp 1
  (`houses_sidereal_matches_compat_houses_ex`,
  `crates/oxiephemeris-cli/tests/cli.rs`). See
  `crates/oxiephemeris-astro/src/ayanamsha.rs` for the ayanamsha model
  itself (mean-equinox value only, by design — each true-equinox-aware
  call site applies `Δψ` itself).

## Deferred to a later release

Full topocentric parallax corrections beyond the geocentric/topocentric
figures above, additional house systems beyond the 7 shipped, additional
ayanamshas beyond the 5 shipped, small-body/asteroid ephemerides, a
Moshier-analytic-series fallback matching Swiss Ephemeris' own, and SPK type
coverage beyond types 2/3/13.

`oxiephemeris-lod` also carries one known, already-documented security
issue: a transitive `quick-xml` RUSTSEC advisory pulled in via `oxigraph`,
with no upstream fix available yet. See [`CHANGELOG.md`](CHANGELOG.md) (the
0.1.1 entry's `### Security` note) and the crate's own README for the
mitigation and full detail; it does not affect any other crate in the
workspace.

## CLI quickstart

```sh
# From crates.io (recommended):
cargo install oxiephemeris-cli

# ...or from a source checkout:
cargo install --path crates/oxiephemeris-cli
```

This installs the `oxieph` binary with nine subcommands: `convert`, `pos`,
`houses`, `chart`, `synastry`, `transit`, `progress`, `composite`, and
`vocab`.

```sh
# Julian Date <-> calendar date conversion (no ephemeris file needed).
oxieph convert --date 2024-03-15T06:30:00
oxieph convert --jd 2451545.25 --json

# Apparent geocentric ecliptic-of-date position of a body (needs a DE
# kernel; see "Building from source" below).
oxieph pos sun 2000-01-01T12:00:00 --scale tt --de data/de440/linux_p1550p2650.440

# Every supported body at once, machine-readable.
oxieph pos --all 2026-07-05T12:00:00 --scale tt --de data/de440/linux_p1550p2650.440 --json

# House cusps and chart angles for a geographic site and UTC epoch.
oxieph houses --system placidus --lat 35.6895 --lon 139.6917 1999-03-30T20:15:00

# A full natal chart: bodies + houses + angles + nodes + aspect table.
oxieph chart 1999-03-30T20:15:00 --lat 43.06 --lon 141.35 \
  --de data/de440/linux_p1550p2650.440 --json

# Synastry: cross-aspects between two natal charts.
oxieph synastry --date-a 1970-01-01T00:00:00 --lat-a 51.4779 --lon-a 0.0 \
  --date-b 2000-01-01T12:00:00 --lat-b 48.8566 --lon-b 2.3522 \
  --de data/de440/linux_p1550p2650.440

# Transit: transiting-body aspects to a natal chart at a given instant.
oxieph transit 1970-01-01T00:00:00 --lat 51.4779 --lon 0.0 \
  --transit 2026-07-11T12:00:00 --de data/de440/linux_p1550p2650.440

# Secondary progression ("a day for a year") to a target date.
oxieph progress 1970-01-01T00:00:00 --lat 51.4779 --lon 0.0 \
  --target 2026-07-11T12:00:00 --de data/de440/linux_p1550p2650.440

# Midpoint composite chart of two natal charts.
oxieph composite --date-a 1970-01-01T00:00:00 --lat-a 51.4779 --lon-a 0.0 \
  --date-b 2000-01-01T12:00:00 --lat-b 48.8566 --lon-b 2.3522 \
  --de data/de440/linux_p1550p2650.440

# The oxa:/oxc:/oxs: RDF vocabulary itself, standalone (no DE file needed).
oxieph vocab --part ontology --format turtle
```

`pos`/`houses`/`chart` also accept `--sidereal <ayanamsha>` (e.g. `lahiri`)
to shift reported longitudes to a sidereal zodiac, and `--json` for the
stable machine-readable schema documented in each subcommand's module doc
comment (`crates/oxiephemeris-cli/src/{pos,houses,chart}.rs`). `pos` also
accepts `--frame {ecliptic,equatorial,j2000}` and
`--center {geo,helio,bary}`. If `--de`/`$OXIEPH_DE` is omitted, `pos`/`chart`
and the four comparison commands (`synastry`/`transit`/`progress`/
`composite`) fall back to `data/de440/linux_p1550p2650.440` relative to the
current directory. If you installed from crates.io (no repo checkout),
download that DE440 file once with

```sh
curl -fsSL --create-dirs -o data/de440/linux_p1550p2650.440 \
  https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440
```

(DE440, public domain, ~114 MB), or point `$OXIEPH_DE` at a copy you already
have; from a repo checkout, `scripts/fetch_de440.sh` fetches the same file.

Every chart-family command (`chart`, `synastry`, `transit`, `progress`,
`composite`) also accepts a unified `--format text|json|turtle|ntriples`
flag (`text` is the default); the legacy `--json` flag is still honored
when `--format` is left at its default. `--base-iri` (RDF formats only)
controls the minted subject IRIs; `chart` alone also accepts `--chart-iri`
to pin an exact IRI instead of minting one. `oxieph vocab` emits the
`oxa:`/`oxc:`/`oxs:` vocabulary standalone — the same one
[`oxiephemeris-lod`](crates/oxiephemeris-lod) serves over HTTP. Run
`oxieph <command> --help` for the complete, current flag reference for any
subcommand.

## Building from source

```sh
git clone https://github.com/cool-japan/oxiephemeris
cd oxiephemeris
cargo build --workspace --all-features
cargo nextest run --workspace --all-features
```

Most of the test suite (and `pos`/`chart`'s DE-backed computations) need
ephemeris and reference data files that are public-domain/published but not
committed to this repository. Fetch scripts are provided for some of them
under `scripts/`; see [`CONTRIBUTING.md`](CONTRIBUTING.md#data-setup) for
the full list of required data files, what each fetch script retrieves, and
which files still require a manual download.

The workspace currently passes, with `unsafe_code = "forbid"` at the
workspace lint level and no `.unwrap()`/`.expect()` in library code
(`clippy::unwrap_used`/`expect_used` denied workspace-wide):

```sh
cargo nextest run --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all -- --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

## Repository

<https://github.com/cool-japan/oxiephemeris>

## License

Apache-2.0 © 2026 COOLJAPAN OU (Team Kitasan). See [`LICENSE`](LICENSE).
