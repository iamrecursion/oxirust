# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - Unreleased

### Changed

- **RDF model and Turtle codec** (`oxiephemeris-rdf`, and through it
  `-chart`, `-cli`, `-py`, `-wasm`): the graphs this workspace builds and
  serializes are now made of `oxixml-model` and `oxixml-turtle` instead of
  `oxrdf` and `oxttl`. **Breaking for downstream consumers**: the
  `oxiephemeris_rdf::oxrdf` re-export keeps its name but now names
  `oxixml-model`, so `oxiephemeris_rdf::oxrdf::Graph`, `NamedNode`, `Term`
  and friends are different types from the `oxrdf` crate's. Code that
  mixes them with terms obtained from `oxrdf` directly (for example via
  `oxigraph::model`) must convert between the two. Emitted Turtle and
  N-Triples are unchanged, and remain byte-stable.
- **API shifts that follow the model change** (`oxiephemeris-rdf`):
  `Graph::iter` and `Graph::objects_for_subject_predicate` now yield owned
  `Triple`/`Term` rather than `TripleRef`/`TermRef`, `Graph::insert` takes
  an owned `Triple`, and `Literal::datatype` returns `&NamedNode`. A new
  `vocab::xsd` module publishes the XSD datatype IRIs as `NamedNodeRef`
  constants, which is the form the ontology's `const` property table needs.
- **SPARQL store** (`oxiephemeris-lod`): keeps `oxigraph` (and therefore
  `oxrdf`) for storage and query evaluation, so `FileBackedStore::load_graph`
  now copies each term from the document model into the store model. The
  copy is structural rather than a serialize/parse round trip, which
  preserves blank-node identity. `load_graph`'s `name` parameter is now an
  `oxigraph::model::NamedNodeRef`.
- **First-run experience** (`oxiephemeris-cli`): documented
  `cargo install oxiephemeris-cli` as the primary install path, and the
  "DE ephemeris file missing" error now also prints a repo-independent
  direct-download command (the exact JPL DE440 URL) next to the existing
  `scripts/fetch_de440.sh` hint — so a crates.io install can obtain the
  ephemeris without a repo checkout.

### Removed

- **`getrandom` dependency** (`oxiephemeris-wasm`): the wasm-backend
  workaround existed only for the transitive `oxrdf -> rand -> getrandom`
  chain. `oxixml-model` needs neither `rand` nor `getrandom`, so both are
  gone from the WebAssembly dependency graph.

## [0.1.1] - 2026-07-11

A Linked Open Data release: a full RDF/SKOS/PROV-O astrology vocabulary
with a SPARQL store and HTTP endpoint, a UI-free chart-computation
facade shared bit-for-bit by the CLI and new Python and WebAssembly
bindings, and a substantially deeper astrology layer — sidereal
ayanamshas, essential dignities, declinations, Arabic Parts, midpoints,
and synastry cross-aspects.

### Added

- **Linked Open Data layer** (`oxiephemeris-rdf`): a custom `oxa:` astrology
  ontology and SKOS concept schemes (`oxc:`/`oxs:`) generated from the
  engine's own enums and dignity tables so they cannot drift from the code
  that computes charts; PROV-O provenance (`add_provenance`); canonical,
  byte-stable Turtle/N-Triples serialization (`to_turtle_string`,
  `to_ntriples_string`, `write_turtle`, `write_ntriples` — triples sorted
  so re-emitting never produces a spurious diff); bilingual English/Japanese
  labels for every astrological concept (`labels` module); Wikidata QID
  cross-references for the ten planets and twelve signs
  (`vocab::PLANET_WIKIDATA`, `vocab::SIGN_WIKIDATA`); and graph-closure
  tests (`tests/closure.rs`) asserting every emitted `oxa:` predicate/type
  is declared in `ontology_graph`.
- **SPARQL store and HTTP endpoint** (`oxiephemeris-lod`): `FileBackedStore`,
  an in-memory `oxigraph` store with pure-Rust whole-file N-Quads
  persistence (no RocksDB); `endpoint::handle`, the SPARQL 1.1 Protocol
  handler implemented as a pure function so the whole HTTP surface is
  testable without binding a socket; and the `oxieph-sparqld` binary
  (`--bind`, `--store`, `--read-only`, `--load`, `--vocab`) for serving the
  published `oxa:`/`oxc:`/`oxs:` IRIs as dereferenceable Turtle.
- **Chart-computation facade** (`oxiephemeris-chart`): `natal_chart`,
  `synastry`/`transit`/`progression`/`composite`, and their JSON/RDF
  serializers (`chart_to_json`, `chart_to_rdf`, `comparison_to_json`,
  `comparison_to_rdf`, `composite_to_json`, `composite_to_rdf`) — the
  single UI-free source of truth now shared bit-identically by the CLI and
  the new Python and WASM bindings, including its own dependency-free
  ISO 8601 parser (`iso8601::parse`) and stable JSON schema
  (`json::ChartJson` and friends).
- **Python bindings** (`oxiephemeris-py`, PyO3, `abi3-py38`): an
  `oxiephemeris` module exposing `julday`, `revjul`, `natal`, `natal_rdf`,
  `synastry`, `synastry_rdf`, `transit`, `progress`, `composite`, and
  `composite_rdf` — DE ephemeris bytes in, a Python `dict` (JSON view) or
  `str` (RDF) out.
- **WebAssembly bindings** (`oxiephemeris-wasm`, `wasm-bindgen`):
  `natal_json`/`natal_turtle`, `synastry_json`/`synastry_turtle`,
  `transit_json`, `progress_json`, and `composite_json`/`composite_turtle`
  — full natal, synastry, transit, progression, and composite charts,
  including Linked Open Data (Turtle) output, computed entirely offline in
  the browser from a DE ephemeris buffer.
- **Astrology layer additions** (`oxiephemeris-astro`): a sidereal-zodiac
  `ayanamsha` module (`Ayanamsha` with `FaganBradley`, `Lahiri`,
  `Krishnamurti`, `Raman`, `J2000Zero`, and `Custom` variants;
  `ayanamsha_rad`, `sidereal_from_tropical`) propagated from cited anchor
  epochs via the IAU 2006 (P03) general-precession polynomial; essential
  dignities (`dignities`: domicile/exaltation/triplicity/term/face rulers,
  detriment/fall/peregrine via `EssentialDignity`/`essential_dignity`, a
  configurable `RulershipScheme`, and element/modality `Distribution`);
  declinations (`declination`: ecliptic→equatorial `declination`,
  `antiscia`/`contra_antiscia`, and parallel/contraparallel/out-of-bounds
  `DeclinationAspect`, cross-checked against an independent rotation-based
  oracle in `tests/declination_oracle.rs`); Arabic Parts (`parts`:
  day/night `Sect`, generic `lot`, `part_of_fortune`, `part_of_spirit`);
  midpoints (`midpoints`: near/far midpoint plus a batch helper for
  composite-chart construction); motion classification
  (`motion::MotionState`: direct/retrograde/stationary); cross-chart
  aspects (`synastry`: `CrossHit`, `for_each_cross_aspect`,
  `cross_aspects_into`, feeding synastry/transit/progression comparisons);
  zodiac decomposition (`zodiac`: `Element`, `Modality`, `Sign`,
  `SignPosition`); and `HouseSystem::ALL`/`HouseSystem::name()` for stable
  enumeration and CLI-matching names.
- **CLI subcommands** (`oxieph`): `synastry` (cross-aspects between two
  natal charts), `transit` (transiting-body aspects to a natal chart),
  `progress` (secondary-progressed "day-for-a-year" chart), `composite`
  (midpoint composite chart), and `vocab` (emit the `oxa:` ontology and
  SKOS concept schemes as RDF); all chart-family commands gain a unified
  `--format text|json|turtle|ntriples` flag (`rdf_args::OutputFormat`),
  with the legacy `--json` flag still honored when `--format` is left at
  its default.
- **Ephemeris provenance** (`oxiephemeris-de`): `DeFile::de_number()`
  exposes the DE header's `NUMDE` field (e.g. `440`, `441`) for recording
  ephemeris identity in RDF provenance output.

### Changed

- **Facade delegation** (`oxiephemeris-cli`): `chart` and `houses` are now
  thin renderers over `oxiephemeris_chart::natal_chart`/`compute_houses`;
  epoch resolution moved out of the CLI's own `astro_epoch` module and into
  `oxiephemeris_chart::epoch`; and `CliError::ChartGeometry`/
  `CliError::Node` were replaced by `CliError::Chart`/`CliError::Rdf`,
  which wrap the facade's own error types — so the CLI, Python, and WASM
  bindings now share one compute path end to end.
- **RDF output plumbing** (`oxiephemeris-cli`): RDF emission in the
  comparison commands (`synastry`, `transit`, `progress`, `composite`) was
  consolidated behind an internal `RdfOutput` struct bundling `base_iri`,
  `format`, and `de_number`, replacing three separate parameters previously
  threaded through `emit_comparison_rdf`/`emit_composite_rdf`.
- **MSRV**: raised from 1.81 to 1.89 (`rust-version` in
  `[workspace.package]`), required by the new RDF/SPARQL dependency stack
  (`oxigraph`/`oxrdf`/`oxttl` declare `rust-version = "1.87"` and
  `edition = "2024"`).
- **Reference chart** (tests, examples, READMEs): switched from an
  arbitrary birth date to the Unix epoch (1970-01-01T00:00:00Z) at the
  Royal Observatory, Greenwich, giving every worked example — including the
  new `oxiephemeris-chart` end-to-end DE440 test — a canonical,
  easily-reproduced reference point.

### Security

- **Known issue** (not fixed in this release): `oxiephemeris-lod` pulls
  in `quick-xml 0.37.5` transitively via `oxigraph 0.5.9`, which carries
  2 HIGH-severity (7.5) RUSTSEC advisories ([RUSTSEC-2026-0194](https://rustsec.org/advisories/RUSTSEC-2026-0194.html),
  [RUSTSEC-2026-0195](https://rustsec.org/advisories/RUSTSEC-2026-0195.html)) —
  both denial-of-service via crafted XML (unbounded namespace-declaration
  allocation; quadratic duplicate-attribute checking). No `oxigraph`
  release fixes this yet, and force-patching `quick-xml` to `>=0.41`
  fails to compile against `oxrdfxml`/`sparesults`'s pinned API (verified).
  **Mitigation**: do not expose `oxieph-sparqld` to untrusted networks
  and do not load untrusted RDF/XML via `--load` until this is resolved
  upstream. Tracked for a future patch release.

## [0.1.0] - 2026-07-10

Initial release: a clean-room, Pure Rust ephemeris engine — an independent
JPL DE/SPK reader and IAU 2006/2000A frame implementation with an
SE-compatible API surface, an astrology layer, a no-data-file analytic
fallback, and a CLI.

### Added

- **Core time scales** (`oxiephemeris-core`): two-part `JulianDate` (`hi`/`lo`)
  arithmetic for extended precision; proleptic Gregorian and Julian calendar
  ↔ Julian Date conversions (`julday`/`revjul`), BCE sign-correct
  (astronomical year numbering); TAI↔TT (constant 32.184 s); an embedded
  UTC leap-second table with a documented update procedure; the
  Fairhead–Bretagnon TT↔TDB analytical series; and the full 15-piece
  Espenak–Meeus ΔT (TT−UT1) polynomial model (−500 to +2150, with
  long-term parabola tails outside that range).
- **JPL DE ephemeris support** (`oxiephemeris-de`): classic binary DE
  header/record parsing with automatic endianness detection (`NCON`
  sanity check); Chebyshev interpolation via Clenshaw recurrence (position
  and velocity); DE440 and DE441 support; and a SPK/DAF (`.bsp`) reader
  (DAF container, little/big-endian, summary and name record chains) for
  segment Types 2 and 3 (Chebyshev) and Type 13 (Hermite on doubled nodes).
- **Reference frames & nutation** (`oxiephemeris-bodies`): IAU 2006
  frame bias and precession (Fukushima–Williams angles); the full
  IAU 2000A_R06 nutation series (luni-solar and planetary terms, generated
  from the IERS Conventions electronic tables) plus a truncated,
  IAU 2000B-accuracy-class fast path selectable at runtime; IAU 2006 mean
  obliquity; and Earth-rotation/sidereal-time support (ERA, GMST, GAST,
  and the equation of the equinoxes with complementary terms).
- **Topocentric observers**: WGS84 geodetic-site → ITRS conversion,
  polar-motion matrix `W(t)` with the TIO locator `s'`, an Earth
  Orientation Parameters (EOP) struct for polar motion and ΔUT1, and
  station state (diurnal aberration + parallax) feeding the apparent-place
  pipeline.
- **Apparent-place pipeline**: light-time iteration, relativistic annual
  aberration, optional solar gravitational light deflection, and
  frame-flag selection (J2000/ICRS, mean-of-date, true-of-date, ecliptic
  mean/true) with daily speeds via central differences; a fixed-star
  apparent-place variant (geocentric and topocentric) covering space
  motion, parallax, and the shared deflection/aberration/frame chain.
- **Astrology layer** (`oxiephemeris-astro`): Ascendant, Midheaven (MC),
  Vertex, and East Point; seven house systems (Placidus, Koch, Whole
  Sign, Equal, Porphyry, Regiomontanus, Campanus), each verified against
  an independent oracle of its own defining condition, with polar
  Placidus/Koch surfaced as a typed error rather than a silent fallback;
  eleven classical aspects (conjunction, opposition, trine, square,
  sextile, semisextile, semisquare, sesquiquadrate, quincunx, quintile,
  biquintile) with configurable per-aspect orbs and
  applying/separating classification; mean and true (osculating) lunar
  node and apogee; a sidereal zodiac with four named ayanamsha systems
  (Fagan/Bradley, Lahiri, Krishnamurti, Raman) plus a J2000-zero and a
  caller-supplied custom-anchor mode; and a 177-star Hipparcos
  (V ≤ 3.0) subset with 18 IAU proper names feeding the fixed-star
  pipeline.
- **SE-compatible surface** (`oxiephemeris-compat`): an instance-local
  `Context` (no process-global state) exposing `julday`, `revjul`,
  `calc`/`calc_ut`, `houses`/`houses_ex`, `sidtime`, `set_topo`
  (and `set_topo_with_eop`), `set_sid_mode`, and `get_ayanamsa`; full
  `SEFLG_*` flag decoding (`HELCTR`, `BARYCTR`, `TOPOCTR`, `J2000`,
  `NONUT`, `ICRS`, `EQUATORIAL`, `XYZ`, `RADIANS`, `SIDEREAL`, `SPEED`,
  `SPEED3`, `TRUEPOS`, `NOABERR`, `NOGDEFL`, with unrecognized bits
  rejected as a typed error); and SE body numbers 0–14 (Sun through
  Earth, including the mean/true lunar node and mean/osculating apogee).
- **Analytic fallback** (`oxiephemeris-analytic`): truncated, committed
  series for VSOP87E (barycentric rectangular Sun + 8 planets, J2000)
  and ELP2000-82B (geocentric Moon, J2000), generated from the published
  CDS tables — no ephemeris data file required (Pluto is not available
  from these theories, since it needs a DE or SPK file).
- **CLI** (`oxieph`): `convert` (Julian Date ↔ proleptic calendar date),
  `pos` (apparent/astrometric-style body position, including `--all` for
  every supported body), `houses` (cusps and chart angles for a
  geographic site and epoch), and `chart` (bodies + houses + angles +
  nodes + a full aspect table) — all with stable `--json` output and
  hand-rolled ISO 8601 parsing (no `chrono` dependency).
- **Verification**: JPL `testpo.440` and `testpo.441` golden-value tests
  (≤ 1e-13 AU); 10 JPL Horizons geocentric spot-checks plus additional
  Horizons fixture sets for sidereal time and topocentric positions
  (all < 0.01″); 960 SE-output fixture comparisons across `calc`/
  `calc_ut`/houses/sidereal-time/topocentric/nodes/apogee (all
  < 0.001° on angles); and an independent, adversarially-verified 9-track
  code-review pass over the hand-written (non-generated) code, which
  found and fixed 2 critical/major defects (both in the SPK Type 13
  reader) and 8 minor issues.
- **Quality**: 306 tests passing across the workspace (0 failures);
  `cargo clippy --workspace --all-features --all-targets -- -D warnings`
  clean; no file over 2000 lines; and zero `unsafe`, zero `unwrap()`,
  and zero `expect()` anywhere in library code.

[0.1.0]: https://github.com/cool-japan/oxiephemeris/releases/tag/v0.1.0
[0.1.1]: https://github.com/cool-japan/oxiephemeris/releases/tag/v0.1.1
