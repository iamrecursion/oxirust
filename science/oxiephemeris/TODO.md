# OxiEphemeris — Phase 1 TODO

Task order per CLAUDE.md. Each step ends with green tests.

## Infrastructure
- [x] Workspace skeleton (5 crates + xtask)
- [x] `scripts/fetch_de440.sh` — testpo.440 + DE440 binary + reference material into `data/` (gitignored)

## Step 1 — testpo parser
- [x] `de/tests/testpo.rs`: parser for JPL `testpo.440` text format; test parses N lines

## Steps 2–3 — core::time
- [x] Two-part JD arithmetic (`JulianDate { hi, lo }`)
- [x] Proleptic Gregorian AND Julian calendar <-> JD, BCE sign-correct (astronomical year numbering)
- [x] Property tests: julday<->revjul round-trip over [-3000, +3000] CE, both calendars
- [x] TAI<->TT (const 32.184 s)
- [x] Leap-second table (const array, documented update procedure), UTC<->TAI<->TT
- [x] TT<->TDB Fairhead–Bretagnon truncated (123 terms, target < 1 µs;
      oracle vs DE440t integrated TT−TDB: max 2.4e-7 s over 1900–2100,
      `de/tests/tdb_oracle.rs`)

## Steps 4–5 — de crate
- [x] Classic binary DE header parse (endianness detection via NCON sanity check)
- [x] Record/granule lookup
- [x] Chebyshev evaluation (Clenshaw, position + velocity)
- [x] **Exit: testpo.440 fully green at <= 1e-13 AU**

## Step 6 — bodies frame math
- [x] IAU 2006 precession + frame bias (Fukushima–Williams angles)
- [x] Truncated nutation, IAU 2000B accuracy class: truncated IAU 2000A_R06
      series (143 luni-solar + 11 planetary terms + constant planetary bias).
      NOTE: the verbatim McCarthy & Luzum (2003) 77-term IAU 2000B table is
      not obtainable from clean-room-permitted sources, so the truncation is
      regenerated from the IERS TN36 electronic tables instead; the API is
      named `nutation_iau2000a_truncated` accordingly
- [x] Obliquity IAU 2006
- [x] Oracle test vs full IAU 2000A_R06 series (data/iers/tab5.3a.txt AND
      tab5.3b.txt, both fetched by scripts/fetch_de440.sh) when present

## Step 7 — apparent-place pipeline
- [x] Light-time iteration (tol 1e-12 d; c/AU taken from the DE header)
- [x] Annual aberration (relativistic, Kaplan et al. 1989 eqs. 16–17)
- [x] Gravitational light deflection by Sun (optional flag; GMS from the
      DE header; occultation no-op)
- [x] Frame flags: J2000/ICRS / mean-of-date / true-of-date / ecliptic
      (mean and true), speeds via central differences with 2π-wrap handling

## Step 8 — CLI
- [x] `oxieph convert` (JD <-> calendar, both directions, BCE-safe, `--json`)
- [x] `oxieph pos <body> <iso8601>` (hand-rolled ISO 8601 parse, NO chrono;
      `--frame`, `--center`, `--scale utc|tt`, `--json`, friendly errors)

## Step 9 — Horizons verification
- [x] 10 Horizons spot-check fixtures committed under `crates/oxiephemeris-bodies/tests/fixtures/`
- [x] Spot checks < 0.01": astrometric gate max 0.000358" (Moon,
      DE441-vs-DE440 level); apparent gate ≤ 2.08 mas after the published
      −52.93 mas equinox tie (IERS TN36 eq. 5.21: dα₀ + ξ₀/tan ε₀; Horizons'
      own footer documents "−53 mas") — tolerance itself untouched at 0.01";
      range gate max 1.4e-12 AU. All fixture epochs inside the Horizons EOP
      data span (see fixtures README, EOP-coverage epoch policy)

# Phase 2 & 3 (authorized by user 2026-07-05: "Phase 2 and beyond")

## Wave A — Earth rotation & ephemeris depth — ALL DONE (2026-07-06)
- [x] `bodies::sidereal`: ERA (TN36 eq. 5.15, two-part-JD exact), GMST
      (IAU 2006), equation of the equinoxes + complementary terms, GAST;
      oracle vs Horizons quantity 7 (LAST) < 1 ms after the documented
      −52.93 mas equinox tie: tied residuals −0.123 ms (1987-04-10),
      −0.039 ms (1998-11-10), −0.051 ms (2019-03-05); the 1975-06-20
      epoch is kept ungated/informational (+1.34 ms tied — pre-VLBI
      UT1-series disagreement, investigated and documented in the
      fixture README; the 1 ms gate was NOT loosened)
- [x] Full IAU 2000A_R06 nutation: real xtask generator
      (`cargo run -p xtask -- gen-nutation`) from data/iers/tab5.3a+b,
      committed chunked tables (1320+38 ψ / 1037+19 ε terms; largest
      table file 732 lines after fmt), oracle exact-match vs runtime
      table evaluation: max 1.677e-14″ (dψ) / 8.386e-15″ (dε) over
      1800–2200 (gate 1e-9″); truncated-vs-full 0.6367/0.3424 mas over
      1995–2050, 1.0294/0.3781 mas over 1900–2100
- [x] DE441: fetch script (`scripts/fetch_de441.sh`), header parse
      (NCON = 645 > 400 name block, NCOEFF = 1018, 342 419 records,
      span JD [−3027215.5, 7930192.5]); testpo.441 green: 360 001
      in-span cases, max 2.842e-14 AU (gate 1e-13, skip-if-absent);
      fixed-column testpo parsing (testeph.f `READ` format) replaced the
      whitespace-split parser that mis-tokenized 5-digit signed years
- [x] `core::time::delta_t`: full 15-piece Espenak–Meeus polynomial
      (−500..2150 + long-term parabola tails), cited, 11 transcription
      checkpoints at 1e-9 s, published-value spot checks, boundary
      continuity across all 14 joins: max |jump| 0.251 s at year 1600
      (gate 1 s)
- [x] Topocentric: WGS84 geodetic → ITRS (NIMA TR8350.2 / EPSG 9602),
      polar motion W(t) with s′ = −47 µas·t + EOP struct, ITRS→GCRS via
      GAST, station state in apparent pipeline (diurnal aberration +
      parallax); Horizons Tokyo spot-checks (Moon ×2 / Mars / Venus)
      with per-epoch EOP from finals2000A.all: astrometric max 0.0012″,
      tied apparent max 0.0025″ (gate 0.01″), range max 8.1e-13 AU;
      Moon parallax 0.974°/0.701° sine-rule-consistent (residual
      ≤ 2.6e-7)
- [x] Integration: full 2000A as pipeline default via
      `apparent::Options::nutation` (`frames::NutationModel`, `Default`
      = full `Iau2000a`; `Iau2000aTruncated` kept as the ≈ 15× cheaper
      option; frames grew `*_with(t, model)` matrix builders). All
      workspace gates re-green 2026-07-06: fmt clean, clippy
      --workspace --all-features --all-targets -D warnings clean,
      nextest 152/152 (incl. golden_de441), largest file 847 lines
      (apparent.rs). Geocentric Horizons re-run under the full-series
      default: astrometric max 0.000358″, frame-tied apparent max
      0.00219″ (was 0.00208″ with the truncated series — both ≪ 0.01″);
      `bodies::sidereal` deliberately stays on the truncated Δψ
      (< 1 mas ≈ 0.07 ms in EE, inside its 1 ms gate; documented)

## Wave B — astro layer & SE-compat surface — ALL DONE (2026-07-06)
- [x] `oxiephemeris-astro` crate: ASC/MC/Vertex/East Point (vector
      closed forms re-derived from the horizon/ecliptic/prime-vertical
      intersection geometry, matrix cross-checked); houses: Placidus
      (fixed-point iteration with a Banach contraction proof, rate
      < 2/3 below the polar circle, bisection fallback), Koch, Whole
      Sign, Equal, Porphyry, Regiomontanus, Campanus — every system
      verified against an independent brute-force oracle of its
      defining condition; polar Placidus/Koch = typed error, never a
      silent fallback. Adversarially verified green (round 1)
- [x] Aspects (11 classical kinds, per-aspect orbs,
      applying/separating incl. retrograde truth table); mean lunar
      node/perigee/apogee (Simon et al. 1994 §3.4(b.3), digit-checked
      against the paper), true (osculating) node/apogee (pure
      (r, v, mu) two-body elements + DE convenience; mu = header GMB,
      never a literature fallback); Horizons osculating-elements
      oracle: 4 epochs, |dOM| max ~2e-4 deg. Verified green (round 1)
- [x] Sidereal modes: Fagan/Bradley (B1950 SVP), Lahiri (1956 CRC
      decree), Krishnamurti (1900 anchor), Raman (1912 anchor, his own
      worked example), J2000Zero + Custom — anchor + IAU2006 p_A
      propagation, honesty-documented accuracy notes. Verified green
      (round 2: Raman anchor transposition caught & fixed)
- [x] `compat` crate: Context (no global state), julday/revjul,
      calc/calc_ut (full SEFLG_* decode: HELCTR/BARYCTR/TOPOCTR/
      J2000/NONUT/ICRS/EQUATORIAL/XYZ/RADIANS/SIDEREAL/SPEED/SPEED3/
      TRUEPOS/NOABERR/NOGDEFL; unknown bits = typed error), houses/
      houses_ex (SE ascmc block), sidtime, set_topo(_with_eop),
      set_sid_mode, get_ayanamsa; SE body numbers 0–14 incl. nodes/
      apogees; documented differences section. 47 tests incl. 12
      DE-identity tests vs bodies/astro (agreement 1e-12 deg)
- [x] CLI: `pos --all` (10-body JSON), `oxieph houses` (7 systems,
      --sidereal, friendly polar errors), `oxieph chart` (bodies +
      houses + angles + nodes + 45-pair aspect table, stable JSON);
      end-to-end verified against direct library computation (1e-9 deg)
      and astronomically sanity-checked (1999-03-31 full-moon chart)
- [x] Workspace gates 2026-07-06: fmt clean, clippy --workspace
      --all-features --all-targets -D warnings clean, RUSTDOCFLAGS
      -D warnings doc clean, nextest 265/265 (incl. golden_de441),
      largest file 847 lines (apparent.rs)
- NOTE: compat + cli tracks lost their workflow agents to the API
      weekly limit mid-run; finished + reviewed + identity-tested
      personally by the orchestrator (no independent adversarial pass —
      compensated by the DE-identity test suite and end-to-end runs)
- [x] SE-output fixture gate (done 2026-07-06): fixtures generated by
      the quarantined external tool `../oxieph-fixgen` (pyswisseph
      **binary wheel only** — no SE source ever fetched; SE driven as a
      black box through its public API; the OxiEphemeris side stays
      clean-room). SE 2.10.03 reads the *same* DE440 classic binary
      (`SEFLG_JPLEPH`, asserted per call), isolating the pipelines from
      ephemeris differences. 960 cases, all < 0.001° on angles
      (`compat/tests/se_fixtures.rs`, skip-if-absent): calc frames max
      5.5e-7° (2 mas), houses 6.1e-6°, sidtime 21 µs, topocentric
      4.6e-5°, calc_ut 8.2e-4° (= known EM-vs-SE ΔT gap at 2024), mean
      node 5.0e-5°, true node 2.1e-6°, oscu apogee 7.7e-5°. Fixture
      findings FIXED in compat/bodies: real SEFLG_TRUEPOS (new
      `Options::light_time` toggle — was documented-simplified),
      SEFLG_ICRS bias-omission modifier semantics (was 1.1 au off on
      Pluto XYZ), sidereal = ayanamsha + Δψ convention (was 14″ off),
      topocentric speed step 0.001 d (was 0.077°/day truncation),
      geocentric speed step 0.05 → 0.01 d. Documented SE-side
      conventions (gated at measured size, not mimicked): SE helio uses
      *barycentric* light-time (≤ 0.4″; proven by output probes), SE
      speeds ≠ derivative of SE's own positions (9e-5 geo / 5e-3
      topo °/day self-inconsistency), SE apparent-distance-rate extra
      term, SE Lilith model 0.115°, ayanamsha anchors ≤ 9.6e-3°

## Wave C — Phase 3 — ALL DONE (2026-07-06)
- [x] SPK/DAF (.bsp) reader (`de::spk`, no_std, done 2026-07-06): DAF
      container (LTL/BIG-IEEE, summary/name chains) + SPK Types 2, 3
      (Chebyshev, reusing the Clenshaw core) and 13 (Hermite on doubled
      nodes, divided differences), from NAIF's public daf.req/spk.req
      docs only. Two-part ET (a single-f64 ET carries ~2 µs ulp ≈ 100 m
      of Mercury — caught by a velocity-proportional error signature and
      fixed with Sterbenz-exact splits). Oracles: de440.bsp vs classic
      DE440 cross-format max 9.5e-7 km / 2.3e-9 km/day over 200 epochs
      × 11 bodies (gates 1e-5 km / 1e-7 km/day); geocentric Moon
      1.2e-10 km; codes_300ast Type 13 node reproduction 8.2e-16 rel;
      Ceres physical sanity + finite-difference velocity consistency
- [x] Analytic fallback crate (no data files, done 2026-07-06):
      `oxiephemeris-analytic` — truncated VSOP87E (barycentric
      rectangular J2000, Sun + 8 planets; Bretagnon & Francou 1988) and
      ELP2000-82B (geocentric Moon; Chapront-Touzé & Chapront 1983/88),
      xtask-generated (`gen-vsop87`, `gen-elp`) from the CDS VI/81 and
      VI/79 tables (fetched by `scripts/fetch_analytic.sh`). Replaces
      the "Moshier" plan with the primary published theories; **no
      Pluto** (needs DE/SPK — documented). VSOP87E: 35,605 of 51,880
      terms kept, per-body cutoffs 1e-9..3e-8 au; analytic-derivative
      velocities; own oracle vs the BDL `vsop87.chk` full-series values
      (90 cases: max 3.64e-7 au / 9.3e-9 au/day = pure truncation
      error). ELP: main problem + ζ-group + planetary Table 2 in full
      with the DE200-fit corrections baked in at generation; Table 1
      cut at 0.0005″ (3,654 of 26,190 kept); Laskar P/Q precession →
      J2000; published ecliptic→FK5(≈ICRS) tie. Oracle vs DE440
      (1900–2100, 84 epochs): barycentric planets carry the common
      DE200-era SSB offset (~9.7e-6 au, all Sun..Mars identical);
      geocentric differences cancel it — apparent Sun 1.9e-7 au
      (~0.04″), Mercury/Venus ~2e-7 au, Jupiter 9.2e-6, Neptune 3.3e-4
      (pre-Voyager DE200 orbit); Moon 2.67 km / 0.64 km/day, and
      1700–2200 Moon 14.6 km (DE200-era tidal-acceleration drift).
      Long-span geocentric planets ≤ 3.7× the century gates (1650–2400)
- [x] Fixed stars (done 2026-07-06): `bodies::star::apparent_star`
      (+ topocentric variant) — rectangular space motion (ESA 1997
      §1.5.5), inverse-parallax barycentric vector, parallax floor
      1 µas, then the shared deflection/aberration/frame chain (Kaplan
      1989 §IV; Circ 179 §7.2). `astro::stars`: 177-star Hipparcos
      V ≤ 3.0 subset via `xtask gen-stars` + 18 IAU proper names.
      Physics tests: identity (< 2e-11 rad), aberration constant
      κ(1±e) at the ecliptic pole, 500 mas parallax circle, 20 yr
      proper-motion linearity (< 0.02″), Polaris apparent pole distance
      2026 ∈ [0.60°, 0.73°] with ICRS cross-check, Sirius apparent 2026
      RA/Dec bands, catalog integrity + named-star identity checks
- [x] DE441 long-span property tests (BCE epochs, done 2026-07-06): six
      internal-consistency laws over the BCE half of the span
      (`de/tests/de441_longspan.rs`, one test so the 2.6 GB binary loads
      once) — two-part-split invariance (5 splits, bit-level), record-
      boundary continuity (49 BCE boundaries, worst gap 1.2e-7 km),
      Chebyshev rate vs central difference, Mercury/Moon orbital-band
      sanity (200 epochs), julday<->revjul round-trip over the full span
      (10,815 epochs × both calendars, endpoint years −13000/+17000),
      span-edge evaluation + EpochOutOfRange just outside

## Independent adversarial verification pass — DONE (2026-07-06)
- [x] 9-track review workflow over all personally-written code (compat
      calc/flags/houses/time, de::spk, bodies::star, analytic VSOP+ELP,
      cli, de441 property tests, recent apparent.rs changes), each
      critical/major finding adversarially verified by 3 independent
      refuters (math re-derivation / code trace / reproduction).
      Result: analytic (both theories), time/houses, and the DE441
      property tests came back CLEAN. Confirmed + fixed:
      * CRITICAL `de::spk` type 13: trailer stores WINDOW−1 (spk.req),
        code used it as WINDOW → degree-13 instead of degree-15
        interpolation, up to 0.78 km off SPICE truth on codes_300ast
        (verifiers proved it from the raw kernel bytes + independent
        Hermite re-implementation; node-reproduction tests are blind to
        window size by construction). Fixed (+1); new synthetic
        degree-15 polynomial off-node test discriminates the bug at
        3 orders of magnitude in both directions (no data files needed)
      * MAJOR `de::spk` type 13: directory count is ⌊(N−1)/100⌋, code
        validated with N/100 → valid kernels with N a multiple of 100
        rejected as Corrupt. Fixed + N=100 synthetic test
      * 8 minors, all fixed: NaN control values / usize overflow in spk
        validation (typed Corrupt now, + NaN-epoch test); calc_node
        silently accepting SEFLG_ICRS-alone; SIDEREAL+J2000/ICRS
        mixed-epoch combination now a typed conflict; star pipeline
        speed steps aligned with the planetary pipeline (0.01/0.001 d);
        cli format_year i32::MIN off-by-one; houses/chart pre-1972
        epochs now WORK via the documented Espenak–Meeus ΔT fallback
        (was: unusable with an error suggesting a flag they don't
        have; + CLI test at 1965); light_time_days doc corrected
      All gates re-green: nextest 294/294, clippy -D warnings, doc, fmt

## Phase 1 exit criteria — ALL MET (2026-07-05)
- [x] testpo.440 green (<= 1e-13 AU): max 1.421e-14 AU over 13,201 cases
      (DE440 and DE440t both)
- [x] 10 Horizons spot-checks < 0.01": astrometric max 0.000358",
      frame-tied apparent max 0.00208" (see step 9 above)
- [x] `cargo clippy --workspace --all-features --all-targets -- -D warnings` clean
- [x] `cargo nextest run --workspace --all-features` green (123/123)
- [x] No file > 2000 lines (largest: nutation_truncated_table.rs, 743)
- [x] `cargo fmt --all` applied; no unsafe / no unwrap / no expect anywhere

# Phase 4 — Linked Open Data & Language Bindings (0.1.1, released 2026-07-11)

Shipped in the 0.1.1 release; see `CHANGELOG.md`'s `[0.1.1]` entry for the
full, source-verified description this section summarizes. Workspace gates
2026-07-11: `cargo nextest run --workspace --all-features` 490/490 green,
0 skipped (see the root README's Crates table for the per-crate
breakdown); no unsafe/unwrap/expect anywhere (workspace lints unchanged
and still enforced on the two new `cdylib` binding crates).

## RDF/SKOS/PROV-O vocabulary — `oxiephemeris-rdf` (84 tests)
- [x] A custom `oxa:` astrology ontology plus `oxc:`/`oxs:` SKOS concept
      schemes, generated from the engine's own enums and dignity tables so
      the vocabulary cannot drift from the code that computes charts
- [x] PROV-O provenance (`add_provenance`)
- [x] Canonical, byte-stable Turtle/N-Triples serialization
      (`to_turtle_string`, `to_ntriples_string`, `write_turtle`,
      `write_ntriples`) — triples sorted so re-emitting a chart never
      produces a spurious diff
- [x] Bilingual English/Japanese `rdfs:label`s for every astrological
      concept (`labels` module)
- [x] Wikidata QID cross-references for the ten planets and twelve signs
      (`vocab::PLANET_WIKIDATA`, `vocab::SIGN_WIKIDATA`)
- [x] Graph-closure tests (`tests/closure.rs`): every emitted `oxa:`
      predicate/type is asserted to be declared in `ontology_graph()`, so
      an undeclared term can never reach a published document
- [x] Skolemized IRIs (not blank nodes) for every placement/cusp/angle/
      node/lot/aspect/dignity assessment, so each is independently
      linkable/queryable on its own

## SPARQL store & HTTP endpoint — `oxiephemeris-lod` (14 tests)
- [x] `FileBackedStore`: an in-memory `oxigraph` store (`default-features
      = false`, so no RocksDB) with pure-Rust whole-file N-Quads
      persistence (atomic tmp-file + rename, no partial writes on crash)
- [x] `endpoint::handle`: the SPARQL 1.1 Protocol handler implemented as a
      pure function `(&FileBackedStore, read_only, &mut Request) ->
      Response` — testable without binding a socket
- [x] `oxieph-sparqld` binary (`--bind`, `--store`, `--read-only`,
      `--load`, `--vocab`) serving the published `oxa:`/`oxc:`/`oxs:`
      IRIs as dereferenceable Turtle, with content negotiation
      (SPARQL-Results JSON/XML/CSV/TSV for SELECT/ASK; Turtle/N-Triples
      for CONSTRUCT/DESCRIBE)

## Chart-computation facade — `oxiephemeris-chart` (27 tests)
- [x] Five compute entry points: `natal_chart`, `synastry`, `transit`,
      `progression`, `composite`
- [x] JSON/RDF serializers for all five: `chart_to_json`/`chart_to_rdf`,
      `comparison_to_json`/`comparison_to_rdf`,
      `composite_to_json`/`composite_to_rdf`
- [x] The single UI-free compute path now shared bit-identically by the
      CLI, the Python bindings, and the WASM bindings
- [x] Own dependency-free ISO 8601 parser (`iso8601::parse`, no `chrono`)
      and a stable JSON schema (`json::ChartJson` and friends)

## Python bindings — `oxiephemeris-py` (PyO3, `abi3-py38`)
- [x] An `oxiephemeris` module exposing `julday`, `revjul`, `natal`,
      `natal_rdf`, `synastry`, `synastry_rdf`, `transit`, `progress`,
      `composite`, `composite_rdf` — DE ephemeris bytes in, a Python
      `dict` (JSON view) or `str` (RDF) out
- [x] No filesystem access from the extension itself: the caller supplies
      the DE bytes

## WebAssembly bindings — `oxiephemeris-wasm` (wasm-bindgen)
- [x] `natal_json`/`natal_turtle`, `synastry_json`/`synastry_turtle`,
      `transit_json`, `progress_json`, `composite_json`/`composite_turtle`
      — full natal, synastry, transit, progression, and composite charts,
      including RDF (Turtle) output, computed entirely offline in the
      browser from a DE ephemeris buffer
- [x] Request/response schemas identical to the Python binding and the
      CLI's JSON output, so results are bit-for-bit the same across all
      three front ends
- [x] Self-contained demo page (`crates/oxiephemeris-wasm/www/index.html`)

## Astrology-layer extensions — `oxiephemeris-astro` (grew 64 -> 108 tests)
- [x] `ayanamsha` module: `Ayanamsha` (`FaganBradley`, `Lahiri`,
      `Krishnamurti`, `Raman`, `J2000Zero`, `Custom`), `ayanamsha_rad`,
      `sidereal_from_tropical` — propagated from cited anchor epochs via
      the IAU 2006 (P03) general-precession polynomial (still 5 named
      ayanamshas; see the root README's "Deferred" note)
- [x] `dignities` module: essential dignities (domicile/exaltation/
      triplicity/term/face rulers, detriment/fall/peregrine via
      `EssentialDignity`/`essential_dignity`), a configurable
      `RulershipScheme`, and element/modality `Distribution`
- [x] `declination` module: ecliptic->equatorial `declination`,
      `antiscia`/`contra_antiscia`, parallel/contraparallel/
      out-of-bounds `DeclinationAspect`, cross-checked against an
      independent rotation-based oracle (`tests/declination_oracle.rs`)
- [x] `parts` module: day/night `Sect`, generic `lot`, `part_of_fortune`,
      `part_of_spirit` (Arabic Parts/Lots)
- [x] `midpoints` module: near/far ecliptic midpoints, plus a batch
      helper for composite-chart construction
- [x] `motion` module: `MotionState` (direct/retrograde/stationary)
      classification from a daily longitude speed
- [x] `synastry` module: `CrossHit`, `for_each_cross_aspect`,
      `cross_aspects_into` — cross-chart aspects feeding synastry,
      transit, and progression comparisons
- [x] `zodiac` module: `Element`, `Modality`, `Sign`, `SignPosition`
      (longitude -> sign/degree/DMS decomposition)
- [x] `HouseSystem::ALL`/`HouseSystem::name()` for stable enumeration and
      CLI-matching names

## CLI: new subcommands & unified RDF output — `oxiephemeris-cli` (grew 64 -> 79 tests)
- [x] `oxieph synastry` — cross-aspects between two natal charts
- [x] `oxieph transit` — transiting-body aspects to a natal chart
- [x] `oxieph progress` — secondary-progressed ("day-for-a-year") chart
- [x] `oxieph composite` — midpoint composite chart
- [x] `oxieph vocab` — emit the `oxa:` ontology and SKOS concept schemes
      as RDF (`--part all|ontology|concepts`)
- [x] Unified `--format text|json|turtle|ntriples` flag
      (`rdf_args::OutputFormat`) on every chart-family command; legacy
      `--json` still honored when `--format` is left at its default
- [x] `chart`/`houses` refactored to thin renderers over
      `oxiephemeris_chart::natal_chart`/`compute_houses`; epoch
      resolution moved into `oxiephemeris_chart::epoch`;
      `CliError::ChartGeometry`/`CliError::Node` replaced by
      `CliError::Chart`/`CliError::Rdf` wrapping the facade's own error
      types, so the CLI/Python/WASM front ends share one compute path
      end to end
- [x] RDF emission in the comparison commands consolidated behind an
      internal `RdfOutput` struct (`base_iri`, `format`, `de_number`),
      replacing three separate parameters previously threaded through
      `emit_comparison_rdf`/`emit_composite_rdf`
- [x] Verified end-to-end against a real DE440 file this session: all
      five new subcommands (`synastry`/`transit`/`progress`/`composite`/
      `vocab`) built and run successfully with real output

## Ephemeris provenance — `oxiephemeris-de`
- [x] `DeFile::de_number()` exposes the DE header's `NUMDE` field (e.g.
      `440`, `441`) for recording ephemeris identity in RDF provenance
      output

## Other workspace changes
- [x] MSRV raised 1.81 -> 1.89 (required by the RDF/SPARQL dependency
      stack: `oxigraph`/`oxrdf`/`oxttl` declare `rust-version = "1.87"`
      and `edition = "2024"`)
- [x] Reference chart (tests, examples, READMEs) switched from an
      arbitrary birth date to the Unix epoch (1970-01-01T00:00:00Z) at
      the Royal Observatory, Greenwich, across the workspace

## Deferred / known gaps (as of 0.1.1, checked 2026-07-11)
- [ ] `oxiephemeris-lod` security: transitive `quick-xml 0.37.5` RUSTSEC
      advisories (RUSTSEC-2026-0194, RUSTSEC-2026-0195 — both HIGH
      severity, denial-of-service via crafted XML) pulled in via
      `oxigraph 0.5.9`; no `oxigraph` release fixes this yet, and
      force-patching `quick-xml` to >=0.41 fails to compile against
      `oxrdfxml`/`sparesults`'s pinned API (verified). Mitigation until
      resolved upstream: do not expose `oxieph-sparqld` to untrusted
      networks, do not load untrusted RDF/XML via `--load` (see
      `CHANGELOG.md` and `crates/oxiephemeris-lod/README.md`). Tracked
      for a future patch release.
- NOTE: `grep -rn "TODO\|FIXME" crates/oxiephemeris-{rdf,lod,chart,py,wasm}/src`
      is clean (checked 2026-07-11) — no in-source stub markers in any of
      the five new crates.
- The root README's "Deferred to a later release" section carries the
      ephemeris/astrology-layer gaps forward unchanged from 0.1.0
      (additional house systems beyond the 7 shipped, additional
      ayanamshas beyond the 5 shipped, small-body/asteroid ephemerides, a
      Moshier-analytic-series fallback, SPK type coverage beyond types
      2/3/13) — none of this was in scope for the 0.1.1 LOD/bindings work.
