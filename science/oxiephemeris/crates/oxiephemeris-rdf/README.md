# oxiephemeris-rdf

The Linked Open Data layer: turns computed chart data into RDF graphs
against a custom `oxa:`/`oxc:`/`oxs:` astrology vocabulary, with SKOS
concept schemes and PROV-O provenance.

This crate sits between the astrology engine and everything that talks
RDF: [`oxiephemeris-chart`](../oxiephemeris-chart) calls it to render
`chart_to_rdf`/`comparison_to_rdf`/`composite_to_rdf` for the CLI, Python,
and WASM bindings; [`oxiephemeris-lod`](../oxiephemeris-lod) loads its
`ontology_graph()`/`concept_scheme_graph()` into a SPARQL store and serves
them as dereferenceable Turtle. It re-exports `oxrdf` (`Graph`,
`NamedNode`, …) so downstream crates can build on the same graph types
without risking a version skew against their own `oxrdf` dependency.

The chart data model (`model::ChartResource`, `model::ChartComparison`,
`model::CompositeChartResource`) is a plain, neutral set of structs
decoupled from `oxiephemeris-astro`'s own types — a caller assembles one
once and can render it as text, JSON, or RDF without the three drifting
apart. Every placement, cusp, angle, node, lot, aspect, and dignity
assessment gets its own IRI (skolemized under the chart's own IRI) rather
than a blank node, so it can be linked to, annotated, or queried on its
own from outside the document. Serialization is deterministic: every
writer sorts triples into a canonical order first, so the same chart
always produces the same bytes.

```rust
use oxiephemeris_astro::dignities::Planet;
use oxiephemeris_astro::houses::HouseSystem;
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::parts::Sect;
use oxiephemeris_astro::zodiac::Sign;
use oxiephemeris_rdf::model::{
    BodyPlacement, ChartKind, ChartResource, DistributionRecord, Observer, Provenance, Zodiac,
};
use oxiephemeris_rdf::{chart_graph, to_turtle_string, ChartKey, IriMinter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A caller normally gets a full ChartResource from
    // `oxiephemeris_chart::natal_chart`; built by hand here for illustration.
    let chart = ChartResource {
        kind: ChartKind::Natal,
        epoch_utc: "1970-01-01T00:00:00Z".to_owned(),
        jd_tt: 2_440_587.500_465_196,
        observer: Some(Observer { lat_deg: 51.4779, lon_deg: 0.0, alt_m: 0.0 }),
        house_system: HouseSystem::Placidus,
        zodiac: Zodiac::Tropical,
        rulership: "traditional".to_owned(),
        sect: Sect::Diurnal,
        bodies: vec![BodyPlacement {
            name: "Sun".to_owned(),
            planet: Some(Planet::Sun),
            lon_deg: 235.509_050_876,
            lat_deg: 0.000_090_525,
            lon_speed_deg_per_day: 1.008_8,
            lat_speed_deg_per_day: 0.0,
            distance_au: 0.988_515_204,
            light_time_days: 0.005_7,
            sign: Sign::Scorpio,
            degrees_in_sign: 25.509_050_876,
            house: 7,
            declination_deg: -19.140_387,
            motion: MotionState::Direct,
            out_of_bounds: false,
        }],
        cusps: vec![],
        angles: vec![],
        nodes: vec![],
        lots: vec![],
        aspects: vec![],
        dignities: vec![],
        distribution: DistributionRecord { elements: [0, 0, 0, 1], modalities: [0, 1, 0] },
        elapsed_years: None,
        provenance: Provenance {
            software_name: "oxiephemeris".to_owned(),
            software_version: "0.1.1".to_owned(),
            ephemeris_label: "DE440".to_owned(),
        },
    };

    let minter = IriMinter::default();
    let key = ChartKey {
        kind: "natal",
        jd_tt: chart.jd_tt,
        lat_deg: 51.4779,
        lon_deg: 0.0,
        house_system: "placidus",
        zodiac: "tropical",
    };
    let chart_iri = minter.chart(&key);
    let graph = chart_graph(&chart, &chart_iri, &minter);
    println!("{}", to_turtle_string(&graph)?);
    Ok(())
}
```

`comparison_graph`/`composite_graph` build the equivalent graphs for
synastry/transit/progression comparisons and midpoint composites. The
vocabulary itself needs no chart data at all — merge the ontology and the
SKOS concept schemes into one document exactly as `oxieph vocab` does:

```rust
use oxiephemeris_rdf::oxrdf::Graph;
use oxiephemeris_rdf::{concept_scheme_graph, ontology_graph, to_turtle_string, RdfError};

fn vocabulary_turtle() -> Result<String, RdfError> {
    let mut merged: Graph = ontology_graph(); // oxa: classes/properties (RDFS/OWL + QUDT units)
    for triple in &concept_scheme_graph() {   // oxc:/oxs: SKOS schemes (signs, planets, aspects, ...)
        merged.insert(triple);
    }
    to_turtle_string(&merged)
}
```

`labels` supplies bilingual English/Japanese `rdfs:label`s for every
concept, and `vocab::PLANET_WIKIDATA`/`vocab::SIGN_WIKIDATA` give Wikidata
QID cross-references for the ten planets and twelve signs.

## Notable design points

- **Skolemized n-ary relations, not blank nodes.** "Sun in Scorpio at
  25°30′ in house 7, retrograde" is a multi-place relation, so it gets its
  own referenceable IRI (`<chart>/position/Sun`) rather than being folded
  into an unaddressable blank node.
- **The ontology is closure-checked.** `tests/closure.rs` walks every
  triple this crate can emit and asserts each `oxa:` predicate/type was
  actually declared in `ontology_graph()` — an undeclared term can never
  reach a published document.
- **No `prov:generatedAtTime`.** A chart is a pure function of its inputs;
  its provenance graph is too, so the same chart always serializes to the
  same bytes rather than picking up a wall-clock timestamp on every run.
- **Dignity assessments are qualified by rulership scheme**
  (`dignity/{scheme}/{planet}`), so a traditional and a modern assessment
  of the same chart can coexist instead of one overwriting the other.

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview. Licensed under Apache-2.0 (see workspace
[LICENSE](../../LICENSE)).
