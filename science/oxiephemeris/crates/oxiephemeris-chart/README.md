# oxiephemeris-chart

The UI-free chart-computation facade: turns a birth (or comparison)
request plus DE ephemeris bytes into a fully computed chart, and renders
it to stable JSON or RDF.

This crate is the single source of truth shared, bit-identically, by
[`oxiephemeris-cli`](../oxiephemeris-cli), the Python binding
(`oxiephemeris-py`), and the WebAssembly binding
(`oxiephemeris-wasm`) — a natal chart computed through any of the three
front ends produces the exact same numbers. It sits above
[`oxiephemeris-core`](../oxiephemeris-core),
[`oxiephemeris-de`](../oxiephemeris-de),
[`oxiephemeris-bodies`](../oxiephemeris-bodies), and
[`oxiephemeris-astro`](../oxiephemeris-astro) (computing every derived
quantity — sign, house, retrograde, declination, essential dignity, Arabic
Part, element/modality distribution — exactly once) and hands the result
to [`oxiephemeris-rdf`](../oxiephemeris-rdf) (`model::ChartResource` /
`ChartComparison` / `CompositeChartResource`) for serialization. It also
owns its own dependency-free ISO 8601 parser (`iso8601::parse`, no
`chrono`) and a stable JSON schema (`json::ChartJson` and friends).

Five compute entry points cover every chart this workspace supports:

- `natal_chart(de, &NatalRequest) -> ChartResource` — a full natal chart:
  bodies, houses, angles, nodes, Lots, dignities, and the aspect table.
- `synastry(de, &a, &b, system, dut1_s)` — cross-aspects between two
  natal charts.
- `transit(de, &natal, transit_date, cal, system, dut1_s)` — aspects from
  transiting bodies at a given instant to a natal chart.
- `progression(de, &natal, target_date, cal, system, dut1_s)` —
  secondary-progressed ("a day for a year") chart and its aspects back to
  the natal chart.
- `composite(de, &a, &b, system, dut1_s, base_iri)` — a midpoint composite
  of two natal charts.

The last four take `PersonInput { date, cal, lat_deg, lon_deg }` and
return a `ChartComparison`/`CompositeChartResource`; `chart_to_json`/
`chart_to_rdf`, `comparison_to_json`/`comparison_to_rdf`, and
`composite_to_json`/`composite_to_rdf` render any of them to a `String`.

```rust
use oxiephemeris_chart::{chart_to_json, chart_to_rdf, natal_chart, NatalSpec, RdfFormat};
use oxiephemeris_de::DeFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("data/de440/linux_p1550p2650.440")?;
    let de = DeFile::parse(&bytes)?;

    // The Unix epoch at the Royal Observatory, Greenwich. `NatalSpec`'s
    // fields are all strings/primitives, ready for JSON/kwargs
    // deserialization from the CLI, Python, or WASM front ends;
    // `into_request` parses and validates them once, in one place.
    let spec = NatalSpec {
        date: "1970-01-01T00:00:00Z".to_owned(),
        lat: 51.4779,
        lon: 0.0,
        alt: 0.0,
        dut1: 0.0,
        cal: "gregorian".to_owned(),
        system: "placidus".to_owned(),
        sidereal: None,
        rulership: "traditional".to_owned(),
    };
    let request = spec.into_request()?;
    let chart = natal_chart(&de, &request)?;

    if let Some(sun) = chart.bodies.iter().find(|b| b.name == "Sun") {
        // Sun at 10 deg 09' Capricorn, house 4.
        println!("Sun: {:.3} deg, house {}", sun.lon_deg, sun.house);
    }

    let json = chart_to_json(&chart)?; // the stable JSON schema
    let turtle = chart_to_rdf(&chart, None, None, RdfFormat::Turtle)?; // RDF, default base IRI
    println!("{json}");
    println!("{turtle}");
    Ok(())
}
```

Bodies, angles, cusps, nodes, and Lots all report the **displayed**
longitude — tropical, or ayanamsha-shifted when `sidereal` names a mode,
so a sidereal chart's Sun really is reported in its sidereal sign.
Essential dignities are always scored from the tropical longitude
(a tropical technique regardless of the chart's display zodiac), and
aspects are computed from tropical longitudes/speeds (a common ayanamsha
cancels out of any pairwise separation anyway).

See the [main OxiEphemeris README](../../README.md) for the full workspace
overview. Licensed under Apache-2.0 (see workspace
[LICENSE](../../LICENSE)).
