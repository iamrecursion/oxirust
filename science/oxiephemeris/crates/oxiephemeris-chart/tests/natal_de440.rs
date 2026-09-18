//! End-to-end facade test against DE440 for the synthetic reference
//! chart used throughout this workspace: the Unix epoch
//! (1970-01-01T00:00:00Z) at the Royal Observatory, Greenwich. Skips if
//! the DE440 fixture is absent.

use std::path::PathBuf;

use oxiephemeris_chart::calendar::CalendarKind;
use oxiephemeris_chart::{
    chart_to_json, chart_to_rdf, comparison_to_rdf, composite, composite_to_rdf, natal_chart,
    synastry, NatalSpec, PersonInput, RdfFormat,
};
use oxiephemeris_de::DeFile;

type TestResult = Result<(), String>;

fn de440_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440")
}

fn load_de() -> Option<Vec<u8>> {
    let path = de440_path();
    if !path.exists() {
        eprintln!(
            "note: skipping (DE440 fixture not present at {})",
            path.display()
        );
        return None;
    }
    std::fs::read(&path).ok()
}

fn person_a() -> PersonInput {
    PersonInput {
        date: "1970-01-01T00:00:00Z".to_owned(),
        cal: CalendarKind::Gregorian,
        lat_deg: 51.4779,
        lon_deg: 0.0,
    }
}

fn person_b() -> PersonInput {
    PersonInput {
        date: "2000-01-01T12:00:00Z".to_owned(),
        cal: CalendarKind::Gregorian,
        lat_deg: 48.8566,
        lon_deg: 2.3522,
    }
}

#[test]
fn natal_matches_the_reference_chart() -> TestResult {
    let Some(bytes) = load_de() else {
        return Ok(());
    };
    let Ok(de) = DeFile::parse(&bytes) else {
        return Err("DE440 must parse".to_owned());
    };
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
    let Ok(request) = spec.into_request() else {
        return Err("spec must resolve".to_owned());
    };
    let Ok(chart) = natal_chart(&de, &request) else {
        return Err("natal_chart must succeed".to_owned());
    };

    // Sun at 10 deg 09' Capricorn, in house 4.
    let Some(sun) = chart.bodies.iter().find(|b| b.name == "Sun") else {
        return Err("Sun must be present".to_owned());
    };
    if sun.sign.name() != "Capricorn" {
        return Err(format!("Sun sign {} != Capricorn", sun.sign.name()));
    }
    if (sun.degrees_in_sign - 10.156).abs() > 0.01 {
        return Err(format!(
            "Sun deg-in-sign {} != ~10.156",
            sun.degrees_in_sign
        ));
    }
    if sun.house != 4 {
        return Err(format!("Sun house {} != 4", sun.house));
    }

    // Mars: night triplicity ruler of a water sign => score +3.
    let Some(mars) = chart.dignities.iter().find(|d| d.planet.name() == "Mars") else {
        return Err("Mars dignity must be present".to_owned());
    };
    if mars.score != 3 {
        return Err(format!("Mars dignity score {} != 3", mars.score));
    }

    // JSON view round-trips and Turtle carries the expected facts.
    let Ok(json) = chart_to_json(&chart) else {
        return Err("chart_to_json must succeed".to_owned());
    };
    if !json.contains("\"Capricorn\"") {
        return Err("JSON must mention Capricorn".to_owned());
    }
    let Ok(turtle) = chart_to_rdf(&chart, None, None, RdfFormat::Turtle) else {
        return Err("chart_to_rdf must succeed".to_owned());
    };
    if !turtle.contains("oxa:inSign sign:Capricorn") {
        return Err("Turtle must place a body in Capricorn".to_owned());
    }
    if !turtle.contains("oxa:dignityScore 3") {
        return Err("Turtle must carry Mars' dignity score".to_owned());
    }
    Ok(())
}

#[test]
fn synastry_and_composite_serialize() -> TestResult {
    let Some(bytes) = load_de() else {
        return Ok(());
    };
    let Ok(de) = DeFile::parse(&bytes) else {
        return Err("DE440 must parse".to_owned());
    };
    let system = oxiephemeris_astro::houses::HouseSystem::Placidus;

    let Ok(syn) = synastry(&de, &person_a(), &person_b(), system, 0.0) else {
        return Err("synastry must succeed".to_owned());
    };
    let Ok(turtle) = comparison_to_rdf(&syn, None, RdfFormat::Turtle) else {
        return Err("synastry RDF must succeed".to_owned());
    };
    if !turtle.contains("oxa:SynastryComparison") {
        return Err("synastry Turtle must declare its class".to_owned());
    }

    let Ok(comp) = composite(&de, &person_a(), &person_b(), system, 0.0, None) else {
        return Err("composite must succeed".to_owned());
    };
    let Ok(comp_ttl) = composite_to_rdf(&comp, None, RdfFormat::Turtle) else {
        return Err("composite RDF must succeed".to_owned());
    };
    if !comp_ttl.contains("oxa:CompositeChart") || !comp_ttl.contains("wasDerivedFrom") {
        return Err("composite Turtle must derive from its sources".to_owned());
    }
    Ok(())
}
