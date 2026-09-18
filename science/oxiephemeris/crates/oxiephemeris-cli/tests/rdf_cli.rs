//! Integration tests for the RDF surface of the `oxieph` binary.
//!
//! # Why these spawn separate processes
//!
//! `oxiephemeris-rdf` claims its RDF output is **byte-identical across
//! runs**. Asserting that inside one process is nearly worthless: a
//! hash-ordering hazard would produce the same order twice within a single
//! `RandomState`. The determinism tests below therefore run the compiled
//! binary *twice, as two separate processes*, and compare the bytes. That
//! is the only way to catch an ordering that depends on per-process state.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_oxieph"))
}

/// Workspace-relative DE440 fixture, resolved from `CARGO_MANIFEST_DIR`
/// (never a hardcoded absolute path).
fn de440_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440")
}

fn run(args: &[&str]) -> (bool, String) {
    let output = match Command::new(bin()).args(args).output() {
        Ok(o) => o,
        Err(e) => panic!("failed to spawn oxieph: {e}"),
    };
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

#[test]
fn vocab_turtle_is_byte_identical_across_processes() {
    let (ok_a, a) = run(&["vocab", "--format", "turtle"]);
    let (ok_b, b) = run(&["vocab", "--format", "turtle"]);
    assert!(ok_a && ok_b, "vocab must succeed");
    assert!(!a.is_empty(), "vocab must emit something");
    assert_eq!(a, b, "Turtle output must not depend on per-process state");
}

#[test]
fn vocab_ntriples_is_byte_identical_across_processes() {
    let (ok_a, a) = run(&["vocab", "--format", "ntriples"]);
    let (ok_b, b) = run(&["vocab", "--format", "ntriples"]);
    assert!(ok_a && ok_b, "vocab must succeed");
    assert!(!a.is_empty(), "vocab must emit something");
    assert_eq!(a, b, "N-Triples output must be deterministic");
}

/// `--format ntriples` is the spelling users type; the hyphenated and
/// short forms must keep working too.
#[test]
fn ntriples_format_aliases_are_accepted() {
    for spelling in ["ntriples", "n-triples", "nt"] {
        let (ok, out) = run(&["vocab", "--format", spelling]);
        assert!(ok, "--format {spelling} must be accepted");
        assert!(!out.is_empty(), "--format {spelling} must emit triples");
    }
}

#[test]
fn vocab_rejects_non_rdf_formats() {
    let (ok, _) = run(&["vocab", "--format", "json"]);
    assert!(!ok, "vocab has no JSON rendering and must fail loudly");
}

#[test]
fn vocab_parts_partition_the_document() {
    let (ok_all, all) = run(&["vocab", "--part", "all", "--format", "ntriples"]);
    let (ok_o, ontology) = run(&["vocab", "--part", "ontology", "--format", "ntriples"]);
    let (ok_c, concepts) = run(&["vocab", "--part", "concepts", "--format", "ntriples"]);
    assert!(ok_all && ok_o && ok_c, "every --part must succeed");
    let count = |s: &str| s.lines().filter(|l| !l.trim().is_empty()).count();
    // Guard against the whole assertion passing vacuously on empty output.
    assert!(count(&ontology) > 0, "ontology must have triples");
    assert!(count(&concepts) > 0, "concepts must have triples");
    assert_eq!(
        count(&all),
        count(&ontology) + count(&concepts),
        "the two parts must exactly partition the whole vocabulary"
    );
}

#[test]
fn chart_turtle_is_deterministic_and_carries_the_expected_facts() {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!(
            "note: skipping (DE440 fixture not present at {}); run scripts/fetch_de440.sh",
            de_path.display()
        );
        return;
    }
    let de = de_path.to_string_lossy().into_owned();
    let args = [
        "chart",
        "1970-01-01T00:00:00Z",
        "--lat",
        "51.4779",
        "--lon",
        "0.0",
        "--de",
        &de,
        "--format",
        "turtle",
    ];
    let (ok_a, a) = run(&args);
    let (ok_b, b) = run(&args);
    assert!(ok_a && ok_b, "chart --format turtle must succeed");
    assert_eq!(a, b, "chart Turtle must be byte-identical across processes");

    // The chart's own identity, and the facts verified elsewhere in this
    // workspace against the reference implementation.
    assert!(a.contains("oxa:NatalChart"), "chart class missing");
    assert!(
        a.contains("oxa:inSign sign:Capricorn"),
        "Sun's sign missing"
    );
    assert!(a.contains("oxa:houseNumber 4"), "Sun's house missing");
    assert!(a.contains("oxa:dignityScore 3"), "Mars' dignity missing");
    assert!(
        a.contains("prov:wasDerivedFrom") && a.contains("ephemeris/DE440"),
        "provenance must name the ephemeris it used"
    );
    // No wall-clock timestamp: that is what makes the two runs identical.
    assert!(
        !a.contains("generatedAtTime"),
        "a generation timestamp would break determinism"
    );
}

/// Extracts every `.../chart/<32 hex>` IRI mentioned in an N-Triples doc.
fn chart_iris(ntriples: &str) -> Vec<String> {
    let mut found: Vec<String> = ntriples
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '/' || c == ':' || c == '.'))
        .filter_map(|token| {
            let (_, tail) = token.split_once("/chart/")?;
            let hex: String = tail.chars().take_while(char::is_ascii_hexdigit).collect();
            (hex.len() == 32).then_some(hex)
        })
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// **The load-bearing property of the LOD layer.** A natal chart emitted on
/// its own and the same chart appearing as one side of a synastry, a
/// transit, or a composite must be the *same resource*, or the documents
/// will never join in a triple store.
#[test]
fn a_chart_keeps_one_identity_across_every_document_that_mentions_it() {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return;
    }
    let de = de_path.to_string_lossy().into_owned();
    let (_, standalone) = run(&[
        "chart",
        "1970-01-01T00:00:00Z",
        "--lat",
        "51.4779",
        "--lon",
        "0.0",
        "--de",
        &de,
        "--format",
        "ntriples",
    ]);
    let natal_iris = chart_iris(&standalone);
    assert_eq!(
        natal_iris.len(),
        1,
        "a chart document names exactly one chart"
    );
    let natal = &natal_iris[0];

    let (ok, synastry) = run(&[
        "synastry",
        "--date-a",
        "1970-01-01T00:00:00Z",
        "--lat-a",
        "51.4779",
        "--lon-a",
        "0.0",
        "--date-b",
        "2000-01-01T12:00:00Z",
        "--lat-b",
        "48.8566",
        "--lon-b",
        "2.3522",
        "--de",
        &de,
        "--format",
        "ntriples",
    ]);
    assert!(ok, "synastry --format ntriples must succeed");
    assert!(
        chart_iris(&synastry).contains(natal),
        "synastry side A must be the same resource as the standalone chart"
    );

    let (ok, transit) = run(&[
        "transit",
        "1970-01-01T00:00:00Z",
        "--lat",
        "51.4779",
        "--lon",
        "0.0",
        "--transit",
        "2026-07-10T00:00:00",
        "--de",
        &de,
        "--format",
        "ntriples",
    ]);
    assert!(ok, "transit --format ntriples must succeed");
    assert!(
        chart_iris(&transit).contains(natal),
        "the transit's natal side must be the same resource"
    );

    let (ok, composite) = run(&[
        "composite",
        "--date-a",
        "1970-01-01T00:00:00Z",
        "--lat-a",
        "51.4779",
        "--lon-a",
        "0.0",
        "--date-b",
        "2000-01-01T12:00:00Z",
        "--lat-b",
        "48.8566",
        "--lon-b",
        "2.3522",
        "--de",
        &de,
        "--format",
        "ntriples",
    ]);
    assert!(ok, "composite --format ntriples must succeed");
    assert!(
        chart_iris(&composite).contains(natal),
        "the composite must be derived from the very same source chart"
    );
    assert!(
        composite.contains("wasDerivedFrom"),
        "a composite must record what it was derived from"
    );
}

/// The rulership scheme is an interpretive lens, not part of a chart's
/// identity — but the two assessments must not overwrite each other.
#[test]
fn rulership_moves_the_dignity_assessment_not_the_chart() {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return;
    }
    let de = de_path.to_string_lossy().into_owned();
    let emit = |scheme: &str| {
        run(&[
            "chart",
            "1970-01-01T00:00:00Z",
            "--lat",
            "51.4779",
            "--lon",
            "0.0",
            "--de",
            &de,
            "--rulership",
            scheme,
            "--format",
            "ntriples",
        ])
        .1
    };
    let traditional = emit("traditional");
    let modern = emit("modern");
    assert_eq!(
        chart_iris(&traditional),
        chart_iris(&modern),
        "the rulership scheme must not change which chart this is"
    );
    assert!(traditional.contains("/dignity/traditional/Mars"));
    assert!(modern.contains("/dignity/modern/Mars"));
    assert!(
        !traditional.contains("/dignity/Mars>"),
        "an unqualified dignity IRI would collide across schemes"
    );
}

#[test]
fn comparison_documents_are_deterministic() {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return;
    }
    let de = de_path.to_string_lossy().into_owned();
    let args = [
        "progress",
        "1970-01-01T00:00:00Z",
        "--lat",
        "51.4779",
        "--lon",
        "0.0",
        "--target",
        "2010-01-01T00:00:00",
        "--de",
        &de,
        "--format",
        "turtle",
    ];
    let (ok_a, a) = run(&args);
    let (_, b) = run(&args);
    assert!(
        ok_a && !a.is_empty(),
        "progress --format turtle must succeed"
    );
    assert_eq!(a, b, "a progression document must be byte-identical");
    assert!(a.contains("oxa:ProgressionComparison"));
    assert!(a.contains("oxa:elapsedYears"));
}

/// The chart IRI is a pure function of the chart's defining inputs, so
/// changing an input must move the IRI, and `--chart-iri` must override it.
#[test]
fn chart_iri_is_derived_from_the_chart_and_overridable() {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return;
    }
    let de = de_path.to_string_lossy().into_owned();
    let base = [
        "chart",
        "1970-01-01T00:00:00Z",
        "--lat",
        "51.4779",
        "--lon",
        "0.0",
        "--de",
        &de,
        "--format",
        "ntriples",
    ];
    let (_, tropical) = run(&base);

    let mut koch = base.to_vec();
    koch.extend_from_slice(&["--system", "koch"]);
    let (_, koch_out) = run(&koch);
    assert_ne!(
        tropical, koch_out,
        "a different house system is a different chart"
    );

    let mut explicit = base.to_vec();
    explicit.extend_from_slice(&["--chart-iri", "https://example.org/my/chart"]);
    let (ok, out) = run(&explicit);
    assert!(ok, "--chart-iri must be accepted");
    assert!(
        out.contains("<https://example.org/my/chart>"),
        "explicit chart IRI must be used verbatim"
    );

    let mut bad = base.to_vec();
    bad.extend_from_slice(&["--base-iri", "not an iri"]);
    let (ok, _) = run(&bad);
    assert!(!ok, "an invalid --base-iri must fail loudly");
}
