//! Closure checks over the emitted graphs.
//!
//! The ontology is only useful if it actually *covers* the data. These
//! tests walk every triple this crate can emit and assert that each
//! `oxa:` term appearing as a predicate or as an `rdf:type` object was
//! declared in [`ontology_graph`]. An undeclared term is a term nobody
//! outside this repository can interpret; catching it here means it can
//! never reach a published document.

use oxiephemeris_rdf::model::{
    AngleKind, AngleRecord, AspectRecord, BodyPlacement, ChartComparison, ChartKind,
    ChartPointRecord, ChartResource, ChartSide, ComparisonKind, CompositeChartResource,
    CrossAspectRecord, CuspRecord, DignityRecord, DignityTier, DistributionRecord, LotKind,
    LotRecord, NodeKind, NodeRecord, Observer, Provenance, Zodiac,
};
use oxiephemeris_rdf::oxrdf::{Graph, NamedOrBlankNodeRef, TermRef};
use oxiephemeris_rdf::vocab::ns;
use oxiephemeris_rdf::{
    chart_graph, comparison_graph, composite_graph, concept_scheme_graph, ontology, ontology_graph,
    ChartKey, ComparisonKey, IriMinter,
};

use oxiephemeris_astro::aspects::AspectKind;
use oxiephemeris_astro::dignities::Planet;
use oxiephemeris_astro::houses::HouseSystem;
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::parts::Sect;
use oxiephemeris_astro::zodiac::Sign;

type TestResult = Result<(), String>;

/// A synthetic sample chart (the Unix epoch at Greenwich) exercising
/// every builder branch. The body positions are illustrative fixtures,
/// not an astronomically resolved chart for the epoch.
fn sample_chart() -> ChartResource {
    ChartResource {
        kind: ChartKind::Natal,
        epoch_utc: "1970-01-01T00:00:00.000000Z".to_owned(),
        jd_tt: 2_440_587.500_465_196,
        observer: Some(Observer {
            lat_deg: 51.4779,
            lon_deg: 0.0,
            alt_m: 0.0,
        }),
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
        cusps: vec![CuspRecord {
            number: 1,
            lon_deg: 29.987_602_661,
        }],
        angles: vec![AngleRecord {
            kind: AngleKind::Ascendant,
            lon_deg: 29.987_602_661,
        }],
        nodes: vec![NodeRecord {
            kind: NodeKind::MeanNode,
            lon_deg: 250.913_166_919,
        }],
        lots: vec![LotRecord {
            kind: LotKind::Fortune,
            lon_deg: 79.494_102_64,
        }],
        aspects: vec![AspectRecord {
            body1: "Sun".to_owned(),
            body2: "Venus".to_owned(),
            kind: AspectKind::Conjunction,
            exact_angle_deg: 0.0,
            offset_deg: -2.917_8,
            applying: false,
        }],
        dignities: vec![DignityRecord {
            planet: Planet::Mars,
            tiers: vec![DignityTier::Domicile, DignityTier::Triplicity],
            score: 8,
        }],
        distribution: DistributionRecord {
            elements: [1, 1, 2, 6],
            modalities: [4, 4, 2],
        },
        elapsed_years: None,
        provenance: Provenance {
            software_name: "oxiephemeris".to_owned(),
            software_version: "0.1.1".to_owned(),
            ephemeris_label: "DE440".to_owned(),
        },
    }
}

fn sample_key(jd_tt: f64) -> ChartKey<'static> {
    ChartKey {
        kind: "natal",
        jd_tt,
        lat_deg: 51.4779,
        lon_deg: 0.0,
        house_system: "placidus",
        zodiac: "tropical",
    }
}

fn sample_provenance() -> Provenance {
    Provenance {
        software_name: "oxiephemeris".to_owned(),
        software_version: "0.1.1".to_owned(),
        ephemeris_label: "DE440".to_owned(),
    }
}

fn sample_chart_graph() -> Graph {
    let minter = IriMinter::default();
    let iri = minter.chart(&sample_key(2_440_587.500_465_196));
    chart_graph(&sample_chart(), &iri, &minter)
}

/// One point of each shape, so a comparison graph exercises both the
/// planet branch and the angle branch.
fn sample_points() -> Vec<ChartPointRecord> {
    vec![
        ChartPointRecord {
            name: "Sun".to_owned(),
            planet: Some(Planet::Sun),
            angle: None,
            lon_deg: 235.5,
            speed_deg_per_day: 1.0,
            sign: Sign::Scorpio,
            degrees_in_sign: 25.5,
            motion: MotionState::Direct,
        },
        ChartPointRecord {
            name: "ASC".to_owned(),
            planet: None,
            angle: Some(AngleKind::Ascendant),
            lon_deg: 29.98,
            speed_deg_per_day: 0.0,
            sign: Sign::Aries,
            degrees_in_sign: 29.98,
            motion: MotionState::Stationary,
        },
    ]
}

fn sample_side(kind: ChartKind, jd_tt: Option<f64>) -> ChartSide {
    ChartSide {
        kind,
        epoch_utc: jd_tt.map(|_| "1970-01-01T00:00:00.000000Z".to_owned()),
        jd_tt,
        observer: jd_tt.map(|_| Observer {
            lat_deg: 51.4779,
            lon_deg: 0.0,
            alt_m: 0.0,
        }),
        house_system: HouseSystem::Placidus,
        points: sample_points(),
    }
}

fn sample_cross() -> Vec<CrossAspectRecord> {
    vec![CrossAspectRecord {
        from_name: "Sun".to_owned(),
        to_name: "ASC".to_owned(),
        kind: AspectKind::Trine,
        offset_deg: 1.5,
        applying: true,
    }]
}

fn sample_comparison_graph(kind: ComparisonKind) -> Graph {
    let minter = IriMinter::default();
    let a = minter.chart(&sample_key(2_440_587.500_465_196));
    let b = minter.chart(&sample_key(2_451_545.0));
    let comparison = ChartComparison {
        kind,
        chart_a: sample_side(ChartKind::Natal, Some(2_440_587.500_465_196)),
        chart_b: sample_side(ChartKind::Natal, Some(2_451_545.0)),
        cross_aspects: sample_cross(),
        elapsed_years: Some(36.0),
        provenance: sample_provenance(),
    };
    let key = ComparisonKey {
        kind: kind.name(),
        chart_a: a.as_str(),
        chart_b: b.as_str(),
    };
    comparison_graph(&comparison, &minter.comparison(&key), &a, &b, &minter)
}

fn sample_composite_graph() -> Graph {
    let minter = IriMinter::default();
    let a = minter.chart(&sample_key(2_440_587.500_465_196));
    let b = minter.chart(&sample_key(2_451_545.0));
    let composite = CompositeChartResource {
        side: sample_side(ChartKind::Composite, None),
        aspects: sample_cross(),
        source_charts: vec![a.as_str().to_owned(), b.as_str().to_owned()],
        provenance: sample_provenance(),
    };
    let key = ComparisonKey {
        kind: "composite",
        chart_a: a.as_str(),
        chart_b: b.as_str(),
    };
    composite_graph(&composite, &minter.chart_from_slug(&key.slug()), &minter)
}

/// The comparison and composite builders must be as disciplined as the
/// chart builder: no term they emit may be missing from the ontology.
#[test]
fn comparison_graphs_use_only_declared_terms() -> TestResult {
    let declared = ontology::declared_terms();
    let mut graphs = vec![("composite", sample_composite_graph())];
    for kind in ComparisonKind::ALL {
        graphs.push((kind.name(), sample_comparison_graph(kind)));
    }
    for (name, graph) in graphs {
        for term in oxa_terms_used(&graph) {
            if !declared.contains(&term) {
                return Err(format!("{name} graph uses undeclared term {term}"));
            }
        }
    }
    Ok(())
}

/// A comparison's sides must carry the very IRIs a standalone chart gets,
/// or the two documents will never merge. This is the load-bearing
/// property of the whole design.
#[test]
fn comparison_sides_reuse_standalone_chart_iris() -> TestResult {
    let minter = IriMinter::default();
    let a = minter.chart(&sample_key(2_440_587.500_465_196));
    let graph = sample_comparison_graph(ComparisonKind::Synastry);
    let sun = IriMinter::child(&a, "position", "Sun");
    let found = graph
        .iter()
        .any(|t| matches!(t.subject, NamedOrBlankNodeRef::NamedNode(n) if n == sun));
    if !found {
        return Err(format!(
            "synastry side A must describe {sun}, the same resource `chart` emits"
        ));
    }
    Ok(())
}

/// A composite chart has no epoch, and none must be invented for it.
#[test]
fn composite_invents_no_epoch() -> TestResult {
    let graph = sample_composite_graph();
    for triple in &graph {
        if triple.predicate.as_str().ends_with("#atEpoch")
            || triple.predicate.as_str().ends_with("#julianDateTT")
        {
            return Err("a composite is a midpoint construction, not a moment".to_owned());
        }
    }
    Ok(())
}

/// Every `oxa:` IRI used as a predicate or as an `rdf:type` object.
fn oxa_terms_used(graph: &Graph) -> Vec<String> {
    let mut used = Vec::new();
    for triple in graph {
        let predicate = triple.predicate.as_str();
        if predicate.starts_with(ns::OXA) {
            used.push(predicate.to_owned());
        }
        if predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            if let TermRef::NamedNode(class) = triple.object {
                if class.as_str().starts_with(ns::OXA) {
                    used.push(class.as_str().to_owned());
                }
            }
        }
    }
    used.sort_unstable();
    used.dedup();
    used
}

#[test]
fn chart_graph_uses_only_declared_terms() -> TestResult {
    let declared = ontology::declared_terms();
    for term in oxa_terms_used(&sample_chart_graph()) {
        if !declared.contains(&term) {
            return Err(format!("chart graph uses undeclared term {term}"));
        }
    }
    Ok(())
}

#[test]
fn concept_scheme_graph_uses_only_declared_terms() -> TestResult {
    let declared = ontology::declared_terms();
    for term in oxa_terms_used(&concept_scheme_graph()) {
        if !declared.contains(&term) {
            return Err(format!("SKOS graph uses undeclared term {term}"));
        }
    }
    Ok(())
}

/// The two documents the vocabulary ships must not overlap: the ontology
/// describes terms, the concept schemes describe individuals.
#[test]
fn ontology_and_concepts_are_disjoint() -> TestResult {
    let ontology = ontology_graph();
    let concepts = concept_scheme_graph();
    for triple in &ontology {
        if concepts.contains(triple) {
            return Err(format!("triple appears in both documents: {triple}"));
        }
    }
    Ok(())
}

/// Every chart sub-resource must be reachable from the chart itself.
/// Skolemized IRIs are only useful if the chart actually links to them.
#[test]
fn every_chart_subresource_is_linked_from_the_chart() -> TestResult {
    let graph = sample_chart_graph();
    let minter = IriMinter::default();
    let key = ChartKey {
        kind: "natal",
        jd_tt: 2_440_587.500_465_196,
        lat_deg: 51.4779,
        lon_deg: 0.0,
        house_system: "placidus",
        zodiac: "tropical",
    };
    let chart = minter.chart(&key);
    let prefix = format!("{}/", chart.as_str());

    // Collect every subject under the chart's IRI space.
    let mut subresources: Vec<String> = Vec::new();
    for triple in &graph {
        if let NamedOrBlankNodeRef::NamedNode(subject) = triple.subject {
            if subject.as_str().starts_with(&prefix) {
                subresources.push(subject.as_str().to_owned());
            }
        }
    }
    subresources.sort_unstable();
    subresources.dedup();
    if subresources.is_empty() {
        return Err("no sub-resources were emitted".to_owned());
    }

    // Every one must appear as the object of some triple whose subject is
    // the chart (hasBodyPosition, atEpoch, wasGeneratedBy, ...).
    for sub in subresources {
        let linked = graph.iter().any(|triple| {
            triple.subject == NamedOrBlankNodeRef::NamedNode(chart.as_ref())
                && matches!(triple.object, TermRef::NamedNode(n) if n.as_str() == sub)
        });
        if !linked {
            return Err(format!(
                "orphan sub-resource, unreachable from chart: {sub}"
            ));
        }
    }
    Ok(())
}

/// No blank nodes: they cannot be referenced across documents, which is
/// the whole reason this crate skolemizes its n-ary relations.
#[test]
fn no_blank_nodes_are_emitted() -> TestResult {
    for (name, graph) in [
        ("chart", sample_chart_graph()),
        ("concepts", concept_scheme_graph()),
        ("ontology", ontology_graph()),
    ] {
        for triple in &graph {
            if matches!(triple.subject, NamedOrBlankNodeRef::BlankNode(_)) {
                return Err(format!("{name} graph has a blank node subject"));
            }
            if matches!(triple.object, TermRef::BlankNode(_)) {
                return Err(format!("{name} graph has a blank node object"));
            }
        }
    }
    Ok(())
}
