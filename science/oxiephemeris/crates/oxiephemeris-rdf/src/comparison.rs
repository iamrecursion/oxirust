//! Two-chart comparisons (synastry, transit, progression) and midpoint
//! composite charts as RDF.
//!
//! # Why the sides are charts, not copies
//!
//! Each side of a comparison is emitted under **its own chart IRI**, minted
//! from that chart's own defining inputs exactly as `oxieph chart` mints
//! it. So the natal chart on the left of a synastry is *the same resource*
//! as the fully-described chart emitted on its own, and loading both into a
//! triple store merges them: the synastry contributes the cross-aspects,
//! the natal document contributes the houses and dignities. That merge is
//! the entire reason to publish Linked Data instead of two JSON files.
//!
//! RDF is open-world, so a side that mentions only ten planets and two
//! angles asserts nothing about houses *not* existing — it simply does not
//! speak of them.
//!
//! # Cross-aspects are point-to-point
//!
//! A cross-aspect's endpoints are concrete placements
//! (`{chartA}/position/Sun`, `{chartB}/angle/ascendant`), not planet
//! concepts. That is what lets it say *whose* Sun, and lets an Ascendant be
//! an endpoint at all.
//!
//! # Composites are charts, not relations
//!
//! A composite's points are midpoints of two other charts' points, so it is
//! a chart in its own right that `prov:wasDerivedFrom` its two sources —
//! not a [`ChartComparison`].

use oxrdf::vocab::{rdf, rdfs};
use oxrdf::{Graph, NamedNode};

use crate::builder::{add, add_double, boolean, date_time_stamp, plain};
use crate::chart::chart_class;
use crate::iri::IriMinter;
use crate::model::{
    ChartComparison, ChartPointRecord, ChartSide, ComparisonKind, CompositeChartResource,
    CrossAspectRecord,
};
use crate::prov::add_provenance;
use crate::skos::{
    angle_concept, aspect_concept, house_system_concept, motion_concept, planet_concept,
    sign_concept,
};
use crate::vocab::{geo, oxa, prov, time};

/// The `oxa:` class for a comparison kind.
fn comparison_class(kind: ComparisonKind) -> oxrdf::NamedNodeRef<'static> {
    match kind {
        ComparisonKind::Synastry => oxa::SYNASTRY_COMPARISON,
        ComparisonKind::Transit => oxa::TRANSIT_COMPARISON,
        ComparisonKind::Progression => oxa::PROGRESSION_COMPARISON,
    }
}

/// The IRI of one chart point, under its own chart.
///
/// Planets live at `{chart}/position/{Name}` and angles at
/// `{chart}/angle/{kind}` — the very same IRIs `crate::chart` mints, which
/// is what makes the two documents merge.
#[must_use]
pub fn point_iri(chart_iri: &NamedNode, point: &ChartPointRecord) -> NamedNode {
    match point.angle {
        Some(kind) => IriMinter::child(chart_iri, "angle", kind.name()),
        None => IriMinter::child(chart_iri, "position", &point.name),
    }
}

/// Finds a side's point by display name.
fn find_point<'a>(side: &'a ChartSide, name: &str) -> Option<&'a ChartPointRecord> {
    side.points.iter().find(|p| p.name == name)
}

/// Emits one chart point and links it from its chart.
fn add_point(graph: &mut Graph, chart_iri: &NamedNode, point: &ChartPointRecord) {
    let node = point_iri(chart_iri, point);
    if let Some(kind) = point.angle {
        add(graph, chart_iri.clone(), oxa::HAS_ANGLE, node.clone());
        add(graph, node.clone(), rdf::TYPE, oxa::CHART_ANGLE);
        add(graph, node.clone(), oxa::ANGLE_KIND_OF, angle_concept(kind));
    } else {
        add(
            graph,
            chart_iri.clone(),
            oxa::HAS_BODY_POSITION,
            node.clone(),
        );
        add(graph, node.clone(), rdf::TYPE, oxa::BODY_POSITION);
    }
    if let Some(planet) = point.planet {
        add(graph, node.clone(), oxa::BODY, planet_concept(planet));
    }
    add(graph, node.clone(), rdfs::LABEL, plain(&point.name));
    add_double(graph, node.clone(), oxa::ECLIPTIC_LONGITUDE, point.lon_deg);
    add(graph, node.clone(), oxa::IN_SIGN, sign_concept(point.sign));
    add_double(
        graph,
        node.clone(),
        oxa::DEGREES_IN_SIGN,
        point.degrees_in_sign,
    );
    add_double(
        graph,
        node.clone(),
        oxa::LONGITUDE_SPEED,
        point.speed_deg_per_day,
    );
    add(
        graph,
        node.clone(),
        oxa::MOTION,
        motion_concept(point.motion),
    );
    add(
        graph,
        node,
        oxa::IS_RETROGRADE,
        boolean(point.motion.is_retrograde()),
    );
}

/// Emits one side of a comparison as a (partially described) chart.
fn add_side(graph: &mut Graph, chart_iri: &NamedNode, side: &ChartSide) {
    add(graph, chart_iri.clone(), rdf::TYPE, oxa::CHART);
    add(graph, chart_iri.clone(), rdf::TYPE, chart_class(side.kind));
    add(
        graph,
        chart_iri.clone(),
        oxa::HOUSE_SYSTEM_OF,
        house_system_concept(side.house_system),
    );
    if let Some(jd_tt) = side.jd_tt {
        add_double(graph, chart_iri.clone(), oxa::JULIAN_DATE_TT, jd_tt);
    }
    if let Some(epoch_utc) = &side.epoch_utc {
        let epoch = IriMinter::part(chart_iri, "epoch");
        add(graph, chart_iri.clone(), oxa::AT_EPOCH, epoch.clone());
        add(graph, epoch.clone(), rdf::TYPE, time::INSTANT);
        add(
            graph,
            epoch,
            time::IN_XSD_DATE_TIME_STAMP,
            date_time_stamp(epoch_utc),
        );
    }
    if let Some(observer) = side.observer {
        let node = IriMinter::part(chart_iri, "observer");
        add(graph, chart_iri.clone(), oxa::HAS_OBSERVER, node.clone());
        add(graph, node.clone(), rdf::TYPE, oxa::OBSERVER);
        add_double(graph, node.clone(), geo::LAT, observer.lat_deg);
        add_double(graph, node.clone(), geo::LONG, observer.lon_deg);
        add_double(graph, node, geo::ALT, observer.alt_m);
    }
    for point in &side.points {
        add_point(graph, chart_iri, point);
    }
}

/// Emits one aspect between two points, of the given class.
fn add_aspect(
    graph: &mut Graph,
    node: &NamedNode,
    class: oxrdf::NamedNodeRef<'static>,
    from: &NamedNode,
    to: &NamedNode,
    record: &CrossAspectRecord,
) {
    add(graph, node.clone(), rdf::TYPE, class);
    add(graph, node.clone(), oxa::FROM_POINT, from.clone());
    add(graph, node.clone(), oxa::TO_POINT, to.clone());
    add(
        graph,
        node.clone(),
        oxa::ASPECT_KIND_OF,
        aspect_concept(record.kind),
    );
    add_double(graph, node.clone(), oxa::ORB, record.offset_deg);
    add(
        graph,
        node.clone(),
        oxa::IS_APPLYING,
        boolean(record.applying),
    );
}

/// Builds the RDF graph of a two-chart comparison.
///
/// `chart_a_iri`/`chart_b_iri` must be the IRIs minted from each side's own
/// [`crate::ChartKey`], so the sides merge with independently published
/// charts.
#[must_use]
// `chart_a_iri`/`chart_b_iri` differ by exactly the letter that carries the
// meaning; renaming them apart would obscure that they are the two sides.
#[allow(clippy::similar_names)]
pub fn comparison_graph(
    comparison: &ChartComparison,
    comparison_iri: &NamedNode,
    chart_a_iri: &NamedNode,
    chart_b_iri: &NamedNode,
    minter: &IriMinter,
) -> Graph {
    let mut graph = Graph::new();

    add(
        &mut graph,
        comparison_iri.clone(),
        rdf::TYPE,
        oxa::CHART_COMPARISON,
    );
    add(
        &mut graph,
        comparison_iri.clone(),
        rdf::TYPE,
        comparison_class(comparison.kind),
    );
    add(
        &mut graph,
        comparison_iri.clone(),
        oxa::CHART_A,
        chart_a_iri.clone(),
    );
    add(
        &mut graph,
        comparison_iri.clone(),
        oxa::CHART_B,
        chart_b_iri.clone(),
    );
    if let Some(years) = comparison.elapsed_years {
        add_double(
            &mut graph,
            comparison_iri.clone(),
            oxa::ELAPSED_YEARS,
            years,
        );
    }

    add_side(&mut graph, chart_a_iri, &comparison.chart_a);
    add_side(&mut graph, chart_b_iri, &comparison.chart_b);

    for (index, record) in comparison.cross_aspects.iter().enumerate() {
        // A cross-aspect whose endpoint is not among the emitted points
        // would dangle; skip it rather than publish a broken link.
        let (Some(from), Some(to)) = (
            find_point(&comparison.chart_a, &record.from_name),
            find_point(&comparison.chart_b, &record.to_name),
        ) else {
            continue;
        };
        let node = IriMinter::child(comparison_iri, "cross-aspect", &index.to_string());
        add(
            &mut graph,
            comparison_iri.clone(),
            oxa::HAS_CROSS_ASPECT,
            node.clone(),
        );
        add_aspect(
            &mut graph,
            &node,
            oxa::CROSS_ASPECT,
            &point_iri(chart_a_iri, from),
            &point_iri(chart_b_iri, to),
            record,
        );
    }

    add_provenance(&mut graph, comparison_iri, minter, &comparison.provenance);
    graph
}

/// Builds the RDF graph of a midpoint composite chart.
#[must_use]
pub fn composite_graph(
    composite: &CompositeChartResource,
    chart_iri: &NamedNode,
    minter: &IriMinter,
) -> Graph {
    let mut graph = Graph::new();
    add_side(&mut graph, chart_iri, &composite.side);

    for source in &composite.source_charts {
        // The source IRIs come from `IriMinter`, so they are valid; a
        // malformed one is dropped rather than corrupting the graph.
        if let Ok(node) = NamedNode::new(source.clone()) {
            add(
                &mut graph,
                chart_iri.clone(),
                prov::WAS_DERIVED_FROM,
                node.clone(),
            );
        }
    }

    for (index, record) in composite.aspects.iter().enumerate() {
        let (Some(from), Some(to)) = (
            find_point(&composite.side, &record.from_name),
            find_point(&composite.side, &record.to_name),
        ) else {
            continue;
        };
        let node = IriMinter::child(chart_iri, "aspect", &index.to_string());
        add(&mut graph, chart_iri.clone(), oxa::HAS_ASPECT, node.clone());
        add_aspect(
            &mut graph,
            &node,
            oxa::ASPECT_OCCURRENCE,
            &point_iri(chart_iri, from),
            &point_iri(chart_iri, to),
            record,
        );
        if let Some(planet) = from.planet {
            add(
                &mut graph,
                node.clone(),
                oxa::FIRST_BODY,
                planet_concept(planet),
            );
        }
        if let Some(planet) = to.planet {
            add(&mut graph, node, oxa::SECOND_BODY, planet_concept(planet));
        }
    }

    add_provenance(&mut graph, chart_iri, minter, &composite.provenance);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AngleKind, ChartKind, Observer, Provenance};
    use crate::{ChartKey, ComparisonKey};
    use oxiephemeris_astro::aspects::AspectKind;
    use oxiephemeris_astro::dignities::Planet;
    use oxiephemeris_astro::houses::HouseSystem;
    use oxiephemeris_astro::motion::MotionState;
    use oxiephemeris_astro::zodiac::Sign;
    use oxrdf::TripleRef;

    fn planet_point(name: &str, planet: Planet, lon: f64, sign: Sign) -> ChartPointRecord {
        ChartPointRecord {
            name: name.to_owned(),
            planet: Some(planet),
            angle: None,
            lon_deg: lon,
            speed_deg_per_day: 1.0,
            sign,
            degrees_in_sign: lon % 30.0,
            motion: MotionState::Direct,
        }
    }

    fn asc_point(lon: f64, sign: Sign) -> ChartPointRecord {
        ChartPointRecord {
            name: "ASC".to_owned(),
            planet: None,
            angle: Some(AngleKind::Ascendant),
            lon_deg: lon,
            speed_deg_per_day: 0.0,
            sign,
            degrees_in_sign: lon % 30.0,
            motion: MotionState::Stationary,
        }
    }

    fn side(kind: ChartKind, jd: f64) -> ChartSide {
        ChartSide {
            kind,
            epoch_utc: Some("1970-01-01T00:00:00.000000Z".to_owned()),
            jd_tt: Some(jd),
            observer: Some(Observer {
                lat_deg: 51.4779,
                lon_deg: 0.0,
                alt_m: 0.0,
            }),
            house_system: HouseSystem::Placidus,
            points: vec![
                planet_point("Sun", Planet::Sun, 235.5, Sign::Scorpio),
                asc_point(29.98, Sign::Aries),
            ],
        }
    }

    fn provenance() -> Provenance {
        Provenance {
            software_name: "oxiephemeris".to_owned(),
            software_version: "0.1.1".to_owned(),
            ephemeris_label: "DE440".to_owned(),
        }
    }

    fn natal_key(jd: f64) -> ChartKey<'static> {
        ChartKey {
            kind: "natal",
            jd_tt: jd,
            lat_deg: 51.4779,
            lon_deg: 0.0,
            house_system: "placidus",
            zodiac: "tropical",
        }
    }

    fn built() -> (Graph, NamedNode, NamedNode, NamedNode) {
        let minter = IriMinter::default();
        let a = minter.chart(&natal_key(2_440_587.500_465_196));
        let b = minter.chart(&natal_key(2_451_545.0));
        let cmp = ChartComparison {
            kind: ComparisonKind::Synastry,
            chart_a: side(ChartKind::Natal, 2_440_587.500_465_196),
            chart_b: side(ChartKind::Natal, 2_451_545.0),
            cross_aspects: vec![
                CrossAspectRecord {
                    from_name: "Sun".to_owned(),
                    to_name: "ASC".to_owned(),
                    kind: AspectKind::Trine,
                    offset_deg: 1.5,
                    applying: true,
                },
                // Dangling: chart B has no "Pluto" point.
                CrossAspectRecord {
                    from_name: "Sun".to_owned(),
                    to_name: "Pluto".to_owned(),
                    kind: AspectKind::Square,
                    offset_deg: 0.5,
                    applying: false,
                },
            ],
            elapsed_years: None,
            provenance: provenance(),
        };
        let key = ComparisonKey {
            kind: "synastry",
            chart_a: a.as_str(),
            chart_b: b.as_str(),
        };
        let iri = minter.comparison(&key);
        let graph = comparison_graph(&cmp, &iri, &a, &b, &minter);
        (graph, iri, a, b)
    }

    /// The point of the whole design: a side's Sun carries the *same* IRI
    /// that `crate::chart` would mint for that chart's Sun, so the two
    /// documents merge.
    #[test]
    fn side_points_reuse_the_chart_position_iris() {
        let (graph, _, a, _) = built();
        let sun = IriMinter::child(&a, "position", "Sun");
        assert!(graph.contains(TripleRef::new(sun.as_ref(), rdf::TYPE, oxa::BODY_POSITION)));
        assert!(graph.contains(TripleRef::new(
            a.as_ref(),
            oxa::HAS_BODY_POSITION,
            sun.as_ref()
        )));
        // And the Ascendant lands under /angle/ascendant, not /position/ASC.
        let asc = IriMinter::child(&a, "angle", "ascendant");
        assert!(graph.contains(TripleRef::new(asc.as_ref(), rdf::TYPE, oxa::CHART_ANGLE)));
        assert!(!graph
            .iter()
            .any(|t| t.subject.to_string().contains("/position/ASC")));
    }

    #[test]
    fn comparison_links_both_charts_and_its_cross_aspects() {
        let (graph, iri, a, b) = built();
        assert!(graph.contains(TripleRef::new(
            iri.as_ref(),
            rdf::TYPE,
            oxa::SYNASTRY_COMPARISON
        )));
        assert!(graph.contains(TripleRef::new(iri.as_ref(), oxa::CHART_A, a.as_ref())));
        assert!(graph.contains(TripleRef::new(iri.as_ref(), oxa::CHART_B, b.as_ref())));
        let cross = IriMinter::child(&iri, "cross-aspect", "0");
        assert!(graph.contains(TripleRef::new(
            cross.as_ref(),
            oxa::FROM_POINT,
            IriMinter::child(&a, "position", "Sun").as_ref()
        )));
        assert!(graph.contains(TripleRef::new(
            cross.as_ref(),
            oxa::TO_POINT,
            IriMinter::child(&b, "angle", "ascendant").as_ref()
        )));
    }

    /// A cross-aspect naming a point that was never emitted would dangle;
    /// it must be dropped, not published as a broken link.
    #[test]
    fn dangling_cross_aspects_are_dropped() {
        let (graph, iri, _, _) = built();
        let orphan = IriMinter::child(&iri, "cross-aspect", "1");
        assert!(
            !graph
                .iter()
                .any(|t| t.subject.to_string().contains(orphan.as_str())),
            "the Pluto cross-aspect has no endpoint and must not be emitted"
        );
    }

    #[test]
    fn a_cross_aspect_is_an_aspect_occurrence_subclass_instance() {
        let (graph, iri, _, _) = built();
        let cross = IriMinter::child(&iri, "cross-aspect", "0");
        assert!(graph.contains(TripleRef::new(cross.as_ref(), rdf::TYPE, oxa::CROSS_ASPECT)));
    }

    #[test]
    fn composite_is_a_chart_derived_from_its_two_sources() {
        let minter = IriMinter::default();
        let a = minter.chart(&natal_key(2_440_587.500_465_196));
        let b = minter.chart(&natal_key(2_451_545.0));
        let mut composite_side = side(ChartKind::Composite, 0.0);
        composite_side.epoch_utc = None;
        composite_side.jd_tt = None;
        composite_side.observer = None;
        let composite = CompositeChartResource {
            side: composite_side,
            aspects: vec![CrossAspectRecord {
                from_name: "Sun".to_owned(),
                to_name: "ASC".to_owned(),
                kind: AspectKind::Trine,
                offset_deg: 1.0,
                applying: false,
            }],
            source_charts: vec![a.as_str().to_owned(), b.as_str().to_owned()],
            provenance: provenance(),
        };
        let key = ComparisonKey {
            kind: "composite",
            chart_a: a.as_str(),
            chart_b: b.as_str(),
        };
        let iri = minter.chart_from_slug(&key.slug());
        let graph = composite_graph(&composite, &iri, &minter);

        assert!(graph.contains(TripleRef::new(
            iri.as_ref(),
            rdf::TYPE,
            oxa::COMPOSITE_CHART
        )));
        assert!(graph.contains(TripleRef::new(
            iri.as_ref(),
            prov::WAS_DERIVED_FROM,
            a.as_ref()
        )));
        assert!(graph.contains(TripleRef::new(
            iri.as_ref(),
            prov::WAS_DERIVED_FROM,
            b.as_ref()
        )));
        // A composite has no epoch: none must be invented.
        assert!(!graph.iter().any(|t| t.predicate == oxa::AT_EPOCH));
        assert!(!graph.iter().any(|t| t.predicate == oxa::JULIAN_DATE_TT));
    }
}
