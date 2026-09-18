//! Turning a [`ChartResource`] into an RDF graph.
//!
//! # Skolemized n-ary relations
//!
//! "Sun is in Scorpio at 25°30′33″ in the 7th house, retrograde" is not a
//! triple — it is a five-place relation about one *placement*. Rather than
//! a blank node (unreferenceable from outside the document), each
//! placement, cusp, angle, node, lot, aspect and dignity assessment gets
//! its own IRI under the chart's:
//!
//! ```text
//! <chart>              oxa:hasBodyPosition <chart/position/Sun> .
//! <chart/position/Sun> oxa:body            oxc:planet/Sun ;
//!                      oxa:inSign          oxc:sign/Scorpio ;
//!                      oxa:houseNumber     7 ;
//!                      oxa:isRetrograde    false .
//! ```
//!
//! A consumer can then link to, annotate, or query any individual
//! placement across documents — which is the whole point of publishing
//! Linked Data rather than a JSON blob.
//!
//! # What is *not* duplicated
//!
//! An aspect occurrence carries its orb and its applying flag, but not its
//! exact angle: that is a property of the aspect *kind*
//! (`oxc:aspect/trine oxa:exactAngle 120`), reachable in one hop. Chart
//! graphs state facts about the chart; definitional facts live in the
//! vocabulary.

use oxrdf::vocab::{rdf, rdfs};
use oxrdf::{Graph, NamedNode};

use oxiephemeris_astro::dignities::Planet;
use oxiephemeris_astro::zodiac::{Sign, SignPosition};

use crate::builder::{add, add_double, boolean, date_time_stamp, integer, plain};
use crate::iri::IriMinter;
use crate::model::{ChartKind, ChartResource, Zodiac};
use crate::prov::add_provenance;
use crate::skos::{
    angle_concept, aspect_concept, ayanamsha_concept, dignity_tier_concept, house_system_concept,
    lot_concept, motion_concept, node_concept, planet_concept, sect_concept, sign_concept,
};
use crate::vocab::{geo, oxa, time};

/// Resolves a body display name to its [`Planet`], if it is one of the ten.
#[must_use]
pub fn planet_by_name(name: &str) -> Option<Planet> {
    Planet::ALL.into_iter().find(|p| p.name() == name)
}

/// The sign and the degrees within it, for a longitude in degrees.
fn sign_and_degrees(lon_deg: f64) -> (Sign, f64) {
    let pos = SignPosition::of(lon_deg.to_radians());
    (pos.sign, pos.degrees_in_sign)
}

/// Adds `oxa:eclipticLongitude`, `oxa:inSign` and `oxa:degreesInSign` for a
/// resource whose sign is derived from its longitude (cusps, angles,
/// nodes, lots — none of which carry a precomputed sign).
fn add_position(graph: &mut Graph, subject: &NamedNode, lon_deg: f64) {
    add_double(graph, subject.clone(), oxa::ECLIPTIC_LONGITUDE, lon_deg);
    if lon_deg.is_finite() {
        let (sign, degrees) = sign_and_degrees(lon_deg);
        add(graph, subject.clone(), oxa::IN_SIGN, sign_concept(sign));
        add_double(graph, subject.clone(), oxa::DEGREES_IN_SIGN, degrees);
    }
}

/// Adds the same three triples for a body, taking the sign from the model
/// rather than re-deriving it: the caller already computed it (possibly in
/// a sidereal zodiac, where re-deriving from the longitude alone would be
/// correct only by coincidence).
fn add_body_position(graph: &mut Graph, subject: &NamedNode, body: &crate::model::BodyPlacement) {
    add_double(
        graph,
        subject.clone(),
        oxa::ECLIPTIC_LONGITUDE,
        body.lon_deg,
    );
    add(
        graph,
        subject.clone(),
        oxa::IN_SIGN,
        sign_concept(body.sign),
    );
    add_double(
        graph,
        subject.clone(),
        oxa::DEGREES_IN_SIGN,
        body.degrees_in_sign,
    );
}

/// The `oxa:` class for a chart kind.
pub(crate) fn chart_class(kind: ChartKind) -> oxrdf::NamedNodeRef<'static> {
    match kind {
        ChartKind::Natal => oxa::NATAL_CHART,
        ChartKind::Composite => oxa::COMPOSITE_CHART,
        ChartKind::Progressed => oxa::PROGRESSED_CHART,
        ChartKind::Transit => oxa::TRANSIT_CHART,
    }
}

/// Builds the RDF graph of `chart`, identified by `chart_iri`.
///
/// `minter` is used only for the provenance resources (agent, ephemeris);
/// every chart sub-resource is skolemized under `chart_iri` itself, so a
/// caller may pass an explicit chart IRI from a different namespace than
/// the minter's base.
#[must_use]
#[allow(clippy::too_many_lines)] // one linear pass per chart section
pub fn chart_graph(chart: &ChartResource, chart_iri: &NamedNode, minter: &IriMinter) -> Graph {
    let mut graph = Graph::new();

    // --- the chart itself -------------------------------------------
    add(&mut graph, chart_iri.clone(), rdf::TYPE, oxa::CHART);
    add(
        &mut graph,
        chart_iri.clone(),
        rdf::TYPE,
        chart_class(chart.kind),
    );
    add_double(
        &mut graph,
        chart_iri.clone(),
        oxa::JULIAN_DATE_TT,
        chart.jd_tt,
    );
    // `oxa:rulershipScheme` is NOT stated here: it qualifies each dignity
    // assessment, not the chart (see `ChartKey`'s doc).
    add(
        &mut graph,
        chart_iri.clone(),
        oxa::HAS_SECT,
        sect_concept(chart.sect),
    );
    add(
        &mut graph,
        chart_iri.clone(),
        oxa::HOUSE_SYSTEM_OF,
        house_system_concept(chart.house_system),
    );
    if let Zodiac::Sidereal { ayanamsha, degrees } = chart.zodiac {
        if let Some(concept) = ayanamsha_concept(ayanamsha) {
            add(&mut graph, chart_iri.clone(), oxa::AYANAMSHA_OF, concept);
        }
        add_double(
            &mut graph,
            chart_iri.clone(),
            oxa::AYANAMSHA_DEGREES,
            degrees,
        );
    }
    if let Some(years) = chart.elapsed_years {
        add_double(&mut graph, chart_iri.clone(), oxa::ELAPSED_YEARS, years);
    }

    // --- epoch (OWL-Time) -------------------------------------------
    let epoch = IriMinter::part(chart_iri, "epoch");
    add(&mut graph, chart_iri.clone(), oxa::AT_EPOCH, epoch.clone());
    add(&mut graph, epoch.clone(), rdf::TYPE, time::INSTANT);
    add(
        &mut graph,
        epoch,
        time::IN_XSD_DATE_TIME_STAMP,
        date_time_stamp(&chart.epoch_utc),
    );

    // --- observer (WGS84 geo) ---------------------------------------
    if let Some(observer) = chart.observer {
        let node = IriMinter::part(chart_iri, "observer");
        add(
            &mut graph,
            chart_iri.clone(),
            oxa::HAS_OBSERVER,
            node.clone(),
        );
        add(&mut graph, node.clone(), rdf::TYPE, oxa::OBSERVER);
        add_double(&mut graph, node.clone(), geo::LAT, observer.lat_deg);
        add_double(&mut graph, node.clone(), geo::LONG, observer.lon_deg);
        add_double(&mut graph, node, geo::ALT, observer.alt_m);
    }

    // --- bodies -------------------------------------------------------
    for body in &chart.bodies {
        let node = IriMinter::child(chart_iri, "position", &body.name);
        add(
            &mut graph,
            chart_iri.clone(),
            oxa::HAS_BODY_POSITION,
            node.clone(),
        );
        add(&mut graph, node.clone(), rdf::TYPE, oxa::BODY_POSITION);
        add(&mut graph, node.clone(), rdfs::LABEL, plain(&body.name));
        if let Some(planet) = body.planet {
            add(&mut graph, node.clone(), oxa::BODY, planet_concept(planet));
        }
        add_body_position(&mut graph, &node, body);
        add_double(
            &mut graph,
            node.clone(),
            oxa::ECLIPTIC_LATITUDE,
            body.lat_deg,
        );
        add_double(
            &mut graph,
            node.clone(),
            oxa::LONGITUDE_SPEED,
            body.lon_speed_deg_per_day,
        );
        add_double(
            &mut graph,
            node.clone(),
            oxa::LATITUDE_SPEED,
            body.lat_speed_deg_per_day,
        );
        add_double(&mut graph, node.clone(), oxa::DISTANCE_AU, body.distance_au);
        add_double(
            &mut graph,
            node.clone(),
            oxa::LIGHT_TIME_DAYS,
            body.light_time_days,
        );
        add_double(
            &mut graph,
            node.clone(),
            oxa::DECLINATION,
            body.declination_deg,
        );
        add(
            &mut graph,
            node.clone(),
            oxa::HOUSE_NUMBER,
            integer(i64::try_from(body.house).unwrap_or(0)),
        );
        add(
            &mut graph,
            node.clone(),
            oxa::MOTION,
            motion_concept(body.motion),
        );
        add(
            &mut graph,
            node.clone(),
            oxa::IS_RETROGRADE,
            boolean(body.motion.is_retrograde()),
        );
        add(
            &mut graph,
            node,
            oxa::IS_OUT_OF_BOUNDS,
            boolean(body.out_of_bounds),
        );
    }

    // --- house cusps ---------------------------------------------------
    for cusp in &chart.cusps {
        let node = IriMinter::child(chart_iri, "cusp", &cusp.number.to_string());
        add(
            &mut graph,
            chart_iri.clone(),
            oxa::HAS_HOUSE_CUSP,
            node.clone(),
        );
        add(&mut graph, node.clone(), rdf::TYPE, oxa::HOUSE_CUSP);
        add(
            &mut graph,
            node.clone(),
            oxa::CUSP_NUMBER,
            integer(i64::try_from(cusp.number).unwrap_or(0)),
        );
        add_position(&mut graph, &node, cusp.lon_deg);
    }

    // --- angles ---------------------------------------------------------
    for angle in &chart.angles {
        let node = IriMinter::child(chart_iri, "angle", angle.kind.name());
        add(&mut graph, chart_iri.clone(), oxa::HAS_ANGLE, node.clone());
        add(&mut graph, node.clone(), rdf::TYPE, oxa::CHART_ANGLE);
        add(
            &mut graph,
            node.clone(),
            oxa::ANGLE_KIND_OF,
            angle_concept(angle.kind),
        );
        add_position(&mut graph, &node, angle.lon_deg);
    }

    // --- lunar nodes ----------------------------------------------------
    for lunar in &chart.nodes {
        let node = IriMinter::child(chart_iri, "node", lunar.kind.name());
        add(
            &mut graph,
            chart_iri.clone(),
            oxa::HAS_LUNAR_NODE,
            node.clone(),
        );
        add(&mut graph, node.clone(), rdf::TYPE, oxa::LUNAR_NODE);
        add(
            &mut graph,
            node.clone(),
            oxa::NODE_KIND_OF,
            node_concept(lunar.kind),
        );
        add_position(&mut graph, &node, lunar.lon_deg);
    }

    // --- lots ------------------------------------------------------------
    for lot in &chart.lots {
        let node = IriMinter::child(chart_iri, "lot", lot.kind.name());
        add(&mut graph, chart_iri.clone(), oxa::HAS_LOT, node.clone());
        add(&mut graph, node.clone(), rdf::TYPE, oxa::LOT);
        add(
            &mut graph,
            node.clone(),
            oxa::LOT_KIND_OF,
            lot_concept(lot.kind),
        );
        add_position(&mut graph, &node, lot.lon_deg);
    }

    // --- aspects ----------------------------------------------------------
    for (index, aspect) in chart.aspects.iter().enumerate() {
        let node = IriMinter::child(chart_iri, "aspect", &index.to_string());
        add(&mut graph, chart_iri.clone(), oxa::HAS_ASPECT, node.clone());
        add(&mut graph, node.clone(), rdf::TYPE, oxa::ASPECT_OCCURRENCE);
        // Point-level endpoints: which *placements* aspect each other, not
        // merely which planet concepts. `firstBody`/`secondBody` below stay
        // as the convenient concept-level shortcut.
        add(
            &mut graph,
            node.clone(),
            oxa::FROM_POINT,
            IriMinter::child(chart_iri, "position", &aspect.body1),
        );
        add(
            &mut graph,
            node.clone(),
            oxa::TO_POINT,
            IriMinter::child(chart_iri, "position", &aspect.body2),
        );
        if let Some(planet) = planet_by_name(&aspect.body1) {
            add(
                &mut graph,
                node.clone(),
                oxa::FIRST_BODY,
                planet_concept(planet),
            );
        }
        if let Some(planet) = planet_by_name(&aspect.body2) {
            add(
                &mut graph,
                node.clone(),
                oxa::SECOND_BODY,
                planet_concept(planet),
            );
        }
        add(
            &mut graph,
            node.clone(),
            oxa::ASPECT_KIND_OF,
            aspect_concept(aspect.kind),
        );
        add_double(&mut graph, node.clone(), oxa::ORB, aspect.offset_deg);
        add(&mut graph, node, oxa::IS_APPLYING, boolean(aspect.applying));
    }

    // --- dignities ----------------------------------------------------------
    // The assessment IRI is qualified by the rulership scheme, so one chart
    // can carry a traditional and a modern assessment of the same planet
    // side by side instead of one silently overwriting the other.
    for dignity in &chart.dignities {
        let node = IriMinter::child_at(
            chart_iri,
            &["dignity", &chart.rulership, dignity.planet.name()],
        );
        add(
            &mut graph,
            chart_iri.clone(),
            oxa::HAS_DIGNITY,
            node.clone(),
        );
        add(&mut graph, node.clone(), rdf::TYPE, oxa::DIGNITY_ASSESSMENT);
        add(
            &mut graph,
            node.clone(),
            oxa::RULERSHIP_SCHEME,
            plain(&chart.rulership),
        );
        add(
            &mut graph,
            node.clone(),
            oxa::BODY,
            planet_concept(dignity.planet),
        );
        for tier in &dignity.tiers {
            add(
                &mut graph,
                node.clone(),
                oxa::HOLDS_TIER,
                dignity_tier_concept(*tier),
            );
        }
        add(
            &mut graph,
            node,
            oxa::DIGNITY_SCORE,
            integer(i64::from(dignity.score)),
        );
    }

    // --- distribution --------------------------------------------------------
    let distribution = IriMinter::part(chart_iri, "distribution");
    add(
        &mut graph,
        chart_iri.clone(),
        oxa::HAS_DISTRIBUTION,
        distribution.clone(),
    );
    add(
        &mut graph,
        distribution.clone(),
        rdf::TYPE,
        oxa::DISTRIBUTION,
    );
    let counts = [
        (oxa::FIRE_COUNT, chart.distribution.elements[0]),
        (oxa::EARTH_COUNT, chart.distribution.elements[1]),
        (oxa::AIR_COUNT, chart.distribution.elements[2]),
        (oxa::WATER_COUNT, chart.distribution.elements[3]),
        (oxa::CARDINAL_COUNT, chart.distribution.modalities[0]),
        (oxa::FIXED_COUNT, chart.distribution.modalities[1]),
        (oxa::MUTABLE_COUNT, chart.distribution.modalities[2]),
    ];
    for (predicate, value) in counts {
        add(
            &mut graph,
            distribution.clone(),
            predicate,
            integer(i64::try_from(value).unwrap_or(0)),
        );
    }

    add_provenance(&mut graph, chart_iri, minter, &chart.provenance);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AngleKind, AngleRecord, AspectRecord, BodyPlacement, CuspRecord, DignityRecord,
        DignityTier, DistributionRecord, LotKind, LotRecord, NodeKind, NodeRecord, Observer,
        Provenance,
    };
    use oxiephemeris_astro::aspects::AspectKind;
    use oxiephemeris_astro::houses::HouseSystem;
    use oxiephemeris_astro::motion::MotionState;
    use oxiephemeris_astro::parts::Sect;
    use oxrdf::{NamedOrBlankNodeRef, TermRef, TripleRef};

    fn sample() -> ChartResource {
        ChartResource {
            kind: ChartKind::Natal,
            epoch_utc: "1970-01-01T00:00:00Z".to_owned(),
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
            bodies: vec![
                BodyPlacement {
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
                },
                BodyPlacement {
                    name: "Saturn".to_owned(),
                    planet: Some(Planet::Saturn),
                    lon_deg: 108.619_436_053,
                    lat_deg: -0.459_040_764,
                    lon_speed_deg_per_day: -0.032_4,
                    lat_speed_deg_per_day: 0.0,
                    distance_au: 8.397_073_769,
                    light_time_days: 0.048_5,
                    sign: Sign::Cancer,
                    degrees_in_sign: 18.619_436_053,
                    house: 4,
                    declination_deg: 21.692_9,
                    motion: MotionState::Retrograde,
                    out_of_bounds: false,
                },
            ],
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
                lon_deg: 79.494_102_640,
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

    fn built() -> (Graph, NamedNode) {
        let minter = IriMinter::default();
        let key = crate::ChartKey {
            kind: "natal",
            jd_tt: 2_440_587.500_465_196,
            lat_deg: 51.4779,
            lon_deg: 0.0,
            house_system: "placidus",
            zodiac: "tropical",
        };
        let iri = minter.chart(&key);
        let g = chart_graph(&sample(), &iri, &minter);
        (g, iri)
    }

    #[test]
    fn planet_lookup_by_name() {
        assert_eq!(planet_by_name("Mars"), Some(Planet::Mars));
        assert_eq!(planet_by_name("ASC"), None);
    }

    #[test]
    fn sun_is_in_scorpio_in_the_seventh_house() {
        let (g, chart) = built();
        let sun = IriMinter::child(&chart, "position", "Sun");
        assert!(g.contains(TripleRef::new(
            sun.as_ref(),
            oxa::IN_SIGN,
            sign_concept(Sign::Scorpio).as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            sun.as_ref(),
            oxa::BODY,
            planet_concept(Planet::Sun).as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            sun.as_ref(),
            oxa::HOUSE_NUMBER,
            TermRef::Literal(integer(7).as_ref())
        )));
        assert!(g.contains(TripleRef::new(
            sun.as_ref(),
            oxa::IS_RETROGRADE,
            TermRef::Literal(boolean(false).as_ref())
        )));
    }

    #[test]
    fn saturn_is_retrograde() {
        let (g, chart) = built();
        let saturn = IriMinter::child(&chart, "position", "Saturn");
        assert!(g.contains(TripleRef::new(
            saturn.as_ref(),
            oxa::IS_RETROGRADE,
            TermRef::Literal(boolean(true).as_ref())
        )));
        assert!(g.contains(TripleRef::new(
            saturn.as_ref(),
            oxa::MOTION,
            motion_concept(MotionState::Retrograde).as_ref()
        )));
    }

    #[test]
    fn aspect_occurrence_has_no_exact_angle_only_a_kind() {
        let (g, chart) = built();
        let aspect = IriMinter::child(&chart, "aspect", "0");
        assert!(g.contains(TripleRef::new(
            aspect.as_ref(),
            oxa::ASPECT_KIND_OF,
            aspect_concept(AspectKind::Conjunction).as_ref()
        )));
        let exact = g
            .objects_for_subject_predicate(
                NamedOrBlankNodeRef::NamedNode(aspect.as_ref()),
                oxa::EXACT_ANGLE,
            )
            .count();
        assert_eq!(
            exact, 0,
            "exactAngle belongs to the aspect kind, not the occurrence"
        );
    }

    /// The assessment lives under its rulership scheme, so a modern
    /// scoring of the same chart cannot overwrite the traditional one.
    #[test]
    fn dignity_assessment_holds_its_tiers_and_score() {
        let (g, chart) = built();
        let mars = IriMinter::child_at(&chart, &["dignity", "traditional", "Mars"]);
        assert!(g.contains(TripleRef::new(
            mars.as_ref(),
            oxa::RULERSHIP_SCHEME,
            TermRef::Literal(crate::builder::plain("traditional").as_ref())
        )));
        // The unqualified IRI must no longer exist.
        let unqualified = IriMinter::child(&chart, "dignity", "Mars");
        assert!(!g
            .iter()
            .any(|t| t.subject.to_string().contains(unqualified.as_str())));
        assert!(g.contains(TripleRef::new(
            mars.as_ref(),
            oxa::HOLDS_TIER,
            dignity_tier_concept(DignityTier::Domicile).as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            mars.as_ref(),
            oxa::HOLDS_TIER,
            dignity_tier_concept(DignityTier::Triplicity).as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            mars.as_ref(),
            oxa::DIGNITY_SCORE,
            TermRef::Literal(integer(8).as_ref())
        )));
    }

    #[test]
    fn chart_carries_epoch_observer_and_provenance() {
        let (g, chart) = built();
        let epoch = IriMinter::part(&chart, "epoch");
        assert!(g.contains(TripleRef::new(
            chart.as_ref(),
            oxa::AT_EPOCH,
            epoch.as_ref()
        )));
        let observer = IriMinter::part(&chart, "observer");
        assert!(g.contains(TripleRef::new(observer.as_ref(), rdf::TYPE, oxa::OBSERVER)));
        let activity = IriMinter::part(&chart, "activity");
        assert!(g.contains(TripleRef::new(
            chart.as_ref(),
            crate::vocab::prov::WAS_GENERATED_BY,
            activity.as_ref()
        )));
    }

    #[test]
    fn graph_is_deterministic_for_the_same_input() {
        let (a, _) = built();
        let (b, _) = built();
        assert_eq!(a.len(), b.len());
        for triple in &a {
            assert!(b.contains(triple), "graphs differ");
        }
    }

    #[test]
    fn distribution_counts_are_emitted() {
        let (g, chart) = built();
        let dist = IriMinter::part(&chart, "distribution");
        assert!(g.contains(TripleRef::new(
            dist.as_ref(),
            oxa::WATER_COUNT,
            TermRef::Literal(integer(6).as_ref())
        )));
        assert!(g.contains(TripleRef::new(
            dist.as_ref(),
            oxa::CARDINAL_COUNT,
            TermRef::Literal(integer(4).as_ref())
        )));
    }
}
