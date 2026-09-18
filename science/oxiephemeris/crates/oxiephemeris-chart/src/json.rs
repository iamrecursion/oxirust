//! The stable JSON view of a chart, comparison, or composite.
//!
//! This is the single JSON schema the CLI (`--format json`), the Python
//! `natal(...)` dict, and the WASM `natal_json` all render, built from the
//! RDF-neutral [`oxiephemeris_rdf::model`] resources so they cannot drift.
//! Degrees throughout; longitudes are the displayed (tropical or sidereal)
//! values.

use serde::Serialize;

use oxiephemeris_rdf::model as m;

/// One body in the JSON `bodies` array.
#[derive(Debug, Serialize)]
pub struct BodyJson {
    /// Body name.
    pub body: String,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
    /// Ecliptic latitude, degrees.
    pub lat_deg: f64,
    /// Daily longitude speed, degrees/day.
    pub lon_speed_deg_per_day: f64,
    /// Daily latitude speed, degrees/day.
    pub lat_speed_deg_per_day: f64,
    /// Geocentric distance, AU.
    pub distance_au: f64,
    /// One-way light time, days.
    pub light_time_days: f64,
    /// Zodiac sign name.
    pub sign: &'static str,
    /// Degrees within the sign, `[0, 30)`.
    pub sign_degrees: f64,
    /// Whether retrograde (negative longitude speed).
    pub retrograde: bool,
    /// Motion state name.
    pub motion: &'static str,
    /// House occupied, `1..=12`.
    pub house: usize,
    /// Equatorial declination, degrees.
    pub declination_deg: f64,
    /// Whether out of bounds in declination.
    pub out_of_bounds: bool,
}

/// The `houses` object.
#[derive(Debug, Serialize)]
pub struct HousesJson {
    /// House-system name.
    pub system: &'static str,
    /// Cusps 1-12, degrees.
    pub cusps_deg: Vec<f64>,
}

/// The `angles` object.
#[derive(Debug, Serialize)]
pub struct AnglesJson {
    /// Ascendant, degrees.
    pub ascendant_deg: f64,
    /// Midheaven, degrees.
    pub mc_deg: f64,
    /// Vertex, degrees.
    pub vertex_deg: f64,
    /// East Point, degrees.
    pub east_point_deg: f64,
}

/// The `nodes` object.
#[derive(Debug, Serialize)]
pub struct NodesJson {
    /// Mean node, degrees.
    pub mean_node_deg: f64,
    /// Mean apogee (Lilith), degrees.
    pub mean_apogee_deg: f64,
    /// True node, degrees.
    pub true_node_deg: f64,
    /// True apogee, degrees.
    pub true_apogee_deg: f64,
}

/// The `lots` object.
#[derive(Debug, Serialize)]
pub struct LotsJson {
    /// Lot of Fortune, degrees.
    pub fortune_deg: f64,
    /// Lot of Spirit, degrees.
    pub spirit_deg: f64,
}

/// The `distribution` object.
#[derive(Debug, Serialize)]
pub struct DistributionJson {
    /// Fire count.
    pub fire: usize,
    /// Earth count.
    pub earth: usize,
    /// Air count.
    pub air: usize,
    /// Water count.
    pub water: usize,
    /// Cardinal count.
    pub cardinal: usize,
    /// Fixed count.
    pub fixed: usize,
    /// Mutable count.
    pub mutable: usize,
}

/// One planet's dignity in the JSON `dignities` array.
#[allow(clippy::struct_excessive_bools)] // one flag per classical tier
#[derive(Debug, Serialize)]
pub struct DignityJson {
    /// Planet name.
    pub body: &'static str,
    /// Rules the sign.
    pub domicile: bool,
    /// Exalted in the sign.
    pub exaltation: bool,
    /// The sect's triplicity ruler.
    pub triplicity: bool,
    /// Rules the term.
    pub term: bool,
    /// Rules the face.
    pub face: bool,
    /// Opposite its domicile.
    pub detriment: bool,
    /// Opposite its exaltation.
    pub fall: bool,
    /// No dignity and no debility.
    pub peregrine: bool,
    /// Lilly point sum.
    pub score: i32,
}

/// One aspect in the JSON `aspects` array.
#[derive(Debug, Serialize)]
pub struct AspectJson {
    /// First body.
    pub body1: String,
    /// Second body.
    pub body2: String,
    /// Aspect kind name.
    pub aspect: &'static str,
    /// Exact separation, degrees.
    pub exact_angle_deg: f64,
    /// Signed offset from exactness, degrees.
    pub offset_deg: f64,
    /// Whether applying.
    pub applying: bool,
}

/// The full natal-chart JSON view.
#[derive(Debug, Serialize)]
pub struct ChartJson {
    /// Chart kind name.
    pub kind: &'static str,
    /// Epoch as a Julian Date in TT.
    pub jd_tt: f64,
    /// Epoch as ISO 8601 UTC.
    pub epoch_utc: String,
    /// Ayanamsha name when sidereal, else absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sidereal: Option<&'static str>,
    /// Ayanamsha value at the epoch, degrees, when sidereal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ayanamsha_deg: Option<f64>,
    /// Rulership scheme used for dignities.
    pub rulership: String,
    /// Chart sect (diurnal/nocturnal).
    pub sect: &'static str,
    /// The bodies.
    pub bodies: Vec<BodyJson>,
    /// The houses.
    pub houses: HousesJson,
    /// The chart angles.
    pub angles: AnglesJson,
    /// The lunar nodes and apogees.
    pub nodes: NodesJson,
    /// The Arabic Parts.
    pub lots: LotsJson,
    /// Element/modality distribution.
    pub distribution: DistributionJson,
    /// Essential dignities.
    pub dignities: Vec<DignityJson>,
    /// The aspects.
    pub aspects: Vec<AspectJson>,
}

fn dignity_json(record: &m::DignityRecord) -> DignityJson {
    let has = |t: m::DignityTier| record.tiers.contains(&t);
    DignityJson {
        body: record.planet.name(),
        domicile: has(m::DignityTier::Domicile),
        exaltation: has(m::DignityTier::Exaltation),
        triplicity: has(m::DignityTier::Triplicity),
        term: has(m::DignityTier::Term),
        face: has(m::DignityTier::Face),
        detriment: has(m::DignityTier::Detriment),
        fall: has(m::DignityTier::Fall),
        peregrine: has(m::DignityTier::Peregrine),
        score: record.score,
    }
}

/// Finds a node/angle/lot longitude by kind, defaulting to `0.0`.
fn node_lon(nodes: &[m::NodeRecord], kind: m::NodeKind) -> f64 {
    nodes
        .iter()
        .find(|n| n.kind == kind)
        .map_or(0.0, |n| n.lon_deg)
}

fn angle_lon(angles: &[m::AngleRecord], kind: m::AngleKind) -> f64 {
    angles
        .iter()
        .find(|a| a.kind == kind)
        .map_or(0.0, |a| a.lon_deg)
}

fn lot_lon(lots: &[m::LotRecord], kind: m::LotKind) -> f64 {
    lots.iter()
        .find(|l| l.kind == kind)
        .map_or(0.0, |l| l.lon_deg)
}

/// Builds the JSON view of a [`m::ChartResource`].
#[must_use]
pub fn chart_json(chart: &m::ChartResource) -> ChartJson {
    let (sidereal, ayanamsha_deg) = match chart.zodiac {
        m::Zodiac::Tropical => (None, None),
        m::Zodiac::Sidereal { ayanamsha, degrees } => (ayanamsha.name(), Some(degrees)),
    };
    ChartJson {
        kind: chart.kind.name(),
        jd_tt: chart.jd_tt,
        epoch_utc: chart.epoch_utc.clone(),
        sidereal,
        ayanamsha_deg,
        rulership: chart.rulership.clone(),
        sect: chart.sect.name(),
        bodies: chart
            .bodies
            .iter()
            .map(|b| BodyJson {
                body: b.name.clone(),
                lon_deg: b.lon_deg,
                lat_deg: b.lat_deg,
                lon_speed_deg_per_day: b.lon_speed_deg_per_day,
                lat_speed_deg_per_day: b.lat_speed_deg_per_day,
                distance_au: b.distance_au,
                light_time_days: b.light_time_days,
                sign: b.sign.name(),
                sign_degrees: b.degrees_in_sign,
                retrograde: b.motion.is_retrograde(),
                motion: b.motion.name(),
                house: b.house,
                declination_deg: b.declination_deg,
                out_of_bounds: b.out_of_bounds,
            })
            .collect(),
        houses: HousesJson {
            system: chart.house_system.name(),
            cusps_deg: chart.cusps.iter().map(|c| c.lon_deg).collect(),
        },
        angles: AnglesJson {
            ascendant_deg: angle_lon(&chart.angles, m::AngleKind::Ascendant),
            mc_deg: angle_lon(&chart.angles, m::AngleKind::Midheaven),
            vertex_deg: angle_lon(&chart.angles, m::AngleKind::Vertex),
            east_point_deg: angle_lon(&chart.angles, m::AngleKind::EastPoint),
        },
        nodes: NodesJson {
            mean_node_deg: node_lon(&chart.nodes, m::NodeKind::MeanNode),
            mean_apogee_deg: node_lon(&chart.nodes, m::NodeKind::MeanApogee),
            true_node_deg: node_lon(&chart.nodes, m::NodeKind::TrueNode),
            true_apogee_deg: node_lon(&chart.nodes, m::NodeKind::TrueApogee),
        },
        lots: LotsJson {
            fortune_deg: lot_lon(&chart.lots, m::LotKind::Fortune),
            spirit_deg: lot_lon(&chart.lots, m::LotKind::Spirit),
        },
        distribution: DistributionJson {
            fire: chart.distribution.elements[0],
            earth: chart.distribution.elements[1],
            air: chart.distribution.elements[2],
            water: chart.distribution.elements[3],
            cardinal: chart.distribution.modalities[0],
            fixed: chart.distribution.modalities[1],
            mutable: chart.distribution.modalities[2],
        },
        dignities: chart.dignities.iter().map(dignity_json).collect(),
        aspects: chart
            .aspects
            .iter()
            .map(|a| AspectJson {
                body1: a.body1.clone(),
                body2: a.body2.clone(),
                aspect: a.kind.name(),
                exact_angle_deg: a.exact_angle_deg,
                offset_deg: a.offset_deg,
                applying: a.applying,
            })
            .collect(),
    }
}

/// One point in a comparison's JSON side.
#[derive(Debug, Serialize)]
pub struct PointJson {
    /// Point name.
    pub point: String,
    /// Longitude, degrees.
    pub lon_deg: f64,
    /// Sign name.
    pub sign: &'static str,
    /// Degrees within the sign.
    pub sign_degrees: f64,
    /// Whether retrograde.
    pub retrograde: bool,
    /// Daily speed, degrees/day.
    pub speed_deg_per_day: f64,
}

/// One cross-aspect in a comparison's JSON.
#[derive(Debug, Serialize)]
pub struct CrossAspectJson {
    /// From-point name.
    pub from: String,
    /// To-point name.
    pub to: String,
    /// Aspect kind name.
    pub aspect: &'static str,
    /// Signed offset from exactness, degrees.
    pub offset_deg: f64,
    /// Whether applying.
    pub applying: bool,
}

/// A comparison JSON view.
#[derive(Debug, Serialize)]
pub struct ComparisonJson {
    /// Comparison kind name.
    pub kind: &'static str,
    /// The moving/left side's points.
    pub chart_a: Vec<PointJson>,
    /// The fixed/right side's points.
    pub chart_b: Vec<PointJson>,
    /// The cross-aspects.
    pub cross_aspects: Vec<CrossAspectJson>,
    /// Elapsed tropical years (progression only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_years: Option<f64>,
}

fn point_json(p: &m::ChartPointRecord) -> PointJson {
    PointJson {
        point: p.name.clone(),
        lon_deg: p.lon_deg,
        sign: p.sign.name(),
        sign_degrees: p.degrees_in_sign,
        retrograde: p.motion.is_retrograde(),
        speed_deg_per_day: p.speed_deg_per_day,
    }
}

fn cross_json(c: &m::CrossAspectRecord) -> CrossAspectJson {
    CrossAspectJson {
        from: c.from_name.clone(),
        to: c.to_name.clone(),
        aspect: c.kind.name(),
        offset_deg: c.offset_deg,
        applying: c.applying,
    }
}

/// Builds the JSON view of a comparison.
#[must_use]
pub fn comparison_json(cmp: &m::ChartComparison) -> ComparisonJson {
    ComparisonJson {
        kind: cmp.kind.name(),
        chart_a: cmp.chart_a.points.iter().map(point_json).collect(),
        chart_b: cmp.chart_b.points.iter().map(point_json).collect(),
        cross_aspects: cmp.cross_aspects.iter().map(cross_json).collect(),
        elapsed_years: cmp.elapsed_years,
    }
}

/// A composite JSON view.
#[derive(Debug, Serialize)]
pub struct CompositeJson {
    /// The composite chart's points.
    pub composite: Vec<PointJson>,
    /// The composite's internal aspects.
    pub aspects: Vec<CrossAspectJson>,
    /// IRIs of the two source charts.
    pub source_charts: Vec<String>,
}

/// Builds the JSON view of a composite chart.
#[must_use]
pub fn composite_json(comp: &m::CompositeChartResource) -> CompositeJson {
    CompositeJson {
        composite: comp.side.points.iter().map(point_json).collect(),
        aspects: comp.aspects.iter().map(cross_json).collect(),
        source_charts: comp.source_charts.clone(),
    }
}
