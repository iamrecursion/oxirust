//! Assembling two-chart comparisons (synastry, transit, progression) and
//! midpoint composites into the RDF layer's models.
//!
//! Each returns a `oxiephemeris_rdf::model` value ready for the JSON view
//! ([`crate::json`]) or the RDF serializers ([`crate::serialize`]). The
//! sides carry enough identity (kind, epoch, place, house system) that the
//! serializer can re-mint each one's own chart IRI, so a comparison side
//! and the standalone chart of the same birth are the same resource.

use oxiephemeris_astro::midpoints::midpoint;
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::zodiac::SignPosition;
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_core::time::JulianDate;
use oxiephemeris_de::DeFile;
use oxiephemeris_rdf::model as m;
use oxiephemeris_rdf::{ChartKey, IriMinter};

use crate::calendar::CalendarKind;
use crate::epoch::{epoch_utc_iso8601, resolve_chart_epoch, ChartEpoch};
use crate::error::ChartError;
use crate::natal::{planet_of, provenance};
use crate::points::{
    collect_cross_aspects, collect_internal_aspects, natal_points, planet_points_at,
    transit_points, ChartPoint,
};

/// Days in a mean tropical year, for the secondary-progression step.
const TROPICAL_YEAR_DAYS: f64 = 365.242_19;
/// Seconds per day, for the progressed-instant offset.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// One person's birth data (typed; the bindings parse this from a spec).
#[derive(Debug, Clone)]
pub struct PersonInput {
    /// ISO 8601 UTC date/time.
    pub date: String,
    /// Calendar the date is written in.
    pub cal: CalendarKind,
    /// Latitude, degrees north.
    pub lat_deg: f64,
    /// Longitude, degrees east.
    pub lon_deg: f64,
}

/// Maps a point display name to its [`AngleKind`], if it is an angle.
fn angle_of(name: &str) -> Option<m::AngleKind> {
    match name {
        "ASC" => Some(m::AngleKind::Ascendant),
        "MC" => Some(m::AngleKind::Midheaven),
        _ => None,
    }
}

/// Converts a [`ChartPoint`] into the RDF model's record.
fn to_record(point: &ChartPoint) -> m::ChartPointRecord {
    let sp = SignPosition::of(point.lon_rad);
    m::ChartPointRecord {
        name: point.name.to_owned(),
        planet: planet_of(point.name),
        angle: angle_of(point.name),
        lon_deg: point.lon_rad * RAD2DEG,
        speed_deg_per_day: point.speed_rad_per_day * RAD2DEG,
        sign: sp.sign,
        degrees_in_sign: sp.degrees_in_sign,
        motion: MotionState::of(point.speed_rad_per_day),
    }
}

/// Builds a chart side from its kind, epoch, observer, house system, and
/// points.
fn side(
    kind: m::ChartKind,
    epoch_utc: Option<String>,
    jd_tt: Option<f64>,
    observer: Option<(f64, f64)>,
    system: oxiephemeris_astro::houses::HouseSystem,
    points: &[ChartPoint],
) -> m::ChartSide {
    m::ChartSide {
        kind,
        epoch_utc,
        jd_tt,
        observer: observer.map(|(lat_deg, lon_deg)| m::Observer {
            lat_deg,
            lon_deg,
            alt_m: 0.0,
        }),
        house_system: system,
        points: points.iter().map(to_record).collect(),
    }
}

/// Cross-aspect records from `(from, to, hit)` triples.
fn cross_records(
    hits: &[(
        &'static str,
        &'static str,
        oxiephemeris_astro::aspects::AspectHit,
    )],
) -> Vec<m::CrossAspectRecord> {
    hits.iter()
        .map(|(from, to, hit)| m::CrossAspectRecord {
            from_name: (*from).to_owned(),
            to_name: (*to).to_owned(),
            kind: hit.aspect.kind,
            offset_deg: hit.offset_rad * RAD2DEG,
            applying: hit.applying,
        })
        .collect()
}

type HouseSystem = oxiephemeris_astro::houses::HouseSystem;

/// Computes a synastry comparison between two natal charts.
///
/// # Errors
///
/// Propagates epoch-resolution, apparent-place, and house-geometry errors.
pub fn synastry(
    de: &DeFile,
    a: &PersonInput,
    b: &PersonInput,
    system: HouseSystem,
    dut1_s: f64,
) -> Result<m::ChartComparison, ChartError> {
    let (epoch_a, points_a) = natal_side(de, a, system, dut1_s)?;
    let (epoch_b, points_b) = natal_side(de, b, system, dut1_s)?;
    Ok(m::ChartComparison {
        kind: m::ComparisonKind::Synastry,
        chart_a: side(
            m::ChartKind::Natal,
            Some(epoch_utc_iso8601(epoch_a.jd_utc)?),
            Some(epoch_a.jd_tt.value()),
            Some((a.lat_deg, a.lon_deg)),
            system,
            &points_a,
        ),
        chart_b: side(
            m::ChartKind::Natal,
            Some(epoch_utc_iso8601(epoch_b.jd_utc)?),
            Some(epoch_b.jd_tt.value()),
            Some((b.lat_deg, b.lon_deg)),
            system,
            &points_b,
        ),
        cross_aspects: cross_records(&collect_cross_aspects(&points_a, &points_b)),
        elapsed_years: None,
        provenance: provenance(de.de_number()),
    })
}

/// Computes a transit comparison: transiting planets vs a natal chart.
///
/// # Errors
///
/// Propagates epoch-resolution, apparent-place, and house-geometry errors.
pub fn transit(
    de: &DeFile,
    natal: &PersonInput,
    transit_date: &str,
    cal: CalendarKind,
    system: HouseSystem,
    dut1_s: f64,
) -> Result<m::ChartComparison, ChartError> {
    let (natal_epoch, natal_points) = natal_side(de, natal, system, dut1_s)?;
    let transit_epoch = resolve_chart_epoch(transit_date, cal, 0.0, dut1_s)?;
    let transiting = transit_points(de, &transit_epoch)?;
    Ok(m::ChartComparison {
        kind: m::ComparisonKind::Transit,
        chart_a: side(
            m::ChartKind::Transit,
            Some(epoch_utc_iso8601(transit_epoch.jd_utc)?),
            Some(transit_epoch.jd_tt.value()),
            None,
            system,
            &transiting,
        ),
        chart_b: side(
            m::ChartKind::Natal,
            Some(epoch_utc_iso8601(natal_epoch.jd_utc)?),
            Some(natal_epoch.jd_tt.value()),
            Some((natal.lat_deg, natal.lon_deg)),
            system,
            &natal_points,
        ),
        cross_aspects: cross_records(&collect_cross_aspects(&transiting, &natal_points)),
        elapsed_years: None,
        provenance: provenance(de.de_number()),
    })
}

/// Computes a secondary-progression comparison ("a day for a year").
///
/// # Errors
///
/// Propagates epoch-resolution, apparent-place, and house-geometry errors.
pub fn progression(
    de: &DeFile,
    natal: &PersonInput,
    target_date: &str,
    cal: CalendarKind,
    system: HouseSystem,
    dut1_s: f64,
) -> Result<m::ChartComparison, ChartError> {
    let (natal_epoch, natal_points) = natal_side(de, natal, system, dut1_s)?;
    let target_epoch = resolve_chart_epoch(target_date, cal, natal.lon_deg, dut1_s)?;
    let elapsed_days = target_epoch.jd_tt.value() - natal_epoch.jd_tt.value();
    let elapsed_years = elapsed_days / TROPICAL_YEAR_DAYS;
    let progressed_jd: JulianDate = natal_epoch
        .jd_tt
        .add_seconds(elapsed_years * SECONDS_PER_DAY);
    let progressed = planet_points_at(de, progressed_jd)?;
    Ok(m::ChartComparison {
        kind: m::ComparisonKind::Progression,
        chart_a: side(
            m::ChartKind::Progressed,
            None,
            Some(progressed_jd.value()),
            Some((natal.lat_deg, natal.lon_deg)),
            system,
            &progressed,
        ),
        chart_b: side(
            m::ChartKind::Natal,
            Some(epoch_utc_iso8601(natal_epoch.jd_utc)?),
            Some(natal_epoch.jd_tt.value()),
            Some((natal.lat_deg, natal.lon_deg)),
            system,
            &natal_points,
        ),
        cross_aspects: cross_records(&collect_cross_aspects(&progressed, &natal_points)),
        elapsed_years: Some(elapsed_years),
        provenance: provenance(de.de_number()),
    })
}

/// Computes a midpoint composite chart of two natal charts.
///
/// # Errors
///
/// Propagates epoch-resolution, apparent-place, and house-geometry errors.
///
/// The two source charts' IRIs are minted under `base_iri` (or the
/// published default), so the composite's `prov:wasDerivedFrom` links and
/// its own derived IRI stay consistent with that base. Serialize the
/// result with the **same** `base_iri`.
pub fn composite(
    de: &DeFile,
    a: &PersonInput,
    b: &PersonInput,
    system: HouseSystem,
    dut1_s: f64,
    base_iri: Option<&str>,
) -> Result<m::CompositeChartResource, ChartError> {
    let (epoch_a, points_a) = natal_side(de, a, system, dut1_s)?;
    let (epoch_b, points_b) = natal_side(de, b, system, dut1_s)?;
    // `natal_points` returns the same points in the same order for both,
    // so zipping pairs them by identity.
    let composite: Vec<ChartPoint> = points_a
        .iter()
        .zip(points_b.iter())
        .map(|(pa, pb)| ChartPoint {
            name: pa.name,
            lon_rad: midpoint(pa.lon_rad, pb.lon_rad),
            speed_rad_per_day: f64::midpoint(pa.speed_rad_per_day, pb.speed_rad_per_day),
        })
        .collect();
    let minter = match base_iri {
        Some(base) => IriMinter::new(base)?,
        None => IriMinter::default(),
    };
    let source_a = source_chart_iri(&minter, &epoch_a, a, system);
    let source_b = source_chart_iri(&minter, &epoch_b, b, system);
    Ok(m::CompositeChartResource {
        side: side(
            m::ChartKind::Composite,
            None,
            None,
            None,
            system,
            &composite,
        ),
        aspects: cross_records(&collect_internal_aspects(&composite)),
        source_charts: vec![source_a, source_b],
        provenance: provenance(de.de_number()),
    })
}

/// Resolves a person's epoch and twelve natal points.
fn natal_side(
    de: &DeFile,
    person: &PersonInput,
    system: HouseSystem,
    dut1_s: f64,
) -> Result<(ChartEpoch, Vec<ChartPoint>), ChartError> {
    let epoch = resolve_chart_epoch(&person.date, person.cal, person.lon_deg, dut1_s)?;
    let points = natal_points(de, &epoch, person.lat_deg.to_radians(), system)?;
    Ok((epoch, points))
}

/// The natal chart IRI of a composite's source, minted from its
/// `ChartKey` — the same IRI the standalone natal chart of that birth
/// carries, so a triple store merges them.
fn source_chart_iri(
    minter: &IriMinter,
    epoch: &ChartEpoch,
    person: &PersonInput,
    system: HouseSystem,
) -> String {
    let key = ChartKey {
        kind: "natal",
        jd_tt: epoch.jd_tt.value(),
        lat_deg: person.lat_deg,
        lon_deg: person.lon_deg,
        house_system: system.name(),
        zodiac: "tropical",
    };
    minter.chart(&key).as_str().to_owned()
}
