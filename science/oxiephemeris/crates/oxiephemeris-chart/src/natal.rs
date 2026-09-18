//! Assembling a full natal chart into a
//! [`oxiephemeris_rdf::model::ChartResource`].
//!
//! This is the compute core the CLI's `chart` command, the Python
//! `natal`, and the WASM `natal_json`/`natal_turtle` all share. Every
//! derived quantity — sign, house, retrograde, declination, dignity,
//! Lot, distribution — is produced here exactly once.
//!
//! # Zodiac conventions
//!
//! Body/angle/cusp/node/lot longitudes are the **displayed** longitude:
//! tropical, or sidereal-shifted when a `sidereal` ayanamsha is set, so a
//! sidereal chart reports sidereal signs. Essential **dignities** are a
//! tropical technique and are always scored from the tropical longitude.
//! Aspects are computed from tropical longitudes/speeds (a common
//! ayanamsha cancels in every pairwise separation).

use oxiephemeris_astro::aspects::{find_aspect, AspectHit, OrbPolicy};
use oxiephemeris_astro::ayanamsha::{ayanamsha_rad, Ayanamsha};
use oxiephemeris_astro::declination::{declination, is_out_of_bounds};
use oxiephemeris_astro::dignities::{
    distribution, essential_dignity, EssentialDignity, Planet, RulershipScheme,
};
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::nodes::{mean_apogee, mean_node, true_node_of_date};
use oxiephemeris_astro::parts::{part_of_fortune, part_of_spirit, sect};
use oxiephemeris_astro::zodiac::{Sign, SignPosition};
use oxiephemeris_core::angle::{normalize_0_two_pi, RAD2DEG};
use oxiephemeris_de::DeFile;
use oxiephemeris_rdf::model as m;

use crate::epoch::{
    resolve_chart_epoch, sidereal_offset_true_equinox_rad,
    sidereal_offset_true_equinox_rate_rad_per_day,
};
use crate::error::ChartError;
use crate::houses::{compute_houses, HousesResult};
use crate::positions::{compute_bodies, BodyPlace};
use crate::request::{rulership_name, NatalRequest};

/// Maps a body display name to its dignity [`Planet`], for the ten planets.
#[must_use]
pub fn planet_of(name: &str) -> Option<Planet> {
    Planet::ALL.into_iter().find(|p| p.name() == name)
}

/// The dignity tiers an [`EssentialDignity`] holds, strongest first.
#[must_use]
pub fn dignity_tiers(d: &EssentialDignity) -> Vec<m::DignityTier> {
    let mut tiers = Vec::new();
    if d.domicile {
        tiers.push(m::DignityTier::Domicile);
    }
    if d.exaltation {
        tiers.push(m::DignityTier::Exaltation);
    }
    if d.triplicity {
        tiers.push(m::DignityTier::Triplicity);
    }
    if d.term {
        tiers.push(m::DignityTier::Term);
    }
    if d.face {
        tiers.push(m::DignityTier::Face);
    }
    if d.detriment {
        tiers.push(m::DignityTier::Detriment);
    }
    if d.fall {
        tiers.push(m::DignityTier::Fall);
    }
    if d.peregrine {
        tiers.push(m::DignityTier::Peregrine);
    }
    tiers
}

/// The house (`1..=12`) a longitude falls in, given the twelve cusp
/// longitudes (radians) in house order.
#[must_use]
fn house_of(lon_rad: f64, cusps_rad: &[f64; 12]) -> usize {
    for i in 0..12 {
        let a = cusps_rad[i];
        let b = cusps_rad[(i + 1) % 12];
        let span = normalize_0_two_pi(b - a);
        let offset = normalize_0_two_pi(lon_rad - a);
        if offset < span {
            return i + 1;
        }
    }
    1
}

/// Shifts a mean-equinox-of-date longitude to the true equinox and, if
/// `sidereal`, on to the sidereal zodiac.
fn to_apparent_sidereal(
    mean_equinox_lon_rad: f64,
    dpsi_rad: f64,
    sidereal: Option<Ayanamsha>,
    t_tt: f64,
) -> f64 {
    let true_equinox_lon = normalize_0_two_pi(mean_equinox_lon_rad + dpsi_rad);
    match sidereal {
        Some(kind) => normalize_0_two_pi(
            true_equinox_lon - sidereal_offset_true_equinox_rad(kind, t_tt, dpsi_rad),
        ),
        None => true_equinox_lon,
    }
}

/// Applies the sidereal shift (if any) to a body's longitude and speed,
/// for output only.
fn to_output_body(
    place: &BodyPlace,
    sidereal: Option<Ayanamsha>,
    t_tt: f64,
    dpsi_rad: f64,
) -> (f64, f64) {
    match sidereal {
        Some(kind) => {
            let offset = sidereal_offset_true_equinox_rad(kind, t_tt, dpsi_rad);
            let lon_rad = normalize_0_two_pi(place.lon_rad - offset);
            let rate = sidereal_offset_true_equinox_rate_rad_per_day(kind, t_tt);
            (lon_rad, place.lon_speed_rad_per_day - rate)
        }
        None => (
            normalize_0_two_pi(place.lon_rad),
            place.lon_speed_rad_per_day,
        ),
    }
}

/// One body's fully-resolved placement (internal to this module).
struct Placement {
    name: &'static str,
    lon_out_rad: f64,
    lon_speed_rad_per_day: f64,
    lon_tropical_rad: f64,
    lat_rad: f64,
    lat_speed_rad_per_day: f64,
    distance_au: f64,
    light_time_days: f64,
    house: usize,
    declination_rad: f64,
}

/// Computes a full natal chart from DE ephemeris bytes and a request.
///
/// # Errors
///
/// Propagates epoch-resolution, apparent-place, house-geometry, and
/// lunar-node errors.
#[allow(clippy::too_many_lines)] // one linear pass over the chart's sections
pub fn natal_chart(de: &DeFile, request: &NatalRequest) -> Result<m::ChartResource, ChartError> {
    let epoch = resolve_chart_epoch(&request.date, request.cal, request.lon_deg, request.dut1_s)?;
    let phi_rad = request.lat_deg.to_radians();
    let sidereal = request.sidereal;
    let ayanamsha_deg = sidereal.map(|kind| ayanamsha_rad(kind, epoch.t_tt) * RAD2DEG);

    let places = compute_bodies(de, epoch.jd_tt)?;
    let houses = compute_houses(request.system, &epoch, phi_rad, sidereal)?;

    let mut cusps_rad = [0.0_f64; 12];
    cusps_rad.copy_from_slice(&houses.cusps_rad);

    let mut placements = Vec::with_capacity(places.len());
    for place in &places {
        let (lon_out_rad, lon_speed_rad_per_day) =
            to_output_body(place, sidereal, epoch.t_tt, epoch.dpsi_rad);
        placements.push(Placement {
            name: place.name,
            lon_out_rad,
            lon_speed_rad_per_day,
            lon_tropical_rad: normalize_0_two_pi(place.lon_rad),
            lat_rad: place.lat_rad,
            lat_speed_rad_per_day: place.lat_speed_rad_per_day,
            distance_au: place.distance_au,
            light_time_days: place.light_time_days,
            house: house_of(lon_out_rad, &cusps_rad),
            declination_rad: declination(place.lon_rad, place.lat_rad, epoch.eps_true_rad),
        });
    }

    let asc_out = houses.ascendant_rad;
    let find = |name: &str| {
        placements
            .iter()
            .find(|p| p.name == name)
            .map_or(0.0, |p| p.lon_out_rad)
    };
    let chart_sect = sect(find("Sun"), asc_out);
    let fortune = part_of_fortune(asc_out, find("Sun"), find("Moon"), chart_sect);
    let spirit = part_of_spirit(asc_out, find("Sun"), find("Moon"), chart_sect);

    let dpsi = epoch.dpsi_rad;
    let mean_node_lon = to_apparent_sidereal(mean_node(epoch.t_tt), dpsi, sidereal, epoch.t_tt);
    let mean_apogee_lon = to_apparent_sidereal(mean_apogee(epoch.t_tt), dpsi, sidereal, epoch.t_tt);
    let osc = true_node_of_date(de, epoch.jd_tt)?;
    let true_node_lon = to_apparent_sidereal(osc.node_lon_rad, dpsi, sidereal, epoch.t_tt);
    let true_apogee_lon = to_apparent_sidereal(osc.apogee_lon_rad, dpsi, sidereal, epoch.t_tt);

    let policy = OrbPolicy::default();
    let mut aspects: Vec<(&'static str, &'static str, AspectHit)> = Vec::new();
    for i in 0..places.len() {
        for j in (i + 1)..places.len() {
            let (a, b) = (&places[i], &places[j]);
            if let Some(hit) = find_aspect(
                a.lon_rad,
                a.lon_speed_rad_per_day,
                b.lon_rad,
                b.lon_speed_rad_per_day,
                &policy,
            ) {
                aspects.push((a.name, b.name, hit));
            }
        }
    }

    let scheme: RulershipScheme = request.rulership;
    let signs: Vec<Sign> = placements
        .iter()
        .map(|p| Sign::of(p.lon_tropical_rad))
        .collect();
    let dist = distribution(&signs);

    let zodiac = match (sidereal, ayanamsha_deg) {
        (Some(ayanamsha), Some(degrees)) => m::Zodiac::Sidereal { ayanamsha, degrees },
        _ => m::Zodiac::Tropical,
    };

    let bodies = placements
        .iter()
        .map(|p| {
            let sp = SignPosition::of(p.lon_out_rad);
            m::BodyPlacement {
                name: p.name.to_owned(),
                planet: planet_of(p.name),
                lon_deg: p.lon_out_rad * RAD2DEG,
                lat_deg: p.lat_rad * RAD2DEG,
                lon_speed_deg_per_day: p.lon_speed_rad_per_day * RAD2DEG,
                lat_speed_deg_per_day: p.lat_speed_rad_per_day * RAD2DEG,
                distance_au: p.distance_au,
                light_time_days: p.light_time_days,
                sign: sp.sign,
                degrees_in_sign: sp.degrees_in_sign,
                house: p.house,
                declination_deg: p.declination_rad * RAD2DEG,
                motion: MotionState::of(p.lon_speed_rad_per_day),
                out_of_bounds: is_out_of_bounds(p.declination_rad, epoch.eps_true_rad),
            }
        })
        .collect();

    let cusps = houses
        .cusps_rad
        .iter()
        .enumerate()
        .map(|(index, rad)| m::CuspRecord {
            number: index + 1,
            lon_deg: rad * RAD2DEG,
        })
        .collect();

    let angles = angle_records(&houses);
    let nodes = node_records(
        mean_node_lon,
        mean_apogee_lon,
        true_node_lon,
        true_apogee_lon,
    );
    let lots = vec![
        m::LotRecord {
            kind: m::LotKind::Fortune,
            lon_deg: fortune * RAD2DEG,
        },
        m::LotRecord {
            kind: m::LotKind::Spirit,
            lon_deg: spirit * RAD2DEG,
        },
    ];

    let aspect_records = aspects
        .iter()
        .map(|(body1, body2, hit)| m::AspectRecord {
            body1: (*body1).to_owned(),
            body2: (*body2).to_owned(),
            kind: hit.aspect.kind,
            exact_angle_deg: hit.aspect.exact_angle_rad * RAD2DEG,
            offset_deg: hit.offset_rad * RAD2DEG,
            applying: hit.applying,
        })
        .collect();

    let dignity_records = placements
        .iter()
        .filter_map(|p| {
            planet_of(p.name).map(|planet| {
                let d: EssentialDignity =
                    essential_dignity(planet, p.lon_tropical_rad, chart_sect, scheme);
                m::DignityRecord {
                    planet,
                    tiers: dignity_tiers(&d),
                    score: d.score,
                }
            })
        })
        .collect();

    Ok(m::ChartResource {
        kind: m::ChartKind::Natal,
        epoch_utc: crate::epoch::epoch_utc_iso8601(epoch.jd_utc)?,
        jd_tt: epoch.jd_tt.value(),
        observer: Some(m::Observer {
            lat_deg: request.lat_deg,
            lon_deg: request.lon_deg,
            alt_m: request.alt_m,
        }),
        house_system: request.system,
        zodiac,
        rulership: rulership_name(scheme).to_owned(),
        sect: chart_sect,
        bodies,
        cusps,
        angles,
        nodes,
        lots,
        aspects: aspect_records,
        dignities: dignity_records,
        distribution: m::DistributionRecord {
            elements: dist.elements,
            modalities: dist.modalities,
        },
        elapsed_years: None,
        provenance: provenance(de.de_number()),
    })
}

/// The four chart angles as records.
fn angle_records(houses: &HousesResult) -> Vec<m::AngleRecord> {
    vec![
        m::AngleRecord {
            kind: m::AngleKind::Ascendant,
            lon_deg: houses.ascendant_rad * RAD2DEG,
        },
        m::AngleRecord {
            kind: m::AngleKind::Midheaven,
            lon_deg: houses.mc_rad * RAD2DEG,
        },
        m::AngleRecord {
            kind: m::AngleKind::Vertex,
            lon_deg: houses.vertex_rad * RAD2DEG,
        },
        m::AngleRecord {
            kind: m::AngleKind::EastPoint,
            lon_deg: houses.east_point_rad * RAD2DEG,
        },
    ]
}

/// The four lunar nodes/apogees as records.
fn node_records(
    mean_node: f64,
    mean_apogee: f64,
    true_node: f64,
    true_apogee: f64,
) -> Vec<m::NodeRecord> {
    vec![
        m::NodeRecord {
            kind: m::NodeKind::MeanNode,
            lon_deg: mean_node * RAD2DEG,
        },
        m::NodeRecord {
            kind: m::NodeKind::MeanApogee,
            lon_deg: mean_apogee * RAD2DEG,
        },
        m::NodeRecord {
            kind: m::NodeKind::TrueNode,
            lon_deg: true_node * RAD2DEG,
        },
        m::NodeRecord {
            kind: m::NodeKind::TrueApogee,
            lon_deg: true_apogee * RAD2DEG,
        },
    ]
}

/// The PROV-O software agent for a chart computed from `de_number`.
pub(crate) fn provenance(de_number: i32) -> m::Provenance {
    m::Provenance {
        software_name: "oxiephemeris".to_owned(),
        software_version: env!("CARGO_PKG_VERSION").to_owned(),
        ephemeris_label: format!("DE{de_number}"),
    }
}
