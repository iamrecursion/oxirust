//! Shared building blocks for the two-chart / time subcommands
//! (`synastry`, `transit`, `progress`, `composite`).
//!
//! The [`ChartPoint`] currency and the point-building helpers
//! ([`natal_points`], [`transit_points`], [`planet_points_at`]) live once in
//! the [`oxiephemeris_chart::points`] facade and are re-exported here; this
//! module keeps only the CLI-specific text (`print_*`) and JSON (`*_json`,
//! `PointJson`, `CrossAspectJson`) renderings.
//!
//! These commands work in the **tropical** zodiac only. Cross-aspects are
//! sidereal-invariant (a common ayanamsha cancels in every pairwise
//! separation), so no accuracy is lost in the aspect grids; only the
//! displayed sign of each raw position would differ under a sidereal
//! shift, and `oxieph chart` already offers full sidereal output for that
//! need.

use oxiephemeris_astro::aspects::OrbPolicy;
use oxiephemeris_astro::synastry::for_each_cross_aspect;
use oxiephemeris_astro::zodiac::SignPosition;
use oxiephemeris_core::angle::RAD2DEG;
use serde::Serialize;

pub use oxiephemeris_chart::points::{natal_points, planet_points_at, transit_points, ChartPoint};
use oxiephemeris_chart::PersonInput;

use crate::chart_render::{format_sign, retrograde_tag};
use crate::convert::CalArg;

/// Builds a facade [`PersonInput`] from CLI-typed birth fields — the input
/// the comparison builders ([`oxiephemeris_chart::synastry`] etc.) consume
/// on the RDF path.
#[must_use]
pub fn person_input(date: &str, cal: CalArg, lat_deg: f64, lon_deg: f64) -> PersonInput {
    PersonInput {
        date: date.to_owned(),
        cal: cal.to_calendar_kind(),
        lat_deg,
        lon_deg,
    }
}

/// Prints a one-line-per-point position table with sign, retrograde tag,
/// and speed.
pub fn print_positions(title: &str, points: &[ChartPoint]) {
    println!("-- {title} --");
    for p in points {
        let tag = retrograde_tag(p.speed_rad_per_day);
        let tag_col = if tag.is_empty() { " " } else { tag };
        println!(
            "{:<8} {:<20} {tag_col:<1}  speed = {:>+8.4} deg/day",
            p.name,
            format_sign(p.lon_rad),
            p.speed_rad_per_day * RAD2DEG,
        );
    }
}

/// Prints every cross-aspect between two point sets, labelled
/// `{a_label} {p_a}  ASPECT  {b_label} {p_b}`.
pub fn print_cross_aspects(a: &[ChartPoint], b: &[ChartPoint], a_label: &str, b_label: &str) {
    let a_pairs: Vec<(f64, f64)> = a.iter().map(|p| (p.lon_rad, p.speed_rad_per_day)).collect();
    let b_pairs: Vec<(f64, f64)> = b.iter().map(|p| (p.lon_rad, p.speed_rad_per_day)).collect();
    let mut any = false;
    for_each_cross_aspect(&a_pairs, &b_pairs, &OrbPolicy::default(), |cross| {
        any = true;
        let pa = a[cross.index_a].name;
        let pb = b[cross.index_b].name;
        let kind = cross.hit.aspect.kind.name();
        let offset_deg = cross.hit.offset_rad * RAD2DEG;
        let applying = if cross.hit.applying {
            "applying"
        } else {
            "separating"
        };
        println!(
            "{a_label} {pa:<8} {kind:<14} {b_label} {pb:<8} \
             offset = {offset_deg:>9.4} deg ({applying})"
        );
    });
    if !any {
        println!("(no cross-aspects within the default orbs)");
    }
}

/// One chart point in JSON form.
#[derive(Debug, Serialize)]
pub struct PointJson {
    /// Point name.
    pub point: &'static str,
    /// Tropical longitude, degrees.
    pub lon_deg: f64,
    /// Zodiac sign name.
    pub sign: &'static str,
    /// Degrees within the sign, `[0, 30)`.
    pub sign_degrees: f64,
    /// Whether the point is retrograde (negative speed).
    pub retrograde: bool,
    /// Daily longitude speed, degrees per day.
    pub speed_deg_per_day: f64,
}

/// One cross-aspect in JSON form.
#[derive(Debug, Serialize)]
pub struct CrossAspectJson {
    /// The `a`-side point name.
    pub a: &'static str,
    /// The `b`-side point name.
    pub b: &'static str,
    /// Aspect kind (stable lower-case name).
    pub aspect: &'static str,
    /// Exact aspect angle, degrees.
    pub exact_angle_deg: f64,
    /// Signed offset from exactness, degrees.
    pub offset_deg: f64,
    /// Whether the aspect is applying.
    pub applying: bool,
}

/// Serializes a point set for `--json`.
#[must_use]
pub fn points_json(points: &[ChartPoint]) -> Vec<PointJson> {
    points
        .iter()
        .map(|p| {
            let sp = SignPosition::of(p.lon_rad);
            PointJson {
                point: p.name,
                lon_deg: p.lon_rad * RAD2DEG,
                sign: sp.sign.name(),
                sign_degrees: sp.degrees_in_sign,
                retrograde: p.speed_rad_per_day < 0.0,
                speed_deg_per_day: p.speed_rad_per_day * RAD2DEG,
            }
        })
        .collect()
}

/// Collects the cross-aspects between two point sets for `--json`.
#[must_use]
pub fn cross_aspects_json(a: &[ChartPoint], b: &[ChartPoint]) -> Vec<CrossAspectJson> {
    let a_pairs: Vec<(f64, f64)> = a.iter().map(|p| (p.lon_rad, p.speed_rad_per_day)).collect();
    let b_pairs: Vec<(f64, f64)> = b.iter().map(|p| (p.lon_rad, p.speed_rad_per_day)).collect();
    let mut out = Vec::new();
    for_each_cross_aspect(&a_pairs, &b_pairs, &OrbPolicy::default(), |cross| {
        out.push(CrossAspectJson {
            a: a[cross.index_a].name,
            b: b[cross.index_b].name,
            aspect: cross.hit.aspect.kind.name(),
            exact_angle_deg: cross.hit.aspect.exact_angle_rad * RAD2DEG,
            offset_deg: cross.hit.offset_rad * RAD2DEG,
            applying: cross.hit.applying,
        });
    });
    out
}
