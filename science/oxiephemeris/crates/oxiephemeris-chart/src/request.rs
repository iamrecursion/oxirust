//! Request types for every chart the facade computes, plus the
//! string-to-enum parsing the Python and WASM bindings need.
//!
//! The CLI builds the typed [`NatalRequest`] etc. directly from its
//! `clap` value-enums. The bindings instead deserialize a JSON/kwargs
//! **spec** ([`NatalSpec`] …) whose fields are strings, then call
//! `into_request()` to parse and validate — so `"placidus"`,
//! `"lahiri"`, `"traditional"` are checked in exactly one place.

use serde::Deserialize;

use oxiephemeris_astro::ayanamsha::Ayanamsha;
use oxiephemeris_astro::dignities::RulershipScheme;
use oxiephemeris_astro::houses::HouseSystem;

use crate::calendar::CalendarKind;
use crate::error::ChartError;

/// Parses a house-system name (kebab-case, matching
/// [`HouseSystem::name`]).
///
/// # Errors
///
/// [`ChartError::BadRequest`] for an unknown name.
pub fn parse_house_system(name: &str) -> Result<HouseSystem, ChartError> {
    HouseSystem::ALL
        .into_iter()
        .find(|s| s.name() == name)
        .ok_or_else(|| {
            ChartError::BadRequest(format!(
                "unknown house system '{name}'; expected one of: placidus, koch, \
                 whole-sign, equal, porphyry, regiomontanus, campanus"
            ))
        })
}

/// Parses an optional ayanamsha name. `None` (or `null`) means the
/// tropical zodiac.
///
/// # Errors
///
/// [`ChartError::BadRequest`] for an unknown name.
pub fn parse_ayanamsha(name: Option<&str>) -> Result<Option<Ayanamsha>, ChartError> {
    match name {
        None => Ok(None),
        Some(name) => Ayanamsha::ALL_NAMED
            .into_iter()
            .find(|a| a.name() == Some(name))
            .map(Some)
            .ok_or_else(|| {
                ChartError::BadRequest(format!(
                    "unknown ayanamsha '{name}'; expected one of: fagan-bradley, \
                     lahiri, krishnamurti, raman, j2000-zero (or omit for tropical)"
                ))
            }),
    }
}

/// The stable name of a [`RulershipScheme`].
#[must_use]
pub fn rulership_name(scheme: RulershipScheme) -> &'static str {
    // `RulershipScheme` is `#[non_exhaustive]`; anything that is not the
    // modern scheme is reported as traditional (its default).
    match scheme {
        RulershipScheme::Modern => "modern",
        _ => "traditional",
    }
}

/// Parses a rulership-scheme name.
///
/// # Errors
///
/// [`ChartError::BadRequest`] for an unknown name.
pub fn parse_rulership(name: &str) -> Result<RulershipScheme, ChartError> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "traditional" => Ok(RulershipScheme::Traditional),
        "modern" => Ok(RulershipScheme::Modern),
        other => Err(ChartError::BadRequest(format!(
            "unknown rulership scheme '{other}'; expected 'traditional' or 'modern'"
        ))),
    }
}

/// A fully-typed natal-chart request.
#[derive(Debug, Clone)]
pub struct NatalRequest {
    /// ISO 8601 UTC date/time string.
    pub date: String,
    /// Calendar the date is written in.
    pub cal: CalendarKind,
    /// Geodetic latitude, degrees north.
    pub lat_deg: f64,
    /// Geodetic longitude, degrees east.
    pub lon_deg: f64,
    /// Height above the ellipsoid, metres (recorded, not used in the
    /// geocentric cusp math).
    pub alt_m: f64,
    /// `UT1 - UTC`, seconds.
    pub dut1_s: f64,
    /// House system.
    pub system: HouseSystem,
    /// Sidereal ayanamsha, or `None` for tropical.
    pub sidereal: Option<Ayanamsha>,
    /// Rulership scheme for the essential dignities.
    pub rulership: RulershipScheme,
}

/// Common default: Placidus, tropical, traditional, `dut1 = 0`, sea level.
fn default_alt() -> f64 {
    0.0
}
fn default_system() -> String {
    "placidus".to_owned()
}
fn default_rulership() -> String {
    "traditional".to_owned()
}
fn default_cal() -> String {
    "gregorian".to_owned()
}

/// A binding-friendly natal request whose enum fields are strings, ready
/// for JSON/kwargs deserialization.
#[derive(Debug, Clone, Deserialize)]
pub struct NatalSpec {
    /// ISO 8601 UTC date/time.
    pub date: String,
    /// Latitude, degrees north.
    pub lat: f64,
    /// Longitude, degrees east.
    pub lon: f64,
    /// Height above the ellipsoid, metres.
    #[serde(default = "default_alt")]
    pub alt: f64,
    /// `UT1 - UTC`, seconds.
    #[serde(default)]
    pub dut1: f64,
    /// Calendar name.
    #[serde(default = "default_cal")]
    pub cal: String,
    /// House-system name.
    #[serde(default = "default_system")]
    pub system: String,
    /// Ayanamsha name, or `null` for tropical.
    #[serde(default)]
    pub sidereal: Option<String>,
    /// Rulership scheme name.
    #[serde(default = "default_rulership")]
    pub rulership: String,
}

impl NatalSpec {
    /// Parses and validates the spec into a typed [`NatalRequest`].
    ///
    /// # Errors
    ///
    /// [`ChartError::BadRequest`] for an unknown calendar/system/
    /// ayanamsha/rulership name.
    pub fn into_request(self) -> Result<NatalRequest, ChartError> {
        Ok(NatalRequest {
            date: self.date,
            cal: CalendarKind::parse(&self.cal)?,
            lat_deg: self.lat,
            lon_deg: self.lon,
            alt_m: self.alt,
            dut1_s: self.dut1,
            system: parse_house_system(&self.system)?,
            sidereal: parse_ayanamsha(self.sidereal.as_deref())?,
            rulership: parse_rulership(&self.rulership)?,
        })
    }
}

/// A synastry (or composite) request: two people, one house system.
#[derive(Debug, Clone, Deserialize)]
pub struct PairSpec {
    /// Person A's date, latitude, longitude.
    pub a: PersonSpec,
    /// Person B's date, latitude, longitude.
    pub b: PersonSpec,
    /// House-system name for both charts.
    #[serde(default = "default_system")]
    pub system: String,
    /// `UT1 - UTC`, seconds, applied to both.
    #[serde(default)]
    pub dut1: f64,
    /// Calendar for both dates.
    #[serde(default = "default_cal")]
    pub cal: String,
}

/// One person in a [`PairSpec`].
#[derive(Debug, Clone, Deserialize)]
pub struct PersonSpec {
    /// ISO 8601 UTC date/time.
    pub date: String,
    /// Latitude, degrees north.
    pub lat: f64,
    /// Longitude, degrees east.
    pub lon: f64,
}

/// A transit request: a natal chart and a transiting instant.
#[derive(Debug, Clone, Deserialize)]
pub struct TransitSpec {
    /// The natal chart.
    pub natal: PersonSpec,
    /// The transiting instant (ISO 8601 UTC).
    pub transit: String,
    /// House-system name for the natal angles.
    #[serde(default = "default_system")]
    pub system: String,
    /// `UT1 - UTC`, seconds.
    #[serde(default)]
    pub dut1: f64,
    /// Calendar for both dates.
    #[serde(default = "default_cal")]
    pub cal: String,
}

/// A secondary-progression request: a natal chart and a target date.
#[derive(Debug, Clone, Deserialize)]
pub struct ProgressSpec {
    /// The natal chart.
    pub natal: PersonSpec,
    /// The date to progress to (ISO 8601 UTC).
    pub target: String,
    /// House-system name for the natal angles.
    #[serde(default = "default_system")]
    pub system: String,
    /// `UT1 - UTC`, seconds.
    #[serde(default)]
    pub dut1: f64,
    /// Calendar for both dates.
    #[serde(default = "default_cal")]
    pub cal: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names() {
        assert_eq!(parse_house_system("koch").ok(), Some(HouseSystem::Koch));
        assert!(parse_house_system("bogus").is_err());
        assert!(parse_ayanamsha(None).ok().flatten().is_none());
        assert_eq!(
            parse_ayanamsha(Some("lahiri")).ok().flatten(),
            Some(Ayanamsha::Lahiri)
        );
        assert!(parse_ayanamsha(Some("nope")).is_err());
        assert_eq!(
            parse_rulership("modern").ok(),
            Some(RulershipScheme::Modern)
        );
        assert_eq!(rulership_name(RulershipScheme::Traditional), "traditional");
    }

    #[test]
    fn natal_spec_defaults_and_resolves() {
        let json = r#"{"date":"1970-01-01T00:00:00Z","lat":51.4779,"lon":0.0}"#;
        let Ok(spec) = serde_json::from_str::<NatalSpec>(json) else {
            panic!("spec must deserialize");
        };
        let Ok(req) = spec.into_request() else {
            panic!("spec must resolve");
        };
        assert_eq!(req.system, HouseSystem::Placidus);
        assert_eq!(req.rulership, RulershipScheme::Traditional);
        assert!(req.sidereal.is_none());
        assert_eq!(req.cal, CalendarKind::Gregorian);
        assert!((req.alt_m - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn natal_spec_rejects_bad_enum() {
        let json = r#"{"date":"x","lat":0,"lon":0,"system":"martian"}"#;
        let Ok(spec) = serde_json::from_str::<NatalSpec>(json) else {
            panic!("spec must deserialize");
        };
        assert!(spec.into_request().is_err());
    }
}
