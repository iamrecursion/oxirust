//! Human-readable formatting shared by the chart-family subcommands:
//! sign/degree notation, house occupancy, and the retrograde tag.
//!
//! These helpers turn the raw radian longitudes produced by the
//! `oxiephemeris-chart` facade into the conventional astrological presentation
//! (e.g. `25°30'33" Scorpio  [R]  (5th house)`) without duplicating the
//! decomposition logic across `chart`, `transit`, `progress`, and
//! `composite`.

use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::zodiac::SignPosition;

/// Formats an ecliptic longitude (radians) as `dd°mm'ss" Sign`, e.g.
/// `25°30'33" Scorpio`.
#[must_use]
// `arcsec` is in `[0, 60)`, so the display cast neither truncates
// meaningfully nor loses a sign.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn format_sign(lon_rad: f64) -> String {
    let pos = SignPosition::of(lon_rad);
    let (deg, arcmin, arcsec) = pos.dms();
    format!(
        "{deg:2}\u{00B0}{arcmin:02}'{:02}\" {}",
        arcsec as u32,
        pos.sign.name()
    )
}

/// The retrograde tag for a daily longitude speed: `"R"` retrograde,
/// `"S"` stationary, empty for direct motion.
#[must_use]
pub fn retrograde_tag(lon_speed_rad_per_day: f64) -> &'static str {
    MotionState::of(lon_speed_rad_per_day).tag()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    #[test]
    fn sign_formatting() {
        // 235.509... deg -> 25°30'.. Scorpio.
        let s = format_sign(deg(235.509_050_876));
        assert!(s.contains("Scorpio"), "{s}");
        assert!(s.starts_with("25\u{00B0}30'"), "{s}");
    }

    #[test]
    fn retrograde_tagging() {
        assert_eq!(retrograde_tag(1.0), "");
        assert_eq!(retrograde_tag(-1.0), "R");
        assert_eq!(retrograde_tag(0.0), "S");
    }
}
