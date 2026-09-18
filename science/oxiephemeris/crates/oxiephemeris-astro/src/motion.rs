//! Apparent direction of motion in ecliptic longitude: direct,
//! retrograde, or (near-)stationary.
//!
//! # Model
//!
//! As seen from Earth, a planet's geocentric ecliptic longitude usually
//! increases with time (**direct** motion, positive daily speed). Near
//! opposition (for the outer planets) or inferior conjunction (for
//! Mercury and Venus) the apparent longitude briefly *decreases*
//! (**retrograde** motion, negative daily speed). At the two turning
//! points the speed passes through zero: those instants are
//! **stationary**.
//!
//! Classification here is purely on the sign of the supplied daily
//! longitude speed, with a small symmetric dead-band around zero to
//! label the turning points as stationary rather than reporting a
//! meaningless direct/retrograde flip that would chatter across the
//! exact station. The dead-band half-width is a caller-supplied
//! tolerance in radians per day; [`MotionState::of`] uses
//! [`STATION_TOL_RAD_PER_DAY`] by default via [`MotionState::classify`].
//!
//! The Sun and Moon (geocentric) are never retrograde; the mean lunar
//! node moves retrograde on average. Nothing in this module special-
//! cases any body — only the sign of the passed speed matters, exactly
//! as in [`crate::aspects`]'s applying/separating rule.
//!
//! # Clean-room provenance
//!
//! "Retrograde = decreasing apparent longitude" is the plain kinematic
//! definition; no ephemeris source was consulted.

/// Default station dead-band half-width, radians per day.
///
/// `1e-4 rad/day` is about `0.006 deg/day` (`20.6"/day`). A true planet
/// crosses this band for only a short interval around each station, so
/// it flags genuine turning points without swallowing ordinary slow
/// motion (e.g. Pluto's `~0.01-0.04 deg/day` direct/retrograde speed is
/// several times this band). It is this crate's own convention, not a
/// published constant.
pub const STATION_TOL_RAD_PER_DAY: f64 = 1.0e-4;

/// The apparent direction of motion in longitude.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotionState {
    /// Longitude increasing: positive daily speed beyond the dead-band.
    Direct,
    /// Longitude decreasing: negative daily speed beyond the dead-band.
    Retrograde,
    /// Speed within the station dead-band: a turning point.
    Stationary,
}

impl MotionState {
    /// All three motion states, in canonical order.
    pub const ALL: [Self; 3] = [Self::Direct, Self::Retrograde, Self::Stationary];

    /// Classifies motion from a daily longitude speed and an explicit
    /// station dead-band half-width, both in radians per day.
    ///
    /// `station_tol` is treated as a magnitude; its absolute value is
    /// used, so a negative tolerance behaves like its positive twin. A
    /// non-finite `lon_speed_rad_per_day` yields [`MotionState::Stationary`]
    /// (neither comparison below is true for NaN).
    #[must_use]
    pub fn classify(lon_speed_rad_per_day: f64, station_tol: f64) -> Self {
        let tol = libm::fabs(station_tol);
        if lon_speed_rad_per_day > tol {
            Self::Direct
        } else if lon_speed_rad_per_day < -tol {
            Self::Retrograde
        } else {
            Self::Stationary
        }
    }

    /// Classifies motion from a daily longitude speed using the default
    /// [`STATION_TOL_RAD_PER_DAY`] dead-band.
    #[must_use]
    pub fn of(lon_speed_rad_per_day: f64) -> Self {
        Self::classify(lon_speed_rad_per_day, STATION_TOL_RAD_PER_DAY)
    }

    /// `true` for [`MotionState::Retrograde`].
    #[must_use]
    pub const fn is_retrograde(self) -> bool {
        matches!(self, Self::Retrograde)
    }

    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Retrograde => "retrograde",
            Self::Stationary => "stationary",
        }
    }

    /// A one-character tag suited to compact tables: `""` for direct,
    /// `"R"` for retrograde, `"S"` for stationary.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Direct => "",
            Self::Retrograde => "R",
            Self::Stationary => "S",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_and_retrograde_by_sign() {
        assert_eq!(MotionState::of(0.5), MotionState::Direct);
        assert_eq!(MotionState::of(-0.5), MotionState::Retrograde);
        assert!(MotionState::of(-0.5).is_retrograde());
        assert!(!MotionState::of(0.5).is_retrograde());
    }

    #[test]
    fn near_zero_is_stationary() {
        assert_eq!(MotionState::of(0.0), MotionState::Stationary);
        assert_eq!(
            MotionState::of(STATION_TOL_RAD_PER_DAY * 0.5),
            MotionState::Stationary
        );
        assert_eq!(
            MotionState::of(-STATION_TOL_RAD_PER_DAY * 0.5),
            MotionState::Stationary
        );
        // Just outside the band resolves to a direction.
        assert_eq!(
            MotionState::of(STATION_TOL_RAD_PER_DAY * 2.0),
            MotionState::Direct
        );
        assert_eq!(
            MotionState::of(-STATION_TOL_RAD_PER_DAY * 2.0),
            MotionState::Retrograde
        );
    }

    #[test]
    fn negative_tolerance_behaves_like_its_magnitude() {
        assert_eq!(
            MotionState::classify(0.05, -0.1),
            MotionState::classify(0.05, 0.1)
        );
    }

    #[test]
    fn nan_speed_is_stationary() {
        assert_eq!(MotionState::of(f64::NAN), MotionState::Stationary);
    }

    #[test]
    fn tags_and_names() {
        assert_eq!(MotionState::Direct.tag(), "");
        assert_eq!(MotionState::Retrograde.tag(), "R");
        assert_eq!(MotionState::Stationary.tag(), "S");
        assert_eq!(MotionState::Retrograde.name(), "retrograde");
    }
}
