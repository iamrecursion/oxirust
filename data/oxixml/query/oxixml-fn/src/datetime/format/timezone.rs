//! The `$place` argument of [`fn:format-dateTime`](super::format_date_time),
//! `fn:format-date` and `fn:format-time`.
//!
//! [F&O 3.1 §9.8.4.3](https://www.w3.org/TR/xpath-functions-31/#date-time-examples) gives
//! `$place` two jobs. An Olson timezone name — `America/New_York` — *adjusts* the value to the
//! offset in force in that place, daylight saving included, so a value written `12:00:00Z` is
//! formatted as if it had been written `07:00:00-05:00`. A country code narrows the choice of
//! the *name* that `[ZN]` writes for the timezone the value already has.
//!
//! §9.8.4.6 leaves the set of timezone names an implementation offers up to the implementation.
//! [`CIVIL_ZONES`] is this one: a small table of the civil timezones in widest use, not a copy
//! of the IANA database. A timezone it cannot name is written as a numeric offset instead.

use super::{Components, civil_year_of, days_from_civil};

/// Minutes in a day, the unit the timezone arithmetic below counts in.
const MINUTES_PER_DAY: i64 = 24 * 60;

/// When a zone's clocks go forward, and by how much.
#[derive(Debug, Clone, Copy)]
struct DaylightSaving {
    /// Minutes east of UTC while the saving is in force.
    offset: i32,
    /// What the zone is called while the saving is in force.
    name: &'static str,
    rule: DaylightRule,
}

/// The two daylight-saving calendars the zones below follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DaylightRule {
    /// North America since 2007: from 02:00 local standard time on the second Sunday in March
    /// until 02:00 local daylight time on the first Sunday in November.
    NorthAmerica,
    /// The European Union: from 01:00 UTC on the last Sunday in March until 01:00 UTC on the
    /// last Sunday in October.
    EuropeanUnion,
}

impl DaylightRule {
    /// Whether the saving is in force at `instant`, which counts minutes from 1970-01-01T00:00Z.
    fn covers(self, instant: i64, standard_offset: i32) -> bool {
        // Both windows sit deep inside the year, far from any boundary a timezone offset could
        // push the date across, so the year of the UTC instant is the year of the rule.
        let year = civil_year_of(instant.div_euclid(MINUTES_PER_DAY));
        let (start, end) = match self {
            Self::NorthAmerica => {
                let standard = i64::from(standard_offset);
                (
                    nth_weekday(year, 3, SUNDAY, 2) * MINUTES_PER_DAY + 2 * 60 - standard,
                    // 02:00 daylight time is 01:00 standard time.
                    nth_weekday(year, 11, SUNDAY, 1) * MINUTES_PER_DAY + 60 - standard,
                )
            }
            Self::EuropeanUnion => (
                last_weekday(year, 3, SUNDAY) * MINUTES_PER_DAY + 60,
                last_weekday(year, 10, SUNDAY) * MINUTES_PER_DAY + 60,
            ),
        };
        instant >= start && instant < end
    }
}

/// The ISO weekday number of Sunday.
const SUNDAY: i64 = 7;

/// One civil timezone: what it is called, where it is used, and when its clocks move.
#[derive(Debug)]
pub(super) struct CivilZone {
    /// The Olson identifiers this entry answers to.
    ids: &'static [&'static str],
    /// The ISO 3166 country codes whose civil time this zone keeps.
    countries: &'static [&'static str],
    /// Minutes east of UTC on standard time.
    standard_offset: i32,
    /// What the zone is called on standard time.
    standard_name: &'static str,
    daylight: Option<DaylightSaving>,
}

impl CivilZone {
    /// The offset and name in force at `instant`, or the zone's standard time when the value
    /// being formatted is an `xs:time` and so names no day to place in the year.
    fn at(&self, instant: Option<i64>) -> (i32, &'static str) {
        match (self.daylight, instant) {
            (Some(daylight), Some(instant))
                if daylight.rule.covers(instant, self.standard_offset) =>
            {
                (daylight.offset, daylight.name)
            }
            _ => (self.standard_offset, self.standard_name),
        }
    }
}

/// The timezone-name database this implementation offers `$place`.
///
/// F&O 3.1 §9.8.4.6 leaves the set of timezone names to the implementation — "the choice of
/// timezone name is implementation-defined" — so this is a deliberately small table of the
/// civil timezones in widest use rather than a copy of the IANA database. It is consulted in
/// order: a zone whose country matches `$place` wins over one that merely has the right offset,
/// which is what makes `-05:00` `EST` in January and `CDT` in July for `$place="us"`.
const CIVIL_ZONES: &[CivilZone] = &[
    CivilZone {
        ids: &["UTC", "Etc/UTC", "GMT", "Etc/GMT", "Zulu"],
        countries: &[],
        standard_offset: 0,
        standard_name: "GMT",
        daylight: None,
    },
    CivilZone {
        ids: &["Europe/London"],
        countries: &["gb", "uk"],
        standard_offset: 0,
        standard_name: "GMT",
        daylight: Some(DaylightSaving {
            offset: 60,
            name: "BST",
            rule: DaylightRule::EuropeanUnion,
        }),
    },
    CivilZone {
        ids: &["Europe/Dublin"],
        countries: &["ie"],
        standard_offset: 0,
        standard_name: "GMT",
        daylight: Some(DaylightSaving {
            offset: 60,
            name: "IST",
            rule: DaylightRule::EuropeanUnion,
        }),
    },
    CivilZone {
        ids: &[
            "Europe/Paris",
            "Europe/Berlin",
            "Europe/Madrid",
            "Europe/Rome",
            "Europe/Amsterdam",
            "Europe/Brussels",
            "Europe/Stockholm",
            "Europe/Vienna",
            "Europe/Zurich",
        ],
        countries: &["fr", "de", "es", "it", "nl", "be", "se", "at", "ch"],
        standard_offset: 60,
        standard_name: "CET",
        daylight: Some(DaylightSaving {
            offset: 120,
            name: "CEST",
            rule: DaylightRule::EuropeanUnion,
        }),
    },
    CivilZone {
        ids: &["Europe/Helsinki", "Europe/Athens", "Europe/Kiev"],
        countries: &["fi", "gr", "ua"],
        standard_offset: 120,
        standard_name: "EET",
        daylight: Some(DaylightSaving {
            offset: 180,
            name: "EEST",
            rule: DaylightRule::EuropeanUnion,
        }),
    },
    CivilZone {
        ids: &["Asia/Kolkata", "Asia/Calcutta"],
        countries: &["in"],
        standard_offset: 330,
        standard_name: "IST",
        daylight: None,
    },
    CivilZone {
        ids: &["Asia/Tokyo"],
        countries: &["jp"],
        standard_offset: 540,
        standard_name: "JST",
        daylight: None,
    },
    CivilZone {
        ids: &["America/New_York", "America/Toronto"],
        countries: &["us", "ca"],
        standard_offset: -300,
        standard_name: "EST",
        daylight: Some(DaylightSaving {
            offset: -240,
            name: "EDT",
            rule: DaylightRule::NorthAmerica,
        }),
    },
    CivilZone {
        ids: &["America/Chicago", "America/Winnipeg"],
        countries: &["us", "ca"],
        standard_offset: -360,
        standard_name: "CST",
        daylight: Some(DaylightSaving {
            offset: -300,
            name: "CDT",
            rule: DaylightRule::NorthAmerica,
        }),
    },
    CivilZone {
        ids: &["America/Denver", "America/Edmonton"],
        countries: &["us", "ca"],
        standard_offset: -420,
        standard_name: "MST",
        daylight: Some(DaylightSaving {
            offset: -360,
            name: "MDT",
            rule: DaylightRule::NorthAmerica,
        }),
    },
    CivilZone {
        ids: &["America/Phoenix"],
        countries: &["us"],
        standard_offset: -420,
        standard_name: "MST",
        daylight: None,
    },
    CivilZone {
        ids: &["America/Los_Angeles", "America/Vancouver"],
        countries: &["us", "ca"],
        standard_offset: -480,
        standard_name: "PST",
        daylight: Some(DaylightSaving {
            offset: -420,
            name: "PDT",
            rule: DaylightRule::NorthAmerica,
        }),
    },
    CivilZone {
        ids: &["America/Anchorage"],
        countries: &["us"],
        standard_offset: -540,
        standard_name: "AKST",
        daylight: Some(DaylightSaving {
            offset: -480,
            name: "AKDT",
            rule: DaylightRule::NorthAmerica,
        }),
    },
    CivilZone {
        ids: &["Pacific/Honolulu"],
        countries: &["us"],
        standard_offset: -600,
        standard_name: "HST",
        daylight: None,
    },
];

/// What `fn:format-dateTime`'s fifth argument turned out to name.
#[derive(Debug, Clone, Copy)]
pub(super) enum Place<'a> {
    /// No `$place` was given, or one this database does not hold: the value is not moved and
    /// its timezone is not named.
    Unknown,
    /// An Olson timezone name the database knows. §9.8.4.3: the value *is adjusted* to the
    /// offset in force there.
    Zone(&'static CivilZone),
    /// An ISO 3166 country code. The value stays where it is; the country only narrows the
    /// search for a name.
    Country(&'a str),
}

impl<'a> Place<'a> {
    pub(super) fn parse(place: Option<&'a str>) -> Self {
        let trimmed = place.map(str::trim).unwrap_or_default();
        if trimmed.is_empty() {
            return Self::Unknown;
        }
        // §9.8.4.3 spells the second recognised form of `$place` as "a country code as defined
        // in ISO 3166" — two letters. It is tested *first*, because only the timezone-name form
        // may move the value and a two-letter place must never do that, whatever the IANA
        // database happens to accept as an alias for a zone.
        if trimmed.len() == 2 && trimmed.bytes().all(|byte| byte.is_ascii_alphabetic()) {
            return Self::Country(trimmed);
        }
        CIVIL_ZONES
            .iter()
            .find(|zone| zone.ids.iter().any(|id| id.eq_ignore_ascii_case(trimmed)))
            // Anything else is a place this implementation cannot place.
            .map_or(Self::Unknown, Self::Zone)
    }
}

/// Applies `$place` to the value: §9.8.4.3 adjusts it to an Olson timezone's offset, and
/// §9.8.4.6 uses the place to choose the name `[ZN]` writes.
pub(super) fn apply_place(mut components: Components, place: Place<'_>) -> Components {
    let Some(offset) = components.timezone else {
        // With no timezone there is no instant to place in the year and nothing to name.
        return components;
    };
    let instant = utc_instant(components, offset);
    // An `xs:time` names no day, so a zone's daylight saving cannot be decided for it and its
    // standard time is used (`format-time-025c` expects `EST`, not `EDT`).
    let dated = components.shape.has_date().then_some(instant);
    match place {
        Place::Unknown => components,
        Place::Zone(zone) => {
            let (target, name) = zone.at(dated);
            components = shift(components, instant, target);
            components.zone_name = Some(name);
            components
        }
        Place::Country(country) => {
            components.zone_name = zone_name(Some(country), offset, dated);
            components
        }
    }
}

/// The value as minutes from 1970-01-01T00:00Z. An `xs:time` counts from midnight instead,
/// which is all its wrap-around arithmetic needs.
fn utc_instant(components: Components, offset: i32) -> i64 {
    let days = if components.shape.has_date() {
        days_from_civil(components.year, components.month, components.day)
    } else {
        0
    };
    days * MINUTES_PER_DAY + i64::from(components.hour) * 60 + i64::from(components.minute)
        - i64::from(offset)
}

/// Rewrites the value's fields to read the same instant in the timezone `target` minutes east
/// of UTC.
fn shift(mut components: Components, instant: i64, target: i32) -> Components {
    let local = instant + i64::from(target);
    let minute_of_day = local.rem_euclid(MINUTES_PER_DAY);
    components.hour = u8::try_from(minute_of_day / 60).unwrap_or(components.hour);
    components.minute = u8::try_from(minute_of_day % 60).unwrap_or(components.minute);
    if components.shape.has_date() {
        let (year, month, day) = civil_from_days(local.div_euclid(MINUTES_PER_DAY));
        components.year = year;
        components.month = month;
        components.day = day;
    }
    components.timezone = Some(target);
    components
}

/// The name the database gives a timezone `offset` minutes east of UTC at `instant`.
///
/// A zone kept by `region` — an ISO 3166 country code — wins over one that merely happens to
/// share the offset, so the same `-05:00` is `EST` in a United States winter and `CDT` in a
/// United States summer.
fn zone_name(region: Option<&str>, offset: i32, instant: Option<i64>) -> Option<&'static str> {
    let mut anywhere = None;
    for zone in CIVIL_ZONES {
        let (zone_offset, name) = zone.at(instant);
        if zone_offset != offset {
            continue;
        }
        if region.is_some_and(|region| {
            zone.countries
                .iter()
                .any(|country| country.eq_ignore_ascii_case(region))
        }) {
            return Some(name);
        }
        anywhere.get_or_insert(name);
    }
    anywhere
}

/// The day number of the `ordinal`th `weekday` of a month — the second Sunday in March, say.
fn nth_weekday(year: i64, month: u8, weekday: i64, ordinal: i64) -> i64 {
    let first = days_from_civil(year, month, 1);
    first + (weekday - weekday_of_day(first)).rem_euclid(7) + (ordinal - 1) * 7
}

/// The day number of the last `weekday` of a month.
fn last_weekday(year: i64, month: u8, weekday: i64) -> i64 {
    let next_month = if month == 12 {
        days_from_civil(year + 1, 1, 1)
    } else {
        days_from_civil(year, month + 1, 1)
    };
    let last = next_month - 1;
    last - (weekday_of_day(last) - weekday).rem_euclid(7)
}

/// The ISO weekday of a day number: 1 is Monday, 7 is Sunday.
fn weekday_of_day(days: i64) -> i64 {
    (days + 3).rem_euclid(7) + 1
}

/// The proleptic Gregorian date of a day number, the inverse of [`days_from_civil`]
/// ([Howard Hinnant's `civil_from_days`](https://howardhinnant.github.io/date_algorithms.html)).
fn civil_from_days(days: i64) -> (i64, u8, u8) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = month_index + if month_index < 10 { 3 } else { -9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (
        year,
        u8::try_from(month).unwrap_or(1),
        u8::try_from(day).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FnContext;
    use crate::datetime::format::{format_date_time, format_time};
    use oxixml_infoset::{AtomicType, AtomicValue};

    fn parse(atomic_type: AtomicType, lexical: &str) -> AtomicValue {
        AtomicValue::parse(atomic_type, lexical).expect("a valid lexical form")
    }

    fn formatted(lexical: &str, picture: &str) -> String {
        let context = FnContext::new();
        format_date_time(
            &context,
            &parse(AtomicType::DateTime, lexical),
            picture,
            None,
            None,
            None,
        )
        .expect("a valid picture")
    }

    fn placed(lexical: &str, picture: &str, place: &str) -> String {
        let context = FnContext::new();
        format_date_time(
            &context,
            &parse(AtomicType::DateTime, lexical),
            picture,
            None,
            None,
            Some(place),
        )
        .expect("a valid picture")
    }

    fn placed_time(lexical: &str, picture: &str, place: &str) -> String {
        let context = FnContext::new();
        format_time(
            &context,
            &parse(AtomicType::Time, lexical),
            picture,
            None,
            None,
            Some(place),
        )
        .expect("a valid picture")
    }

    /// §9.8.4.3: an Olson timezone name in `$place` adjusts the value to the offset in force
    /// there, daylight saving included (`format-dateTime-019a`/`-019b`, `-025c`, `-025e`).
    #[test]
    fn an_olson_place_adjusts_the_value() {
        for (value, expected) in [
            // 025c: winter in New York is `EST`, five hours behind UTC.
            ("2018-01-01T12:00:00-10:00", "17:00 EST"),
            ("2018-01-01T12:00:00-05:00", "12:00 EST"),
            ("2018-01-01T12:00:00Z", "07:00 EST"),
            ("2018-01-01T12:00:00+05:30", "01:30 EST"),
            ("2018-01-01T12:00:00+13:00", "18:00 EST"),
            // 025e: summer is `EDT`, four hours behind.
            ("2018-07-01T12:00:00-10:00", "18:00 EDT"),
            ("2018-07-01T12:00:00-05:00", "13:00 EDT"),
            ("2018-07-01T12:00:00Z", "08:00 EDT"),
            ("2018-07-01T12:00:00+05:30", "02:30 EDT"),
            ("2018-07-01T12:00:00+13:00", "19:00 EDT"),
        ] {
            assert_eq!(
                placed(value, "[H00]:[m00] [ZN]", "America/New_York"),
                expected,
                "{value}"
            );
        }
        // 019a/019b: the adjusted *offset* is what `[Z]` writes, and the date follows the
        // adjustment across midnight.
        assert_eq!(
            placed(
                "2015-02-15T12:00:00Z",
                "[D] [MNn] [Y0001] @ [H01]:[m01] [Z]",
                "America/New_York"
            ),
            "15 February 2015 @ 07:00 -05:00"
        );
        assert_eq!(
            placed(
                "2015-08-15T12:00:00Z",
                "[D] [MNn] [Y0001] @ [H01]:[m01] [Z]",
                "America/New_York"
            ),
            "15 August 2015 @ 08:00 -04:00"
        );
        assert_eq!(
            placed(
                "2018-01-01T02:00:00Z",
                "[Y0001]-[M01]-[D01]",
                "America/New_York"
            ),
            "2017-12-31"
        );
        // 019e-019h: the same for a zone that keeps European summer time.
        assert_eq!(
            placed(
                "2015-02-15T12:00:00Z",
                "[H01]:[m01] [Z] [ZN]",
                "Europe/Paris"
            ),
            "13:00 +01:00 CET"
        );
        assert_eq!(
            placed(
                "2015-08-15T12:00:00Z",
                "[H01]:[m01] [Z] [ZN]",
                "Europe/Paris"
            ),
            "14:00 +02:00 CEST"
        );
    }

    /// An `xs:time` names no day, so a zone's daylight saving cannot be decided for it and its
    /// standard time is used — `format-time-025c` expects `EST` all year.
    #[test]
    fn an_olson_place_keeps_standard_time_for_a_bare_time() {
        for (value, expected) in [
            ("12:00:00-10:00", "17:00 EST"),
            ("12:00:00-05:00", "12:00 EST"),
            ("12:00:00Z", "07:00 EST"),
            ("12:00:00+05:30", "01:30 EST"),
            // The conversion wraps around midnight rather than carrying into a date.
            ("12:00:00+13:00", "18:00 EST"),
        ] {
            assert_eq!(
                placed_time(value, "[H00]:[m00] [ZN]", "America/New_York"),
                expected,
                "{value}"
            );
        }
    }

    /// §9.8.4.6: a country in `$place` does not move the value, it chooses the *name* `[ZN]`
    /// writes — and the same offset has different names in different seasons
    /// (`format-dateTime-025b`, `-025d`, `format-time-025b`).
    #[test]
    fn a_country_place_names_the_timezone() {
        // 025b, a winter date: `-05:00` is the eastern standard time of the United States.
        for (value, expected) in [
            ("2018-01-01T12:00:00-10:00", "HST"),
            ("2018-01-01T12:00:00-05:00", "EST"),
            ("2018-01-01T12:00:00Z", "GMT"),
            ("2018-01-01T12:00:00+05:30", "IST"),
            // No zone of the database keeps `+13:00`, so the offset itself is written.
            ("2018-01-01T12:00:00+13:00", "+13:00"),
        ] {
            assert_eq!(placed(value, "[ZN]", "us"), expected, "{value}");
        }
        // 025d, a summer date: `-05:00` is now the *central daylight* time of the United States,
        // because the eastern zone has moved to `-04:00`.
        for (value, expected) in [
            ("2018-07-01T12:00:00-10:00", "HST"),
            ("2018-07-01T12:00:00-05:00", "CDT"),
            ("2018-07-01T12:00:00-04:00", "EDT"),
            ("2018-07-01T12:00:00Z", "GMT"),
            ("2018-07-01T12:00:00+05:30", "IST"),
            ("2018-07-01T12:00:00+13:00", "+13:00"),
        ] {
            assert_eq!(placed(value, "[ZN]", "us"), expected, "{value}");
        }
        // format-time-025b: standard time throughout, for want of a date.
        for (value, expected) in [
            ("12:00:00-10:00", "HST"),
            ("12:00:00-05:00", "EST"),
            ("12:00:00Z", "GMT"),
            ("12:00:00+05:30", "IST"),
            ("12:00:00+13:00", "+13:00"),
        ] {
            assert_eq!(placed_time(value, "[ZN]", "us"), expected, "{value}");
        }
    }

    /// Without a `$place` there is no database to consult, so `[ZN]` keeps writing the offset —
    /// which is what `format-dateTime-025a` and `format-time-025a` pin down for every other
    /// timezone presentation as well.
    #[test]
    fn the_timezone_presentations_are_unchanged_without_a_place() {
        let value = "2018-01-01T12:00:00-05:00";
        assert_eq!(formatted(value, "[ZN]"), "-05:00");
        assert_eq!(formatted(value, "[Z]"), "-05:00");
        assert_eq!(formatted(value, "[Z0]"), "-5");
        assert_eq!(formatted(value, "[Z0000]"), "-0500");
        assert_eq!(formatted(value, "[z]"), "GMT-05:00");
        assert_eq!(formatted(value, "[ZZ]"), "R");
        assert_eq!(formatted("2018-01-01T12:00:00Z", "[ZZ]"), "Z");
        assert_eq!(formatted("2018-01-01T12:00:00+13:00", "[ZZ]"), "+13:00");
        // A place that is neither a zone the database holds nor a country code names nothing
        // either, and never moves the value.
        assert_eq!(placed(value, "[ZN]", "Mars/Olympus_Mons"), "-05:00");
        assert_eq!(
            placed(value, "[H01]:[m01][ZN]", "America/Sao_Paulo"),
            "12:00-05:00"
        );
    }

    /// A country code only chooses a name; §9.8.4.3 reserves the adjustment for the timezone
    /// *name* form. A two-letter place that the IANA database happens to accept as an alias for
    /// a zone — `GB` for `Europe/London` — must therefore still leave the value where it is.
    #[test]
    fn a_country_code_never_moves_the_value() {
        let value = "2018-01-01T12:00:00-05:00";
        assert_eq!(placed(value, "[H01]:[m01][Z]", "GB"), "12:00-05:00");
        assert_eq!(placed(value, "[H01]:[m01][Z]", "gb"), "12:00-05:00");
        // The zone the alias stands for does adjust, and does name itself.
        assert_eq!(
            placed(value, "[H01]:[m01] [Z] [ZN]", "Europe/London"),
            "17:00 +00:00 GMT"
        );
        // A country code names a timezone the country keeps, in its season.
        assert_eq!(placed("2018-01-01T12:00:00Z", "[ZN]", "gb"), "GMT");
        assert_eq!(placed("2018-07-01T12:00:00+01:00", "[ZN]", "gb"), "BST");
    }

    /// The daylight-saving windows are the ones the two rules define, to the hour.
    #[test]
    fn daylight_saving_starts_and_ends_on_the_right_day() {
        // The United States: 02:00 local standard time on the second Sunday in March, which is
        // 07:00 UTC for the eastern zone.
        assert_eq!(
            placed("2018-03-11T06:59:00Z", "[ZN]", "America/New_York"),
            "EST"
        );
        assert_eq!(
            placed("2018-03-11T07:00:00Z", "[ZN]", "America/New_York"),
            "EDT"
        );
        // …until 02:00 local daylight time on the first Sunday in November, 06:00 UTC.
        assert_eq!(
            placed("2018-11-04T05:59:00Z", "[ZN]", "America/New_York"),
            "EDT"
        );
        assert_eq!(
            placed("2018-11-04T06:00:00Z", "[ZN]", "America/New_York"),
            "EST"
        );
        // The European Union: 01:00 UTC on the last Sunday in March and in October.
        assert_eq!(
            placed("2018-03-25T00:59:00Z", "[ZN]", "Europe/Paris"),
            "CET"
        );
        assert_eq!(
            placed("2018-03-25T01:00:00Z", "[ZN]", "Europe/Paris"),
            "CEST"
        );
        assert_eq!(
            placed("2018-10-28T00:59:00Z", "[ZN]", "Europe/Paris"),
            "CEST"
        );
        assert_eq!(
            placed("2018-10-28T01:00:00Z", "[ZN]", "Europe/Paris"),
            "CET"
        );
    }

    /// A value with no timezone has nothing to adjust and nothing to name.
    #[test]
    fn a_value_without_a_timezone_is_left_alone() {
        assert_eq!(
            placed("2018-07-01T12:00:00", "[H01]:[m01][ZN]", "America/New_York"),
            "12:00"
        );
    }

    /// [`civil_from_days`] inverts [`days_from_civil`], including across the leap-day and
    /// century rules and before the epoch.
    #[test]
    fn day_numbers_round_trip_through_the_calendar() {
        for (year, month, day) in [
            (1970, 1, 1),
            (1969, 12, 31),
            (1600, 2, 29),
            (1900, 3, 1),
            (2000, 2, 29),
            (2018, 11, 4),
            (2400, 12, 31),
        ] {
            let days = days_from_civil(year, month, day);
            assert_eq!(civil_from_days(days), (year, month, day));
        }
    }
}
