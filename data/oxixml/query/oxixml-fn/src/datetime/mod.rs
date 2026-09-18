//! The functions on durations, dates and times of
//! [F&O 3.1 §9](https://www.w3.org/TR/xpath-functions-31/#durations-dates-times).
//!
//! * [`components`] — the component extractors, `fn:adjust-*-to-timezone`, the `fn:current-*`
//!   functions and `fn:dateTime`.
//! * [`mod@format_integer`] — `fn:format-integer` and the numbering sequences the date/time
//!   pictures are defined in terms of.
//! * [`mod@format`] — `fn:format-dateTime`, `fn:format-date` and `fn:format-time`.
//! * [`ietf`] — `fn:parse-ietf-date`.
//!
//! The arithmetic operators on dates, times and durations live in
//! [`crate::numeric::arith`], because they share the promotion machinery with the numeric
//! operators.

pub mod components;
pub mod format;
pub mod format_integer;
pub mod ietf;

pub use components::{
    TimezoneArgument, adjust_date_time_to_timezone, adjust_date_to_timezone,
    adjust_time_to_timezone, as_duration, current_date, current_date_time, current_time, date_time,
    day_from_date, day_from_date_time, days_from_duration, decimal_value, hours_from_date_time,
    hours_from_duration, hours_from_time, implicit_timezone, integer_value, minutes_from_date_time,
    minutes_from_duration, minutes_from_time, month_from_date, month_from_date_time,
    months_from_duration, seconds_from_date_time, seconds_from_duration, seconds_from_time,
    timezone_from_date, timezone_from_date_time, timezone_from_time, timezone_value,
    year_from_date, year_from_date_time, years_from_duration,
};
pub use format::{format_date, format_date_time, format_time};
pub use format_integer::{IntegerPicture, digit_family_zero, format_integer, ordinal_suffix};
pub use ietf::parse_ietf_date;
