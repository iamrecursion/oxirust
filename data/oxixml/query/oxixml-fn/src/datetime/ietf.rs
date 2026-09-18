//! [`fn:parse-ietf-date`](https://www.w3.org/TR/xpath-functions-31/#func-parse-ietf-date):
//! the date formats of RFC 5322, RFC 2616 and `asctime`, read into an `xs:dateTime`.
//!
//! The grammar is the one F&O 3.1 §9.9.2 spells out, and this module follows it literally:
//!
//! ```text
//! input        ::= S? (dayname ","? S)? ((datespec S time) | asctime) S?
//! datespec     ::= daynum dsep monthname dsep year
//! asctime      ::= monthname dsep daynum S time S year
//! dsep         ::= S | (S? "-" S?)
//! time         ::= hours ":" minutes (":" seconds)? (S? timezone)?
//! timezone     ::= tzname | tzoffset (S? "(" S? tzname S? ")")?
//! tzoffset     ::= ("+"|"-") hours ":"? minutes?
//! ```
//!
//! Names are matched without regard to case, a two-digit year means the nineteen-hundreds, and
//! an absent timezone means UTC. Anything the grammar does not accept — including a day number
//! that does not exist in its month — is `FORG0010`.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue};

/// The day names the grammar admits, short form first.
const DAY_NAMES: [(&str, &str); 7] = [
    ("mon", "monday"),
    ("tue", "tuesday"),
    ("wed", "wednesday"),
    ("thu", "thursday"),
    ("fri", "friday"),
    ("sat", "saturday"),
    ("sun", "sunday"),
];

/// The month names the grammar admits, in calendar order.
const MONTH_NAMES: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

/// The timezone names the grammar admits, with their offsets in minutes.
const ZONE_NAMES: [(&str, i32); 11] = [
    ("ut", 0),
    ("utc", 0),
    ("gmt", 0),
    ("est", -5 * 60),
    ("edt", -4 * 60),
    ("cst", -6 * 60),
    ("cdt", -5 * 60),
    ("mst", -7 * 60),
    ("mdt", -6 * 60),
    ("pst", -8 * 60),
    ("pdt", -7 * 60),
];

/// [`fn:parse-ietf-date`](https://www.w3.org/TR/xpath-functions-31/#func-parse-ietf-date).
///
/// # Errors
///
/// [`FnErrorCode::Forg0010`] when the string does not match the grammar, or names a date that
/// does not exist.
///
/// # Examples
///
/// ```
/// use oxixml_fn::datetime::parse_ietf_date;
///
/// assert_eq!(
///     parse_ietf_date("Wed, 20 Aug 2014 19:36:01 GMT")?.string_value(),
///     "2014-08-20T19:36:01Z"
/// );
/// assert_eq!(
///     parse_ietf_date("Wed Aug 20 14:36:01-05:00 2014")?.string_value(),
///     "2014-08-20T14:36:01-05:00"
/// );
/// assert!(parse_ietf_date("2014-08-20T19:36:01Z").is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn parse_ietf_date(input: &str) -> FnResult<AtomicValue> {
    let mut scanner = Scanner {
        chars: input.chars().collect(),
        position: 0,
    };
    let parsed = scanner.input().ok_or_else(|| malformed(input))?;
    let lexical = parsed.lexical();
    AtomicValue::parse(AtomicType::DateTime, &lexical).map_err(|_| malformed(input))
}

fn malformed(input: &str) -> FnError {
    FnError::new(
        FnErrorCode::Forg0010,
        format!("`{input}` is not an IETF-format date"),
    )
}

/// The pieces one input yields, before they are spelled as an `xs:dateTime`.
#[derive(Debug)]
struct Parsed {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    fraction: String,
    offset_minutes: i32,
}

impl Parsed {
    fn lexical(&self) -> String {
        let sign = if self.offset_minutes < 0 { '-' } else { '+' };
        let zone = if self.offset_minutes == 0 {
            "Z".to_owned()
        } else {
            let magnitude = self.offset_minutes.abs();
            format!("{sign}{:02}:{:02}", magnitude / 60, magnitude % 60)
        };
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{zone}",
            self.year, self.month, self.day, self.hour, self.minute, self.second, self.fraction
        )
    }
}

struct Scanner {
    chars: Vec<char>,
    position: usize,
}

/// A parsed time-of-day, with the timezone that followed it.
struct TimeOfDay {
    hour: u32,
    minute: u32,
    second: u32,
    fraction: String,
    offset_minutes: i32,
}

impl Scanner {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).copied()
    }

    fn at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.position + offset).copied()
    }

    fn eat(&mut self, wanted: char) -> bool {
        if self.peek() == Some(wanted) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    /// `S`, one or more spaces; returns whether any were consumed.
    fn space(&mut self) -> bool {
        let start = self.position;
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.position += 1;
        }
        self.position > start
    }

    /// `dsep ::= S | (S? "-" S?)`.
    fn separator(&mut self) -> bool {
        let spaced = self.space();
        if self.eat('-') {
            self.space();
            return true;
        }
        spaced
    }

    /// The whole input, or `None` if it does not match the grammar.
    fn input(&mut self) -> Option<Parsed> {
        self.space();
        self.day_name();
        let parsed = if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.datespec_form()?
        } else {
            self.asctime_form()?
        };
        self.space();
        (self.position == self.chars.len()).then_some(parsed)
    }

    /// `(dayname ","? S)?`, which is skipped when no day name is there.
    fn day_name(&mut self) {
        let saved = self.position;
        let Some(word) = self.word() else {
            return;
        };
        let lowered = word.to_ascii_lowercase();
        if !DAY_NAMES
            .iter()
            .any(|(short, long)| lowered == *short || lowered == *long)
        {
            self.position = saved;
            return;
        }
        self.eat(',');
        if !self.space() {
            self.position = saved;
        }
    }

    /// `datespec S time`, with `datespec ::= daynum dsep monthname dsep year`.
    fn datespec_form(&mut self) -> Option<Parsed> {
        let day = self.day_number()?;
        if !self.separator() {
            return None;
        }
        let month = self.month_name()?;
        if !self.separator() {
            return None;
        }
        let year = self.year()?;
        if !self.space() {
            return None;
        }
        let time = self.time()?;
        Some(self.assemble(year, month, day, time))
    }

    /// `asctime ::= monthname dsep daynum S time S year`.
    fn asctime_form(&mut self) -> Option<Parsed> {
        let month = self.month_name()?;
        if !self.separator() {
            return None;
        }
        let day = self.day_number()?;
        if !self.space() {
            return None;
        }
        let time = self.time()?;
        if !self.space() {
            return None;
        }
        let year = self.year()?;
        Some(self.assemble(year, month, day, time))
    }

    fn assemble(&self, year: i32, month: u32, day: u32, time: TimeOfDay) -> Parsed {
        Parsed {
            year,
            month,
            day,
            hour: time.hour,
            minute: time.minute,
            second: time.second,
            fraction: time.fraction,
            offset_minutes: time.offset_minutes,
        }
    }

    /// A run of ASCII letters.
    fn word(&mut self) -> Option<String> {
        let start = self.position;
        while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            self.position += 1;
        }
        (self.position > start).then(|| self.chars[start..self.position].iter().collect())
    }

    /// Exactly `count` decimal digits, as a number.
    fn digits(&mut self, count: usize) -> Option<u32> {
        let mut value = 0_u32;
        for _ in 0..count {
            let digit = self.peek()?.to_digit(10)?;
            value = value * 10 + digit;
            self.position += 1;
        }
        Some(value)
    }

    /// `daynum ::= digit digit?`, which may not be followed by a third digit.
    fn day_number(&mut self) -> Option<u32> {
        let first = self.digits(1)?;
        let value = match self.peek().and_then(|c| c.to_digit(10)) {
            Some(second) => {
                self.position += 1;
                first * 10 + second
            }
            None => first,
        };
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return None;
        }
        (1..=31).contains(&value).then_some(value)
    }

    /// `year ::= digit digit (digit digit)?`, with a two-digit year in the nineteen-hundreds.
    fn year(&mut self) -> Option<i32> {
        let first = self.digits(2)?;
        if self
            .at(0)
            .is_some_and(|c| c.is_ascii_digit() && self.at(1).is_some_and(|d| d.is_ascii_digit()))
        {
            let second = self.digits(2)?;
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return None;
            }
            #[allow(clippy::cast_possible_wrap)]
            return Some((first * 100 + second) as i32);
        }
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return None;
        }
        #[allow(clippy::cast_possible_wrap)]
        Some(1900 + first as i32)
    }

    fn month_name(&mut self) -> Option<u32> {
        let saved = self.position;
        let word = self.word()?;
        let lowered = word.to_ascii_lowercase();
        match MONTH_NAMES.iter().position(|name| *name == lowered) {
            #[allow(clippy::cast_possible_truncation)]
            Some(index) => Some(index as u32 + 1),
            None => {
                self.position = saved;
                None
            }
        }
    }

    /// `time ::= hours ":" minutes (":" seconds)? (S? timezone)?`.
    fn time(&mut self) -> Option<TimeOfDay> {
        let hour = self.one_or_two_digits()?;
        if hour > 24 {
            return None;
        }
        if !self.eat(':') {
            return None;
        }
        let minute = self.digits(2)?;
        if minute > 59 || self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return None;
        }
        let mut second = 0;
        let mut fraction = String::new();
        if self.peek() == Some(':') && self.at(1).is_some_and(|c| c.is_ascii_digit()) {
            self.position += 1;
            second = self.digits(2)?;
            if second > 59 || self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return None;
            }
            if self.peek() == Some('.') {
                self.position += 1;
                let start = self.position;
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.position += 1;
                }
                if self.position == start {
                    return None;
                }
                fraction = format!(
                    ".{}",
                    self.chars[start..self.position].iter().collect::<String>()
                );
            }
        } else if self.peek() == Some(':') {
            return None;
        }
        let offset_minutes = self.timezone()?;
        Some(TimeOfDay {
            hour,
            minute,
            second,
            fraction,
            offset_minutes,
        })
    }

    /// `hours ::= digit digit?`, preferring two digits.
    fn one_or_two_digits(&mut self) -> Option<u32> {
        let first = self.digits(1)?;
        match self.peek().and_then(|c| c.to_digit(10)) {
            Some(second) => {
                self.position += 1;
                Some(first * 10 + second)
            }
            None => Some(first),
        }
    }

    /// `(S? timezone)?`; an absent timezone is UTC.
    fn timezone(&mut self) -> Option<i32> {
        let saved = self.position;
        self.space();
        match self.peek() {
            Some('+' | '-') => match self.zone_offset() {
                Some(offset) => {
                    if self.parenthesized_zone_name().is_none() {
                        self.position = saved;
                        return Some(0);
                    }
                    Some(offset)
                }
                None => {
                    self.position = saved;
                    Some(0)
                }
            },
            Some(c) if c.is_ascii_alphabetic() => {
                let attempt = self.position;
                match self.word().map(|word| word.to_ascii_lowercase()) {
                    Some(word) => match ZONE_NAMES.iter().find(|(name, _)| *name == word) {
                        Some((_, offset)) => Some(*offset),
                        None => {
                            self.position = attempt;
                            self.position = saved;
                            Some(0)
                        }
                    },
                    None => {
                        self.position = saved;
                        Some(0)
                    }
                }
            }
            _ => {
                self.position = saved;
                Some(0)
            }
        }
    }

    /// `tzoffset ::= ("+"|"-") hours ":"? minutes?`, where a two-digit hour that leaves an
    /// unusable remainder is retried as a one-digit hour — `-500` is five hours, not fifty.
    fn zone_offset(&mut self) -> Option<i32> {
        let start = self.position;
        let negative = match self.peek() {
            Some('+') => false,
            Some('-') => true,
            _ => return None,
        };
        self.position += 1;
        let after_sign = self.position;
        for hour_digits in [2_usize, 1] {
            self.position = after_sign;
            let Some(hours) = self.digits(hour_digits) else {
                continue;
            };
            let colon = self.eat(':');
            let minutes = if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                match self.digits(2) {
                    Some(minutes) => minutes,
                    None => continue,
                }
            } else {
                0
            };
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
            if hours > 14 || minutes > 59 || (hours == 14 && minutes > 0) {
                continue;
            }
            let _ = colon;
            #[allow(clippy::cast_possible_wrap)]
            let total = (hours * 60 + minutes) as i32;
            return Some(if negative { -total } else { total });
        }
        self.position = start;
        None
    }

    /// `(S? "(" S? tzname S? ")")?` — the comment form that may follow a numeric offset.
    fn parenthesized_zone_name(&mut self) -> Option<()> {
        let saved = self.position;
        self.space();
        if !self.eat('(') {
            self.position = saved;
            return Some(());
        }
        self.space();
        let word = self.word()?.to_ascii_lowercase();
        if !ZONE_NAMES.iter().any(|(name, _)| *name == word) {
            return None;
        }
        self.space();
        self.eat(')').then_some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(input: &str) -> String {
        parse_ietf_date(input)
            .map(|value| value.string_value())
            .unwrap_or_else(|error| error.name().local_name().to_owned())
    }

    #[test]
    fn the_rfc_forms_all_read() {
        for input in [
            "Wed, 20 Aug 2014 19:36:01 GMT",
            "  Wed, 20 Aug 2014 19:36:01 GMT",
            "Wed 20 Aug 2014 19:36:01 GMT",
            "Wednesday 20 Aug 2014 19:36:01 GMT",
            "wed, 20 aug 2014 19:36:01 gmt",
            "Wed, 20 Aug 2014 19:36:01",
            "Wed, 20 Aug 2014 19:36:01GMT",
            " 20 Aug 2014 19:36:01 GMT",
            "Wed, 20 - Aug - 2014 19:36:01",
            "Wed, 20 Aug 2014 19:36:01 UT",
            "Wed Aug 20 19:36:01 2014",
        ] {
            assert_eq!(parsed(input), "2014-08-20T19:36:01Z", "{input}");
        }
    }

    #[test]
    fn named_zones_shift_the_instant() {
        assert_eq!(
            parsed("Wed, 20 Aug 2014 14:36:01 EST"),
            "2014-08-20T14:36:01-05:00"
        );
        assert_eq!(
            parsed("Wed, 20 Aug 2014 11:36:01 PST"),
            "2014-08-20T11:36:01-08:00"
        );
    }

    #[test]
    fn numeric_offsets_take_every_spelling() {
        for input in [
            "Wed, 20 Aug 2014 14:36:01 -05:00(EST)",
            "Wed, 20 Aug 2014 14:36:01-05:00",
            "Wed, 20 Aug 2014 14:36:01-05",
            "Wed, 20 Aug 2014 14:36:01-05:",
            "Wed, 20 Aug 2014 14:36:01 -0500",
            "Wed, 20 Aug 2014 14:36:01 -05:00  (  EST  )",
        ] {
            assert_eq!(parsed(input), "2014-08-20T14:36:01-05:00", "{input}");
        }
        assert_eq!(
            parsed("Aug 20 4:36:01 -500 2014"),
            "2014-08-20T04:36:01-05:00"
        );
    }

    #[test]
    fn a_two_digit_year_is_in_the_nineteen_hundreds() {
        assert_eq!(parsed("20 Aug 14 19:36:01"), "1914-08-20T19:36:01Z");
        assert_eq!(parsed("Feb-02 02:02-02: 02"), "1902-02-02T02:02:00-02:00");
    }

    #[test]
    fn the_grammar_rejects_what_it_should() {
        for input in [
            "2014-08-20T19:36:01Z",
            "Wed, 020 Aug 2014 19:36:01 GMT",
            "Wed, 20 August 2014 19:36:01 GMT",
            "Wed, 20 Aug 114 19:36:01 GMT",
            "Wed, 20 Aug 2014 19:36:01 CET",
            "Mon, 32 Aug 2014 19:36:01 GMT",
            "Sat, 29 Feb 2014 19:36:01 GMT",
            "Wed, 00 Aug 2014 19:36:01 GMT",
            "Boy 20 Aug 2014 19:36:01 GMT",
            "Manchester, 20 Aug 2014 19:36:01 GMT",
            "Aug,20 19:36:01 2014",
            "Aug20 19:36:01 2014",
            "Wed,20 Aug 2014 19:36:01",
            "Aug 2014 19:36:01",
            "20Aug 2014 19:36:01",
            "20 Aug2014 19:36:01",
            "Aug 20 19:36:01GMT2014",
            "Aug 20 19:36:01GMT 014",
            "Wed, 20 Aug 2014 19:36:01 GMT Manchester",
            "Aug 20 19.36.01GMT 2014",
            "Aug 20 19:3:01GMT 2014",
            "Aug 20 19:36:0.1GMT 2014",
            "Aug 20 19:36:GMT 2014",
            "Aug 20 19:36:01.GMT 2014",
            "Aug 20 19:36:01 -05:0 2014",
            "Aug 20 19:36:01 -05:00 () 2014",
            "Aug 20 19:36:01 -05:00 (CET) 2014",
            "Aug 20 19:36:01 -05:00 EST 2014",
            "Aug 20 19:36(EST) 2014",
            "",
            "Aug 20 29:36:01GMT 2014",
            "Aug 20 19:66:01GMT 2014",
            "Aug 20 19:36:71GMT 2014",
            "Aug 20 19:36:01 -15:00 2014",
            "Aug 20 19:36:01 -05:60 2014",
            "Feb 28 19:36:01 GMT -2000",
        ] {
            assert_eq!(parsed(input), "FORG0010", "{input}");
        }
    }

    #[test]
    fn a_twenty_four_hour_rolls_over() {
        assert_eq!(parsed("Aug 20 24:00:00 2014"), "2014-08-21T00:00:00Z");
        assert_eq!(parsed("Aug 20 24:00 2014"), "2014-08-21T00:00:00Z");
    }
}
