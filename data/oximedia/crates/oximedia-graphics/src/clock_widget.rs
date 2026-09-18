//! On-screen clock graphics widget.
//!
//! Provides digital, analog (simplified), and countdown clock rendering
//! with configurable formats and timezone offsets.

/// Clock display format.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ClockFormat {
    /// HH:MM:SS — 24-hour with seconds.
    HhMmSs,
    /// HH:MM — 24-hour without seconds.
    HhMm,
    /// HH:MM AM/PM — 12-hour format.
    HhMmAmPm,
    /// Unix epoch timestamp (integer seconds).
    Epoch,
    /// HH:MM followed by the timezone abbreviation offset.
    TimezoneAbbrev,
}

/// Clock visual style.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ClockStyle {
    /// Text-based digital display.
    Digital,
    /// Simplified circular analog display.
    Analog,
    /// Countdown timer display.
    Countdown,
}

/// Configuration for an on-screen clock widget.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ClockConfig {
    /// Display format.
    pub format: ClockFormat,
    /// Visual style.
    pub style: ClockStyle,
    /// Timezone offset in whole hours (e.g. -5 for EST, +9 for JST).
    pub timezone_offset_hours: i8,
    /// Whether to display the current date below the time.
    pub show_date: bool,
    /// Clock color as RGBA.
    pub color: [u8; 4],
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self {
            format: ClockFormat::HhMmSs,
            style: ClockStyle::Digital,
            timezone_offset_hours: 0,
            show_date: false,
            color: [255, 255, 255, 255],
        }
    }
}

/// Renderer for digital clock text output.
pub struct DigitalClockRenderer;

impl DigitalClockRenderer {
    /// Render the formatted time string from a millisecond timestamp (UTC).
    ///
    /// The timezone offset in `config.timezone_offset_hours` is applied
    /// before formatting.
    #[allow(dead_code)]
    pub fn render_text(time_ms: u64, config: &ClockConfig) -> String {
        // Convert ms → seconds, apply timezone offset
        let offset_secs = i64::from(config.timezone_offset_hours) * 3600;
        let total_secs = (time_ms / 1000) as i64 + offset_secs;

        // Keep secs positive for modular arithmetic
        let total_secs = total_secs.rem_euclid(86400); // seconds in a day

        let hh = (total_secs / 3600) as u32;
        let mm = ((total_secs % 3600) / 60) as u32;
        let ss = (total_secs % 60) as u32;

        match config.format {
            ClockFormat::HhMmSs => format!("{hh:02}:{mm:02}:{ss:02}"),
            ClockFormat::HhMm => format!("{hh:02}:{mm:02}"),
            ClockFormat::HhMmAmPm => {
                let (display_h, suffix) = if hh == 0 {
                    (12, "AM")
                } else if hh < 12 {
                    (hh, "AM")
                } else if hh == 12 {
                    (12, "PM")
                } else {
                    (hh - 12, "PM")
                };
                format!("{display_h:02}:{mm:02} {suffix}")
            }
            ClockFormat::Epoch => {
                // Return the original epoch seconds (ignoring timezone)
                format!("{}", time_ms / 1000)
            }
            ClockFormat::TimezoneAbbrev => {
                let sign = if config.timezone_offset_hours >= 0 {
                    '+'
                } else {
                    '-'
                };
                let abs_offset = config.timezone_offset_hours.unsigned_abs();
                format!("{hh:02}:{mm:02}:{ss:02} UTC{sign}{abs_offset:02}")
            }
        }
    }
}

/// Countdown clock that tracks time remaining until a target.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CountdownClock {
    /// Target time in milliseconds (Unix epoch ms).
    pub target_ms: u64,
}

impl CountdownClock {
    /// Create a new countdown clock targeting the given millisecond timestamp.
    #[allow(dead_code)]
    pub fn new(target_ms: u64) -> Self {
        Self { target_ms }
    }

    /// Returns the remaining milliseconds until the target, or `None` if expired.
    #[allow(dead_code)]
    pub fn remaining(&self, current_ms: u64) -> Option<u64> {
        if current_ms >= self.target_ms {
            None
        } else {
            Some(self.target_ms - current_ms)
        }
    }

    /// Format the remaining time as "HH:MM:SS".
    ///
    /// Caps display at 99:59:59 to avoid overflow in display.
    #[allow(dead_code)]
    pub fn format(&self, remaining_ms: u64) -> String {
        let total_secs = remaining_ms / 1000;
        let hh = (total_secs / 3600).min(99);
        let mm = (total_secs % 3600) / 60;
        let ss = total_secs % 60;
        format!("{hh:02}:{mm:02}:{ss:02}")
    }
}

/// Simplified analog clock that computes hand angles.
pub struct AnalogClock;

impl AnalogClock {
    /// Compute the angles (in degrees) for hour, minute, and second hands.
    ///
    /// - Returns `(hour_deg, min_deg, sec_deg)`.
    /// - Hour hand: full rotation every 12 hours (smoothly interpolated).
    /// - Minute hand: full rotation every 60 minutes.
    /// - Second hand: full rotation every 60 seconds.
    #[allow(dead_code)]
    pub fn hands_angles(time_ms: u64) -> (f32, f32, f32) {
        let total_secs = (time_ms / 1000) % 86400;
        let hours_12 = total_secs % 43200; // seconds within 12-hour cycle
        let minutes = total_secs % 3600;
        let seconds = total_secs % 60;
        let ms_frac = (time_ms % 1000) as f32 / 1000.0;

        let sec_deg = (seconds as f32 + ms_frac) * (360.0 / 60.0);
        let min_deg = (minutes as f32 + (seconds as f32 + ms_frac) / 60.0) * (360.0 / 3600.0);
        let hour_deg = hours_12 as f32 * (360.0 / 43200.0);

        (hour_deg, min_deg, sec_deg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc_ms(h: u64, m: u64, s: u64) -> u64 {
        (h * 3600 + m * 60 + s) * 1000
    }

    #[test]
    fn test_digital_render_hh_mm_ss() {
        let config = ClockConfig {
            format: ClockFormat::HhMmSs,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(utc_ms(14, 30, 45), &config);
        assert_eq!(result, "14:30:45");
    }

    #[test]
    fn test_digital_render_hh_mm() {
        let config = ClockConfig {
            format: ClockFormat::HhMm,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(utc_ms(9, 5, 0), &config);
        assert_eq!(result, "09:05");
    }

    #[test]
    fn test_digital_render_am_pm_morning() {
        let config = ClockConfig {
            format: ClockFormat::HhMmAmPm,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(utc_ms(9, 30, 0), &config);
        assert!(result.ends_with("AM"), "Expected AM, got: {result}");
        assert!(result.starts_with("09:30"));
    }

    #[test]
    fn test_digital_render_am_pm_afternoon() {
        let config = ClockConfig {
            format: ClockFormat::HhMmAmPm,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(utc_ms(15, 0, 0), &config);
        assert!(result.ends_with("PM"), "Expected PM, got: {result}");
        assert!(result.starts_with("03:00"));
    }

    #[test]
    fn test_digital_render_am_pm_midnight() {
        let config = ClockConfig {
            format: ClockFormat::HhMmAmPm,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(0, &config);
        assert!(result.starts_with("12:00"));
        assert!(result.ends_with("AM"));
    }

    #[test]
    fn test_digital_render_epoch() {
        let config = ClockConfig {
            format: ClockFormat::Epoch,
            ..ClockConfig::default()
        };
        let ms = 1_700_000_000_000u64;
        let result = DigitalClockRenderer::render_text(ms, &config);
        assert_eq!(result, "1700000000");
    }

    #[test]
    fn test_digital_render_timezone_abbrev() {
        let config = ClockConfig {
            format: ClockFormat::TimezoneAbbrev,
            timezone_offset_hours: 9,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(utc_ms(0, 0, 0), &config);
        assert!(result.contains("UTC+09"), "Got: {result}");
    }

    #[test]
    fn test_digital_render_timezone_negative() {
        let config = ClockConfig {
            format: ClockFormat::TimezoneAbbrev,
            timezone_offset_hours: -5,
            ..ClockConfig::default()
        };
        let result = DigitalClockRenderer::render_text(utc_ms(12, 0, 0), &config);
        assert!(result.contains("UTC-05"), "Got: {result}");
    }

    #[test]
    fn test_countdown_remaining_active() {
        let clock = CountdownClock::new(10_000);
        assert_eq!(clock.remaining(5_000), Some(5_000));
    }

    #[test]
    fn test_countdown_remaining_expired() {
        let clock = CountdownClock::new(5_000);
        assert_eq!(clock.remaining(10_000), None);
        assert_eq!(clock.remaining(5_000), None); // exactly at target
    }

    #[test]
    fn test_countdown_format() {
        let clock = CountdownClock::new(0);
        // 3723000ms = 1h 2m 3s
        assert_eq!(clock.format(3_723_000), "01:02:03");
    }

    #[test]
    fn test_countdown_format_zero() {
        let clock = CountdownClock::new(0);
        assert_eq!(clock.format(0), "00:00:00");
    }

    #[test]
    fn test_analog_clock_noon() {
        // 12:00:00 UTC
        let (h, m, s) = AnalogClock::hands_angles(utc_ms(12, 0, 0));
        // All hands at top (0° or 360°)
        assert!((h % 360.0).abs() < 0.1, "Hour hand at noon: {h}");
        assert!((m % 360.0).abs() < 0.1, "Minute hand at noon: {m}");
        assert!((s % 360.0).abs() < 0.1, "Second hand at noon: {s}");
    }

    #[test]
    fn test_analog_clock_three_oclock() {
        // 3:00:00 — hour hand should be at 90°
        let (h, m, s) = AnalogClock::hands_angles(utc_ms(3, 0, 0));
        assert!((h - 90.0).abs() < 0.5, "Hour at 3 o'clock: {h}");
        assert!(m.abs() < 0.1 || (m - 360.0).abs() < 0.1, "Minute: {m}");
        assert!(s.abs() < 0.1 || (s - 360.0).abs() < 0.1, "Second: {s}");
    }
}
