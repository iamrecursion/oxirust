//! Countdown timer graphics widget for broadcast productions.
//!
//! Provides countdown and count-up timer displays with customizable
//! visual styles, alert thresholds, and animation effects.

#![allow(dead_code)]

/// Timer direction: counting down or counting up
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerDirection {
    /// Count down from a start value to zero
    CountDown,
    /// Count up from zero
    CountUp,
}

/// Timer state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerState {
    /// Timer is idle / not started
    Idle,
    /// Timer is actively running
    Running,
    /// Timer is paused
    Paused,
    /// Timer has finished (countdown reached zero or target reached)
    Finished,
}

/// Alert level based on remaining time
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertLevel {
    /// Normal (plenty of time remaining)
    Normal,
    /// Warning (approaching threshold)
    Warning,
    /// Critical (very little time remaining)
    Critical,
    /// Expired (countdown past zero)
    Expired,
}

impl AlertLevel {
    /// Returns the suggested text color (RGBA) for this alert level
    pub fn suggested_color(&self) -> [u8; 4] {
        match self {
            AlertLevel::Normal => [255, 255, 255, 255],
            AlertLevel::Warning => [255, 200, 0, 255],
            AlertLevel::Critical => [255, 80, 0, 255],
            AlertLevel::Expired => [255, 0, 0, 255],
        }
    }
}

/// Timer display format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerFormat {
    /// HH:MM:SS
    HoursMinutesSeconds,
    /// MM:SS
    MinutesSeconds,
    /// MM:SS.f (with tenths)
    MinutesSecondsTenths,
    /// SS (seconds only)
    SecondsOnly,
    /// SS.ff (seconds and hundredths)
    SecondsHundredths,
}

impl TimerFormat {
    /// Formats a duration in milliseconds to a string
    pub fn format(&self, ms: i64) -> String {
        let negative = ms < 0;
        let ms_abs = ms.unsigned_abs();
        let total_secs = ms_abs / 1000;
        let millis = ms_abs % 1000;
        let secs = total_secs % 60;
        let mins = (total_secs / 60) % 60;
        let hours = total_secs / 3600;
        let sign = if negative { "-" } else { "" };

        match self {
            TimerFormat::HoursMinutesSeconds => {
                format!("{sign}{hours:02}:{mins:02}:{secs:02}")
            }
            TimerFormat::MinutesSeconds => {
                let total_mins = total_secs / 60;
                format!("{sign}{total_mins:02}:{secs:02}")
            }
            TimerFormat::MinutesSecondsTenths => {
                let total_mins = total_secs / 60;
                let tenths = millis / 100;
                format!("{sign}{total_mins:02}:{secs:02}.{tenths}")
            }
            TimerFormat::SecondsOnly => {
                format!("{sign}{total_secs}")
            }
            TimerFormat::SecondsHundredths => {
                let hundredths = millis / 10;
                format!("{sign}{total_secs:02}.{hundredths:02}")
            }
        }
    }
}

/// Visual style for the countdown timer
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerStyle {
    /// Plain text display
    Plain,
    /// Digital LCD-style display
    Digital,
    /// Circular/pie countdown indicator
    Circular,
    /// Progress bar style
    ProgressBar,
    /// Flip clock style
    FlipClock,
}

/// Threshold configuration for alert levels
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Transition to Warning below this many milliseconds
    pub warning_ms: i64,
    /// Transition to Critical below this many milliseconds
    pub critical_ms: i64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            warning_ms: 30_000,  // 30 seconds
            critical_ms: 10_000, // 10 seconds
        }
    }
}

impl AlertThresholds {
    /// Creates thresholds with custom values
    pub fn new(warning_ms: i64, critical_ms: i64) -> Self {
        Self {
            warning_ms,
            critical_ms,
        }
    }

    /// Determines alert level for a remaining time in ms
    pub fn level_for(&self, remaining_ms: i64) -> AlertLevel {
        if remaining_ms <= 0 {
            AlertLevel::Expired
        } else if remaining_ms <= self.critical_ms {
            AlertLevel::Critical
        } else if remaining_ms <= self.warning_ms {
            AlertLevel::Warning
        } else {
            AlertLevel::Normal
        }
    }
}

/// Countdown timer widget configuration
#[derive(Debug, Clone)]
pub struct CountdownTimerConfig {
    /// Timer direction
    pub direction: TimerDirection,
    /// Display format
    pub format: TimerFormat,
    /// Visual style
    pub style: TimerStyle,
    /// Initial duration in milliseconds (for countdown)
    pub duration_ms: i64,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Font size in points
    pub font_size: f32,
    /// Whether to show milliseconds
    pub show_ms: bool,
    /// Whether to flash when expired
    pub flash_on_expire: bool,
}

impl Default for CountdownTimerConfig {
    fn default() -> Self {
        Self {
            direction: TimerDirection::CountDown,
            format: TimerFormat::MinutesSeconds,
            style: TimerStyle::Digital,
            duration_ms: 60_000,
            thresholds: AlertThresholds::default(),
            width: 300,
            height: 100,
            font_size: 72.0,
            show_ms: false,
            flash_on_expire: true,
        }
    }
}

/// Main countdown timer widget
#[derive(Debug)]
pub struct CountdownTimer {
    /// Configuration
    pub config: CountdownTimerConfig,
    /// Current state
    pub state: TimerState,
    /// Elapsed time in milliseconds (always counts up internally)
    elapsed_ms: i64,
    /// Target time in ms for completion callback
    target_ms: Option<i64>,
    /// Label text shown above/below the timer
    pub label: Option<String>,
    /// Flash state for expired animation
    flash_visible: bool,
    /// Frame count for flash timing
    flash_frame: u32,
}

impl CountdownTimer {
    /// Creates a new countdown timer with default config
    pub fn new() -> Self {
        Self {
            config: CountdownTimerConfig::default(),
            state: TimerState::Idle,
            elapsed_ms: 0,
            target_ms: None,
            label: None,
            flash_visible: true,
            flash_frame: 0,
        }
    }

    /// Creates a countdown timer with a specific duration
    pub fn with_duration(duration_ms: i64) -> Self {
        let mut timer = Self::new();
        timer.config.duration_ms = duration_ms;
        timer
    }

    /// Creates a count-up timer
    pub fn count_up() -> Self {
        let mut timer = Self::new();
        timer.config.direction = TimerDirection::CountUp;
        timer
    }

    /// Starts the timer
    pub fn start(&mut self) {
        self.state = TimerState::Running;
    }

    /// Pauses the timer
    pub fn pause(&mut self) {
        if self.state == TimerState::Running {
            self.state = TimerState::Paused;
        }
    }

    /// Resumes a paused timer
    pub fn resume(&mut self) {
        if self.state == TimerState::Paused {
            self.state = TimerState::Running;
        }
    }

    /// Resets the timer to initial state
    pub fn reset(&mut self) {
        self.elapsed_ms = 0;
        self.state = TimerState::Idle;
        self.flash_visible = true;
        self.flash_frame = 0;
    }

    /// Advances the timer by the given delta in milliseconds
    pub fn tick(&mut self, delta_ms: i64) {
        if self.state != TimerState::Running {
            return;
        }
        self.elapsed_ms += delta_ms;

        // Check completion for countdown
        if self.config.direction == TimerDirection::CountDown
            && self.elapsed_ms >= self.config.duration_ms
        {
            self.state = TimerState::Finished;
        }

        // Update flash state
        self.flash_frame = self.flash_frame.wrapping_add(1);
        if self.flash_frame % 30 == 0 {
            self.flash_visible = !self.flash_visible;
        }
    }

    /// Returns the current display time in milliseconds
    pub fn display_ms(&self) -> i64 {
        match self.config.direction {
            TimerDirection::CountDown => self.config.duration_ms - self.elapsed_ms,
            TimerDirection::CountUp => self.elapsed_ms,
        }
    }

    /// Returns the formatted time string
    pub fn display_string(&self) -> String {
        self.config.format.format(self.display_ms())
    }

    /// Returns the current alert level
    pub fn alert_level(&self) -> AlertLevel {
        match self.config.direction {
            TimerDirection::CountDown => self.config.thresholds.level_for(self.display_ms()),
            TimerDirection::CountUp => AlertLevel::Normal,
        }
    }

    /// Returns whether the timer should be visible (handles flashing)
    pub fn is_visible(&self) -> bool {
        if self.state == TimerState::Finished && self.config.flash_on_expire {
            self.flash_visible
        } else {
            true
        }
    }

    /// Returns the progress ratio from 0.0 to 1.0
    pub fn progress(&self) -> f64 {
        if self.config.duration_ms <= 0 {
            return 0.0;
        }
        match self.config.direction {
            TimerDirection::CountDown => {
                let remaining = (self.config.duration_ms - self.elapsed_ms).max(0);
                remaining as f64 / self.config.duration_ms as f64
            }
            TimerDirection::CountUp => {
                if let Some(target) = self.target_ms {
                    (self.elapsed_ms as f64 / target as f64).min(1.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Sets a target for count-up progress display
    pub fn set_target(&mut self, target_ms: i64) {
        self.target_ms = Some(target_ms);
    }

    /// Sets the timer label
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }

    /// Returns elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> i64 {
        self.elapsed_ms
    }
}

impl Default for CountdownTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// A preset timer for common broadcast scenarios
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerPreset {
    /// Commercial break (typically 30, 60, or 120 seconds)
    CommercialBreak(u32),
    /// Half-time show countdown
    HalfTimeShow,
    /// News segment timing
    NewsSegment,
    /// Sports clock (counts up)
    SportsClock,
    /// Voting deadline countdown
    VotingDeadline(u32),
}

impl TimerPreset {
    /// Creates a `CountdownTimer` configured for this preset
    pub fn create_timer(&self) -> CountdownTimer {
        let mut timer = CountdownTimer::new();
        match self {
            TimerPreset::CommercialBreak(secs) => {
                timer.config.duration_ms = *secs as i64 * 1000;
                timer.config.format = TimerFormat::SecondsOnly;
                timer.set_label("COMMERCIAL");
            }
            TimerPreset::HalfTimeShow => {
                timer.config.duration_ms = 20 * 60 * 1000; // 20 minutes
                timer.config.format = TimerFormat::MinutesSeconds;
                timer.set_label("HALF TIME");
            }
            TimerPreset::NewsSegment => {
                timer.config.duration_ms = 3 * 60 * 1000; // 3 minutes
                timer.config.format = TimerFormat::MinutesSecondsTenths;
                timer.set_label("SEGMENT");
            }
            TimerPreset::SportsClock => {
                timer.config.direction = TimerDirection::CountUp;
                timer.config.format = TimerFormat::MinutesSeconds;
                timer.set_label("GAME CLOCK");
            }
            TimerPreset::VotingDeadline(mins) => {
                timer.config.duration_ms = *mins as i64 * 60 * 1000;
                timer.config.format = TimerFormat::MinutesSeconds;
                timer.set_label("VOTING CLOSES");
            }
        }
        timer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_format_hours_minutes_seconds() {
        let fmt = TimerFormat::HoursMinutesSeconds;
        assert_eq!(fmt.format(3_661_000), "01:01:01");
        assert_eq!(fmt.format(0), "00:00:00");
    }

    #[test]
    fn test_timer_format_minutes_seconds() {
        let fmt = TimerFormat::MinutesSeconds;
        assert_eq!(fmt.format(90_000), "01:30");
        assert_eq!(fmt.format(0), "00:00");
    }

    #[test]
    fn test_timer_format_seconds_only() {
        let fmt = TimerFormat::SecondsOnly;
        assert_eq!(fmt.format(30_000), "30");
        assert_eq!(fmt.format(0), "0");
    }

    #[test]
    fn test_timer_format_negative() {
        let fmt = TimerFormat::MinutesSeconds;
        let s = fmt.format(-5_000);
        assert!(s.starts_with('-'));
    }

    #[test]
    fn test_timer_format_tenths() {
        let fmt = TimerFormat::MinutesSecondsTenths;
        let s = fmt.format(5_500);
        assert!(s.contains('.'));
    }

    #[test]
    fn test_alert_level_suggested_color() {
        let normal_color = AlertLevel::Normal.suggested_color();
        let critical_color = AlertLevel::Critical.suggested_color();
        assert_ne!(normal_color, critical_color);
        assert_eq!(AlertLevel::Expired.suggested_color(), [255, 0, 0, 255]);
    }

    #[test]
    fn test_alert_thresholds_level_for() {
        let thresholds = AlertThresholds::new(30_000, 10_000);
        assert_eq!(thresholds.level_for(60_000), AlertLevel::Normal);
        assert_eq!(thresholds.level_for(20_000), AlertLevel::Warning);
        assert_eq!(thresholds.level_for(5_000), AlertLevel::Critical);
        assert_eq!(thresholds.level_for(0), AlertLevel::Expired);
        assert_eq!(thresholds.level_for(-1), AlertLevel::Expired);
    }

    #[test]
    fn test_countdown_timer_new() {
        let timer = CountdownTimer::new();
        assert_eq!(timer.state, TimerState::Idle);
        assert_eq!(timer.elapsed_ms(), 0);
    }

    #[test]
    fn test_countdown_timer_start_pause_resume() {
        let mut timer = CountdownTimer::new();
        timer.start();
        assert_eq!(timer.state, TimerState::Running);
        timer.pause();
        assert_eq!(timer.state, TimerState::Paused);
        timer.resume();
        assert_eq!(timer.state, TimerState::Running);
    }

    #[test]
    fn test_countdown_timer_tick() {
        let mut timer = CountdownTimer::with_duration(60_000);
        timer.start();
        timer.tick(5_000);
        assert_eq!(timer.elapsed_ms(), 5_000);
        assert_eq!(timer.display_ms(), 55_000);
    }

    #[test]
    fn test_countdown_timer_completion() {
        let mut timer = CountdownTimer::with_duration(10_000);
        timer.start();
        timer.tick(10_000);
        assert_eq!(timer.state, TimerState::Finished);
    }

    #[test]
    fn test_countdown_timer_reset() {
        let mut timer = CountdownTimer::with_duration(10_000);
        timer.start();
        timer.tick(5_000);
        timer.reset();
        assert_eq!(timer.state, TimerState::Idle);
        assert_eq!(timer.elapsed_ms(), 0);
    }

    #[test]
    fn test_countdown_timer_display_string() {
        let mut timer = CountdownTimer::with_duration(90_000);
        timer.start();
        timer.tick(30_000);
        let s = timer.display_string();
        assert_eq!(s, "01:00");
    }

    #[test]
    fn test_countdown_timer_alert_level() {
        let mut timer = CountdownTimer::with_duration(60_000);
        timer.start();
        assert_eq!(timer.alert_level(), AlertLevel::Normal);
        timer.tick(35_000); // 25s remaining
        assert_eq!(timer.alert_level(), AlertLevel::Warning);
        timer.tick(16_000); // 9s remaining
        assert_eq!(timer.alert_level(), AlertLevel::Critical);
    }

    #[test]
    fn test_countdown_timer_progress() {
        let mut timer = CountdownTimer::with_duration(100_000);
        timer.start();
        timer.tick(25_000);
        let progress = timer.progress();
        assert!((progress - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_count_up_timer() {
        let mut timer = CountdownTimer::count_up();
        timer.start();
        timer.tick(5_000);
        assert_eq!(timer.display_ms(), 5_000);
        assert_eq!(timer.alert_level(), AlertLevel::Normal);
    }

    #[test]
    fn test_count_up_timer_progress_with_target() {
        let mut timer = CountdownTimer::count_up();
        timer.set_target(100_000);
        timer.start();
        timer.tick(50_000);
        let progress = timer.progress();
        assert!((progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_timer_no_tick_when_paused() {
        let mut timer = CountdownTimer::with_duration(60_000);
        timer.start();
        timer.tick(5_000);
        timer.pause();
        timer.tick(10_000); // Should not advance
        assert_eq!(timer.elapsed_ms(), 5_000);
    }

    #[test]
    fn test_timer_preset_commercial_break() {
        let timer = TimerPreset::CommercialBreak(30).create_timer();
        assert_eq!(timer.config.duration_ms, 30_000);
        assert!(timer.label.is_some());
    }

    #[test]
    fn test_timer_preset_sports_clock() {
        let timer = TimerPreset::SportsClock.create_timer();
        assert_eq!(timer.config.direction, TimerDirection::CountUp);
    }

    #[test]
    fn test_timer_preset_voting_deadline() {
        let timer = TimerPreset::VotingDeadline(15).create_timer();
        assert_eq!(timer.config.duration_ms, 15 * 60 * 1000);
    }

    #[test]
    fn test_timer_set_label() {
        let mut timer = CountdownTimer::new();
        timer.set_label("TEST LABEL");
        assert_eq!(timer.label.as_deref(), Some("TEST LABEL"));
    }
}
