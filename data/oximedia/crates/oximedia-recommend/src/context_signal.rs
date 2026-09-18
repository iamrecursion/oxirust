#![allow(dead_code)]
//! Contextual signals for context-aware media recommendations.
//!
//! Enriches recommendation scoring with real-time contextual data such as
//! time of day, day of week, device type, network quality, user location,
//! and seasonal patterns. Signals are normalized to [0, 1] and combined
//! into a composite context score that modulates base recommendation scores.

use std::collections::HashMap;

/// Type of contextual signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalKind {
    /// Time-of-day signal (morning/afternoon/evening/night).
    TimeOfDay,
    /// Day-of-week signal (weekday vs weekend).
    DayOfWeek,
    /// Device type signal (mobile, tablet, desktop, TV).
    DeviceType,
    /// Network quality signal (bandwidth tier).
    NetworkQuality,
    /// Geographic region signal.
    Region,
    /// Season/holiday signal.
    Season,
    /// Content recency signal.
    Recency,
}

impl std::fmt::Display for SignalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimeOfDay => write!(f, "TimeOfDay"),
            Self::DayOfWeek => write!(f, "DayOfWeek"),
            Self::DeviceType => write!(f, "DeviceType"),
            Self::NetworkQuality => write!(f, "NetworkQuality"),
            Self::Region => write!(f, "Region"),
            Self::Season => write!(f, "Season"),
            Self::Recency => write!(f, "Recency"),
        }
    }
}

/// Normalized contextual signal value in [0, 1].
#[derive(Debug, Clone, Copy)]
pub struct SignalValue {
    /// The signal kind.
    pub kind: SignalKind,
    /// Normalized value (0.0 = lowest relevance, 1.0 = highest).
    pub value: f64,
    /// Confidence in this signal (0.0-1.0).
    pub confidence: f64,
}

impl SignalValue {
    /// Create a new signal value, clamping to \[0,1\].
    #[must_use]
    pub fn new(kind: SignalKind, value: f64, confidence: f64) -> Self {
        Self {
            kind,
            value: value.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Effective value weighted by confidence.
    #[must_use]
    pub fn effective(&self) -> f64 {
        self.value * self.confidence
    }
}

/// Time period categories for time-of-day signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePeriod {
    /// Early morning (5-8).
    EarlyMorning,
    /// Morning (8-12).
    Morning,
    /// Afternoon (12-17).
    Afternoon,
    /// Evening (17-21).
    Evening,
    /// Late night (21-5).
    LateNight,
}

impl TimePeriod {
    /// Classify an hour (0-23) into a time period.
    #[must_use]
    pub fn from_hour(hour: u8) -> Self {
        match hour {
            5..=7 => Self::EarlyMorning,
            8..=11 => Self::Morning,
            12..=16 => Self::Afternoon,
            17..=20 => Self::Evening,
            _ => Self::LateNight,
        }
    }

    /// Returns a base signal value for content consumption likelihood.
    /// Evening and late night are peak viewing times.
    #[must_use]
    pub fn consumption_signal(self) -> f64 {
        match self {
            Self::EarlyMorning => 0.3,
            Self::Morning => 0.5,
            Self::Afternoon => 0.6,
            Self::Evening => 0.9,
            Self::LateNight => 0.8,
        }
    }
}

impl std::fmt::Display for TimePeriod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EarlyMorning => write!(f, "EarlyMorning"),
            Self::Morning => write!(f, "Morning"),
            Self::Afternoon => write!(f, "Afternoon"),
            Self::Evening => write!(f, "Evening"),
            Self::LateNight => write!(f, "LateNight"),
        }
    }
}

/// Device categories for device-type signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCategory {
    /// Mobile phone.
    Mobile,
    /// Tablet.
    Tablet,
    /// Desktop/laptop.
    Desktop,
    /// Smart TV or set-top box.
    Tv,
}

impl DeviceCategory {
    /// Returns a content-length preference signal.
    /// Mobile users prefer shorter content; TV users prefer longer.
    #[must_use]
    pub fn length_preference_signal(self) -> f64 {
        match self {
            Self::Mobile => 0.3,
            Self::Tablet => 0.5,
            Self::Desktop => 0.7,
            Self::Tv => 0.9,
        }
    }

    /// Returns a quality preference signal.
    /// TV and desktop users prefer higher quality.
    #[must_use]
    pub fn quality_preference_signal(self) -> f64 {
        match self {
            Self::Mobile => 0.4,
            Self::Tablet => 0.6,
            Self::Desktop => 0.8,
            Self::Tv => 1.0,
        }
    }
}

impl std::fmt::Display for DeviceCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mobile => write!(f, "Mobile"),
            Self::Tablet => write!(f, "Tablet"),
            Self::Desktop => write!(f, "Desktop"),
            Self::Tv => write!(f, "TV"),
        }
    }
}

/// Weight configuration for combining signals.
#[derive(Debug, Clone)]
pub struct SignalWeights {
    /// Weights per signal kind.
    pub weights: HashMap<SignalKind, f64>,
}

impl SignalWeights {
    /// Create equal weights for all signal kinds.
    #[must_use]
    pub fn equal() -> Self {
        let mut weights = HashMap::new();
        let kinds = [
            SignalKind::TimeOfDay,
            SignalKind::DayOfWeek,
            SignalKind::DeviceType,
            SignalKind::NetworkQuality,
            SignalKind::Region,
            SignalKind::Season,
            SignalKind::Recency,
        ];
        for kind in &kinds {
            weights.insert(*kind, 1.0);
        }
        Self { weights }
    }

    /// Set the weight for a specific signal kind.
    pub fn set_weight(&mut self, kind: SignalKind, weight: f64) {
        self.weights.insert(kind, weight.max(0.0));
    }

    /// Get the weight for a signal kind (defaults to 0.0 if not set).
    #[must_use]
    pub fn get_weight(&self, kind: SignalKind) -> f64 {
        self.weights.get(&kind).copied().unwrap_or(0.0)
    }

    /// Total weight (for normalization).
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.weights.values().sum()
    }
}

impl Default for SignalWeights {
    fn default() -> Self {
        Self::equal()
    }
}

/// A collection of contextual signals for a single recommendation request.
#[derive(Debug, Clone)]
pub struct ContextSignals {
    /// Individual signal values.
    pub signals: Vec<SignalValue>,
}

impl ContextSignals {
    /// Create an empty signal collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            signals: Vec::new(),
        }
    }

    /// Add a signal.
    pub fn add(&mut self, signal: SignalValue) {
        self.signals.push(signal);
    }

    /// Add a time-of-day signal from an hour value.
    pub fn add_time_of_day(&mut self, hour: u8) {
        let period = TimePeriod::from_hour(hour);
        self.signals.push(SignalValue::new(
            SignalKind::TimeOfDay,
            period.consumption_signal(),
            1.0,
        ));
    }

    /// Add a device type signal.
    pub fn add_device(&mut self, device: DeviceCategory) {
        self.signals.push(SignalValue::new(
            SignalKind::DeviceType,
            device.length_preference_signal(),
            1.0,
        ));
    }

    /// Add a weekend vs weekday signal (true = weekend).
    pub fn add_day_of_week(&mut self, is_weekend: bool) {
        let value = if is_weekend { 0.9 } else { 0.5 };
        self.signals
            .push(SignalValue::new(SignalKind::DayOfWeek, value, 1.0));
    }

    /// Add a recency signal (days since publication; newer = higher signal).
    #[allow(clippy::cast_precision_loss)]
    pub fn add_recency(&mut self, days_old: u32, max_days: u32) {
        let value = if max_days == 0 {
            1.0
        } else {
            1.0 - (f64::from(days_old.min(max_days)) / f64::from(max_days))
        };
        self.signals
            .push(SignalValue::new(SignalKind::Recency, value, 0.9));
    }

    /// Compute a composite context score using the given weights.
    #[must_use]
    pub fn composite_score(&self, weights: &SignalWeights) -> f64 {
        let total_weight = weights.total_weight();
        if total_weight <= 0.0 || self.signals.is_empty() {
            return 0.5; // Neutral default.
        }
        let weighted_sum: f64 = self
            .signals
            .iter()
            .map(|s| s.effective() * weights.get_weight(s.kind))
            .sum();
        let used_weight: f64 = self
            .signals
            .iter()
            .map(|s| weights.get_weight(s.kind))
            .sum();
        if used_weight <= 0.0 {
            return 0.5;
        }
        (weighted_sum / used_weight).clamp(0.0, 1.0)
    }

    /// Number of signals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Whether the signal set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Get the signal value for a specific kind (first match).
    #[must_use]
    pub fn get(&self, kind: SignalKind) -> Option<&SignalValue> {
        self.signals.iter().find(|s| s.kind == kind)
    }
}

impl Default for ContextSignals {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies context modulation to base recommendation scores.
#[derive(Debug)]
pub struct ContextModulator {
    /// Signal weights.
    weights: SignalWeights,
    /// Modulation strength (0.0 = no effect, 1.0 = full effect).
    strength: f64,
}

impl ContextModulator {
    /// Create a new context modulator.
    #[must_use]
    pub fn new(weights: SignalWeights, strength: f64) -> Self {
        Self {
            weights,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Modulate a base score using contextual signals.
    ///
    /// Result = `base_score` * (1 - strength + strength * `context_score`).
    #[must_use]
    pub fn modulate(&self, base_score: f64, signals: &ContextSignals) -> f64 {
        let ctx = signals.composite_score(&self.weights);
        let factor = 1.0 - self.strength + self.strength * ctx;
        (base_score * factor).clamp(0.0, 1.0)
    }

    /// Get the modulation strength.
    #[must_use]
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// Set the modulation strength.
    pub fn set_strength(&mut self, strength: f64) {
        self.strength = strength.clamp(0.0, 1.0);
    }
}

impl Default for ContextModulator {
    fn default() -> Self {
        Self::new(SignalWeights::default(), 0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_kind_display() {
        assert_eq!(SignalKind::TimeOfDay.to_string(), "TimeOfDay");
        assert_eq!(SignalKind::DeviceType.to_string(), "DeviceType");
        assert_eq!(SignalKind::Region.to_string(), "Region");
    }

    #[test]
    fn test_signal_value_clamp() {
        let sv = SignalValue::new(SignalKind::Recency, 1.5, -0.2);
        assert!((sv.value - 1.0).abs() < f64::EPSILON);
        assert!((sv.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_signal_value_effective() {
        let sv = SignalValue::new(SignalKind::TimeOfDay, 0.8, 0.5);
        assert!((sv.effective() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_period_from_hour() {
        assert_eq!(TimePeriod::from_hour(6), TimePeriod::EarlyMorning);
        assert_eq!(TimePeriod::from_hour(10), TimePeriod::Morning);
        assert_eq!(TimePeriod::from_hour(14), TimePeriod::Afternoon);
        assert_eq!(TimePeriod::from_hour(19), TimePeriod::Evening);
        assert_eq!(TimePeriod::from_hour(23), TimePeriod::LateNight);
        assert_eq!(TimePeriod::from_hour(3), TimePeriod::LateNight);
    }

    #[test]
    fn test_time_period_consumption_signal() {
        let evening = TimePeriod::Evening.consumption_signal();
        let morning = TimePeriod::Morning.consumption_signal();
        assert!(evening > morning);
    }

    #[test]
    fn test_device_category_signals() {
        let mobile_len = DeviceCategory::Mobile.length_preference_signal();
        let tv_len = DeviceCategory::Tv.length_preference_signal();
        assert!(tv_len > mobile_len);

        let mobile_q = DeviceCategory::Mobile.quality_preference_signal();
        let tv_q = DeviceCategory::Tv.quality_preference_signal();
        assert!(tv_q > mobile_q);
    }

    #[test]
    fn test_signal_weights_equal() {
        let w = SignalWeights::equal();
        assert!((w.get_weight(SignalKind::TimeOfDay) - 1.0).abs() < f64::EPSILON);
        assert!((w.get_weight(SignalKind::Season) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_signal_weights_set() {
        let mut w = SignalWeights::equal();
        w.set_weight(SignalKind::TimeOfDay, 2.0);
        assert!((w.get_weight(SignalKind::TimeOfDay) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_signals_add_time() {
        let mut ctx = ContextSignals::new();
        ctx.add_time_of_day(19); // Evening.
        assert_eq!(ctx.len(), 1);
        let sig = ctx
            .get(SignalKind::TimeOfDay)
            .expect("should succeed in test");
        assert!((sig.value - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_signals_add_device() {
        let mut ctx = ContextSignals::new();
        ctx.add_device(DeviceCategory::Tv);
        let sig = ctx
            .get(SignalKind::DeviceType)
            .expect("should succeed in test");
        assert!((sig.value - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_signals_add_recency() {
        let mut ctx = ContextSignals::new();
        ctx.add_recency(0, 30); // Brand new.
        let sig = ctx
            .get(SignalKind::Recency)
            .expect("should succeed in test");
        assert!((sig.value - 1.0).abs() < f64::EPSILON);

        let mut ctx2 = ContextSignals::new();
        ctx2.add_recency(30, 30); // Max age.
        let sig2 = ctx2
            .get(SignalKind::Recency)
            .expect("should succeed in test");
        assert!((sig2.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_composite_score_empty() {
        let ctx = ContextSignals::new();
        let w = SignalWeights::equal();
        assert!((ctx.composite_score(&w) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_composite_score_single_signal() {
        let mut ctx = ContextSignals::new();
        ctx.add(SignalValue::new(SignalKind::TimeOfDay, 0.8, 1.0));
        let w = SignalWeights::equal();
        let score = ctx.composite_score(&w);
        assert!((score - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_modulator_no_effect() {
        let modulator = ContextModulator::new(SignalWeights::equal(), 0.0);
        let ctx = ContextSignals::new();
        let result = modulator.modulate(0.7, &ctx);
        assert!((result - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_modulator_full_effect() {
        let modulator = ContextModulator::new(SignalWeights::equal(), 1.0);
        let mut ctx = ContextSignals::new();
        ctx.add(SignalValue::new(SignalKind::TimeOfDay, 1.0, 1.0));
        let result = modulator.modulate(0.5, &ctx);
        // factor = 1.0 - 1.0 + 1.0 * 1.0 = 1.0, result = 0.5 * 1.0 = 0.5
        assert!((result - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_device_category_display() {
        assert_eq!(DeviceCategory::Mobile.to_string(), "Mobile");
        assert_eq!(DeviceCategory::Tv.to_string(), "TV");
    }
}
