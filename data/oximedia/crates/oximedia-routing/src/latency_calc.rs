#![allow(dead_code)]
//! Latency calculation and budgeting for media routing paths.
//!
//! Models the end-to-end latency of a signal path by summing the
//! contributions of individual processing stages (codec, network hop,
//! buffer, etc.). Provides helpers for budget validation, worst-case
//! analysis, and jitter estimation.

use std::fmt;

/// Category of a latency contributor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatencyKind {
    /// Codec encode/decode latency.
    Codec,
    /// Network transport latency.
    Network,
    /// Buffer / queue latency.
    Buffer,
    /// Processing stage (e.g., effects, mixing).
    Processing,
    /// Display / output device latency.
    Display,
    /// Other / miscellaneous.
    Other,
}

impl fmt::Display for LatencyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec => write!(f, "Codec"),
            Self::Network => write!(f, "Network"),
            Self::Buffer => write!(f, "Buffer"),
            Self::Processing => write!(f, "Processing"),
            Self::Display => write!(f, "Display"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// A single latency contributor in a signal path.
#[derive(Debug, Clone)]
pub struct LatencyStage {
    /// Human-readable name of the stage.
    pub name: String,
    /// Category.
    pub kind: LatencyKind,
    /// Nominal (typical) latency in microseconds.
    pub nominal_us: f64,
    /// Worst-case latency in microseconds.
    pub worst_us: f64,
    /// Jitter (variation) in microseconds.
    pub jitter_us: f64,
}

impl LatencyStage {
    /// Create a new latency stage.
    pub fn new(name: &str, kind: LatencyKind, nominal_us: f64, worst_us: f64) -> Self {
        Self {
            name: name.to_owned(),
            kind,
            nominal_us,
            worst_us,
            jitter_us: worst_us - nominal_us,
        }
    }

    /// Create a fixed-latency stage (zero jitter).
    pub fn fixed(name: &str, kind: LatencyKind, latency_us: f64) -> Self {
        Self {
            name: name.to_owned(),
            kind,
            nominal_us: latency_us,
            worst_us: latency_us,
            jitter_us: 0.0,
        }
    }

    /// Nominal latency in milliseconds.
    pub fn nominal_ms(&self) -> f64 {
        self.nominal_us / 1_000.0
    }

    /// Worst-case latency in milliseconds.
    pub fn worst_ms(&self) -> f64 {
        self.worst_us / 1_000.0
    }
}

impl fmt::Display for LatencyStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}]: {:.1}us nom / {:.1}us worst",
            self.name, self.kind, self.nominal_us, self.worst_us
        )
    }
}

/// Result of a latency budget check.
#[derive(Debug, Clone)]
pub struct BudgetResult {
    /// Whether the path meets the budget.
    pub within_budget: bool,
    /// Budget limit in microseconds.
    pub budget_us: f64,
    /// Total nominal latency in microseconds.
    pub total_nominal_us: f64,
    /// Total worst-case latency in microseconds.
    pub total_worst_us: f64,
    /// Margin (budget minus worst-case) in microseconds.
    pub margin_us: f64,
}

/// Per-kind breakdown entry.
#[derive(Debug, Clone)]
pub struct KindBreakdown {
    /// The kind.
    pub kind: LatencyKind,
    /// Sum of nominal latency in microseconds.
    pub nominal_us: f64,
    /// Sum of worst-case latency in microseconds.
    pub worst_us: f64,
    /// Number of stages of this kind.
    pub count: usize,
}

/// A latency calculator that models a complete signal path.
#[derive(Debug, Clone)]
pub struct LatencyCalc {
    /// Name of the path being modeled.
    name: String,
    /// Ordered list of stages along the path.
    stages: Vec<LatencyStage>,
}

impl LatencyCalc {
    /// Create a new calculator for the named path.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            stages: Vec::new(),
        }
    }

    /// Add a stage to the path (appended at the end).
    pub fn add_stage(&mut self, stage: LatencyStage) {
        self.stages.push(stage);
    }

    /// Number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Read-only access to stages.
    pub fn stages(&self) -> &[LatencyStage] {
        &self.stages
    }

    /// Total nominal latency in microseconds.
    pub fn total_nominal_us(&self) -> f64 {
        self.stages.iter().map(|s| s.nominal_us).sum()
    }

    /// Total worst-case latency in microseconds.
    pub fn total_worst_us(&self) -> f64 {
        self.stages.iter().map(|s| s.worst_us).sum()
    }

    /// Total nominal latency in milliseconds.
    pub fn total_nominal_ms(&self) -> f64 {
        self.total_nominal_us() / 1_000.0
    }

    /// Total worst-case latency in milliseconds.
    pub fn total_worst_ms(&self) -> f64 {
        self.total_worst_us() / 1_000.0
    }

    /// Root-sum-square jitter estimate in microseconds.
    pub fn rss_jitter_us(&self) -> f64 {
        let sum_sq: f64 = self.stages.iter().map(|s| s.jitter_us * s.jitter_us).sum();
        sum_sq.sqrt()
    }

    /// Check whether the path fits within the given budget (microseconds).
    pub fn check_budget(&self, budget_us: f64) -> BudgetResult {
        let total_worst = self.total_worst_us();
        BudgetResult {
            within_budget: total_worst <= budget_us,
            budget_us,
            total_nominal_us: self.total_nominal_us(),
            total_worst_us: total_worst,
            margin_us: budget_us - total_worst,
        }
    }

    /// Break down latency by kind.
    pub fn breakdown_by_kind(&self) -> Vec<KindBreakdown> {
        use std::collections::HashMap;
        let mut map: HashMap<LatencyKind, (f64, f64, usize)> = HashMap::new();
        for s in &self.stages {
            let entry = map.entry(s.kind).or_insert((0.0, 0.0, 0));
            entry.0 += s.nominal_us;
            entry.1 += s.worst_us;
            entry.2 += 1;
        }
        let mut out: Vec<KindBreakdown> = map
            .into_iter()
            .map(|(kind, (nom, worst, count))| KindBreakdown {
                kind,
                nominal_us: nom,
                worst_us: worst,
                count,
            })
            .collect();
        out.sort_by(|a, b| {
            b.worst_us
                .partial_cmp(&a.worst_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// Return the stage with the highest worst-case latency.
    pub fn bottleneck(&self) -> Option<&LatencyStage> {
        self.stages.iter().max_by(|a, b| {
            a.worst_us
                .partial_cmp(&b.worst_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Estimate number of video frames of latency at the given frame rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn frames_of_latency(&self, fps: f64) -> f64 {
        let total_sec = self.total_worst_us() / 1_000_000.0;
        total_sec * fps
    }

    /// Path name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        let mut lines = vec![format!("Latency Report: {}", self.name)];
        for (i, s) in self.stages.iter().enumerate() {
            lines.push(format!("  [{i}] {s}"));
        }
        lines.push(format!(
            "Total: {:.1}us nominal / {:.1}us worst / {:.1}us RSS-jitter",
            self.total_nominal_us(),
            self.total_worst_us(),
            self.rss_jitter_us(),
        ));
        lines.join("\n")
    }

    /// Clear all stages.
    pub fn clear(&mut self) {
        self.stages.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_new() {
        let s = LatencyStage::new("Encoder", LatencyKind::Codec, 1000.0, 1500.0);
        assert_eq!(s.name, "Encoder");
        assert_eq!(s.kind, LatencyKind::Codec);
        assert!((s.jitter_us - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_stage_fixed() {
        let s = LatencyStage::fixed("Buffer", LatencyKind::Buffer, 2000.0);
        assert!((s.jitter_us).abs() < 1e-9);
        assert!((s.nominal_us - s.worst_us).abs() < 1e-9);
    }

    #[test]
    fn test_stage_ms_conversion() {
        let s = LatencyStage::fixed("X", LatencyKind::Other, 5000.0);
        assert!((s.nominal_ms() - 5.0).abs() < 1e-9);
        assert!((s.worst_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_stage_display() {
        let s = LatencyStage::new("Enc", LatencyKind::Codec, 100.0, 200.0);
        let d = format!("{s}");
        assert!(d.contains("Enc"));
        assert!(d.contains("Codec"));
    }

    #[test]
    fn test_calc_totals() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Codec, 1000.0));
        calc.add_stage(LatencyStage::fixed("B", LatencyKind::Network, 2000.0));
        assert_eq!(calc.stage_count(), 2);
        assert!((calc.total_nominal_us() - 3000.0).abs() < 1e-9);
        assert!((calc.total_worst_us() - 3000.0).abs() < 1e-9);
    }

    #[test]
    fn test_calc_ms_totals() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Codec, 10_000.0));
        assert!((calc.total_nominal_ms() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_rss_jitter() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::new("A", LatencyKind::Codec, 100.0, 400.0)); // jitter=300
        calc.add_stage(LatencyStage::new("B", LatencyKind::Network, 100.0, 500.0)); // jitter=400
                                                                                    // RSS = sqrt(300^2 + 400^2) = sqrt(250000) = 500
        assert!((calc.rss_jitter_us() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_check_budget_pass() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Codec, 1000.0));
        let result = calc.check_budget(5000.0);
        assert!(result.within_budget);
        assert!((result.margin_us - 4000.0).abs() < 1e-9);
    }

    #[test]
    fn test_check_budget_fail() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Codec, 6000.0));
        let result = calc.check_budget(5000.0);
        assert!(!result.within_budget);
        assert!(result.margin_us < 0.0);
    }

    #[test]
    fn test_bottleneck() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("Small", LatencyKind::Buffer, 100.0));
        calc.add_stage(LatencyStage::fixed("Big", LatencyKind::Codec, 5000.0));
        let bn = calc.bottleneck().expect("should succeed in test");
        assert_eq!(bn.name, "Big");
    }

    #[test]
    fn test_frames_of_latency() {
        let mut calc = LatencyCalc::new("test");
        // 40ms = 1 frame at 25fps
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Buffer, 40_000.0));
        let frames = calc.frames_of_latency(25.0);
        assert!((frames - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_breakdown_by_kind() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("Enc", LatencyKind::Codec, 1000.0));
        calc.add_stage(LatencyStage::fixed("Dec", LatencyKind::Codec, 1000.0));
        calc.add_stage(LatencyStage::fixed("Net", LatencyKind::Network, 500.0));
        let bd = calc.breakdown_by_kind();
        let codec_entry = bd
            .iter()
            .find(|e| e.kind == LatencyKind::Codec)
            .expect("should succeed in test");
        assert_eq!(codec_entry.count, 2);
        assert!((codec_entry.nominal_us - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn test_report() {
        let mut calc = LatencyCalc::new("MyPath");
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Codec, 1000.0));
        let r = calc.report();
        assert!(r.contains("MyPath"));
        assert!(r.contains("Total"));
    }

    #[test]
    fn test_clear() {
        let mut calc = LatencyCalc::new("test");
        calc.add_stage(LatencyStage::fixed("A", LatencyKind::Codec, 1000.0));
        calc.clear();
        assert_eq!(calc.stage_count(), 0);
        assert!((calc.total_nominal_us()).abs() < 1e-9);
    }

    #[test]
    fn test_latency_kind_display() {
        assert_eq!(format!("{}", LatencyKind::Codec), "Codec");
        assert_eq!(format!("{}", LatencyKind::Network), "Network");
        assert_eq!(format!("{}", LatencyKind::Display), "Display");
    }
}
