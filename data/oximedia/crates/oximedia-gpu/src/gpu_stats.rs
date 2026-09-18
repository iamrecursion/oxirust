#![allow(dead_code)]
//! GPU statistics collection and monitoring.

/// A measurable GPU statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuStat {
    /// Core utilization (0–100 %).
    Utilization,
    /// VRAM currently in use (bytes).
    MemoryUsed,
    /// Die temperature (degrees Celsius).
    Temperature,
    /// Board power draw (milliwatts).
    PowerDraw,
}

impl GpuStat {
    /// Return the SI/display unit for this statistic.
    #[must_use]
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Utilization => "%",
            Self::MemoryUsed => "bytes",
            Self::Temperature => "°C",
            Self::PowerDraw => "mW",
        }
    }

    /// Returns `true` if this is a percentage-based stat.
    #[must_use]
    pub fn is_percentage(&self) -> bool {
        matches!(self, Self::Utilization)
    }

    /// Returns `true` if this is a thermal stat.
    #[must_use]
    pub fn is_thermal(&self) -> bool {
        matches!(self, Self::Temperature)
    }
}

/// A single sample of a GPU statistic at a point in time.
#[derive(Debug, Clone)]
pub struct GpuStatSample {
    /// Which statistic was measured.
    pub stat: GpuStat,
    /// The measured value.
    pub value: f64,
    /// Threshold above which the value is considered critical.
    pub critical_threshold: f64,
}

impl GpuStatSample {
    /// Create a new sample.
    #[must_use]
    pub fn new(stat: GpuStat, value: f64, critical_threshold: f64) -> Self {
        Self {
            stat,
            value,
            critical_threshold,
        }
    }

    /// Returns `true` when the value exceeds the critical threshold.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.value >= self.critical_threshold
    }

    /// Returns how far the value is from the critical threshold (negative = safe).
    #[must_use]
    pub fn headroom(&self) -> f64 {
        self.critical_threshold - self.value
    }
}

/// Accumulated GPU statistics over a recording period.
#[derive(Debug, Default)]
pub struct GpuStats {
    utilization_samples: Vec<f64>,
    memory_used_samples: Vec<u64>,
    temperature_samples: Vec<f64>,
    power_draw_samples: Vec<f64>,
    total_memory_bytes: u64,
}

impl GpuStats {
    /// Create a new collector knowing total VRAM.
    #[must_use]
    pub fn new(total_memory_bytes: u64) -> Self {
        Self {
            total_memory_bytes,
            ..Default::default()
        }
    }

    /// Record a [`GpuStatSample`].
    pub fn record(&mut self, sample: &GpuStatSample) {
        match sample.stat {
            GpuStat::Utilization => self.utilization_samples.push(sample.value),
            GpuStat::MemoryUsed => self.memory_used_samples.push(sample.value as u64),
            GpuStat::Temperature => self.temperature_samples.push(sample.value),
            GpuStat::PowerDraw => self.power_draw_samples.push(sample.value),
        }
    }

    /// Average utilization percentage over all recorded samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn utilization_pct(&self) -> f64 {
        if self.utilization_samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.utilization_samples.iter().sum();
        sum / self.utilization_samples.len() as f64
    }

    /// Average memory usage as a percentage of total VRAM.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn memory_pct(&self) -> f64 {
        if self.memory_used_samples.is_empty() || self.total_memory_bytes == 0 {
            return 0.0;
        }
        let sum: u64 = self.memory_used_samples.iter().sum();
        let avg = sum as f64 / self.memory_used_samples.len() as f64;
        (avg / self.total_memory_bytes as f64) * 100.0
    }

    /// Peak temperature recorded.
    #[must_use]
    pub fn peak_temperature(&self) -> Option<f64> {
        self.temperature_samples.iter().copied().reduce(f64::max)
    }

    /// Average power draw in milliwatts.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_power_draw_mw(&self) -> f64 {
        if self.power_draw_samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.power_draw_samples.iter().sum();
        sum / self.power_draw_samples.len() as f64
    }

    /// Total number of recorded samples across all stat types.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.utilization_samples.len()
            + self.memory_used_samples.len()
            + self.temperature_samples.len()
            + self.power_draw_samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_stat_unit_utilization() {
        assert_eq!(GpuStat::Utilization.unit(), "%");
    }

    #[test]
    fn test_gpu_stat_unit_memory() {
        assert_eq!(GpuStat::MemoryUsed.unit(), "bytes");
    }

    #[test]
    fn test_gpu_stat_unit_temperature() {
        assert_eq!(GpuStat::Temperature.unit(), "°C");
    }

    #[test]
    fn test_gpu_stat_unit_power() {
        assert_eq!(GpuStat::PowerDraw.unit(), "mW");
    }

    #[test]
    fn test_gpu_stat_is_percentage() {
        assert!(GpuStat::Utilization.is_percentage());
        assert!(!GpuStat::MemoryUsed.is_percentage());
        assert!(!GpuStat::Temperature.is_percentage());
    }

    #[test]
    fn test_gpu_stat_is_thermal() {
        assert!(GpuStat::Temperature.is_thermal());
        assert!(!GpuStat::Utilization.is_thermal());
    }

    #[test]
    fn test_sample_is_critical_true() {
        let s = GpuStatSample::new(GpuStat::Temperature, 95.0, 90.0);
        assert!(s.is_critical());
    }

    #[test]
    fn test_sample_is_critical_false() {
        let s = GpuStatSample::new(GpuStat::Temperature, 75.0, 90.0);
        assert!(!s.is_critical());
    }

    #[test]
    fn test_sample_is_critical_at_threshold() {
        let s = GpuStatSample::new(GpuStat::Utilization, 90.0, 90.0);
        assert!(s.is_critical());
    }

    #[test]
    fn test_sample_headroom() {
        let s = GpuStatSample::new(GpuStat::PowerDraw, 200.0, 250.0);
        assert!((s.headroom() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_utilization_pct_empty() {
        let stats = GpuStats::new(8 * 1024 * 1024 * 1024);
        assert!((stats.utilization_pct() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_record_and_utilization_pct() {
        let mut stats = GpuStats::new(8 * 1024 * 1024 * 1024);
        stats.record(&GpuStatSample::new(GpuStat::Utilization, 80.0, 100.0));
        stats.record(&GpuStatSample::new(GpuStat::Utilization, 60.0, 100.0));
        assert!((stats.utilization_pct() - 70.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_memory_pct() {
        let total = 8_000_000_000u64;
        let mut stats = GpuStats::new(total);
        stats.record(&GpuStatSample::new(
            GpuStat::MemoryUsed,
            4_000_000_000.0,
            f64::MAX,
        ));
        let pct = stats.memory_pct();
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_peak_temperature() {
        let mut stats = GpuStats::new(0);
        assert!(stats.peak_temperature().is_none());
        stats.record(&GpuStatSample::new(GpuStat::Temperature, 60.0, 100.0));
        stats.record(&GpuStatSample::new(GpuStat::Temperature, 85.0, 100.0));
        assert_eq!(stats.peak_temperature(), Some(85.0));
    }

    #[test]
    fn test_stats_avg_power_draw() {
        let mut stats = GpuStats::new(0);
        stats.record(&GpuStatSample::new(GpuStat::PowerDraw, 100.0, 300.0));
        stats.record(&GpuStatSample::new(GpuStat::PowerDraw, 200.0, 300.0));
        assert!((stats.avg_power_draw_mw() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_sample_count() {
        let mut stats = GpuStats::new(0);
        stats.record(&GpuStatSample::new(GpuStat::Utilization, 50.0, 100.0));
        stats.record(&GpuStatSample::new(GpuStat::Temperature, 70.0, 90.0));
        assert_eq!(stats.sample_count(), 2);
    }
}
