//! SFP+ transceiver monitoring for broadcast video-over-IP infrastructure.
//!
//! Provides types and utilities for monitoring SFP+ optical transceivers used
//! in professional broadcast networks (10GbE, 25GbE). Covers:
//!
//! - Transceiver identity (vendor, part number, serial)
//! - Digital Optical Monitoring (DOM / DDM) — temperature, voltage, optical power, bias current
//! - Threshold-based alarm and warning detection per SFF-8472 / SFF-8636

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{VideoIpError, VideoIpResult};

// ─── Connector type ──────────────────────────────────────────────────────────

/// Physical connector type of an SFP/SFP+ module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectorType {
    /// LC duplex (most common for 10G SFP+).
    Lc,
    /// SC duplex.
    Sc,
    /// ST connector.
    St,
    /// RJ45 copper.
    Rj45,
    /// Unknown or vendor-specific connector.
    Unknown,
}

impl ConnectorType {
    /// Returns a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Lc => "LC",
            Self::Sc => "SC",
            Self::St => "ST",
            Self::Rj45 => "RJ45",
            Self::Unknown => "Unknown",
        }
    }
}

// ─── Transceiver identity ────────────────────────────────────────────────────

/// Identity information for an SFP+ transceiver module.
#[derive(Debug, Clone)]
pub struct SfpTransceiver {
    /// Logical module identifier (slot number, 0-based).
    pub module_id: u8,
    /// Vendor name (up to 16 ASCII characters per SFF-8472).
    pub vendor: String,
    /// Vendor part number (up to 16 ASCII characters).
    pub part_number: String,
    /// Module serial number (up to 16 ASCII characters).
    pub serial: String,
    /// Physical connector type.
    pub connector_type: ConnectorType,
    /// Nominal wavelength in nanometres (e.g. 850, 1310, 1550).
    pub wavelength_nm: u16,
    /// Rated maximum fibre distance in metres.
    pub max_distance_m: u32,
}

impl SfpTransceiver {
    /// Creates a new `SfpTransceiver`.
    #[must_use]
    pub fn new(
        module_id: u8,
        vendor: impl Into<String>,
        part_number: impl Into<String>,
        serial: impl Into<String>,
        connector_type: ConnectorType,
        wavelength_nm: u16,
        max_distance_m: u32,
    ) -> Self {
        Self {
            module_id,
            vendor: vendor.into(),
            part_number: part_number.into(),
            serial: serial.into(),
            connector_type,
            wavelength_nm,
            max_distance_m,
        }
    }

    /// Constructs a typical 10GBase-SR SFP+ module for testing.
    #[must_use]
    pub fn typical_10g_sr(module_id: u8) -> Self {
        Self::new(
            module_id,
            "FINISAR CORP",
            "FTLX8574D3BCL",
            format!("PTE1234{module_id:02X}"),
            ConnectorType::Lc,
            850,
            300,
        )
    }
}

// ─── Diagnostics ────────────────────────────────────────────────────────────

/// Digital Optical Monitoring (DOM) diagnostics snapshot for an SFP+ module.
///
/// All power values are in dBm; bias current in mA.
#[derive(Debug, Clone, Copy)]
pub struct SfpDiagnostics {
    /// Module case temperature in °C.
    pub temperature_c: f32,
    /// Supply voltage in V (nominal 3.3 V).
    pub voltage_v: f32,
    /// Transmit optical power in dBm.
    pub tx_power_dbm: f32,
    /// Receive optical power in dBm.
    pub rx_power_dbm: f32,
    /// Transmit laser bias current in mA.
    pub tx_bias_ma: f32,
}

impl SfpDiagnostics {
    /// Constructs a snapshot with nominal healthy values for a 10G SR module.
    #[must_use]
    pub fn nominal() -> Self {
        Self {
            temperature_c: 42.5,
            voltage_v: 3.30,
            tx_power_dbm: -2.5,
            rx_power_dbm: -5.0,
            tx_bias_ma: 6.5,
        }
    }

    /// Constructs a simulated diagnostics snapshot by varying `nominal()` with
    /// a deterministic pseudo-jitter derived from `seed`.
    #[must_use]
    pub fn simulated(seed: u8) -> Self {
        let jitter = f32::from(seed) * 0.01;
        Self {
            temperature_c: 40.0 + jitter * 50.0,
            voltage_v: 3.28 + jitter * 0.05,
            tx_power_dbm: -3.0 + jitter * 2.0,
            rx_power_dbm: -6.0 + jitter * 4.0,
            tx_bias_ma: 5.5 + jitter * 5.0,
        }
    }
}

// ─── Alarm / warning ────────────────────────────────────────────────────────

/// Which diagnostic field triggered an alarm or warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfpField {
    /// Module temperature.
    Temperature,
    /// Supply voltage.
    Voltage,
    /// Transmit optical power.
    TxPower,
    /// Receive optical power.
    RxPower,
    /// Transmit laser bias current.
    TxBias,
}

impl SfpField {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Temperature => "Temperature",
            Self::Voltage => "Voltage",
            Self::TxPower => "TX Power",
            Self::RxPower => "RX Power",
            Self::TxBias => "TX Bias",
        }
    }
}

/// Severity level of an SFP alarm or warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlarmLevel {
    /// Value is between the low-warning and low-alarm thresholds.
    LowWarning,
    /// Value has crossed the low-alarm threshold.
    LowAlarm,
    /// Value is between the high-warning and high-alarm thresholds.
    HighWarning,
    /// Value has crossed the high-alarm threshold.
    HighAlarm,
}

impl AlarmLevel {
    /// Returns `true` if this is an alarm (not just a warning).
    #[must_use]
    pub fn is_alarm(self) -> bool {
        matches!(self, Self::LowAlarm | Self::HighAlarm)
    }
}

/// A single SFP alarm or warning event.
#[derive(Debug, Clone)]
pub struct SfpAlarm {
    /// Which diagnostic field triggered the event.
    pub field: SfpField,
    /// Severity level.
    pub level: AlarmLevel,
    /// Measured value at the time of the alarm.
    pub value: f32,
    /// The threshold that was crossed.
    pub threshold: f32,
}

// ─── Thresholds ──────────────────────────────────────────────────────────────

/// Alarm and warning thresholds for all SFP+ DOM fields.
///
/// Typical values follow SFF-8472 / vendor datasheets for a 10GBase-SR module.
#[derive(Debug, Clone)]
pub struct SfpThresholds {
    // Temperature °C
    /// High-alarm threshold for temperature.
    pub temp_high_alarm: f32,
    /// High-warning threshold for temperature.
    pub temp_high_warn: f32,
    /// Low-warning threshold for temperature.
    pub temp_low_warn: f32,
    /// Low-alarm threshold for temperature.
    pub temp_low_alarm: f32,

    // Voltage V
    /// High-alarm threshold for voltage.
    pub voltage_high_alarm: f32,
    /// High-warning threshold for voltage.
    pub voltage_high_warn: f32,
    /// Low-warning threshold for voltage.
    pub voltage_low_warn: f32,
    /// Low-alarm threshold for voltage.
    pub voltage_low_alarm: f32,

    // TX Power dBm
    /// High-alarm threshold for TX optical power.
    pub tx_power_high_alarm: f32,
    /// High-warning threshold for TX optical power.
    pub tx_power_high_warn: f32,
    /// Low-warning threshold for TX optical power.
    pub tx_power_low_warn: f32,
    /// Low-alarm threshold for TX optical power.
    pub tx_power_low_alarm: f32,

    // RX Power dBm
    /// High-alarm threshold for RX optical power.
    pub rx_power_high_alarm: f32,
    /// High-warning threshold for RX optical power.
    pub rx_power_high_warn: f32,
    /// Low-warning threshold for RX optical power.
    pub rx_power_low_warn: f32,
    /// Low-alarm threshold for RX optical power.
    pub rx_power_low_alarm: f32,

    // TX Bias mA
    /// High-alarm threshold for TX bias current.
    pub tx_bias_high_alarm: f32,
    /// High-warning threshold for TX bias current.
    pub tx_bias_high_warn: f32,
    /// Low-warning threshold for TX bias current.
    pub tx_bias_low_warn: f32,
    /// Low-alarm threshold for TX bias current.
    pub tx_bias_low_alarm: f32,
}

impl SfpThresholds {
    /// Typical broadcast-grade 10GBase-SR SFP+ thresholds.
    #[must_use]
    pub fn typical() -> Self {
        Self {
            // Temperature
            temp_high_alarm: 85.0,
            temp_high_warn: 75.0,
            temp_low_warn: -5.0,
            temp_low_alarm: -15.0,

            // Voltage (3.3 V nominal)
            voltage_high_alarm: 3.63,
            voltage_high_warn: 3.47,
            voltage_low_warn: 3.13,
            voltage_low_alarm: 2.97,

            // TX Power (dBm)
            tx_power_high_alarm: 0.0,
            tx_power_high_warn: -1.0,
            tx_power_low_warn: -7.0,
            tx_power_low_alarm: -10.0,

            // RX Power (dBm)
            rx_power_high_alarm: 0.5,
            rx_power_high_warn: -1.0,
            rx_power_low_warn: -14.0,
            rx_power_low_alarm: -17.0,

            // TX Bias (mA)
            tx_bias_high_alarm: 17.0,
            tx_bias_high_warn: 14.0,
            tx_bias_low_warn: 2.0,
            tx_bias_low_alarm: 0.5,
        }
    }
}

/// Helper: check a single scalar against high/low thresholds.
fn check_thresholds(
    field: SfpField,
    value: f32,
    high_alarm: f32,
    high_warn: f32,
    low_warn: f32,
    low_alarm: f32,
    alarms: &mut Vec<SfpAlarm>,
) {
    if value >= high_alarm {
        alarms.push(SfpAlarm {
            field,
            level: AlarmLevel::HighAlarm,
            value,
            threshold: high_alarm,
        });
    } else if value >= high_warn {
        alarms.push(SfpAlarm {
            field,
            level: AlarmLevel::HighWarning,
            value,
            threshold: high_warn,
        });
    } else if value <= low_alarm {
        alarms.push(SfpAlarm {
            field,
            level: AlarmLevel::LowAlarm,
            value,
            threshold: low_alarm,
        });
    } else if value <= low_warn {
        alarms.push(SfpAlarm {
            field,
            level: AlarmLevel::LowWarning,
            value,
            threshold: low_warn,
        });
    }
}

// ─── Monitor ─────────────────────────────────────────────────────────────────

/// Monitor for one or more SFP+ transceiver modules.
///
/// In a production system the `read_diagnostics` method would perform actual
/// I²C / MDIO reads from the module's DOM registers (SFF-8472 bytes 96-105).
/// This implementation returns deterministic simulated values suitable for
/// testing and integration.
#[derive(Debug)]
pub struct SfpMonitor {
    /// Registered transceiver modules, keyed by `module_id`.
    pub transceivers: HashMap<u8, SfpTransceiver>,
    /// Cached diagnostics (updated by `read_diagnostics`).
    pub diagnostics_cache: HashMap<u8, SfpDiagnostics>,
    /// Alarm / warning thresholds applied to all modules.
    pub thresholds: SfpThresholds,
}

impl SfpMonitor {
    /// Creates a new monitor with typical SFP+ thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transceivers: HashMap::new(),
            diagnostics_cache: HashMap::new(),
            thresholds: SfpThresholds::typical(),
        }
    }

    /// Creates a monitor with custom thresholds.
    #[must_use]
    pub fn with_thresholds(thresholds: SfpThresholds) -> Self {
        Self {
            transceivers: HashMap::new(),
            diagnostics_cache: HashMap::new(),
            thresholds,
        }
    }

    /// Registers an SFP+ module with the monitor.
    pub fn register_module(&mut self, transceiver: SfpTransceiver) {
        self.transceivers.insert(transceiver.module_id, transceiver);
    }

    /// Reads (simulated) DOM diagnostics for a module.
    ///
    /// Returns an error if the `module_id` is not registered.
    pub fn read_diagnostics(&mut self, module_id: u8) -> VideoIpResult<SfpDiagnostics> {
        if !self.transceivers.contains_key(&module_id) {
            return Err(VideoIpError::ServiceNotFound(format!(
                "SFP module {module_id} not registered"
            )));
        }
        let diag = SfpDiagnostics::simulated(module_id);
        self.diagnostics_cache.insert(module_id, diag);
        Ok(diag)
    }

    /// Evaluates `diag` against the monitor's thresholds and returns any alarms.
    #[must_use]
    pub fn check_alarms(&self, _module_id: u8, diag: &SfpDiagnostics) -> Vec<SfpAlarm> {
        let mut alarms: Vec<SfpAlarm> = Vec::new();
        let t = &self.thresholds;

        check_thresholds(
            SfpField::Temperature,
            diag.temperature_c,
            t.temp_high_alarm,
            t.temp_high_warn,
            t.temp_low_warn,
            t.temp_low_alarm,
            &mut alarms,
        );
        check_thresholds(
            SfpField::Voltage,
            diag.voltage_v,
            t.voltage_high_alarm,
            t.voltage_high_warn,
            t.voltage_low_warn,
            t.voltage_low_alarm,
            &mut alarms,
        );
        check_thresholds(
            SfpField::TxPower,
            diag.tx_power_dbm,
            t.tx_power_high_alarm,
            t.tx_power_high_warn,
            t.tx_power_low_warn,
            t.tx_power_low_alarm,
            &mut alarms,
        );
        check_thresholds(
            SfpField::RxPower,
            diag.rx_power_dbm,
            t.rx_power_high_alarm,
            t.rx_power_high_warn,
            t.rx_power_low_warn,
            t.rx_power_low_alarm,
            &mut alarms,
        );
        check_thresholds(
            SfpField::TxBias,
            diag.tx_bias_ma,
            t.tx_bias_high_alarm,
            t.tx_bias_high_warn,
            t.tx_bias_low_warn,
            t.tx_bias_low_alarm,
            &mut alarms,
        );

        alarms
    }

    /// Returns `true` if the link is considered up based on received optical power.
    ///
    /// A received power above −30 dBm indicates a connected fibre carrying signal.
    #[must_use]
    pub fn is_link_up(&self, _module_id: u8, diag: &SfpDiagnostics) -> bool {
        diag.rx_power_dbm > -30.0
    }

    /// Returns all registered module IDs, sorted.
    #[must_use]
    pub fn module_ids(&self) -> Vec<u8> {
        let mut ids: Vec<u8> = self.transceivers.keys().copied().collect();
        ids.sort();
        ids
    }
}

impl Default for SfpMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor_with_module(id: u8) -> SfpMonitor {
        let mut m = SfpMonitor::new();
        m.register_module(SfpTransceiver::typical_10g_sr(id));
        m
    }

    // ── Registration ─────────────────────────────────────────────────────

    #[test]
    fn test_register_module() {
        let mut monitor = SfpMonitor::new();
        monitor.register_module(SfpTransceiver::typical_10g_sr(0));
        assert!(monitor.transceivers.contains_key(&0));
    }

    #[test]
    fn test_register_multiple_modules() {
        let mut monitor = SfpMonitor::new();
        for id in 0..4 {
            monitor.register_module(SfpTransceiver::typical_10g_sr(id));
        }
        assert_eq!(monitor.module_ids(), vec![0, 1, 2, 3]);
    }

    // ── Diagnostics reading ───────────────────────────────────────────────

    #[test]
    fn test_read_diagnostics_known_module() {
        let mut monitor = make_monitor_with_module(0);
        let diag = monitor.read_diagnostics(0).expect("should succeed");
        // Simulated values should be in plausible ranges
        assert!(diag.temperature_c >= 0.0 && diag.temperature_c < 200.0);
        assert!(diag.voltage_v > 0.0 && diag.voltage_v < 5.0);
        assert!(diag.tx_bias_ma >= 0.0);
    }

    #[test]
    fn test_read_diagnostics_unknown_module_error() {
        let mut monitor = SfpMonitor::new();
        assert!(monitor.read_diagnostics(99).is_err());
    }

    #[test]
    fn test_diagnostics_cached_after_read() {
        let mut monitor = make_monitor_with_module(1);
        monitor.read_diagnostics(1).expect("read");
        assert!(monitor.diagnostics_cache.contains_key(&1));
    }

    #[test]
    fn test_diagnostics_simulated_varies_by_seed() {
        let d0 = SfpDiagnostics::simulated(0);
        let d1 = SfpDiagnostics::simulated(1);
        // Different seeds → different values
        assert!((d0.temperature_c - d1.temperature_c).abs() > f32::EPSILON);
    }

    // ── Alarm detection ────────────────────────────────────────────────────

    #[test]
    fn test_no_alarms_nominal() {
        let monitor = SfpMonitor::new();
        let diag = SfpDiagnostics::nominal();
        let alarms = monitor.check_alarms(0, &diag);
        assert!(
            alarms.is_empty(),
            "nominal values should not trigger alarms"
        );
    }

    #[test]
    fn test_high_temperature_alarm() {
        let monitor = SfpMonitor::new();
        let diag = SfpDiagnostics {
            temperature_c: 90.0, // above 85°C alarm threshold
            ..SfpDiagnostics::nominal()
        };
        let alarms = monitor.check_alarms(0, &diag);
        let temp_alarm = alarms.iter().find(|a| a.field == SfpField::Temperature);
        assert!(temp_alarm.is_some());
        assert_eq!(temp_alarm.expect("alarm").level, AlarmLevel::HighAlarm);
    }

    #[test]
    fn test_high_temperature_warning() {
        let monitor = SfpMonitor::new();
        let diag = SfpDiagnostics {
            temperature_c: 78.0, // above 75°C warn, below 85°C alarm
            ..SfpDiagnostics::nominal()
        };
        let alarms = monitor.check_alarms(0, &diag);
        let temp_alarm = alarms.iter().find(|a| a.field == SfpField::Temperature);
        assert!(temp_alarm.is_some());
        assert_eq!(temp_alarm.expect("warn").level, AlarmLevel::HighWarning);
    }

    #[test]
    fn test_low_rx_power_alarm() {
        let monitor = SfpMonitor::new();
        let diag = SfpDiagnostics {
            rx_power_dbm: -20.0, // below −17 dBm low-alarm threshold
            ..SfpDiagnostics::nominal()
        };
        let alarms = monitor.check_alarms(0, &diag);
        let rx_alarm = alarms.iter().find(|a| a.field == SfpField::RxPower);
        assert!(rx_alarm.is_some());
        assert_eq!(rx_alarm.expect("alarm").level, AlarmLevel::LowAlarm);
    }

    #[test]
    fn test_alarm_is_alarm_vs_warning() {
        assert!(AlarmLevel::HighAlarm.is_alarm());
        assert!(AlarmLevel::LowAlarm.is_alarm());
        assert!(!AlarmLevel::HighWarning.is_alarm());
        assert!(!AlarmLevel::LowWarning.is_alarm());
    }

    // ── Link status ────────────────────────────────────────────────────────

    #[test]
    fn test_link_up_nominal() {
        let monitor = SfpMonitor::new();
        let diag = SfpDiagnostics::nominal();
        assert!(monitor.is_link_up(0, &diag));
    }

    #[test]
    fn test_link_down_low_rx_power() {
        let monitor = SfpMonitor::new();
        let diag = SfpDiagnostics {
            rx_power_dbm: -35.0, // below −30 dBm → no signal
            ..SfpDiagnostics::nominal()
        };
        assert!(!monitor.is_link_up(0, &diag));
    }

    // ── Connector type / field label ───────────────────────────────────────

    #[test]
    fn test_connector_type_labels() {
        assert_eq!(ConnectorType::Lc.label(), "LC");
        assert_eq!(ConnectorType::Rj45.label(), "RJ45");
        assert_eq!(ConnectorType::Unknown.label(), "Unknown");
    }

    #[test]
    fn test_sfp_field_labels() {
        assert_eq!(SfpField::Temperature.label(), "Temperature");
        assert_eq!(SfpField::RxPower.label(), "RX Power");
    }
}
