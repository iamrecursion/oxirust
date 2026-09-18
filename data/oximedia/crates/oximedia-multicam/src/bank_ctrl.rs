//! Camera bank control system for multicam production.
//!
//! Provides [`CameraBankCtrl`], [`PreviewMode`], [`BankSwitchEvent`], and
//! [`BankControlSystem`] for managing camera bank assignments and live switching.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── PreviewMode ───────────────────────────────────────────────────────────────

/// How cameras in a bank are displayed in the preview monitor wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    /// All cameras shown in a grid.
    Grid,
    /// Main camera with a smaller inset picture.
    PIP,
    /// Single full-screen camera.
    Single,
    /// Four cameras in a 2×2 layout.
    Quad,
}

impl PreviewMode {
    /// Maximum number of simultaneously visible cameras for this mode.
    #[must_use]
    pub fn max_cameras(&self) -> usize {
        match self {
            Self::Grid => 16,
            Self::PIP => 2,
            Self::Single => 1,
            Self::Quad => 4,
        }
    }
}

// ── CameraBankCtrl ────────────────────────────────────────────────────────────

/// A logical grouping of camera sources with a shared preview mode.
#[derive(Debug, Clone)]
pub struct CameraBankCtrl {
    /// Bank identifier.
    pub bank_id: u32,
    /// Human-readable label.
    pub name: String,
    /// Camera source IDs assigned to this bank.
    pub camera_ids: Vec<u32>,
    /// Current preview layout.
    pub preview_mode: PreviewMode,
}

impl CameraBankCtrl {
    /// Number of cameras in this bank.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.camera_ids.len()
    }

    /// Returns `true` if `camera_id` is assigned to this bank.
    #[must_use]
    pub fn contains(&self, camera_id: u32) -> bool {
        self.camera_ids.contains(&camera_id)
    }
}

// ── BankSwitchEvent ───────────────────────────────────────────────────────────

/// Records a single bank-switch action.
#[derive(Debug, Clone)]
pub struct BankSwitchEvent {
    /// Bank that was active before the switch.
    pub from_bank: u32,
    /// Bank that became active after the switch.
    pub to_bank: u32,
    /// Wall-clock timestamp of the switch in milliseconds.
    pub timestamp_ms: u64,
    /// Operator who initiated the switch.
    pub operator: String,
}

impl BankSwitchEvent {
    /// Returns `true` when the switch stays within the same bank (no-op switch).
    #[must_use]
    pub fn is_self_switch(&self) -> bool {
        self.from_bank == self.to_bank
    }
}

// ── BankControlSystem ─────────────────────────────────────────────────────────

/// Manages multiple camera banks, tracks the active bank, and logs switches.
#[derive(Debug, Default)]
pub struct BankControlSystem {
    /// Registered banks.
    pub banks: Vec<CameraBankCtrl>,
    /// ID of the currently active bank.
    pub active_bank_id: u32,
    /// Chronological log of bank-switch events.
    pub switch_log: Vec<BankSwitchEvent>,
}

impl BankControlSystem {
    /// Create an empty bank control system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a bank.
    pub fn add_bank(&mut self, bank: CameraBankCtrl) {
        self.banks.push(bank);
    }

    /// Switch to `bank_id`.
    ///
    /// Returns `true` on success, `false` if the bank does not exist.
    pub fn switch_to(&mut self, bank_id: u32, operator: &str, now_ms: u64) -> bool {
        if !self.banks.iter().any(|b| b.bank_id == bank_id) {
            return false;
        }
        let evt = BankSwitchEvent {
            from_bank: self.active_bank_id,
            to_bank: bank_id,
            timestamp_ms: now_ms,
            operator: operator.to_owned(),
        };
        self.switch_log.push(evt);
        self.active_bank_id = bank_id;
        true
    }

    /// Return a reference to the currently active bank, if it exists.
    #[must_use]
    pub fn active_bank(&self) -> Option<&CameraBankCtrl> {
        self.banks.iter().find(|b| b.bank_id == self.active_bank_id)
    }

    /// Estimate the bank-switch rate (switches per hour) within the most recent
    /// `window_ms` milliseconds.
    #[must_use]
    pub fn switch_rate_per_hour(&self, window_ms: u64) -> f64 {
        if window_ms == 0 || self.switch_log.is_empty() {
            return 0.0;
        }
        let last_ts = self.switch_log.last().map_or(0, |e| e.timestamp_ms);
        let cutoff = last_ts.saturating_sub(window_ms);
        let count = self
            .switch_log
            .iter()
            .filter(|e| e.timestamp_ms >= cutoff && !e.is_self_switch())
            .count();
        let window_hours = window_ms as f64 / 3_600_000.0;
        count as f64 / window_hours
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bank(id: u32, cameras: &[u32]) -> CameraBankCtrl {
        CameraBankCtrl {
            bank_id: id,
            name: format!("Bank {id}"),
            camera_ids: cameras.to_vec(),
            preview_mode: PreviewMode::Grid,
        }
    }

    // PreviewMode ──────────────────────────────────────────────────────────────

    #[test]
    fn test_grid_max_cameras() {
        assert_eq!(PreviewMode::Grid.max_cameras(), 16);
    }

    #[test]
    fn test_pip_max_cameras() {
        assert_eq!(PreviewMode::PIP.max_cameras(), 2);
    }

    #[test]
    fn test_single_max_cameras() {
        assert_eq!(PreviewMode::Single.max_cameras(), 1);
    }

    #[test]
    fn test_quad_max_cameras() {
        assert_eq!(PreviewMode::Quad.max_cameras(), 4);
    }

    // CameraBankCtrl ───────────────────────────────────────────────────────────

    #[test]
    fn test_camera_count() {
        let b = make_bank(1, &[10, 20, 30]);
        assert_eq!(b.camera_count(), 3);
    }

    #[test]
    fn test_contains_true() {
        let b = make_bank(1, &[5, 10, 15]);
        assert!(b.contains(10));
    }

    #[test]
    fn test_contains_false() {
        let b = make_bank(1, &[5, 10, 15]);
        assert!(!b.contains(99));
    }

    // BankSwitchEvent ──────────────────────────────────────────────────────────

    #[test]
    fn test_is_self_switch_true() {
        let evt = BankSwitchEvent {
            from_bank: 3,
            to_bank: 3,
            timestamp_ms: 1000,
            operator: "op".into(),
        };
        assert!(evt.is_self_switch());
    }

    #[test]
    fn test_is_self_switch_false() {
        let evt = BankSwitchEvent {
            from_bank: 1,
            to_bank: 2,
            timestamp_ms: 2000,
            operator: "op".into(),
        };
        assert!(!evt.is_self_switch());
    }

    // BankControlSystem ────────────────────────────────────────────────────────

    #[test]
    fn test_switch_to_valid_bank() {
        let mut sys = BankControlSystem::new();
        sys.add_bank(make_bank(1, &[1]));
        sys.add_bank(make_bank(2, &[2]));
        assert!(sys.switch_to(2, "alice", 5000));
        assert_eq!(sys.active_bank_id, 2);
    }

    #[test]
    fn test_switch_to_invalid_bank_returns_false() {
        let mut sys = BankControlSystem::new();
        sys.add_bank(make_bank(1, &[1]));
        assert!(!sys.switch_to(99, "alice", 0));
    }

    #[test]
    fn test_active_bank_found() {
        let mut sys = BankControlSystem::new();
        sys.add_bank(make_bank(7, &[1, 2]));
        sys.active_bank_id = 7;
        assert!(sys.active_bank().is_some());
    }

    #[test]
    fn test_active_bank_not_found() {
        let sys = BankControlSystem::new();
        assert!(sys.active_bank().is_none());
    }

    #[test]
    fn test_switch_log_grows() {
        let mut sys = BankControlSystem::new();
        sys.add_bank(make_bank(0, &[]));
        sys.add_bank(make_bank(1, &[]));
        sys.switch_to(1, "op", 1000);
        sys.switch_to(0, "op", 2000);
        assert_eq!(sys.switch_log.len(), 2);
    }

    #[test]
    fn test_switch_rate_per_hour_simple() {
        let mut sys = BankControlSystem::new();
        sys.add_bank(make_bank(0, &[]));
        sys.add_bank(make_bank(1, &[]));
        // Perform 2 real switches within a 1-hour window
        sys.switch_to(1, "op", 0);
        sys.switch_to(0, "op", 1_800_000); // 30 min later
        let rate = sys.switch_rate_per_hour(3_600_000); // 1-hour window
                                                        // 2 switches in 1 hour = rate 2.0
        assert!(rate > 0.0);
    }

    #[test]
    fn test_switch_rate_zero_window() {
        let sys = BankControlSystem::new();
        assert_eq!(sys.switch_rate_per_hour(0), 0.0);
    }
}
