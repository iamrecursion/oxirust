//! BBR (Bottleneck Bandwidth and Round-trip propagation time) congestion control.
//!
//! BBR is a model-based congestion controller originally developed at Google
//! (Cardwell et al., 2016).  Rather than reacting to packet loss (like CUBIC
//! or Reno), BBR maintains explicit estimates of two path parameters:
//!
//! - **BtlBw** (bottleneck bandwidth): the maximum delivery rate observed
//!   over a sliding window of recent round trips.
//! - **RTprop** (round-trip propagation time): the minimum RTT observed over
//!   a longer measurement interval (default 10 s).
//!
//! These two estimates define the **Bandwidth-Delay Product (BDP)**, which
//! BBR uses to set both pacing rate and congestion window.
//!
//! # State Machine
//!
//! ```text
//!                  ┌─────────┐
//!            ┌────>│ Startup │─── BtlBw not growing ──>┐
//!            │     └─────────┘                          │
//!            │                                          v
//!            │                                     ┌───────┐
//!            │                                     │ Drain  │
//!            │                                     └───────┘
//!            │                                          │
//!            │                                    queue drained
//!            │                                          v
//!         ┌──┴──────────────────────────────────────────────┐
//!         │                   ProbeBw                        │
//!         │  (cyclic gain: [5/4, 3/4, 1, 1, 1, 1, 1, 1])   │
//!         └─────────────────────────────────────────────────┘
//!                          │  RTprop expired
//!                          v
//!                     ┌──────────┐
//!                     │ ProbeRtt │──── 200 ms ────> ProbeBw
//!                     └──────────┘
//! ```
//!
//! # Reference
//!
//! - [BBR: Congestion-Based Congestion Control](https://research.google/pubs/pub45646/)
//! - [RFC 9438](https://datatracker.ietf.org/doc/html/rfc9438) (BBRv2 informational)

use std::collections::VecDeque;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Startup pacing gain: 2/ln(2) ≈ 2.885.
///
/// This is the standard BBR startup gain that allows the pipe to fill
/// roughly one RTT per doubling.
const DEFAULT_STARTUP_GAIN: f64 = 2.885;

/// Minimum pacing rate as a fraction of the estimated bottleneck bandwidth
/// during ProbeRtt.  We pace at this rate to drain the queue without
/// completely stalling delivery.
const PROBE_RTT_PACING_GAIN: f64 = 1.0;

/// Minimum cwnd floor in bytes (4 × MSS, MSS = 1460 bytes).
const MIN_CWND_BYTES: u64 = 4 * 1460;

/// ProbeBw cycle of gains (8 phases).  The first phase probes up (+25 %),
/// the second drains (-25 %), then six steady phases at 1.0.
const PROBE_BW_GAINS: [f64; 8] = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Duration of ProbeRtt phase in milliseconds.
const PROBE_RTT_DURATION_MS: u64 = 200;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Configuration knobs for [`BbrController`].
#[derive(Debug, Clone)]
pub struct BbrConfig {
    /// Startup pacing gain (default ≈ 2.885).
    pub startup_gain: f64,
    /// Drain pacing gain = 1/startup_gain so the queue drains in one RTT.
    pub drain_gain: f64,
    /// ProbeBw probe-up gain (default 1.25 = +25 %).
    pub probe_bw_gain: f64,
    /// RTprop measurement window in milliseconds (default 10 000 ms = 10 s).
    pub rtprop_filter_len_ms: u64,
    /// BtlBw measurement window in round trips (default 10).
    pub btlbw_filter_len: usize,
}

impl Default for BbrConfig {
    fn default() -> Self {
        Self {
            startup_gain: DEFAULT_STARTUP_GAIN,
            drain_gain: 1.0 / DEFAULT_STARTUP_GAIN,
            probe_bw_gain: 1.25,
            rtprop_filter_len_ms: 10_000,
            btlbw_filter_len: 10,
        }
    }
}

/// BBR state machine phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BbrState {
    /// Exponential bandwidth probing at startup.
    Startup,
    /// Drain the queue built during startup.
    Drain,
    /// Steady-state operation with periodic bandwidth probing.
    ProbeBw,
    /// Temporarily reduce cwnd to measure the true propagation RTT.
    ProbeRtt,
}

impl std::fmt::Display for BbrState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Startup => "startup",
            Self::Drain => "drain",
            Self::ProbeBw => "probe_bw",
            Self::ProbeRtt => "probe_rtt",
        };
        write!(f, "{s}")
    }
}

/// A single delivery-rate / RTT measurement from an ACK event.
#[derive(Debug, Clone, Copy)]
pub struct AckSample {
    /// Bytes delivered since the last ACK sample.
    pub delivered: u64,
    /// Wall-clock time elapsed since the last ACK sample (seconds).
    pub elapsed_secs: f64,
    /// Round-trip time measured for this ACK (seconds).
    pub rtt_secs: f64,
    /// Whether the sender was application-limited when the packet was sent.
    pub is_app_limited: bool,
}

/// Bandwidth sample stored in the BtlBw max-filter.
#[derive(Debug, Clone, Copy)]
struct BwSample {
    /// Delivery rate in bytes/second.
    rate_bps: f64,
    /// Round count when this sample was taken.
    #[allow(dead_code)]
    round: u64,
}

/// RTT sample stored in the RTprop min-filter.
#[derive(Debug, Clone, Copy)]
struct RttSample {
    /// Measured RTT in seconds.
    rtt_secs: f64,
    /// Monotonic timestamp in milliseconds when the sample was taken.
    timestamp_ms: u64,
}

/// BBR congestion controller.
///
/// Call [`BbrController::on_ack`] for every ACK event to update internal
/// state.  Read [`BbrController::pacing_rate`] and [`BbrController::cwnd`]
/// to determine how fast to send and how much data may be in-flight.
pub struct BbrController {
    config: BbrConfig,
    state: BbrState,

    /// Bottleneck bandwidth estimate (bytes/second).
    btlbw: f64,
    /// Round-trip propagation time estimate (seconds).
    rtprop: f64,
    /// Monotonic clock offset (ms) when we last measured a new RTprop minimum.
    rtprop_stamp_ms: u64,
    /// True when RTprop measurement window has expired.
    rtprop_expired: bool,

    /// Pacing rate in bytes/second.
    pacing_rate: f64,
    /// Congestion window in bytes.
    cwnd: u64,

    /// Monotonic "clock" maintained by summing elapsed_secs × 1000 on each ACK.
    now_ms: u64,

    /// Round trip counter.  Incremented each time we see delivery progress.
    round_count: u64,
    /// Number of consecutive rounds where BtlBw did not increase (startup exit criterion).
    full_bw_count: u32,
    /// BtlBw value at the last full-bw check.
    full_bw: f64,

    /// Sliding window of bandwidth samples (length = config.btlbw_filter_len).
    bw_samples: VecDeque<BwSample>,

    /// Ring buffer of RTT samples for RTprop min-filter.
    rtt_samples: VecDeque<RttSample>,

    /// ProbeBw gain cycle index (0–7).
    probe_bw_cycle_idx: usize,
    /// Round count when the current ProbeBw cycle phase started.
    probe_bw_cycle_start_round: u64,

    /// Monotonic timestamp (ms) when ProbeRtt entered the rtt-measurement phase.
    probe_rtt_start_ms: Option<u64>,

    /// Total ACK count (for diagnostics).
    ack_count: u64,
}

impl BbrController {
    /// Create a new [`BbrController`] with the given configuration.
    #[must_use]
    pub fn new(config: BbrConfig) -> Self {
        let startup_gain = config.startup_gain;
        // Initial estimates: 1 Mbps bandwidth, 100 ms RTprop
        let initial_bw: f64 = 125_000.0; // 1 Mbps in bytes/sec
        let initial_rtt: f64 = 0.1; // 100 ms

        let initial_cwnd = ((initial_bw * initial_rtt * startup_gain) as u64).max(MIN_CWND_BYTES);

        Self {
            config,
            state: BbrState::Startup,
            btlbw: initial_bw,
            rtprop: initial_rtt,
            rtprop_stamp_ms: 0,
            rtprop_expired: false,
            pacing_rate: initial_bw * startup_gain,
            cwnd: initial_cwnd,
            now_ms: 0,
            round_count: 0,
            full_bw_count: 0,
            full_bw: 0.0,
            bw_samples: VecDeque::new(),
            rtt_samples: VecDeque::new(),
            probe_bw_cycle_idx: 0,
            probe_bw_cycle_start_round: 0,
            probe_rtt_start_ms: None,
            ack_count: 0,
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Process an ACK event and update all BBR internal state.
    ///
    /// This is the primary entry point.  Call once per ACK with a filled-in
    /// [`AckSample`].
    pub fn on_ack(&mut self, sample: AckSample) {
        if sample.elapsed_secs <= 0.0 || sample.rtt_secs <= 0.0 {
            return;
        }

        self.ack_count += 1;

        // Advance our internal clock.
        let elapsed_ms = (sample.elapsed_secs * 1_000.0) as u64;
        self.now_ms = self.now_ms.saturating_add(elapsed_ms);

        // Update RTprop (minimum RTT over the filter window).
        self.update_rtprop(sample.rtt_secs);

        // Compute delivery rate for this sample.
        let delivery_rate = if sample.elapsed_secs > 0.0 {
            sample.delivered as f64 / sample.elapsed_secs
        } else {
            0.0
        };

        // Update BtlBw (maximum delivery rate) only for non-app-limited samples.
        if !sample.is_app_limited || delivery_rate > self.btlbw {
            self.update_btlbw(delivery_rate);
        }

        // Advance round counter.
        self.round_count = self.round_count.saturating_add(1);

        // Run state machine.
        match self.state {
            BbrState::Startup => self.handle_startup(),
            BbrState::Drain => self.handle_drain(),
            BbrState::ProbeBw => self.handle_probe_bw(),
            BbrState::ProbeRtt => self.handle_probe_rtt(),
        }

        // Recompute cwnd and pacing_rate from current estimates.
        self.update_pacing_and_cwnd();
    }

    /// Current pacing rate in bytes per second.
    #[must_use]
    pub fn pacing_rate(&self) -> f64 {
        self.pacing_rate
    }

    /// Current congestion window in bytes.
    #[must_use]
    pub fn cwnd(&self) -> u64 {
        self.cwnd
    }

    /// Current BBR state machine phase.
    #[must_use]
    pub fn state(&self) -> &BbrState {
        &self.state
    }

    /// Target in-flight data volume in bytes (BDP + headroom).
    ///
    /// This is `BtlBw × RTprop × gain + MIN_CWND_BYTES` to ensure the
    /// pipe stays full under high throughput.
    #[must_use]
    pub fn inflight_target(&self) -> u64 {
        let bdp = self.btlbw * self.rtprop;
        let gain = self.current_cwnd_gain();
        ((bdp * gain) as u64).max(MIN_CWND_BYTES)
    }

    /// Bottleneck bandwidth estimate (bytes/second).
    #[must_use]
    pub fn btlbw(&self) -> f64 {
        self.btlbw
    }

    /// RTprop estimate (seconds).
    #[must_use]
    pub fn rtprop(&self) -> f64 {
        self.rtprop
    }

    // ── Internal: RTprop filter ─────────────────────────────────────────────

    fn update_rtprop(&mut self, rtt_secs: f64) {
        // Expire old RTprop measurement if the window has lapsed.
        let window_ms = self.config.rtprop_filter_len_ms;
        let age_ms = self.now_ms.saturating_sub(self.rtprop_stamp_ms);
        self.rtprop_expired = age_ms > window_ms;

        // Store sample.
        self.rtt_samples.push_back(RttSample {
            rtt_secs,
            timestamp_ms: self.now_ms,
        });

        // Evict samples older than the filter window.
        while let Some(front) = self.rtt_samples.front() {
            if self.now_ms.saturating_sub(front.timestamp_ms) > window_ms {
                self.rtt_samples.pop_front();
            } else {
                break;
            }
        }

        // Recompute RTprop as the minimum in the window.
        if let Some(min_sample) = self.rtt_samples.iter().min_by(|a, b| {
            a.rtt_secs
                .partial_cmp(&b.rtt_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if min_sample.rtt_secs < self.rtprop || self.rtprop_expired {
                self.rtprop = min_sample.rtt_secs;
                self.rtprop_stamp_ms = self.now_ms;
                self.rtprop_expired = false;
            }
        }
    }

    // ── Internal: BtlBw filter ──────────────────────────────────────────────

    fn update_btlbw(&mut self, delivery_rate: f64) {
        self.bw_samples.push_back(BwSample {
            rate_bps: delivery_rate,
            round: self.round_count,
        });

        // Keep only the last `btlbw_filter_len` round trips.
        while self.bw_samples.len() > self.config.btlbw_filter_len {
            self.bw_samples.pop_front();
        }

        // BtlBw = max delivery rate in the window.
        if let Some(max_sample) = self.bw_samples.iter().max_by(|a, b| {
            a.rate_bps
                .partial_cmp(&b.rate_bps)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.btlbw = max_sample.rate_bps;
        }
    }

    // ── Internal: state handlers ────────────────────────────────────────────

    fn handle_startup(&mut self) {
        // Check if we have reached full pipe: BtlBw did not grow by ≥ 25 %
        // for `full_bw_count_threshold` consecutive rounds.
        const FULL_BW_GROWTH_THRESHOLD: f64 = 1.25;
        const FULL_BW_COUNT_THRESHOLD: u32 = 3;

        if self.btlbw >= self.full_bw * FULL_BW_GROWTH_THRESHOLD {
            self.full_bw = self.btlbw;
            self.full_bw_count = 0;
        } else {
            self.full_bw_count += 1;
        }

        if self.full_bw_count >= FULL_BW_COUNT_THRESHOLD {
            self.state = BbrState::Drain;
        }
    }

    fn handle_drain(&mut self) {
        // Transition to ProbeBw when in-flight data falls to/below BDP.
        let bdp = self.btlbw * self.rtprop;
        let inflight = self.cwnd; // cwnd approximates in-flight here
        if (inflight as f64) <= bdp {
            self.state = BbrState::ProbeBw;
            self.probe_bw_cycle_idx = 0;
            self.probe_bw_cycle_start_round = self.round_count;
        }
    }

    fn handle_probe_bw(&mut self) {
        // Advance ProbeBw gain cycle every round.
        let rounds_in_phase = self
            .round_count
            .saturating_sub(self.probe_bw_cycle_start_round);
        if rounds_in_phase >= 1 {
            self.probe_bw_cycle_idx = (self.probe_bw_cycle_idx + 1) % PROBE_BW_GAINS.len();
            self.probe_bw_cycle_start_round = self.round_count;
        }

        // Trigger ProbeRtt if RTprop measurement has expired.
        if self.rtprop_expired {
            self.state = BbrState::ProbeRtt;
            self.probe_rtt_start_ms = Some(self.now_ms);
        }
    }

    fn handle_probe_rtt(&mut self) {
        // Stay in ProbeRtt for at least PROBE_RTT_DURATION_MS.
        let start_ms = self.probe_rtt_start_ms.unwrap_or(self.now_ms);
        let elapsed_ms = self.now_ms.saturating_sub(start_ms);

        if elapsed_ms >= PROBE_RTT_DURATION_MS {
            // Re-enter ProbeBw after a ProbeRtt.
            self.probe_rtt_start_ms = None;
            self.state = BbrState::ProbeBw;
            self.probe_bw_cycle_idx = 0;
            self.probe_bw_cycle_start_round = self.round_count;
            // Fresh RTprop stamp.
            self.rtprop_stamp_ms = self.now_ms;
            self.rtprop_expired = false;
        }
    }

    // ── Internal: pacing & cwnd ─────────────────────────────────────────────

    /// Returns the current pacing gain factor based on the state machine.
    fn current_pacing_gain(&self) -> f64 {
        match self.state {
            BbrState::Startup => self.config.startup_gain,
            BbrState::Drain => self.config.drain_gain,
            BbrState::ProbeBw => PROBE_BW_GAINS[self.probe_bw_cycle_idx],
            BbrState::ProbeRtt => PROBE_RTT_PACING_GAIN,
        }
    }

    /// Returns the current cwnd gain factor.
    fn current_cwnd_gain(&self) -> f64 {
        match self.state {
            BbrState::Startup => self.config.startup_gain,
            BbrState::Drain => self.config.drain_gain,
            // ProbeBw cwnd is always 2×BDP to allow the probe-up phase to work.
            BbrState::ProbeBw => 2.0,
            // ProbeRtt uses minimal cwnd to measure propagation RTT.
            BbrState::ProbeRtt => 1.0,
        }
    }

    fn update_pacing_and_cwnd(&mut self) {
        let pacing_gain = self.current_pacing_gain();
        self.pacing_rate = self.btlbw * pacing_gain;

        let bdp = self.btlbw * self.rtprop;
        let cwnd_gain = self.current_cwnd_gain();
        let target = ((bdp * cwnd_gain) as u64).max(MIN_CWND_BYTES);
        self.cwnd = target;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(delivered: u64, elapsed_secs: f64, rtt_secs: f64) -> AckSample {
        AckSample {
            delivered,
            elapsed_secs,
            rtt_secs,
            is_app_limited: false,
        }
    }

    // ── BbrConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = BbrConfig::default();
        assert!((cfg.startup_gain - DEFAULT_STARTUP_GAIN).abs() < 1e-6);
        assert!((cfg.drain_gain - 1.0 / DEFAULT_STARTUP_GAIN).abs() < 1e-6);
        assert!((cfg.probe_bw_gain - 1.25).abs() < 1e-6);
        assert_eq!(cfg.rtprop_filter_len_ms, 10_000);
        assert_eq!(cfg.btlbw_filter_len, 10);
    }

    // ── Initial state ───────────────────────────────────────────────────────

    #[test]
    fn test_initial_state_is_startup() {
        let ctrl = BbrController::new(BbrConfig::default());
        assert_eq!(*ctrl.state(), BbrState::Startup);
    }

    #[test]
    fn test_initial_pacing_rate_positive() {
        let ctrl = BbrController::new(BbrConfig::default());
        assert!(
            ctrl.pacing_rate() > 0.0,
            "initial pacing rate should be positive"
        );
    }

    #[test]
    fn test_initial_cwnd_at_least_min() {
        let ctrl = BbrController::new(BbrConfig::default());
        assert!(ctrl.cwnd() >= MIN_CWND_BYTES, "initial cwnd below floor");
    }

    // ── Startup phase ───────────────────────────────────────────────────────

    #[test]
    fn test_startup_pacing_gain_applied() {
        let cfg = BbrConfig::default();
        let mut ctrl = BbrController::new(cfg.clone());

        // Feed a single strong sample; check pacing rate reflects startup gain.
        let sample = make_sample(125_000, 0.001, 0.01); // 125 MB/s, 10 ms RTT
        ctrl.on_ack(sample);
        assert_eq!(*ctrl.state(), BbrState::Startup);
        // Pacing rate = BtlBw × startup_gain
        let expected = ctrl.btlbw() * cfg.startup_gain;
        assert!(
            (ctrl.pacing_rate() - expected).abs() < 1.0,
            "unexpected pacing rate in startup"
        );
    }

    #[test]
    fn test_startup_exits_after_full_bw_detected() {
        let mut ctrl = BbrController::new(BbrConfig::default());

        // Simulate bandwidth plateau: same delivery rate for many rounds.
        let flat_sample = make_sample(12_500, 0.001, 0.01); // 12.5 MB/s constant
        for _ in 0..20 {
            ctrl.on_ack(flat_sample);
        }

        // After enough flat rounds, BBR should exit Startup (into Drain or ProbeBw).
        assert_ne!(
            *ctrl.state(),
            BbrState::Startup,
            "should have exited Startup after flat bandwidth"
        );
    }

    #[test]
    fn test_startup_rate_increases_with_growing_bw() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        let r0 = ctrl.pacing_rate();

        // Feeding a high-bandwidth sample should push up the estimate.
        ctrl.on_ack(make_sample(1_000_000, 0.001, 0.005));
        assert!(
            ctrl.pacing_rate() > r0,
            "pacing rate should increase after high-bw sample"
        );
    }

    // ── Drain phase ─────────────────────────────────────────────────────────

    #[test]
    fn test_drain_uses_lower_pacing_gain() {
        let cfg = BbrConfig::default();
        // Force Drain state.
        let mut ctrl = BbrController::new(cfg.clone());
        ctrl.state = BbrState::Drain;
        ctrl.btlbw = 1_000_000.0;
        ctrl.rtprop = 0.01;
        ctrl.update_pacing_and_cwnd();

        let expected = ctrl.btlbw() * cfg.drain_gain;
        assert!(
            (ctrl.pacing_rate() - expected).abs() < 1.0,
            "drain pacing rate incorrect: {} vs {}",
            ctrl.pacing_rate(),
            expected
        );
    }

    // ── ProbeBw phase ───────────────────────────────────────────────────────

    #[test]
    fn test_probe_bw_gain_cycles() {
        // Verify we eventually step through gain > 1 and gain < 1 phases.
        let mut ctrl = BbrController::new(BbrConfig::default());
        ctrl.state = BbrState::ProbeBw;
        ctrl.btlbw = 1_000_000.0;
        ctrl.rtprop = 0.01;

        let mut saw_high = false;
        let mut saw_low = false;

        for _ in 0..32 {
            ctrl.on_ack(make_sample(1_000, 0.001, 0.01));
            let rate = ctrl.pacing_rate() / ctrl.btlbw();
            if rate > 1.1 {
                saw_high = true;
            }
            if rate < 0.9 {
                saw_low = true;
            }
        }

        assert!(saw_high, "expected a high-gain phase in ProbeBw");
        assert!(saw_low, "expected a drain-gain phase in ProbeBw");
    }

    // ── ProbeRtt phase ──────────────────────────────────────────────────────

    #[test]
    fn test_probe_rtt_reduces_cwnd() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        ctrl.state = BbrState::ProbeBw;
        ctrl.btlbw = 1_000_000.0;
        ctrl.rtprop = 0.01;
        ctrl.update_pacing_and_cwnd();
        let cwnd_before = ctrl.cwnd();

        // Force ProbeRtt.
        ctrl.state = BbrState::ProbeRtt;
        ctrl.probe_rtt_start_ms = Some(ctrl.now_ms);
        ctrl.update_pacing_and_cwnd();

        // ProbeRtt cwnd gain = 1.0 vs ProbeBw gain = 2.0 → smaller cwnd.
        assert!(
            ctrl.cwnd() <= cwnd_before,
            "ProbeRtt should reduce cwnd: {} vs {}",
            ctrl.cwnd(),
            cwnd_before
        );
    }

    #[test]
    fn test_probe_rtt_exits_after_duration() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        ctrl.state = BbrState::ProbeRtt;
        ctrl.probe_rtt_start_ms = Some(0);
        ctrl.now_ms = 0;

        // Before duration expires, stay in ProbeRtt.
        ctrl.on_ack(make_sample(1_000, 0.001, 0.01));
        // Advance time beyond PROBE_RTT_DURATION_MS.
        ctrl.now_ms = PROBE_RTT_DURATION_MS + 1;
        ctrl.on_ack(make_sample(1_000, 0.001, 0.01));

        assert_eq!(
            *ctrl.state(),
            BbrState::ProbeBw,
            "should transition to ProbeBw after ProbeRtt duration"
        );
    }

    // ── RTprop filter ───────────────────────────────────────────────────────

    #[test]
    fn test_rtprop_tracks_minimum() {
        let mut ctrl = BbrController::new(BbrConfig::default());

        // Feed samples with varying RTTs; RTprop should be the minimum.
        ctrl.on_ack(make_sample(1_000, 0.001, 0.050));
        ctrl.on_ack(make_sample(1_000, 0.001, 0.010)); // new minimum
        ctrl.on_ack(make_sample(1_000, 0.001, 0.030));

        assert!(
            ctrl.rtprop() <= 0.010 + 1e-9,
            "RTprop should track minimum RTT, got {}",
            ctrl.rtprop()
        );
    }

    #[test]
    fn test_rtprop_updates_on_better_rtt() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        ctrl.on_ack(make_sample(1_000, 0.001, 0.100));
        let rtt_before = ctrl.rtprop();

        ctrl.on_ack(make_sample(1_000, 0.001, 0.005)); // much smaller
        assert!(
            ctrl.rtprop() < rtt_before,
            "RTprop should update to smaller RTT"
        );
    }

    // ── BtlBw filter ────────────────────────────────────────────────────────

    #[test]
    fn test_btlbw_tracks_maximum() {
        let mut ctrl = BbrController::new(BbrConfig::default());

        // Vary delivered rate; BtlBw should be the max.
        ctrl.on_ack(make_sample(100_000, 0.001, 0.01)); // 100 MB/s
        ctrl.on_ack(make_sample(50_000, 0.001, 0.01)); // 50 MB/s

        // BtlBw should be ≥ 100 MB/s (the highest sample).
        assert!(
            ctrl.btlbw() >= 100_000.0 / 0.001 * 0.9,
            "BtlBw should hold peak value, got {}",
            ctrl.btlbw()
        );
    }

    #[test]
    fn test_app_limited_does_not_reduce_btlbw() {
        let mut ctrl = BbrController::new(BbrConfig::default());

        // Establish a high estimate.
        ctrl.on_ack(make_sample(100_000, 0.001, 0.01));
        let bw_high = ctrl.btlbw();

        // App-limited sample with low rate should not reduce BtlBw.
        let limited = AckSample {
            delivered: 100,
            elapsed_secs: 0.001,
            rtt_secs: 0.01,
            is_app_limited: true,
        };
        ctrl.on_ack(limited);

        assert!(
            ctrl.btlbw() >= bw_high * 0.99,
            "app-limited sample should not reduce BtlBw: {} < {}",
            ctrl.btlbw(),
            bw_high
        );
    }

    // ── inflight_target ──────────────────────────────────────────────────────

    #[test]
    fn test_inflight_target_at_least_min_cwnd() {
        let ctrl = BbrController::new(BbrConfig::default());
        assert!(
            ctrl.inflight_target() >= MIN_CWND_BYTES,
            "inflight_target should be >= MIN_CWND"
        );
    }

    #[test]
    fn test_inflight_target_proportional_to_bdp() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        ctrl.btlbw = 1_000_000.0; // 1 MB/s
        ctrl.rtprop = 0.1; // 100 ms
                           // BDP = 100 000 bytes; with startup gain ≈ 2.885 → ~288 500 bytes
        let target = ctrl.inflight_target();
        assert!(
            target > 100_000,
            "inflight_target should exceed BDP, got {target}"
        );
    }

    // ── Display ─────────────────────────────────────────────────────────────

    #[test]
    fn test_state_display() {
        assert_eq!(BbrState::Startup.to_string(), "startup");
        assert_eq!(BbrState::Drain.to_string(), "drain");
        assert_eq!(BbrState::ProbeBw.to_string(), "probe_bw");
        assert_eq!(BbrState::ProbeRtt.to_string(), "probe_rtt");
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_zero_elapsed_sample_ignored() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        let initial_bw = ctrl.btlbw();

        // A zero-elapsed sample must not cause division by zero or state corruption.
        let bad = AckSample {
            delivered: 1000,
            elapsed_secs: 0.0,
            rtt_secs: 0.01,
            is_app_limited: false,
        };
        ctrl.on_ack(bad);

        // State should be unchanged (we early-return on invalid samples).
        assert_eq!(ctrl.btlbw(), initial_bw);
        assert_eq!(*ctrl.state(), BbrState::Startup);
    }

    #[test]
    fn test_zero_rtt_sample_ignored() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        let initial_rtprop = ctrl.rtprop();

        let bad = AckSample {
            delivered: 1000,
            elapsed_secs: 0.001,
            rtt_secs: 0.0,
            is_app_limited: false,
        };
        ctrl.on_ack(bad);
        assert_eq!(ctrl.rtprop(), initial_rtprop);
    }

    #[test]
    fn test_many_acks_no_panic() {
        let mut ctrl = BbrController::new(BbrConfig::default());
        for i in 0..1_000u64 {
            let rtt = 0.005 + (i % 20) as f64 * 0.001;
            let delivered = 10_000 + (i % 5) * 1_000;
            ctrl.on_ack(make_sample(delivered, 0.001, rtt));
        }
        // Verify we end up in a defined state with sane values.
        assert!(ctrl.pacing_rate() > 0.0);
        assert!(ctrl.cwnd() >= MIN_CWND_BYTES);
    }
}
