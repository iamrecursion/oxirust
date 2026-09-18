//! SRT congestion control algorithm.
//!
//! Implements a window-based congestion control similar to TCP with modifications
//! for real-time streaming.

use std::time::{Duration, Instant};

/// Congestion control state.
#[derive(Debug)]
pub struct CongestionControl {
    /// Current congestion window size (packets).
    cwnd: f64,
    /// Slow start threshold.
    ssthresh: f64,
    /// Maximum window size.
    max_window: f64,
    /// Minimum RTT observed (microseconds).
    min_rtt: u32,
    /// Current RTT estimate (microseconds).
    rtt: u32,
    /// RTT variance (microseconds).
    rtt_var: u32,
    /// Packet delivery rate (packets/second).
    delivery_rate: f64,
    /// Number of packets acknowledged since last update.
    acked_packets: u32,
    /// Last congestion event time.
    last_congestion: Instant,
    /// Packet loss detected.
    loss_detected: bool,
    /// Current phase (slow start or congestion avoidance).
    in_slow_start: bool,
}

impl CongestionControl {
    /// Creates a new congestion control instance.
    #[must_use]
    pub fn new(initial_window: u32, max_window: u32) -> Self {
        Self {
            cwnd: f64::from(initial_window),
            ssthresh: f64::from(max_window),
            max_window: f64::from(max_window),
            min_rtt: 100_000, // 100ms initial
            rtt: 100_000,
            rtt_var: 50_000,
            delivery_rate: 0.0,
            acked_packets: 0,
            last_congestion: Instant::now(),
            loss_detected: false,
            in_slow_start: true,
        }
    }

    /// Returns the current congestion window size.
    #[must_use]
    pub fn window_size(&self) -> u32 {
        self.cwnd as u32
    }

    /// Returns the current RTT estimate in microseconds.
    #[must_use]
    pub const fn rtt(&self) -> u32 {
        self.rtt
    }

    /// Returns the retransmission timeout (RTO) in microseconds.
    #[must_use]
    pub fn rto(&self) -> u32 {
        // RTO = RTT + 4 * RTT_VAR (RFC 6298)
        let rto = self.rtt + 4 * self.rtt_var;
        rto.clamp(200_000, 3_000_000) // 200ms to 3s
    }

    /// Updates RTT estimate with a new measurement.
    pub fn update_rtt(&mut self, measured_rtt: u32) {
        if measured_rtt < self.min_rtt {
            self.min_rtt = measured_rtt;
        }

        // RFC 6298 RTT estimation
        if self.rtt == 0 {
            self.rtt = measured_rtt;
            self.rtt_var = measured_rtt / 2;
        } else {
            let diff = if measured_rtt > self.rtt {
                measured_rtt - self.rtt
            } else {
                self.rtt - measured_rtt
            };

            // RTT_VAR = (1 - beta) * RTT_VAR + beta * |RTT - measured|
            self.rtt_var = (3 * self.rtt_var + diff) / 4;

            // RTT = (1 - alpha) * RTT + alpha * measured
            self.rtt = (7 * self.rtt + measured_rtt) / 8;
        }
    }

    /// Called when packets are acknowledged.
    pub fn on_ack(&mut self, num_packets: u32) {
        self.acked_packets += num_packets;
        self.loss_detected = false;

        if self.in_slow_start {
            // Slow start: increase cwnd by number of acked packets
            self.cwnd += f64::from(num_packets);

            if self.cwnd >= self.ssthresh {
                self.in_slow_start = false;
            }
        } else {
            // Congestion avoidance: increase cwnd by 1/cwnd per ACK
            self.cwnd += f64::from(num_packets) / self.cwnd;
        }

        // Cap at max window
        if self.cwnd > self.max_window {
            self.cwnd = self.max_window;
        }
    }

    /// Called when packet loss is detected.
    pub fn on_loss(&mut self) {
        if self.loss_detected {
            return; // Already in loss recovery
        }

        self.loss_detected = true;
        self.last_congestion = Instant::now();

        // Multiplicative decrease
        self.ssthresh = (self.cwnd / 2.0).max(2.0);
        self.cwnd = self.ssthresh;
        self.in_slow_start = false;
    }

    /// Called when congestion is detected (not from loss).
    pub fn on_congestion(&mut self) {
        // Rate-based congestion control adjustment
        self.ssthresh = (self.cwnd * 0.875).max(2.0);
        self.cwnd = self.ssthresh;
    }

    /// Updates delivery rate estimation.
    pub fn update_delivery_rate(&mut self, packets: u32, duration: Duration) {
        if duration.as_secs_f64() > 0.0 {
            let rate = f64::from(packets) / duration.as_secs_f64();
            // Exponential moving average
            self.delivery_rate = 0.8 * self.delivery_rate + 0.2 * rate;
        }
    }

    /// Returns the estimated delivery rate in packets/second.
    #[must_use]
    pub const fn delivery_rate(&self) -> f64 {
        self.delivery_rate
    }

    /// Returns true if currently in slow start phase.
    #[must_use]
    pub const fn in_slow_start(&self) -> bool {
        self.in_slow_start
    }

    /// Resets to initial state.
    pub fn reset(&mut self) {
        self.cwnd = 2.0;
        self.ssthresh = self.max_window;
        self.in_slow_start = true;
        self.loss_detected = false;
        self.acked_packets = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_congestion_control_new() {
        let cc = CongestionControl::new(10, 1000);
        assert_eq!(cc.window_size(), 10);
        assert!(cc.in_slow_start());
    }

    #[test]
    fn test_slow_start() {
        let mut cc = CongestionControl::new(10, 10000);
        cc.on_ack(5);
        assert_eq!(cc.window_size(), 15);
        assert!(cc.in_slow_start());
    }

    #[test]
    fn test_congestion_avoidance() {
        let mut cc = CongestionControl::new(10, 10000);
        cc.ssthresh = 15.0;
        cc.cwnd = 20.0;
        cc.max_window = 10000.0;
        cc.in_slow_start = false;

        let before = cc.window_size();
        cc.on_ack(50);
        assert!(cc.window_size() > before);
        assert!(!cc.in_slow_start());
    }

    #[test]
    fn test_on_loss() {
        let mut cc = CongestionControl::new(100, 1000);
        cc.cwnd = 100.0;
        cc.on_loss();

        assert_eq!(cc.window_size(), 50);
        assert!(!cc.in_slow_start());
    }

    #[test]
    fn test_rtt_update() {
        let mut cc = CongestionControl::new(10, 10000);
        // Initial RTT is 100_000, first update uses EWMA
        cc.update_rtt(50_000);
        // Result: (7*100_000 + 50_000)/8 = 93_750
        assert_eq!(cc.rtt(), 93_750);

        cc.update_rtt(60_000);
        // Should decrease from 93_750 toward 60_000
        assert!(cc.rtt() < 93_750 && cc.rtt() > 60_000);
    }

    #[test]
    fn test_rto() {
        let mut cc = CongestionControl::new(10, 10000);
        cc.rtt = 100_000;
        cc.rtt_var = 10_000;
        let rto = cc.rto();
        assert!(rto >= 200_000); // Should be at least min RTO
    }
}
