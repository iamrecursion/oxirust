//! NDI bandwidth management and adaptive bitrate control.
//!
//! This module provides tools for managing NDI network bandwidth usage,
//! including bandwidth mode selection, EWMA-based estimation, and QoS configuration.

#![allow(dead_code)]

/// NDI bandwidth mode determining the quality/bandwidth tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdiBandwidthMode {
    /// Lowest bandwidth mode - heavily compressed, suitable for constrained networks.
    Lowest,
    /// Highest bandwidth mode - best quality, suitable for high-bandwidth networks.
    Highest,
    /// Audio-only mode - no video is transmitted.
    AudioOnly,
    /// Metadata-only mode - no audio or video, only control metadata.
    MetadataOnly,
}

impl NdiBandwidthMode {
    /// Returns the target bitrate in Mbps for this bandwidth mode.
    pub fn bitrate_mbps(&self) -> f32 {
        match self {
            Self::Lowest => 5.0,
            Self::Highest => 100.0,
            Self::AudioOnly => 0.5,
            Self::MetadataOnly => 0.01,
        }
    }
}

/// Network statistics for an NDI connection.
#[derive(Debug, Clone, Copy, Default)]
pub struct NdiNetworkStats {
    /// Total bytes sent over this connection.
    pub bytes_sent: u64,
    /// Total bytes received over this connection.
    pub bytes_received: u64,
    /// Number of packets lost.
    pub packets_lost: u32,
    /// Jitter in milliseconds.
    pub jitter_ms: f32,
    /// Round-trip time in milliseconds.
    pub rtt_ms: f32,
}

/// Controls NDI bandwidth by selecting appropriate modes based on available network capacity.
#[derive(Debug, Clone)]
pub struct BandwidthController {
    /// The target bandwidth mode.
    pub target_mode: NdiBandwidthMode,
    /// Current measured bitrate in Mbps.
    pub current_bitrate_mbps: f32,
}

impl BandwidthController {
    /// Creates a new `BandwidthController` with the given target mode.
    pub fn new(target_mode: NdiBandwidthMode) -> Self {
        let current_bitrate_mbps = target_mode.bitrate_mbps();
        Self {
            target_mode,
            current_bitrate_mbps,
        }
    }

    /// Adjusts the bandwidth mode based on the available network capacity (in Mbps).
    ///
    /// Returns the recommended `NdiBandwidthMode` for the available bandwidth.
    pub fn adjust(&mut self, available_mbps: f32) -> NdiBandwidthMode {
        self.current_bitrate_mbps = available_mbps;

        if available_mbps < NdiBandwidthMode::AudioOnly.bitrate_mbps() {
            self.target_mode = NdiBandwidthMode::MetadataOnly;
        } else if available_mbps < NdiBandwidthMode::Lowest.bitrate_mbps() {
            self.target_mode = NdiBandwidthMode::AudioOnly;
        } else if available_mbps < NdiBandwidthMode::Highest.bitrate_mbps() {
            self.target_mode = NdiBandwidthMode::Lowest;
        } else {
            self.target_mode = NdiBandwidthMode::Highest;
        }

        self.target_mode
    }
}

impl Default for BandwidthController {
    fn default() -> Self {
        Self::new(NdiBandwidthMode::Highest)
    }
}

/// Adaptive bitrate controller using Exponentially Weighted Moving Average (EWMA).
///
/// Estimates available network bandwidth and recommends appropriate `NdiBandwidthMode`.
#[derive(Debug, Clone)]
pub struct AdaptiveBitrateController {
    /// EWMA smoothing factor (0 < alpha <= 1). Higher values give more weight to recent samples.
    alpha: f64,
    /// Current EWMA bandwidth estimate in Mbps.
    estimated_mbps: f64,
    /// Whether at least one sample has been recorded.
    initialized: bool,
}

impl AdaptiveBitrateController {
    /// Creates a new `AdaptiveBitrateController`.
    ///
    /// `alpha` is the EWMA smoothing factor. A value of `0.1` is a conservative (slow-reacting)
    /// estimate; `0.5` reacts more quickly to changes.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            estimated_mbps: 0.0,
            initialized: false,
        }
    }

    /// Updates the bandwidth estimate given the number of bytes observed in a time window.
    ///
    /// # Arguments
    /// * `bytes_in_window` - Number of bytes transferred during the measurement window.
    /// * `window_ms` - Duration of the measurement window in milliseconds.
    pub fn update(&mut self, bytes_in_window: u64, window_ms: u64) {
        if window_ms == 0 {
            return;
        }
        // Convert to Mbps: (bytes * 8 bits/byte) / (window_ms * 1000 us/ms) -> Mbps
        let measured_mbps = (bytes_in_window as f64 * 8.0) / (window_ms as f64 * 1000.0);

        if self.initialized {
            self.estimated_mbps =
                self.alpha * measured_mbps + (1.0 - self.alpha) * self.estimated_mbps;
        } else {
            self.estimated_mbps = measured_mbps;
            self.initialized = true;
        }
    }

    /// Returns the recommended `NdiBandwidthMode` based on the current bandwidth estimate.
    pub fn recommended_mode(&self) -> NdiBandwidthMode {
        let mbps = self.estimated_mbps as f32;

        if mbps < NdiBandwidthMode::AudioOnly.bitrate_mbps() {
            NdiBandwidthMode::MetadataOnly
        } else if mbps < NdiBandwidthMode::Lowest.bitrate_mbps() {
            NdiBandwidthMode::AudioOnly
        } else if mbps < NdiBandwidthMode::Highest.bitrate_mbps() {
            NdiBandwidthMode::Lowest
        } else {
            NdiBandwidthMode::Highest
        }
    }

    /// Returns the current EWMA bandwidth estimate in Mbps.
    pub fn estimated_mbps(&self) -> f64 {
        self.estimated_mbps
    }
}

impl Default for AdaptiveBitrateController {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// QoS (Quality of Service) configuration for NDI streams.
#[derive(Debug, Clone, Copy)]
pub struct NdiQosConfig {
    /// Stream priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// DSCP (Differentiated Services Code Point) class for IP traffic marking.
    pub dscp_class: u8,
    /// Maximum allowed buffer time in milliseconds before dropping frames.
    pub max_buffer_ms: u32,
}

impl NdiQosConfig {
    /// Creates a realtime QoS configuration suitable for live production.
    ///
    /// Uses highest priority and EF (Expedited Forwarding) DSCP class.
    pub fn realtime() -> Self {
        Self {
            priority: 255,
            dscp_class: 46, // EF (Expedited Forwarding) - lowest latency
            max_buffer_ms: 16,
        }
    }

    /// Creates a broadcast QoS configuration suitable for high-quality broadcast delivery.
    ///
    /// Uses high priority and AF41 DSCP class.
    pub fn broadcast() -> Self {
        Self {
            priority: 200,
            dscp_class: 34, // AF41 - high-priority assured forwarding
            max_buffer_ms: 100,
        }
    }
}

impl Default for NdiQosConfig {
    fn default() -> Self {
        Self::broadcast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_mode_bitrates() {
        assert!(NdiBandwidthMode::Lowest.bitrate_mbps() < NdiBandwidthMode::Highest.bitrate_mbps());
        assert!(
            NdiBandwidthMode::AudioOnly.bitrate_mbps() < NdiBandwidthMode::Lowest.bitrate_mbps()
        );
        assert!(
            NdiBandwidthMode::MetadataOnly.bitrate_mbps()
                < NdiBandwidthMode::AudioOnly.bitrate_mbps()
        );
    }

    #[test]
    fn test_bandwidth_mode_highest_bitrate() {
        assert_eq!(NdiBandwidthMode::Highest.bitrate_mbps(), 100.0);
    }

    #[test]
    fn test_bandwidth_mode_audio_only() {
        let mode = NdiBandwidthMode::AudioOnly;
        assert!(mode.bitrate_mbps() > 0.0);
        assert!(mode.bitrate_mbps() < 5.0);
    }

    #[test]
    fn test_network_stats_default() {
        let stats = NdiNetworkStats::default();
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
        assert_eq!(stats.packets_lost, 0);
    }

    #[test]
    fn test_bandwidth_controller_adjust_high() {
        let mut ctrl = BandwidthController::default();
        let mode = ctrl.adjust(200.0);
        assert_eq!(mode, NdiBandwidthMode::Highest);
    }

    #[test]
    fn test_bandwidth_controller_adjust_low() {
        let mut ctrl = BandwidthController::default();
        let mode = ctrl.adjust(2.0);
        assert_eq!(mode, NdiBandwidthMode::AudioOnly);
    }

    #[test]
    fn test_bandwidth_controller_adjust_very_low() {
        let mut ctrl = BandwidthController::default();
        let mode = ctrl.adjust(0.001);
        assert_eq!(mode, NdiBandwidthMode::MetadataOnly);
    }

    #[test]
    fn test_bandwidth_controller_adjust_medium() {
        let mut ctrl = BandwidthController::default();
        let mode = ctrl.adjust(10.0);
        assert_eq!(mode, NdiBandwidthMode::Lowest);
    }

    #[test]
    fn test_adaptive_controller_ewma_update() {
        let mut ctrl = AdaptiveBitrateController::new(0.5);
        // 125 MB in 1000ms = 1 Gbps = 1000 Mbps
        ctrl.update(125_000_000, 1000);
        assert!(ctrl.estimated_mbps() > 0.0);
    }

    #[test]
    fn test_adaptive_controller_recommended_mode() {
        let mut ctrl = AdaptiveBitrateController::new(1.0);
        // 12.5 MB in 1000ms = 100 Mbps -> Highest
        ctrl.update(12_500_000, 1000);
        assert_eq!(ctrl.recommended_mode(), NdiBandwidthMode::Highest);
    }

    #[test]
    fn test_adaptive_controller_low_bandwidth() {
        let mut ctrl = AdaptiveBitrateController::new(1.0);
        // 1250 bytes in 1000ms = 0.01 Mbps -> MetadataOnly (< AudioOnly threshold of 0.5 Mbps)
        ctrl.update(1_250, 1000);
        assert_eq!(ctrl.recommended_mode(), NdiBandwidthMode::MetadataOnly);
    }

    #[test]
    fn test_adaptive_controller_window_zero() {
        let mut ctrl = AdaptiveBitrateController::new(0.5);
        ctrl.update(1_000_000, 0); // should not panic, should be ignored
        assert_eq!(ctrl.estimated_mbps(), 0.0);
    }

    #[test]
    fn test_qos_config_realtime() {
        let qos = NdiQosConfig::realtime();
        assert_eq!(qos.priority, 255);
        assert_eq!(qos.dscp_class, 46);
        assert!(qos.max_buffer_ms <= 20);
    }

    #[test]
    fn test_qos_config_broadcast() {
        let qos = NdiQosConfig::broadcast();
        assert!(qos.priority > 100);
        assert!(qos.max_buffer_ms >= 50);
    }

    #[test]
    fn test_qos_realtime_lower_latency_than_broadcast() {
        let rt = NdiQosConfig::realtime();
        let bc = NdiQosConfig::broadcast();
        assert!(rt.max_buffer_ms < bc.max_buffer_ms);
        assert!(rt.priority >= bc.priority);
    }
}
