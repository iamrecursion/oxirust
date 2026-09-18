//! Neuromorphic Spike Events
//!
//! This module defines the fundamental spike event structure used throughout
//! neuromorphic computing systems. Spike events represent the basic unit of
//! communication in spiking neural networks.

/// Neuromorphic spike event
///
/// Represents a discrete spike occurring in a neuromorphic system. Each spike
/// carries information about its source, timing, amplitude, and associated
/// spatial coordinates.
///
/// # Fields
/// - `neuron_id`: Unique identifier for the neuron that generated the spike
/// - `timestamp`: Time of spike occurrence in simulation time units
/// - `amplitude`: Strength/amplitude of the spike
/// - `spatial_coords`: Spatial coordinates associated with the spike
///
/// # Example
/// ```rust
/// use scirs2_spatial::neuromorphic::core::SpikeEvent;
///
/// let spike = SpikeEvent {
///     neuron_id: 42,
///     timestamp: 1000.5,
///     amplitude: 1.0,
///     spatial_coords: vec![0.5, 0.3, 0.8],
/// };
///
/// println!("Neuron {} spiked at time {}", spike.neuron_id, spike.timestamp);
/// ```
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Source neuron ID
    pub neuron_id: usize,
    /// Spike timestamp (in simulation time units)
    pub timestamp: f64,
    /// Spike amplitude
    pub amplitude: f64,
    /// Spatial coordinates associated with the spike
    pub spatial_coords: Vec<f64>,
}

impl SpikeEvent {
    /// Create a new spike event
    ///
    /// # Arguments
    /// * `neuron_id` - ID of the neuron generating the spike
    /// * `timestamp` - Time of spike occurrence
    /// * `amplitude` - Spike amplitude (typically 1.0)
    /// * `spatial_coords` - Associated spatial coordinates
    ///
    /// # Returns
    /// A new `SpikeEvent` instance
    pub fn new(neuron_id: usize, timestamp: f64, amplitude: f64, spatial_coords: Vec<f64>) -> Self {
        Self {
            neuron_id,
            timestamp,
            amplitude,
            spatial_coords,
        }
    }

    /// Get the neuron ID that generated this spike
    pub fn neuron_id(&self) -> usize {
        self.neuron_id
    }

    /// Get the timestamp of this spike
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// Get the amplitude of this spike
    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Get the spatial coordinates associated with this spike
    pub fn spatial_coords(&self) -> &[f64] {
        &self.spatial_coords
    }

    /// Get the number of spatial dimensions
    pub fn spatial_dims(&self) -> usize {
        self.spatial_coords.len()
    }

    /// Calculate the temporal distance to another spike
    ///
    /// # Arguments
    /// * `other` - Another spike event
    ///
    /// # Returns
    /// Absolute temporal distance between the two spikes
    pub fn temporal_distance(&self, other: &SpikeEvent) -> f64 {
        (self.timestamp - other.timestamp).abs()
    }

    /// Calculate the spatial distance to another spike
    ///
    /// # Arguments
    /// * `other` - Another spike event
    ///
    /// # Returns
    /// Euclidean distance between spatial coordinates, or None if dimensions don't match
    pub fn spatial_distance(&self, other: &SpikeEvent) -> Option<f64> {
        if self.spatial_coords.len() != other.spatial_coords.len() {
            return None;
        }

        let distance = self
            .spatial_coords
            .iter()
            .zip(other.spatial_coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        Some(distance)
    }

    /// Check if this spike occurs before another spike
    ///
    /// # Arguments
    /// * `other` - Another spike event
    ///
    /// # Returns
    /// True if this spike occurs before the other spike
    pub fn is_before(&self, other: &SpikeEvent) -> bool {
        self.timestamp < other.timestamp
    }

    /// Check if this spike occurs after another spike
    ///
    /// # Arguments
    /// * `other` - Another spike event
    ///
    /// # Returns
    /// True if this spike occurs after the other spike
    pub fn is_after(&self, other: &SpikeEvent) -> bool {
        self.timestamp > other.timestamp
    }

    /// Check if two spikes are from the same neuron
    ///
    /// # Arguments
    /// * `other` - Another spike event
    ///
    /// # Returns
    /// True if both spikes are from the same neuron
    pub fn same_neuron(&self, other: &SpikeEvent) -> bool {
        self.neuron_id == other.neuron_id
    }
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.neuron_id == other.neuron_id
            && (self.timestamp - other.timestamp).abs() < 1e-9
            && (self.amplitude - other.amplitude).abs() < 1e-9
            && self.spatial_coords == other.spatial_coords
    }
}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.timestamp.partial_cmp(&other.timestamp)
    }
}

/// Collection of spike events with utilities for temporal and spatial analysis
#[derive(Debug, Clone)]
pub struct SpikeSequence {
    /// Vector of spike events
    events: Vec<SpikeEvent>,
}

impl SpikeSequence {
    /// Create a new empty spike sequence
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Create a spike sequence from a vector of events
    ///
    /// # Arguments
    /// * `events` - Vector of spike events
    ///
    /// # Returns
    /// A new `SpikeSequence` containing the events
    pub fn from_events(events: Vec<SpikeEvent>) -> Self {
        Self { events }
    }

    /// Add a spike event to the sequence
    ///
    /// # Arguments
    /// * `event` - Spike event to add
    pub fn add_event(&mut self, event: SpikeEvent) {
        self.events.push(event);
    }

    /// Get all spike events
    pub fn events(&self) -> &[SpikeEvent] {
        &self.events
    }

    /// Get the number of spike events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Sort events by timestamp
    pub fn sort_by_time(&mut self) {
        self.events.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .expect("Operation failed")
        });
    }

    /// Get events within a time window
    ///
    /// # Arguments
    /// * `start_time` - Start of time window
    /// * `end_time` - End of time window
    ///
    /// # Returns
    /// Vector of spike events within the time window
    pub fn events_in_window(&self, start_time: f64, end_time: f64) -> Vec<&SpikeEvent> {
        self.events
            .iter()
            .filter(|event| event.timestamp >= start_time && event.timestamp <= end_time)
            .collect()
    }

    /// Get events from a specific neuron
    ///
    /// # Arguments
    /// * `neuron_id` - ID of the neuron
    ///
    /// # Returns
    /// Vector of spike events from the specified neuron
    pub fn events_from_neuron(&self, neuron_id: usize) -> Vec<&SpikeEvent> {
        self.events
            .iter()
            .filter(|event| event.neuron_id == neuron_id)
            .collect()
    }

    /// Calculate average firing rate for a neuron
    ///
    /// # Arguments
    /// * `neuron_id` - ID of the neuron
    /// * `time_window` - Duration of the time window
    ///
    /// # Returns
    /// Average firing rate in spikes per time unit
    pub fn firing_rate(&self, neuron_id: usize, time_window: f64) -> f64 {
        let neuron_events = self.events_from_neuron(neuron_id);
        if neuron_events.is_empty() || time_window <= 0.0 {
            return 0.0;
        }

        neuron_events.len() as f64 / time_window
    }

    /// Get the time span of the sequence
    ///
    /// # Returns
    /// Tuple of (earliest_time, latest_time), or None if sequence is empty
    pub fn time_span(&self) -> Option<(f64, f64)> {
        if self.events.is_empty() {
            return None;
        }

        let times: Vec<f64> = self.events.iter().map(|event| event.timestamp).collect();
        let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Some((min_time, max_time))
    }
}

impl Default for SpikeSequence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_creation() {
        let spike = SpikeEvent::new(42, 1000.5, 1.0, vec![0.5, 0.3, 0.8]);

        assert_eq!(spike.neuron_id(), 42);
        assert_eq!(spike.timestamp(), 1000.5);
        assert_eq!(spike.amplitude(), 1.0);
        assert_eq!(spike.spatial_coords(), &[0.5, 0.3, 0.8]);
        assert_eq!(spike.spatial_dims(), 3);
    }

    #[test]
    fn test_temporal_distance() {
        let spike1 = SpikeEvent::new(1, 100.0, 1.0, vec![0.0]);
        let spike2 = SpikeEvent::new(2, 105.0, 1.0, vec![1.0]);

        assert_eq!(spike1.temporal_distance(&spike2), 5.0);
        assert_eq!(spike2.temporal_distance(&spike1), 5.0);
    }

    #[test]
    fn test_spatial_distance() {
        let spike1 = SpikeEvent::new(1, 100.0, 1.0, vec![0.0, 0.0]);
        let spike2 = SpikeEvent::new(2, 105.0, 1.0, vec![3.0, 4.0]);

        assert_eq!(spike1.spatial_distance(&spike2), Some(5.0));

        // Test mismatched dimensions
        let spike3 = SpikeEvent::new(3, 110.0, 1.0, vec![1.0]);
        assert_eq!(spike1.spatial_distance(&spike3), None);
    }

    #[test]
    fn test_spike_ordering() {
        let spike1 = SpikeEvent::new(1, 100.0, 1.0, vec![0.0]);
        let spike2 = SpikeEvent::new(2, 105.0, 1.0, vec![1.0]);

        assert!(spike1.is_before(&spike2));
        assert!(spike2.is_after(&spike1));
        assert!(!spike1.is_after(&spike2));
        assert!(!spike2.is_before(&spike1));
    }

    #[test]
    fn test_spike_sequence() {
        let mut sequence = SpikeSequence::new();
        assert!(sequence.is_empty());

        let spike1 = SpikeEvent::new(1, 100.0, 1.0, vec![0.0]);
        let spike2 = SpikeEvent::new(2, 105.0, 1.0, vec![1.0]);

        sequence.add_event(spike1);
        sequence.add_event(spike2);

        assert_eq!(sequence.len(), 2);
        assert!(!sequence.is_empty());

        let time_span = sequence.time_span().expect("Operation failed");
        assert_eq!(time_span.0, 100.0);
        assert_eq!(time_span.1, 105.0);
    }

    #[test]
    fn test_events_in_window() {
        let mut sequence = SpikeSequence::new();
        sequence.add_event(SpikeEvent::new(1, 95.0, 1.0, vec![0.0]));
        sequence.add_event(SpikeEvent::new(2, 100.0, 1.0, vec![1.0]));
        sequence.add_event(SpikeEvent::new(3, 105.0, 1.0, vec![2.0]));
        sequence.add_event(SpikeEvent::new(4, 110.0, 1.0, vec![3.0]));

        let windowed_events = sequence.events_in_window(100.0, 105.0);
        assert_eq!(windowed_events.len(), 2);
        assert_eq!(windowed_events[0].neuron_id, 2);
        assert_eq!(windowed_events[1].neuron_id, 3);
    }

    #[test]
    fn test_firing_rate() {
        let mut sequence = SpikeSequence::new();
        sequence.add_event(SpikeEvent::new(1, 100.0, 1.0, vec![0.0]));
        sequence.add_event(SpikeEvent::new(1, 105.0, 1.0, vec![0.0]));
        sequence.add_event(SpikeEvent::new(1, 110.0, 1.0, vec![0.0]));
        sequence.add_event(SpikeEvent::new(2, 107.0, 1.0, vec![1.0]));

        // Neuron 1 has 3 spikes in 10 time units = 0.3 spikes per unit
        assert_eq!(sequence.firing_rate(1, 10.0), 0.3);

        // Neuron 2 has 1 spike in 10 time units = 0.1 spikes per unit
        assert_eq!(sequence.firing_rate(2, 10.0), 0.1);

        // Non-existent neuron
        assert_eq!(sequence.firing_rate(99, 10.0), 0.0);
    }
}
