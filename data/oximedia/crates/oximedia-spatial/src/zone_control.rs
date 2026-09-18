//! Multi-zone spatial audio rendering.
//!
//! The `zone_control` module partitions a physical space into named zones, each
//! with its own set of loudspeakers and gain/routing settings.  Audio objects
//! can be assigned to zones, and the renderer produces per-zone mixes.
//!
//! # Architecture
//!
//! ```text
//!  AudioObject ──► ZoneRouter ──► Zone A (speakers 0..3, gain 1.0)
//!                             └──► Zone B (speakers 4..7, gain 0.5)
//!                             └──► Zone C (overhead, gain 0.8)
//! ```
//!
//! # Use cases
//!
//! - **Theme park** installations: different audio in adjacent rooms.
//! - **Open-plan offices**: localised audio feeds to different areas.
//! - **Cinema**: independent front/surround/Atmos zone gain control.
//! - **Live event**: monitor mixes for stage, front-of-house, and delay towers.

use crate::SpatialError;
use std::collections::HashMap;

// ─── Types ────────────────────────────────────────────────────────────────────

/// A physical zone in the listening environment.
///
/// Each zone has a name, a list of output channel indices, and an overall gain.
#[derive(Debug, Clone)]
pub struct Zone {
    /// Unique zone identifier (must be unique within a [`ZoneManager`]).
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Output channel indices that belong to this zone.
    pub channels: Vec<usize>,
    /// Zone-wide linear gain multiplier.
    pub gain: f32,
    /// Whether this zone is currently active (muted if false).
    pub active: bool,
    /// Spatial position of the zone centre (for distance-based routing).
    /// Expressed as `[x, y, z]` in metres.
    pub position: [f32; 3],
}

/// An audio object being routed through the zone system.
#[derive(Debug, Clone)]
pub struct ZonedObject {
    /// Object ID.
    pub id: u32,
    /// Audio signal (one mono buffer).
    pub signal: Vec<f32>,
    /// World-space position `[x, y, z]` in metres.
    pub position: [f32; 3],
    /// Zone IDs this object is routed to.  Empty = broadcast to all zones.
    pub zone_ids: Vec<u32>,
    /// Object-level gain.
    pub gain: f32,
}

/// Per-zone routing policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingPolicy {
    /// Route audio only to the nearest zone.
    NearestZone,
    /// Route to all zones within `radius` metres.
    ByRadius { radius: f32 },
    /// Use the explicit zone list on each `ZonedObject`.
    Explicit,
    /// Broadcast to every active zone.
    Broadcast,
}

/// The output of the zone renderer for a single render call.
#[derive(Debug, Clone)]
pub struct ZoneRenderOutput {
    /// Rendered audio per zone: `zone_id → mono_mix_buffer`.
    pub zone_mixes: HashMap<u32, Vec<f32>>,
    /// Number of samples rendered.
    pub num_samples: usize,
}

// ─── Zone ─────────────────────────────────────────────────────────────────────

impl Zone {
    /// Create a new zone.
    pub fn new(id: u32, name: impl Into<String>, channels: Vec<usize>, position: [f32; 3]) -> Self {
        Self {
            id,
            name: name.into(),
            channels,
            gain: 1.0,
            active: true,
            position,
        }
    }

    /// Euclidean distance from this zone's centre to a point.
    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ─── ZoneManager ──────────────────────────────────────────────────────────────

/// Manages a collection of zones and their routing.
#[derive(Debug, Clone)]
pub struct ZoneManager {
    /// All registered zones, keyed by zone ID.
    pub zones: HashMap<u32, Zone>,
    /// Routing policy controlling how objects are assigned to zones.
    pub routing_policy: RoutingPolicy,
    /// Global output gain applied to all zones.
    pub master_gain: f32,
    /// Total number of output channels across all zones.
    num_output_channels: usize,
}

impl ZoneManager {
    /// Create a new zone manager.
    pub fn new(routing_policy: RoutingPolicy) -> Self {
        Self {
            zones: HashMap::new(),
            routing_policy,
            master_gain: 1.0,
            num_output_channels: 0,
        }
    }

    /// Add a zone to the manager.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if a zone with the same ID already exists.
    pub fn add_zone(&mut self, zone: Zone) -> Result<(), SpatialError> {
        if self.zones.contains_key(&zone.id) {
            return Err(SpatialError::InvalidConfig(format!(
                "Zone {} already exists",
                zone.id
            )));
        }
        // Track max output channel to determine buffer size.
        if let Some(&max_ch) = zone.channels.iter().max() {
            if max_ch + 1 > self.num_output_channels {
                self.num_output_channels = max_ch + 1;
            }
        }
        self.zones.insert(zone.id, zone);
        Ok(())
    }

    /// Remove a zone by ID.  Returns the removed zone, or `None` if not found.
    pub fn remove_zone(&mut self, id: u32) -> Option<Zone> {
        self.zones.remove(&id)
    }

    /// Set gain for a specific zone.
    pub fn set_zone_gain(&mut self, id: u32, gain: f32) {
        if let Some(zone) = self.zones.get_mut(&id) {
            zone.gain = gain;
        }
    }

    /// Mute or unmute a zone.
    pub fn set_zone_active(&mut self, id: u32, active: bool) {
        if let Some(zone) = self.zones.get_mut(&id) {
            zone.active = active;
        }
    }

    /// Find the zone IDs that should receive a given object's audio.
    fn resolve_zones(&self, obj: &ZonedObject) -> Vec<u32> {
        match self.routing_policy {
            RoutingPolicy::Explicit => obj.zone_ids.clone(),
            RoutingPolicy::Broadcast => self.zones.keys().copied().collect(),
            RoutingPolicy::NearestZone => {
                let nearest = self.zones.values().filter(|z| z.active).min_by(|a, b| {
                    a.distance_to(obj.position)
                        .partial_cmp(&b.distance_to(obj.position))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                nearest.map(|z| vec![z.id]).unwrap_or_default()
            }
            RoutingPolicy::ByRadius { radius } => self
                .zones
                .values()
                .filter(|z| z.active && z.distance_to(obj.position) <= radius)
                .map(|z| z.id)
                .collect(),
        }
    }

    /// Render a list of zoned objects into per-zone mono mixes.
    ///
    /// # Parameters
    /// - `objects`: audio objects to render.
    /// - `num_samples`: number of samples to render per zone.
    ///
    /// # Returns
    /// A [`ZoneRenderOutput`] containing one output buffer per active zone.
    pub fn render(&self, objects: &[ZonedObject], num_samples: usize) -> ZoneRenderOutput {
        let mut zone_mixes: HashMap<u32, Vec<f32>> = self
            .zones
            .keys()
            .map(|&id| (id, vec![0.0_f32; num_samples]))
            .collect();

        for obj in objects {
            let target_zones = self.resolve_zones(obj);
            let signal_len = obj.signal.len().min(num_samples);

            for zone_id in &target_zones {
                if let (Some(zone), Some(mix)) =
                    (self.zones.get(zone_id), zone_mixes.get_mut(zone_id))
                {
                    if !zone.active {
                        continue;
                    }
                    // Distance attenuation: inverse linear with zone distance.
                    let dist = zone.distance_to(obj.position).max(0.1);
                    let dist_gain = (1.0 / dist).min(1.0);
                    let effective_gain = obj.gain * zone.gain * self.master_gain * dist_gain;

                    for (n, &s) in obj.signal[..signal_len].iter().enumerate() {
                        mix[n] += s * effective_gain;
                    }
                }
            }
        }

        ZoneRenderOutput {
            zone_mixes,
            num_samples,
        }
    }

    /// Number of registered zones.
    pub fn num_zones(&self) -> usize {
        self.zones.len()
    }

    /// Iterate over all active zones.
    pub fn active_zones(&self) -> impl Iterator<Item = &Zone> {
        self.zones.values().filter(|z| z.active)
    }
}

// ─── ZoneCrossfader ───────────────────────────────────────────────────────────

/// Smooth gain crossfader between two zones.
///
/// When an audio object moves from one zone to another, abrupt gain changes
/// cause clicks.  The `ZoneCrossfader` provides a linear ramp over a
/// configurable number of samples.
#[derive(Debug, Clone)]
pub struct ZoneCrossfader {
    /// Current gain (starts at 0).
    current_gain: f32,
    /// Target gain.
    target_gain: f32,
    /// Increment per sample.
    increment: f32,
    /// Remaining ramp samples.
    remaining: usize,
}

impl ZoneCrossfader {
    /// Create a new crossfader.
    ///
    /// - `ramp_samples`: number of samples for the full gain transition.
    pub fn new(ramp_samples: usize) -> Self {
        Self {
            current_gain: 0.0,
            target_gain: 0.0,
            increment: 0.0,
            remaining: ramp_samples,
        }
    }

    /// Set a new target gain and start ramping.
    pub fn set_target(&mut self, target: f32, ramp_samples: usize) {
        self.target_gain = target;
        let diff = target - self.current_gain;
        if ramp_samples == 0 {
            self.current_gain = target;
            self.increment = 0.0;
            self.remaining = 0;
        } else {
            self.increment = diff / ramp_samples as f32;
            self.remaining = ramp_samples;
        }
    }

    /// Process one sample: advance the ramp and return the current gain.
    pub fn next_gain(&mut self) -> f32 {
        if self.remaining > 0 {
            self.current_gain += self.increment;
            self.remaining -= 1;
            // Clamp to target when ramp finishes.
            if self.remaining == 0 {
                self.current_gain = self.target_gain;
            }
        }
        self.current_gain
    }

    /// Apply the ramp to a buffer in-place.
    pub fn apply_to(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s *= self.next_gain();
        }
    }

    /// Return the current gain without advancing.
    pub fn current(&self) -> f32 {
        self.current_gain
    }

    /// Return `true` if the ramp is still active.
    pub fn is_ramping(&self) -> bool {
        self.remaining > 0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(policy: RoutingPolicy) -> ZoneManager {
        let mut mgr = ZoneManager::new(policy);
        mgr.add_zone(Zone::new(0, "Front", vec![0, 1], [0.0, 0.0, 0.0]))
            .expect("adding Front zone should succeed");
        mgr.add_zone(Zone::new(1, "Rear", vec![2, 3], [0.0, 5.0, 0.0]))
            .expect("adding Rear zone should succeed");
        mgr.add_zone(Zone::new(2, "Height", vec![4, 5], [0.0, 2.5, 3.0]))
            .expect("adding Height zone should succeed");
        mgr
    }

    fn make_object(id: u32, pos: [f32; 3]) -> ZonedObject {
        ZonedObject {
            id,
            signal: vec![1.0_f32; 64],
            position: pos,
            zone_ids: Vec::new(),
            gain: 1.0,
        }
    }

    // ── Zone ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_zone_distance_same_position() {
        let z = Zone::new(0, "A", vec![0], [1.0, 2.0, 3.0]);
        assert!((z.distance_to([1.0, 2.0, 3.0])).abs() < 1e-5);
    }

    #[test]
    fn test_zone_distance_known() {
        let z = Zone::new(0, "A", vec![0], [0.0, 0.0, 0.0]);
        let d = z.distance_to([3.0, 4.0, 0.0]);
        assert!((d - 5.0).abs() < 1e-4, "distance should be 5, got {d}");
    }

    // ── ZoneManager ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_zone_increases_count() {
        let mut mgr = ZoneManager::new(RoutingPolicy::Broadcast);
        mgr.add_zone(Zone::new(0, "A", vec![0], [0.0, 0.0, 0.0]))
            .expect("adding zone A should succeed");
        assert_eq!(mgr.num_zones(), 1);
    }

    #[test]
    fn test_add_duplicate_zone_fails() {
        let mut mgr = ZoneManager::new(RoutingPolicy::Broadcast);
        mgr.add_zone(Zone::new(0, "A", vec![0], [0.0, 0.0, 0.0]))
            .expect("adding first zone should succeed");
        assert!(mgr
            .add_zone(Zone::new(0, "B", vec![1], [1.0, 0.0, 0.0]))
            .is_err());
    }

    #[test]
    fn test_remove_zone_decreases_count() {
        let mut mgr = make_manager(RoutingPolicy::Broadcast);
        mgr.remove_zone(0);
        assert_eq!(mgr.num_zones(), 2);
    }

    #[test]
    fn test_set_zone_gain() {
        let mut mgr = make_manager(RoutingPolicy::Broadcast);
        mgr.set_zone_gain(0, 0.5);
        assert!((mgr.zones[&0].gain - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mute_zone() {
        let mut mgr = make_manager(RoutingPolicy::Broadcast);
        mgr.set_zone_active(0, false);
        assert!(!mgr.zones[&0].active);
    }

    // ── Render / routing ─────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_renders_to_all_zones() {
        let mgr = make_manager(RoutingPolicy::Broadcast);
        let obj = make_object(0, [0.0, 0.0, 0.0]);
        let out = mgr.render(&[obj], 64);
        assert_eq!(out.zone_mixes.len(), 3, "Should have 3 zones");
        for (_id, mix) in &out.zone_mixes {
            let energy: f32 = mix.iter().map(|x| x * x).sum();
            assert!(energy > 0.0, "All zones should receive energy");
        }
    }

    #[test]
    fn test_nearest_zone_routes_to_one_zone() {
        let mgr = make_manager(RoutingPolicy::NearestZone);
        let obj = make_object(0, [0.0, 0.0, 0.0]); // at origin → nearest to Front (zone 0)
        let out = mgr.render(&[obj], 64);

        // Zone 0 (Front) should have energy; others should be silent or less energetic.
        let e0: f32 = out.zone_mixes[&0].iter().map(|x| x * x).sum();
        let e1: f32 = out.zone_mixes[&1].iter().map(|x| x * x).sum();
        assert!(e0 > 0.0, "Front zone should receive energy");
        assert_eq!(
            e1, 0.0,
            "Rear zone should not receive energy with NearestZone routing"
        );
    }

    #[test]
    fn test_by_radius_routing() {
        let mgr = make_manager(RoutingPolicy::ByRadius { radius: 3.0 });
        let obj = make_object(0, [0.0, 0.0, 0.0]); // Front at 0m, Rear at 5m, Height at ~3.9m
        let out = mgr.render(&[obj], 64);

        let e0: f32 = out.zone_mixes[&0].iter().map(|x| x * x).sum();
        let e1: f32 = out.zone_mixes[&1].iter().map(|x| x * x).sum();

        assert!(e0 > 0.0, "Front (within radius) should receive energy");
        assert_eq!(e1, 0.0, "Rear (outside radius) should not receive energy");
    }

    #[test]
    fn test_explicit_routing() {
        let mgr = make_manager(RoutingPolicy::Explicit);
        let mut obj = make_object(0, [0.0, 0.0, 0.0]);
        obj.zone_ids = vec![1]; // route only to Rear
        let out = mgr.render(&[obj], 64);

        let e0: f32 = out.zone_mixes[&0].iter().map(|x| x * x).sum();
        let e1: f32 = out.zone_mixes[&1].iter().map(|x| x * x).sum();
        assert_eq!(e0, 0.0, "Front should not receive explicitly routed object");
        assert!(e1 > 0.0, "Rear should receive explicitly routed object");
    }

    #[test]
    fn test_muted_zone_receives_no_audio() {
        let mut mgr = make_manager(RoutingPolicy::Broadcast);
        mgr.set_zone_active(0, false);
        let obj = make_object(0, [0.0, 0.0, 0.0]);
        let out = mgr.render(&[obj], 64);
        let e0: f32 = out.zone_mixes[&0].iter().map(|x| x * x).sum();
        assert_eq!(e0, 0.0, "Muted zone should receive no audio");
    }

    #[test]
    fn test_render_num_samples_matches() {
        let mgr = make_manager(RoutingPolicy::Broadcast);
        let obj = make_object(0, [0.0, 0.0, 0.0]);
        let out = mgr.render(&[obj], 128);
        assert_eq!(out.num_samples, 128);
        for mix in out.zone_mixes.values() {
            assert_eq!(mix.len(), 128);
        }
    }

    #[test]
    fn test_multiple_objects_accumulate() {
        let mgr = make_manager(RoutingPolicy::Broadcast);
        let obj1 = make_object(0, [0.0, 0.0, 0.0]);
        let obj2 = make_object(1, [0.0, 0.0, 0.0]);
        let out1 = mgr.render(std::slice::from_ref(&obj1), 64);
        let out2 = mgr.render(&[obj1, obj2], 64);

        let e_single: f32 = out1.zone_mixes[&0].iter().map(|x| x * x).sum();
        let e_double: f32 = out2.zone_mixes[&0].iter().map(|x| x * x).sum();
        assert!(
            e_double > e_single,
            "Two objects should accumulate more energy: single={e_single}, double={e_double}"
        );
    }

    // ── ZoneCrossfader ────────────────────────────────────────────────────────

    #[test]
    fn test_crossfader_ramps_to_target() {
        let mut cf = ZoneCrossfader::new(100);
        cf.set_target(1.0, 100);
        for _ in 0..100 {
            cf.next_gain();
        }
        assert!(
            (cf.current() - 1.0).abs() < 1e-5,
            "Should reach target after ramp"
        );
    }

    #[test]
    fn test_crossfader_instant_transition() {
        let mut cf = ZoneCrossfader::new(0);
        cf.set_target(0.7, 0);
        assert!(
            (cf.current() - 0.7).abs() < 1e-5,
            "Instant transition should set gain immediately"
        );
    }

    #[test]
    fn test_crossfader_is_ramping() {
        let mut cf = ZoneCrossfader::new(50);
        cf.set_target(1.0, 50);
        assert!(cf.is_ramping(), "Should be ramping");
        for _ in 0..50 {
            cf.next_gain();
        }
        assert!(!cf.is_ramping(), "Should not be ramping after completion");
    }

    #[test]
    fn test_crossfader_apply_to_buffer() {
        let mut cf = ZoneCrossfader::new(64);
        cf.set_target(1.0, 64);
        let mut buf = vec![1.0_f32; 64];
        cf.apply_to(&mut buf);
        // Gain starts at 0 and ramps to 1 → first sample should be near 0.
        assert!(buf[0].abs() < 0.1, "First sample gain should be near 0");
        assert!(buf[63] > 0.9, "Last sample gain should be near 1");
    }

    #[test]
    fn test_crossfader_monotonic_increasing() {
        let mut cf = ZoneCrossfader::new(10);
        cf.set_target(1.0, 10);
        let mut prev = 0.0_f32;
        for _ in 0..10 {
            let g = cf.next_gain();
            assert!(
                g >= prev - 1e-6,
                "Gain should be monotonically non-decreasing"
            );
            prev = g;
        }
    }

    #[test]
    fn test_active_zones_iterator() {
        let mut mgr = make_manager(RoutingPolicy::Broadcast);
        mgr.set_zone_active(1, false);
        let active_count = mgr.active_zones().count();
        assert_eq!(active_count, 2, "Should have 2 active zones");
    }
}
