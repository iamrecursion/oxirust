//! Full mixer scene and snapshot management.
//!
//! Provides serializable snapshots of the entire mixer state, a snapshot library,
//! diff utilities, and safe partial recall.

use std::collections::HashMap;

/// State of a single channel at snapshot time.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelState {
    /// Channel index.
    pub channel_id: u32,
    /// Fader gain in dB.
    pub gain_db: f32,
    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pub pan: f32,
    /// Muted state.
    pub muted: bool,
    /// Send configurations: `(bus_id, send_level)`.
    pub sends: Vec<(u32, f32)>,
}

impl ChannelState {
    /// Create a default channel state.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(channel_id: u32) -> Self {
        Self {
            channel_id,
            gain_db: 0.0,
            pan: 0.0,
            muted: false,
            sends: Vec::new(),
        }
    }
}

/// State of a single bus at snapshot time.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct BusState {
    /// Bus index.
    pub bus_id: u32,
    /// Fader gain in dB.
    pub gain_db: f32,
    /// Muted state.
    pub muted: bool,
    /// Solo state.
    pub solo: bool,
}

impl BusState {
    /// Create a default bus state.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(bus_id: u32) -> Self {
        Self {
            bus_id,
            gain_db: 0.0,
            muted: false,
            solo: false,
        }
    }
}

/// A complete snapshot of the mixer state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MixerSnapshot {
    /// Unique snapshot identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Creation timestamp (milliseconds since Unix epoch).
    pub created_at_ms: u64,
    /// Per-channel states.
    pub channel_states: Vec<ChannelState>,
    /// Per-bus states.
    pub bus_states: Vec<BusState>,
}

impl MixerSnapshot {
    /// Create a new snapshot.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        created_at_ms: u64,
        channel_states: Vec<ChannelState>,
        bus_states: Vec<BusState>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            created_at_ms,
            channel_states,
            bus_states,
        }
    }
}

/// Library of named mixer snapshots.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct SnapshotLibrary {
    snapshots: HashMap<String, MixerSnapshot>,
}

impl SnapshotLibrary {
    /// Create a new empty library.
    #[must_use]
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
        }
    }

    /// Store a snapshot in the library.
    ///
    /// Overwrites any existing snapshot with the same ID.
    #[allow(dead_code)]
    pub fn store(&mut self, snap: MixerSnapshot) {
        self.snapshots.insert(snap.id.clone(), snap);
    }

    /// Recall a snapshot by its ID.
    #[must_use]
    #[allow(dead_code)]
    pub fn recall(&self, id: &str) -> Option<&MixerSnapshot> {
        self.snapshots.get(id)
    }

    /// List all snapshot names.
    #[must_use]
    #[allow(dead_code)]
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.snapshots.values().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Delete a snapshot by ID. Returns `true` if it existed.
    #[allow(dead_code)]
    pub fn delete(&mut self, id: &str) -> bool {
        self.snapshots.remove(id).is_some()
    }

    /// Number of stored snapshots.
    #[must_use]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` if no snapshots are stored.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

/// Computes a human-readable diff between two snapshots.
pub struct SnapshotDiff;

impl SnapshotDiff {
    /// Compute a list of change descriptions between snapshots `a` and `b`.
    #[must_use]
    #[allow(dead_code)]
    pub fn compute(a: &MixerSnapshot, b: &MixerSnapshot) -> Vec<String> {
        let mut changes = Vec::new();

        // Index states by channel ID
        let a_ch: HashMap<u32, &ChannelState> =
            a.channel_states.iter().map(|s| (s.channel_id, s)).collect();
        let b_ch: HashMap<u32, &ChannelState> =
            b.channel_states.iter().map(|s| (s.channel_id, s)).collect();

        // Channels present in both
        for (&id, &a_state) in &a_ch {
            if let Some(&b_state) = b_ch.get(&id) {
                if (a_state.gain_db - b_state.gain_db).abs() > 0.01 {
                    changes.push(format!(
                        "Channel {id}: gain_db {:.2} → {:.2}",
                        a_state.gain_db, b_state.gain_db
                    ));
                }
                if (a_state.pan - b_state.pan).abs() > 0.001 {
                    changes.push(format!(
                        "Channel {id}: pan {:.3} → {:.3}",
                        a_state.pan, b_state.pan
                    ));
                }
                if a_state.muted != b_state.muted {
                    changes.push(format!(
                        "Channel {id}: muted {} → {}",
                        a_state.muted, b_state.muted
                    ));
                }
            }
        }

        // Channels only in B (added)
        for &id in b_ch.keys() {
            if !a_ch.contains_key(&id) {
                changes.push(format!("Channel {id}: added"));
            }
        }

        // Channels only in A (removed)
        for &id in a_ch.keys() {
            if !b_ch.contains_key(&id) {
                changes.push(format!("Channel {id}: removed"));
            }
        }

        // Buses
        let a_bus: HashMap<u32, &BusState> = a.bus_states.iter().map(|s| (s.bus_id, s)).collect();
        let b_bus: HashMap<u32, &BusState> = b.bus_states.iter().map(|s| (s.bus_id, s)).collect();

        for (&id, &a_state) in &a_bus {
            if let Some(&b_state) = b_bus.get(&id) {
                if (a_state.gain_db - b_state.gain_db).abs() > 0.01 {
                    changes.push(format!(
                        "Bus {id}: gain_db {:.2} → {:.2}",
                        a_state.gain_db, b_state.gain_db
                    ));
                }
                if a_state.muted != b_state.muted {
                    changes.push(format!(
                        "Bus {id}: muted {} → {}",
                        a_state.muted, b_state.muted
                    ));
                }
                if a_state.solo != b_state.solo {
                    changes.push(format!(
                        "Bus {id}: solo {} → {}",
                        a_state.solo, b_state.solo
                    ));
                }
            }
        }

        changes.sort();
        changes
    }
}

/// Safe (partial) recall filter — only recalls specific channels and/or parameters.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SafeRecall {
    /// Only recall these channel IDs (empty = recall all channels).
    pub filter_channels: Vec<u32>,
    /// Only recall these parameter names: "gain", "pan", "muted", "sends".
    /// Empty = recall all parameters.
    pub filter_params: Vec<String>,
}

impl SafeRecall {
    /// Create a new `SafeRecall` with no filters (recalls everything).
    #[must_use]
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a partial recall from `source` to `target`, respecting filters.
    ///
    /// Returns a new `MixerSnapshot` that represents `target` updated with filtered values from `source`.
    #[must_use]
    #[allow(dead_code)]
    pub fn apply(&self, source: &MixerSnapshot, target: &MixerSnapshot) -> MixerSnapshot {
        let recall_all_channels = self.filter_channels.is_empty();
        let recall_all_params = self.filter_params.is_empty();

        let recall_param = |param: &str| -> bool {
            recall_all_params || self.filter_params.iter().any(|p| p == param)
        };

        let recall_channel =
            |id: u32| -> bool { recall_all_channels || self.filter_channels.contains(&id) };

        // Build lookup from source
        let src_ch: HashMap<u32, &ChannelState> = source
            .channel_states
            .iter()
            .map(|s| (s.channel_id, s))
            .collect();

        let new_channels: Vec<ChannelState> = target
            .channel_states
            .iter()
            .map(|tgt| {
                if !recall_channel(tgt.channel_id) {
                    return tgt.clone();
                }
                if let Some(&src) = src_ch.get(&tgt.channel_id) {
                    ChannelState {
                        channel_id: tgt.channel_id,
                        gain_db: if recall_param("gain") {
                            src.gain_db
                        } else {
                            tgt.gain_db
                        },
                        pan: if recall_param("pan") {
                            src.pan
                        } else {
                            tgt.pan
                        },
                        muted: if recall_param("muted") {
                            src.muted
                        } else {
                            tgt.muted
                        },
                        sends: if recall_param("sends") {
                            src.sends.clone()
                        } else {
                            tgt.sends.clone()
                        },
                    }
                } else {
                    tgt.clone()
                }
            })
            .collect();

        MixerSnapshot {
            id: target.id.clone(),
            name: target.name.clone(),
            created_at_ms: target.created_at_ms,
            channel_states: new_channels,
            bus_states: target.bus_states.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(id: &str, name: &str, gain: f32, muted: bool) -> MixerSnapshot {
        let ch = ChannelState {
            channel_id: 0,
            gain_db: gain,
            pan: 0.0,
            muted,
            sends: Vec::new(),
        };
        let bus = BusState {
            bus_id: 0,
            gain_db: 0.0,
            muted: false,
            solo: false,
        };
        MixerSnapshot::new(id, name, 1000, vec![ch], vec![bus])
    }

    #[test]
    fn test_channel_state_new() {
        let s = ChannelState::new(5);
        assert_eq!(s.channel_id, 5);
        assert_eq!(s.gain_db, 0.0);
        assert!(!s.muted);
    }

    #[test]
    fn test_bus_state_new() {
        let s = BusState::new(3);
        assert_eq!(s.bus_id, 3);
        assert!(!s.solo);
    }

    #[test]
    fn test_library_store_and_recall() {
        let mut lib = SnapshotLibrary::new();
        let snap = make_snapshot("snap1", "Main Mix", -6.0, false);
        lib.store(snap);
        assert!(lib.recall("snap1").is_some());
        assert!(lib.recall("nonexistent").is_none());
    }

    #[test]
    fn test_library_list() {
        let mut lib = SnapshotLibrary::new();
        lib.store(make_snapshot("s2", "Verse", 0.0, false));
        lib.store(make_snapshot("s1", "Chorus", 0.0, false));
        let names = lib.list();
        assert_eq!(names.len(), 2);
        // Names are sorted
        assert!(names[0] <= names[1]);
    }

    #[test]
    fn test_library_delete() {
        let mut lib = SnapshotLibrary::new();
        lib.store(make_snapshot("s1", "Main", 0.0, false));
        assert!(lib.delete("s1"));
        assert!(!lib.delete("s1")); // already deleted
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_len() {
        let mut lib = SnapshotLibrary::new();
        assert_eq!(lib.len(), 0);
        lib.store(make_snapshot("a", "A", 0.0, false));
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn test_snapshot_diff_no_changes() {
        let a = make_snapshot("s", "Mix", -6.0, false);
        let b = make_snapshot("s", "Mix", -6.0, false);
        let changes = SnapshotDiff::compute(&a, &b);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_snapshot_diff_gain_change() {
        let a = make_snapshot("s", "Mix", -6.0, false);
        let b = make_snapshot("s", "Mix", -12.0, false);
        let changes = SnapshotDiff::compute(&a, &b);
        assert!(!changes.is_empty());
        assert!(changes.iter().any(|c| c.contains("gain_db")));
    }

    #[test]
    fn test_snapshot_diff_mute_change() {
        let a = make_snapshot("s", "Mix", 0.0, false);
        let b = make_snapshot("s", "Mix", 0.0, true);
        let changes = SnapshotDiff::compute(&a, &b);
        assert!(changes.iter().any(|c| c.contains("muted")));
    }

    #[test]
    fn test_safe_recall_all_params() {
        let source = make_snapshot("src", "Source", -6.0, true);
        let target = make_snapshot("tgt", "Target", 0.0, false);
        let recall = SafeRecall::new();
        let result = recall.apply(&source, &target);
        assert!((result.channel_states[0].gain_db - (-6.0)).abs() < f32::EPSILON);
        assert!(result.channel_states[0].muted);
    }

    #[test]
    fn test_safe_recall_filter_param() {
        let source = make_snapshot("src", "Source", -6.0, true);
        let target = make_snapshot("tgt", "Target", 0.0, false);
        let recall = SafeRecall {
            filter_channels: Vec::new(),
            filter_params: vec!["gain".to_string()],
        };
        let result = recall.apply(&source, &target);
        // gain should be recalled
        assert!((result.channel_states[0].gain_db - (-6.0)).abs() < f32::EPSILON);
        // muted should remain from target
        assert!(!result.channel_states[0].muted);
    }

    #[test]
    fn test_safe_recall_filter_channel() {
        let source = MixerSnapshot::new(
            "src",
            "Source",
            0,
            vec![
                ChannelState {
                    channel_id: 0,
                    gain_db: -6.0,
                    pan: 0.0,
                    muted: false,
                    sends: vec![],
                },
                ChannelState {
                    channel_id: 1,
                    gain_db: -12.0,
                    pan: 0.0,
                    muted: false,
                    sends: vec![],
                },
            ],
            Vec::new(),
        );
        let target = MixerSnapshot::new(
            "tgt",
            "Target",
            0,
            vec![
                ChannelState {
                    channel_id: 0,
                    gain_db: 0.0,
                    pan: 0.0,
                    muted: false,
                    sends: vec![],
                },
                ChannelState {
                    channel_id: 1,
                    gain_db: 0.0,
                    pan: 0.0,
                    muted: false,
                    sends: vec![],
                },
            ],
            Vec::new(),
        );
        let recall = SafeRecall {
            filter_channels: vec![0],
            filter_params: Vec::new(),
        };
        let result = recall.apply(&source, &target);
        // Channel 0 should be recalled
        assert!((result.channel_states[0].gain_db - (-6.0)).abs() < f32::EPSILON);
        // Channel 1 should remain from target
        assert!((result.channel_states[1].gain_db - 0.0).abs() < f32::EPSILON);
    }
}
