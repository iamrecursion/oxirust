//! Professional audio mixing bus with routing matrix.
//!
//! Provides [`MixBus`] for per-bus accumulation and fader control, and
//! [`MixMatrix`] for an N-bus routing graph with cycle detection and a
//! single-pass render path.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────── BusSendEntry ───

/// Auxiliary send from one bus to another.
#[derive(Debug, Clone)]
pub struct BusSendEntry {
    /// ID of the destination bus.
    pub destination_bus: String,
    /// Send level in dB (negative = attenuate, positive = boost).
    pub level_db: f32,
    /// Pre-fader send: signal tapped before the master fader is applied.
    pub pre_fader: bool,
}

impl BusSendEntry {
    /// Create a new send entry.
    #[must_use]
    pub fn new(destination_bus: &str, level_db: f32, pre_fader: bool) -> Self {
        Self {
            destination_bus: destination_bus.to_string(),
            level_db,
            pre_fader,
        }
    }
}

// ─────────────────────────────────────────────────────────────── MixBus ───

/// Single bus in a [`MixMatrix`].
///
/// A bus accumulates audio from one or more sources into an internal buffer,
/// applies a master fader, and can optionally route signal to other buses via
/// aux sends.
#[derive(Debug, Clone)]
pub struct MixBus {
    /// Unique bus identifier.
    pub id: String,
    /// Human-readable bus name.
    pub name: String,
    /// Number of channels (1 = mono, 2 = stereo, 6 = 5.1, 8 = 7.1).
    pub channels: usize,
    /// Master fader level in dB (−∞ to +12 dB; use −144 for silence).
    pub fader_db: f32,
    /// Whether this bus is muted.
    pub muted: bool,
    /// Whether this bus is soloed.
    pub soloed: bool,
    /// Aux sends to downstream buses.
    pub sends: Vec<BusSendEntry>,
    /// Internal mix accumulation buffer (interleaved channels × samples).
    buffer: Vec<f32>,
}

impl MixBus {
    /// Create a new, silent bus.
    ///
    /// The internal buffer has zero length until [`add_samples`](MixBus::add_samples)
    /// is first called; it is reallocated as needed.
    #[must_use]
    pub fn new(id: &str, name: &str, channels: usize) -> Self {
        let ch = channels.max(1);
        Self {
            id: id.to_string(),
            name: name.to_string(),
            channels: ch,
            fader_db: 0.0,
            muted: false,
            soloed: false,
            sends: Vec::new(),
            buffer: Vec::new(),
        }
    }

    /// Accumulate `samples` into the internal buffer.
    ///
    /// If `samples` is longer than the current buffer it is extended with zeros
    /// then the new samples are added.  If it is shorter, only the available
    /// positions are touched.
    pub fn add_samples(&mut self, samples: &[f32]) {
        if self.buffer.len() < samples.len() {
            self.buffer.resize(samples.len(), 0.0);
        }
        for (dst, &src) in self.buffer.iter_mut().zip(samples.iter()) {
            *dst += src;
        }
    }

    /// Apply the master fader to a copy of the internal buffer and return it.
    ///
    /// If the bus is muted the returned buffer is all zeros regardless of
    /// fader settings.
    #[must_use]
    pub fn apply_fader(&self) -> Vec<f32> {
        if self.muted {
            return vec![0.0_f32; self.buffer.len()];
        }
        let gain = db_to_linear(self.fader_db);
        self.buffer.iter().map(|&s| s * gain).collect()
    }

    /// Clear the internal accumulation buffer without deallocating memory.
    pub fn clear_buffer(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
    }

    /// Read the raw accumulation buffer (before fader).
    #[must_use]
    pub fn buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Add an aux send.
    pub fn add_send(&mut self, send: BusSendEntry) {
        self.sends.push(send);
    }

    /// Remove all sends to a given destination bus.
    pub fn remove_send(&mut self, destination_bus: &str) {
        self.sends.retain(|s| s.destination_bus != destination_bus);
    }
}

// ────────────────────────────────────────────────────────── MixMatrix ───

/// Complete N-bus routing matrix with a single master output.
///
/// # Routing graph
///
/// Routings are directed edges `(src_bus, dst_bus, gain_linear)`.  The graph
/// must be a DAG; [`route`](MixMatrix::route) checks for newly created cycles
/// before committing.
///
/// # Render pass
///
/// [`render`](MixMatrix::render) performs a full mix in topological order:
///
/// 1. Clear all bus buffers.
/// 2. Add each input to its assigned bus.
/// 3. Propagate fader-applied bus output through the routing edges.
/// 4. Return the master bus output.
#[derive(Debug)]
pub struct MixMatrix {
    buses: HashMap<String, MixBus>,
    /// Routing edges: `(src_bus_id, dst_bus_id, gain_linear)`.
    routing: Vec<(String, String, f32)>,
    master_bus: String,
    /// Sample rate stored for future time-based features.
    #[allow(dead_code)]
    sample_rate: u32,
}

impl MixMatrix {
    /// Create a new matrix with a stereo `"Master"` bus already present.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let master = MixBus::new("master", "Master", 2);
        let mut buses = HashMap::new();
        buses.insert("master".to_string(), master);
        Self {
            buses,
            routing: Vec::new(),
            master_bus: "master".to_string(),
            sample_rate,
        }
    }

    /// Add a bus to the matrix.
    ///
    /// If a bus with the same ID already exists it is replaced.
    pub fn add_bus(&mut self, bus: MixBus) {
        self.buses.insert(bus.id.clone(), bus);
    }

    /// Remove a bus by ID.
    ///
    /// Returns `true` if the bus existed and was removed.
    /// All routing edges that reference this bus are also removed.
    /// The master bus cannot be removed; attempting to do so returns `false`.
    pub fn remove_bus(&mut self, id: &str) -> bool {
        if id == self.master_bus {
            return false;
        }
        let existed = self.buses.remove(id).is_some();
        if existed {
            self.routing.retain(|(src, dst, _)| src != id && dst != id);
        }
        existed
    }

    /// Add a routing edge from `src` to `dst` with gain `gain_db`.
    ///
    /// Returns an error if either bus does not exist or the edge would create a
    /// cycle in the routing graph.  Duplicate edges (same src + dst) are
    /// updated rather than duplicated.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` when the buses don't exist or a cycle is detected.
    pub fn route(&mut self, src: &str, dst: &str, gain_db: f32) -> Result<(), String> {
        if !self.buses.contains_key(src) {
            return Err(format!("Source bus not found: {src}"));
        }
        if !self.buses.contains_key(dst) {
            return Err(format!("Destination bus not found: {dst}"));
        }

        // Update or insert edge
        if let Some(edge) = self
            .routing
            .iter_mut()
            .find(|(s, d, _)| s == src && d == dst)
        {
            edge.2 = db_to_linear(gain_db);
        } else {
            self.routing
                .push((src.to_string(), dst.to_string(), db_to_linear(gain_db)));
        }

        if self.detect_cycle() {
            // Roll back
            self.routing.retain(|(s, d, _)| !(s == src && d == dst));
            return Err(format!("Routing {src} → {dst} would create a cycle"));
        }

        Ok(())
    }

    /// Remove any routing edge between `src` and `dst`.
    pub fn unroute(&mut self, src: &str, dst: &str) {
        self.routing.retain(|(s, d, _)| !(s == src && d == dst));
    }

    /// Perform a complete mix render.
    ///
    /// `inputs` is a slice of `(bus_id, samples)` pairs.  Each input's samples
    /// are accumulated into the named bus.  After all inputs are placed, bus
    /// outputs are propagated through the routing graph in topological order.
    /// The master bus output (post-fader) is returned.
    #[must_use]
    pub fn render(&mut self, inputs: &[(String, Vec<f32>)]) -> Vec<f32> {
        // 1. Clear all bus buffers
        for bus in self.buses.values_mut() {
            bus.clear_buffer();
        }

        // 2. Add input samples to their target buses
        for (bus_id, samples) in inputs {
            if let Some(bus) = self.buses.get_mut(bus_id.as_str()) {
                bus.add_samples(samples);
            }
        }

        // 3. Propagate in topological order
        //    Build adjacency for the topological sort on the fly.
        let order = self.topological_order();

        // We need to iterate in topo order but also mutate buses.
        // Collect fader-applied output first for each bus, then route.
        // Two-phase approach: snapshot outputs → apply to destinations.
        for src_id in &order {
            // Collect outgoing edges for this source
            let edges: Vec<(String, f32)> = self
                .routing
                .iter()
                .filter(|(s, _, _)| s == src_id)
                .map(|(_, d, g)| (d.clone(), *g))
                .collect();

            if edges.is_empty() {
                continue;
            }

            // Compute the output of the source bus
            let src_output = match self.buses.get(src_id.as_str()) {
                Some(bus) => bus.apply_fader(),
                None => continue,
            };

            // Route to each destination
            for (dst_id, gain) in edges {
                let routed: Vec<f32> = src_output.iter().map(|&s| s * gain).collect();
                if let Some(dst_bus) = self.buses.get_mut(dst_id.as_str()) {
                    dst_bus.add_samples(&routed);
                }
            }
        }

        // 4. Return master bus output
        match self.buses.get(self.master_bus.as_str()) {
            Some(bus) => bus.apply_fader(),
            None => Vec::new(),
        }
    }

    /// Set the fader level of a bus.
    ///
    /// Returns `true` if the bus was found.
    pub fn set_fader(&mut self, bus_id: &str, db: f32) -> bool {
        match self.buses.get_mut(bus_id) {
            Some(bus) => {
                bus.fader_db = db;
                true
            }
            None => false,
        }
    }

    /// Set the mute state of a bus.
    ///
    /// Returns `true` if the bus was found.
    pub fn set_mute(&mut self, bus_id: &str, muted: bool) -> bool {
        match self.buses.get_mut(bus_id) {
            Some(bus) => {
                bus.muted = muted;
                true
            }
            None => false,
        }
    }

    /// Solo a bus — all other buses are muted; the soloed bus is unmuted.
    ///
    /// Calling with `soloed = false` clears all solo/mute flags across all
    /// buses so the mix returns to normal.
    pub fn set_solo(&mut self, bus_id: &str, soloed: bool) {
        if soloed {
            // Mark the target soloed, mute everything else
            for (id, bus) in &mut self.buses {
                if id == bus_id {
                    bus.soloed = true;
                    bus.muted = false;
                } else {
                    bus.soloed = false;
                    bus.muted = true;
                }
            }
        } else {
            // Clear all solo/mute flags
            for bus in self.buses.values_mut() {
                bus.soloed = false;
                bus.muted = false;
            }
        }
    }

    /// Detect a cycle in the routing graph using iterative DFS.
    ///
    /// Returns `true` if the graph contains at least one cycle.
    #[must_use]
    pub fn detect_cycle(&self) -> bool {
        // Build adjacency list
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for (src, dst, _) in &self.routing {
            adj.entry(src.as_str()).or_default().push(dst.as_str());
        }

        // States: 0 = unvisited, 1 = in-stack, 2 = done
        let mut state: HashMap<&str, u8> = HashMap::new();

        let nodes: Vec<&str> = self.buses.keys().map(String::as_str).collect();

        for start in nodes {
            if state.get(start).copied().unwrap_or(0) == 0 {
                // Iterative DFS with explicit stack
                // Stack entries: (node, iterator-index-into-neighbours)
                let mut stack: Vec<(&str, usize)> = vec![(start, 0)];
                state.insert(start, 1);

                while let Some((node, idx)) = stack.last_mut() {
                    let node = *node;
                    let neighbours = adj.get(node).map(Vec::as_slice).unwrap_or(&[]);
                    if *idx < neighbours.len() {
                        let neighbour = neighbours[*idx];
                        *idx += 1;
                        match state.get(neighbour).copied().unwrap_or(0) {
                            1 => return true, // back edge → cycle
                            0 => {
                                state.insert(neighbour, 1);
                                stack.push((neighbour, 0));
                            }
                            _ => {} // already done
                        }
                    } else {
                        state.insert(node, 2);
                        stack.pop();
                    }
                }
            }
        }

        false
    }

    /// Get a bus by ID.
    #[must_use]
    pub fn get_bus(&self, id: &str) -> Option<&MixBus> {
        self.buses.get(id)
    }

    /// Get a mutable bus by ID.
    #[must_use]
    pub fn get_bus_mut(&mut self, id: &str) -> Option<&mut MixBus> {
        self.buses.get_mut(id)
    }

    /// Get all buses.
    #[must_use]
    pub fn buses(&self) -> &HashMap<String, MixBus> {
        &self.buses
    }

    /// Get routing edges.
    #[must_use]
    pub fn routing(&self) -> &[(String, String, f32)] {
        &self.routing
    }

    /// Get master bus ID.
    #[must_use]
    pub fn master_bus_id(&self) -> &str {
        &self.master_bus
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Return bus IDs in topological order (sources before sinks).
    ///
    /// Uses Kahn's algorithm on the routing graph.  Any bus not present in the
    /// routing graph is included at the front (no in-edges).
    fn topological_order(&self) -> Vec<String> {
        // Compute in-degree for each bus
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for id in self.buses.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
        }
        for (_, dst, _) in &self.routing {
            *in_degree.entry(dst.as_str()).or_insert(0) += 1;
        }

        // Kahn's BFS
        let mut queue: std::collections::VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order: Vec<String> = Vec::with_capacity(self.buses.len());

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            for (src, dst, _) in &self.routing {
                if src == node {
                    let deg = in_degree.entry(dst.as_str()).or_insert(0);
                    if *deg > 0 {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dst.as_str());
                        }
                    }
                }
            }
        }

        // Append any buses that weren't in the routing graph
        for id in self.buses.keys() {
            if !order.contains(id) {
                order.push(id.clone());
            }
        }

        order
    }
}

// ──────────────────────────────────────────────────────────── utilities ───

/// Convert dB to linear gain.  Values ≤ −144 dB are treated as silence (0.0).
#[must_use]
pub fn db_to_linear(db: f32) -> f32 {
    if db <= -144.0 {
        0.0
    } else {
        10.0_f32.powf(db / 20.0)
    }
}

/// Convert linear gain to dB.  Zero returns `f32::NEG_INFINITY`.
#[must_use]
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

// ──────────────────────────────────────────────────────────────── tests ───

#[cfg(test)]
mod tests {
    use super::*;

    // ── MixBus ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mix_bus_new() {
        let bus = MixBus::new("drums", "Drums", 2);
        assert_eq!(bus.id, "drums");
        assert_eq!(bus.name, "Drums");
        assert_eq!(bus.channels, 2);
        assert!((bus.fader_db).abs() < f32::EPSILON);
        assert!(!bus.muted);
        assert!(!bus.soloed);
    }

    #[test]
    fn test_mix_bus_add_samples() {
        let mut bus = MixBus::new("b", "Bus", 1);
        let samples = vec![0.5_f32; 4];
        bus.add_samples(&samples);
        assert_eq!(bus.buffer().len(), 4);
        // add again – values should double
        bus.add_samples(&samples);
        assert!((bus.buffer()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mix_bus_apply_fader_0db() {
        let mut bus = MixBus::new("b", "Bus", 1);
        bus.add_samples(&[1.0, 1.0]);
        let out = bus.apply_fader();
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mix_bus_apply_fader_minus6db() {
        let mut bus = MixBus::new("b", "Bus", 1);
        bus.fader_db = -6.0206; // ≈ half amplitude
        bus.add_samples(&[1.0]);
        let out = bus.apply_fader();
        assert!((out[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mix_bus_muted() {
        let mut bus = MixBus::new("b", "Bus", 1);
        bus.muted = true;
        bus.add_samples(&[1.0, 1.0]);
        let out = bus.apply_fader();
        assert!(out.iter().all(|&s| s.abs() < f32::EPSILON));
    }

    #[test]
    fn test_mix_bus_clear_buffer() {
        let mut bus = MixBus::new("b", "Bus", 1);
        bus.add_samples(&[0.8, 0.9]);
        bus.clear_buffer();
        assert!(bus.buffer().iter().all(|&s| s.abs() < f32::EPSILON));
    }

    // ── MixMatrix ────────────────────────────────────────────────────────────

    #[test]
    fn test_mix_matrix_new_has_master() {
        let mx = MixMatrix::new(48_000);
        assert!(mx.get_bus("master").is_some());
        assert_eq!(mx.master_bus_id(), "master");
    }

    #[test]
    fn test_mix_matrix_add_remove_bus() {
        let mut mx = MixMatrix::new(48_000);
        mx.add_bus(MixBus::new("group1", "Group 1", 2));
        assert!(mx.get_bus("group1").is_some());
        assert!(mx.remove_bus("group1"));
        assert!(mx.get_bus("group1").is_none());
    }

    #[test]
    fn test_mix_matrix_cannot_remove_master() {
        let mut mx = MixMatrix::new(48_000);
        assert!(!mx.remove_bus("master"));
    }

    #[test]
    fn test_mix_matrix_route_and_render() {
        let mut mx = MixMatrix::new(48_000);
        mx.add_bus(MixBus::new("kick", "Kick", 1));
        mx.route("kick", "master", 0.0)
            .expect("routing should succeed");

        let inputs = vec![("kick".to_string(), vec![0.5_f32; 4])];
        let out = mx.render(&inputs);
        // kick → master at 0 dB, master at 0 dB → output ≈ 0.5
        assert!(!out.is_empty());
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_mix_matrix_detect_cycle() {
        let mut mx = MixMatrix::new(48_000);
        mx.add_bus(MixBus::new("a", "A", 1));
        mx.add_bus(MixBus::new("b", "B", 1));
        mx.route("a", "b", 0.0).expect("no cycle yet");
        // a → b already; adding b → a should fail (cycle)
        let result = mx.route("b", "a", 0.0);
        assert!(result.is_err());
        // Routing graph must still be acyclic
        assert!(!mx.detect_cycle());
    }

    #[test]
    fn test_mix_matrix_set_fader() {
        let mut mx = MixMatrix::new(48_000);
        assert!(mx.set_fader("master", -6.0));
        assert!(
            (mx.get_bus("master").expect("master exists").fader_db - (-6.0)).abs() < f32::EPSILON
        );
        // Non-existent bus
        assert!(!mx.set_fader("nonexistent", 0.0));
    }

    #[test]
    fn test_mix_matrix_set_mute() {
        let mut mx = MixMatrix::new(48_000);
        assert!(mx.set_mute("master", true));
        assert!(mx.get_bus("master").expect("master exists").muted);
    }

    #[test]
    fn test_mix_matrix_set_solo() {
        let mut mx = MixMatrix::new(48_000);
        mx.add_bus(MixBus::new("ch1", "Channel 1", 2));
        mx.add_bus(MixBus::new("ch2", "Channel 2", 2));
        mx.set_solo("ch1", true);
        assert!(!mx.get_bus("ch1").expect("ch1 exists").muted);
        assert!(mx.get_bus("ch1").expect("ch1 exists").soloed);
        assert!(mx.get_bus("ch2").expect("ch2 exists").muted);
        // Clear solo
        mx.set_solo("ch1", false);
        assert!(!mx.get_bus("ch1").expect("ch1 exists").muted);
        assert!(!mx.get_bus("ch2").expect("ch2 exists").muted);
    }

    #[test]
    fn test_db_to_linear_and_back() {
        let db = -12.0_f32;
        let lin = db_to_linear(db);
        let db2 = linear_to_db(lin);
        assert!((db - db2).abs() < 0.001);
    }

    #[test]
    fn test_db_to_linear_silence() {
        assert_eq!(db_to_linear(-200.0), 0.0);
    }

    #[test]
    fn test_linear_to_db_zero() {
        assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
    }
}
