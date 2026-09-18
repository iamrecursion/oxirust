//! Distributed cache coordination primitives.
//!
//! This module provides the building blocks for routing cache operations across
//! a cluster of nodes:
//!
//! - [`NodeId`] — a lightweight opaque node identifier.
//! - [`ConsistentHash`] — a virtual-node consistent-hash ring for stable key
//!   routing as nodes join and leave.
//! - [`DistributedCacheClient`] — per-node view with a routing helper.
//! - [`ReplicationFactor`] — quorum read/write logic.
//! - [`CacheCoordinator`] — cluster-level coordinator that ties it all
//!   together.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

// ── FNV-1a (same scheme as bloom_filter but local to avoid cross-module dep) ──

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325u64;
const FNV_PRIME: u64 = 0x00000100000001b3u64;

fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── NodeId ────────────────────────────────────────────────────────────────────

/// Opaque identifier for a cache cluster node.
///
/// Implements `Copy`, `Eq`, `Hash`, and `Display` so it can be used both as a
/// map key and in format strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node:{}", self.0)
    }
}

// ── ConsistentHash ────────────────────────────────────────────────────────────

/// Virtual-node consistent-hash ring.
///
/// Each real node is mapped to `virtual_nodes_per_node` positions on a
/// `u64` hash ring using FNV-1a hashes of the string `"<node_id>_<i>"`.
/// Key routing finds the first virtual-node position ≥ `hash(key)` on the
/// ring (wrapping around).
#[derive(Debug, Clone)]
pub struct ConsistentHash {
    /// `ring_position → NodeId` sorted map representing the virtual-node ring.
    ring: BTreeMap<u64, NodeId>,
    /// How many virtual nodes each real node occupies.
    virtual_nodes_per_node: u32,
}

impl ConsistentHash {
    /// Create an empty ring with the given number of virtual nodes per real node.
    pub fn new(virtual_nodes: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes_per_node: virtual_nodes.max(1),
        }
    }

    /// Add `node_id` to the ring by inserting `virtual_nodes_per_node` hash
    /// positions derived from `"<node_id>_<i>"` for `i` in `0..virtual_nodes`.
    pub fn add_node(&mut self, node_id: NodeId) {
        for i in 0..self.virtual_nodes_per_node {
            let label = format!("{node_id}_{i}");
            let pos = fnv1a_64(label.as_bytes());
            self.ring.insert(pos, node_id);
        }
    }

    /// Remove all virtual nodes associated with `node_id`.
    pub fn remove_node(&mut self, node_id: NodeId) {
        // Collect positions to remove first to avoid borrow conflicts.
        let to_remove: Vec<u64> = self
            .ring
            .iter()
            .filter_map(|(&pos, &nid)| if nid == node_id { Some(pos) } else { None })
            .collect();
        for pos in to_remove {
            self.ring.remove(&pos);
        }
    }

    /// Route `key` to the first node whose ring position is ≥ `hash(key)`,
    /// wrapping around to the lowest position if needed.
    ///
    /// Returns `None` when the ring is empty.
    pub fn get_node(&self, key: &[u8]) -> Option<NodeId> {
        if self.ring.is_empty() {
            return None;
        }
        let pos = fnv1a_64(key);
        // Try to find the first entry ≥ pos (successor).
        self.ring
            .range(pos..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, &nid)| nid)
    }

    /// Return up to `n` distinct successor nodes for `key` (for replication).
    ///
    /// Starts at the primary successor and walks the ring clockwise, collecting
    /// distinct `NodeId`s until `n` unique nodes are found or the ring is
    /// exhausted.
    pub fn get_n_nodes(&self, key: &[u8], n: usize) -> Vec<NodeId> {
        if self.ring.is_empty() || n == 0 {
            return Vec::new();
        }
        let pos = fnv1a_64(key);

        // Build an iterator that walks the ring starting at `pos`, wrapping.
        let after = self.ring.range(pos..).map(|(_, nid)| *nid);
        let before = self.ring.range(..pos).map(|(_, nid)| *nid);
        let full_circle = after.chain(before);

        let mut seen: Vec<NodeId> = Vec::with_capacity(n);
        for node in full_circle {
            if !seen.contains(&node) {
                seen.push(node);
                if seen.len() == n {
                    break;
                }
            }
        }
        seen
    }

    /// Return the number of virtual nodes currently in the ring.
    pub fn virtual_node_count(&self) -> usize {
        self.ring.len()
    }

    /// Return the number of distinct real nodes in the ring.
    pub fn real_node_count(&self) -> usize {
        let mut nodes: Vec<NodeId> = self.ring.values().copied().collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len()
    }
}

// ── DistributedCacheClient ────────────────────────────────────────────────────

/// Per-node client that wraps a [`ConsistentHash`] ring and provides key
/// routing from the perspective of `local_node`.
#[derive(Debug, Clone)]
pub struct DistributedCacheClient {
    /// The node this client represents.
    pub local_node: NodeId,
    /// Shared ring (each client holds its own copy for isolation in this
    /// in-process model; in a real system this would be a shared reference).
    pub ring: ConsistentHash,
}

impl DistributedCacheClient {
    /// Create a new client for `local_node` with the given ring.
    pub fn new(local_node: NodeId, ring: ConsistentHash) -> Self {
        Self { local_node, ring }
    }

    /// Determine which node should own `key`.
    ///
    /// Returns `local_node` when the ring is empty (degenerate single-node
    /// mode).
    pub fn route_key(&self, key: &[u8]) -> NodeId {
        self.ring.get_node(key).unwrap_or(self.local_node)
    }

    /// Return `true` if `key` maps to `local_node` (i.e. this node is the
    /// primary owner).
    pub fn is_local_key(&self, key: &[u8]) -> bool {
        self.route_key(key) == self.local_node
    }
}

// ── ReplicationFactor ─────────────────────────────────────────────────────────

/// Quorum-based replication configuration.
///
/// A write quorum requires acknowledgement from at least `writes` nodes; a
/// read quorum requires responses from at least `reads` nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicationFactor {
    /// Number of read responses required to satisfy a quorum.
    pub reads: u8,
    /// Number of write acknowledgements required to satisfy a quorum.
    pub writes: u8,
}

impl ReplicationFactor {
    /// Construct a new `ReplicationFactor`.
    pub fn new(reads: u8, writes: u8) -> Self {
        Self { reads, writes }
    }

    /// Return `true` if `responses` meets or exceeds the read quorum.
    pub fn is_quorum_read_met(&self, responses: u8) -> bool {
        responses >= self.reads
    }

    /// Return `true` if `responses` meets or exceeds the write quorum.
    pub fn is_quorum_write_met(&self, responses: u8) -> bool {
        responses >= self.writes
    }

    /// Convenience constructor for a standard RF-3 cluster (R=2, W=2).
    pub fn rf3() -> Self {
        Self {
            reads: 2,
            writes: 2,
        }
    }

    /// Convenience constructor for a strongly consistent RF-3 cluster
    /// (R+W > N, so R=3, W=3 for N=3).
    pub fn rf3_strong() -> Self {
        Self {
            reads: 3,
            writes: 3,
        }
    }
}

impl Default for ReplicationFactor {
    fn default() -> Self {
        Self::rf3()
    }
}

// ── CacheCoordinator ──────────────────────────────────────────────────────────

/// Cluster-level coordinator.
///
/// Tracks all node clients and the replication policy for the cluster.  In a
/// real distributed system the coordinator would issue RPCs; here it simulates
/// the routing and quorum decisions.
#[derive(Debug)]
pub struct CacheCoordinator {
    /// Map from `NodeId` to per-node client.
    pub clients: HashMap<NodeId, DistributedCacheClient>,
    /// Cluster-wide replication factor.
    pub replication: ReplicationFactor,
}

impl CacheCoordinator {
    /// Create a new `CacheCoordinator` with the given replication factor.
    pub fn new(replication: ReplicationFactor) -> Self {
        Self {
            clients: HashMap::new(),
            replication,
        }
    }

    /// Register a `DistributedCacheClient` for its `local_node`.
    pub fn add_client(&mut self, client: DistributedCacheClient) {
        self.clients.insert(client.local_node, client);
    }

    /// Remove the client (and node) identified by `node_id`.
    pub fn remove_client(&mut self, node_id: NodeId) {
        self.clients.remove(&node_id);
    }

    /// Determine the primary owner of `key` according to the first registered
    /// client's ring.
    ///
    /// Returns `None` when no clients are registered.
    pub fn primary_node_for(&self, key: &[u8]) -> Option<NodeId> {
        self.clients.values().next().map(|c| c.route_key(key))
    }

    /// Return up to `n` replica nodes for `key` according to the first
    /// registered client's ring.
    pub fn replica_nodes_for(&self, key: &[u8], n: usize) -> Vec<NodeId> {
        self.clients
            .values()
            .next()
            .map(|c| c.ring.get_n_nodes(key, n))
            .unwrap_or_default()
    }

    /// Simulate a write operation: determine the owner nodes for `key` and
    /// check whether a quorum can be formed from `available_nodes`.
    pub fn can_write_quorum(&self, key: &[u8], available_nodes: &[NodeId]) -> bool {
        let replicas = self.replica_nodes_for(key, self.replication.writes as usize);
        let ack_count = replicas
            .iter()
            .filter(|nid| available_nodes.contains(nid))
            .count() as u8;
        self.replication.is_quorum_write_met(ack_count)
    }

    /// Simulate a read operation: determine the owner nodes for `key` and
    /// check whether a quorum can be formed from `available_nodes`.
    pub fn can_read_quorum(&self, key: &[u8], available_nodes: &[NodeId]) -> bool {
        let replicas = self.replica_nodes_for(key, self.replication.reads as usize);
        let response_count = replicas
            .iter()
            .filter(|nid| available_nodes.contains(nid))
            .count() as u8;
        self.replication.is_quorum_read_met(response_count)
    }

    /// Return the number of registered clients/nodes.
    pub fn node_count(&self) -> usize {
        self.clients.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ring_with_nodes(vn: u32, ids: &[u64]) -> ConsistentHash {
        let mut ring = ConsistentHash::new(vn);
        for &id in ids {
            ring.add_node(NodeId(id));
        }
        ring
    }

    // 1. NodeId display
    #[test]
    fn test_node_id_display() {
        let nid = NodeId(42);
        assert_eq!(format!("{nid}"), "node:42");
    }

    // 2. Empty ring returns None
    #[test]
    fn test_empty_ring_get_node() {
        let ring = ConsistentHash::new(10);
        assert!(ring.get_node(b"any_key").is_none());
    }

    // 3. Single node always wins
    #[test]
    fn test_single_node_routing() {
        let ring = make_ring_with_nodes(20, &[1]);
        for key in [b"a".as_ref(), b"hello", b"oximedia"] {
            assert_eq!(ring.get_node(key), Some(NodeId(1)));
        }
    }

    // 4. Adding two nodes splits the keyspace
    #[test]
    fn test_two_nodes_split_keyspace() {
        let ring = make_ring_with_nodes(150, &[1, 2]);
        let mut counts = [0usize; 2];
        for i in 0u32..1000 {
            let key = i.to_le_bytes();
            match ring.get_node(&key) {
                Some(NodeId(1)) => counts[0] += 1,
                Some(NodeId(2)) => counts[1] += 1,
                _ => {}
            }
        }
        // Each node should get a non-trivial fraction.
        assert!(counts[0] > 100, "node 1 got too few keys: {}", counts[0]);
        assert!(counts[1] > 100, "node 2 got too few keys: {}", counts[1]);
    }

    // 5. virtual_node_count matches expected
    #[test]
    fn test_virtual_node_count() {
        let ring = make_ring_with_nodes(50, &[1, 2, 3]);
        assert_eq!(ring.virtual_node_count(), 150);
    }

    // 6. real_node_count
    #[test]
    fn test_real_node_count() {
        let ring = make_ring_with_nodes(20, &[10, 20, 30, 40]);
        assert_eq!(ring.real_node_count(), 4);
    }

    // 7. remove_node shrinks the ring
    #[test]
    fn test_remove_node() {
        let mut ring = make_ring_with_nodes(10, &[1, 2]);
        ring.remove_node(NodeId(1));
        assert_eq!(ring.real_node_count(), 1);
        assert_eq!(ring.virtual_node_count(), 10);
        for i in 0u32..50 {
            assert_eq!(ring.get_node(&i.to_le_bytes()), Some(NodeId(2)));
        }
    }

    // 8. Stability: re-adding the same node does not double-add
    #[test]
    fn test_add_node_twice_does_not_double_positions() {
        let mut ring = ConsistentHash::new(10);
        ring.add_node(NodeId(7));
        ring.add_node(NodeId(7)); // second add should overwrite same positions
                                  // BTreeMap deduplicates by position key, so count <= 10.
        assert!(ring.virtual_node_count() <= 10);
    }

    // 9. get_n_nodes returns distinct nodes
    #[test]
    fn test_get_n_nodes_distinct() {
        let ring = make_ring_with_nodes(100, &[1, 2, 3]);
        let nodes = ring.get_n_nodes(b"replicated_key", 3);
        assert_eq!(nodes.len(), 3);
        let unique: std::collections::HashSet<_> = nodes.iter().cloned().collect();
        assert_eq!(unique.len(), 3);
    }

    // 10. get_n_nodes with n > real nodes returns all real nodes
    #[test]
    fn test_get_n_nodes_exceeds_real_count() {
        let ring = make_ring_with_nodes(50, &[1, 2]);
        let nodes = ring.get_n_nodes(b"key", 10);
        // Only 2 real nodes exist.
        assert_eq!(nodes.len(), 2);
    }

    // 11. get_n_nodes with n=0 returns empty
    #[test]
    fn test_get_n_nodes_zero() {
        let ring = make_ring_with_nodes(50, &[1, 2, 3]);
        assert!(ring.get_n_nodes(b"key", 0).is_empty());
    }

    // 12. get_n_nodes on empty ring returns empty
    #[test]
    fn test_get_n_nodes_empty_ring() {
        let ring = ConsistentHash::new(10);
        assert!(ring.get_n_nodes(b"key", 3).is_empty());
    }

    // 13. Consistent routing: same key always maps to same node
    #[test]
    fn test_consistent_routing() {
        let ring = make_ring_with_nodes(100, &[1, 2, 3, 4, 5]);
        for key in [b"video_001".as_ref(), b"audio_002", b"manifest"] {
            let first = ring.get_node(key);
            for _ in 0..10 {
                assert_eq!(ring.get_node(key), first, "routing is not deterministic");
            }
        }
    }

    // 14. DistributedCacheClient::route_key
    #[test]
    fn test_distributed_cache_client_route() {
        let ring = make_ring_with_nodes(100, &[1, 2, 3]);
        let client = DistributedCacheClient::new(NodeId(1), ring);
        // Must return a valid node, not panic.
        let routed = client.route_key(b"some_key");
        assert!(routed.0 >= 1 && routed.0 <= 3);
    }

    // 15. is_local_key when ring has only local node
    #[test]
    fn test_is_local_key_single_node() {
        let mut ring = ConsistentHash::new(50);
        ring.add_node(NodeId(99));
        let client = DistributedCacheClient::new(NodeId(99), ring);
        assert!(client.is_local_key(b"anything"));
    }

    // 16. ReplicationFactor quorum read
    #[test]
    fn test_replication_factor_read_quorum() {
        let rf = ReplicationFactor::new(2, 2);
        assert!(!rf.is_quorum_read_met(1));
        assert!(rf.is_quorum_read_met(2));
        assert!(rf.is_quorum_read_met(3));
    }

    // 17. ReplicationFactor quorum write
    #[test]
    fn test_replication_factor_write_quorum() {
        let rf = ReplicationFactor::new(2, 3);
        assert!(!rf.is_quorum_write_met(2));
        assert!(rf.is_quorum_write_met(3));
    }

    // 18. rf3 default quorum
    #[test]
    fn test_rf3_defaults() {
        let rf = ReplicationFactor::rf3();
        assert_eq!(rf.reads, 2);
        assert_eq!(rf.writes, 2);
    }

    // 19. CacheCoordinator add / remove clients
    #[test]
    fn test_cache_coordinator_node_count() {
        let mut coord = CacheCoordinator::new(ReplicationFactor::rf3());
        let ring = make_ring_with_nodes(50, &[1, 2, 3]);
        for id in 1..=3u64 {
            coord.add_client(DistributedCacheClient::new(NodeId(id), ring.clone()));
        }
        assert_eq!(coord.node_count(), 3);
        coord.remove_client(NodeId(2));
        assert_eq!(coord.node_count(), 2);
    }

    // 20. CacheCoordinator can_write_quorum all available
    #[test]
    fn test_can_write_quorum_all_nodes_up() {
        let ring = make_ring_with_nodes(100, &[1, 2, 3]);
        let mut coord = CacheCoordinator::new(ReplicationFactor::new(2, 2));
        for id in 1..=3u64 {
            coord.add_client(DistributedCacheClient::new(NodeId(id), ring.clone()));
        }
        let all_nodes = vec![NodeId(1), NodeId(2), NodeId(3)];
        assert!(coord.can_write_quorum(b"key", &all_nodes));
    }

    // 21. CacheCoordinator can_write_quorum insufficient nodes
    #[test]
    fn test_can_write_quorum_insufficient() {
        let ring = make_ring_with_nodes(100, &[1, 2, 3]);
        let mut coord = CacheCoordinator::new(ReplicationFactor::new(2, 3));
        for id in 1..=3u64 {
            coord.add_client(DistributedCacheClient::new(NodeId(id), ring.clone()));
        }
        // Only one node available.
        let partial = vec![NodeId(1)];
        assert!(!coord.can_write_quorum(b"key", &partial));
    }

    // 22. primary_node_for returns Some when nodes are registered
    #[test]
    fn test_primary_node_for() {
        let ring = make_ring_with_nodes(100, &[5, 6, 7]);
        let mut coord = CacheCoordinator::new(ReplicationFactor::default());
        coord.add_client(DistributedCacheClient::new(NodeId(5), ring));
        let primary = coord.primary_node_for(b"video_segment");
        assert!(primary.is_some());
    }

    // 23. primary_node_for returns None when no clients
    #[test]
    fn test_primary_node_for_empty() {
        let coord = CacheCoordinator::new(ReplicationFactor::default());
        assert!(coord.primary_node_for(b"key").is_none());
    }

    // 24. Adding many nodes preserves routing consistency after removals
    #[test]
    fn test_routing_consistency_after_removal() {
        let mut ring = make_ring_with_nodes(100, &[1, 2, 3, 4, 5]);
        let key = b"stable_key";
        let before = ring.get_node(key);
        ring.remove_node(NodeId(99)); // remove a node that was never added
        let after = ring.get_node(key);
        assert_eq!(before, after, "routing changed when removing absent node");
    }

    // 25. Keyspace distribution is roughly uniform across 3 nodes
    #[test]
    fn test_uniform_distribution_three_nodes() {
        let ring = make_ring_with_nodes(200, &[1, 2, 3]);
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for i in 0u32..3000 {
            let key = format!("key_{i}");
            if let Some(nid) = ring.get_node(key.as_bytes()) {
                *counts.entry(nid.0).or_insert(0) += 1;
            }
        }
        // Each node should receive between 20% and 80% of keys.
        for (node, count) in &counts {
            assert!(
                *count > 300 && *count < 2400,
                "node {node} has unbalanced load: {count} / 3000"
            );
        }
    }
}
