#![allow(dead_code)]

//! Link aggregation for redundant media routing.
//!
//! Groups multiple physical links into a single logical link for
//! increased bandwidth and fault tolerance. Supports round-robin,
//! active-backup, and weighted load-balancing modes.

/// Mode of link aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMode {
    /// Distribute traffic evenly across all active links (round-robin).
    RoundRobin,
    /// Use one active link; failover to the next on failure.
    ActiveBackup,
    /// Distribute traffic proportionally to link weights.
    Weighted,
}

/// State of a physical link in the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    /// Link is operational.
    Up,
    /// Link is down or unreachable.
    Down,
    /// Link is in standby (only used for backup).
    Standby,
}

/// A single physical link within an aggregation group.
#[derive(Debug, Clone)]
pub struct PhysicalLink {
    /// Link identifier.
    pub id: String,
    /// Current state.
    pub state: LinkState,
    /// Capacity in megabits per second.
    pub capacity_mbps: u64,
    /// Weight for weighted balancing (higher = more traffic).
    pub weight: u32,
    /// Packets sent through this link.
    pub packets_sent: u64,
}

impl PhysicalLink {
    /// Create a new link.
    pub fn new(id: impl Into<String>, capacity_mbps: u64) -> Self {
        Self {
            id: id.into(),
            state: LinkState::Up,
            capacity_mbps,
            weight: 1,
            packets_sent: 0,
        }
    }

    /// Set the weight.
    pub fn with_weight(mut self, w: u32) -> Self {
        self.weight = w;
        self
    }

    /// Whether the link can carry traffic.
    pub fn is_active(&self) -> bool {
        self.state == LinkState::Up
    }
}

/// A logical group of aggregated links.
#[derive(Debug, Clone)]
pub struct LinkGroup {
    /// Group name.
    pub name: String,
    /// Aggregation mode.
    pub mode: AggregationMode,
    /// Physical links in the group.
    links: Vec<PhysicalLink>,
    /// Round-robin index.
    rr_index: usize,
}

impl LinkGroup {
    /// Create a new link group.
    pub fn new(name: impl Into<String>, mode: AggregationMode) -> Self {
        Self {
            name: name.into(),
            mode,
            links: Vec::new(),
            rr_index: 0,
        }
    }

    /// Add a link to the group.
    pub fn add_link(&mut self, link: PhysicalLink) {
        self.links.push(link);
    }

    /// Number of links.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Number of active (Up) links.
    pub fn active_count(&self) -> usize {
        self.links.iter().filter(|l| l.is_active()).count()
    }

    /// Total capacity of active links in Mbps.
    pub fn aggregate_capacity_mbps(&self) -> u64 {
        self.links
            .iter()
            .filter(|l| l.is_active())
            .map(|l| l.capacity_mbps)
            .sum()
    }

    /// Set a link's state by id.
    pub fn set_link_state(&mut self, link_id: &str, state: LinkState) -> bool {
        if let Some(link) = self.links.iter_mut().find(|l| l.id == link_id) {
            link.state = state;
            true
        } else {
            false
        }
    }

    /// Select the next link to use for sending, based on the aggregation mode.
    /// Returns the link id or `None` if no active link is available.
    pub fn select_link(&mut self) -> Option<&str> {
        match self.mode {
            AggregationMode::RoundRobin => self.select_round_robin(),
            AggregationMode::ActiveBackup => self.select_active_backup(),
            AggregationMode::Weighted => self.select_weighted(),
        }
    }

    fn select_round_robin(&mut self) -> Option<&str> {
        let n = self.links.len();
        if n == 0 {
            return None;
        }
        for _ in 0..n {
            let idx = self.rr_index % n;
            self.rr_index = self.rr_index.wrapping_add(1);
            if self.links[idx].is_active() {
                self.links[idx].packets_sent += 1;
                return Some(&self.links[idx].id);
            }
        }
        None
    }

    fn select_active_backup(&mut self) -> Option<&str> {
        // Pick the first active link
        for link in &mut self.links {
            if link.is_active() {
                link.packets_sent += 1;
                return Some(&link.id);
            }
        }
        None
    }

    #[allow(clippy::cast_precision_loss)]
    fn select_weighted(&mut self) -> Option<&str> {
        let active: Vec<usize> = self
            .links
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_active())
            .map(|(i, _)| i)
            .collect();

        if active.is_empty() {
            return None;
        }

        // Pick the active link with the lowest (packets_sent / weight) ratio
        let best = active
            .into_iter()
            .min_by(|&a, &b| {
                let ratio_a =
                    self.links[a].packets_sent as f64 / self.links[a].weight.max(1) as f64;
                let ratio_b =
                    self.links[b].packets_sent as f64 / self.links[b].weight.max(1) as f64;
                ratio_a
                    .partial_cmp(&ratio_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("should succeed in test");

        self.links[best].packets_sent += 1;
        Some(&self.links[best].id)
    }

    /// Reset packet counters on all links.
    pub fn reset_counters(&mut self) {
        for link in &mut self.links {
            link.packets_sent = 0;
        }
        self.rr_index = 0;
    }

    /// Get a link by id.
    pub fn get_link(&self, id: &str) -> Option<&PhysicalLink> {
        self.links.iter().find(|l| l.id == id)
    }
}

/// Manages multiple link groups.
#[derive(Debug, Clone)]
pub struct LinkAggregator {
    groups: Vec<LinkGroup>,
}

impl LinkAggregator {
    /// Create a new aggregator.
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Add a link group.
    pub fn add_group(&mut self, group: LinkGroup) {
        self.groups.push(group);
    }

    /// Number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get a mutable group by name.
    pub fn get_group_mut(&mut self, name: &str) -> Option<&mut LinkGroup> {
        self.groups.iter_mut().find(|g| g.name == name)
    }

    /// Get a group by name.
    pub fn get_group(&self, name: &str) -> Option<&LinkGroup> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Total capacity across all groups.
    pub fn total_capacity_mbps(&self) -> u64 {
        self.groups
            .iter()
            .map(|g| g.aggregate_capacity_mbps())
            .sum()
    }
}

impl Default for LinkAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_link_group(mode: AggregationMode) -> LinkGroup {
        let mut g = LinkGroup::new("bond0", mode);
        g.add_link(PhysicalLink::new("eth0", 1000));
        g.add_link(PhysicalLink::new("eth1", 1000));
        g
    }

    #[test]
    fn test_link_is_active() {
        let l = PhysicalLink::new("eth0", 1000);
        assert!(l.is_active());
    }

    #[test]
    fn test_link_down_not_active() {
        let mut l = PhysicalLink::new("eth0", 1000);
        l.state = LinkState::Down;
        assert!(!l.is_active());
    }

    #[test]
    fn test_group_link_count() {
        let g = two_link_group(AggregationMode::RoundRobin);
        assert_eq!(g.link_count(), 2);
        assert_eq!(g.active_count(), 2);
    }

    #[test]
    fn test_aggregate_capacity() {
        let g = two_link_group(AggregationMode::RoundRobin);
        assert_eq!(g.aggregate_capacity_mbps(), 2000);
    }

    #[test]
    fn test_round_robin_alternates() {
        let mut g = two_link_group(AggregationMode::RoundRobin);
        let first = g.select_link().expect("should succeed in test").to_string();
        let second = g.select_link().expect("should succeed in test").to_string();
        assert_ne!(first, second);
    }

    #[test]
    fn test_active_backup_uses_first() {
        let mut g = two_link_group(AggregationMode::ActiveBackup);
        assert_eq!(g.select_link().expect("should succeed in test"), "eth0");
        assert_eq!(g.select_link().expect("should succeed in test"), "eth0");
    }

    #[test]
    fn test_active_backup_failover() {
        let mut g = two_link_group(AggregationMode::ActiveBackup);
        g.set_link_state("eth0", LinkState::Down);
        assert_eq!(g.select_link().expect("should succeed in test"), "eth1");
    }

    #[test]
    fn test_no_active_links() {
        let mut g = two_link_group(AggregationMode::RoundRobin);
        g.set_link_state("eth0", LinkState::Down);
        g.set_link_state("eth1", LinkState::Down);
        assert!(g.select_link().is_none());
    }

    #[test]
    fn test_weighted_prefers_higher_weight() {
        let mut g = LinkGroup::new("bond0", AggregationMode::Weighted);
        g.add_link(PhysicalLink::new("slow", 100).with_weight(1));
        g.add_link(PhysicalLink::new("fast", 1000).with_weight(10));
        // Send several packets; "fast" should get most
        for _ in 0..11 {
            g.select_link();
        }
        let fast = g.get_link("fast").expect("should succeed in test");
        let slow = g.get_link("slow").expect("should succeed in test");
        assert!(fast.packets_sent > slow.packets_sent);
    }

    #[test]
    fn test_set_link_state() {
        let mut g = two_link_group(AggregationMode::RoundRobin);
        assert!(g.set_link_state("eth0", LinkState::Standby));
        assert!(!g.set_link_state("missing", LinkState::Down));
    }

    #[test]
    fn test_reset_counters() {
        let mut g = two_link_group(AggregationMode::RoundRobin);
        g.select_link();
        g.select_link();
        g.reset_counters();
        assert_eq!(
            g.get_link("eth0")
                .expect("should succeed in test")
                .packets_sent,
            0
        );
        assert_eq!(
            g.get_link("eth1")
                .expect("should succeed in test")
                .packets_sent,
            0
        );
    }

    #[test]
    fn test_aggregator_add_and_count() {
        let mut agg = LinkAggregator::new();
        agg.add_group(two_link_group(AggregationMode::RoundRobin));
        assert_eq!(agg.group_count(), 1);
    }

    #[test]
    fn test_aggregator_total_capacity() {
        let mut agg = LinkAggregator::new();
        agg.add_group(two_link_group(AggregationMode::RoundRobin));
        assert_eq!(agg.total_capacity_mbps(), 2000);
    }

    #[test]
    fn test_aggregator_get_group() {
        let mut agg = LinkAggregator::new();
        agg.add_group(two_link_group(AggregationMode::RoundRobin));
        assert!(agg.get_group("bond0").is_some());
        assert!(agg.get_group("missing").is_none());
    }

    #[test]
    fn test_link_with_weight_builder() {
        let l = PhysicalLink::new("eth0", 1000).with_weight(5);
        assert_eq!(l.weight, 5);
    }
}
