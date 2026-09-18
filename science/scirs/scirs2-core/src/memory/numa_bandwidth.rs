//! Cross-NUMA bandwidth measurement and routing.
//!
//! Measures effective memory bandwidth between NUMA nodes and uses the
//! resulting matrix to make optimal data-placement decisions.
//!
//! On systems without `libnuma` (or without multiple NUMA nodes), the module
//! transparently returns a single-node matrix so code using it is portable.
//!
//! # Design
//!
//! ```text
//!   probe_bandwidth_matrix()
//!         │
//!         ├── measure_copy_bandwidth(size)   (warm-up + 3 timed copies)
//!         └── NumaBandwidthMatrix::uniform(1, bw, lat)  (fallback)
//! ```
//!
//! For systems with runtime NUMA topology detection (future: via `libnuma`
//! feature), the matrix would be filled with per-pair measurements.  The
//! present implementation probes the local node and builds a uniform matrix;
//! the API is designed to accommodate full multi-node measurement without
//! breaking changes.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::memory::numa_bandwidth::{probe_bandwidth_matrix, optimal_placement_node};
//!
//! let matrix = probe_bandwidth_matrix();
//! let target = optimal_placement_node(&matrix, 0, 4 * 1024 * 1024);
//! assert!(target < matrix.n_nodes);
//! ```

use std::time::Instant;

// ---------------------------------------------------------------------------
// BandwidthMeasurement
// ---------------------------------------------------------------------------

/// Bandwidth measurement between two NUMA nodes (or within a single node).
#[derive(Debug, Clone)]
pub struct BandwidthMeasurement {
    /// Source NUMA node index.
    pub from_node: usize,
    /// Destination NUMA node index.
    pub to_node: usize,
    /// Measured bandwidth in GB/s.
    pub bandwidth_gb_s: f64,
    /// Average per-transfer latency in nanoseconds.
    pub latency_ns: f64,
    /// Transfer size used for the measurement.
    pub transfer_size_bytes: usize,
}

// ---------------------------------------------------------------------------
// NumaBandwidthMatrix
// ---------------------------------------------------------------------------

/// NUMA bandwidth matrix: `bandwidth[from][to]` in GB/s, `latency[from][to]` in ns.
#[derive(Debug, Clone)]
pub struct NumaBandwidthMatrix {
    /// Number of NUMA nodes.
    pub n_nodes: usize,
    /// Bandwidth from node `i` to node `j` in GB/s.  Row-major: `[from][to]`.
    pub bandwidth: Vec<Vec<f64>>,
    /// Latency from node `i` to node `j` in nanoseconds.  Row-major: `[from][to]`.
    pub latency: Vec<Vec<f64>>,
}

impl NumaBandwidthMatrix {
    /// Create a uniform matrix where every (i, j) pair has the same bandwidth
    /// and latency.  Useful as a single-node fallback.
    pub fn uniform(n_nodes: usize, bandwidth_gb_s: f64, latency_ns: f64) -> Self {
        let row = vec![bandwidth_gb_s; n_nodes.max(1)];
        let lat_row = vec![latency_ns; n_nodes.max(1)];
        let n = n_nodes.max(1);
        Self {
            n_nodes: n,
            bandwidth: vec![row; n],
            latency: vec![lat_row; n],
        }
    }

    /// Get bandwidth from node `from` to node `to` in GB/s.
    ///
    /// Returns `0.0` if either index is out of range.
    pub fn get_bandwidth(&self, from: usize, to: usize) -> f64 {
        self.bandwidth
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get latency from node `from` to node `to` in nanoseconds.
    ///
    /// Returns `f64::MAX` if either index is out of range.
    pub fn get_latency(&self, from: usize, to: usize) -> f64 {
        self.latency
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(f64::MAX)
    }

    /// Find the highest-bandwidth route from `src` to `dst`.
    ///
    /// For a 1-node matrix this is always the diagonal.
    /// For multi-node matrices, returns the direct bandwidth (no intermediate
    /// hop logic needed for cache-coherent NUMA).
    ///
    /// Returns the bandwidth value in GB/s.
    pub fn best_route(&self, src: usize, dst: usize) -> f64 {
        self.get_bandwidth(src, dst)
    }

    /// Find the node with the highest average outgoing bandwidth.
    ///
    /// Ties are broken by choosing the lower node index.
    pub fn highest_bandwidth_node(&self) -> usize {
        let mut best_node = 0usize;
        let mut best_avg = f64::NEG_INFINITY;

        for from in 0..self.n_nodes {
            let row = &self.bandwidth[from];
            let avg = if row.is_empty() {
                0.0
            } else {
                row.iter().sum::<f64>() / row.len() as f64
            };
            if avg > best_avg {
                best_avg = avg;
                best_node = from;
            }
        }
        best_node
    }

    /// Format the bandwidth matrix as a human-readable table.
    pub fn display(&self) -> String {
        let mut out = String::from("NUMA Bandwidth Matrix (GB/s):\n");
        out.push_str("     ");
        for j in 0..self.n_nodes {
            out.push_str(&format!("  {:>6}", format!("N{j}")));
        }
        out.push('\n');

        for (i, row) in self.bandwidth.iter().enumerate() {
            out.push_str(&format!("N{i:<4}"));
            for &bw in row {
                out.push_str(&format!("  {:>6.2}", bw));
            }
            out.push('\n');
        }
        out
    }

    /// Number of NUMA nodes in this matrix.
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }
}

// ---------------------------------------------------------------------------
// measure_copy_bandwidth
// ---------------------------------------------------------------------------

/// Measure actual memory copy bandwidth using a warm-up pass followed by
/// three timed copies of `transfer_size_bytes` bytes.
///
/// The result represents intra-node (local) bandwidth; use multiple calls to
/// measure cross-node bandwidth once libnuma pinning is available.
pub fn measure_copy_bandwidth(transfer_size_bytes: usize) -> BandwidthMeasurement {
    // Allocate source and destination buffers.
    let src: Vec<u8> = vec![0xABu8; transfer_size_bytes];
    let mut dst = vec![0u8; transfer_size_bytes];

    // Warm-up pass (pulls buffers into cache / TLB).
    dst.copy_from_slice(&src);

    // Prevent the compiler from eliding the copies.
    let _ = dst[transfer_size_bytes / 2];

    // Timed measurement: 3 copies.
    let repetitions: u64 = 3;
    let start = Instant::now();
    for _ in 0..repetitions {
        dst.copy_from_slice(&src);
    }
    let elapsed = start.elapsed();

    // Prevent elision of the last copy.
    let _ = dst[0];

    let bytes_transferred = transfer_size_bytes as u64 * repetitions;
    let elapsed_secs = elapsed.as_secs_f64();

    // Guard against zero elapsed time (possible on very fast CPUs).
    let bandwidth_gb_s = if elapsed_secs > 0.0 {
        bytes_transferred as f64 / elapsed_secs / 1e9
    } else {
        f64::MAX
    };

    let latency_ns = if repetitions > 0 {
        elapsed.as_nanos() as f64 / repetitions as f64
    } else {
        0.0
    };

    BandwidthMeasurement {
        from_node: 0,
        to_node: 0,
        bandwidth_gb_s,
        latency_ns,
        transfer_size_bytes,
    }
}

// ---------------------------------------------------------------------------
// probe_bandwidth_matrix
// ---------------------------------------------------------------------------

/// Build a bandwidth matrix by probing memory copy throughput.
///
/// On systems without `libnuma` or without multiple NUMA nodes, returns a
/// 1-node uniform matrix populated with the measured copy bandwidth.
///
/// This function performs real memory copies (4 MiB probe by default) so
/// it should be called once at startup and the result cached.
pub fn probe_bandwidth_matrix() -> NumaBandwidthMatrix {
    // Future: detect n_nodes from libnuma or sysfs when available.
    // Current: single-node fallback.
    let n_nodes = detect_numa_node_count();
    let probe_size = 4 * 1024 * 1024; // 4 MiB
    let measurement = measure_copy_bandwidth(probe_size);

    NumaBandwidthMatrix::uniform(n_nodes, measurement.bandwidth_gb_s, measurement.latency_ns)
}

/// Detect the number of NUMA nodes available on this system.
///
/// Falls back to 1 when NUMA topology cannot be determined without `libnuma`.
fn detect_numa_node_count() -> usize {
    // On Linux we can read /sys/devices/system/node/online to get the count.
    #[cfg(target_os = "linux")]
    {
        if let Some(count) = try_read_linux_numa_count() {
            return count;
        }
    }
    1
}

#[cfg(target_os = "linux")]
fn try_read_linux_numa_count() -> Option<usize> {
    use std::fs;
    let contents = fs::read_to_string("/sys/devices/system/node/online").ok()?;
    // Format is e.g. "0-3" or "0" or "0,2-4".
    // Count the nodes by parsing the range/list.
    parse_node_count_from_range(contents.trim())
}

/// Parse a node count from a Linux cpumask/nodelist string like "0-3" or "0,2,4-6".
fn parse_node_count_from_range(s: &str) -> Option<usize> {
    let mut count = 0usize;
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo_str, hi_str)) = part.split_once('-') {
            let lo: usize = lo_str.trim().parse().ok()?;
            let hi: usize = hi_str.trim().parse().ok()?;
            if hi >= lo {
                count += hi - lo + 1;
            }
        } else {
            // Single node id.
            let _id: usize = part.parse().ok()?;
            count += 1;
        }
    }
    if count > 0 {
        Some(count)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// optimal_placement_node
// ---------------------------------------------------------------------------

/// Route a data transfer to maximise bandwidth.
///
/// Returns the optimal target NUMA node for placement of `data_size` bytes
/// that will be accessed from `src_node`.
///
/// For same-node access this returns `src_node`; for cross-node access it
/// returns the node with the highest bandwidth from `src_node`.
pub fn optimal_placement_node(
    matrix: &NumaBandwidthMatrix,
    src_node: usize,
    data_size: usize,
) -> usize {
    let _ = data_size; // Size could influence threshold decisions in future.

    if src_node >= matrix.n_nodes {
        return 0;
    }

    // Find the destination node with the highest bandwidth from src_node.
    let bandwidth_row = &matrix.bandwidth[src_node];
    let mut best_node = src_node;
    let mut best_bw = bandwidth_row.get(src_node).copied().unwrap_or(0.0);

    for (to, &bw) in bandwidth_row.iter().enumerate() {
        if bw > best_bw {
            best_bw = bw;
            best_node = to;
        }
    }
    best_node
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_matrix_uniform() {
        let matrix = NumaBandwidthMatrix::uniform(2, 50.0, 100.0);
        assert_eq!(matrix.n_nodes, 2);
        assert_eq!(matrix.get_bandwidth(0, 0), 50.0);
        assert_eq!(matrix.get_bandwidth(0, 1), 50.0);
        assert_eq!(matrix.get_bandwidth(1, 0), 50.0);
        assert_eq!(matrix.get_bandwidth(1, 1), 50.0);
        assert_eq!(matrix.get_latency(0, 1), 100.0);
        // Out of range returns sentinel.
        assert_eq!(matrix.get_bandwidth(5, 0), 0.0);
        assert_eq!(matrix.get_latency(5, 0), f64::MAX);
    }

    #[test]
    fn test_bandwidth_matrix_single_node_fallback() {
        // uniform with 0 nodes should produce a 1-node matrix.
        let matrix = NumaBandwidthMatrix::uniform(0, 10.0, 200.0);
        assert_eq!(matrix.n_nodes, 1);
        assert_eq!(matrix.get_bandwidth(0, 0), 10.0);
    }

    #[test]
    fn test_bandwidth_measure() {
        // 1 MB probe — just verify we get a positive bandwidth.
        let m = measure_copy_bandwidth(1024 * 1024);
        assert!(m.bandwidth_gb_s > 0.0, "bandwidth must be positive");
        assert!(m.latency_ns > 0.0, "latency must be positive");
        assert_eq!(m.transfer_size_bytes, 1024 * 1024);
        assert_eq!(m.from_node, 0);
        assert_eq!(m.to_node, 0);
    }

    #[test]
    fn test_bandwidth_matrix_display() {
        let matrix = NumaBandwidthMatrix::uniform(2, 42.0, 80.0);
        let s = matrix.display();
        assert!(!s.is_empty(), "display string should not be empty");
        assert!(s.contains("42.00"), "should contain bandwidth value");
    }

    #[test]
    fn test_optimal_placement_single_node() {
        let matrix = NumaBandwidthMatrix::uniform(1, 50.0, 100.0);
        let node = optimal_placement_node(&matrix, 0, 4 * 1024 * 1024);
        assert_eq!(node, 0, "single-node system => always node 0");
    }

    #[test]
    fn test_optimal_placement_out_of_range() {
        let matrix = NumaBandwidthMatrix::uniform(2, 50.0, 100.0);
        // src_node >= n_nodes should return 0 safely.
        let node = optimal_placement_node(&matrix, 99, 1024);
        assert_eq!(node, 0, "out-of-range src should return 0");
    }

    #[test]
    fn test_optimal_placement_multi_node_prefers_high_bw() {
        let n = 3;
        let mut matrix = NumaBandwidthMatrix::uniform(n, 10.0, 100.0);
        // Make node 0 -> node 2 the highest bandwidth link.
        matrix.bandwidth[0][2] = 100.0;
        let node = optimal_placement_node(&matrix, 0, 1024);
        assert_eq!(node, 2, "should prefer node 2 with highest bandwidth");
    }

    #[test]
    fn test_highest_bandwidth_node() {
        let n = 3;
        let mut matrix = NumaBandwidthMatrix::uniform(n, 10.0, 100.0);
        // Make node 1 have the highest outgoing bandwidth overall.
        for j in 0..n {
            matrix.bandwidth[1][j] = 50.0;
        }
        assert_eq!(
            matrix.highest_bandwidth_node(),
            1,
            "node 1 should have the highest average outgoing BW"
        );
    }

    #[test]
    fn test_best_route() {
        let matrix = NumaBandwidthMatrix::uniform(2, 30.0, 50.0);
        assert_eq!(matrix.best_route(0, 1), 30.0);
        assert_eq!(matrix.best_route(1, 0), 30.0);
    }

    #[test]
    fn test_probe_bandwidth_matrix_returns_valid() {
        let matrix = probe_bandwidth_matrix();
        assert!(matrix.n_nodes >= 1, "must have at least one node");
        assert!(
            matrix.get_bandwidth(0, 0) > 0.0,
            "local bandwidth must be positive"
        );
    }

    #[test]
    fn test_parse_node_count_from_range() {
        assert_eq!(parse_node_count_from_range("0"), Some(1));
        assert_eq!(parse_node_count_from_range("0-3"), Some(4));
        assert_eq!(parse_node_count_from_range("0,2"), Some(2));
        assert_eq!(parse_node_count_from_range("0-1,4-7"), Some(6));
        assert_eq!(parse_node_count_from_range(""), None);
    }
}
