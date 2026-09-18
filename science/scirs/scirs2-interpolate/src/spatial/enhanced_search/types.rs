//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;

/// Locality Sensitive Hashing index for approximate search
#[derive(Debug)]
pub struct LSHIndex<F: Float> {
    /// Hash tables
    #[allow(dead_code)]
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
    /// Random projection matrices
    #[allow(dead_code)]
    projections: Vec<Array2<F>>,
    /// Number of hash functions per table
    #[allow(dead_code)]
    hash_functions_per_table: usize,
    /// Number of hash tables
    #[allow(dead_code)]
    num_tables: usize,
    /// Hash bucket width
    #[allow(dead_code)]
    bucket_width: F,
}
impl<F: Float + FromPrimitive> LSHIndex<F> {
    pub fn new(points: &Array2<F>, config: &SearchConfig) -> InterpolateResult<Self> {
        if points.is_empty() {
            return Err(InterpolateError::invalid_input("Points array is empty"));
        }
        let dimensions = points.ncols();
        let num_points = points.nrows();
        let num_tables = if num_points > 10000 { 20 } else { 10 };
        let hash_functions_per_table = if dimensions > 50 { 10 } else { 6 };
        let bucket_width = F::from_f64(config.approximation_factor * 0.5).ok_or_else(|| {
            InterpolateError::ComputationError(
                "Failed to convert bucket width to float type".to_string(),
            )
        })?;
        let mut hash_tables = Vec::with_capacity(num_tables);
        let mut projections = Vec::with_capacity(num_tables);
        let mut seed = 42u64;
        for _ in 0..num_tables {
            hash_tables.push(HashMap::new());
            let mut projection = Array2::zeros((hash_functions_per_table, dimensions));
            for i in 0..hash_functions_per_table {
                for j in 0..dimensions {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let random_val = (seed as f64) / (u64::MAX as f64) * 2.0 - 1.0;
                    projection[[i, j]] = F::from_f64(random_val).ok_or_else(|| {
                        InterpolateError::ComputationError(
                            "Failed to convert random value to float type".to_string(),
                        )
                    })?;
                }
            }
            projections.push(projection);
        }
        let mut index = Self {
            hash_tables,
            projections,
            hash_functions_per_table,
            num_tables,
            bucket_width,
        };
        for (point_idx, point) in points.axis_iter(scirs2_core::ndarray::Axis(0)).enumerate() {
            for table_idx in 0..num_tables {
                let hash_key = index.compute_hash(&point, table_idx)?;
                index.hash_tables[table_idx]
                    .entry(hash_key)
                    .or_default()
                    .push(point_idx);
            }
        }
        Ok(index)
    }
    /// Compute hash key for a point using a specific table's projection
    fn compute_hash(&self, point: &ArrayView1<F>, tableidx: usize) -> InterpolateResult<u64> {
        let projection = &self.projections[tableidx];
        let mut hash_key = 0u64;
        for hash_func_idx in 0..self.hash_functions_per_table {
            let projection_row = projection.slice(scirs2_core::ndarray::s![hash_func_idx, ..]);
            let dot_product = point
                .iter()
                .zip(projection_row.iter())
                .map(|(&p, &proj)| p * proj)
                .fold(F::zero(), |acc, x| acc + x);
            let bucket = (dot_product / self.bucket_width)
                .floor()
                .to_u64()
                .unwrap_or(0);
            hash_key ^= bucket.wrapping_shl(hash_func_idx as u32);
        }
        Ok(hash_key)
    }
    pub fn approximate_k_nearest_neighbors(
        &self,
        query: &ArrayView1<F>,
        k: usize,
        points: &Array2<F>,
        stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut candidates = std::collections::HashSet::new();
        for table_idx in 0..self.num_tables {
            let hash_key = self.compute_hash(query, table_idx)?;
            for offset in -1i64..=1i64 {
                let neighbor_key = (hash_key as i64).wrapping_add(offset) as u64;
                if let Some(bucket_points) = self.hash_tables[table_idx].get(&neighbor_key) {
                    for &point_idx in bucket_points {
                        candidates.insert(point_idx);
                    }
                }
            }
        }
        if candidates.len() < k * 2 {
            for table_idx in 0..self.num_tables {
                let hash_key = self.compute_hash(query, table_idx)?;
                for offset in -2i64..=2i64 {
                    let neighbor_key = (hash_key as i64).wrapping_add(offset) as u64;
                    if let Some(bucket_points) = self.hash_tables[table_idx].get(&neighbor_key) {
                        for &point_idx in bucket_points {
                            candidates.insert(point_idx);
                        }
                    }
                }
            }
        }
        let mut neighbors = BinaryHeap::new();
        for &candidate_idx in &candidates {
            let point = points.slice(scirs2_core::ndarray::s![candidate_idx, ..]);
            let distance_squared = query
                .iter()
                .zip(point.iter())
                .map(|(&q, &p)| {
                    let diff = q - p;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x);
            stats.distance_computations += 1;
            let distance = distance_squared.sqrt();
            if neighbors.len() < k {
                neighbors.push(std::cmp::Reverse((OrderedFloat(distance), candidate_idx)));
            } else if let Some(&std::cmp::Reverse((OrderedFloat(max_dist), _))) = neighbors.peek() {
                if distance < max_dist {
                    neighbors.pop();
                    neighbors.push(std::cmp::Reverse((OrderedFloat(distance), candidate_idx)));
                }
            }
        }
        let mut result: Vec<_> = neighbors
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(dist), idx))| (idx, dist))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }
    pub fn approximate_radius_neighbors(
        &self,
        query: &ArrayView1<F>,
        radius: F,
        points: &Array2<F>,
        stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut candidates = std::collections::HashSet::new();
        let radius_squared = radius * radius;
        for table_idx in 0..self.num_tables {
            let hash_key = self.compute_hash(query, table_idx)?;
            let search_width = ((radius / self.bucket_width).ceil().to_i64().unwrap_or(3)).max(3);
            for offset in -search_width..=search_width {
                let neighbor_key = (hash_key as i64).wrapping_add(offset) as u64;
                if let Some(bucket_points) = self.hash_tables[table_idx].get(&neighbor_key) {
                    for &point_idx in bucket_points {
                        candidates.insert(point_idx);
                    }
                }
            }
        }
        let mut neighbors = Vec::new();
        for &candidate_idx in &candidates {
            let point = points.slice(scirs2_core::ndarray::s![candidate_idx, ..]);
            let distance_squared = query
                .iter()
                .zip(point.iter())
                .map(|(&q, &p)| {
                    let diff = q - p;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x);
            stats.distance_computations += 1;
            if distance_squared <= radius_squared {
                neighbors.push((candidate_idx, distance_squared.sqrt()));
            }
        }
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(neighbors)
    }
}
/// Helper struct for priority queue in k-nearest neighbors search
#[derive(Debug, Clone)]
pub struct Neighbor<F: Float> {
    pub(super) index: usize,
    pub(super) distance: OrderedFloat<F>,
}
/// Ball tree implementation for high-dimensional data
#[derive(Debug)]
pub struct BallTreeIndex<F: Float> {
    /// Tree nodes
    nodes: Vec<BallTreeNode<F>>,
    /// Root node index
    root: usize,
    /// Number of dimensions
    dimensions: usize,
    /// Original points data
    points: Array2<F>,
}
impl<F: Float + FromPrimitive> BallTreeIndex<F> {
    /// Build a new Ball tree from points
    pub fn new(points: &Array2<F>) -> InterpolateResult<Self> {
        if points.is_empty() {
            return Err(InterpolateError::invalid_input("Points array is empty"));
        }
        let dimensions = points.ncols();
        let num_points = points.nrows();
        let mut indices: Vec<usize> = (0..num_points).collect();
        let mut nodes = Vec::new();
        let root = Self::build_tree_recursive(points, &mut indices, dimensions, &mut nodes)?;
        Ok(Self {
            nodes,
            root,
            dimensions,
            points: points.to_owned(),
        })
    }
    /// Recursively build the Ball tree
    fn build_tree_recursive(
        points: &Array2<F>,
        indices: &mut [usize],
        dimensions: usize,
        nodes: &mut Vec<BallTreeNode<F>>,
    ) -> InterpolateResult<usize> {
        if indices.is_empty() {
            return Err(InterpolateError::invalid_input(
                "Empty indices in tree building",
            ));
        }
        let mut centroid = vec![F::zero(); dimensions];
        for &idx in indices.iter() {
            for d in 0..dimensions {
                centroid[d] = centroid[d] + points[[idx, d]];
            }
        }
        let n = F::from_usize(indices.len()).unwrap_or(F::one());
        for item in centroid.iter_mut().take(dimensions) {
            *item = *item / n;
        }
        let mut radius = F::zero();
        for &idx in indices.iter() {
            let mut dist_sq = F::zero();
            for d in 0..dimensions {
                let diff = points[[idx, d]] - centroid[d];
                dist_sq = dist_sq + diff * diff;
            }
            radius = radius.max(dist_sq.sqrt());
        }
        let node_idx = nodes.len();
        let is_leaf = indices.len() <= 1;
        nodes.push(BallTreeNode {
            point_indices: indices.to_vec(),
            center: centroid,
            radius,
            left: None,
            right: None,
            is_leaf,
        });
        if !is_leaf {
            let mut best_dim = 0;
            let mut max_spread = F::zero();
            for d in 0..dimensions {
                let mut min_val = F::infinity();
                let mut max_val = F::neg_infinity();
                for &idx in indices.iter() {
                    let val = points[[idx, d]];
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
                let spread = max_val - min_val;
                if spread > max_spread {
                    max_spread = spread;
                    best_dim = d;
                }
            }
            indices.sort_by(|&a, &b| {
                let val_a = points[[a, best_dim]];
                let val_b = points[[b, best_dim]];
                val_a.partial_cmp(&val_b).unwrap_or(Ordering::Equal)
            });
            let split_idx = indices.len() / 2;
            let left_idx =
                Self::build_tree_recursive(points, &mut indices[..split_idx], dimensions, nodes)?;
            nodes[node_idx].left = Some(left_idx);
            let right_idx =
                Self::build_tree_recursive(points, &mut indices[split_idx..], dimensions, nodes)?;
            nodes[node_idx].right = Some(right_idx);
        }
        Ok(node_idx)
    }
    /// Find k nearest neighbors
    pub fn k_nearest_neighbors(
        &self,
        query: &ArrayView1<F>,
        k: usize,
        stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut best = BinaryHeap::new();
        self.search_knn_recursive(self.root, query, k, &mut best, stats)?;
        let mut result = Vec::new();
        while let Some(neighbor) = best.pop() {
            result.push((neighbor.index, neighbor.distance.0));
        }
        result.reverse();
        Ok(result)
    }
    /// Recursive k-nearest neighbors search
    fn search_knn_recursive(
        &self,
        node_idx: usize,
        query: &ArrayView1<F>,
        k: usize,
        best: &mut BinaryHeap<Neighbor<F>>,
        stats: &mut SearchStats,
    ) -> InterpolateResult<()> {
        stats.nodes_visited += 1;
        let node = &self.nodes[node_idx];
        let mut center_dist_sq = F::zero();
        for d in 0..self.dimensions {
            let diff = query[d] - node.center[d];
            center_dist_sq = center_dist_sq + diff * diff;
        }
        let center_dist = center_dist_sq.sqrt();
        let worst_dist = best.peek().map(|n| n.distance.0).unwrap_or(F::infinity());
        let min_possible_dist = (center_dist - node.radius).max(F::zero());
        if best.len() >= k && min_possible_dist >= worst_dist {
            return Ok(());
        }
        if node.is_leaf {
            for &point_idx in &node.point_indices {
                let mut distance_sq = F::zero();
                for d in 0..self.dimensions {
                    let diff = query[d] - self.points[[point_idx, d]];
                    distance_sq = distance_sq + diff * diff;
                }
                let distance = distance_sq.sqrt();
                let neighbor = Neighbor {
                    index: point_idx,
                    distance: OrderedFloat(distance),
                };
                if best.len() < k {
                    best.push(neighbor);
                } else if neighbor.distance < best.peek().expect("Operation failed").distance {
                    best.pop();
                    best.push(neighbor);
                }
            }
        } else {
            if let Some(left_idx) = node.left {
                self.search_knn_recursive(left_idx, query, k, best, stats)?;
            }
            if let Some(right_idx) = node.right {
                self.search_knn_recursive(right_idx, query, k, best, stats)?;
            }
        }
        Ok(())
    }
    /// Find all neighbors within a given radius
    pub fn radius_neighbors(
        &self,
        query: &ArrayView1<F>,
        radius: F,
        stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut result = Vec::new();
        self.search_radius_recursive(self.root, query, radius, &mut result, stats)?;
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }
    /// Recursive radius search
    fn search_radius_recursive(
        &self,
        node_idx: usize,
        query: &ArrayView1<F>,
        radius: F,
        result: &mut Vec<(usize, F)>,
        stats: &mut SearchStats,
    ) -> InterpolateResult<()> {
        stats.nodes_visited += 1;
        let node = &self.nodes[node_idx];
        let mut center_dist_sq = F::zero();
        for d in 0..self.dimensions {
            let diff = query[d] - node.center[d];
            center_dist_sq = center_dist_sq + diff * diff;
        }
        let center_dist = center_dist_sq.sqrt();
        let min_possible_dist = (center_dist - node.radius).max(F::zero());
        if min_possible_dist > radius {
            return Ok(());
        }
        if node.is_leaf {
            for &point_idx in &node.point_indices {
                let mut distance_sq = F::zero();
                for d in 0..self.dimensions {
                    let diff = query[d] - self.points[[point_idx, d]];
                    distance_sq = distance_sq + diff * diff;
                }
                let distance = distance_sq.sqrt();
                if distance <= radius {
                    result.push((point_idx, distance));
                }
            }
        } else {
            if let Some(left_idx) = node.left {
                self.search_radius_recursive(left_idx, query, radius, result, stats)?;
            }
            if let Some(right_idx) = node.right {
                self.search_radius_recursive(right_idx, query, radius, result, stats)?;
            }
        }
        Ok(())
    }
}
/// Enhanced nearest neighbor searcher with multiple algorithms
#[derive(Debug)]
pub struct EnhancedNearestNeighborSearcher<F: Float + FromPrimitive + Debug + Send + Sync> {
    /// Training data points
    points: Array2<F>,
    /// Spatial index type currently in use
    index_type: IndexType,
    /// Search configuration
    config: SearchConfig,
    /// KD-tree index (if applicable)
    kdtree: Option<KdTreeIndex<F>>,
    /// Ball tree index (if applicable)
    balltree: Option<BallTreeIndex<F>>,
    /// LSH index for approximate search (if applicable)
    lsh_index: Option<LSHIndex<F>>,
    /// Query cache for repeated searches
    query_cache: HashMap<QueryKey, Vec<(usize, F)>>,
    /// Performance statistics
    stats: SearchStats,
}
impl<F> EnhancedNearestNeighborSearcher<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    /// Create a new enhanced nearest neighbor searcher
    ///
    /// # Arguments
    ///
    /// * `points` - Training data points with shape (n_points, n_dims)
    /// * `index_type` - Type of spatial index to use
    /// * `config` - Search configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_core::ndarray::Array2;
    /// use scirs2_interpolate::spatial::enhanced_search::{
    ///     EnhancedNearestNeighborSearcher, IndexType, SearchConfig
    /// };
    ///
    /// let points = Array2::from_shape_vec((5, 2), vec![
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 1.0,
    ///     0.5, 0.5,
    /// ]).expect("Operation failed");
    ///
    /// let config = SearchConfig::default();
    /// let searcher = EnhancedNearestNeighborSearcher::new(
    ///     points, IndexType::Adaptive, config
    /// ).expect("Operation failed");
    /// ```
    pub fn new(
        points: Array2<F>,
        index_type: IndexType,
        config: SearchConfig,
    ) -> InterpolateResult<Self> {
        let n_points = points.nrows();
        let n_dims = points.ncols();
        if n_points == 0 {
            return Err(InterpolateError::invalid_input(
                "Cannot create searcher with zero points".to_string(),
            ));
        }
        let actual_index_type = match index_type {
            IndexType::Adaptive => Self::choose_index_type(n_points, n_dims, &config),
            other => other,
        };
        let mut searcher = Self {
            points,
            index_type: actual_index_type.clone(),
            config,
            kdtree: None,
            balltree: None,
            lsh_index: None,
            query_cache: HashMap::new(),
            stats: SearchStats::default(),
        };
        searcher.build_index(actual_index_type)?;
        Ok(searcher)
    }
    /// Choose the best index type based on data characteristics
    fn choose_index_type(n_points: usize, n_dims: usize, config: &SearchConfig) -> IndexType {
        if config.approximation_factor > 1.0 && n_points > 10000 {
            IndexType::LSH
        } else if n_dims <= 10 && n_points > 100 {
            IndexType::KdTree
        } else if n_dims <= 50 && n_points > 50 {
            IndexType::BallTree
        } else {
            IndexType::BruteForce
        }
    }
    /// Build the spatial index
    fn build_index(&mut self, indextype: IndexType) -> InterpolateResult<()> {
        match indextype {
            IndexType::KdTree => {
                self.kdtree = Some(KdTreeIndex::new(&self.points)?);
            }
            IndexType::BallTree => {
                self.balltree = Some(BallTreeIndex::new(&self.points)?);
            }
            IndexType::LSH => {
                self.lsh_index = Some(LSHIndex::new(&self.points, &self.config)?);
            }
            IndexType::BruteForce | IndexType::Adaptive => {}
        }
        Ok(())
    }
    /// Find k nearest neighbors for a query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) pairs sorted by distance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_core::ndarray::{Array1, Array2};
    /// use scirs2_interpolate::spatial::enhanced_search::{
    ///     EnhancedNearestNeighborSearcher, IndexType, SearchConfig
    /// };
    ///
    /// let points = Array2::from_shape_vec((4, 2), vec![
    ///     0.0, 0.0,
    ///     1.0, 0.0,
    ///     0.0, 1.0,
    ///     1.0, 1.0,
    /// ]).expect("Operation failed");
    ///
    /// let mut searcher = EnhancedNearestNeighborSearcher::new(
    ///     points, IndexType::BruteForce, SearchConfig::default()
    /// ).expect("Operation failed");
    ///
    /// let query = Array1::from_vec(vec![0.5, 0.5]);
    /// let neighbors = searcher.k_nearest_neighbors(&query.view(), 2).expect("Operation failed");
    ///
    /// assert_eq!(neighbors.len(), 2);
    /// ```
    pub fn k_nearest_neighbors(
        &mut self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let start_time = std::time::Instant::now();
        self.stats.total_queries += 1;
        if self.config.cache_results {
            let cache_key =
                QueryKey::from_coords(query.as_slice().expect("Operation failed"), k, None);
            if let Some(cached_result) = self.query_cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                return Ok(cached_result.clone());
            }
        }
        let result = match &self.index_type {
            IndexType::BruteForce => self.brute_force_knn(query, k),
            IndexType::KdTree => {
                if let Some(ref kdtree) = self.kdtree {
                    kdtree.k_nearest_neighbors(query, k, &mut self.stats)
                } else {
                    self.brute_force_knn(query, k)
                }
            }
            IndexType::BallTree => {
                if let Some(ref balltree) = self.balltree {
                    balltree.k_nearest_neighbors(query, k, &mut self.stats)
                } else {
                    self.brute_force_knn(query, k)
                }
            }
            IndexType::LSH => {
                if let Some(ref lsh) = self.lsh_index {
                    lsh.approximate_k_nearest_neighbors(query, k, &self.points, &mut self.stats)
                } else {
                    self.brute_force_knn(query, k)
                }
            }
            IndexType::Adaptive => self.brute_force_knn(query, k),
        };
        if self.config.cache_results {
            if let Ok(ref neighbors) = result {
                let cache_key =
                    QueryKey::from_coords(query.as_slice().expect("Operation failed"), k, None);
                self.query_cache.insert(cache_key, neighbors.clone());
            }
        }
        let elapsed = start_time.elapsed();
        self.stats.total_query_time_us += elapsed.as_micros() as u64;
        result
    }
    /// Find all neighbors within a given radius
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) pairs within the radius
    pub fn radius_neighbors(
        &mut self,
        query: &ArrayView1<F>,
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let start_time = std::time::Instant::now();
        self.stats.total_queries += 1;
        if self.config.cache_results {
            let cache_key =
                QueryKey::from_coords(query.as_slice().expect("Operation failed"), 0, Some(radius));
            if let Some(cached_result) = self.query_cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                return Ok(cached_result.clone());
            }
        }
        let result = match &self.index_type {
            IndexType::BruteForce => self.brute_force_radius(query, radius),
            IndexType::KdTree => {
                if let Some(ref kdtree) = self.kdtree {
                    kdtree.radius_neighbors(query, radius, &mut self.stats)
                } else {
                    self.brute_force_radius(query, radius)
                }
            }
            IndexType::BallTree => {
                if let Some(ref balltree) = self.balltree {
                    balltree.radius_neighbors(query, radius, &mut self.stats)
                } else {
                    self.brute_force_radius(query, radius)
                }
            }
            IndexType::LSH => {
                if let Some(ref lsh) = self.lsh_index {
                    lsh.approximate_radius_neighbors(query, radius, &self.points, &mut self.stats)
                } else {
                    self.brute_force_radius(query, radius)
                }
            }
            IndexType::Adaptive => self.brute_force_radius(query, radius),
        };
        if self.config.cache_results {
            if let Ok(ref neighbors) = result {
                let cache_key = QueryKey::from_coords(
                    query.as_slice().expect("Operation failed"),
                    0,
                    Some(radius),
                );
                self.query_cache.insert(cache_key, neighbors.clone());
            }
        }
        let elapsed = start_time.elapsed();
        self.stats.total_query_time_us += elapsed.as_micros() as u64;
        result
    }
    /// Perform k-nearest neighbor search on multiple query points in parallel
    ///
    /// # Arguments
    ///
    /// * `queries` - Query points with shape (n_queries, n_dims)
    /// * `k` - Number of nearest neighbors per query
    ///
    /// # Returns
    ///
    /// Vector of neighbor results for each query point
    pub fn batch_k_nearest_neighbors(
        &mut self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        let n_queries = queries.nrows();
        if !self.config.parallel_search || n_queries < 10 {
            let mut results = Vec::with_capacity(n_queries);
            for i in 0..n_queries {
                let query = queries.slice(scirs2_core::ndarray::s![i, ..]);
                results.push(self.k_nearest_neighbors(&query, k)?);
            }
            Ok(results)
        } else {
            self.parallel_batch_knn(queries, k)
        }
    }
    /// Parallel implementation of batch k-nearest neighbor search
    fn parallel_batch_knn(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        let queries_owned = queries.to_owned();
        let points = &self.points;
        let results: Result<Vec<_>, InterpolateError> =
            if let Some(_num_threads) = self.config.num_threads {
                (0..queries_owned.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let query = queries_owned.slice(scirs2_core::ndarray::s![i, ..]);
                        Self::parallel_brute_force_knn(&query, k, points)
                    })
                    .collect()
            } else {
                (0..queries_owned.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let query = queries_owned.slice(scirs2_core::ndarray::s![i, ..]);
                        Self::parallel_brute_force_knn(&query, k, points)
                    })
                    .collect()
            };
        results
    }
    /// Thread-safe brute force k-NN for parallel execution
    fn parallel_brute_force_knn(
        query: &ArrayView1<F>,
        k: usize,
        points: &Array2<F>,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut neighbors = BinaryHeap::new();
        for (idx, point) in points.axis_iter(Axis(0)).enumerate() {
            let distance = Self::euclidean_distance_squared(query, &point);
            if neighbors.len() < k {
                neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
            } else if let Some(&std::cmp::Reverse((OrderedFloat(max_dist), _))) = neighbors.peek() {
                if distance < max_dist {
                    neighbors.pop();
                    neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
                }
            }
        }
        let mut result: Vec<_> = neighbors
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(dist), idx))| (idx, dist.sqrt()))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }
    /// Brute force k-nearest neighbor search
    fn brute_force_knn(
        &mut self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut neighbors = BinaryHeap::new();
        for (idx, point) in self.points.axis_iter(Axis(0)).enumerate() {
            let distance = Self::euclidean_distance_squared(query, &point);
            self.stats.distance_computations += 1;
            if neighbors.len() < k {
                neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
            } else if let Some(&std::cmp::Reverse((OrderedFloat(max_dist), _))) = neighbors.peek() {
                if distance < max_dist {
                    neighbors.pop();
                    neighbors.push(std::cmp::Reverse((OrderedFloat(distance), idx)));
                }
            }
        }
        let mut result: Vec<_> = neighbors
            .into_iter()
            .map(|std::cmp::Reverse((OrderedFloat(dist), idx))| (idx, dist.sqrt()))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }
    /// Brute force radius neighbor search
    fn brute_force_radius(
        &mut self,
        query: &ArrayView1<F>,
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut neighbors = Vec::new();
        let radius_squared = radius * radius;
        for (idx, point) in self.points.axis_iter(Axis(0)).enumerate() {
            let distance_squared = Self::euclidean_distance_squared(query, &point);
            self.stats.distance_computations += 1;
            if distance_squared <= radius_squared {
                neighbors.push((idx, distance_squared.sqrt()));
            }
        }
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(neighbors)
    }
    /// Compute squared Euclidean distance between two points
    fn euclidean_distance_squared(p1: &ArrayView1<F>, p2: &ArrayView1<F>) -> F {
        p1.iter()
            .zip(p2.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
    }
    /// Get performance statistics
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }
    /// Clear the query cache
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
    }
    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.stats.total_queries == 0 {
            0.0
        } else {
            self.stats.cache_hits as f64 / self.stats.total_queries as f64
        }
    }
    /// Get average query time in microseconds
    pub fn average_query_time_us(&self) -> f64 {
        if self.stats.total_queries == 0 {
            0.0
        } else {
            self.stats.total_query_time_us as f64 / self.stats.total_queries as f64
        }
    }
}
/// KD-tree node
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct KdTreeNode<F: Float> {
    /// Point index in original data
    point_idx: Option<usize>,
    /// Splitting dimension
    split_dim: usize,
    /// Splitting value
    split_value: F,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Bounding box for pruning
    bbox_min: Vec<F>,
    bbox_max: Vec<F>,
}
/// Performance statistics for search operations
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Total number of queries processed
    pub total_queries: usize,
    /// Total query time in microseconds
    pub total_query_time_us: u64,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of distance computations
    pub distance_computations: usize,
    /// Number of nodes visited (for tree-based methods)
    pub nodes_visited: usize,
}
/// Ball tree node
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BallTreeNode<F: Float> {
    /// Point indices in this ball
    point_indices: Vec<usize>,
    /// Center of the ball
    center: Vec<F>,
    /// Radius of the ball
    radius: F,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Whether this is a leaf node
    is_leaf: bool,
}
/// Simple KD-tree implementation for low-dimensional data
#[derive(Debug)]
pub struct KdTreeIndex<F: Float> {
    /// Tree nodes
    nodes: Vec<KdTreeNode<F>>,
    /// Root node index
    root: usize,
    /// Number of dimensions
    dimensions: usize,
    /// Original points data
    points: Array2<F>,
}
/// Implementation of KD-tree for efficient nearest neighbor search
impl<F: Float + FromPrimitive> KdTreeIndex<F> {
    /// Build a new KD-tree from points
    pub fn new(points: &Array2<F>) -> InterpolateResult<Self> {
        if points.is_empty() {
            return Err(InterpolateError::invalid_input("Points array is empty"));
        }
        let dimensions = points.ncols();
        let num_points = points.nrows();
        let mut indices: Vec<usize> = (0..num_points).collect();
        let mut nodes = Vec::new();
        let root = Self::build_tree_recursive(points, &mut indices, 0, dimensions, &mut nodes)?;
        Ok(Self {
            nodes,
            root,
            dimensions,
            points: points.to_owned(),
        })
    }
    /// Recursively build the KD-tree
    fn build_tree_recursive(
        points: &Array2<F>,
        indices: &mut [usize],
        depth: usize,
        dimensions: usize,
        nodes: &mut Vec<KdTreeNode<F>>,
    ) -> InterpolateResult<usize> {
        if indices.is_empty() {
            return Err(InterpolateError::invalid_input(
                "Empty indices in tree building",
            ));
        }
        let split_dim = depth % dimensions;
        indices.sort_by(|&a, &b| {
            let val_a = points[[a, split_dim]];
            let val_b = points[[b, split_dim]];
            val_a.partial_cmp(&val_b).unwrap_or(Ordering::Equal)
        });
        let median_idx = indices.len() / 2;
        let point_idx = indices[median_idx];
        let split_value = points[[point_idx, split_dim]];
        let mut bbox_min = vec![F::infinity(); dimensions];
        let mut bbox_max = vec![F::neg_infinity(); dimensions];
        for &idx in indices.iter() {
            for d in 0..dimensions {
                let val = points[[idx, d]];
                bbox_min[d] = bbox_min[d].min(val);
                bbox_max[d] = bbox_max[d].max(val);
            }
        }
        let node_idx = nodes.len();
        nodes.push(KdTreeNode {
            point_idx: Some(point_idx),
            split_dim,
            split_value,
            left: None,
            right: None,
            bbox_min,
            bbox_max,
        });
        if median_idx > 0 {
            let left_idx = Self::build_tree_recursive(
                points,
                &mut indices[..median_idx],
                depth + 1,
                dimensions,
                nodes,
            )?;
            nodes[node_idx].left = Some(left_idx);
        }
        if median_idx + 1 < indices.len() {
            let right_idx = Self::build_tree_recursive(
                points,
                &mut indices[median_idx + 1..],
                depth + 1,
                dimensions,
                nodes,
            )?;
            nodes[node_idx].right = Some(right_idx);
        }
        Ok(node_idx)
    }
    /// Find k nearest neighbors
    pub fn k_nearest_neighbors(
        &self,
        query: &ArrayView1<F>,
        k: usize,
        stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut best = BinaryHeap::new();
        self.search_knn_recursive(self.root, query, k, &mut best, stats)?;
        let mut result = Vec::new();
        while let Some(neighbor) = best.pop() {
            result.push((neighbor.index, neighbor.distance.0));
        }
        result.reverse();
        Ok(result)
    }
    /// Recursive k-nearest neighbors search
    fn search_knn_recursive(
        &self,
        node_idx: usize,
        query: &ArrayView1<F>,
        k: usize,
        best: &mut BinaryHeap<Neighbor<F>>,
        stats: &mut SearchStats,
    ) -> InterpolateResult<()> {
        stats.nodes_visited += 1;
        let node = &self.nodes[node_idx];
        if let Some(point_idx) = node.point_idx {
            let mut distance_sq = F::zero();
            for d in 0..self.dimensions {
                let diff = query[d] - self.points[[point_idx, d]];
                distance_sq = distance_sq + diff * diff;
            }
            let distance = distance_sq.sqrt();
            let neighbor = Neighbor {
                index: point_idx,
                distance: OrderedFloat(distance),
            };
            if best.len() < k {
                best.push(neighbor);
            } else if neighbor.distance < best.peek().expect("Operation failed").distance {
                best.pop();
                best.push(neighbor);
            }
        }
        let split_dim = node.split_dim;
        let query_val = query[split_dim];
        let (first, second) = if query_val < node.split_value {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(child_idx) = first {
            self.search_knn_recursive(child_idx, query, k, best, stats)?;
        }
        if let Some(child_idx) = second {
            let worst_dist = best.peek().map(|n| n.distance.0).unwrap_or(F::infinity());
            let axis_dist = (query_val - node.split_value).abs();
            if best.len() < k || axis_dist < worst_dist {
                self.search_knn_recursive(child_idx, query, k, best, stats)?;
            }
        }
        Ok(())
    }
    /// Find all neighbors within a given radius
    pub fn radius_neighbors(
        &self,
        query: &ArrayView1<F>,
        radius: F,
        stats: &mut SearchStats,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let mut result = Vec::new();
        self.search_radius_recursive(self.root, query, radius, &mut result, stats)?;
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(result)
    }
    /// Recursive radius search
    fn search_radius_recursive(
        &self,
        node_idx: usize,
        query: &ArrayView1<F>,
        radius: F,
        result: &mut Vec<(usize, F)>,
        stats: &mut SearchStats,
    ) -> InterpolateResult<()> {
        stats.nodes_visited += 1;
        let node = &self.nodes[node_idx];
        if let Some(point_idx) = node.point_idx {
            let mut distance_sq = F::zero();
            for d in 0..self.dimensions {
                let diff = query[d] - self.points[[point_idx, d]];
                distance_sq = distance_sq + diff * diff;
            }
            let distance = distance_sq.sqrt();
            if distance <= radius {
                result.push((point_idx, distance));
            }
        }
        let split_dim = node.split_dim;
        let query_val = query[split_dim];
        if let Some(left_idx) = node.left {
            if query_val - radius <= node.split_value {
                self.search_radius_recursive(left_idx, query, radius, result, stats)?;
            }
        }
        if let Some(right_idx) = node.right {
            if query_val + radius >= node.split_value {
                self.search_radius_recursive(right_idx, query, radius, result, stats)?;
            }
        }
        Ok(())
    }
}
/// Wrapper for floating point values to make them orderable
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedFloat<F: Float>(pub(super) F);
/// Configuration for enhanced nearest neighbor search
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of neighbors to search for
    pub max_neighbors: usize,
    /// Search radius (for radius-based searches)
    pub radius: Option<f64>,
    /// Approximation factor (1.0 = exact, >1.0 = approximate)
    pub approximation_factor: f64,
    /// Enable parallel search for large query sets
    pub parallel_search: bool,
    /// Number of threads for parallel search (None = auto)
    pub num_threads: Option<usize>,
    /// Use adaptive indexing for dynamic data
    pub adaptive_indexing: bool,
    /// Cache search results for repeated queries
    pub cache_results: bool,
}
/// Types of spatial indices available
#[derive(Debug, Clone, PartialEq)]
pub enum IndexType {
    /// Brute force search (O(n) per query)
    BruteForce,
    /// KD-tree (efficient for low dimensions)
    KdTree,
    /// Ball tree (efficient for high dimensions)
    BallTree,
    /// Locality Sensitive Hashing (approximate)
    LSH,
    /// Adaptive hybrid approach
    Adaptive,
}
/// Query key for caching (using approximate floating point comparison)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QueryKey {
    /// Quantized coordinates for hashing
    coords: Vec<i64>,
    /// Number of neighbors requested
    k: usize,
    /// Quantized radius (if applicable)
    radius: Option<i64>,
}
impl QueryKey {
    /// Create a new query key from coordinates
    fn from_coords<F: Float + FromPrimitive>(coords: &[F], k: usize, radius: Option<F>) -> Self {
        const QUANTIZATION_FACTOR: f64 = 1000.0;
        let quantized_coords: Vec<i64> = coords
            .iter()
            .map(|&x| (x.to_f64().unwrap_or(0.0) * QUANTIZATION_FACTOR).round() as i64)
            .collect();
        let quantized_radius =
            radius.map(|r| (r.to_f64().unwrap_or(0.0) * QUANTIZATION_FACTOR).round() as i64);
        Self {
            coords: quantized_coords,
            k,
            radius: quantized_radius,
        }
    }
}
