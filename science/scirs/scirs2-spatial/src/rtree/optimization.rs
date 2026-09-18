use crate::error::SpatialResult;
use crate::rtree::node::{Entry, Node, RTree};
use crate::rtree::Rectangle;
use scirs2_core::ndarray::Array1;

impl<T: Clone> RTree<T> {
    /// Optimize the R-tree by rebuilding it with the current data
    ///
    /// This can significantly improve query performance by reducing overlap
    /// and creating a more balanced tree.
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing nothing if successful
    pub fn optimize(&mut self) -> SpatialResult<()> {
        // Collect all data points
        let data_points = self.collect_all_data_points()?;

        if data_points.is_empty() {
            return Ok(());
        }

        // Create a new, empty R-tree
        // These parameters are not used in this method but would be used
        // if we were creating a new R-tree
        // let _ndim = self.ndim();
        // let _min_entries = self.min_entries;
        // let _max_entries = self.maxentries;

        // Save current size to check at the end
        let size = self.size();

        // Clear current tree
        self.clear();

        // Re-insert all data points (bulk loading would be more efficient)
        for (point, data, _) in data_points {
            self.insert(point, data)?;
        }

        // Verify integrity
        assert_eq!(self.size(), size, "Size mismatch after optimization");

        Ok(())
    }

    /// Collect all data points in the R-tree
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing a vector of (point, data, index) tuples
    fn collect_all_data_points(&self) -> SpatialResult<Vec<(Array1<f64>, T, usize)>> {
        let mut points = Vec::new();
        self.collect_data_points_recursive(&self.root, &mut points)?;
        Ok(points)
    }

    /// Recursively collect data points from a node
    #[allow(clippy::only_used_in_recursion)]
    fn collect_data_points_recursive(
        &self,
        node: &Node<T>,
        points: &mut Vec<(Array1<f64>, T, usize)>,
    ) -> SpatialResult<()> {
        for entry in &node.entries {
            match entry {
                Entry::Leaf { mbr, data, index } => {
                    // For leaf entries, the MBR should be a point (min == max)
                    points.push((mbr.min.clone(), data.clone(), *index));
                }
                Entry::NonLeaf { child, .. } => {
                    // For non-leaf entries, recursively collect from children
                    self.collect_data_points_recursive(child, points)?;
                }
            }
        }
        Ok(())
    }

    /// Perform bulk loading of the R-tree with sorted data points
    ///
    /// This is more efficient than inserting points one by one.
    ///
    /// # Arguments
    ///
    /// * `points` - A vector of (point, data) pairs to insert
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing a new R-tree built from the data points
    pub fn bulk_load(
        ndim: usize,
        min_entries: usize,
        max_entries: usize,
        points: Vec<(Array1<f64>, T)>,
    ) -> SpatialResult<Self> {
        // Create a new, empty R-tree
        let mut rtree = RTree::new(ndim, min_entries, max_entries)?;

        if points.is_empty() {
            return Ok(rtree);
        }

        // Implement Sort-Tile-Recursive (STR) bulk loading algorithm

        // Validate all points have correct dimensions
        for (i, (point, _)) in points.iter().enumerate() {
            if point.len() != ndim {
                return Err(crate::error::SpatialError::DimensionError(format!(
                    "Point at index {} has dimension {} but tree dimension is {}",
                    i,
                    point.len(),
                    ndim
                )));
            }
        }

        // Convert points to leaf _entries
        let mut entries: Vec<Entry<T>> = points
            .into_iter()
            .enumerate()
            .map(|(index, (point, data))| Entry::Leaf {
                mbr: Rectangle::from_point(&point.view()),
                data,
                index,
            })
            .collect();

        // Store the number of points for size tracking
        let num_points = entries.len();

        // Build the tree recursively
        rtree.root = rtree.str_build_node(&mut entries, 0)?;
        rtree.root._isleaf =
            rtree.root.entries.is_empty() || matches!(rtree.root.entries[0], Entry::Leaf { .. });

        // Update tree height
        let height = rtree.calculate_height(&rtree.root);
        for _ in 1..height {
            rtree.increment_height();
        }

        // Update the tree size to reflect the number of data points loaded
        for _ in 0..num_points {
            rtree.increment_size();
        }

        Ok(rtree)
    }

    /// Build a node using the STR algorithm
    fn str_build_node(&self, entries: &mut Vec<Entry<T>>, level: usize) -> SpatialResult<Node<T>> {
        let n = entries.len();

        if n == 0 {
            return Ok(Node::new(level == 0, level));
        }

        // If we can fit all _entries in one node, create it
        if n <= self.maxentries {
            let mut node = Node::new(level == 0, level);
            node.entries = std::mem::take(entries);
            return Ok(node);
        }

        // Calculate the number of leaf nodes needed
        let leaf_capacity = self.maxentries;
        let num_leaves = n.div_ceil(leaf_capacity);

        // Calculate the number of slices along each dimension
        let slice_count = (num_leaves as f64).powf(1.0 / self.ndim() as f64).ceil() as usize;

        // Sort _entries by the first dimension
        let dim = level % self.ndim();
        entries.sort_by(|a, b| {
            let a_center = (a.mbr().min[dim] + a.mbr().max[dim]) / 2.0;
            let b_center = (b.mbr().min[dim] + b.mbr().max[dim]) / 2.0;
            a_center
                .partial_cmp(&b_center)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create child nodes
        let mut children = Vec::new();
        let entries_per_slice = n.div_ceil(slice_count);

        for i in 0..slice_count {
            let start = i * entries_per_slice;
            let end = ((i + 1) * entries_per_slice).min(n);

            if start >= n {
                break;
            }

            let mut slice_entries: Vec<Entry<T>> = entries[start..end].to_vec();

            // Recursively build child nodes
            if level == 0 {
                // These are leaf entries, group them into leaf nodes
                while !slice_entries.is_empty() {
                    let mut node = Node::new(true, 0);
                    let take_count = slice_entries.len().min(self.maxentries);
                    node.entries = slice_entries.drain(..take_count).collect();

                    if let Ok(Some(mbr)) = node.mbr() {
                        children.push(Entry::NonLeaf {
                            mbr,
                            child: Box::new(node),
                        });
                    }
                }
            } else {
                // Build non-leaf nodes recursively
                let child_node = self.str_build_node(&mut slice_entries, level - 1)?;
                if let Ok(Some(mbr)) = child_node.mbr() {
                    children.push(Entry::NonLeaf {
                        mbr,
                        child: Box::new(child_node),
                    });
                }
            }
        }

        // Clear the input _entries as they've been moved to children
        entries.clear();

        // If we have too many children, build another level
        if children.len() > self.maxentries {
            self.str_build_node(&mut children, level + 1)
        } else {
            let mut node = Node::new(false, level + 1);
            node.entries = children;
            Ok(node)
        }
    }

    /// Calculate the height of the tree
    #[allow(clippy::only_used_in_recursion)]
    fn calculate_height(&self, node: &Node<T>) -> usize {
        if node._isleaf {
            1
        } else if let Some(Entry::NonLeaf { child, .. }) = node.entries.first() {
            1 + self.calculate_height(child)
        } else {
            1
        }
    }

    /// Calculate the total overlap in the R-tree
    ///
    /// This is a quality metric for the tree. Lower overlap generally means
    /// better query performance.
    ///
    /// # Returns
    ///
    /// The total overlap area between all pairs of nodes at each level
    pub fn calculate_total_overlap(&self) -> SpatialResult<f64> {
        let mut total_overlap = 0.0;

        // Calculate overlap at each level, starting from the root
        let mut current_level_nodes = vec![&self.root];

        while !current_level_nodes.is_empty() {
            // Calculate overlap between nodes at this level
            for i in 0..current_level_nodes.len() - 1 {
                let node_i_mbr = match current_level_nodes[i].mbr() {
                    Ok(Some(mbr)) => mbr,
                    _ => continue,
                };

                for node_j in current_level_nodes.iter().skip(i + 1) {
                    let node_j_mbr = match node_j.mbr() {
                        Ok(Some(mbr)) => mbr,
                        _ => continue,
                    };

                    // Check if MBRs intersect
                    if node_i_mbr.intersects(&node_j_mbr)? {
                        // Calculate intersection area
                        if let Ok(intersection) = node_i_mbr.intersection(&node_j_mbr) {
                            total_overlap += intersection.area();
                        }
                    }
                }
            }

            // Move to the next level
            let mut next_level_nodes = Vec::new();
            for node in current_level_nodes {
                for entry in &node.entries {
                    if let Entry::NonLeaf { child, .. } = entry {
                        next_level_nodes.push(&**child);
                    }
                }
            }

            current_level_nodes = next_level_nodes;
        }

        Ok(total_overlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rtree_optimize() {
        // Create a new R-tree
        let mut rtree: RTree<i32> = RTree::new(2, 2, 4).expect("Operation failed");

        // Insert some points
        let points = vec![
            (array![0.0, 0.0], 0),
            (array![1.0, 0.0], 1),
            (array![0.0, 1.0], 2),
            (array![1.0, 1.0], 3),
            (array![0.5, 0.5], 4),
            (array![2.0, 2.0], 5),
            (array![3.0, 3.0], 6),
            (array![4.0, 4.0], 7),
            (array![5.0, 5.0], 8),
            (array![6.0, 6.0], 9),
        ];

        for (point, value) in points {
            rtree.insert(point, value).expect("Operation failed");
        }

        // Optimize the tree
        rtree.optimize().expect("Operation failed");

        // Check that all data is still present
        assert_eq!(rtree.size(), 10);

        // Try to search for a point
        let results = rtree
            .search_range(&array![0.4, 0.4].view(), &array![0.6, 0.6].view())
            .expect("Operation failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 4);
    }

    #[test]
    fn test_rtree_bulk_load() {
        // Create points
        let points = vec![
            (array![0.0, 0.0], 0),
            (array![1.0, 0.0], 1),
            (array![0.0, 1.0], 2),
            (array![1.0, 1.0], 3),
            (array![0.5, 0.5], 4),
            (array![2.0, 2.0], 5),
            (array![3.0, 3.0], 6),
            (array![4.0, 4.0], 7),
            (array![5.0, 5.0], 8),
            (array![6.0, 6.0], 9),
        ];

        // Bulk load
        let rtree = RTree::bulk_load(2, 2, 4, points).expect("Operation failed");

        // Check that all data is present
        assert_eq!(rtree.size(), 10);

        // Try to search for a point
        let results = rtree
            .search_range(&array![0.4, 0.4].view(), &array![0.6, 0.6].view())
            .expect("Operation failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 4);
    }
}
