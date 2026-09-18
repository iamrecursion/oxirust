use crate::error::SpatialResult;
use crate::rtree::node::{Entry, Node, RTree, Rectangle};
use scirs2_core::ndarray::Array1;
use std::collections::HashSet;

impl<T: Clone> RTree<T> {
    /// Insert a data point into the R-tree
    ///
    /// # Arguments
    ///
    /// * `point` - The point coordinates
    /// * `data` - The data associated with the point
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing nothing if successful, or an error if the point has invalid dimensions
    pub fn insert(&mut self, point: Array1<f64>, data: T) -> SpatialResult<()> {
        if point.len() != self.ndim() {
            return Err(crate::error::SpatialError::DimensionError(format!(
                "Point dimension {} does not match RTree dimension {}",
                point.len(),
                self.ndim()
            )));
        }

        // Create a leaf entry for the point
        let mbr = Rectangle::from_point(&point.view());
        let entry = Entry::Leaf {
            mbr,
            data,
            index: self.size(),
        };

        // Insert the entry and handle node splits
        self.insert_entry(&entry, 0)?;

        // Increment the size after successful insertion
        self.increment_size();

        Ok(())
    }

    /// Insert a rectangle into the R-tree
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum coordinates of the rectangle
    /// * `max` - The maximum coordinates of the rectangle
    /// * `data` - The data associated with the rectangle
    ///
    /// # Returns
    ///
    /// A `SpatialResult` containing nothing if successful, or an error if the rectangle has invalid dimensions
    pub fn insert_rectangle(
        &mut self,
        min: Array1<f64>,
        max: Array1<f64>,
        data: T,
    ) -> SpatialResult<()> {
        if min.len() != self.ndim() {
            return Err(crate::error::SpatialError::DimensionError(format!(
                "Min coordinate dimension {} does not match RTree dimension {}",
                min.len(),
                self.ndim()
            )));
        }

        if max.len() != self.ndim() {
            return Err(crate::error::SpatialError::DimensionError(format!(
                "Max coordinate dimension {} does not match RTree dimension {}",
                max.len(),
                self.ndim()
            )));
        }

        // Create a leaf entry for the rectangle
        let mbr = Rectangle::new(min, max)?;
        let entry = Entry::Leaf {
            mbr,
            data,
            index: self.size(),
        };

        // Insert the entry and handle node splits
        self.insert_entry(&entry, 0)?;

        // Increment the size after successful insertion
        self.increment_size();

        Ok(())
    }

    /// Helper function to insert an entry into the tree
    pub(crate) fn insert_entry(
        &mut self,
        entry: &Entry<T>,
        level: usize,
    ) -> SpatialResult<Option<Node<T>>> {
        // For leaf entries, we should insert at level 0
        // If the root is a leaf node (level 0), insert directly
        if self.root._isleaf && level == 0 {
            self.root.entries.push(entry.clone());

            // If the node overflows, split it
            if self.root.size() > self.maxentries {
                // Take ownership of the root temporarily
                let mut root = std::mem::replace(&mut self.root, Node::new(true, 0));
                let (node1, node2) = self.split_node(&mut root)?;

                // Create a new root
                let mut new_root = Node::new(false, 1);

                // Add both split nodes as children
                if let Ok(Some(mbr1)) = node1.mbr() {
                    new_root.entries.push(Entry::NonLeaf {
                        mbr: mbr1,
                        child: Box::new(node1),
                    });
                }

                if let Ok(Some(mbr2)) = node2.mbr() {
                    new_root.entries.push(Entry::NonLeaf {
                        mbr: mbr2,
                        child: Box::new(node2),
                    });
                }

                self.root = new_root;
                self.increment_height();
            }

            return Ok(None);
        }

        // If root is not a leaf, we need to insert recursively
        if !self.root._isleaf {
            // Choose the best subtree to insert into
            let subtree_index = self.choose_subtree(&self.root, entry.mbr(), level)?;

            // Get the subtree
            let child: &mut Box<Node<T>> = match &mut self.root.entries[subtree_index] {
                Entry::NonLeaf { child, .. } => child,
                Entry::Leaf { .. } => {
                    return Err(crate::error::SpatialError::ComputationError(
                        "Expected a non-leaf entry".into(),
                    ))
                }
            };

            // Get child as mutable raw pointer to avoid borrowing conflicts
            let child_ptr = Box::as_mut(child) as *mut Node<T>;

            // Recursively insert into the subtree
            let maybe_split =
                unsafe { self.insert_entry_recursive(entry, level, &mut *child_ptr) }?;

            // If the child was split, add the new node as a sibling
            if let Some(split_node) = maybe_split {
                // Create a new entry for the split node
                if let Ok(Some(mbr)) = split_node.mbr() {
                    self.root.entries.push(Entry::NonLeaf {
                        mbr,
                        child: Box::new(split_node),
                    });

                    // If the node overflows, split it
                    if self.root.size() > self.maxentries {
                        // Take ownership of the root temporarily
                        let mut root = std::mem::replace(&mut self.root, Node::new(false, 1));
                        let root_level = root.level;
                        let (node1, node2) = self.split_node(&mut root)?;

                        // Create a new root
                        let mut new_root = Node::new(false, root_level + 1);

                        // Add both split nodes as children
                        if let Ok(Some(mbr1)) = node1.mbr() {
                            new_root.entries.push(Entry::NonLeaf {
                                mbr: mbr1,
                                child: Box::new(node1),
                            });
                        }

                        if let Ok(Some(mbr2)) = node2.mbr() {
                            new_root.entries.push(Entry::NonLeaf {
                                mbr: mbr2,
                                child: Box::new(node2),
                            });
                        }

                        self.root = new_root;
                        self.increment_height();
                    }
                }
            }

            // Update the MBR of the parent entry
            if let Entry::NonLeaf { mbr, child } = &mut self.root.entries[subtree_index] {
                if let Ok(Some(child_mbr)) = child.mbr() {
                    *mbr = child_mbr;
                }
            }
        }

        Ok(None)
    }

    /// Recursively insert an entry into a node
    pub(crate) fn insert_entry_recursive(
        &mut self,
        entry: &Entry<T>,
        level: usize,
        node: &mut Node<T>,
    ) -> SpatialResult<Option<Node<T>>> {
        // If we're at the target level, add the entry to this node
        if level == node.level {
            node.entries.push(entry.clone());

            // If the node overflows, split it
            if node.size() > self.maxentries {
                let (new_node, split_node) = self.split_node(node)?;
                *node = new_node;
                return Ok(Some(split_node));
            }

            return Ok(None);
        }

        // Choose the best subtree to insert into
        let subtree_index = self.choose_subtree(node, entry.mbr(), level)?;

        // Get the subtree
        let child: &mut Box<Node<T>> = match &mut node.entries[subtree_index] {
            Entry::NonLeaf { child, .. } => child,
            Entry::Leaf { .. } => {
                return Err(crate::error::SpatialError::ComputationError(
                    "Expected a non-leaf entry".into(),
                ))
            }
        };

        // Get child as mutable raw pointer to avoid borrowing conflicts
        let child_ptr = Box::as_mut(child) as *mut Node<T>;

        // Recursively insert into the subtree
        let maybe_split = unsafe { self.insert_entry_recursive(entry, level, &mut *child_ptr) }?;

        // If the child was split, add the new node as a sibling
        if let Some(split_node) = maybe_split {
            // Create a new entry for the split node
            if let Ok(Some(mbr)) = split_node.mbr() {
                node.entries.push(Entry::NonLeaf {
                    mbr,
                    child: Box::new(split_node),
                });

                // If the node overflows, split it
                if node.size() > self.maxentries {
                    let (new_node, split_node) = self.split_node(node)?;
                    *node = new_node;
                    return Ok(Some(split_node));
                }
            }
        }

        // Update the MBR of the parent entry
        if let Entry::NonLeaf { mbr, child } = &mut node.entries[subtree_index] {
            if let Ok(Some(child_mbr)) = child.mbr() {
                *mbr = child_mbr;
            }
        }

        Ok(None)
    }

    /// Choose the best subtree to insert an entry
    pub(crate) fn choose_subtree(
        &self,
        node: &Node<T>,
        mbr: &Rectangle,
        level: usize,
    ) -> SpatialResult<usize> {
        let mut min_enlargement = f64::MAX;
        let mut min_area = f64::MAX;
        let mut chosen_index = 0;

        // If this is a leaf node, we shouldn't be choosing a subtree
        if node._isleaf {
            return Err(crate::error::SpatialError::ComputationError(
                "Cannot choose subtree in a leaf node".into(),
            ));
        }

        for (i, entry) in node.entries.iter().enumerate() {
            let entry_mbr = entry.mbr();

            // Calculate the enlargement needed to include the new entry
            let enlargement = entry_mbr.enlargement_area(mbr)?;

            // Choose the entry with the minimum enlargement
            if enlargement < min_enlargement
                || (enlargement == min_enlargement && entry_mbr.area() < min_area)
            {
                min_enlargement = enlargement;
                min_area = entry_mbr.area();
                chosen_index = i;
            }
        }

        Ok(chosen_index)
    }

    /// Split a node that has more than max_entries
    pub(crate) fn split_node(&self, node: &mut Node<T>) -> SpatialResult<(Node<T>, Node<T>)> {
        // Create two new nodes: the original (which will be replaced) and the split node
        let mut node1 = Node::new(node._isleaf, node.level);
        let mut node2 = Node::new(node._isleaf, node.level);

        // Choose two seed entries to initialize the split
        let (seed1, seed2) = self.choose_split_seeds(node)?;

        // Create two groups with the seeds
        node1.entries.push(node.entries[seed1].clone());
        node2.entries.push(node.entries[seed2].clone());

        // Use a set to track which entries have been assigned
        let mut assigned = HashSet::new();
        assigned.insert(seed1);
        assigned.insert(seed2);

        // Get the MBRs of the two groups
        let mut mbr1 = node1.entries[0].mbr().clone();
        let mut mbr2 = node2.entries[0].mbr().clone();

        // Assign the remaining entries to the groups using the quadratic cost algorithm
        while assigned.len() < node.size() {
            // If one group needs to be filled to meet the minimum entries requirement
            let remaining = node.size() - assigned.len();
            if node1.size() + remaining == self.min_entries {
                // Assign all remaining entries to group 1
                for i in 0..node.size() {
                    if !assigned.contains(&i) {
                        node1.entries.push(node.entries[i].clone());
                        mbr1 = mbr1.enlarge(node.entries[i].mbr())?;
                    }
                }
                break;
            } else if node2.size() + remaining == self.min_entries {
                // Assign all remaining entries to group 2
                for i in 0..node.size() {
                    if !assigned.contains(&i) {
                        node2.entries.push(node.entries[i].clone());
                        mbr2 = mbr2.enlarge(node.entries[i].mbr())?;
                    }
                }
                break;
            }

            // Find the entry that has the maximum difference in enlargement
            let mut max_diff = -f64::MAX;
            let mut chosen_index = 0;
            let mut add_to_group1 = true;

            for i in 0..node.size() {
                if assigned.contains(&i) {
                    continue;
                }

                // Calculate the enlargement needed for both groups
                let entry_mbr = node.entries[i].mbr();
                let enlargement1 = mbr1.enlargement_area(entry_mbr)?;
                let enlargement2 = mbr2.enlargement_area(entry_mbr)?;

                // Calculate the difference in enlargement
                let diff = (enlargement1 - enlargement2).abs();
                if diff > max_diff {
                    max_diff = diff;
                    chosen_index = i;
                    add_to_group1 = enlargement1 < enlargement2;
                }
            }

            // Add the chosen entry to the preferred group
            if add_to_group1 {
                node1.entries.push(node.entries[chosen_index].clone());
                mbr1 = mbr1.enlarge(node.entries[chosen_index].mbr())?;
            } else {
                node2.entries.push(node.entries[chosen_index].clone());
                mbr2 = mbr2.enlarge(node.entries[chosen_index].mbr())?;
            }

            assigned.insert(chosen_index);
        }

        // Replace the old node with the first group
        *node = node1;

        Ok((node.clone(), node2))
    }

    /// Choose two entries to be the seeds for node splitting
    pub(crate) fn choose_split_seeds(&self, node: &Node<T>) -> SpatialResult<(usize, usize)> {
        let mut max_waste = -f64::MAX;
        let mut seed1 = 0;
        let mut seed2 = 0;

        // Find the two entries that would waste the most area if put together
        for i in 0..node.size() - 1 {
            for j in i + 1..node.size() {
                let mbr_i = node.entries[i].mbr();
                let mbr_j = node.entries[j].mbr();

                // Calculate the area of the MBR containing both entries
                let combined_mbr = mbr_i.enlarge(mbr_j)?;
                let combined_area = combined_mbr.area();

                // Calculate the wasted area
                let waste = combined_area - mbr_i.area() - mbr_j.area();

                if waste > max_waste {
                    max_waste = waste;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }

        Ok((seed1, seed2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rtree_insert_and_search() {
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

        // Check the size
        assert_eq!(rtree.size(), 10);
        assert!(!rtree.is_empty());
    }

    #[test]
    fn test_rtree_insert_rectangle() {
        // Create a new R-tree
        let mut rtree: RTree<&str> = RTree::new(2, 2, 4).expect("Operation failed");

        // Insert some rectangles
        let rectangles = vec![
            (array![0.0, 0.0], array![1.0, 1.0], "A"),
            (array![2.0, 2.0], array![3.0, 3.0], "B"),
            (array![1.5, 0.0], array![2.5, 1.0], "C"),
            (array![0.0, 1.5], array![1.0, 2.5], "D"),
        ];

        for (min, max, value) in rectangles {
            rtree
                .insert_rectangle(min, max, value)
                .expect("Operation failed");
        }

        // Check the size
        assert_eq!(rtree.size(), 4);

        // Test searching for rectangles that overlap with a query range
        let results = rtree
            .search_range(&array![0.5, 0.5].view(), &array![2.0, 2.0].view())
            .expect("Operation failed");

        // Should find all rectangles A, B, C, and D (B overlaps at boundary point (2.0, 2.0))
        assert_eq!(results.len(), 4);

        // Verify that the results contain all expected values
        let result_values: Vec<&str> = results.iter().map(|(_, v)| *v).collect();
        assert!(result_values.contains(&"A"));
        assert!(result_values.contains(&"B")); // B overlaps at boundary
        assert!(result_values.contains(&"C"));
        assert!(result_values.contains(&"D"));
    }
}
