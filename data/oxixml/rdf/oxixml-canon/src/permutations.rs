//! Lazy generation of the gossip-path permutations explored by
//! [hash-n-degree-quads](https://www.w3.org/TR/rdf-canon/#hash-nd-quads) step 5.4.
//!
//! The permutations are produced one at a time, in code point order, so that the caller can
//! charge the [`WorkBudget`](crate::WorkBudget) *before* each permutation is examined. Generating
//! them eagerly would allocate a factorial amount of memory before any budget check could fire,
//! defeating the poison-dataset mitigation the spec's
//! [security considerations](https://www.w3.org/TR/rdf-canon/#security-considerations) ask for.

/// An iterator over every distinct permutation of a list, in ascending code point order.
///
/// The input is sorted on construction, therefore the first item yielded is the smallest
/// permutation and the last is the largest. Equal elements are interchangeable, so only
/// *distinct* permutations are produced: a list such as `["e0", "e0"]` yields the single
/// permutation `["e0", "e0"]` rather than two identical ones. This matches the observable
/// behaviour of the specification (which permutes the multiset) while avoiding redundant work.
pub(crate) struct Permutations<T> {
    items: Vec<T>,
    started: bool,
    exhausted: bool,
}

impl<T: Ord> Permutations<T> {
    /// Creates an iterator over every distinct permutation of `items`.
    pub(crate) fn new(mut items: Vec<T>) -> Self {
        items.sort();
        Self {
            items,
            started: false,
            exhausted: false,
        }
    }
}

impl<T: Ord + Clone> Iterator for Permutations<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        if !self.started {
            self.started = true;
            return Some(self.items.clone());
        }
        if next_permutation(&mut self.items) {
            Some(self.items.clone())
        } else {
            self.exhausted = true;
            None
        }
    }
}

/// Rewrites `items` with the next permutation in ascending order, returning `false` (and leaving
/// `items` sorted ascending) when the current permutation is already the largest one.
fn next_permutation<T: Ord>(items: &mut [T]) -> bool {
    if items.len() < 2 {
        return false;
    }
    // Find the longest non-increasing suffix; `pivot` is the element just before it.
    let mut pivot = items.len() - 1;
    while pivot > 0 && items[pivot - 1] >= items[pivot] {
        pivot -= 1;
    }
    if pivot == 0 {
        items.reverse();
        return false;
    }
    let pivot = pivot - 1;
    // Find the rightmost successor of the pivot in the suffix and swap it in.
    let mut successor = items.len() - 1;
    while items[successor] <= items[pivot] {
        successor -= 1;
    }
    items.swap(pivot, successor);
    items[pivot + 1..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::Permutations;

    #[test]
    fn permutes_empty_and_singleton_lists() {
        assert_eq!(
            Permutations::new(Vec::<u8>::new()).collect::<Vec<_>>(),
            vec![Vec::<u8>::new()]
        );
        assert_eq!(
            Permutations::new(vec![7]).collect::<Vec<_>>(),
            vec![vec![7]]
        );
    }

    #[test]
    fn permutes_in_ascending_order() {
        let permutations = Permutations::new(vec![2_u8, 1, 3]).collect::<Vec<_>>();
        assert_eq!(
            permutations,
            vec![
                vec![1, 2, 3],
                vec![1, 3, 2],
                vec![2, 1, 3],
                vec![2, 3, 1],
                vec![3, 1, 2],
                vec![3, 2, 1],
            ]
        );
    }

    #[test]
    fn collapses_equal_elements() {
        assert_eq!(
            Permutations::new(vec![1_u8, 1]).collect::<Vec<_>>(),
            vec![vec![1, 1]]
        );
        assert_eq!(
            Permutations::new(vec![1_u8, 1, 2]).collect::<Vec<_>>(),
            vec![vec![1, 1, 2], vec![1, 2, 1], vec![2, 1, 1]]
        );
    }

    #[test]
    fn permutation_count_is_factorial() {
        assert_eq!(Permutations::new(vec![1_u8, 2, 3, 4, 5]).count(), 120);
    }
}
