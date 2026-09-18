//! Regression tests for [`SlabPointLocator`]: the nearest segment above a
//! query point (not the topmost one) and the per-cell face labels.

use super::{Point2, Segment2D, SlabPointLocator};

fn horizontal(x0: f64, x1: f64, y: f64) -> Segment2D {
    Segment2D::new(Point2::new(x0, y), Point2::new(x1, y))
}

#[test]
fn locate_returns_the_segment_immediately_above_not_the_topmost() {
    let segs = vec![
        horizontal(0.0, 4.0, 1.0),
        horizontal(0.0, 4.0, 2.0),
        horizontal(0.0, 4.0, 3.0),
    ];
    let locator = SlabPointLocator::build(&segs, &[]);
    assert_eq!(locator.locate(Point2::new(2.0, 1.5)), Some(1));
    assert_eq!(locator.locate(Point2::new(2.0, 0.5)), Some(0));
    assert_eq!(locator.locate(Point2::new(2.0, 2.5)), Some(2));
    assert_eq!(locator.locate(Point2::new(2.0, 3.5)), None);
    // A point on a segment finds that very segment.
    assert_eq!(locator.locate(Point2::new(2.0, 2.0)), Some(1));
    assert_eq!(locator.locate(Point2::new(2.0, 1.0)), Some(0));
}

#[test]
fn locate_reports_the_original_index_for_unsorted_input() {
    // Segment indices are neither bottom-to-top nor top-to-bottom.
    let segs = vec![
        horizontal(0.0, 4.0, 3.0),
        horizontal(0.0, 4.0, 1.0),
        horizontal(0.0, 4.0, 2.0),
    ];
    let locator = SlabPointLocator::build(&segs, &[]);
    // Every slab lists its segments bottom-to-top.
    assert_eq!(locator.slab_segments, vec![vec![1, 2, 0]]);
    assert_eq!(locator.locate(Point2::new(2.0, 1.5)), Some(2));
    assert_eq!(locator.locate(Point2::new(2.0, 0.0)), Some(1));
    assert_eq!(locator.locate(Point2::new(2.0, 2.2)), Some(0));
}

#[test]
fn locate_follows_sloped_segments_at_the_query_abscissa() {
    // y = x / 2, y = 2 + x and y = 8: ordered bottom-to-top over the whole range.
    let segs = vec![
        Segment2D::new(Point2::new(0.0, 0.0), Point2::new(4.0, 2.0)),
        Segment2D::new(Point2::new(0.0, 2.0), Point2::new(4.0, 6.0)),
        horizontal(0.0, 4.0, 8.0),
    ];
    let locator = SlabPointLocator::build(&segs, &[]);
    // At x = 2 the heights are 1, 4, 8.
    assert_eq!(locator.locate(Point2::new(2.0, 1.5)), Some(1));
    assert_eq!(locator.locate(Point2::new(2.0, 0.9)), Some(0));
    assert_eq!(locator.locate(Point2::new(2.0, 4.5)), Some(2));
    // At x = 3.5 the lowest segment is at 1.75.
    assert_eq!(locator.locate(Point2::new(3.5, 1.7)), Some(0));
    assert_eq!(locator.locate(Point2::new(3.5, 1.8)), Some(1));
}

#[test]
fn locate_uses_the_segments_of_the_query_slab_only() {
    let segs = vec![
        horizontal(0.0, 2.0, 1.0),
        horizontal(2.0, 6.0, 5.0),
        horizontal(0.0, 6.0, 3.0),
    ];
    let locator = SlabPointLocator::build(&segs, &[]);
    assert_eq!(locator.slab_xs, vec![0.0, 2.0, 6.0]);
    // Slab [0, 2]: segments 0 and 2.
    assert_eq!(locator.locate(Point2::new(1.0, 0.5)), Some(0));
    assert_eq!(locator.locate(Point2::new(1.0, 2.0)), Some(2));
    assert_eq!(locator.locate(Point2::new(1.0, 3.5)), None);
    // Slab [2, 6]: segments 2 and 1.
    assert_eq!(locator.locate(Point2::new(4.0, 2.0)), Some(2));
    assert_eq!(locator.locate(Point2::new(4.0, 4.0)), Some(1));
    assert_eq!(locator.locate(Point2::new(4.0, 6.0)), None);
    // Both x-extremes belong to the structure, anything beyond does not.
    assert_eq!(locator.locate(Point2::new(6.0, 4.0)), Some(1));
    assert_eq!(locator.locate(Point2::new(0.0, 0.0)), Some(0));
    assert_eq!(locator.locate(Point2::new(6.5, 1.0)), None);
    assert_eq!(locator.locate(Point2::new(-0.5, 1.0)), None);
}

#[test]
fn vertical_segments_and_invalid_queries_are_ignored() {
    let segs = vec![
        horizontal(0.0, 6.0, 2.0),
        Segment2D::new(Point2::new(3.0, 0.0), Point2::new(3.0, 5.0)),
    ];
    let locator = SlabPointLocator::build(&segs, &[]);
    // The vertical segment only contributes a slab boundary.
    assert_eq!(locator.slab_xs, vec![0.0, 3.0, 6.0]);
    assert_eq!(locator.slab_segments, vec![vec![0], vec![0]]);
    assert_eq!(locator.locate(Point2::new(1.0, 1.0)), Some(0));
    assert_eq!(locator.locate(Point2::new(4.0, 1.0)), Some(0));
    assert_eq!(locator.locate(Point2::new(f64::NAN, 1.0)), None);
    assert_eq!(locator.locate(Point2::new(1.0, f64::NAN)), None);
    // Empty and single-abscissa structures locate nothing.
    assert_eq!(
        SlabPointLocator::default().locate(Point2::new(0.0, 0.0)),
        None
    );
    let only_vertical = SlabPointLocator::build(&segs[1..], &[]);
    assert_eq!(only_vertical.locate(Point2::new(3.0, 1.0)), None);
}

#[test]
fn face_labels_are_stored_per_cell_and_returned_by_locate_face() {
    // Segment indices 0..3 sit at y = 3, 1, 2; the face below each carries a
    // distinct label.
    let segs = vec![
        horizontal(0.0, 4.0, 3.0),
        horizontal(0.0, 4.0, 1.0),
        horizontal(0.0, 4.0, 2.0),
    ];
    let labels = [Some(30), Some(10), Some(20)];
    let locator = SlabPointLocator::build(&segs, &labels);
    // slab_faces follows the bottom-to-top order of slab_segments.
    assert_eq!(locator.slab_segments, vec![vec![1, 2, 0]]);
    assert_eq!(locator.slab_faces, vec![vec![Some(10), Some(20), Some(30)]]);
    assert_eq!(locator.locate_face(Point2::new(2.0, 0.5)), Some(10));
    assert_eq!(locator.locate_face(Point2::new(2.0, 1.5)), Some(20));
    assert_eq!(locator.locate_face(Point2::new(2.0, 2.5)), Some(30));
    // Above every segment there is no cell below a segment.
    assert_eq!(locator.locate_face(Point2::new(2.0, 3.5)), None);
    assert_eq!(locator.locate_face(Point2::new(9.0, 1.5)), None);
}

#[test]
fn missing_or_short_label_lists_leave_cells_unlabelled() {
    let segs = vec![
        horizontal(0.0, 4.0, 1.0),
        horizontal(0.0, 4.0, 2.0),
        horizontal(0.0, 4.0, 3.0),
    ];
    // Only the first segment has a label; the second is explicitly unlabelled.
    let locator = SlabPointLocator::build(&segs, &[Some(7), None]);
    assert_eq!(
        locator.slab_faces,
        vec![vec![Some(7), None, None]],
        "labels beyond the slice and None entries stay unlabelled"
    );
    assert_eq!(locator.locate_face(Point2::new(2.0, 0.5)), Some(7));
    assert_eq!(locator.locate_face(Point2::new(2.0, 1.5)), None);
    assert_eq!(locator.locate_face(Point2::new(2.0, 2.5)), None);
}
