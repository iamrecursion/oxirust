# Polygon Operations

The polygon module in `scirs2-spatial` provides a comprehensive set of operations for working with 2D polygons, enabling tasks such as point-in-polygon testing, area calculation, centroid computation, and more.

## Point-in-Polygon Testing

One of the most fundamental operations in computational geometry is determining whether a point lies inside a polygon. The polygon module provides efficient implementations for this test using the ray casting algorithm.

```rust
use scirs2_spatial::polygon::point_in_polygon;
use ndarray::array;

// Create a polygon
let polygon = array![
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

// Test if a point is inside
let inside = point_in_polygon(&[0.5, 0.5], &polygon.view());
assert!(inside);
```

## Point-on-Boundary Testing

In addition to testing if a point is inside a polygon, the module can also determine if a point lies precisely on the boundary of the polygon.

```rust
use scirs2_spatial::polygon::point_on_boundary;
use ndarray::array;

let polygon = array![
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

// Test if a point is on the boundary with a small epsilon
let on_boundary = point_on_boundary(&[0.5, 0.0], &polygon.view(), 1e-10);
assert!(on_boundary);
```

## Area and Centroid Calculation

The module provides functions to compute the area and centroid of a polygon, which are useful for many geometric applications.

```rust
use scirs2_spatial::polygon::{polygon_area, polygon_centroid};
use ndarray::array;

let polygon = array![
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

// Calculate the area
let area = polygon_area(&polygon.view());
assert_eq!(area, 1.0);

// Calculate the centroid
let centroid = polygon_centroid(&polygon.view());
assert_eq!(centroid[0], 0.5);
assert_eq!(centroid[1], 0.5);
```

## Simple Polygon Testing

A simple polygon is one that doesn't intersect itself. The module provides a function to test if a polygon is simple.

```rust
use scirs2_spatial::polygon::is_simple_polygon;
use ndarray::array;

// A simple square
let simple = array![
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

// A self-intersecting "bow tie"
let complex = array![
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
];

assert!(is_simple_polygon(&simple.view()));
assert!(!is_simple_polygon(&complex.view()));
```

## Polygon Containment

The module can determine if one polygon fully contains another polygon.

```rust
use scirs2_spatial::polygon::polygon_contains_polygon;
use ndarray::array;

// Outer polygon
let outer = array![
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 2.0],
    [0.0, 2.0],
];

// Inner polygon
let inner = array![
    [0.5, 0.5],
    [1.5, 0.5],
    [1.5, 1.5],
    [0.5, 1.5],
];

assert!(polygon_contains_polygon(&outer.view(), &inner.view()));
assert!(!polygon_contains_polygon(&inner.view(), &outer.view()));
```

## Convex Hull Computation

The module implements the Graham scan algorithm for computing the convex hull of a set of points.

```rust
use scirs2_spatial::polygon::convex_hull_graham;
use ndarray::array;

// A set of points
let points = array![
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 0.5],  // Inside point
    [1.0, 1.0],
    [0.0, 1.0],
];

let hull = convex_hull_graham(&points.view());

// The hull should have only 4 points (the corners)
assert_eq!(hull.shape()[0], 4);
```

## Future Extensions

Future improvements to the polygon module may include:

1. Boolean operations (union, difference, intersection)
2. Polygon simplification algorithms
3. Polygon triangulation
4. Support for 3D polyhedra
5. More efficient point-in-polygon testing for large polygons (e.g., using quadtrees)