# Set-Based Distances

The `set_distance` module in `scirs2-spatial` provides metrics for measuring distances between sets of points rather than individual points. These distance measures are particularly useful for comparing shapes, point clouds, or spatial distributions.

## Hausdorff Distance

The Hausdorff distance is one of the most widely used measures for comparing two sets of points. It captures the maximum distance from any point in one set to the closest point in the other set.

Formally, the Hausdorff distance between two sets A and B is defined as:

```
H(A, B) = max(h(A, B), h(B, A))
```

where `h(A, B)` is the directed Hausdorff distance:

```
h(A, B) = max_{a in A} min_{b in B} ||a - b||
```

```rust
use scirs2_spatial::set_distance::hausdorff_distance;
use ndarray::array;

// Create two point sets
let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];

// Compute the Hausdorff distance
let dist = hausdorff_distance(&set1.view(), &set2.view());
println!("Hausdorff distance: {}", dist);
```

### Directed Hausdorff Distance

The module also provides the directed Hausdorff distance function, which returns additional information about which points realize the distance.

```rust
use scirs2_spatial::set_distance::directed_hausdorff;
use ndarray::array;

// Create two point sets
let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];

// Compute the directed Hausdorff distance from set1 to set2
let (dist, idx1, idx2) = directed_hausdorff(&set1.view(), &set2.view(), Some(42));
println!("Directed Hausdorff distance: {}", dist);
println!("Point in set1: {:?}", set1.row(idx1));
println!("Corresponding point in set2: {:?}", set2.row(idx2));
```

## Wasserstein Distance (Earth Mover's Distance)

The Wasserstein distance, also known as the Earth Mover's Distance (EMD), measures the minimum amount of "work" required to transform one distribution into another. It's a more refined measure than the Hausdorff distance as it considers all points in both sets rather than just the extreme cases.

```rust
use scirs2_spatial::set_distance::wasserstein_distance;
use ndarray::array;

// Create two point distributions
let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];

// Compute the approximate Wasserstein distance
match wasserstein_distance(&set1.view(), &set2.view()) {
    Ok(dist) => println!("Wasserstein distance: {}", dist),
    Err(e) => println!("Error: {}", e),
}
```

## Gromov-Hausdorff Distance

The Gromov-Hausdorff distance measures the similarity between metric spaces. Unlike the standard Hausdorff distance which compares point sets directly, this distance considers the intrinsic geometry of the spaces.

```rust
use scirs2_spatial::set_distance::gromov_hausdorff_distance;
use ndarray::array;

// Create two point sets that have similar shapes but different scales
let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
let set2 = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];

// Compute the Gromov-Hausdorff distance
let dist = gromov_hausdorff_distance(&set1.view(), &set2.view());
println!("Gromov-Hausdorff distance: {}", dist);
```

## Applications

Set-based distances are useful in many applications:

1. **Shape matching**: Compare the similarity of geometric shapes
2. **Image comparison**: Measure distances between sets of features in images
3. **Point cloud registration**: Align 3D scans of objects
4. **Clustering validation**: Evaluate the quality of clustering algorithms
5. **Pattern recognition**: Identify similar patterns across different datasets

## Performance Considerations

The implementation of the set-based distances includes several optimizations:

1. **Shuffling**: Points are randomly shuffled to improve the likelihood of early termination in search loops
2. **Early termination**: When computing directed Hausdorff distance, the algorithm can terminate early if it finds a point that is closer than the current maximum
3. **Squared distances**: Distance calculations use squared distances internally to avoid expensive square root operations until the final result

For very large point sets, additional optimizations may be needed, such as spatial indexing structures (e.g., KD-trees) to accelerate nearest neighbor searches.