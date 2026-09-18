use scirs2_core::ndarray::array;
use scirs2_spatial::voronoi::Voronoi;

#[allow(dead_code)]
fn main() {
    // Create a set of 2D points
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];

    println!("Computing Voronoi diagram for {} points", points.nrows());

    // Compute Voronoi diagram
    let vor = match Voronoi::new(&points.view(), false) {
        Ok(v) => v,
        Err(e) => {
            println!("Error computing Voronoi diagram: {e}");
            return;
        }
    };

    // Print Voronoi vertices
    println!("\nVoronoi vertices:");
    let vertices = vor.vertices();
    for i in 0..vertices.nrows() {
        println!("  Vertex {}: {:?}", i, vertices.row(i));
    }

    // Print Voronoi regions
    println!("\nVoronoi regions:");
    let regions = vor.regions();
    for (i, region) in regions.iter().enumerate() {
        println!("  Region {i}: {region:?}");
    }

    // Print point to region mapping
    println!("\nPoint to region mapping:");
    let point_region = vor.point_region();
    for i in 0..point_region.len() {
        println!("  Point {} -> Region {}", i, point_region[i]);
    }

    // Print ridge information
    println!("\nVoronoi ridges:");
    let ridge_points = vor.ridgepoints();
    let ridge_vertices = vor.ridge_vertices();
    for i in 0..ridge_points.len() {
        println!(
            "  Ridge {}: Between points {:?}, Vertices: {:?}",
            i, ridge_points[i], ridge_vertices[i]
        );
    }

    // Compute furthest-site Voronoi diagram
    println!("\nComputing furthest-site Voronoi diagram");
    let furthest_vor = match Voronoi::new(&points.view(), true) {
        Ok(v) => v,
        Err(e) => {
            println!("Error computing furthest-site Voronoi diagram: {e}");
            return;
        }
    };

    // Print furthest-site Voronoi vertices
    println!("\nFurthest-site Voronoi vertices:");
    let furthest_vertices = furthest_vor.vertices();
    for i in 0..furthest_vertices.nrows() {
        println!("  Vertex {}: {:?}", i, furthest_vertices.row(i));
    }
}
