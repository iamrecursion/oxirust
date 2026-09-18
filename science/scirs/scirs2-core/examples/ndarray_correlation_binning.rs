use scirs2_core::ndarray_ext::stats::{
    bincount, corrcoef, cov, digitize, histogram, histogram2d, quantile,
};
use scirs2_core::ndarray_ext::{array, Array2};

#[allow(dead_code)]
fn main() {
    println!("Advanced Statistical Analysis with scirs2-core\n");

    // Example 1: Working with binned data
    println!("=== Binning and Counting ===");

    // Create sample data
    let data = array![1, 2, 3, 1, 2, 1, 0, 1, 3, 2, 4, 5, 3, 2, 1, 0];
    println!("Sample data: {data:?}");

    // Basic counting
    let counts = bincount(data.view(), None, None).expect("Operation failed");
    println!("\nValue counts:");
    for (i, &count) in counts.iter().enumerate() {
        println!("  {i}: {count}");
    }

    // Digitize values into bins
    let values = array![0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 4.3, 5.5, 8.1, 3.2];
    println!("\nValues to digitize: {values:?}");

    let bins = array![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    println!("Bin edges: {bins:?}");

    let indices = digitize(values.view(), bins.view(), true, "indices").expect("Operation failed");
    println!("Bin indices: {indices:?}");

    // Create a frequency table
    println!("\nFrequency table:");
    println!("  Bin Range    | Count");
    println!("  -------------|------");

    let hist_result =
        histogram(values.view(), 5, Some((0.0, 10.0)), None).expect("Operation failed");
    let (hist, bin_edges) = hist_result;

    for i in 0..hist.len() {
        println!(
            "  {:.1} - {:.1} | {}",
            bin_edges[i],
            bin_edges[i + 1],
            "#".repeat(hist[i])
        );
    }

    // Example 2: Correlation analysis
    println!("\n\n=== Correlation Analysis ===");

    // Create correlated variables
    let n = 10;
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    // Strong positive correlation
    let y_pos = array![1.2, 1.9, 3.2, 3.9, 4.8, 6.2, 7.1, 7.9, 9.1, 10.2];
    // Strong negative correlation
    let y_neg = array![10.1, 9.2, 8.1, 7.3, 6.2, 5.1, 4.0, 3.1, 2.0, 0.9];
    // No correlation
    let y_none = array![5.5, 2.1, 8.3, 3.7, 9.2, 1.5, 6.8, 4.2, 7.9, 3.3];

    println!("x: {x:?}");
    println!("ypos: {y_pos:?}");
    println!("yneg: {y_neg:?}");
    println!("ynone: {y_none:?}");

    let corr_pos = corrcoef(x.view(), y_pos.view()).expect("Operation failed");
    let corr_neg = corrcoef(x.view(), y_neg.view()).expect("Operation failed");
    let corr_none = corrcoef(x.view(), y_none.view()).expect("Operation failed");

    println!("\nCorrelation coefficients:");
    println!("  Positive correlation: {corr_pos:.4}");
    println!("  Negative correlation: {corr_neg:.4}");
    println!("  No correlation: {corr_none:.4}");

    // Create a multivariate dataset and compute its covariance matrix
    let data = Array2::from_shape_vec(
        (n, 3),
        vec![
            x[0], y_pos[0], y_neg[0], x[1], y_pos[1], y_neg[1], x[2], y_pos[2], y_neg[2], x[3],
            y_pos[3], y_neg[3], x[4], y_pos[4], y_neg[4], x[5], y_pos[5], y_neg[5], x[6], y_pos[6],
            y_neg[6], x[7], y_pos[7], y_neg[7], x[8], y_pos[8], y_neg[8], x[9], y_pos[9], y_neg[9],
        ],
    )
    .expect("Operation failed");

    println!("\nCovariance matrix:");
    let covmatrix = cov(data.view(), 1).expect("Operation failed");

    println!("  Variances:");
    println!("    Var(x): {:.2}", covmatrix[[0, 0]]);
    println!("    Var(y_pos): {:.2}", covmatrix[[1, 1]]);
    println!("    Var(y_neg): {:.2}", covmatrix[[2, 2]]);

    println!("  Covariances:");
    println!("    Cov(x, y_pos): {:.2}", covmatrix[[0, 1]]);
    println!("    Cov(x, y_neg): {:.2}", covmatrix[[0, 2]]);
    println!("    Cov(y_pos, y_neg): {:.2}", covmatrix[[1, 2]]);

    // Example 3: Quantile analysis
    println!("\n\n=== Quantile Analysis ===");

    // Create some data with outliers
    let data = array![
        12.5, 10.2, 10.8, 11.5, 11.0, 12.0, 11.8, 12.2, 10.5, 11.2, 24.5, 9.8, 8.5, 25.0, 11.5,
        12.5, 9.9, 10.2, 23.5, 10.8
    ];

    println!("Data with outliers: {data:?}");

    // Calculate quartiles
    let quartiles =
        quantile(data.view(), array![0.25, 0.5, 0.75].view(), None).expect("Operation failed");
    println!("\nQuartiles:");
    println!("  Q1 (25%): {:.2}", quartiles[0]);
    println!("  Q2 (50%, median): {:.2}", quartiles[1]);
    println!("  Q3 (75%): {:.2}", quartiles[2]);

    // Calculate IQR (Interquartile Range)
    let iqr = quartiles[2] - quartiles[0];
    println!("  IQR: {iqr:.2}");

    // Define outlier boundaries
    let lower_bound = quartiles[0] - 1.5 * iqr;
    let upper_bound = quartiles[2] + 1.5 * iqr;
    println!("  Outlier boundaries: ({lower_bound:.2}, {upper_bound:.2})");

    // Count outliers
    let mut outliers = 0;
    for &value in data.iter() {
        if value < lower_bound || value > upper_bound {
            outliers += 1;
        }
    }
    println!("  Number of outliers: {outliers}");

    // Create a 2D histogram for bivariate data
    println!("\n=== 2D Histogram ===");

    // Generate x and y data with some correlation
    let x_data = array![
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 1.2, 1.8, 2.3, 2.7, 3.2, 3.6, 4.1, 4.6,
        5.1, 5.6
    ];
    let y_data = array![
        1.2, 1.6, 2.3, 2.6, 3.1, 3.5, 4.2, 4.4, 5.3, 5.7, 5.5, 4.5, 4.3, 3.8, 3.2, 2.7, 2.2, 1.8,
        1.3, 0.8
    ];

    println!("X values: {x_data:?}");
    println!("Y values: {y_data:?}");

    let (hist_2d, x_edges, y_edges) = histogram2d(
        x_data.view(),
        y_data.view(),
        Some((4, 4)),
        Some(((1.0, 6.0), (0.0, 6.0))),
        None,
    )
    .expect("Operation failed");

    println!("\n2D Histogram:");

    // Print the 2D histogram as a heatmap
    for i in (0..hist_2d.shape()[0]).rev() {
        print!("{:.1}-{:.1} |", y_edges[i], y_edges[i + 1]);
        for j in 0..hist_2d.shape()[1] {
            let count = hist_2d[[i, j]];
            match count {
                0 => print!("   "),
                1 => print!(" ▁ "),
                2 => print!(" ▃ "),
                3 => print!(" ▅ "),
                _ => print!(" █ "),
            }
        }
        println!();
    }

    print!("       |");
    for _ in 0..hist_2d.shape()[1] {
        print!("---");
    }
    println!();

    print!("       |");
    for j in 0..hist_2d.shape()[1] {
        print!("{:.1} ", x_edges[j]);
    }
    println!("{:.1}", x_edges[hist_2d.shape()[1]]);
}
