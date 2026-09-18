use scirs2_core::ndarray::array;
use scirs2_spatial::distance::*;

#[allow(dead_code)]
fn main() {
    // Define some test vectors
    let u = vec![1.0, 2.0, 3.0];
    let v = vec![4.0, 5.0, 6.0];

    // Numeric vector metrics
    println!("Numeric vector metrics:");
    println!("  Euclidean:    {:.6}", euclidean(&u, &v));
    println!("  Manhattan:    {:.6}", manhattan(&u, &v));
    println!("  Chebyshev:    {:.6}", chebyshev(&u, &v));
    println!("  Minkowski(3): {:.6}", minkowski(&u, &v, 3.0));
    println!("  Canberra:     {:.6}", canberra(&u, &v));
    println!("  Cosine:       {:.6}", cosine(&u, &v));
    println!("  Correlation:  {:.6}", correlation(&u, &v));
    println!("  Bray-Curtis:  {:.6}", braycurtis(&u, &v));

    // Standardized Euclidean with variance vector
    let variance = vec![0.5, 1.0, 2.0]; // Custom variance for each dimension
    println!("  Seuclidean:   {:.6}", seuclidean(&u, &v, &variance));

    // Mahalanobis with inverse covariance matrix
    let vi = array![[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]];
    println!("  Mahalanobis:  {:.6}", mahalanobis(&u, &v, &vi));

    // Boolean vector metrics
    let bu = vec![true, false, true, false, true];
    let bv = vec![true, true, false, false, false];

    println!("\nBoolean vector metrics:");
    println!("  Dice:           {:.6}", dice::<f64>(&bu, &bv));
    println!("  Kulsinski:      {:.6}", kulsinski::<f64>(&bu, &bv));
    println!("  Rogers-Tanimoto: {:.6}", rogerstanimoto::<f64>(&bu, &bv));
    println!("  Russell-Rao:    {:.6}", russellrao::<f64>(&bu, &bv));
    println!("  Sokal-Michener: {:.6}", sokalmichener::<f64>(&bu, &bv));
    println!("  Sokal-Sneath:   {:.6}", sokalsneath::<f64>(&bu, &bv));
    println!("  Yule:           {:.6}", yule::<f64>(&bu, &bv));
}
