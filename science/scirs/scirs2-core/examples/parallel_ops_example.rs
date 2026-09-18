//! Example usage of the parallel operations abstraction layer

use scirs2_core::parallel_ops::{par_range, IntoParallelIterator, ParallelIterator};

#[allow(dead_code)]
fn main() {
    println!("Parallel operations example");

    // Example 1: Parallel range iteration
    let squares: Vec<i32> = par_range(0, 10).map(|x| (x * x) as i32).collect();
    println!("Squares: {:?}", squares);

    // Example 2: Parallel vector processing
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let doubled: Vec<i32> = data.into_par_iter().map(|x| x * 2).collect();
    println!("Doubled: {:?}", doubled);

    // Example 3: Parallel filtering
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let evens: Vec<i32> = numbers.into_par_iter().filter(|x| x % 2 == 0).collect();
    println!("Even numbers: {:?}", evens);

    // Example 4: Error handling in parallel
    let values = vec![1, 2, 3, 4, 5];
    let result = values.into_par_iter().try_for_each(|x| {
        if x < 10 {
            println!("Processing: {}", x);
            Ok(())
        } else {
            Err("Value too large")
        }
    });

    match result {
        Ok(()) => println!("All values processed successfully"),
        Err(e) => println!("Error: {}", e),
    }

    // Example 5: Check if parallel is enabled
    println!(
        "Parallel processing enabled: {}",
        scirs2_core::parallel_ops::is_parallel_enabled()
    );
    println!(
        "Number of threads: {}",
        scirs2_core::parallel_ops::num_threads()
    );
}
