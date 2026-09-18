// The distributed module is currently gated behind a feature flag
// This example is temporarily disabled until the module is available

#[allow(dead_code)]
fn main() {
    println!("Distributed FFT Example");
    println!("----------------------");
    println!("This example is currently disabled as the distributed module is not available.");
    println!("Please use other examples in this directory for FFT functionality.");

    // Explain what distributed FFT is about
    println!("\nDistributed FFT supports the following decomposition strategies:");
    println!("1. Slab decomposition: 1D partitioning, good for smaller problems");
    println!("2. Pencil decomposition: 2D partitioning, better for large 3D problems");
    println!("3. Volumetric decomposition: 3D partitioning, for very large problems");
    println!("4. Adaptive decomposition: Automatically chooses optimal strategy");

    // Explain communication patterns
    println!("\nSupported communication patterns:");
    println!("1. AllToAll: All processes communicate with all others (most general)");
    println!("2. PointToPoint: Direct process-to-process communication");
    println!("3. Neighbor: Only communicate with neighboring processes");
    println!("4. Hybrid: Combination of different patterns for optimal performance");

    // Explain implementation requirements
    println!("\nTo use actual distributed computation, you would need:");
    println!("1. An MPI implementation (e.g., OpenMPI, MPICH)");
    println!("2. The mpi crate for Rust or custom bindings");
    println!("3. Multiple compute nodes or processes");
    println!("4. A domain decomposition strategy appropriate for your problem");
}
