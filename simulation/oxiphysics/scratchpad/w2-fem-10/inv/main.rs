#![allow(dead_code)]
pub mod error {
    #[derive(Debug)]
    pub enum Error {
        General(String),
        InvalidInput(String),
        DofOutOfRange { dof: usize, n: usize },
        SingularSystem,
        NotConverged { solver: String, iterations: usize, residual: f64 },
    }
    pub type Result<T> = std::result::Result<T, Error>;
}
#[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/dense_lu.rs"]
pub mod dense_lu;
#[path = "/Users/kitasan/work/oxiphysics/crates/oxiphysics-fem/src/inverse_problems.rs"]
pub mod inverse_problems;
fn main() {}
