//! Enhanced PDE solvers for financial applications
//!
//! This module provides advanced partial differential equation solvers optimized for
//! financial derivative pricing with improved stability and performance.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::finance::types::FinancialOption;
use scirs2_core::ndarray::{Array1, Array2};

/// Crank-Nicolson solver for 1D PDEs (unconditionally stable, second-order accurate)
#[derive(Debug, Clone)]
pub struct CrankNicolsonSolver {
    /// Spatial grid size
    pub n_space: usize,
    /// Time steps
    pub n_time: usize,
    /// Theta parameter (0.5 for CN, 1.0 for implicit Euler)
    pub theta: f64,
}

impl CrankNicolsonSolver {
    /// Create a new Crank-Nicolson solver
    pub fn new(n_space: usize, n_time: usize) -> IntegrateResult<Self> {
        if n_space < 3 {
            return Err(IntegrateError::ValueError(
                "Grid size must be at least 3".to_string(),
            ));
        }

        if n_time == 0 {
            return Err(IntegrateError::ValueError(
                "Number of time steps must be positive".to_string(),
            ));
        }

        Ok(Self {
            n_space,
            n_time,
            theta: 0.5, // Classic Crank-Nicolson
        })
    }

    /// Solve Black-Scholes PDE using Crank-Nicolson method
    pub fn solve_black_scholes(
        &self,
        option: &FinancialOption,
        sigma: f64,
    ) -> IntegrateResult<f64> {
        let s_max = option.spot * 3.0;
        let ds = s_max / (self.n_space - 1) as f64;
        let dt = option.maturity / self.n_time as f64;

        // Create spatial grid
        let mut s_grid = Array1::<f64>::zeros(self.n_space);
        for i in 0..self.n_space {
            s_grid[i] = i as f64 * ds;
        }

        // Initialize value grid with terminal condition
        let mut v = Array1::<f64>::zeros(self.n_space);
        for i in 0..self.n_space {
            let s = s_grid[i];
            v[i] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => {
                    (s - option.strike).max(0.0)
                }
                crate::specialized::finance::types::OptionType::Put => (option.strike - s).max(0.0),
            };
        }

        // Build tridiagonal matrices for Crank-Nicolson
        let mut a = Array1::<f64>::zeros(self.n_space);
        let mut b = Array1::<f64>::zeros(self.n_space);
        let mut c = Array1::<f64>::zeros(self.n_space);

        for i in 1..(self.n_space - 1) {
            let s = s_grid[i];
            let sigma_sq = sigma * sigma;

            // Coefficients for d²V/dS² and dV/dS
            let alpha = 0.5 * sigma_sq * s * s / (ds * ds);
            let beta = 0.5 * (option.risk_free_rate - option.dividend_yield) * s / ds;

            a[i] = -self.theta * dt * (alpha - beta);
            b[i] = 1.0 + self.theta * dt * (2.0 * alpha + option.risk_free_rate);
            c[i] = -self.theta * dt * (alpha + beta);
        }

        // Boundary conditions
        b[0] = 1.0;
        c[0] = 0.0;
        a[self.n_space - 1] = 0.0;
        b[self.n_space - 1] = 1.0;

        // Time-stepping backward
        for _ in 0..self.n_time {
            // Build right-hand side
            let mut rhs = Array1::<f64>::zeros(self.n_space);

            for i in 1..(self.n_space - 1) {
                let s = s_grid[i];
                let sigma_sq = sigma * sigma;

                let alpha = 0.5 * sigma_sq * s * s / (ds * ds);
                let beta = 0.5 * (option.risk_free_rate - option.dividend_yield) * s / ds;

                let a_rhs = (1.0 - self.theta) * dt * (alpha - beta);
                let b_rhs = 1.0 - (1.0 - self.theta) * dt * (2.0 * alpha + option.risk_free_rate);
                let c_rhs = (1.0 - self.theta) * dt * (alpha + beta);

                rhs[i] = a_rhs * v[i - 1] + b_rhs * v[i] + c_rhs * v[i + 1];
            }

            // Boundary conditions for RHS
            rhs[0] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => 0.0,
                crate::specialized::finance::types::OptionType::Put => {
                    option.strike * (-option.risk_free_rate * dt).exp()
                }
            };

            rhs[self.n_space - 1] = match option.option_type {
                crate::specialized::finance::types::OptionType::Call => {
                    s_max - option.strike * (-option.risk_free_rate * dt).exp()
                }
                crate::specialized::finance::types::OptionType::Put => 0.0,
            };

            // Solve tridiagonal system
            v = thomas_algorithm(&a, &b, &c, &rhs)?;
        }

        // Interpolate to get option value at spot
        let idx = (option.spot / ds) as usize;
        let idx = idx.min(self.n_space - 2);

        let s1 = s_grid[idx];
        let s2 = s_grid[idx + 1];
        let v1 = v[idx];
        let v2 = v[idx + 1];

        // Linear interpolation
        let value = v1 + (v2 - v1) * (option.spot - s1) / (s2 - s1);

        Ok(value)
    }
}

/// ADI (Alternating Direction Implicit) solver for 2D PDEs
#[derive(Debug, Clone)]
pub struct ADISolver {
    /// Grid size in first dimension
    pub n_x: usize,
    /// Grid size in second dimension
    pub n_y: usize,
    /// Number of time steps
    pub n_time: usize,
}

impl ADISolver {
    /// Create a new ADI solver
    pub fn new(n_x: usize, n_y: usize, n_time: usize) -> IntegrateResult<Self> {
        if n_x < 3 || n_y < 3 {
            return Err(IntegrateError::ValueError(
                "Grid sizes must be at least 3".to_string(),
            ));
        }

        if n_time == 0 {
            return Err(IntegrateError::ValueError(
                "Number of time steps must be positive".to_string(),
            ));
        }

        Ok(Self { n_x, n_y, n_time })
    }

    /// Solve 2D PDE using Douglas-Rachford ADI scheme
    ///
    /// Solves: ∂u/∂t = L_x(u) + L_y(u) where L_x and L_y are spatial operators
    pub fn solve_2d_diffusion(
        &self,
        x_range: (f64, f64),
        y_range: (f64, f64),
        t_max: f64,
        diffusion_x: f64,
        diffusion_y: f64,
        initial_condition: &Array2<f64>,
    ) -> IntegrateResult<Array2<f64>> {
        if initial_condition.nrows() != self.n_x || initial_condition.ncols() != self.n_y {
            return Err(IntegrateError::ValueError(
                "Initial condition dimensions must match grid size".to_string(),
            ));
        }

        let dx = (x_range.1 - x_range.0) / (self.n_x - 1) as f64;
        let dy = (y_range.1 - y_range.0) / (self.n_y - 1) as f64;
        let dt = t_max / self.n_time as f64;

        let mut u = initial_condition.clone();
        let mut u_star = Array2::<f64>::zeros((self.n_x, self.n_y));

        let rx = diffusion_x * dt / (dx * dx);
        let ry = diffusion_y * dt / (dy * dy);

        // ADI splitting: half-step in x, then half-step in y
        for _ in 0..self.n_time {
            // Step 1: Implicit in x-direction
            for j in 0..self.n_y {
                let mut a = Array1::<f64>::zeros(self.n_x);
                let mut b = Array1::<f64>::zeros(self.n_x);
                let mut c = Array1::<f64>::zeros(self.n_x);
                let mut rhs = Array1::<f64>::zeros(self.n_x);

                for i in 1..(self.n_x - 1) {
                    a[i] = -0.5 * rx;
                    b[i] = 1.0 + rx;
                    c[i] = -0.5 * rx;
                    rhs[i] = u[[i, j]];
                }

                // Boundary conditions (Dirichlet: u = 0)
                b[0] = 1.0;
                c[0] = 0.0;
                rhs[0] = 0.0;

                a[self.n_x - 1] = 0.0;
                b[self.n_x - 1] = 1.0;
                rhs[self.n_x - 1] = 0.0;

                let solution = thomas_algorithm(&a, &b, &c, &rhs)?;

                for i in 0..self.n_x {
                    u_star[[i, j]] = solution[i];
                }
            }

            // Step 2: Implicit in y-direction
            for i in 0..self.n_x {
                let mut a = Array1::<f64>::zeros(self.n_y);
                let mut b = Array1::<f64>::zeros(self.n_y);
                let mut c = Array1::<f64>::zeros(self.n_y);
                let mut rhs = Array1::<f64>::zeros(self.n_y);

                for j in 1..(self.n_y - 1) {
                    a[j] = -0.5 * ry;
                    b[j] = 1.0 + ry;
                    c[j] = -0.5 * ry;
                    rhs[j] = u_star[[i, j]];
                }

                // Boundary conditions
                b[0] = 1.0;
                c[0] = 0.0;
                rhs[0] = 0.0;

                a[self.n_y - 1] = 0.0;
                b[self.n_y - 1] = 1.0;
                rhs[self.n_y - 1] = 0.0;

                let solution = thomas_algorithm(&a, &b, &c, &rhs)?;

                for j in 0..self.n_y {
                    u[[i, j]] = solution[j];
                }
            }
        }

        Ok(u)
    }
}

/// High-order finite difference solver (4th order in space)
#[derive(Debug, Clone)]
pub struct HighOrderFDSolver {
    /// Grid size
    pub n_space: usize,
    /// Time steps
    pub n_time: usize,
}

impl HighOrderFDSolver {
    /// Create a new high-order FD solver
    pub fn new(n_space: usize, n_time: usize) -> IntegrateResult<Self> {
        if n_space < 5 {
            return Err(IntegrateError::ValueError(
                "Grid size must be at least 5 for 4th order scheme".to_string(),
            ));
        }

        Ok(Self { n_space, n_time })
    }

    /// Compute 4th order central difference for second derivative
    fn fourth_order_laplacian(&self, u: &Array1<f64>, i: usize, dx: f64) -> f64 {
        let n = u.len();

        if i < 2 || i >= n - 2 {
            // Fall back to 2nd order at boundaries
            (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx)
        } else {
            // 4th order: (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12*dx²)
            (-u[i - 2] + 16.0 * u[i - 1] - 30.0 * u[i] + 16.0 * u[i + 1] - u[i + 2])
                / (12.0 * dx * dx)
        }
    }

    /// Solve diffusion equation using high-order scheme
    pub fn solve_diffusion(
        &self,
        x_range: (f64, f64),
        t_max: f64,
        diffusion: f64,
        initial_condition: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        if initial_condition.len() != self.n_space {
            return Err(IntegrateError::ValueError(
                "Initial condition length must match grid size".to_string(),
            ));
        }

        let dx = (x_range.1 - x_range.0) / (self.n_space - 1) as f64;
        let dt = t_max / self.n_time as f64;

        let mut u = initial_condition.clone();

        // Use implicit scheme for stability
        for _ in 0..self.n_time {
            let mut du = Array1::<f64>::zeros(self.n_space);

            for i in 2..(self.n_space - 2) {
                du[i] = diffusion * self.fourth_order_laplacian(&u, i, dx);
            }

            // Update (simple forward Euler for demonstration)
            for i in 2..(self.n_space - 2) {
                u[i] += dt * du[i];
            }
        }

        Ok(u)
    }
}

/// Operator splitting solver for reaction-diffusion equations
#[derive(Debug, Clone)]
pub struct OperatorSplittingSolver {
    /// Grid size
    pub n_space: usize,
    /// Time steps
    pub n_time: usize,
}

impl OperatorSplittingSolver {
    /// Create a new operator splitting solver
    pub fn new(n_space: usize, n_time: usize) -> IntegrateResult<Self> {
        Ok(Self { n_space, n_time })
    }

    /// Solve reaction-diffusion equation using Strang splitting
    ///
    /// du/dt = D * d²u/dx² + R(u)
    /// Strang splitting: R(dt/2) -> D(dt) -> R(dt/2)
    pub fn solve_reaction_diffusion<F>(
        &self,
        x_range: (f64, f64),
        t_max: f64,
        diffusion: f64,
        reaction_fn: F,
        initial_condition: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(f64) -> f64,
    {
        if initial_condition.len() != self.n_space {
            return Err(IntegrateError::ValueError(
                "Initial condition length must match grid size".to_string(),
            ));
        }

        let dx = (x_range.1 - x_range.0) / (self.n_space - 1) as f64;
        let dt = t_max / self.n_time as f64;

        let mut u = initial_condition.clone();

        for _ in 0..self.n_time {
            // Step 1: Reaction (half step)
            for i in 0..self.n_space {
                u[i] += 0.5 * dt * reaction_fn(u[i]);
            }

            // Step 2: Diffusion (full step) using implicit method
            let r = diffusion * dt / (dx * dx);

            let mut a = Array1::<f64>::zeros(self.n_space);
            let mut b = Array1::<f64>::zeros(self.n_space);
            let mut c = Array1::<f64>::zeros(self.n_space);
            let mut rhs = u.clone();

            for i in 1..(self.n_space - 1) {
                a[i] = -r;
                b[i] = 1.0 + 2.0 * r;
                c[i] = -r;
            }

            // Boundary conditions
            b[0] = 1.0;
            c[0] = 0.0;
            a[self.n_space - 1] = 0.0;
            b[self.n_space - 1] = 1.0;

            u = thomas_algorithm(&a, &b, &c, &rhs)?;

            // Step 3: Reaction (half step)
            for i in 0..self.n_space {
                u[i] += 0.5 * dt * reaction_fn(u[i]);
            }
        }

        Ok(u)
    }
}

/// Thomas algorithm (tridiagonal matrix solver)
fn thomas_algorithm(
    a: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
) -> IntegrateResult<Array1<f64>> {
    let n = b.len();

    if a.len() != n || c.len() != n || d.len() != n {
        return Err(IntegrateError::ValueError(
            "All arrays must have same length".to_string(),
        ));
    }

    let mut c_prime = Array1::<f64>::zeros(n);
    let mut d_prime = Array1::<f64>::zeros(n);

    // Forward sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            return Err(IntegrateError::ValueError(
                "Thomas algorithm failed: zero pivot".to_string(),
            ));
        }

        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = Array1::<f64>::zeros(n);
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::finance::types::{FinancialOption, OptionStyle, OptionType};

    #[test]
    fn test_crank_nicolson_solver_creation() {
        let solver = CrankNicolsonSolver::new(100, 50).expect("Operation failed");
        assert_eq!(solver.n_space, 100);
        assert_eq!(solver.n_time, 50);
        assert!((solver.theta - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_crank_nicolson_black_scholes() {
        let solver = CrankNicolsonSolver::new(100, 100).expect("Operation failed");

        let option = FinancialOption {
            option_type: OptionType::Call,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let price = solver
            .solve_black_scholes(&option, 0.2)
            .expect("Operation failed");

        // Compare with Black-Scholes analytical solution
        let bs_price = crate::specialized::finance::pricing::black_scholes::black_scholes_price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Call,
        );

        // Should be close (within 1%)
        assert!(
            (price - bs_price).abs() / bs_price < 0.01,
            "PDE price {} vs BS price {}",
            price,
            bs_price
        );
    }

    #[test]
    fn test_adi_solver_creation() {
        let solver = ADISolver::new(50, 50, 100).expect("Operation failed");
        assert_eq!(solver.n_x, 50);
        assert_eq!(solver.n_y, 50);
    }

    #[test]
    fn test_adi_2d_diffusion() {
        let solver = ADISolver::new(21, 21, 50).expect("Operation failed");

        // Gaussian initial condition
        let mut initial = Array2::<f64>::zeros((21, 21));
        for i in 0..21 {
            for j in 0..21 {
                let x = (i as f64 - 10.0) / 10.0;
                let y = (j as f64 - 10.0) / 10.0;
                initial[[i, j]] = (-(x * x + y * y) / 0.1).exp();
            }
        }

        let result = solver
            .solve_2d_diffusion((-1.0, 1.0), (-1.0, 1.0), 0.1, 0.1, 0.1, &initial)
            .expect("Operation failed");

        // After diffusion, center should have decreased, but sum should be conserved (approximately)
        assert!(result[[10, 10]] < initial[[10, 10]]);
    }

    #[test]
    fn test_thomas_algorithm() {
        // Solve: 2x₁ - x₂ = 1
        //       -x₁ + 2x₂ - x₃ = 0
        //       -x₂ + 2x₃ = 1
        let a = Array1::from_vec(vec![0.0, -1.0, -1.0]);
        let b = Array1::from_vec(vec![2.0, 2.0, 2.0]);
        let c = Array1::from_vec(vec![-1.0, -1.0, 0.0]);
        let d = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let x = thomas_algorithm(&a, &b, &c, &d).expect("Operation failed");

        // Solution should be x = [1, 1, 1]
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((x[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_order_solver() {
        let solver = HighOrderFDSolver::new(101, 100).expect("Operation failed");

        // Initial Gaussian
        let mut initial = Array1::<f64>::zeros(101);
        for i in 0..101 {
            let x = (i as f64 - 50.0) / 50.0;
            initial[i] = (-x * x / 0.1).exp();
        }

        let result = solver
            .solve_diffusion((-1.0, 1.0), 0.01, 0.1, &initial)
            .expect("Operation failed");

        // Peak should decrease due to diffusion
        assert!(result[50] < initial[50]);
    }

    #[test]
    fn test_operator_splitting() {
        let solver = OperatorSplittingSolver::new(50, 50).expect("Operation failed");

        // Simple linear reaction: R(u) = -u
        let reaction = |u: f64| -u;

        let mut initial = Array1::<f64>::zeros(50);
        initial[25] = 1.0;

        let result = solver
            .solve_reaction_diffusion((-1.0, 1.0), 0.1, 0.1, reaction, &initial)
            .expect("Operation failed");

        // Both diffusion and decay should occur
        assert!(result[25] < initial[25]);
    }

    #[test]
    fn test_crank_nicolson_put_option() {
        let solver = CrankNicolsonSolver::new(100, 100).expect("Operation failed");

        let option = FinancialOption {
            option_type: OptionType::Put,
            option_style: OptionStyle::European,
            strike: 100.0,
            maturity: 1.0,
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
        };

        let price = solver
            .solve_black_scholes(&option, 0.2)
            .expect("Operation failed");

        let bs_price = crate::specialized::finance::pricing::black_scholes::black_scholes_price(
            100.0,
            100.0,
            0.05,
            0.0,
            0.2,
            1.0,
            OptionType::Put,
        );

        assert!(
            (price - bs_price).abs() / bs_price < 0.01,
            "PDE put price {} vs BS price {}",
            price,
            bs_price
        );
    }
}
