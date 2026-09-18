use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_integrate::specialized::finance::{
    derivatives::{
        exotic::{AsianOption, AveragingMethod, BarrierOption, BarrierType},
        vanilla::EuropeanOption,
        variance_swaps::VarianceSwap,
    },
    models::VolatilityModel,
    pricing::{
        finite_difference::price_finite_difference, monte_carlo::price_monte_carlo,
        tree::price_tree,
    },
    solvers::StochasticPDESolver,
    types::{FinanceMethod, FinancialOption, OptionStyle, OptionType},
};
use std::time::Duration;

fn bench_black_scholes_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("black_scholes_pricing");
    group.measurement_time(Duration::from_secs(10));

    // Standard ATM European call option
    let option = FinancialOption {
        option_type: OptionType::Call,
        option_style: OptionStyle::European,
        strike: 100.0,
        maturity: 1.0,
        spot: 100.0,
        risk_free_rate: 0.05,
        dividend_yield: 0.0,
    };

    // Benchmark analytical Black-Scholes
    let vanilla = EuropeanOption::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
    group.bench_function("analytical_black_scholes", |b| b.iter(|| vanilla.price()));

    // Benchmark Crank-Nicolson finite difference (various grid sizes)
    for n in [50, 100, 200] {
        let solver = StochasticPDESolver::new(
            n,
            n,
            VolatilityModel::Constant(0.2),
            FinanceMethod::FiniteDifference,
        );
        group.bench_with_input(BenchmarkId::new("crank_nicolson", n), &n, |b, _| {
            b.iter(|| price_finite_difference(&solver, &option).expect("Operation failed"))
        });
    }

    // Benchmark binomial tree (various steps)
    for n_steps in [50, 100, 200] {
        let solver = StochasticPDESolver::new(
            n_steps,
            n_steps,
            VolatilityModel::Constant(0.2),
            FinanceMethod::Tree { n_steps },
        );
        group.bench_with_input(
            BenchmarkId::new("binomial_tree", n_steps),
            &n_steps,
            |b, _| b.iter(|| price_tree(&solver, &option, n_steps).expect("Operation failed")),
        );
    }

    // Benchmark Monte Carlo (various path counts)
    for n_paths in [10000, 50000] {
        let solver = StochasticPDESolver::new(
            100,
            50,
            VolatilityModel::Constant(0.2),
            FinanceMethod::MonteCarlo {
                n_paths,
                antithetic: true,
            },
        );
        group.bench_with_input(
            BenchmarkId::new("monte_carlo", n_paths),
            &n_paths,
            |b, _| {
                b.iter(|| {
                    price_monte_carlo(&solver, &option, n_paths, true).expect("Operation failed")
                })
            },
        );
    }

    group.finish();
}

fn bench_exotic_options(c: &mut Criterion) {
    let mut group = c.benchmark_group("exotic_options");
    group.measurement_time(Duration::from_secs(10));

    // Barrier option pricing
    let barrier = BarrierOption::new(
        100.0, // spot
        100.0, // strike
        120.0, // barrier
        0.0,   // rebate
        0.05,  // rate
        0.0,   // dividend
        0.2,   // volatility
        1.0,   // maturity
        OptionType::Call,
        BarrierType::UpAndOut,
    )
    .expect("Operation failed");

    group.bench_function("barrier_option_monte_carlo", |b| {
        b.iter(|| {
            barrier
                .price_monte_carlo(50000, 252)
                .expect("Operation failed")
        })
    });

    // Asian option pricing
    let asian = AsianOption::new(
        100.0,
        100.0,
        0.05,
        0.0,
        0.2,
        1.0,
        OptionType::Call,
        AveragingMethod::Geometric,
        252,
    )
    .expect("Operation failed");

    group.bench_function("asian_geometric_closed_form", |b| {
        b.iter(|| {
            asian
                .price_geometric_closed_form()
                .expect("Operation failed")
        })
    });

    group.bench_function("asian_monte_carlo", |b| {
        b.iter(|| asian.price_monte_carlo(50000).expect("Operation failed"))
    });

    group.finish();
}

fn bench_variance_swaps(c: &mut Criterion) {
    let mut group = c.benchmark_group("variance_swaps");
    group.measurement_time(Duration::from_secs(5));

    // Variance swap with realized variance calculation
    let var_swap = VarianceSwap::new(
        0.04,  // strike
        100.0, // notional
        1.0,   // maturity
        252,   // observations_per_year
        0.05,  // risk_free_rate
    )
    .expect("Operation failed");

    // Benchmark payoff calculation
    group.bench_function("variance_swap_payoff", |b| b.iter(|| var_swap.payoff(0.05)));

    // Benchmark realized variance from price series
    let prices: Vec<f64> = (0..252)
        .map(|i| 100.0 * (1.0 + 0.02 * (i as f64 / 252.0).sin()))
        .collect();

    group.bench_function("realized_variance_252_days", |b| {
        b.iter(|| VarianceSwap::realized_variance(&prices, 252.0))
    });

    group.finish();
}

fn bench_greeks_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("greeks");
    group.measurement_time(Duration::from_secs(5));

    let option = EuropeanOption::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);

    group.bench_function("all_greeks", |b| b.iter(|| option.greeks()));

    group.finish();
}

fn bench_implied_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("implied_volatility");
    group.measurement_time(Duration::from_secs(5));

    let option = EuropeanOption::new(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, OptionType::Call);
    let market_price = option.price();

    group.bench_function("newton_raphson", |b| {
        b.iter(|| {
            option
                .implied_volatility(market_price)
                .expect("Operation failed")
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_black_scholes_methods,
    bench_exotic_options,
    bench_variance_swaps,
    bench_greeks_calculation,
    bench_implied_volatility,
);
criterion_main!(benches);
