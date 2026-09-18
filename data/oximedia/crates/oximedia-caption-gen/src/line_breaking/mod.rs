//! Caption line-breaking algorithms: greedy, optimal (Knuth-Plass-inspired DP),
//! reading-speed helpers, and line-balance optimisation.

pub mod balance;
pub mod config;
pub mod greedy;
pub mod kp_common;
pub mod larsch;
pub mod optimal;
pub mod smawk;

pub use balance::{
    adjust_duration_for_reading, compute_cps, reading_speed_ok, rebalance_lines, LineBalance,
};
pub use config::{reading_speed_ok_for_audience, AudienceProfile, CpsCache, LineBreakConfig};
pub use greedy::{cjk_break, greedy_break, language_aware_break, LineBreakAlgorithm};
pub use larsch::{optimal_break_larsch, Larsch};
pub use optimal::optimal_break;
pub use smawk::{optimal_break_smawk, smawk_row_minima, TotallyMonotoneMatrix};

#[cfg(test)]
mod api_tests {
    use crate::line_breaking::AudienceProfile;
    use crate::line_breaking::CpsCache;
    use crate::line_breaking::Larsch;
    use crate::line_breaking::LineBalance;
    use crate::line_breaking::LineBreakAlgorithm;
    use crate::line_breaking::LineBreakConfig;
    use crate::line_breaking::{
        adjust_duration_for_reading, compute_cps, reading_speed_ok, rebalance_lines,
    };
    use crate::line_breaking::{cjk_break, language_aware_break, reading_speed_ok_for_audience};
    use crate::line_breaking::{greedy_break, optimal_break, optimal_break_smawk};
    use crate::line_breaking::{optimal_break_larsch, smawk_row_minima};

    #[test]
    fn test_module_split_reexports_preserve_api() {
        // Verify all major public symbols are importable through the module.
        let _ = LineBreakAlgorithm::Greedy;
        let _ = LineBreakAlgorithm::Optimal;
        let _ = LineBreakAlgorithm::Fixed(42);

        let cfg = LineBreakConfig::default_broadcast();
        assert_eq!(cfg.max_chars_per_line, 42);

        let _ = AudienceProfile::Adults.max_cps();

        let mut cache = CpsCache::new();
        let _ = cache.compute_cps("hello", 1000);

        let lines = vec!["hi".to_string(), "there".to_string()];
        let _ = LineBalance::balance_factor(&lines);

        let mut larsch = Larsch::new(4);
        let cost = |r: usize, c: usize| -> i64 { (r as i64 - c as i64).pow(2) };
        for r in 0..4 {
            larsch.add_row(r, &cost);
        }

        let _ = greedy_break("hello world", 10);
        let _ = optimal_break("hello world", 10);
        let _ = optimal_break_smawk("hello world", 10);
        let _ = optimal_break_larsch("hello world", 10);
        let _ = smawk_row_minima;

        let _ = compute_cps("hello", 1000);
        let _ = reading_speed_ok("hello", 1000, 17.0);
        let _ = adjust_duration_for_reading("hello", 500, 17.0);
        let _ = rebalance_lines(lines, 40);
        let _ = cjk_break("日本語", 5);
        let _ = language_aware_break("hello world", 10);
        let _ = reading_speed_ok_for_audience("hello", 1000, AudienceProfile::Adults);
    }
}
