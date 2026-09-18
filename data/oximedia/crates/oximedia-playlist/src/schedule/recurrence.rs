//! Recurring event patterns.

use chrono::{DateTime, Datelike, Duration, Utc, Weekday};
use serde::{Deserialize, Serialize};

/// Recurrence pattern for scheduled events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecurrencePattern {
    /// Daily recurrence.
    Daily {
        /// Repeat every N days.
        interval: u32,
    },

    /// Weekly recurrence.
    Weekly {
        /// Repeat every N weeks.
        interval: u32,
        /// Days of the week (0 = Monday, 6 = Sunday).
        days: Vec<Weekday>,
    },

    /// Monthly recurrence.
    Monthly {
        /// Repeat every N months.
        interval: u32,
        /// Day of month (1-31).
        day_of_month: u32,
    },

    /// Yearly recurrence.
    Yearly {
        /// Month (1-12).
        month: u32,
        /// Day of month (1-31).
        day: u32,
    },

    /// Custom recurrence defined by a cron-like expression.
    Custom {
        /// Cron expression.
        expression: String,
    },
}

/// Recurring event configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recurrence {
    /// Recurrence pattern.
    pub pattern: RecurrencePattern,

    /// Start date for recurrence.
    pub start_date: DateTime<Utc>,

    /// End date for recurrence (None = infinite).
    pub end_date: Option<DateTime<Utc>>,

    /// Maximum number of occurrences (None = infinite).
    pub max_occurrences: Option<u32>,
}

impl Recurrence {
    /// Creates a new daily recurrence.
    #[must_use]
    pub fn daily(start_date: DateTime<Utc>, interval: u32) -> Self {
        Self {
            pattern: RecurrencePattern::Daily { interval },
            start_date,
            end_date: None,
            max_occurrences: None,
        }
    }

    /// Creates a new weekly recurrence.
    #[must_use]
    pub fn weekly(start_date: DateTime<Utc>, interval: u32, days: Vec<Weekday>) -> Self {
        Self {
            pattern: RecurrencePattern::Weekly { interval, days },
            start_date,
            end_date: None,
            max_occurrences: None,
        }
    }

    /// Creates a new monthly recurrence.
    #[must_use]
    pub fn monthly(start_date: DateTime<Utc>, interval: u32, day_of_month: u32) -> Self {
        Self {
            pattern: RecurrencePattern::Monthly {
                interval,
                day_of_month,
            },
            start_date,
            end_date: None,
            max_occurrences: None,
        }
    }

    /// Sets the end date.
    #[must_use]
    pub fn with_end_date(mut self, end_date: DateTime<Utc>) -> Self {
        self.end_date = Some(end_date);
        self
    }

    /// Sets the maximum number of occurrences.
    #[must_use]
    pub const fn with_max_occurrences(mut self, max: u32) -> Self {
        self.max_occurrences = Some(max);
        self
    }

    /// Calculates the next occurrence after a given time.
    #[must_use]
    pub fn next_occurrence(&self, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
        if let Some(end) = self.end_date {
            if after >= end {
                return None;
            }
        }

        let mut current = if after < self.start_date {
            self.start_date
        } else {
            after
        };

        match &self.pattern {
            RecurrencePattern::Daily { interval } => {
                let days = (*interval).max(1);
                current += Duration::days(i64::from(days));
            }

            RecurrencePattern::Weekly { interval, days } => {
                if days.is_empty() {
                    return None;
                }

                // Find the next matching weekday
                for _ in 0..7 {
                    current += Duration::days(1);
                    if days.contains(&current.weekday()) {
                        // Check interval
                        let weeks_diff = (current - self.start_date).num_weeks();
                        if weeks_diff % i64::from(*interval) == 0 {
                            break;
                        }
                    }
                }
            }

            RecurrencePattern::Monthly {
                interval,
                day_of_month,
            } => {
                let mut month = current.month();
                let mut year = current.year();

                month += interval;
                while month > 12 {
                    month -= 12;
                    year += 1;
                }

                // Create new date with the target day
                if let Some(dt) = current
                    .with_year(year)
                    .and_then(|d| d.with_month(month))
                    .and_then(|d| d.with_day(*day_of_month))
                {
                    current = dt;
                }
            }

            RecurrencePattern::Yearly { month, day } => {
                let year = current.year() + 1;

                if let Some(dt) = current
                    .with_year(year)
                    .and_then(|d| d.with_month(*month))
                    .and_then(|d| d.with_day(*day))
                {
                    current = dt;
                }
            }

            RecurrencePattern::Custom { .. } => {
                // Custom cron expression handling would go here
                return None;
            }
        }

        if let Some(end) = self.end_date {
            if current >= end {
                return None;
            }
        }

        Some(current)
    }

    /// Generates a list of occurrences in a time range.
    #[must_use]
    pub fn occurrences_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        max_count: usize,
    ) -> Vec<DateTime<Utc>> {
        let mut occurrences = Vec::new();
        let mut current = if start < self.start_date {
            self.start_date
        } else {
            start
        };

        while current < end && occurrences.len() < max_count {
            if current >= start {
                occurrences.push(current);
            }

            if let Some(next) = self.next_occurrence(current) {
                current = next;
            } else {
                break;
            }
        }

        occurrences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daily_recurrence() {
        let start = Utc::now();
        let recurrence = Recurrence::daily(start, 1);

        let next = recurrence
            .next_occurrence(start)
            .expect("should succeed in test");
        assert!(next > start);
    }

    #[test]
    fn test_weekly_recurrence() {
        let start = Utc::now();
        let recurrence = Recurrence::weekly(start, 1, vec![Weekday::Mon, Weekday::Wed]);

        let occurrences = recurrence.occurrences_in_range(start, start + Duration::days(14), 10);
        assert!(!occurrences.is_empty());
    }

    #[test]
    fn test_monthly_recurrence() {
        let start = Utc::now();
        let recurrence = Recurrence::monthly(start, 1, 15);

        let next = recurrence.next_occurrence(start);
        assert!(next.is_some());
    }
}
