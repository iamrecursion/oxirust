//! Gender detection from voice characteristics.

/// Gender classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    /// Male voice
    Male,
    /// Female voice
    Female,
    /// Unknown/ambiguous
    Unknown,
}

/// Detect gender from fundamental frequency and formants.
///
/// Uses typical ranges:
/// - Male F0: 85-180 Hz, F1: ~500 Hz, F2: ~1500 Hz
/// - Female F0: 165-255 Hz, F1: ~600 Hz, F2: ~1800 Hz
///
/// # Arguments
/// * `f0` - Fundamental frequency in Hz
/// * `formants` - Formant frequencies [F1, F2, ...]
///
/// # Returns
/// Detected gender
#[must_use]
pub fn detect_gender(f0: f32, formants: &[f32]) -> Gender {
    // F0-based classification
    let gender_from_f0 = if f0 < 140.0 {
        Gender::Male
    } else if f0 > 200.0 {
        Gender::Female
    } else {
        Gender::Unknown
    };

    // Formant-based classification
    if formants.len() >= 2 {
        let f1 = formants[0];
        let f2 = formants[1];

        // Typical formant values
        let male_score = formant_similarity(f1, f2, 500.0, 1500.0);
        let female_score = formant_similarity(f1, f2, 600.0, 1800.0);

        let gender_from_formants = if male_score > female_score {
            Gender::Male
        } else if female_score > male_score {
            Gender::Female
        } else {
            Gender::Unknown
        };

        // Combine F0 and formant evidence
        match (gender_from_f0, gender_from_formants) {
            (Gender::Male, Gender::Male) => Gender::Male,
            (Gender::Female, Gender::Female) => Gender::Female,
            (Gender::Unknown, g) | (g, Gender::Unknown) => g,
            _ => Gender::Unknown, // Conflicting evidence
        }
    } else {
        gender_from_f0
    }
}

/// Compute similarity score between observed and expected formants.
fn formant_similarity(f1: f32, f2: f32, expected_f1: f32, expected_f2: f32) -> f32 {
    let f1_diff = (f1 - expected_f1).abs() / expected_f1;
    let f2_diff = (f2 - expected_f2).abs() / expected_f2;
    1.0 / (1.0 + f1_diff + f2_diff)
}

/// Gender detection with confidence score.
#[must_use]
pub fn detect_gender_with_confidence(f0: f32, formants: &[f32]) -> (Gender, f32) {
    let gender = detect_gender(f0, formants);

    // Compute confidence based on how clearly the features match expectations
    let confidence = if formants.len() >= 2 {
        let f1 = formants[0];
        let f2 = formants[1];

        let male_score = formant_similarity(f1, f2, 500.0, 1500.0);
        let female_score = formant_similarity(f1, f2, 600.0, 1800.0);

        let max_score = male_score.max(female_score);
        let diff = (male_score - female_score).abs();

        // High confidence if scores differ significantly
        (max_score * diff).min(1.0)
    } else {
        // Lower confidence with F0 only
        if (120.0..=220.0).contains(&f0) {
            0.3
        } else {
            0.7
        }
    };

    (gender, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gender_detection() {
        // Typical male voice
        let male_f0 = 120.0;
        let male_formants = vec![500.0, 1500.0, 2500.0];
        assert_eq!(detect_gender(male_f0, &male_formants), Gender::Male);

        // Typical female voice
        let female_f0 = 220.0;
        let female_formants = vec![600.0, 1800.0, 2800.0];
        assert_eq!(detect_gender(female_f0, &female_formants), Gender::Female);

        // Ambiguous
        let ambiguous_f0 = 170.0;
        let result = detect_gender(ambiguous_f0, &[]);
        assert_eq!(result, Gender::Unknown);
    }

    #[test]
    fn test_gender_with_confidence() {
        let f0 = 100.0; // Clearly male
        let formants = vec![500.0, 1500.0];
        let (gender, confidence) = detect_gender_with_confidence(f0, &formants);
        assert_eq!(gender, Gender::Male);
        assert!(confidence >= 0.0 && confidence <= 1.0); // Valid confidence range
    }
}
