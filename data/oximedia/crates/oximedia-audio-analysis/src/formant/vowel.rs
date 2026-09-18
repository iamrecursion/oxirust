//! Vowel detection from formant frequencies.

/// Vowel classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vowel {
    /// /i/ as in "beat"
    I,
    /// /ɪ/ as in "bit"
    ISmall,
    /// /e/ as in "bait"
    E,
    /// /ɛ/ as in "bet"
    Epsilon,
    /// /æ/ as in "bat"
    Ae,
    /// /ɑ/ as in "bot"
    Alpha,
    /// /ɔ/ as in "bought"
    OpenO,
    /// /o/ as in "boat"
    O,
    /// /ʊ/ as in "book"
    USmall,
    /// /u/ as in "boot"
    U,
    /// /ʌ/ as in "but"
    Caret,
    /// /ə/ as in "about" (schwa)
    Schwa,
    /// Unknown/unclear
    Unknown,
}

/// Detect vowel from formant frequencies.
///
/// Uses F1 and F2 to classify vowels based on typical formant values:
/// - /i/: F1=270, F2=2290
/// - /ɪ/: F1=390, F2=1990
/// - /e/: F1=530, F2=1840
/// - /æ/: F1=660, F2=1720
/// - /ɑ/: F1=730, F2=1090
/// - /ɔ/: F1=570, F2=840
/// - /o/: F1=440, F2=1020
/// - /u/: F1=300, F2=870
/// - /ʌ/: F1=640, F2=1190
///
/// # Arguments
/// * `formants` - Formant frequencies [F1, F2, ...]
///
/// # Returns
/// Detected vowel
#[must_use]
pub fn detect_vowel(formants: &[f32]) -> Vowel {
    if formants.len() < 2 {
        return Vowel::Unknown;
    }

    let f1 = formants[0];
    let f2 = formants[1];

    // Define prototypical formant values for each vowel
    let vowels = [
        (Vowel::I, 270.0, 2290.0),
        (Vowel::ISmall, 390.0, 1990.0),
        (Vowel::E, 530.0, 1840.0),
        (Vowel::Epsilon, 610.0, 1900.0),
        (Vowel::Ae, 660.0, 1720.0),
        (Vowel::Alpha, 730.0, 1090.0),
        (Vowel::OpenO, 570.0, 840.0),
        (Vowel::O, 440.0, 1020.0),
        (Vowel::USmall, 440.0, 1020.0),
        (Vowel::U, 300.0, 870.0),
        (Vowel::Caret, 640.0, 1190.0),
        (Vowel::Schwa, 500.0, 1500.0),
    ];

    // Find closest vowel using Euclidean distance
    let mut min_distance = f32::INFINITY;
    let mut best_vowel = Vowel::Unknown;

    for (vowel, f1_proto, f2_proto) in vowels {
        let distance = ((f1 - f1_proto).powi(2) + (f2 - f2_proto).powi(2)).sqrt();
        if distance < min_distance {
            min_distance = distance;
            best_vowel = vowel;
        }
    }

    // Only return vowel if distance is reasonable
    if min_distance < 400.0 {
        best_vowel
    } else {
        Vowel::Unknown
    }
}

/// Detect vowel with confidence score.
#[must_use]
pub fn detect_vowel_with_confidence(formants: &[f32]) -> (Vowel, f32) {
    let vowel = detect_vowel(formants);

    if vowel == Vowel::Unknown || formants.len() < 2 {
        return (Vowel::Unknown, 0.0);
    }

    let f1 = formants[0];
    let f2 = formants[1];

    // Compute distance to detected vowel prototype
    let (proto_f1, proto_f2) = match vowel {
        Vowel::I => (270.0, 2290.0),
        Vowel::ISmall => (390.0, 1990.0),
        Vowel::E => (530.0, 1840.0),
        Vowel::Epsilon => (610.0, 1900.0),
        Vowel::Ae => (660.0, 1720.0),
        Vowel::Alpha => (730.0, 1090.0),
        Vowel::OpenO => (570.0, 840.0),
        Vowel::O | Vowel::USmall => (440.0, 1020.0),
        Vowel::U => (300.0, 870.0),
        Vowel::Caret => (640.0, 1190.0),
        Vowel::Schwa => (500.0, 1500.0),
        Vowel::Unknown => return (Vowel::Unknown, 0.0),
    };

    let distance = ((f1 - proto_f1).powi(2) + (f2 - proto_f2).powi(2)).sqrt();

    // Convert distance to confidence (0-1)
    let confidence = (1.0 - distance / 400.0).max(0.0);

    (vowel, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vowel_detection() {
        // /i/ vowel
        let formants_i = vec![270.0, 2290.0, 3000.0, 3500.0];
        assert_eq!(detect_vowel(&formants_i), Vowel::I);

        // /æ/ vowel
        let formants_ae = vec![660.0, 1720.0, 2500.0, 3500.0];
        assert_eq!(detect_vowel(&formants_ae), Vowel::Ae);

        // /u/ vowel
        let formants_u = vec![300.0, 870.0, 2500.0, 3500.0];
        assert_eq!(detect_vowel(&formants_u), Vowel::U);
    }

    #[test]
    fn test_vowel_with_confidence() {
        let formants = vec![270.0, 2290.0];
        let (vowel, confidence) = detect_vowel_with_confidence(&formants);

        assert_eq!(vowel, Vowel::I);
        assert!(confidence > 0.9);
    }

    #[test]
    fn test_unknown_vowel() {
        let formants = vec![100.0, 5000.0]; // Unrealistic formants
        assert_eq!(detect_vowel(&formants), Vowel::Unknown);
    }
}
