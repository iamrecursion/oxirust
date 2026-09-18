// Statistical spelling correction demo
//
// This example demonstrates the enhanced statistical spelling correction
// functionality in the scirs2-text crate, including context-aware correction.

use scirs2_text::{
    DictionaryCorrector, ErrorModel, SpellingCorrector, StatisticalCorrector,
    StatisticalCorrectorConfig,
};
use std::time::Instant;

// Sample text with misspellings in different contexts
const TEXT_WITH_CONTEXT_MISSPELLINGS: &str =
    "I went to the bnk to deposit some money. The river bnk was muddy after the rain. \
    I recieved your mesage about the meeting. He recieved many complements on his work. \
    Their was a problem with there computer. The museum disply had many historical artefcts.";

// Expected corrected text
const EXPECTED_CORRECTED_TEXT: &str =
    "I went to the bank to deposit some money. The river bank was muddy after the rain. \
    I received your message about the meeting. He received many compliments on his work. \
    There was a problem with their computer. The museum display had many historical artifacts.";

// Sample text for language model training
const SAMPLE_TRAINING_TEXT: &str =
    "I went to the bank to deposit some money yesterday. The bank offers good interest rates. \
    The river bank was muddy after the rain. We sat on the bank of the river and watched the sunset. \
    I received your message about the meeting. Thank you for the message you sent. \
    He received many compliments on his work. The teacher gave compliments to the students. \
    There was a problem with their computer. Their car broke down on the highway. \
    The museum display had many historical artifacts. The ancient artifacts were well preserved. \
    The display was impressive and educational.";

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Statistical Spelling Correction Demo\n");

    // Create a dictionary corrector as baseline
    let dict_corrector = DictionaryCorrector::default();

    // Create a statistical corrector
    let mut stat_corrector = StatisticalCorrector::default();

    // Train the language model
    train_language_model(&mut stat_corrector);

    // Add specific words to ensure consistent behavior in examples
    add_example_words(&mut stat_corrector);

    // Compare dictionary and statistical correctors
    compare_correctors(&dict_corrector, &stat_corrector)?;

    // Demonstrate context-aware correction
    context_aware_correction_demo(&stat_corrector)?;

    // Performance test
    performance_test(&dict_corrector, &stat_corrector)?;

    // Configuration demo
    configuration_demo()?;

    // Noise model demo
    noise_model_demo()?;

    Ok(())
}

// Function to train the language model with sample text
#[allow(dead_code)]
fn train_language_model(corrector: &mut StatisticalCorrector) {
    println!("Training language model with sample text...");

    // Add sample training text
    corrector.add_trainingtext(SAMPLE_TRAINING_TEXT);

    // Add more specialized training examples for context disambiguation
    let additional_examples = [
        // Bank context examples
        "I went to the bank to deposit money.",
        "The bank is open until 5pm.",
        "She works at the bank downtown.",
        "I need to check my bank account.",
        // River bank context examples
        "We sat on the bank of the river.",
        "The river bank was covered with flowers.",
        "They fished from the bank of the lake.",
        "The boat was tied to the bank.",
        // Homophone examples for there/their/they're
        "There is a book on the table.",
        "Their house is very beautiful.",
        "They're going to the movies tonight.",
        "There was a problem with the system.",
        "Their car broke down yesterday.",
        "They're planning a vacation next month.",
        // Complement/compliment examples
        "He received many compliments on his presentation.",
        "She gave him a compliment about his new haircut.",
        "Red and green are complementary colors.",
        "This wine complements the meal perfectly.",
        // Message examples
        "I received your message yesterday.",
        "Please send me a message when you arrive.",
        "The message was unclear and confusing.",
        "She left a message on my voicemail.",
    ];

    for example in &additional_examples {
        corrector.add_trainingtext(example);
    }

    println!(
        "Language model trained with {} words vocabulary\n",
        corrector.vocabulary_size()
    );
}

// Function to add specific words for consistent example behavior
#[allow(dead_code)]
fn add_example_words(corrector: &mut StatisticalCorrector) {
    // Add specific words to the dictionary
    let word_frequencies = [
        // Common misspelled words
        ("bank", 100),
        ("river", 100),
        ("deposit", 100),
        ("money", 100),
        ("received", 100),
        ("message", 100),
        ("meeting", 100),
        ("compliments", 100),
        ("complements", 100),
        ("work", 100),
        ("there", 100),
        ("their", 100),
        ("they're", 100),
        ("was", 100),
        ("problem", 100),
        ("computer", 100),
        ("museum", 100),
        ("display", 100),
        ("historical", 100),
        ("artifacts", 100),
    ];

    for (word, freq) in &word_frequencies {
        corrector.add_word(word, *freq);
    }
}

// Compare dictionary-based and statistical spelling correction
#[allow(dead_code)]
fn compare_correctors(
    dict_corrector: &DictionaryCorrector,
    stat_corrector: &StatisticalCorrector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dictionary vs. Statistical Correction ===\n");

    // Define test cases with known misspellings
    let test_cases = [
        ("recieve", "receive"),
        ("mesage", "message"),
        ("bnk", "bank"),
        ("thier", "their"),
        ("complements", "compliments"), // Can be correct in some contexts
        ("artefacts", "artifacts"),
        ("disply", "display"),
        ("definately", "definitely"),
    ];

    println!(
        "{:<15} {:<15} {:<15}",
        "Misspelled", "Dictionary", "Statistical"
    );
    println!("{:-<45}", "");

    for (misspelled, _expected) in &test_cases {
        let dict_correction = dict_corrector.correct(misspelled)?;
        let stat_correction = stat_corrector.correct(misspelled)?;

        println!("{misspelled:<15} {dict_correction:<15} {stat_correction:<15}");
    }

    println!("\nDictionary sizes:");
    println!(
        "  - Dictionary _corrector: {} words",
        dict_corrector.dictionary_size()
    );
    println!(
        "  - Statistical _corrector: {} words (+ {} in language model)",
        stat_corrector.dictionary_size(),
        stat_corrector.vocabulary_size()
    );

    Ok(())
}

// Demonstrate context-aware correction
#[allow(dead_code)]
fn context_aware_correction_demo(
    corrector: &StatisticalCorrector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Context-Aware Correction Demo ===\n");

    println!("Original text with misspellings:");
    println!("{TEXT_WITH_CONTEXT_MISSPELLINGS}\n");

    // Correct the text
    let correctedtext = corrector.correcttext(TEXT_WITH_CONTEXT_MISSPELLINGS)?;

    println!("Corrected text:");
    println!("{correctedtext}\n");

    println!("Expected text:");
    println!("{EXPECTED_CORRECTED_TEXT}\n");

    // Compare specific correction examples
    println!("Specific context examples:\n");

    // Example 1: bnk -> bank in different contexts
    let text1 = "I went to the bnk to deposit some money.";
    let text2 = "The river bnk was muddy after the rain.";

    println!("Example 1: 'bnk' in financial context");
    println!("Before: {text1}");
    println!("After:  {}\n", corrector.correcttext(text1)?);

    println!("Example 2: 'bnk' in geographical context");
    println!("Before: {text2}");
    println!("After:  {}\n", corrector.correcttext(text2)?);

    // Example 2: there/their homophone confusion
    let text3 = "Their was a problem with the computer.";
    let text4 = "There car broke down on the highway.";

    println!("Example 3: 'their' used incorrectly");
    println!("Before: {text3}");
    println!("After:  {}\n", corrector.correcttext(text3)?);

    println!("Example 4: 'there' used incorrectly");
    println!("Before: {text4}");
    println!("After:  {}\n", corrector.correcttext(text4)?);

    Ok(())
}

// Test performance of different correctors and configurations
#[allow(dead_code)]
fn performance_test(
    dict_corrector: &DictionaryCorrector,
    stat_corrector: &StatisticalCorrector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Performance Test ===\n");

    // Create test text with a mix of correct and incorrect words
    let testtext = TEXT_WITH_CONTEXT_MISSPELLINGS.repeat(10);

    // Measure dictionary _corrector performance
    let start = Instant::now();
    let _ = dict_corrector.correcttext(&testtext)?;
    let dict_time = start.elapsed();

    // Measure statistical _corrector performance
    let start = Instant::now();
    let _ = stat_corrector.correcttext(&testtext)?;
    let stat_time = start.elapsed();

    // Create a non-contextual statistical _corrector for comparison
    let non_context_config = StatisticalCorrectorConfig {
        use_context: false,
        ..Default::default()
    };
    let mut non_context_corrector = StatisticalCorrector::new(non_context_config);

    // Add training data to ensure fair comparison
    train_language_model(&mut non_context_corrector);
    add_example_words(&mut non_context_corrector);

    // Measure non-contextual statistical _corrector performance
    let start = Instant::now();
    let _ = non_context_corrector.correcttext(&testtext)?;
    let non_context_time = start.elapsed();

    println!(
        "Performance comparison on text with {} characters:",
        testtext.len()
    );
    println!("  - Dictionary corrector: {dict_time:?}");
    println!("  - Statistical _corrector (without context): {non_context_time:?}");
    println!("  - Statistical _corrector (with context): {stat_time:?}");

    Ok(())
}

// Demonstrate different configurations for statistical correction
#[allow(dead_code)]
fn configuration_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Configuration Options Demo ===\n");

    // Create configurations with different settings
    let configs = [
        ("Default", StatisticalCorrectorConfig::default()),
        (
            "Conservative (max_edit_distance=1)",
            StatisticalCorrectorConfig {
                max_edit_distance: 1,
                ..Default::default()
            },
        ),
        (
            "Aggressive (max_edit_distance=3)",
            StatisticalCorrectorConfig {
                max_edit_distance: 3,
                ..Default::default()
            },
        ),
        (
            "Language model focused (weight=0.9)",
            StatisticalCorrectorConfig {
                language_model_weight: 0.9,
                edit_distance_weight: 0.1,
                ..Default::default()
            },
        ),
        (
            "Edit distance focused (weight=0.9)",
            StatisticalCorrectorConfig {
                language_model_weight: 0.1,
                edit_distance_weight: 0.9,
                ..Default::default()
            },
        ),
        (
            "No context",
            StatisticalCorrectorConfig {
                use_context: false,
                ..Default::default()
            },
        ),
    ];

    // Sample misspelled words with varied edit distances
    let test_cases = [
        "recieve",     // Should be "receive"
        "accidant",    // Could be "accident" or "accidental" depending on max_edit_distance
        "programing",  // Single-m, should be "programming"
        "thier",       // Common misspelling of "their"
        "complements", // Could be "compliments" depending on context
    ];

    // Test each configuration
    for (name, config) in &configs {
        let mut corrector = StatisticalCorrector::new(config.clone());

        // Train the model to ensure consistent behavior
        train_language_model(&mut corrector);
        add_example_words(&mut corrector);

        println!("{name} configuration:");
        println!("  max_editdistance: {}", config.max_edit_distance);
        println!("  language_modelweight: {}", config.language_model_weight);
        println!("  edit_distanceweight: {}", config.edit_distance_weight);
        println!("  usecontext: {}", config.use_context);

        println!("\n  Correction examples:");
        for word in &test_cases {
            let corrected = corrector.correct(word)?;
            println!("    {word} -> {corrected}");
        }

        // Show a context example if context is enabled
        if config.use_context {
            let context_example = "Going to the bnk to deposit money. The river bnk was muddy.";
            let corrected = corrector.correcttext(context_example)?;
            println!("\n  Context example:");
            println!("    Before: {context_example}");
            println!("    After:  {corrected}");
        }

        println!("\n{:-<60}", "");
    }

    Ok(())
}

// Demonstrate the error model (noisy channel model)
#[allow(dead_code)]
fn noise_model_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Error Model Demo ===\n");

    // Create different error models with varying error type probabilities
    let models = [
        ("Default", ErrorModel::default()),
        ("Deletion-heavy", ErrorModel::new(0.7, 0.1, 0.1, 0.1)),
        ("Insertion-heavy", ErrorModel::new(0.1, 0.7, 0.1, 0.1)),
        ("Substitution-heavy", ErrorModel::new(0.1, 0.1, 0.7, 0.1)),
        ("Transposition-heavy", ErrorModel::new(0.1, 0.1, 0.1, 0.7)),
    ];

    // Test cases for different error types
    let test_pairs = [
        ("recieve", "receive"),        // Transposition (i and e)
        ("acheive", "achieve"),        // Transposition (i and e)
        ("languge", "language"),       // Deletion (missing 'a')
        ("programing", "programming"), // Deletion (missing 'm')
        ("probblem", "problem"),       // Insertion (extra 'b')
        ("committe", "committee"),     // Insertion (missing 'e')
        ("definately", "definitely"),  // Substitution ('a' instead of 'i')
        ("seperate", "separate"),      // Substitution ('e' instead of 'a')
    ];

    // Test each error model
    println!(
        "{:<20} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Model", "Delete Prob", "Insert Prob", "Subst Prob", "Transp Prob", "Example"
    );
    println!("{:-<80}", "");

    for (name, model) in &models {
        // Pick one example to show
        let (typo, correct) = test_pairs[0];
        let probability = model.error_probability(typo, correct);

        println!(
            "{:<20} {:<12.2} {:<12.2} {:<12.2} {:<12.2} {:<12.4}",
            name,
            model.p_deletion,
            model.p_insertion,
            model.p_substitution,
            model.p_transposition,
            probability
        );
    }

    println!("\nError probabilities for different error types (using default model):");

    let default_model = ErrorModel::default();

    for (typo, correct) in &test_pairs {
        let prob = default_model.error_probability(typo, correct);
        println!("{typo:<12} -> {correct:<12}: {prob:.6}");
    }

    println!("\nImpact on correction with custom error model:");

    // Create a statistical corrector with a custom error model
    let custom_config = StatisticalCorrectorConfig {
        language_model_weight: 0.3,
        edit_distance_weight: 0.7,
        ..Default::default()
    };

    let mut custom_corrector = StatisticalCorrector::new(custom_config);
    train_language_model(&mut custom_corrector);
    add_example_words(&mut custom_corrector);

    // Create a transposition-heavy error model (good for common spelling errors)
    let transposition_model = ErrorModel::new(0.1, 0.1, 0.1, 0.7);
    custom_corrector.set_error_model(transposition_model);

    // Test some examples
    println!("\nCorrecting text with transposition-heavy error model:");
    let testtext = "I recieved a mesage about thier acheivements.";
    let corrected = custom_corrector.correcttext(testtext)?;

    println!("Before: {testtext}");
    println!("After:  {corrected}");

    Ok(())
}
