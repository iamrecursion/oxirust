//! Common type definitions for interactive tutorial examples
//!
//! This module contains shared type definitions used across multiple
//! tutorial example files to avoid code duplication and keep examples
//! under the 2000 line limit.

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TutorialSystem {
    pub user_profile: UserProfile,
    pub available_modules: Vec<TutorialModule>,
    pub current_session: TutorialSession,
    pub learning_analytics: LearningAnalytics,
    pub conceptual_graph: ConceptualGraph,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct UserProfile {
    pub name: String,
    pub learning_style: LearningStyle,
    pub skill_assessment: HashMap<String, SkillLevel>,
    pub preferences: LearningPreferences,
    pub progress_history: Vec<ProgressRecord>,
    pub achievements: Vec<String>,
    pub total_study_time: Duration,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum LearningStyle {
    Visual,
    Analytical,
    Intuitive,
    Applied,
    Historical,
    Experimental,
    Hybrid(Vec<LearningStyle>),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum SkillLevel {
    Novice,
    Developing,
    Proficient,
    Advanced,
    Expert,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LearningPreferences {
    pub preferred_pace: PacePreference,
    pub complexity_tolerance: f64,
    pub proof_detail_level: ProofDetailLevel,
    pub application_focus: Vec<ApplicationDomain>,
    pub interaction_style: InteractionStyle,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum PacePreference {
    SelfPaced,
    Guided,
    Intensive,
    Casual,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ProofDetailLevel {
    Overview,
    Standard,
    Detailed,
    Rigorous,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ApplicationDomain {
    PureMathematics,
    Physics,
    Engineering,
    Statistics,
    ComputerScience,
    Finance,
    Biology,
    SignalProcessing,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum InteractionStyle {
    Exploratory,
    Structured,
    Competitive,
    Collaborative,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TutorialModule {
    pub id: String,
    pub title: String,
    pub description: String,
    pub prerequisites: Vec<String>,
    pub learning_objectives: Vec<String>,
    pub difficulty_level: u32,
    pub estimated_time: Duration,
    pub concepts: Vec<MathematicalConcept>,
    pub assessments: Vec<Assessment>,
    pub applications: Vec<PracticalApplication>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MathematicalConcept {
    pub name: String,
    pub definition: String,
    pub intuitive_explanation: String,
    pub mathematical_formulation: String,
    pub visual_representations: Vec<VisualizationSpec>,
    pub key_properties: Vec<Property>,
    pub connections: Vec<ConceptConnection>,
    pub examples: Vec<WorkedExample>,
    pub common_misconceptions: Vec<Misconception>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VisualizationSpec {
    pub title: String,
    pub description: String,
    pub plot_type: PlotType,
    pub parameters: HashMap<String, PlotParameter>,
    pub interactive_elements: Vec<InteractiveElement>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum PlotType {
    Function2D { domain: (f64, f64), range: (f64, f64) },
    Function3D { domain: ((f64, f64), (f64, f64)), range: (f64, f64) },
    ComplexPlane { radius: f64 },
    Contour { levels: Vec<f64> },
    ParametricCurve { parameter_range: (f64, f64) },
    Animation { frames: usize, duration: Duration },
    InteractiveGraph { controls: Vec<String> },
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PlotParameter {
    pub name: String,
    pub current_value: f64,
    pub range: (f64, f64),
    pub step: f64,
    pub description: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum InteractiveElement {
    Slider { name: String, min: f64, max: f64, step: f64, default: f64 },
    Checkbox { name: String, default: bool },
    Dropdown { name: String, options: Vec<String>, default: usize },
    Input { name: String, validation: String },
    Button { name: String, action: String },
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Property {
    pub statement: String,
    pub proof_sketch: String,
    pub importance: String,
    pub related_properties: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConceptConnection {
    pub target_concept: String,
    pub relationship_type: RelationshipType,
    pub explanation: String,
    pub strength: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum RelationshipType {
    Generalization,
    Specialization,
    Analogy,
    Application,
    DualConcept,
    Transformation,
    LimitingCase,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct WorkedExample {
    pub title: String,
    pub problem_statement: String,
    pub solution_steps: Vec<SolutionStep>,
    pub key_insights: Vec<String>,
    pub variations: Vec<String>,
    pub difficulty: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SolutionStep {
    pub description: String,
    pub mathematical_content: String,
    pub justification: String,
    pub alternative_approaches: Vec<String>,
    pub common_errors: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Misconception {
    pub description: String,
    pub why_it_occurs: String,
    pub correction: String,
    pub clarifying_example: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Assessment {
    pub id: String,
    pub assessment_type: AssessmentType,
    pub questions: Vec<Question>,
    pub scoring_rubric: ScoringRubric,
    pub adaptive_parameters: AdaptiveParameters,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AssessmentType {
    Diagnostic,
    Formative,
    Summative,
    Adaptive,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub question_type: QuestionType,
    pub content: String,
    pub difficulty: u32,
    pub concepts_tested: Vec<String>,
    pub hints: Vec<Hint>,
    pub solution: DetailedSolution,
    pub metacognitive_prompts: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum QuestionType {
    MultipleChoice { options: Vec<String>, correct: Vec<usize> },
    NumericalAnswer { expected: f64, tolerance: f64, units: Option<String> },
    ExpressionMatching { expected_form: String, equivalence_rules: Vec<String> },
    ProofConstruction { steps: Vec<String>, ordering: bool },
    ConceptMapping { concepts: Vec<String>, relationships: Vec<(usize, usize, String)> },
    GraphicalAnalysis { image_data: Vec<u8>, expected_features: Vec<String> },
    CodeCompletion { template: String, expected_functions: Vec<String> },
    OpenEnded { rubric: Vec<RubricCriterion> },
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Hint {
    pub level: u32,
    pub content: String,
    pub hint_type: HintType,
    pub when_to_show: HintTrigger,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum HintType {
    Conceptual,
    Strategic,
    Procedural,
    Motivational,
    Corrective,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum HintTrigger {
    OnRequest,
    AfterTime(Duration),
    AfterAttempts(u32),
    OnSpecificError(String),
    OnLowConfidence,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DetailedSolution {
    pub overview: String,
    pub detailed_steps: Vec<SolutionStep>,
    pub alternative_solutions: Vec<AlternativeSolution>,
    pub verification_methods: Vec<String>,
    pub extensions: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AlternativeSolution {
    pub approach_name: String,
    pub description: String,
    pub when_to_use: String,
    pub trade_offs: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RubricCriterion {
    pub criterion: String,
    pub levels: Vec<(String, u32)>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ScoringRubric {
    pub total_points: u32,
    pub criteria: Vec<RubricCriterion>,
    pub partial_credit_rules: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AdaptiveParameters {
    pub difficulty_adjustment: f64,
    pub hint_frequency: f64,
    pub pacing_adjustment: f64,
    pub content_selection_weights: HashMap<String, f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PracticalApplication {
    pub title: String,
    pub domain: ApplicationDomain,
    pub problem_description: String,
    pub mathematical_model: String,
    pub solution_approach: String,
    pub real_world_context: String,
    pub computational_aspects: Vec<String>,
    pub extensions: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TutorialSession {
    pub start_time: Instant,
    pub current_module: Option<String>,
    pub session_progress: SessionProgress,
    pub user_interactions: Vec<UserInteraction>,
    pub performance_metrics: PerformanceMetrics,
    pub adaptive_state: AdaptiveState,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SessionProgress {
    pub concepts_covered: Vec<String>,
    pub exercises_completed: Vec<String>,
    pub assessments_taken: Vec<String>,
    pub time_per_concept: HashMap<String, Duration>,
    pub difficulty_progression: Vec<(String, u32)>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct UserInteraction {
    pub timestamp: Instant,
    pub interaction_type: InteractionType,
    pub context: String,
    pub user_input: String,
    pub system_response: String,
    pub correctness: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum InteractionType {
    QuestionAnswer,
    ConceptExploration,
    VisualizationInteraction,
    HintRequest,
    HelpRequest,
    NavigationAction,
    PreferenceChange,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub accuracy_by_concept: HashMap<String, f64>,
    pub time_efficiency: HashMap<String, f64>,
    pub hint_usage_rate: f64,
    pub engagement_level: f64,
    pub confidence_ratings: Vec<(String, f64)>,
    pub learning_velocity: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AdaptiveState {
    pub current_difficulty: f64,
    pub learning_rate_estimate: f64,
    pub concept_mastery_estimates: HashMap<String, f64>,
    pub preferred_explanation_style: ExplanationStyle,
    pub attention_span_estimate: Duration,
    pub motivation_level: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ExplanationStyle {
    Concise,
    Detailed,
    ExampleDriven,
    ProofOriented,
    VisualFirst,
    Historical,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LearningAnalytics {
    pub session_data: Vec<SessionData>,
    pub learning_patterns: LearningPatterns,
    pub knowledge_graph_state: KnowledgeGraphState,
    pub predictive_models: PredictiveModels,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SessionData {
    pub date: SystemTime,
    pub duration: Duration,
    pub concepts_studied: Vec<String>,
    pub performance_summary: PerformanceMetrics,
    pub user_feedback: Option<UserFeedback>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct UserFeedback {
    pub satisfaction_rating: u32,
    pub difficulty_perception: u32,
    pub engagement_rating: u32,
    pub suggestions: String,
    pub preferred_improvements: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LearningPatterns {
    pub optimal_session_length: Duration,
    pub best_time_of_day: Option<u32>,
    pub effective_difficulty_progression: f64,
    pub concept_learning_order: Vec<String>,
    pub retention_rates: HashMap<String, f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KnowledgeGraphState {
    pub mastered_concepts: Vec<String>,
    pub partially_understood: Vec<String>,
    pub prerequisite_gaps: Vec<String>,
    pub concept_connections_strength: HashMap<(String, String), f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PredictiveModels {
    pub mastery_prediction: HashMap<String, f64>,
    pub time_to_mastery: HashMap<String, Duration>,
    pub optimal_next_concept: String,
    pub dropout_risk: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConceptualGraph {
    pub nodes: HashMap<String, ConceptNode>,
    pub edges: HashMap<(String, String), ConceptEdge>,
    pub learning_paths: Vec<LearningPath>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConceptNode {
    pub id: String,
    pub name: String,
    pub difficulty: u32,
    pub importance: f64,
    pub prerequisites: Vec<String>,
    pub learning_objectives: Vec<String>,
    pub estimated_learning_time: Duration,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConceptEdge {
    pub source: String,
    pub target: String,
    pub relationship: RelationshipType,
    pub strength: f64,
    pub bidirectional: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LearningPath {
    pub name: String,
    pub description: String,
    pub concept_sequence: Vec<String>,
    pub estimated_duration: Duration,
    pub difficulty_curve: Vec<u32>,
    pub target_audience: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ProgressRecord {
    pub timestamp: SystemTime,
    pub concept: String,
    pub mastery_level: f64,
    pub time_spent: Duration,
    pub attempts: u32,
    pub final_score: f64,
}
