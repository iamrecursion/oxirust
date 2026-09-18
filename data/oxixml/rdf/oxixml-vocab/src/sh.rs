//! [SHACL](https://www.w3.org/TR/shacl/) — the W3C Shapes Constraint Language vocabulary, including SHACL Core, SHACL-SPARQL, and SHACL-JS terms. This is the complete, official SHACL vocabulary (183 terms) as published in the normative `shacl.ttl` vocabulary document.
//!
//! Source: <https://www.w3.org/ns/shacl#>

use oxixml_model::NamedNodeRef;

/// The `sh` namespace IRI (`http://www.w3.org/ns/shacl#`).
pub const NAMESPACE: &str = "http://www.w3.org/ns/shacl#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/ns/shacl#", $local));
    };
}

term!(
    SHAPE,
    "Shape",
    "A shape is a collection of constraints that may be targeted for certain nodes."
);
term!(
    NODE_SHAPE,
    "NodeShape",
    "A node shape is a shape that specifies constraint that need to be met with respect to focus nodes."
);
term!(
    PROPERTY_SHAPE,
    "PropertyShape",
    "A property shape is a shape that specifies constraints on the values of a focus node for a given property or path."
);
term!(
    DEACTIVATED,
    "deactivated",
    "If set to true then all nodes conform to this."
);
term!(
    TARGET_CLASS,
    "targetClass",
    "Links a shape to a class, indicating that all instances of the class must conform to the shape."
);
term!(
    TARGET_NODE,
    "targetNode",
    "Links a shape to individual nodes, indicating that these nodes must conform to the shape."
);
term!(
    TARGET_OBJECTS_OF,
    "targetObjectsOf",
    "Links a shape to a property, indicating that all all objects of triples that have the given property as their predicate must conform to the shape."
);
term!(
    TARGET_SUBJECTS_OF,
    "targetSubjectsOf",
    "Links a shape to a property, indicating that all subjects of triples that have the given property as their predicate must conform to the shape."
);
term!(
    MESSAGE,
    "message",
    "A human-readable message (possibly with placeholders for variables) explaining the cause of the result."
);
term!(
    SEVERITY,
    "severity",
    "Defines the severity that validation results produced by a shape must have. Defaults to sh:Violation."
);
term!(
    NODE_KIND_CLASS,
    "NodeKind",
    "The class of all node kinds, including sh:BlankNode, sh:IRI, sh:Literal or the combinations of these: sh:BlankNodeOrIRI, sh:BlankNodeOrLiteral, sh:IRIOrLiteral."
);
term!(BLANK_NODE, "BlankNode", "The node kind of all blank nodes.");
term!(
    BLANK_NODE_OR_IRI,
    "BlankNodeOrIRI",
    "The node kind of all blank nodes or IRIs."
);
term!(
    BLANK_NODE_OR_LITERAL,
    "BlankNodeOrLiteral",
    "The node kind of all blank nodes or literals."
);
term!(IRI, "IRI", "The node kind of all IRIs.");
term!(
    IRI_OR_LITERAL,
    "IRIOrLiteral",
    "The node kind of all IRIs or literals."
);
term!(LITERAL, "Literal", "The node kind of all literals.");
term!(
    VALIDATION_REPORT,
    "ValidationReport",
    "The class of SHACL validation reports."
);
term!(
    CONFORMS,
    "conforms",
    "True if the validation did not produce any validation results, and false otherwise."
);
term!(
    RESULT,
    "result",
    "The validation results contained in a validation report."
);
term!(
    SHAPES_GRAPH_WELL_FORMED,
    "shapesGraphWellFormed",
    "If true then the validation engine was certain that the shapes graph has passed all SHACL syntax requirements during the validation process."
);
term!(
    ABSTRACT_RESULT,
    "AbstractResult",
    "The base class of validation results, typically not instantiated directly."
);
term!(
    VALIDATION_RESULT,
    "ValidationResult",
    "The class of validation results."
);
term!(
    SEVERITY_CLASS,
    "Severity",
    "The class of validation result severity levels, including violation and warning levels."
);
term!(
    INFO,
    "Info",
    "The severity for an informational validation result."
);
term!(
    VIOLATION,
    "Violation",
    "The severity for a violation validation result."
);
term!(
    WARNING,
    "Warning",
    "The severity for a warning validation result."
);
term!(
    DETAIL,
    "detail",
    "Links a result with other results that provide more details, for example to describe violations against nested shapes."
);
term!(
    FOCUS_NODE,
    "focusNode",
    "The focus node that was validated when the result was produced."
);
term!(
    RESULT_MESSAGE,
    "resultMessage",
    "Human-readable messages explaining the cause of the result."
);
term!(
    RESULT_PATH,
    "resultPath",
    "The path of a validation result, based on the path of the validated property shape."
);
term!(
    RESULT_SEVERITY,
    "resultSeverity",
    "The severity of the result, e.g. warning."
);
term!(
    SOURCE_CONSTRAINT,
    "sourceConstraint",
    "The constraint that was validated when the result was produced."
);
term!(
    SOURCE_SHAPE,
    "sourceShape",
    "The shape that is was validated when the result was produced."
);
term!(
    SOURCE_CONSTRAINT_COMPONENT,
    "sourceConstraintComponent",
    "The constraint component that is the source of the result."
);
term!(VALUE, "value", "An RDF node that has caused the result.");
term!(
    SHAPES_GRAPH,
    "shapesGraph",
    "Shapes graphs that should be used when validating this data graph."
);
term!(
    SUGGESTED_SHAPES_GRAPH,
    "suggestedShapesGraph",
    "Suggested shapes graphs for this ontology. The values of this property may be used in the absence of specific sh:shapesGraph statements."
);
term!(
    ENTAILMENT,
    "entailment",
    "An entailment regime that indicates what kind of inferencing is required by a shapes graph."
);
term!(
    PATH,
    "path",
    "Specifies the property path of a property shape."
);
term!(
    INVERSE_PATH,
    "inversePath",
    "The (single) value of this property represents an inverse path (object to subject)."
);
term!(
    ALTERNATIVE_PATH,
    "alternativePath",
    "The (single) value of this property must be a list of path elements, representing the elements of alternative paths."
);
term!(
    ZERO_OR_MORE_PATH,
    "zeroOrMorePath",
    "The (single) value of this property represents a path that is matched zero or more times."
);
term!(
    ONE_OR_MORE_PATH,
    "oneOrMorePath",
    "The (single) value of this property represents a path that is matched one or more times."
);
term!(
    ZERO_OR_ONE_PATH,
    "zeroOrOnePath",
    "The (single) value of this property represents a path that is matched zero or one times."
);
term!(
    PARAMETERIZABLE,
    "Parameterizable",
    "Superclass of components that can take parameters, especially functions and constraint components."
);
term!(
    PARAMETER,
    "parameter",
    "The parameters of a function or constraint component."
);
term!(
    LABEL_TEMPLATE,
    "labelTemplate",
    "Outlines how human-readable labels of instances of the associated Parameterizable shall be produced. The values can contain {?paramName} as placeholders for the actual values of the given parameter."
);
term!(
    PARAMETER_CLASS,
    "Parameter",
    "The class of parameter declarations, consisting of a path predicate and (possibly) information about allowed value type, cardinality and other characteristics."
);
term!(
    OPTIONAL,
    "optional",
    "Indicates whether a parameter is optional."
);
term!(
    CONSTRAINT_COMPONENT,
    "ConstraintComponent",
    "The class of constraint components."
);
term!(
    VALIDATOR,
    "validator",
    "The validator(s) used to evaluate constraints of either node or property shapes."
);
term!(
    NODE_VALIDATOR,
    "nodeValidator",
    "The validator(s) used to evaluate a constraint in the context of a node shape."
);
term!(
    PROPERTY_VALIDATOR,
    "propertyValidator",
    "The validator(s) used to evaluate a constraint in the context of a property shape."
);
term!(
    VALIDATOR_CLASS,
    "Validator",
    "The class of validators, which provide instructions on how to process a constraint definition. This class serves as base class for the SPARQL-based validators and other possible implementations."
);
term!(
    SPARQL_ASK_VALIDATOR,
    "SPARQLAskValidator",
    "The class of validators based on SPARQL ASK queries. The queries are evaluated for each value node and are supposed to return true if the given node conforms."
);
term!(
    SPARQL_SELECT_VALIDATOR,
    "SPARQLSelectValidator",
    "The class of validators based on SPARQL SELECT queries. The queries are evaluated for each focus node and are supposed to produce bindings for all focus nodes that do not conform."
);
term!(
    AND_CONSTRAINT_COMPONENT,
    "AndConstraintComponent",
    "A constraint component that can be used to test whether a value node conforms to all members of a provided list of shapes."
);
term!(
    AND,
    "and",
    "RDF list of shapes to validate the value nodes against."
);
term!(
    CLASS_CONSTRAINT_COMPONENT,
    "ClassConstraintComponent",
    "A constraint component that can be used to verify that each value node is an instance of a given type."
);
term!(CLASS, "class", "The type that all value nodes must have.");
term!(
    CLOSED_CONSTRAINT_COMPONENT,
    "ClosedConstraintComponent",
    "A constraint component that can be used to indicate that focus nodes must only have values for those properties that have been explicitly enumerated via sh:property/sh:path."
);
term!(CLOSED, "closed", "If set to true then the shape is closed.");
term!(
    IGNORED_PROPERTIES,
    "ignoredProperties",
    "An optional RDF list of properties that are also permitted in addition to those explicitly enumerated via sh:property/sh:path."
);
term!(
    DATATYPE_CONSTRAINT_COMPONENT,
    "DatatypeConstraintComponent",
    "A constraint component that can be used to restrict the datatype of all value nodes."
);
term!(
    DATATYPE,
    "datatype",
    "Specifies an RDF datatype that all value nodes must have."
);
term!(
    DISJOINT_CONSTRAINT_COMPONENT,
    "DisjointConstraintComponent",
    "A constraint component that can be used to verify that the set of value nodes is disjoint with the the set of nodes that have the focus node as subject and the value of a given property as predicate."
);
term!(
    DISJOINT,
    "disjoint",
    "Specifies a property where the set of values must be disjoint with the value nodes."
);
term!(
    EQUALS_CONSTRAINT_COMPONENT,
    "EqualsConstraintComponent",
    "A constraint component that can be used to verify that the set of value nodes is equal to the set of nodes that have the focus node as subject and the value of a given property as predicate."
);
term!(
    EQUALS,
    "equals",
    "Specifies a property that must have the same values as the value nodes."
);
term!(
    HAS_VALUE_CONSTRAINT_COMPONENT,
    "HasValueConstraintComponent",
    "A constraint component that can be used to verify that one of the value nodes is a given RDF node."
);
term!(
    HAS_VALUE,
    "hasValue",
    "Specifies a value that must be among the value nodes."
);
term!(
    IN_CONSTRAINT_COMPONENT,
    "InConstraintComponent",
    "A constraint component that can be used to exclusively enumerate the permitted value nodes."
);
term!(
    IN,
    "in",
    "Specifies a list of allowed values so that each value node must be among the members of the given list."
);
term!(
    LANGUAGE_IN_CONSTRAINT_COMPONENT,
    "LanguageInConstraintComponent",
    "A constraint component that can be used to enumerate language tags that all value nodes must have."
);
term!(
    LANGUAGE_IN,
    "languageIn",
    "Specifies a list of language tags that all value nodes must have."
);
term!(
    LESS_THAN_CONSTRAINT_COMPONENT,
    "LessThanConstraintComponent",
    "A constraint component that can be used to verify that each value node is smaller than all the nodes that have the focus node as subject and the value of a given property as predicate."
);
term!(
    LESS_THAN,
    "lessThan",
    "Specifies a property that must have smaller values than the value nodes."
);
term!(
    LESS_THAN_OR_EQUALS_CONSTRAINT_COMPONENT,
    "LessThanOrEqualsConstraintComponent",
    "A constraint component that can be used to verify that every value node is smaller than all the nodes that have the focus node as subject and the value of a given property as predicate."
);
term!(
    LESS_THAN_OR_EQUALS,
    "lessThanOrEquals",
    "Specifies a property that must have smaller or equal values than the value nodes."
);
term!(
    MAX_COUNT_CONSTRAINT_COMPONENT,
    "MaxCountConstraintComponent",
    "A constraint component that can be used to restrict the maximum number of value nodes."
);
term!(
    MAX_COUNT,
    "maxCount",
    "Specifies the maximum number of values in the set of value nodes."
);
term!(
    MAX_EXCLUSIVE_CONSTRAINT_COMPONENT,
    "MaxExclusiveConstraintComponent",
    "A constraint component that can be used to restrict the range of value nodes with a maximum exclusive value."
);
term!(
    MAX_EXCLUSIVE,
    "maxExclusive",
    "Specifies the maximum exclusive value of each value node."
);
term!(
    MAX_INCLUSIVE_CONSTRAINT_COMPONENT,
    "MaxInclusiveConstraintComponent",
    "A constraint component that can be used to restrict the range of value nodes with a maximum inclusive value."
);
term!(
    MAX_INCLUSIVE,
    "maxInclusive",
    "Specifies the maximum inclusive value of each value node."
);
term!(
    MAX_LENGTH_CONSTRAINT_COMPONENT,
    "MaxLengthConstraintComponent",
    "A constraint component that can be used to restrict the maximum string length of value nodes."
);
term!(
    MAX_LENGTH,
    "maxLength",
    "Specifies the maximum string length of each value node."
);
term!(
    MIN_COUNT_CONSTRAINT_COMPONENT,
    "MinCountConstraintComponent",
    "A constraint component that can be used to restrict the minimum number of value nodes."
);
term!(
    MIN_COUNT,
    "minCount",
    "Specifies the minimum number of values in the set of value nodes."
);
term!(
    MIN_EXCLUSIVE_CONSTRAINT_COMPONENT,
    "MinExclusiveConstraintComponent",
    "A constraint component that can be used to restrict the range of value nodes with a minimum exclusive value."
);
term!(
    MIN_EXCLUSIVE,
    "minExclusive",
    "Specifies the minimum exclusive value of each value node."
);
term!(
    MIN_INCLUSIVE_CONSTRAINT_COMPONENT,
    "MinInclusiveConstraintComponent",
    "A constraint component that can be used to restrict the range of value nodes with a minimum inclusive value."
);
term!(
    MIN_INCLUSIVE,
    "minInclusive",
    "Specifies the minimum inclusive value of each value node."
);
term!(
    MIN_LENGTH_CONSTRAINT_COMPONENT,
    "MinLengthConstraintComponent",
    "A constraint component that can be used to restrict the minimum string length of value nodes."
);
term!(
    MIN_LENGTH,
    "minLength",
    "Specifies the minimum string length of each value node."
);
term!(
    NODE_CONSTRAINT_COMPONENT,
    "NodeConstraintComponent",
    "A constraint component that can be used to verify that all value nodes conform to the given node shape."
);
term!(
    NODE,
    "node",
    "Specifies the node shape that all value nodes must conform to."
);
term!(
    NODE_KIND_CONSTRAINT_COMPONENT,
    "NodeKindConstraintComponent",
    "A constraint component that can be used to restrict the RDF node kind of each value node."
);
term!(
    NODE_KIND,
    "nodeKind",
    "Specifies the node kind (e.g. IRI or literal) each value node."
);
term!(
    NOT_CONSTRAINT_COMPONENT,
    "NotConstraintComponent",
    "A constraint component that can be used to verify that value nodes do not conform to a given shape."
);
term!(
    NOT,
    "not",
    "Specifies a shape that the value nodes must not conform to."
);
term!(
    OR_CONSTRAINT_COMPONENT,
    "OrConstraintComponent",
    "A constraint component that can be used to restrict the value nodes so that they conform to at least one out of several provided shapes."
);
term!(
    OR,
    "or",
    "Specifies a list of shapes so that the value nodes must conform to at least one of the shapes."
);
term!(
    PATTERN_CONSTRAINT_COMPONENT,
    "PatternConstraintComponent",
    "A constraint component that can be used to verify that every value node matches a given regular expression."
);
term!(
    FLAGS,
    "flags",
    "An optional flag to be used with regular expression pattern matching."
);
term!(
    PATTERN,
    "pattern",
    "Specifies a regular expression pattern that the string representations of the value nodes must match."
);
term!(
    PROPERTY_CONSTRAINT_COMPONENT,
    "PropertyConstraintComponent",
    "A constraint component that can be used to verify that all value nodes conform to the given property shape."
);
term!(
    PROPERTY,
    "property",
    "Links a shape to its property shapes."
);
term!(
    QUALIFIED_MAX_COUNT_CONSTRAINT_COMPONENT,
    "QualifiedMaxCountConstraintComponent",
    "A constraint component that can be used to verify that a specified maximum number of value nodes conforms to a given shape."
);
term!(
    QUALIFIED_MIN_COUNT_CONSTRAINT_COMPONENT,
    "QualifiedMinCountConstraintComponent",
    "A constraint component that can be used to verify that a specified minimum number of value nodes conforms to a given shape."
);
term!(
    QUALIFIED_MAX_COUNT,
    "qualifiedMaxCount",
    "The maximum number of value nodes that can conform to the shape."
);
term!(
    QUALIFIED_MIN_COUNT,
    "qualifiedMinCount",
    "The minimum number of value nodes that must conform to the shape."
);
term!(
    QUALIFIED_VALUE_SHAPE,
    "qualifiedValueShape",
    "The shape that a specified number of values must conform to."
);
term!(
    QUALIFIED_VALUE_SHAPES_DISJOINT,
    "qualifiedValueShapesDisjoint",
    "Can be used to mark the qualified value shape to be disjoint with its sibling shapes."
);
term!(
    UNIQUE_LANG_CONSTRAINT_COMPONENT,
    "UniqueLangConstraintComponent",
    "A constraint component that can be used to specify that no pair of value nodes may use the same language tag."
);
term!(
    UNIQUE_LANG,
    "uniqueLang",
    "Specifies whether all node values must have a unique (or no) language tag."
);
term!(
    XONE_CONSTRAINT_COMPONENT,
    "XoneConstraintComponent",
    "A constraint component that can be used to restrict the value nodes so that they conform to exactly one out of several provided shapes."
);
term!(
    XONE,
    "xone",
    "Specifies a list of shapes so that the value nodes must conform to exactly one of the shapes."
);
term!(
    SPARQL_EXECUTABLE,
    "SPARQLExecutable",
    "The class of resources that encapsulate a SPARQL query."
);
term!(
    SPARQL_ASK_EXECUTABLE,
    "SPARQLAskExecutable",
    "The class of SPARQL executables that are based on an ASK query."
);
term!(ASK, "ask", "The SPARQL ASK query to execute.");
term!(
    SPARQL_CONSTRUCT_EXECUTABLE,
    "SPARQLConstructExecutable",
    "The class of SPARQL executables that are based on a CONSTRUCT query."
);
term!(
    CONSTRUCT,
    "construct",
    "The SPARQL CONSTRUCT query to execute."
);
term!(
    SPARQL_SELECT_EXECUTABLE,
    "SPARQLSelectExecutable",
    "The class of SPARQL executables based on a SELECT query."
);
term!(SELECT, "select", "The SPARQL SELECT query to execute.");
term!(
    SPARQL_UPDATE_EXECUTABLE,
    "SPARQLUpdateExecutable",
    "The class of SPARQL executables based on a SPARQL UPDATE."
);
term!(UPDATE, "update", "The SPARQL UPDATE to execute.");
term!(
    PREFIXES,
    "prefixes",
    "The prefixes that shall be applied before parsing the associated SPARQL query."
);
term!(
    PREFIX_DECLARATION,
    "PrefixDeclaration",
    "The class of prefix declarations, consisting of pairs of a prefix with a namespace."
);
term!(
    DECLARE,
    "declare",
    "Links a resource with its namespace prefix declarations."
);
term!(PREFIX, "prefix", "The prefix of a prefix declaration.");
term!(
    PREFIX_NAMESPACE,
    "namespace",
    "The namespace associated with a prefix in a prefix declaration."
);
term!(
    SPARQL_CONSTRAINT_COMPONENT,
    "SPARQLConstraintComponent",
    "A constraint component that can be used to define constraints based on SPARQL queries."
);
term!(SPARQL, "sparql", "Links a shape with SPARQL constraints.");
term!(
    SPARQL_CONSTRAINT,
    "SPARQLConstraint",
    "The class of constraints based on SPARQL SELECT queries."
);
term!(
    DEFAULT_VALUE,
    "defaultValue",
    "A default value for a property, for example for user interface tools to pre-populate input fields."
);
term!(
    DESCRIPTION,
    "description",
    "Human-readable descriptions for the property in the context of the surrounding shape."
);
term!(
    GROUP,
    "group",
    "Can be used to link to a property group to indicate that a property shape belongs to a group of related property shapes."
);
term!(
    NAME,
    "name",
    "Human-readable labels for the property in the context of the surrounding shape."
);
term!(
    ORDER,
    "order",
    "Specifies the relative order of this compared to its siblings. For example use 0 for the first, 1 for the second."
);
term!(
    PROPERTY_GROUP,
    "PropertyGroup",
    "Instances of this class represent groups of property shapes that belong together."
);
term!(
    TARGET,
    "target",
    "Links a shape to a target specified by an extension language, for example instances of sh:SPARQLTarget."
);
term!(
    GENERIC_TARGET_CLASS,
    "Target",
    "The base class of targets such as those based on SPARQL queries."
);
term!(
    TARGET_TYPE,
    "TargetType",
    "The (meta) class for parameterizable targets. Instances of this are instantiated as values of the sh:target property."
);
term!(
    SPARQL_TARGET,
    "SPARQLTarget",
    "The class of targets that are based on SPARQL queries."
);
term!(
    SPARQL_TARGET_TYPE,
    "SPARQLTargetType",
    "The (meta) class for parameterizable targets that are based on SPARQL queries."
);
term!(FUNCTION, "Function", "The class of SHACL functions.");
term!(
    RETURN_TYPE,
    "returnType",
    "The expected type of values returned by the associated function."
);
term!(
    SPARQL_FUNCTION,
    "SPARQLFunction",
    "A function backed by a SPARQL query - either ASK or SELECT."
);
term!(
    RESULT_ANNOTATION,
    "resultAnnotation",
    "Links a SPARQL validator with zero or more sh:ResultAnnotation instances, defining how to derive additional result properties based on the variables of the SELECT query."
);
term!(
    RESULT_ANNOTATION_CLASS,
    "ResultAnnotation",
    "A class of result annotations, which define the rules to derive the values of a given annotation property as extra values for a validation result."
);
term!(
    ANNOTATION_PROPERTY,
    "annotationProperty",
    "The annotation property that shall be set."
);
term!(
    ANNOTATION_VALUE,
    "annotationValue",
    "The (default) values of the annotation property."
);
term!(
    ANNOTATION_VAR_NAME,
    "annotationVarName",
    "The name of the SPARQL variable from the SELECT clause that shall be used for the values."
);
term!(
    THIS,
    "this",
    "A node expression that represents the current focus node."
);
term!(
    FILTER_SHAPE,
    "filterShape",
    "The shape that all input nodes of the expression need to conform to."
);
term!(
    NODES,
    "nodes",
    "The node expression producing the input nodes of a filter shape expression."
);
term!(
    INTERSECTION,
    "intersection",
    "A list of node expressions that shall be intersected."
);
term!(
    UNION,
    "union",
    "A list of node expressions that shall be used together."
);
term!(
    EXPRESSION_CONSTRAINT_COMPONENT,
    "ExpressionConstraintComponent",
    "A constraint component that can be used to verify that a given node expression produces true for all value nodes."
);
term!(
    EXPRESSION,
    "expression",
    "The node expression that must return true for the value nodes."
);
term!(
    RULE_CLASS,
    "Rule",
    "The class of SHACL rules. Never instantiated directly."
);
term!(RULE, "rule", "The rules linked to a shape.");
term!(
    CONDITION,
    "condition",
    "The shapes that the focus nodes need to conform to before a rule is executed on them."
);
term!(
    TRIPLE_RULE,
    "TripleRule",
    "A rule based on triple (subject, predicate, object) pattern."
);
term!(
    SUBJECT,
    "subject",
    "An expression producing the resources that shall be inferred as subjects."
);
term!(
    PREDICATE,
    "predicate",
    "An expression producing the properties that shall be inferred as predicates."
);
term!(
    OBJECT,
    "object",
    "An expression producing the nodes that shall be inferred as objects."
);
term!(
    SPARQL_RULE,
    "SPARQLRule",
    "The class of SHACL rules based on SPARQL CONSTRUCT queries."
);
term!(
    JS_EXECUTABLE,
    "JSExecutable",
    "Abstract base class of resources that declare an executable JavaScript."
);
term!(
    JS_TARGET,
    "JSTarget",
    "The class of targets that are based on JavaScript functions."
);
term!(
    JS_TARGET_TYPE,
    "JSTargetType",
    "The (meta) class for parameterizable targets that are based on JavaScript functions."
);
term!(
    JS_CONSTRAINT,
    "JSConstraint",
    "The class of constraints backed by a JavaScript function."
);
term!(
    JS_CONSTRAINT_COMPONENT,
    "JSConstraintComponent",
    "A constraint component with the parameter sh:js linking to a sh:JSConstraint containing a sh:script."
);
term!(JS, "js", "Constraints expressed in JavaScript.");
term!(
    JS_FUNCTION_NAME,
    "jsFunctionName",
    "The name of the JavaScript function to execute."
);
term!(
    JS_LIBRARY,
    "jsLibrary",
    "Declares which JavaScript libraries are needed to execute this."
);
term!(
    JS_LIBRARY_URL,
    "jsLibraryURL",
    "Declares the URLs of a JavaScript library. This should be the absolute URL of a JavaScript file. Implementations may redirect those to local files."
);
term!(
    JS_FUNCTION,
    "JSFunction",
    "The class of SHACL functions that execute a JavaScript function when called."
);
term!(
    JS_LIBRARY_CLASS,
    "JSLibrary",
    "Represents a JavaScript library, typically identified by one or more URLs of files to include."
);
term!(
    JS_RULE,
    "JSRule",
    "The class of SHACL rules expressed using JavaScript."
);
term!(
    JS_VALIDATOR,
    "JSValidator",
    "A SHACL validator based on JavaScript. This can be used to declare SHACL constraint components that perform JavaScript-based validation when used."
);

/// All 183 terms defined in the [`sh`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    SHAPE,
    NODE_SHAPE,
    PROPERTY_SHAPE,
    DEACTIVATED,
    TARGET_CLASS,
    TARGET_NODE,
    TARGET_OBJECTS_OF,
    TARGET_SUBJECTS_OF,
    MESSAGE,
    SEVERITY,
    NODE_KIND_CLASS,
    BLANK_NODE,
    BLANK_NODE_OR_IRI,
    BLANK_NODE_OR_LITERAL,
    IRI,
    IRI_OR_LITERAL,
    LITERAL,
    VALIDATION_REPORT,
    CONFORMS,
    RESULT,
    SHAPES_GRAPH_WELL_FORMED,
    ABSTRACT_RESULT,
    VALIDATION_RESULT,
    SEVERITY_CLASS,
    INFO,
    VIOLATION,
    WARNING,
    DETAIL,
    FOCUS_NODE,
    RESULT_MESSAGE,
    RESULT_PATH,
    RESULT_SEVERITY,
    SOURCE_CONSTRAINT,
    SOURCE_SHAPE,
    SOURCE_CONSTRAINT_COMPONENT,
    VALUE,
    SHAPES_GRAPH,
    SUGGESTED_SHAPES_GRAPH,
    ENTAILMENT,
    PATH,
    INVERSE_PATH,
    ALTERNATIVE_PATH,
    ZERO_OR_MORE_PATH,
    ONE_OR_MORE_PATH,
    ZERO_OR_ONE_PATH,
    PARAMETERIZABLE,
    PARAMETER,
    LABEL_TEMPLATE,
    PARAMETER_CLASS,
    OPTIONAL,
    CONSTRAINT_COMPONENT,
    VALIDATOR,
    NODE_VALIDATOR,
    PROPERTY_VALIDATOR,
    VALIDATOR_CLASS,
    SPARQL_ASK_VALIDATOR,
    SPARQL_SELECT_VALIDATOR,
    AND_CONSTRAINT_COMPONENT,
    AND,
    CLASS_CONSTRAINT_COMPONENT,
    CLASS,
    CLOSED_CONSTRAINT_COMPONENT,
    CLOSED,
    IGNORED_PROPERTIES,
    DATATYPE_CONSTRAINT_COMPONENT,
    DATATYPE,
    DISJOINT_CONSTRAINT_COMPONENT,
    DISJOINT,
    EQUALS_CONSTRAINT_COMPONENT,
    EQUALS,
    HAS_VALUE_CONSTRAINT_COMPONENT,
    HAS_VALUE,
    IN_CONSTRAINT_COMPONENT,
    IN,
    LANGUAGE_IN_CONSTRAINT_COMPONENT,
    LANGUAGE_IN,
    LESS_THAN_CONSTRAINT_COMPONENT,
    LESS_THAN,
    LESS_THAN_OR_EQUALS_CONSTRAINT_COMPONENT,
    LESS_THAN_OR_EQUALS,
    MAX_COUNT_CONSTRAINT_COMPONENT,
    MAX_COUNT,
    MAX_EXCLUSIVE_CONSTRAINT_COMPONENT,
    MAX_EXCLUSIVE,
    MAX_INCLUSIVE_CONSTRAINT_COMPONENT,
    MAX_INCLUSIVE,
    MAX_LENGTH_CONSTRAINT_COMPONENT,
    MAX_LENGTH,
    MIN_COUNT_CONSTRAINT_COMPONENT,
    MIN_COUNT,
    MIN_EXCLUSIVE_CONSTRAINT_COMPONENT,
    MIN_EXCLUSIVE,
    MIN_INCLUSIVE_CONSTRAINT_COMPONENT,
    MIN_INCLUSIVE,
    MIN_LENGTH_CONSTRAINT_COMPONENT,
    MIN_LENGTH,
    NODE_CONSTRAINT_COMPONENT,
    NODE,
    NODE_KIND_CONSTRAINT_COMPONENT,
    NODE_KIND,
    NOT_CONSTRAINT_COMPONENT,
    NOT,
    OR_CONSTRAINT_COMPONENT,
    OR,
    PATTERN_CONSTRAINT_COMPONENT,
    FLAGS,
    PATTERN,
    PROPERTY_CONSTRAINT_COMPONENT,
    PROPERTY,
    QUALIFIED_MAX_COUNT_CONSTRAINT_COMPONENT,
    QUALIFIED_MIN_COUNT_CONSTRAINT_COMPONENT,
    QUALIFIED_MAX_COUNT,
    QUALIFIED_MIN_COUNT,
    QUALIFIED_VALUE_SHAPE,
    QUALIFIED_VALUE_SHAPES_DISJOINT,
    UNIQUE_LANG_CONSTRAINT_COMPONENT,
    UNIQUE_LANG,
    XONE_CONSTRAINT_COMPONENT,
    XONE,
    SPARQL_EXECUTABLE,
    SPARQL_ASK_EXECUTABLE,
    ASK,
    SPARQL_CONSTRUCT_EXECUTABLE,
    CONSTRUCT,
    SPARQL_SELECT_EXECUTABLE,
    SELECT,
    SPARQL_UPDATE_EXECUTABLE,
    UPDATE,
    PREFIXES,
    PREFIX_DECLARATION,
    DECLARE,
    PREFIX,
    PREFIX_NAMESPACE,
    SPARQL_CONSTRAINT_COMPONENT,
    SPARQL,
    SPARQL_CONSTRAINT,
    DEFAULT_VALUE,
    DESCRIPTION,
    GROUP,
    NAME,
    ORDER,
    PROPERTY_GROUP,
    TARGET,
    GENERIC_TARGET_CLASS,
    TARGET_TYPE,
    SPARQL_TARGET,
    SPARQL_TARGET_TYPE,
    FUNCTION,
    RETURN_TYPE,
    SPARQL_FUNCTION,
    RESULT_ANNOTATION,
    RESULT_ANNOTATION_CLASS,
    ANNOTATION_PROPERTY,
    ANNOTATION_VALUE,
    ANNOTATION_VAR_NAME,
    THIS,
    FILTER_SHAPE,
    NODES,
    INTERSECTION,
    UNION,
    EXPRESSION_CONSTRAINT_COMPONENT,
    EXPRESSION,
    RULE_CLASS,
    RULE,
    CONDITION,
    TRIPLE_RULE,
    SUBJECT,
    PREDICATE,
    OBJECT,
    SPARQL_RULE,
    JS_EXECUTABLE,
    JS_TARGET,
    JS_TARGET_TYPE,
    JS_CONSTRAINT,
    JS_CONSTRAINT_COMPONENT,
    JS,
    JS_FUNCTION_NAME,
    JS_LIBRARY,
    JS_LIBRARY_URL,
    JS_FUNCTION,
    JS_LIBRARY_CLASS,
    JS_RULE,
    JS_VALIDATOR,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn namespace_ends_with_separator() {
        assert!(NAMESPACE.ends_with('#') || NAMESPACE.ends_with('/'));
    }

    #[test]
    fn terms_are_namespace_prefixed() {
        for term in ALL {
            assert!(
                term.as_str().starts_with(NAMESPACE),
                "{} is not prefixed by {}",
                term.as_str(),
                NAMESPACE
            );
            assert!(term.as_str().len() > NAMESPACE.len());
        }
    }

    #[test]
    fn no_duplicate_terms() {
        let mut seen = HashSet::new();
        for term in ALL {
            assert!(
                seen.insert(term.as_str()),
                "duplicate IRI: {}",
                term.as_str()
            );
        }
    }

    #[test]
    fn expected_term_count() {
        assert_eq!(ALL.len(), 183);
    }
}
