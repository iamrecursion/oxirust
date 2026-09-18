//! [PROV-O](https://www.w3.org/TR/prov-o/) — the W3C PROVenance Ontology.
//!
//! Source: <https://www.w3.org/ns/prov-o#>

use oxixml_model::NamedNodeRef;

/// The `prov` namespace IRI (`http://www.w3.org/ns/prov#`).
pub const NAMESPACE: &str = "http://www.w3.org/ns/prov#";

macro_rules! term {
    ($name:ident, $local:literal, $doc:expr) => {
        #[doc = $doc]
        pub const $name: NamedNodeRef<'static> =
            NamedNodeRef::new_unchecked(concat!("http://www.w3.org/ns/prov#", $local));
    };
}

term!(
    UNQUALIFIED_FORM,
    "unqualifiedForm",
    "Classes and properties used to qualify relationships are annotated with prov:unqualifiedForm to indicate the property used to assert an unqualified provenance relation."
);
term!(
    WAS_REVISION_OF,
    "wasRevisionOf",
    "A revision is a derivation that revises an entity into a revised version."
);
term!(
    AQ,
    "aq",
    "Annotation property (a sub-property of `rdfs:seeAlso`) used by the PROV-O specification itself to cross-reference an alternate qualified/unqualified form of a term."
);
term!(
    SHARES_DEFINITION_WITH,
    "sharesDefinitionWith",
    "Annotation property (a sub-property of `rdfs:seeAlso`) indicating that two OWL terms share the same underlying PROV-DM definition."
);
term!(
    DEFINITION,
    "definition",
    "A definition quoted from PROV-DM or PROV-CONSTRAINTS that describes the concept expressed with this OWL term."
);
term!(
    EDITORIAL_NOTE,
    "editorialNote",
    "A note by the OWL development team about how this term expresses the PROV-DM concept, or how it should be used in context of semantic web or linked data."
);
term!(
    INVERSE,
    "inverse",
    "PROV-O does not define all property inverses. The directionalities defined in PROV-O should be given preference over those not defined. However, if users wish to name the inverse of a PROV-O property, the local name given by prov:inverse should be used."
);
term!(
    CONSTRAINTS,
    "constraints",
    "A reference to the principal section of the PROV-CONSTRAINTS document that describes this concept."
);
term!(
    SPECIALIZATION_OF,
    "specializationOf",
    "An entity that is a specialization of another shares all aspects of the latter, and additionally presents more specific aspects of the same thing as the latter. In particular, the lifetime of the entity being specialized contains that of any specialization. Examples of aspects include a time period, an abstraction, and a context associated with the entity."
);
term!(
    DM,
    "dm",
    "A reference to the principal section of the PROV-DM document that describes this concept."
);
term!(
    CATEGORY,
    "category",
    "Classify prov-o terms into three categories, including 'starting-point', 'qualifed', and 'extended'. This classification is used by the prov-o html document to gently introduce prov-o terms to its users."
);
term!(
    ORDER,
    "order",
    "The position that this OWL term should be listed within documentation. The scope of the documentation (e.g., among all terms, among terms within a prov:category, among properties applying to a particular class, etc.) is unspecified."
);
term!(
    EDITORS_DEFINITION,
    "editorsDefinition",
    "When the prov-o term does not have a definition drawn from prov-dm, and the prov-o editor provides one."
);
term!(
    COMPONENT,
    "component",
    "Classify prov-o terms into six components according to prov-dm, including 'agents-responsibility', 'alternate', 'annotations', 'collections', 'derivations', and 'entities-activities'. This classification is used so that readers of prov-o specification can find its correspondence with the prov-dm specification."
);
term!(
    QUALIFIED_FORM,
    "qualifiedForm",
    "This annotation property links a subproperty of prov:wasInfluencedBy with the subclass of prov:Influence and the qualifying property that are used to qualify it. Example annotation: prov:wasGeneratedBy prov:qualifiedForm prov:qualifiedGeneration, prov:Generation . Then this unqualified assertion: :entity1 prov:wasGeneratedBy :activity1 . can be qualified by adding: :entity1 prov:qualifiedGeneration :entity1Gen . :entity1Gen a prov:Generation, prov:Influence; prov:activity :activity1; :customValue 1337 . Note how the value of the unqualified influence (prov:wasGeneratedBy :activity1) is mirrored as the value of the prov:activity (or prov:entity, or prov:agent) property on the influence class."
);
term!(
    TODO,
    "todo",
    "Annotation property used internally by the PROV-O editors to flag terms with outstanding editorial work; carries no formal semantics."
);
term!(
    N,
    "n",
    "A reference to the principal section of the PROV-DM document that describes this concept."
);
term!(
    ACTED_ON_BEHALF_OF,
    "actedOnBehalfOf",
    "An object property to express the accountability of an agent towards another agent. The subordinate agent acted on behalf of the responsible agent in an actual activity."
);
term!(
    QUALIFIED_DELEGATION,
    "qualifiedDelegation",
    "If this Agent prov:actedOnBehalfOf Agent :ag, then it can qualify how with prov:qualifiedResponsibility [ a prov:Responsibility; prov:agent :ag; :foo :bar ]."
);
term!(
    AGENT,
    "agent",
    "The prov:agent property references an prov:Agent which influenced a resource. This property applies to an prov:AgentInfluence, which is given by a subproperty of prov:qualifiedInfluence from the influenced prov:Entity, prov:Activity or prov:Agent."
);
term!(
    ACTIVITY,
    "activity",
    "The prov:activity property references an prov:Activity which influenced a resource. This property applies to an prov:ActivityInfluence, which is given by a subproperty of prov:qualifiedInfluence from the influenced prov:Entity, prov:Activity or prov:Agent."
);
term!(
    ALTERNATE_OF,
    "alternateOf",
    "Two alternate entities present aspects of the same thing. These aspects may be the same or different, and the alternate entities may or may not overlap in time."
);
term!(AT_LOCATION, "atLocation", "The Location of any resource.");
term!(
    ACTIVITY_CLASS,
    "Activity",
    "An activity is something that occurs over a period of time and acts upon or with entities; it may include consuming, processing, transforming, modifying, relocating, using, or generating entities."
);
term!(
    AGENT_CLASS,
    "Agent",
    "An agent is something that bears some form of responsibility for an activity taking place, for the existence of an entity, or for another agent's activity."
);
term!(
    ENTITY_CLASS,
    "Entity",
    "An entity is a physical, digital, conceptual, or other kind of thing with some fixed aspects; entities may be real or imaginary."
);
term!(
    INSTANTANEOUS_EVENT,
    "InstantaneousEvent",
    "An instantaneous event, or event for short, happens in the world and marks a change in the world, in its activities and in its entities. The term 'event' is commonly used in process algebra with a similar meaning. Events represent communications or interactions; they are assumed to be atomic and instantaneous."
);
term!(
    ENTITY,
    "entity",
    "The prov:entity property references an prov:Entity which influenced a resource. This property applies to an prov:EntityInfluence, which is given by a subproperty of prov:qualifiedInfluence from the influenced prov:Entity, prov:Activity or prov:Agent."
);
term!(
    GENERATED,
    "generated",
    "Relates an activity to an entity it generated; the inverse of `wasGeneratedBy`."
);
term!(
    HAD_ACTIVITY,
    "hadActivity",
    "The _optional_ Activity of an Influence, which used, generated, invalidated, or was the responsibility of some Entity. This property is _not_ used by ActivityInfluence (use prov:activity instead)."
);
term!(
    DELEGATION,
    "Delegation",
    "An instance of prov:Delegation provides additional descriptions about the binary prov:actedOnBehalfOf relation from a performing prov:Agent to some prov:Agent for whom it was performed. For example, :mixing prov:wasAssociatedWith :toddler . :toddler prov:actedOnBehalfOf :mother; prov:qualifiedDelegation [ a prov:Delegation; prov:entity :mother; :foo :bar ]."
);
term!(
    DERIVATION,
    "Derivation",
    "An instance of prov:Derivation provides additional descriptions about the binary prov:wasDerivedFrom relation from some derived prov:Entity to another prov:Entity from which it was derived. For example, :chewed_bubble_gum prov:wasDerivedFrom :unwrapped_bubble_gum; prov:qualifiedDerivation [ a prov:Derivation; prov:entity :unwrapped_bubble_gum; :foo :bar ]."
);
term!(
    END,
    "End",
    "An instance of prov:End provides additional descriptions about the binary prov:wasEndedBy relation from some ended prov:Activity to an prov:Entity that ended it. For example, :ball_game prov:wasEndedBy :buzzer; prov:qualifiedEnd [ a prov:End; prov:entity :buzzer; :foo :bar; prov:atTime '2012-03-09T08:05:08-05:00'^^xsd:dateTime ]."
);
term!(
    START,
    "Start",
    "An instance of prov:Start provides additional descriptions about the binary prov:wasStartedBy relation from some started prov:Activity to an prov:Entity that started it. For example, :foot_race prov:wasStartedBy :bang; prov:qualifiedStart [ a prov:Start; prov:entity :bang; :foo :bar; prov:atTime '2012-03-09T08:05:08-05:00'^^xsd:dateTime ] ."
);
term!(
    HAD_GENERATION,
    "hadGeneration",
    "The _optional_ Generation involved in an Entity's Derivation."
);
term!(
    HAD_MEMBER,
    "hadMember",
    "The hadMember relation links a collection entity to an entity that is a member of that collection."
);
term!(
    HAD_PLAN,
    "hadPlan",
    "The _optional_ Plan adopted by an Agent in Association with some Activity. Plan specifications are out of the scope of this specification."
);
term!(
    HAD_PRIMARY_SOURCE,
    "hadPrimarySource",
    "A primary source relates a secondary entity to an earlier, primary entity it was derived from, by copying or transcribing preserving the essential content."
);
term!(
    QUALIFIED_PRIMARY_SOURCE,
    "qualifiedPrimarySource",
    "If this Entity prov:hadPrimarySource Entity :e, then it can qualify how using prov:qualifiedPrimarySource [ a prov:PrimarySource; prov:entity :e; :foo :bar ]."
);
term!(
    HAD_ROLE,
    "hadRole",
    "The _optional_ Role that an Entity assumed in the context of an Activity. For example, :baking prov:used :spoon; prov:qualified [ a prov:Usage; prov:entity :spoon; prov:hadRole roles:mixing_implement ]."
);
term!(
    ASSOCIATION,
    "Association",
    "An instance of prov:Association provides additional descriptions about the binary prov:wasAssociatedWith relation from an prov:Activity to some prov:Agent that had some responsiblity for it. For example, :baking prov:wasAssociatedWith :baker; prov:qualifiedAssociation [ a prov:Association; prov:agent :baker; :foo :bar ]."
);
term!(
    HAD_USAGE,
    "hadUsage",
    "The _optional_ Usage involved in an Entity's Derivation."
);
term!(
    INFLUENCED,
    "influenced",
    "Relates an influencer resource to the entity, activity, or agent it influenced; the inverse of `wasInfluencedBy`."
);
term!(
    INFLUENCER,
    "influencer",
    "Subproperties of prov:influencer are used to cite the object of an unqualified PROV-O triple whose predicate is a subproperty of prov:wasInfluencedBy (e.g. prov:used, prov:wasGeneratedBy). prov:influencer is used much like rdf:object is used."
);
term!(
    INVALIDATED,
    "invalidated",
    "Relates an activity to an entity it invalidated; the inverse of `wasInvalidatedBy`."
);
term!(
    QUALIFIED_ASSOCIATION,
    "qualifiedAssociation",
    "If this Activity prov:wasAssociatedWith Agent :ag, then it can qualify the Association using prov:qualifiedAssociation [ a prov:Association; prov:agent :ag; :foo :bar ]."
);
term!(
    QUALIFIED_ATTRIBUTION,
    "qualifiedAttribution",
    "If this Entity prov:wasAttributedTo Agent :ag, then it can qualify how it was influenced using prov:qualifiedAttribution [ a prov:Attribution; prov:agent :ag; :foo :bar ]."
);
term!(
    QUALIFIED_COMMUNICATION,
    "qualifiedCommunication",
    "If this Activity prov:wasInformedBy Activity :a, then it can qualify how it was influenced using prov:qualifiedCommunication [ a prov:Communication; prov:activity :a; :foo :bar ]."
);
term!(
    QUALIFIED_DERIVATION,
    "qualifiedDerivation",
    "If this Entity prov:wasDerivedFrom Entity :e, then it can qualify how it was derived using prov:qualifiedDerivation [ a prov:Derivation; prov:entity :e; :foo :bar ]."
);
term!(
    QUALIFIED_END,
    "qualifiedEnd",
    "If this Activity prov:wasEndedBy Entity :e1, then it can qualify how it was ended using prov:qualifiedEnd [ a prov:End; prov:entity :e1; :foo :bar ]."
);
term!(
    QUALIFIED_GENERATION,
    "qualifiedGeneration",
    "If this Activity prov:generated Entity :e, then it can qualify how it performed the Generation using prov:qualifiedGeneration [ a prov:Generation; prov:entity :e; :foo :bar ]."
);
term!(
    QUALIFIED_INFLUENCE,
    "qualifiedInfluence",
    "Because prov:qualifiedInfluence is a broad relation, the more specific relations (qualifiedCommunication, qualifiedDelegation, qualifiedEnd, etc.) should be used when applicable."
);
term!(
    QUALIFIED_INVALIDATION,
    "qualifiedInvalidation",
    "If this Entity prov:wasInvalidatedBy Activity :a, then it can qualify how it was invalidated using prov:qualifiedInvalidation [ a prov:Invalidation; prov:activity :a; :foo :bar ]."
);
term!(
    QUALIFIED_QUOTATION,
    "qualifiedQuotation",
    "If this Entity prov:wasQuotedFrom Entity :e, then it can qualify how using prov:qualifiedQuotation [ a prov:Quotation; prov:entity :e; :foo :bar ]."
);
term!(
    QUALIFIED_REVISION,
    "qualifiedRevision",
    "If this Entity prov:wasRevisionOf Entity :e, then it can qualify how it was revised using prov:qualifiedRevision [ a prov:Revision; prov:entity :e; :foo :bar ]."
);
term!(
    QUALIFIED_START,
    "qualifiedStart",
    "If this Activity prov:wasStartedBy Entity :e1, then it can qualify how it was started using prov:qualifiedStart [ a prov:Start; prov:entity :e1; :foo :bar ]."
);
term!(
    QUALIFIED_USAGE,
    "qualifiedUsage",
    "If this Activity prov:used Entity :e, then it can qualify how it used it using prov:qualifiedUsage [ a prov:Usage; prov:entity :e; :foo :bar ]."
);
term!(
    USED,
    "used",
    "A prov:Entity that was used by this prov:Activity. For example, :baking prov:used :spoon, :egg, :oven ."
);
term!(
    WAS_ASSOCIATED_WITH,
    "wasAssociatedWith",
    "An prov:Agent that had some (unspecified) responsibility for the occurrence of this prov:Activity."
);
term!(
    WAS_ATTRIBUTED_TO,
    "wasAttributedTo",
    "Attribution is the ascribing of an entity to an agent."
);
term!(
    WAS_DERIVED_FROM,
    "wasDerivedFrom",
    "The more specific subproperties of prov:wasDerivedFrom (i.e., prov:wasQuotedFrom, prov:wasRevisionOf, prov:hadPrimarySource) should be used when applicable."
);
term!(
    WAS_ENDED_BY,
    "wasEndedBy",
    "End is when an activity is deemed to have ended. An end may refer to an entity, known as trigger, that terminated the activity."
);
term!(
    WAS_GENERATED_BY,
    "wasGeneratedBy",
    "The generation of an entity by an activity, marking the entity's beginning of existence and its first availability for use."
);
term!(
    WAS_INFLUENCED_BY,
    "wasInfluencedBy",
    "Because prov:wasInfluencedBy is a broad relation, its more specific subproperties (e.g. prov:wasInformedBy, prov:actedOnBehalfOf, prov:wasEndedBy, etc.) should be used when applicable."
);
term!(
    WAS_INFORMED_BY,
    "wasInformedBy",
    "An activity a2 is dependent on or informed by another activity a1, by way of some unspecified entity that is generated by a1 and used by a2."
);
term!(
    WAS_INVALIDATED_BY,
    "wasInvalidatedBy",
    "The invalidation of an entity by an activity, marking the entity's end of existence and its unavailability for further use."
);
term!(
    WAS_QUOTED_FROM,
    "wasQuotedFrom",
    "An entity is derived from an original entity by copying, or 'quoting', some or all of it."
);
term!(
    WAS_STARTED_BY,
    "wasStartedBy",
    "Start is when an activity is deemed to have started. A start may refer to an entity, known as trigger, that initiated the activity."
);
term!(
    AT_TIME,
    "atTime",
    "The time at which an InstantaneousEvent occurred, in the form of xsd:dateTime."
);
term!(
    ENDED_AT_TIME,
    "endedAtTime",
    "The time at which an activity ended. See also prov:startedAtTime."
);
term!(
    GENERATED_AT_TIME,
    "generatedAtTime",
    "The time at which an entity was completely created and is available for use."
);
term!(
    INVALIDATED_AT_TIME,
    "invalidatedAtTime",
    "The time at which an entity was invalidated (i.e., no longer usable)."
);
term!(
    STARTED_AT_TIME,
    "startedAtTime",
    "The time at which an activity started. See also prov:endedAtTime."
);
term!(
    VALUE,
    "value",
    "Provides a value that is a direct representation of an entity."
);
term!(
    ACTIVITY_INFLUENCE,
    "ActivityInfluence",
    "ActivityInfluence provides additional descriptions of an Activity's binary influence upon any other kind of resource. Instances of ActivityInfluence use the prov:activity property to cite the influencing Activity."
);
term!(
    AGENT_INFLUENCE,
    "AgentInfluence",
    "AgentInfluence provides additional descriptions of an Agent's binary influence upon any other kind of resource. Instances of AgentInfluence use the prov:agent property to cite the influencing Agent."
);
term!(
    ATTRIBUTION,
    "Attribution",
    "An instance of prov:Attribution provides additional descriptions about the binary prov:wasAttributedTo relation from an prov:Entity to some prov:Agent that had some responsible for it. For example, :cake prov:wasAttributedTo :baker; prov:qualifiedAttribution [ a prov:Attribution; prov:entity :baker; :foo :bar ]."
);
term!(
    BUNDLE,
    "Bundle",
    "Note that there are kinds of bundles (e.g. handwritten letters, audio recordings, etc.) that are not expressed in PROV-O, but can be still be described by PROV-O."
);
term!(
    COLLECTION,
    "Collection",
    "A collection is an entity that provides a structure to some constituents, which are themselves entities. These constituents are said to be member of the collections."
);
term!(
    COMMUNICATION,
    "Communication",
    "An instance of prov:Communication provides additional descriptions about the binary prov:wasInformedBy relation from an informed prov:Activity to the prov:Activity that informed it. For example, :you_jumping_off_bridge prov:wasInformedBy :everyone_else_jumping_off_bridge; prov:qualifiedCommunication [ a prov:Communication; prov:activity :everyone_else_jumping_off_bridge; :foo :bar ]."
);
term!(
    EMPTY_COLLECTION,
    "EmptyCollection",
    "An empty collection is a collection without members."
);
term!(
    ENTITY_INFLUENCE,
    "EntityInfluence",
    "EntityInfluence provides additional descriptions of an Entity's binary influence upon any other kind of resource. Instances of EntityInfluence use the prov:entity property to cite the influencing Entity."
);
term!(
    GENERATION,
    "Generation",
    "An instance of prov:Generation provides additional descriptions about the binary prov:wasGeneratedBy relation from a generated prov:Entity to the prov:Activity that generated it. For example, :cake prov:wasGeneratedBy :baking; prov:qualifiedGeneration [ a prov:Generation; prov:activity :baking; :foo :bar ]."
);
term!(
    INFLUENCE,
    "Influence",
    "An instance of prov:Influence provides additional descriptions about the binary prov:wasInfluencedBy relation from some influenced Activity, Entity, or Agent to the influencing Activity, Entity, or Agent. For example, :stomach_ache prov:wasInfluencedBy :spoon; prov:qualifiedInfluence [ a prov:Influence; prov:entity :spoon; :foo :bar ] . Because prov:Influence is a broad relation, the more specific relations (Communication, Delegation, End, etc.) should be used when applicable."
);
term!(
    INVALIDATION,
    "Invalidation",
    "An instance of prov:Invalidation provides additional descriptions about the binary prov:wasInvalidatedBy relation from an invalidated prov:Entity to the prov:Activity that invalidated it. For example, :uncracked_egg prov:wasInvalidatedBy :baking; prov:qualifiedInvalidation [ a prov:Invalidation; prov:activity :baking; :foo :bar ]."
);
term!(
    LOCATION,
    "Location",
    "A location can be an identifiable geographic place (ISO 19112), but it can also be a non-geographic place such as a directory, row, or column. As such, there are numerous ways in which location can be expressed, such as by a coordinate, address, landmark, and so forth."
);
term!(
    ORGANIZATION,
    "Organization",
    "An organization is a social or legal institution such as a company, society, etc."
);
term!(PERSON, "Person", "Person agents are people.");
term!(
    PLAN,
    "Plan",
    "There exist no prescriptive requirement on the nature of plans, their representation, the actions or steps they consist of, or their intended goals. Since plans may evolve over time, it may become necessary to track their provenance, so plans themselves are entities. Representing the plan explicitly in the provenance can be useful for various tasks: for example, to validate the execution as represented in the provenance record, to manage expectation failures, or to provide explanations."
);
term!(
    PRIMARY_SOURCE,
    "PrimarySource",
    "An instance of prov:PrimarySource provides additional descriptions about the binary prov:hadPrimarySource relation from some secondary prov:Entity to an earlier, primary prov:Entity. For example, :blog prov:hadPrimarySource :newsArticle; prov:qualifiedPrimarySource [ a prov:PrimarySource; prov:entity :newsArticle; :foo :bar ] ."
);
term!(
    QUOTATION,
    "Quotation",
    "An instance of prov:Quotation provides additional descriptions about the binary prov:wasQuotedFrom relation from some taken prov:Entity from an earlier, larger prov:Entity. For example, :here_is_looking_at_you_kid prov:wasQuotedFrom :casablanca_script; prov:qualifiedQuotation [ a prov:Quotation; prov:entity :casablanca_script; :foo :bar ]."
);
term!(
    REVISION,
    "Revision",
    "An instance of prov:Revision provides additional descriptions about the binary prov:wasRevisionOf relation from some newer prov:Entity to an earlier prov:Entity. For example, :draft_2 prov:wasRevisionOf :draft_1; prov:qualifiedRevision [ a prov:Revision; prov:entity :draft_1; :foo :bar ]."
);
term!(
    ROLE,
    "Role",
    "A role is the function of an entity or agent with respect to an activity, in the context of a usage, generation, invalidation, association, start, and end."
);
term!(
    SOFTWARE_AGENT,
    "SoftwareAgent",
    "A software agent is running software."
);
term!(
    USAGE,
    "Usage",
    "An instance of prov:Usage provides additional descriptions about the binary prov:used relation from some prov:Activity to an prov:Entity that it used. For example, :keynote prov:used :podium; prov:qualifiedUsage [ a prov:Usage; prov:entity :podium; :foo :bar ]."
);

/// All 95 terms defined in the [`prov`](self) module, in declaration order.
///
/// Exposed for introspection and exhaustiveness testing.
pub const ALL: &[NamedNodeRef<'static>] = &[
    UNQUALIFIED_FORM,
    WAS_REVISION_OF,
    AQ,
    SHARES_DEFINITION_WITH,
    DEFINITION,
    EDITORIAL_NOTE,
    INVERSE,
    CONSTRAINTS,
    SPECIALIZATION_OF,
    DM,
    CATEGORY,
    ORDER,
    EDITORS_DEFINITION,
    COMPONENT,
    QUALIFIED_FORM,
    TODO,
    N,
    ACTED_ON_BEHALF_OF,
    QUALIFIED_DELEGATION,
    AGENT,
    ACTIVITY,
    ALTERNATE_OF,
    AT_LOCATION,
    ACTIVITY_CLASS,
    AGENT_CLASS,
    ENTITY_CLASS,
    INSTANTANEOUS_EVENT,
    ENTITY,
    GENERATED,
    HAD_ACTIVITY,
    DELEGATION,
    DERIVATION,
    END,
    START,
    HAD_GENERATION,
    HAD_MEMBER,
    HAD_PLAN,
    HAD_PRIMARY_SOURCE,
    QUALIFIED_PRIMARY_SOURCE,
    HAD_ROLE,
    ASSOCIATION,
    HAD_USAGE,
    INFLUENCED,
    INFLUENCER,
    INVALIDATED,
    QUALIFIED_ASSOCIATION,
    QUALIFIED_ATTRIBUTION,
    QUALIFIED_COMMUNICATION,
    QUALIFIED_DERIVATION,
    QUALIFIED_END,
    QUALIFIED_GENERATION,
    QUALIFIED_INFLUENCE,
    QUALIFIED_INVALIDATION,
    QUALIFIED_QUOTATION,
    QUALIFIED_REVISION,
    QUALIFIED_START,
    QUALIFIED_USAGE,
    USED,
    WAS_ASSOCIATED_WITH,
    WAS_ATTRIBUTED_TO,
    WAS_DERIVED_FROM,
    WAS_ENDED_BY,
    WAS_GENERATED_BY,
    WAS_INFLUENCED_BY,
    WAS_INFORMED_BY,
    WAS_INVALIDATED_BY,
    WAS_QUOTED_FROM,
    WAS_STARTED_BY,
    AT_TIME,
    ENDED_AT_TIME,
    GENERATED_AT_TIME,
    INVALIDATED_AT_TIME,
    STARTED_AT_TIME,
    VALUE,
    ACTIVITY_INFLUENCE,
    AGENT_INFLUENCE,
    ATTRIBUTION,
    BUNDLE,
    COLLECTION,
    COMMUNICATION,
    EMPTY_COLLECTION,
    ENTITY_INFLUENCE,
    GENERATION,
    INFLUENCE,
    INVALIDATION,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PLAN,
    PRIMARY_SOURCE,
    QUOTATION,
    REVISION,
    ROLE,
    SOFTWARE_AGENT,
    USAGE,
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
        assert_eq!(ALL.len(), 95);
    }
}
