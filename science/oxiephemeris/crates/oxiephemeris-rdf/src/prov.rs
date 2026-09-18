//! PROV-O provenance: which software, from which ephemeris, produced a
//! chart.
//!
//! This workspace has always insisted on saying where a number came from —
//! every module cites its primary source and disclaims any ephemeris
//! source it did not read. PROV-O is that same discipline in machine
//! -readable form:
//!
//! ```text
//! <chart>    a prov:Entity ; prov:wasGeneratedBy <chart/activity> ;
//!                            prov:wasDerivedFrom <base/ephemeris/DE440> .
//! <activity> a prov:Activity ; prov:used <base/ephemeris/DE440> ;
//!                              prov:wasAssociatedWith <base/agent/oxiephemeris-0.1.1> .
//! <agent>    a prov:SoftwareAgent ; dct:hasVersion "0.1.1" .
//! ```
//!
//! # No generation timestamp
//!
//! PROV-O offers `prov:generatedAtTime`, and it is deliberately **not**
//! emitted. A wall-clock reading would make the graph for a given chart
//! differ on every run, which would break both the byte-identical
//! serialization this crate guarantees and the canonicalization
//! round-trip tests. A chart is a pure function of its inputs; its
//! provenance should be too. Callers who need a generation time can add
//! that triple themselves at publication time.

use oxrdf::vocab::rdfs;
use oxrdf::{vocab::rdf, Graph, NamedNode};

use crate::builder::{add, plain};
use crate::iri::IriMinter;
use crate::model::Provenance;
use crate::vocab::{dct, prov};

/// Adds the PROV-O provenance of `chart_iri` to `graph`.
pub fn add_provenance(
    graph: &mut Graph,
    chart_iri: &NamedNode,
    minter: &IriMinter,
    provenance: &Provenance,
) {
    let activity = IriMinter::part(chart_iri, "activity");
    let agent = minter.agent(&provenance.software_name, &provenance.software_version);
    let ephemeris = minter.ephemeris(&provenance.ephemeris_label);

    add(graph, chart_iri.clone(), rdf::TYPE, prov::ENTITY);
    add(
        graph,
        chart_iri.clone(),
        prov::WAS_GENERATED_BY,
        activity.clone(),
    );
    add(
        graph,
        chart_iri.clone(),
        prov::WAS_DERIVED_FROM,
        ephemeris.clone(),
    );

    add(graph, activity.clone(), rdf::TYPE, prov::ACTIVITY);
    add(graph, activity.clone(), prov::USED, ephemeris.clone());
    add(graph, activity, prov::WAS_ASSOCIATED_WITH, agent.clone());

    add(graph, agent.clone(), rdf::TYPE, prov::SOFTWARE_AGENT);
    add(
        graph,
        agent.clone(),
        rdfs::LABEL,
        plain(&format!(
            "{} {}",
            provenance.software_name, provenance.software_version
        )),
    );
    add(
        graph,
        agent,
        dct::HAS_VERSION,
        plain(&provenance.software_version),
    );

    add(graph, ephemeris.clone(), rdf::TYPE, prov::ENTITY);
    add(
        graph,
        ephemeris,
        rdfs::LABEL,
        plain(&provenance.ephemeris_label),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxrdf::TripleRef;

    fn provenance() -> Provenance {
        Provenance {
            software_name: "oxiephemeris".to_owned(),
            software_version: "0.1.1".to_owned(),
            ephemeris_label: "DE440".to_owned(),
        }
    }

    fn built() -> (Graph, NamedNode, IriMinter) {
        let minter = IriMinter::default();
        let chart = NamedNode::new_unchecked("https://example.org/id/chart/abc");
        let mut graph = Graph::new();
        add_provenance(&mut graph, &chart, &minter, &provenance());
        (graph, chart, minter)
    }

    #[test]
    fn chart_is_an_entity_generated_by_an_activity() {
        let (g, chart, _) = built();
        assert!(g.contains(TripleRef::new(chart.as_ref(), rdf::TYPE, prov::ENTITY)));
        let activity = IriMinter::part(&chart, "activity");
        assert!(g.contains(TripleRef::new(
            chart.as_ref(),
            prov::WAS_GENERATED_BY,
            activity.as_ref()
        )));
        assert!(g.contains(TripleRef::new(activity.as_ref(), rdf::TYPE, prov::ACTIVITY)));
    }

    #[test]
    fn activity_used_the_ephemeris_and_the_software_agent() {
        let (g, chart, minter) = built();
        let activity = IriMinter::part(&chart, "activity");
        let ephemeris = minter.ephemeris("DE440");
        let agent = minter.agent("oxiephemeris", "0.1.1");
        assert!(g.contains(TripleRef::new(
            activity.as_ref(),
            prov::USED,
            ephemeris.as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            activity.as_ref(),
            prov::WAS_ASSOCIATED_WITH,
            agent.as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            agent.as_ref(),
            rdf::TYPE,
            prov::SOFTWARE_AGENT
        )));
        assert!(g.contains(TripleRef::new(
            chart.as_ref(),
            prov::WAS_DERIVED_FROM,
            ephemeris.as_ref()
        )));
    }

    #[test]
    fn no_generated_at_time_is_emitted() {
        let (g, _, _) = built();
        let generated_at = NamedNode::new_unchecked("http://www.w3.org/ns/prov#generatedAtTime");
        assert!(
            !g.iter().any(|t| t.predicate == generated_at.as_ref()),
            "a wall-clock timestamp would break deterministic output"
        );
    }

    #[test]
    fn provenance_is_deterministic() {
        let (a, _, _) = built();
        let (b, _, _) = built();
        assert_eq!(a.len(), b.len());
        for triple in &a {
            assert!(b.contains(triple));
        }
    }
}
