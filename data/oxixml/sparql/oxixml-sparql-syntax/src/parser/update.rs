//! Update productions: LOAD/CLEAR/DROP/CREATE/ADD/MOVE/COPY, INSERT/DELETE DATA, DELETE WHERE and
//! the general `Modify` (DELETE/INSERT ... WHERE) operation.

use super::build::fold_graph_patterns;
use super::{Parser, R, alt};
use crate::algebra::{GraphPattern, GraphTarget, QueryDataset};
use crate::term::{
    GraphName, GraphNamePattern, GroundQuad, GroundQuadPattern, NamedNode, Quad, QuadPattern,
    TriplePattern, Variable,
};
use crate::update::{
    ClearOperation, CreateOperation, DeleteDataOperation, DeleteInsertOperation, DropOperation,
    GraphUpdateOperation, InsertDataOperation, LoadOperation,
};

impl Parser<'_> {
    pub(crate) fn update_unit(&mut self) -> R<Vec<GraphUpdateOperation>> {
        let mut ops = Vec::new();
        loop {
            self.ws();
            self.prologue()?;
            self.ws();
            let save = self.pos;
            match self.update1() {
                Ok(mut v) => ops.append(&mut v),
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
            self.ws();
            if self.try_str(";") {
                continue;
            }
            break;
        }
        Ok(ops)
    }

    /// `Update1` — a single update operation.
    ///
    /// Blank node labels are scoped to the whole update request: reusing a label in a later
    /// operation of the same request is a syntax error, so the labels seen by the previous
    /// operation are promoted into [`Parser::used_bnodes`] instead of being cleared.
    fn update1(&mut self) -> R<Vec<GraphUpdateOperation>> {
        let taken = std::mem::take(&mut self.currently_used_bnodes);
        self.used_bnodes.extend(taken);
        alt!(
            self,
            self.load_op(),
            self.clear_op(),
            self.drop_op(),
            self.add_op(),
            self.move_op(),
            self.copy_op(),
            self.create_op(),
            self.insert_data_op(),
            self.delete_data_op(),
            self.delete_where_op(),
            self.modify_op(),
        )
    }

    fn silent(&mut self) -> bool {
        self.ws();
        let s = self.try_kw("SILENT");
        self.ws();
        s
    }

    fn load_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("LOAD")?;
        let silent = self.silent();
        let source = self.iri()?;
        self.ws();
        let destination = if self.try_kw("INTO") {
            self.ws();
            GraphName::NamedNode(self.graph_ref()?)
        } else {
            GraphName::DefaultGraph
        };
        Ok(vec![GraphUpdateOperation::Load(LoadOperation {
            silent,
            source,
            destination,
        })])
    }

    fn clear_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("CLEAR")?;
        let silent = self.silent();
        let graph = self.graph_ref_all()?;
        Ok(vec![GraphUpdateOperation::Clear(ClearOperation {
            silent,
            graph,
        })])
    }

    fn drop_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("DROP")?;
        let silent = self.silent();
        let graph = self.graph_ref_all()?;
        Ok(vec![GraphUpdateOperation::Drop(DropOperation {
            silent,
            graph,
        })])
    }

    fn create_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("CREATE")?;
        let silent = self.silent();
        let graph = self.graph_ref()?;
        Ok(vec![GraphUpdateOperation::Create(CreateOperation {
            silent,
            graph,
        })])
    }

    fn add_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("ADD")?;
        let _ = self.silent();
        let from = self.graph_or_default()?;
        self.ws();
        self.kw("TO")?;
        self.ws();
        let to = self.graph_or_default()?;
        if from == to {
            Ok(Vec::new())
        } else {
            Ok(vec![copy_graph(from, to)])
        }
    }

    fn move_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("MOVE")?;
        let silent = self.silent();
        let from = self.graph_or_default()?;
        self.ws();
        self.kw("TO")?;
        self.ws();
        let to = self.graph_or_default()?;
        if from == to {
            Ok(Vec::new())
        } else {
            Ok(vec![
                GraphUpdateOperation::Drop(DropOperation {
                    silent: true,
                    graph: to.clone().into(),
                }),
                copy_graph(from.clone(), to),
                GraphUpdateOperation::Drop(DropOperation {
                    silent,
                    graph: from.into(),
                }),
            ])
        }
    }

    fn copy_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("COPY")?;
        let _ = self.silent();
        let from = self.graph_or_default()?;
        self.ws();
        self.kw("TO")?;
        self.ws();
        let to = self.graph_or_default()?;
        if from == to {
            Ok(Vec::new())
        } else {
            Ok(vec![
                GraphUpdateOperation::Drop(DropOperation {
                    silent: true,
                    graph: to.clone().into(),
                }),
                copy_graph(from, to),
            ])
        }
    }

    fn insert_data_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("INSERT")?;
        self.ws();
        self.kw("DATA")?;
        self.ws();
        let quads = self.quad_data()?;
        let data = quads
            .into_iter()
            .map(Quad::try_from)
            .collect::<Result<Vec<_>, ()>>();
        match data {
            Ok(data) => Ok(vec![GraphUpdateOperation::InsertData(
                InsertDataOperation { data },
            )]),
            Err(()) => self.fatal("Variables are not allowed in INSERT DATA"),
        }
    }

    fn delete_data_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("DELETE")?;
        self.ws();
        self.kw("DATA")?;
        self.ws();
        let quads = self.quad_data()?;
        let data = quads
            .into_iter()
            .map(|q| GroundQuad::try_from(Quad::try_from(q)?))
            .collect::<Result<Vec<_>, ()>>();
        match data {
            Ok(data) => Ok(vec![GraphUpdateOperation::DeleteData(
                DeleteDataOperation { data },
            )]),
            Err(()) => self.fatal("Variables and blank nodes are not allowed in DELETE DATA"),
        }
    }

    fn delete_where_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        self.kw("DELETE")?;
        self.ws();
        self.kw("WHERE")?;
        self.ws();
        let quads = self.quad_pattern()?;
        // One `GraphPattern::Graph`/`Bgp` element per quad, joined into a left-deep tree exactly
        // like `build_bgp`'s split elements: `fold_graph_patterns` charges the existing recursion
        // guard once per join level, so a `DELETE WHERE` with a very large quad pattern is rejected
        // before the oversized tree it would have produced is ever constructed.
        let elements = quads.iter().map(|q| {
            let bgp = GraphPattern::Bgp {
                patterns: vec![TriplePattern::new(
                    q.subject.clone(),
                    q.predicate.clone(),
                    q.object.clone(),
                )],
            };
            match &q.graph_name {
                GraphNamePattern::NamedNode(g) => GraphPattern::Graph {
                    name: g.clone().into(),
                    inner: Box::new(bgp),
                },
                GraphNamePattern::DefaultGraph => bgp,
                GraphNamePattern::Variable(g) => GraphPattern::Graph {
                    name: g.clone().into(),
                    inner: Box::new(bgp),
                },
            }
        });
        let pattern = fold_graph_patterns(self, elements)?;
        let delete = quads
            .into_iter()
            .map(GroundQuadPattern::try_from)
            .collect::<Result<Vec<_>, ()>>();
        match delete {
            Ok(delete) => Ok(vec![GraphUpdateOperation::DeleteInsert(
                DeleteInsertOperation {
                    delete,
                    insert: Vec::new(),
                    using: None,
                    pattern: Box::new(pattern),
                },
            )]),
            Err(()) => self.fatal("Blank nodes are not allowed in DELETE WHERE"),
        }
    }

    fn modify_op(&mut self) -> R<Vec<GraphUpdateOperation>> {
        let with = if self.try_kw("WITH") {
            self.ws();
            let iri = self.iri()?;
            self.ws();
            Some(iri)
        } else {
            None
        };
        // The `DELETE`/`INSERT` templates form their own blank node scope: only the `WHERE`
        // clause contributes labels to the request-wide scope, exactly like in `spargebra`.
        let outer_used = std::mem::take(&mut self.used_bnodes);
        let outer_current = std::mem::take(&mut self.currently_used_bnodes);
        let clauses = self.delete_insert_clauses();
        self.used_bnodes = outer_used;
        self.currently_used_bnodes = outer_current;
        let (mut delete, mut insert) = clauses?;

        // USING clauses.
        let mut default = Vec::new();
        let mut named = Vec::new();
        let mut has_using = false;
        loop {
            self.ws();
            let save = self.pos;
            if !self.try_kw("USING") {
                self.pos = save;
                break;
            }
            self.ws();
            if self.try_kw("NAMED") {
                self.ws();
                named.push(self.iri()?);
            } else {
                default.push(self.iri()?);
            }
            has_using = true;
        }

        self.ws();
        self.kw("WHERE")?;
        self.ws();
        let pattern = self.group_graph_pattern()?;

        let mut using = if has_using {
            Some(QueryDataset {
                default,
                named: Some(named),
            })
        } else {
            None
        };

        if let Some(with) = with {
            delete = delete
                .into_iter()
                .map(|q| {
                    if q.graph_name == GraphNamePattern::DefaultGraph {
                        GroundQuadPattern {
                            subject: q.subject,
                            predicate: q.predicate,
                            object: q.object,
                            graph_name: with.clone().into(),
                        }
                    } else {
                        q
                    }
                })
                .collect();
            insert = insert
                .into_iter()
                .map(|q| {
                    if q.graph_name == GraphNamePattern::DefaultGraph {
                        QuadPattern {
                            subject: q.subject,
                            predicate: q.predicate,
                            object: q.object,
                            graph_name: with.clone().into(),
                        }
                    } else {
                        q
                    }
                })
                .collect();
            if using.is_none() {
                using = Some(QueryDataset {
                    default: vec![with],
                    named: None,
                });
            }
        }

        Ok(vec![GraphUpdateOperation::DeleteInsert(
            DeleteInsertOperation {
                delete,
                insert,
                using,
                pattern: Box::new(pattern),
            },
        )])
    }

    /// `( DeleteClause InsertClause? | InsertClause )` of the `Modify` production.
    fn delete_insert_clauses(&mut self) -> R<(Vec<GroundQuadPattern>, Vec<QuadPattern>)> {
        if let Some(delete) = self.opt_delete_clause()? {
            self.ws();
            let insert = self.opt_insert_clause()?.unwrap_or_default();
            Ok((delete, insert))
        } else if let Some(insert) = self.opt_insert_clause()? {
            Ok((Vec::new(), insert))
        } else {
            self.mk_err("expected DELETE or INSERT clause")
        }
    }

    fn opt_delete_clause(&mut self) -> R<Option<Vec<GroundQuadPattern>>> {
        let save = self.pos;
        if !self.try_kw("DELETE") {
            self.pos = save;
            return Ok(None);
        }
        self.ws();
        let quads = self.quad_pattern()?;
        let delete = quads
            .into_iter()
            .map(GroundQuadPattern::try_from)
            .collect::<Result<Vec<_>, ()>>();
        match delete {
            Ok(v) => Ok(Some(v)),
            Err(()) => self.fatal("Blank nodes are not allowed in DELETE clauses"),
        }
    }

    fn opt_insert_clause(&mut self) -> R<Option<Vec<QuadPattern>>> {
        let save = self.pos;
        if !self.try_kw("INSERT") {
            self.pos = save;
            return Ok(None);
        }
        self.ws();
        Ok(Some(self.quad_pattern()?))
    }

    // --- quads --------------------------------------------------------------

    fn quad_pattern(&mut self) -> R<Vec<QuadPattern>> {
        self.eat("{")?;
        self.ws();
        let quads = self.quads()?;
        self.ws();
        self.eat("}")?;
        Ok(quads)
    }

    fn quad_data(&mut self) -> R<Vec<QuadPattern>> {
        self.quad_pattern()
    }

    fn quads(&mut self) -> R<Vec<QuadPattern>> {
        let mut result = Vec::new();
        // Optional leading default-graph triples.
        self.push_default_triples(&mut result)?;
        loop {
            self.ws();
            let save = self.pos;
            match self.quads_not_triples() {
                Ok(mut q) => {
                    result.append(&mut q);
                    self.ws();
                    self.try_str(".");
                    self.push_default_triples(&mut result)?;
                }
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    break;
                }
            }
        }
        Ok(result)
    }

    fn push_default_triples(&mut self, result: &mut Vec<QuadPattern>) -> R<()> {
        let save = self.pos;
        let mut triples = Vec::new();
        match self.triples_template(&mut triples) {
            Ok(()) => {
                for t in triples {
                    result.push(QuadPattern::new(
                        t.subject,
                        t.predicate,
                        t.object,
                        GraphNamePattern::DefaultGraph,
                    ));
                }
                Ok(())
            }
            Err(e) if e.fatal => Err(e),
            Err(_) => {
                self.pos = save;
                Ok(())
            }
        }
    }

    fn quads_not_triples(&mut self) -> R<Vec<QuadPattern>> {
        self.kw("GRAPH")?;
        self.ws();
        let g = self.var_or_iri()?;
        let graph: GraphNamePattern = g.into();
        self.ws();
        self.eat("{")?;
        self.ws();
        let mut triples = Vec::new();
        let save = self.pos;
        if self.triples_template(&mut triples).is_err() {
            self.pos = save;
            triples.clear();
        }
        self.ws();
        self.eat("}")?;
        Ok(triples
            .into_iter()
            .map(|t| QuadPattern::new(t.subject, t.predicate, t.object, graph.clone()))
            .collect())
    }

    // --- graph references ---------------------------------------------------

    fn graph_or_default(&mut self) -> R<GraphName> {
        self.ws();
        if self.try_kw("DEFAULT") {
            Ok(GraphName::DefaultGraph)
        } else {
            let save = self.pos;
            self.try_kw("GRAPH");
            self.ws();
            let iri = match self.iri() {
                Ok(i) => i,
                Err(e) if e.fatal => return Err(e),
                Err(_) => {
                    self.pos = save;
                    return self.mk_err("expected graph");
                }
            };
            Ok(GraphName::NamedNode(iri))
        }
    }

    fn graph_ref(&mut self) -> R<NamedNode> {
        self.kw("GRAPH")?;
        self.ws();
        self.iri()
    }

    fn graph_ref_all(&mut self) -> R<GraphTarget> {
        self.ws();
        if self.try_kw("DEFAULT") {
            Ok(GraphTarget::DefaultGraph)
        } else if self.try_kw("NAMED") {
            Ok(GraphTarget::NamedGraphs)
        } else if self.try_kw("ALL") {
            Ok(GraphTarget::AllGraphs)
        } else {
            Ok(self.graph_ref()?.into())
        }
    }
}

/// Rewrites a graph copy as a `DeleteInsert` operation, per the update spec.
fn copy_graph(from: GraphName, to: GraphName) -> GraphUpdateOperation {
    let bgp = GraphPattern::Bgp {
        patterns: vec![TriplePattern::new(
            Variable::new_unchecked("s"),
            Variable::new_unchecked("p"),
            Variable::new_unchecked("o"),
        )],
    };
    let to_pattern: GraphNamePattern = to.into();
    GraphUpdateOperation::DeleteInsert(DeleteInsertOperation {
        delete: Vec::new(),
        insert: vec![QuadPattern::new(
            Variable::new_unchecked("s"),
            Variable::new_unchecked("p"),
            Variable::new_unchecked("o"),
            to_pattern,
        )],
        using: None,
        pattern: Box::new(match from {
            GraphName::NamedNode(from) => GraphPattern::Graph {
                name: from.into(),
                inner: Box::new(bgp),
            },
            GraphName::DefaultGraph => bgp,
        }),
    })
}
