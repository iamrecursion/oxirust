//! The instruction dispatcher and the node-construction instructions.
//!
//! [`Transformer::run_instruction`] is the interpreter's central `match`; the methods below
//! it are the constructors and helpers that a single instruction reaches for, and the free
//! functions resolve the names an `xsl:element`, `xsl:attribute` or `xsl:namespace` computes
//! at run time.

use super::super::{Bindings, Context, Emitter, Transformer};
use super::Flow;
use crate::compile::{Instruction, LiteralElement};
use crate::error::{XsltError, XsltResult};
use crate::names::XSLT_NAMESPACE;
use crate::output::OutputProperties;
use oxixml_infoset::{ExpandedName, Item, NodeKind, Sequence, XdmNode};
use oxixml_qname::{XML_NAMESPACE, XMLNS_NAMESPACE, is_ncname};
use std::sync::Arc;

/// The error a terminating `xsl:message` raises when its `error-code` is absent, cannot be
/// resolved, or does not even lexically look like a `QName` —
/// [XSLT 3.0 §6.7](https://www.w3.org/TR/xslt-30/#element-message)'s default `err:XTMM9000`.
fn default_error_code() -> (Option<String>, String) {
    (
        Some(crate::error::ERROR_NAMESPACE.to_owned()),
        "XTMM9000".to_owned(),
    )
}

/// The string value of a sequence, as `xsl:assert` and `xsl:message` form their description.
fn sequence_text(sequence: &Sequence) -> XsltResult<String> {
    let mut out = String::new();
    for item in sequence.iter() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&item.string_value()?);
    }
    Ok(out)
}

impl Transformer {
    pub(super) fn run_instruction(
        &mut self,
        instruction: &Instruction,
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        self.charge()?;
        // An instruction the compiler gave a position to hands it to whatever it raises, so that
        // an `xsl:catch` body can say where the failure is written; `XsltError::at` keeps the
        // innermost attachment, which is the one XSLT 3.0 §8.3 asks for. The wrapper is peeled
        // here rather than dispatched through, because the dispatcher's own frame is large and a
        // second one for every located instruction would halve the recursion depth that fits on
        // a thread — the wrapper is not an instruction of the stylesheet's, and it costs nothing.
        let mut inner = instruction;
        let mut position = None;
        while let Instruction::Positioned {
            position: written_at,
            inner: nested,
        } = inner
        {
            position = position.or(Some(*written_at));
            inner = nested.as_ref();
        }
        let outcome = self.dispatch_instruction(inner, context, out);
        match position {
            Some(written_at) => outcome.map_err(|error| error.at(written_at)),
            None => outcome,
        }
    }

    /// Runs one instruction, without charging the budget for it — the caller has done that.
    fn dispatch_instruction(
        &mut self,
        instruction: &Instruction,
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        match instruction {
            // `run_instruction` peels the provenance wrapper before dispatching, so this arm
            // stands for exhaustiveness; delegating keeps it right whatever else arrives here.
            Instruction::Positioned { position, inner } => self
                .run_instruction(inner, context, out)
                .map_err(|error| error.at(*position)),
            Instruction::Variable(_) => Ok(Flow::Normal),
            Instruction::Text {
                text,
                disable_escaping,
            } => {
                out.text_node(text, *disable_escaping && self.escaping_may_be_disabled());
                Ok(Flow::Normal)
            }
            Instruction::TextTemplate {
                template,
                disable_escaping,
            } => {
                let text = self.evaluate_avt(template, context)?;
                out.text(&text, *disable_escaping && self.escaping_may_be_disabled());
                Ok(Flow::Normal)
            }
            Instruction::Literal(literal) => self.literal_element(literal, context, out),
            Instruction::ValueOf {
                select,
                content,
                separator,
                disable_escaping,
            } => {
                let separator = match separator {
                    Some(avt) => self.evaluate_avt(avt, context)?,
                    None if select.is_some()
                        && self.ir().version != crate::compile::XsltVersion::V10 =>
                    {
                        " ".to_owned()
                    }
                    None => String::new(),
                };
                let text = match select {
                    Some(select) => self.evaluate_string(select, context, &separator)?,
                    None => self.run_to_simple_content(content, context, &separator)?,
                };
                out.text_node(&text, *disable_escaping && self.escaping_may_be_disabled());
                Ok(Flow::Normal)
            }
            Instruction::Sequence { select, content } => {
                match select {
                    Some(select) => {
                        let value = self.evaluate(select, context)?;
                        let sequence = value.into_sequence()?;
                        for item in sequence.iter() {
                            out.item(item)?;
                        }
                    }
                    None => {
                        let (_, sequence) = self.run_to_sequence(content, None, context)?;
                        for item in sequence.iter() {
                            out.item(item)?;
                        }
                    }
                }
                Ok(Flow::Normal)
            }
            Instruction::CopyOf {
                select,
                copy_namespaces,
            } => {
                let value = self.evaluate(select, context)?;
                for item in value.into_sequence()?.iter() {
                    match item {
                        Item::Node(node) => out.copy_node_with(node, *copy_namespaces)?,
                        other => out.item(other)?,
                    }
                }
                Ok(Flow::Normal)
            }
            Instruction::Copy {
                select,
                content,
                use_attribute_sets,
                copy_namespaces,
                inherit_namespaces,
            } => {
                let scoped = match select {
                    Some(select) => {
                        let value = self.evaluate(select, context)?;
                        // XSLT 3.0 §11.9.1: the `select` expression of `xsl:copy` must yield
                        // at most one item.
                        if value.len() > 1 {
                            return Err(XsltError::dynamic(
                                "XTTE3180",
                                "the select expression of xsl:copy produced more than one item",
                            ));
                        }
                        let Some(item) = value.first().and_then(|item| item.as_item().cloned())
                        else {
                            // An empty `select` copies nothing at all.
                            return Ok(Flow::Normal);
                        };
                        let mut scoped = context.clone();
                        scoped.item = Some(item);
                        scoped.position = 1;
                        scoped.size = 1;
                        // XSLT 3.0 §5.5: an instruction that changes the focus leaves the
                        // current template rule absent, so an `xsl:next-match` inside is
                        // `XTDE0560`.
                        scoped.rule = None;
                        scoped
                    }
                    None => context.clone(),
                };
                self.copy(
                    content,
                    use_attribute_sets,
                    *copy_namespaces,
                    *inherit_namespaces,
                    &scoped,
                    out,
                )
            }
            Instruction::ApplyTemplates {
                select,
                mode,
                sort,
                params,
            } => self.apply_templates(select.as_ref(), mode, sort, params, context, out),
            Instruction::ApplyImports { params } => self.apply_next(params, context, out, true),
            Instruction::NextMatch { params } => self.apply_next(params, context, out, false),
            Instruction::CallTemplate { name, params } => {
                self.call_template(name, params, context, out)
            }
            Instruction::ForEach {
                select,
                sort,
                content,
            } => {
                let value = self.evaluate(select, context)?;
                let mut items: Vec<Item> = value.into_sequence()?.into_items();
                if !sort.is_empty() {
                    items = self.sort_items(items, sort, context)?;
                }
                let size = items.len();
                for (offset, item) in items.into_iter().enumerate() {
                    let scoped = context.with_focus(item, offset + 1, size);
                    let flow = self.run_sequence(content, &scoped, out)?;
                    if !flow.is_normal() {
                        return Ok(flow);
                    }
                }
                Ok(Flow::Normal)
            }
            Instruction::ForEachGroup {
                select,
                grouping,
                composite,
                collation,
                sort,
                content,
            } => self.for_each_group(
                select,
                grouping,
                *composite,
                collation.as_ref(),
                sort,
                content,
                context,
                out,
            ),
            Instruction::If { test, content } => {
                if self.evaluate_boolean(test, context)? {
                    return self.run_sequence(content, context, out);
                }
                Ok(Flow::Normal)
            }
            Instruction::Choose { branches } => {
                for (test, content) in branches {
                    let taken = match test {
                        Some(test) => self.evaluate_boolean(test, context)?,
                        None => true,
                    };
                    if taken {
                        return self.run_sequence(content, context, out);
                    }
                }
                Ok(Flow::Normal)
            }
            Instruction::Element {
                name,
                namespace,
                prefixes,
                use_attribute_sets,
                inherit_namespaces,
                content,
            } => {
                let lexical = self.evaluate_avt(name, context)?;
                let namespace = match namespace {
                    Some(avt) => Some(self.evaluate_avt(avt, context)?),
                    None => None,
                };
                let (expanded, prefix) =
                    resolve_computed_name(&lexical, namespace.as_deref(), prefixes, false)?;
                out.start_element_with(expanded, prefix, *inherit_namespaces);
                self.apply_attribute_sets(use_attribute_sets, context, out)?;
                let flow = self.run_sequence(content, context, out)?;
                out.end_element()?;
                Ok(flow)
            }
            Instruction::Attribute {
                name,
                namespace,
                prefixes,
                select,
                separator,
                content,
            } => {
                let lexical = self.evaluate_avt(name, context)?;
                let namespace = match namespace {
                    Some(avt) => Some(self.evaluate_avt(avt, context)?),
                    None => None,
                };
                let (expanded, prefix) =
                    resolve_computed_name(&lexical, namespace.as_deref(), prefixes, true)?;
                let separator = match separator {
                    Some(avt) => self.evaluate_avt(avt, context)?,
                    None if select.is_some() => " ".to_owned(),
                    None => String::new(),
                };
                let value = match select {
                    Some(select) => self.evaluate_string(select, context, &separator)?,
                    None => self.run_to_simple_content(content, context, &separator)?,
                };
                out.attribute(expanded, prefix, value)?;
                Ok(Flow::Normal)
            }
            Instruction::Namespace {
                name,
                select,
                content,
            } => {
                let prefix = self.evaluate_avt(name, context)?;
                // XSLT 3.0 §5.7.2: instructions without a `separator` attribute always use
                // a single space, whether the content came from `select` or from a sequence
                // constructor.
                let uri = match select {
                    Some(select) => self.evaluate_string(select, context, " ")?,
                    None => self.run_to_simple_content(content, context, " ")?,
                };
                check_namespace_instruction(&prefix, &uri)?;
                out.namespace((!prefix.is_empty()).then_some(prefix), &uri)?;
                Ok(Flow::Normal)
            }
            Instruction::Comment { select, content } => {
                let text = match select {
                    Some(select) => self.evaluate_string(select, context, " ")?,
                    None => self.run_to_simple_content(content, context, " ")?,
                };
                // XSLT 3.0 §11.8 replaced the 2.0 error with a recovery: "the processor must
                // insert a space after each hyphen that is followed by another hyphen or that
                // ends the comment".
                out.comment(&space_out_hyphens(&text));
                Ok(Flow::Normal)
            }
            Instruction::ProcessingInstruction {
                name,
                select,
                content,
            } => {
                let target = self.evaluate_avt(name, context)?;
                // XSLT 3.0 §11.8: the effective name must be both an `NCName` and a
                // `PITarget`, and XML 1.0 §2.6 excludes the three case variations of `xml`
                // from `PITarget`. The `NCName` half is checked where the node is built; this
                // is the half that is about the target's spelling alone.
                if target.eq_ignore_ascii_case("xml") {
                    return Err(XsltError::dynamic(
                        "XTDE0890",
                        format!("`{target}` is not a usable processing-instruction target"),
                    ));
                }
                let text = match select {
                    Some(select) => self.evaluate_string(select, context, " ")?,
                    None => self.run_to_simple_content(content, context, " ")?,
                };
                // XSLT 3.0 §11.9: a `?>` in the content is repaired by inserting a space
                // between the two characters, rather than reported.
                let text = text.replace("?>", "? >");
                // The whitespace the content begins with is not part of it. XML 1.0 §2.6 spells
                // a processing instruction `<?target S content?>`, and XML Infoset §2.6 defines
                // the `[content]` property as "the content of the PI, excluding the target and
                // any white space immediately following it" — so leading whitespace could not
                // survive a serialization and could not be read back. `construct-node-021` and
                // `seqtor-037a`/`-037c`/`-037d` assert exactly that.
                out.processing_instruction(&target, text.trim_start())?;
                Ok(Flow::Normal)
            }
            Instruction::Document { content } => {
                // XSLT 3.0 §11.9: the instruction constructs a document node, which at the
                // top level of a sequence is itself the item.
                let tree = self.run_to_tree(content, None, context)?;
                out.append_node(&tree.root())?;
                Ok(Flow::Normal)
            }
            Instruction::Number(spec) => {
                let text = self.format_number_instruction(spec, context)?;
                out.text(&text, false);
                Ok(Flow::Normal)
            }
            Instruction::Message {
                select,
                content,
                terminate,
                error_code,
            } => {
                // XSLT 3.0 §6.7 (suite: bug 30283 — "the XSLT 3.0 spec is clear that the
                // transformation must not fail"): a dynamic error evaluating the message's own
                // `select` or content — dividing by zero, or a sequence `xsl:value-of`'s string
                // reduction cannot convert, such as a map, an array or a function item — must
                // not abort the transformation. The instruction recovers, treating the message
                // text as unavailable ("the message content is undefined", as the suite's own
                // map/function-item cases put it), rather than propagating the failure.
                //
                // The message is *both* a string, which is what a message sink receives, and a
                // sequence, which [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#element-catch)
                // hands a catching `xsl:catch` as `$err:value` — message-0501 reads an element
                // with a comment child back out of it, so the nodes have to be kept alongside
                // the string rather than replaced by it.
                let produced = self.message_content(select.as_ref(), content, context);
                let (text, value) = produced.unwrap_or_default();
                self.runtime.messages.borrow_mut().push(text.clone());
                let terminate = self.evaluate_avt(terminate, context)?;
                // XSLT 3.0 §3.7: an attribute written as an attribute value template must
                // still produce one of the values the attribute permits; anything else is
                // `XTDE0030`.
                let terminate = match terminate.trim() {
                    "yes" | "true" | "1" => true,
                    "no" | "false" | "0" => false,
                    other => {
                        return Err(XsltError::dynamic(
                            "XTDE0030",
                            format!("`{other}` is not a value the `terminate` attribute permits"),
                        ));
                    }
                };
                if !terminate {
                    return Ok(Flow::Normal);
                }
                // XSLT 3.0 §6.7: a terminating message raises `XTMM9000` unless `error-code`
                // names another error — and falls back to `XTMM9000` too when `error-code`
                // cannot be resolved at all, for the same "must not abort" reason as above.
                let (namespace, local) = match error_code {
                    Some(code) => self.resolve_error_code(code, context),
                    None => Ok(default_error_code()),
                }
                .unwrap_or_else(|_| default_error_code());
                Err(XsltError::named(namespace, local, text).with_value(value))
            }
            Instruction::AnalyzeString {
                select,
                regex,
                flags,
                matching,
                non_matching,
            } => self.analyze_string(select, regex, flags, matching, non_matching, context, out),
            Instruction::PerformSort {
                select,
                sort,
                content,
            } => {
                let items = match select {
                    Some(select) => self
                        .evaluate(select, context)?
                        .into_sequence()?
                        .into_items(),
                    None => self.run_to_sequence(content, None, context)?.1.into_items(),
                };
                let sorted = self.sort_items(items, sort, context)?;
                for item in &sorted {
                    out.item(item)?;
                }
                Ok(Flow::Normal)
            }
            Instruction::ResultDocument {
                href,
                output,
                deferred,
                content,
            } => {
                let settled;
                let output = match deferred {
                    None => output,
                    Some(deferred) => {
                        settled = self.settle_output(output, deferred, context)?;
                        &settled
                    }
                };
                // XSLT 3.0 §23.1: the instruction may not be evaluated in temporary output
                // state.
                if self.runtime.temporary_output.get() > 0 {
                    return Err(XsltError::dynamic(
                        "XTDE1480",
                        "xsl:result-document was evaluated in temporary output state",
                    ));
                }
                let href = match href {
                    Some(avt) => self.evaluate_avt(avt, context)?,
                    None => String::new(),
                };
                if href.is_empty() {
                    // `href=""` — and no `href` at all — names the principal result. XSLT 3.0
                    // §23.1 makes that a final result tree of its own, not content of whatever
                    // tree is being built around the instruction, so a second claim on the
                    // destination — by another such instruction, or by the transformation
                    // writing to the principal result directly — is `XTDE1490`.
                    if self.runtime.principal_claimed.replace(true) {
                        return Err(XsltError::dynamic(
                            "XTDE1490",
                            "two result documents were written to the principal result destination",
                        ));
                    }
                    *self.runtime.principal_output.borrow_mut() = Some(Arc::clone(output));
                    self.runtime
                        .open_result_document_uris
                        .borrow_mut()
                        .push(None);
                    let separator = output.item_separator.as_deref();
                    let tree = self.principal_destination(content, context, output, separator);
                    self.runtime.open_result_document_uris.borrow_mut().pop();
                    *self.runtime.principal_result.borrow_mut() = Some(tree?);
                    return Ok(Flow::Normal);
                }
                // XSLT 3.0 §23.1: "the href attribute is interpreted as a relative URI
                // reference, which is resolved against the base output URI". A caller that
                // declared none leaves the stylesheet's own base URI as the only thing to
                // resolve against — which locates the file the stylesheet meant, and is what
                // this processor did before a base output URI could be declared at all.
                let base = self
                    .runtime
                    .base_output_uri()
                    .or_else(|| self.runtime.base_uri.clone());
                let resolved = match &base {
                    Some(base) => crate::resolve::resolve_uri(&href, Some(base)),
                    None => href,
                };
                // XSLT 3.0 §23.1: a transformation may not write a final result tree to a URI
                // it also reads a document from.
                if self.runtime.documents_read.was_read(&resolved) {
                    return Err(XsltError::dynamic(
                        "XTDE1500",
                        format!("`{resolved}` is both read and written by this transformation"),
                    ));
                }
                // XSLT 3.0 §18.9: `fn:current-output-uri()` reports this destination while
                // the instruction's content is being evaluated; pop it again — even on
                // failure — so a sibling or an enclosing instruction does not inherit it.
                self.runtime
                    .open_result_document_uris
                    .borrow_mut()
                    .push(Some(resolved.clone()));
                // XSLT 3.0 §26: the `json` and `adaptive` methods build no final result tree,
                // so the body's *raw result* is what is serialized — `result-document-0303`
                // writes an `xsl:map`, which no tree could hold.
                let separator = output.item_separator.as_deref();
                let text = if crate::serialize::builds_a_tree(output) {
                    self.run_to_final_tree(content, context, separator)
                        .and_then(|tree| super::super::serialize(&tree, output))
                } else {
                    self.run_to_raw_result(content, context, separator)
                        .and_then(|raw| crate::serialize::serialize_raw(&raw, output))
                };
                self.runtime.open_result_document_uris.borrow_mut().pop();
                let text = text?;
                // XSLT 3.0 §23.1: no two final result trees may share a URI.
                if self
                    .runtime
                    .result_documents
                    .borrow()
                    .iter()
                    .any(|(existing, _)| *existing == resolved)
                {
                    return Err(XsltError::dynamic(
                        "XTDE1490",
                        format!("two result documents were written to `{resolved}`"),
                    ));
                }
                self.runtime
                    .result_documents
                    .borrow_mut()
                    .push((resolved, text));
                Ok(Flow::Normal)
            }
            Instruction::Try {
                select,
                content,
                catches,
            } => self.run_try(select.as_ref(), content, catches, context, out),
            Instruction::Assert {
                test,
                select,
                content,
                error_code,
            } => {
                if self
                    .evaluate(test, context)?
                    .into_sequence()?
                    .effective_boolean_value()?
                {
                    return Ok(Flow::Normal);
                }
                let (namespace, local) = match error_code {
                    Some(code) => self.resolve_error_code(code, context)?,
                    // [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#assert) makes `XTMM9001`
                    // the default code of a failed assertion.
                    None => (
                        Some(crate::error::ERROR_NAMESPACE.to_owned()),
                        "XTMM9001".to_owned(),
                    ),
                };
                let description = match (select, content.is_empty()) {
                    (Some(select), _) => {
                        let value = self.evaluate(select, context)?.into_sequence()?;
                        sequence_text(&value)?
                    }
                    (None, false) => {
                        let (_, items) = self.run_to_sequence(content, None, context)?;
                        sequence_text(&items)?
                    }
                    (None, true) => "an xsl:assert test was not satisfied".to_owned(),
                };
                Err(XsltError::named(namespace, local, description))
            }
            Instruction::Iterate {
                select,
                params,
                content,
                on_completion,
            } => self.iterate(select, params, content, on_completion, context, out),
            Instruction::NextIteration { params } => {
                let mut values = Vec::with_capacity(params.len());
                for param in params {
                    values.push((param.name.clone(), self.with_param_value(param, context)?));
                }
                Ok(Flow::NextIteration(values))
            }
            Instruction::Break { select, content } => {
                let value = match select {
                    Some(select) => self.evaluate(select, context)?.into_sequence()?,
                    None => self.run_to_sequence(content, None, context)?.1,
                };
                Ok(Flow::Break(value))
            }
            Instruction::OnEmpty {
                select, content, ..
            } => {
                // Reached only when the instruction stands alone; `run_sequence` handles the
                // usual case, where its siblings decide whether it fires.
                self.emit_selected(select.as_ref(), content, context, out)
            }
            Instruction::WherePopulated { select, content } => {
                // The operand is judged as the *sequence* XSLT 3.0 §8.4.1 describes, not as a
                // tree: its `deemed-empty` predicate is defined over items, and a map or an
                // array is one of the item kinds it names — which a tree-shaped probe could not
                // even hold (`XTDE0450`). The suite's coco-103 is exactly that case.
                let mut probe = Emitter::sequence();
                let quiet = self.runtime.messages.borrow().len();
                self.emit_selected(select.as_ref(), content, context, &mut probe)?;
                self.runtime.messages.borrow_mut().truncate(quiet);
                let (_, items) = probe.finish()?;
                if super::super::populated::is_populated(&items) {
                    return self.emit_selected(select.as_ref(), content, context, out);
                }
                Ok(Flow::Normal)
            }
            Instruction::Map { content } => {
                let map = self.evaluate_map(content, context)?;
                out.item(&Item::Map(map))?;
                Ok(Flow::Normal)
            }
            Instruction::MapEntry {
                key,
                select,
                content,
            } => {
                let entry = self.evaluate_map_entry(key, select.as_ref(), content, context)?;
                out.item(&Item::Map(entry))?;
                Ok(Flow::Normal)
            }
            Instruction::BackwardsCompatible { name } => Err(XsltError::dynamic(
                "XTDE0160",
                format!(
                    "`{name}` has an effective version of 1.0, and this processor does not \
                     implement backwards compatible behaviour for XSLT 1.0"
                ),
            )),
            Instruction::Fallback => Ok(Flow::Normal),
            Instruction::Unknown {
                name,
                has_fallback,
                fallback,
            } => {
                if !has_fallback {
                    return Err(XsltError::dynamic(
                        "XTDE1450",
                        format!("`{name}` is not implemented and has no xsl:fallback"),
                    ));
                }
                self.run_sequence(fallback, context, out)
            }
        }
    }

    /// What an `xsl:message` produced: its text, and the sequence a catching `xsl:catch` reads
    /// as `$err:value`.
    ///
    /// The two are formed exactly as [XSLT 3.0 §6.7](https://www.w3.org/TR/xslt-30/#element-message)
    /// prescribes and as this instruction has always formed the text: a `select` attribute is
    /// reduced by the simple-content rule with a single space between adjacent atomic values,
    /// while a sequence constructor builds a tree, whose string value runs the parts together.
    /// The value is the same content seen as a sequence — the children of that tree, or the
    /// items the expression selected.
    ///
    /// # Errors
    ///
    /// Whatever the expression or the constructor raises. The caller recovers from it: §6.7 (and
    /// the suite's bug 30283) require a message whose own content fails not to abort the
    /// transformation.
    fn message_content(
        &mut self,
        select: Option<&oxixml_xpath::CompiledXPath>,
        content: &[Instruction],
        context: &Context,
    ) -> XsltResult<(String, Sequence)> {
        match select {
            Some(select) => {
                let value = self.evaluate(select, context)?;
                let text = self.simple_content_of(&value, " ")?;
                Ok((text, value.into_sequence()?))
            }
            None => {
                let tree = self.run_to_tree(content, None, context)?;
                let root = tree.root();
                let items = root.children().into_iter().map(Item::Node);
                Ok((root.string_value(), Sequence::from_items(items)))
            }
        }
    }

    /// Settles the serialization parameters of an `xsl:result-document` whose `format` or whose
    /// serialization attributes are attribute value templates with an embedded expression.
    ///
    /// The properties the compiler folded the constant attributes into are the starting point
    /// only when no `format` is deferred; a deferred `format` names an output definition of its
    /// own, so the effective value decides the base and the instruction's own parameters are
    /// applied over it — the order [XSLT 3.0 §23.1](https://www.w3.org/TR/xslt-30/#element-result-document)
    /// prescribes.
    ///
    /// # Errors
    ///
    /// `XTDE1460` when a deferred `format` is not an `EQName` or names no output definition, and
    /// whatever the serialization specification gives a parameter whose effective value is
    /// outside its lexical space.
    ///
    /// The properties are built and returned behind an [`Arc`] rather than by value, and the
    /// method is never inlined: `run_instruction` is the interpreter's recursion point, and a
    /// several-hundred-byte temporary in its frame costs every nested instruction that much
    /// native stack.
    #[inline(never)]
    fn settle_output(
        &mut self,
        compiled: &OutputProperties,
        deferred: &crate::compile::DeferredOutput,
        context: &Context,
    ) -> XsltResult<Arc<OutputProperties>> {
        let mut output = match &deferred.format {
            None => compiled.clone(),
            Some(template) => {
                let format = self.evaluate_avt(template, context)?;
                let names = crate::compile::decl::OutputNames::Snapshot(&deferred.namespaces);
                let name = names.resolve_output_definition(&format)?;
                match self.ir().named_outputs.get(&name) {
                    Some(named) => (**named).clone(),
                    None => {
                        return Err(XsltError::dynamic(
                            "XTDE1460",
                            format!("no `xsl:output` is named `{format}`"),
                        ));
                    }
                }
            }
        };
        let names = crate::compile::decl::OutputNames::Snapshot(&deferred.namespaces);
        for (local, template) in &deferred.parameters {
            let value = self.evaluate_avt(template, context)?;
            crate::compile::decl::apply_output_attribute(
                &mut output,
                local,
                &value,
                &names,
                deferred.version,
            )?;
        }
        output.character_map.extend(
            deferred
                .character_map
                .iter()
                .map(|(key, value)| (*key, value.clone())),
        );
        Ok(Arc::new(output))
    }

    /// Runs an `xsl:try`, handing any dynamic error to the first `xsl:catch` that admits it.
    ///
    /// The try body is written to a private emitter so that output produced before the failure
    /// is discarded — the *rollback* [XSLT 3.0 §8.3](https://www.w3.org/TR/xslt-30/#try-catch)
    /// requires.
    /// Runs the body of an `xsl:result-document` that names the *principal* result destination.
    ///
    /// A destination whose serialization method builds no final result tree — `json`, `adaptive`
    /// ([XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization)) — produces a raw result
    /// instead, which is serialized here, where the errors that raises can still be reported, and
    /// carried on the (empty) tree that stands for the destination. Every other method builds its
    /// tree as before.
    ///
    /// # Errors
    ///
    /// Any dynamic error the body raises, plus the serialization errors the method raises over
    /// the raw result.
    fn principal_destination(
        &mut self,
        content: &[crate::compile::Instruction],
        context: &Context,
        output: &OutputProperties,
        item_separator: Option<&str>,
    ) -> XsltResult<crate::tree::ResultTree> {
        if crate::serialize::builds_a_tree(output) {
            return self.run_to_final_tree(content, context, item_separator);
        }
        let raw = self.run_to_raw_result(content, context, item_separator)?;
        let text = crate::serialize::serialize_raw(&raw, output)?;
        Ok(self
            .run_to_final_tree(&[], context, None)?
            .with_raw_serialization(text))
    }

    fn run_try(
        &mut self,
        select: Option<&oxixml_xpath::CompiledXPath>,
        content: &[Instruction],
        catches: &[crate::compile::CatchClause],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let attempt = match select {
            Some(select) => self
                .evaluate(select, context)
                .and_then(|value| Ok(value.into_sequence()?))
                .map(|sequence| (Flow::Normal, sequence.iter().cloned().collect::<Vec<_>>())),
            // The body is collected so that it can be discarded when it fails, not because it
            // is building a value: `xsl:try` does not enter temporary output state, which is
            // what lets `doe-0190` and `doe-0191` keep their disabled escaping.
            None => self
                .run_to_sequence_in_place(content, context)
                .map(|(flow, _, sequence)| (flow, sequence.iter().cloned().collect::<Vec<_>>())),
        };
        let error = match attempt {
            // XSLT 3.0 §8.8: `xsl:try` is not a barrier to `xsl:break`. The body succeeded, so
            // what it produced is written and the outcome it ended on — an `xsl:break` that
            // must still end the enclosing `xsl:iterate` (the suite's iterate-035) — is handed
            // on rather than settled here.
            Ok((flow, items)) => {
                for item in &items {
                    out.item(item)?;
                }
                return Ok(flow);
            }
            Err(error) => error,
        };
        // A budget or scope failure is the harness's business, not the stylesheet's.
        if error.is("OXIXML0001") {
            return Err(error);
        }
        let namespace = error.namespace().map(str::to_owned);
        let local = error.code().unwrap_or_default().to_owned();
        for clause in catches {
            if !clause
                .errors
                .iter()
                .any(|test| test.admits(namespace.as_deref(), &local))
            {
                continue;
            }
            let scoped = self.with_error_variables(context, &error);
            return self.emit_selected(clause.select.as_ref(), &clause.content, &scoped, out);
        }
        Err(error)
    }

    /// Binds the `err:` variables an `xsl:catch` body sees.
    fn with_error_variables(&self, context: &Context, error: &XsltError) -> Context {
        let mut scoped = context.clone();
        let bind = |bindings: Bindings, local: &str, value: oxixml_infoset::AtomicValue| {
            bindings.bind(
                ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), local),
                oxixml_xpath::XpathValue::from(Sequence::from(value)),
            )
        };
        // The prefix is part of the value: `{$err:code}` stringifies a `QName` through its
        // lexical form, and every processor writes a specification error as `err:FOAR0001`.
        // `err` is the prefix the compiler already binds to the error namespace whether or not
        // the stylesheet declares it, so it is the one the value carries; an error in some
        // other namespace keeps no prefix, having none to take.
        let prefix = error
            .namespace()
            .filter(|namespace| *namespace == crate::error::ERROR_NAMESPACE)
            .map(|_| "err".to_owned());
        let code = oxixml_infoset::AtomicValue::qname(oxixml_infoset::QNameValue::new(
            prefix,
            error.namespace().map(str::to_owned),
            error.code().unwrap_or_default().to_owned(),
        ));
        let mut bindings = scoped.bindings.bind(
            ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), "code"),
            oxixml_xpath::XpathValue::from(Sequence::from(code)),
        );
        bindings = bind(
            bindings,
            "description",
            oxixml_infoset::AtomicValue::string(error.message()),
        );
        // XSLT 3.0 §8.3 makes `$err:value` "a value associated with the error", which only a
        // construct that supplies one has: a terminating `xsl:message` passes its content
        // (message-0501), and the conditions the specification names raise no value at all.
        bindings = bindings.bind(
            ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), "value"),
            oxixml_xpath::XpathValue::from(error.value().cloned().unwrap_or_else(Sequence::empty)),
        );
        bindings = bind(
            bindings,
            "module",
            oxixml_infoset::AtomicValue::string(self.runtime.base_uri.clone().unwrap_or_default()),
        );
        // §8.3 declares `$err:line-number` and `$err:column-number` as `xs:integer?`: "the value
        // will be an empty sequence if the information is not available". It is available when
        // the stylesheet was parsed with source positions recorded and the failure came out of
        // an instruction the compiler located; otherwise the *empty* sequence says so, where a
        // sentinel such as `-1` would be read as a real line number (`try-002` accepts the line
        // only as `17` or as nothing at all).
        let position = error.position();
        for (local, coordinate) in [
            ("line-number", position.map(|at| at.line)),
            ("column-number", position.map(|at| at.column)),
        ] {
            let value = match coordinate {
                Some(number) => {
                    Sequence::from(oxixml_infoset::AtomicValue::integer(i64::from(number)))
                }
                None => Sequence::empty(),
            };
            bindings = bindings.bind(
                ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), local),
                oxixml_xpath::XpathValue::from(value),
            );
        }
        scoped.bindings = bindings;
        scoped
    }

    /// Resolves an `error-code` attribute to the name the error is raised under.
    ///
    /// [XSLT 3.0 §6.7](https://www.w3.org/TR/xslt-30/#element-message) makes the attribute an
    /// `EQName`: `Q{uri}local` names the namespace outright, a prefix is resolved against the
    /// namespaces in scope where the attribute was written, and an unprefixed name is in no
    /// namespace — *provided* the value is a lexical `QName` at all; the caller ([`Instruction::Message`]'s
    /// handler) falls back to the default `XTMM9000` both when this returns `Err` and when the
    /// value plainly is not one, which is handled here rather than there, since only this
    /// method reads the raw lexical text.
    ///
    /// # Errors
    ///
    /// `XTDE0030` when a prefix is not bound.
    fn resolve_error_code(
        &self,
        code: &crate::compile::ErrorCode,
        context: &Context,
    ) -> XsltResult<(Option<String>, String)> {
        let lexical = self.evaluate_avt(&code.template, context)?;
        let trimmed = lexical.trim();
        if let Some(rest) = trimmed.strip_prefix("Q{")
            && let Some((uri, local)) = rest.split_once('}')
        {
            let namespace = (!uri.is_empty()).then(|| uri.to_owned());
            return Ok((namespace, local.to_owned()));
        }
        // XSLT 3.0 §6.7: a value that is not even a lexical `QName` — `23CODE`, say, which
        // cannot be a `QName` because a `QName` never starts with a digit — reverts to the
        // default `XTMM9000` instead of being raised under that ill-formed name verbatim.
        if !oxixml_qname::is_qname(trimmed) {
            return Ok(default_error_code());
        }
        match trimmed.split_once(':') {
            None => Ok((None, trimmed.to_owned())),
            Some((prefix, local)) => {
                if prefix == "xml" {
                    return Ok((
                        Some(oxixml_qname::XML_NAMESPACE.to_owned()),
                        local.to_owned(),
                    ));
                }
                match code
                    .namespaces
                    .iter()
                    .find(|(bound, _)| bound == prefix)
                    .map(|(_, uri)| uri.clone())
                {
                    Some(uri) => Ok((Some(uri), local.to_owned())),
                    None => Err(XsltError::dynamic(
                        "XTDE0030",
                        format!("the prefix `{prefix}` of an error code is not bound"),
                    )),
                }
            }
        }
    }

    /// Emits either a `select` expression's items or a sequence constructor's output.
    pub(super) fn emit_selected(
        &mut self,
        select: Option<&oxixml_xpath::CompiledXPath>,
        content: &[Instruction],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        match select {
            Some(select) => {
                let value = self.evaluate(select, context)?.into_sequence()?;
                for item in value.iter() {
                    out.item(item)?;
                }
                Ok(Flow::Normal)
            }
            None => self.run_plain(content, context, out),
        }
    }

    fn literal_element(
        &mut self,
        literal: &LiteralElement,
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let name = self.aliased_element_name(&literal.name);
        let prefix = (name.namespace() == literal.name.namespace())
            .then(|| literal.prefix.clone())
            .flatten();
        out.start_element_with(name, prefix, literal.inherit_namespaces);
        for (prefix, uri) in &literal.namespaces {
            let uri = self.aliased_namespace(uri);
            out.namespace(prefix.clone(), &uri)?;
        }
        self.apply_attribute_sets(&literal.use_attribute_sets, context, out)?;
        for (name, prefix, value) in &literal.attributes {
            let value = self.evaluate_avt(value, context)?;
            out.attribute(self.aliased_name(name), prefix.clone(), value)?;
        }
        let flow = self.run_sequence(&literal.content, context, out)?;
        out.end_element()?;
        Ok(flow)
    }

    /// Applies `xsl:namespace-alias` to an *attribute* name written in the stylesheet.
    ///
    /// An unprefixed attribute name is in no namespace and stays there: the default namespace
    /// never applies to an attribute, so `xsl:namespace-alias` declaring `#default` as its
    /// stylesheet-prefix leaves `proc="bah"` alone while it rewrites the element around it
    /// (`namespace-alias-1904`).
    fn aliased_name(&self, name: &ExpandedName) -> ExpandedName {
        match name.namespace() {
            Some(namespace) => self.rename(name, namespace),
            None => name.clone(),
        }
    }

    /// Applies `xsl:namespace-alias` to a literal result *element* name.
    ///
    /// [XSLT 3.0 §11.1.2](https://www.w3.org/TR/xslt-30/#literal-namespace-alias) lets the
    /// stylesheet-prefix be `#default`, which names the null namespace — so an element written
    /// with no prefix at all is aliased too. `namespace-alias-1901` and `namespace-alias-1904`
    /// both alias `#default` to the XSLT namespace and expect a bare `<stylesheet>` in the
    /// stylesheet to become `{http://www.w3.org/1999/XSL/Transform}stylesheet` in the result.
    fn aliased_element_name(&self, name: &ExpandedName) -> ExpandedName {
        self.rename(name, name.namespace().unwrap_or_default())
    }

    /// `name` with `namespace` replaced by whatever the alias table maps it to.
    fn rename(&self, name: &ExpandedName, namespace: &str) -> ExpandedName {
        let replacement = self.aliased_namespace(namespace);
        if replacement == namespace {
            return name.clone();
        }
        ExpandedName::new(
            (!replacement.is_empty()).then_some(replacement),
            name.local_name(),
        )
    }

    fn aliased_namespace(&self, namespace: &str) -> String {
        let mut best: Option<(usize, &str)> = None;
        for (from, to, precedence) in &self.ir().namespace_aliases {
            if from == namespace && best.is_none_or(|(current, _)| *precedence >= current) {
                best = Some((*precedence, to.as_str()));
            }
        }
        best.map_or_else(|| namespace.to_owned(), |(_, to)| to.to_owned())
    }

    fn copy(
        &mut self,
        content: &[Instruction],
        use_attribute_sets: &[ExpandedName],
        copy_namespaces: bool,
        inherit_namespaces: bool,
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let Some(Item::Node(node)) = &context.item else {
            if let Some(Item::Atomic(value)) = &context.item {
                out.atomic(value.clone())?;
                return Ok(Flow::Normal);
            }
            if context.item.is_some() {
                return Ok(Flow::Normal);
            }
            // XSLT 3.0 §11.9.1: with no `select`, `xsl:copy` needs a context item.
            return Err(XsltError::dynamic(
                "XTTE0945",
                "xsl:copy has no select attribute and there is no context item",
            ));
        };
        match node.node_kind() {
            NodeKind::Element => {
                let name = node
                    .node_name()
                    .unwrap_or_else(|| ExpandedName::new(None::<&str>, "element"));
                let prefix = node.namespaces().into_iter().find_map(|binding| {
                    (Some(binding.string_value().as_str()) == name.namespace())
                        .then(|| binding.node_name().map(|n| n.local_name().to_owned()))
                        .flatten()
                });
                out.start_element_with(name, prefix, inherit_namespaces);
                // XSLT 3.0 §11.9.1: "when a node is copied, its base URI is copied".
                out.note_copied_base_uri(node.base_uri());
                if copy_namespaces {
                    for binding in node.namespaces() {
                        let prefix = binding.node_name().map(|n| n.local_name().to_owned());
                        if prefix.as_deref() == Some("xml") {
                            continue;
                        }
                        out.namespace(prefix, &binding.string_value())?;
                    }
                }
                self.apply_attribute_sets(use_attribute_sets, context, out)?;
                let flow = self.run_sequence(content, context, out)?;
                out.end_element()?;
                Ok(flow)
            }
            // The copy survives as a document node of its own, so it carries the original's
            // document URI and DTD declarations — `unparsed-entity-06` reads back through
            // `fn:unparsed-entity-uri` what `<xsl:copy/>` produced.
            NodeKind::Document if out.wants_document_item() => {
                let tree = self.run_to_document_copy(content, context, node)?;
                out.append_node(&tree.root())?;
                Ok(Flow::Normal)
            }
            NodeKind::Document => {
                // XSLT 3.0 §11.9.1 makes this build a *document node*, whose content may hold
                // neither an attribute nor a namespace node; writing the content straight into
                // whatever element happens to be open around the instruction would silently
                // attach one to that element instead of reporting `XTDE0420`. Only the content
                // survives here — the document node itself is replaced by its children — so
                // there are no document properties to carry.
                let tree = self.run_to_tree_in_place(content, context)?;
                for child in tree.root().children() {
                    out.copy_node(&child)?;
                }
                Ok(Flow::Normal)
            }
            NodeKind::Attribute | NodeKind::Namespace => {
                out.copy_node(node)?;
                Ok(Flow::Normal)
            }
            _ => {
                out.copy_node(node)?;
                Ok(Flow::Normal)
            }
        }
    }

    fn apply_attribute_sets(
        &mut self,
        names: &[ExpandedName],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<()> {
        // XSLT 3.0 §10.2: an attribute set is a *declaration*, so only global variables and
        // parameters are in scope inside it — the local variables in force where the set is
        // used are not (the suite's attribute-set-1802). The focus, by contrast, is the one the
        // using instruction has.
        let mut context = context.clone();
        context.bindings = Bindings::default();
        let context = &context;
        for name in names {
            let sets = Arc::clone(&self.runtime.ir);
            let Some(definitions) = sets.attribute_sets.get(name) else {
                return Err(XsltError::dynamic(
                    "XTSE0710",
                    format!("no xsl:attribute-set named `{name}` is declared"),
                ));
            };
            let mut ordered: Vec<&crate::compile::AttributeSet> = definitions.iter().collect();
            ordered.sort_by_key(|definition| definition.import_precedence);
            for definition in ordered {
                self.apply_attribute_sets(&definition.uses, context, out)?;
                self.run_sequence(&definition.content, context, out)?;
            }
        }
        Ok(())
    }
}

/// Resolves a `QName` computed at run time against the namespace bindings the instruction saw.
///
/// [XSLT 3.0 §11.5.2](https://www.w3.org/TR/xslt-30/#computed-attributes) and
/// [§11.7.2](https://www.w3.org/TR/xslt-30/#computed-elements) give `xsl:attribute` and
/// `xsl:element` the *same* two failure conditions — the effective `name` is not a lexical
/// `QName`, or its prefix is not bound — but under *different* codes: `XTDE0850`/`XTDE0860` for
/// an attribute, `XTDE0820`/`XTDE0830` for an element, which `attribute` tells apart.
///
/// # Errors
///
/// A malformed name is `XTDE0850` (`attribute`) or `XTDE0820` (element); an unbound prefix is
/// `XTDE0860` (`attribute`) or `XTDE0830` (element).
pub(crate) fn resolve_computed_name(
    lexical: &str,
    namespace: Option<&str>,
    prefixes: &[(String, String)],
    attribute: bool,
) -> XsltResult<(ExpandedName, Option<String>)> {
    let malformed_name_code = if attribute { "XTDE0850" } else { "XTDE0820" };
    let unbound_prefix_code = if attribute { "XTDE0860" } else { "XTDE0830" };
    let lexical = lexical.trim();
    let (prefix, local) = match lexical.split_once(':') {
        Some((prefix, local)) => (Some(prefix.to_owned()), local),
        None => (None, lexical),
    };
    if !is_ncname(local) || prefix.as_deref().is_some_and(|prefix| !is_ncname(prefix)) {
        return Err(XsltError::dynamic(
            malformed_name_code,
            format!("`{lexical}` is not a valid QName"),
        ));
    }
    if local == "xmlns" && prefix.is_none() && attribute {
        return Err(XsltError::dynamic(
            "XTDE0855",
            "an attribute may not be named `xmlns`",
        ));
    }
    let uri = match namespace {
        Some(namespace) => (!namespace.is_empty()).then(|| namespace.to_owned()),
        None => match &prefix {
            Some(prefix) if prefix == "xml" => Some(XML_NAMESPACE.to_owned()),
            Some(prefix) => Some(
                prefixes
                    .iter()
                    .find(|(candidate, _)| candidate == prefix)
                    .map(|(_, uri)| uri.clone())
                    .ok_or_else(|| {
                        XsltError::dynamic(
                            unbound_prefix_code,
                            format!("the prefix `{prefix}` is not bound"),
                        )
                    })?,
            ),
            None if attribute => None,
            None => prefixes
                .iter()
                .find(|(candidate, _)| candidate.is_empty())
                .map(|(_, uri)| uri.clone()),
        },
    };
    if uri.as_deref() == Some(XMLNS_NAMESPACE) {
        // The two instructions part company again here: [§11.5.2](https://www.w3.org/TR/xslt-30/#computed-attributes)
        // makes an `xsl:attribute` whose namespace is the `xmlns` namespace `XTDE0865`
        // (`namespace-2623` builds the URI out of an attribute value template so that no static
        // check can see it), while [§11.7.2](https://www.w3.org/TR/xslt-30/#computed-elements)
        // gives `xsl:element` `XTDE0835` for the same fault.
        return Err(XsltError::dynamic(
            if attribute { "XTDE0865" } else { "XTDE0835" },
            "a name may not be in the xmlns namespace",
        ));
    }
    let prefix = match (&uri, prefix) {
        (Some(_), Some(prefix)) => Some(prefix),
        (Some(uri), None) if uri == XSLT_NAMESPACE => None,
        (Some(_), None) if attribute => None,
        (_, prefix) => prefix,
    };
    Ok((ExpandedName::new(uri, local), prefix))
}

/// The checks [XSLT 3.0 §11.7.3](https://www.w3.org/TR/xslt-30/#creating-namespace-nodes) puts
/// on the namespace node `xsl:namespace` constructs.
///
/// The instruction's `name` is a prefix and its content is a URI, and each has its own error
/// code: a name that is neither empty nor an `NCName`, or that is `xmlns`, is `XTDE0920`; an
/// empty URI is `XTDE0930`; a URI outside the lexical space of `xs:anyURI`, or the reserved
/// `http://www.w3.org/2000/xmlns/`, is `XTDE0905`; and binding the `xml` prefix to anything but
/// the XML namespace — or that namespace to anything but `xml` — is `XTDE0925`.
fn check_namespace_instruction(prefix: &str, uri: &str) -> XsltResult<()> {
    if prefix == "xmlns" || (!prefix.is_empty() && !oxixml_qname::is_ncname(prefix)) {
        return Err(XsltError::dynamic(
            "XTDE0920",
            format!("`{prefix}` is not a namespace prefix xsl:namespace may create"),
        ));
    }
    if uri.is_empty() {
        return Err(XsltError::dynamic(
            "XTDE0930",
            "xsl:namespace was given a zero-length namespace URI",
        ));
    }
    if (prefix == "xml") != (uri == oxixml_qname::XML_NAMESPACE) {
        return Err(XsltError::dynamic(
            "XTDE0925",
            "the `xml` prefix and the XML namespace may only be bound to each other",
        ));
    }
    // `xs:anyURI` has `whiteSpace="collapse"`, so a value that is only whitespace has nothing
    // left to validate — `on-empty-115b` builds exactly that and expects no error.
    //
    // The lexical space itself is the data type's, not the IRI grammar's: XSD 1.1 §3.3.17 maps a
    // lexical form to an IRI by escaping the characters an IRI may not carry ("disallowed
    // characters … are escaped"), so a URI holding a space or a quotation mark is a perfectly
    // good `xs:anyURI` (`snapshot-0103` binds a namespace to `http://www.example.com/"` and
    // expects it to be accepted). Asking the atomic type settles it the same way every cast to
    // `xs:anyURI` in this workspace does, rather than a second, stricter rule written here.
    let candidate = uri.trim();
    if uri == "http://www.w3.org/2000/xmlns/"
        || (!candidate.is_empty()
            && oxixml_infoset::AtomicValue::parse(oxixml_infoset::AtomicType::AnyUri, candidate)
                .is_err())
    {
        return Err(XsltError::dynamic(
            "XTDE0905",
            format!("`{uri}` is not in the lexical space of xs:anyURI"),
        ));
    }
    Ok(())
}

/// Inserts a space after every hyphen that is followed by another hyphen or that ends the
/// string, which [XSLT 3.0 §11.8](https://www.w3.org/TR/xslt-30/#creating-comments) requires of
/// the content of a comment.
fn space_out_hyphens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        out.push(character);
        if character == '-' && matches!(characters.peek(), Some('-') | None) {
            out.push(' ');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computed_names_resolve_against_the_snapshot() -> XsltResult<()> {
        let prefixes = vec![
            ("p".to_owned(), "urn:p".to_owned()),
            (String::new(), "urn:default".to_owned()),
        ];
        let (name, prefix) = resolve_computed_name("p:a", None, &prefixes, false)?;
        assert_eq!(name, ExpandedName::new(Some("urn:p"), "a"));
        assert_eq!(prefix.as_deref(), Some("p"));

        let (name, _) = resolve_computed_name("a", None, &prefixes, false)?;
        assert_eq!(name, ExpandedName::new(Some("urn:default"), "a"));

        let (name, _) = resolve_computed_name("a", None, &prefixes, true)?;
        assert_eq!(name, ExpandedName::new(None::<&str>, "a"));

        assert_eq!(
            resolve_computed_name("q:a", None, &prefixes, false)
                .expect_err("unbound")
                .code(),
            Some("XTDE0830")
        );
        assert_eq!(
            resolve_computed_name("not a name", None, &prefixes, false)
                .expect_err("malformed")
                .code(),
            Some("XTDE0820")
        );
        Ok(())
    }

    /// `xsl:attribute`'s computed name shares `xsl:element`'s two failure conditions but not
    /// their codes — the suite's error-0850a and error-0860a turn on exactly this distinction.
    #[test]
    fn attribute_computed_names_use_their_own_codes() {
        let prefixes = vec![("p".to_owned(), "urn:p".to_owned())];
        assert_eq!(
            resolve_computed_name("a#", None, &prefixes, true)
                .expect_err("`a#` is not a QName")
                .code(),
            Some("XTDE0850")
        );
        assert_eq!(
            resolve_computed_name("your:ns", None, &prefixes, true)
                .expect_err("`your` is not bound")
                .code(),
            Some("XTDE0860")
        );
        // The element codes are unaffected.
        assert_eq!(
            resolve_computed_name("a#", None, &prefixes, false)
                .expect_err("`a#` is not a QName")
                .code(),
            Some("XTDE0820")
        );
        assert_eq!(
            resolve_computed_name("your:ns", None, &prefixes, false)
                .expect_err("`your` is not bound")
                .code(),
            Some("XTDE0830")
        );
    }
}
