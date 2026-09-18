//! The instruction interpreter.
//!
//! This module holds the [`Transformer`] methods that run a sequence constructor and
//! evaluate the values a variable, parameter or stylesheet function binds. The instruction
//! dispatch itself lives in [`instruction`], template-rule application in [`templates`], and
//! the iteration constructs in [`control`].

use super::{Bindings, Context, Emitter, Transformer};
use crate::compile::{Instruction, VariableDecl};
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{ExpandedName, Item, Sequence};
use oxixml_xpath::XpathValue;
use std::sync::Arc;

mod control;
mod instruction;
mod templates;

/// The most results one `cache="yes"` `xsl:function` memoizes within a transformation.
///
/// [XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions-caching) makes `cache`
/// a hint, so a ceiling keeps a function called with a great many distinct argument lists from
/// turning the cache into an unbounded memory sink: beyond it, further calls recompute rather
/// than store, which is always a correct answer. The figure is generous enough that ordinary
/// memoization — a recursion over a bounded domain — never reaches it.
const MAX_FUNCTION_CACHE_ENTRIES: usize = 100_000;

/// The memoization key for a `cache="yes"` call, or [`None`] when the call cannot be cached.
///
/// A call is cacheable only when every argument is a sequence of atomic values: those have a
/// stable [`Hash`]/[`Eq`] identity (`AtomicValue` keys the same `1` to the same slot however it
/// is spelled, and distinguishes it from `1.0` by type), whereas a node, map, array or function
/// argument carries identity or context the cache could not honour. Returning [`None`] leaves
/// the call to ordinary evaluation, which is always correct.
fn atomic_arguments(arguments: &[Sequence]) -> Option<super::FunctionCacheKey> {
    let mut key = Vec::with_capacity(arguments.len());
    for argument in arguments {
        let mut atoms = Vec::with_capacity(argument.len());
        for item in argument.iter() {
            match item {
                Item::Atomic(value) => atoms.push(value.clone()),
                _ => return None,
            }
        }
        key.push(atoms);
    }
    Some(key)
}

/// How a sequence constructor finished.
#[derive(Debug)]
pub(crate) enum Flow {
    /// It ran to the end.
    Normal,
    /// `xsl:break` was reached, with the value it produced.
    Break(Sequence),
    /// `xsl:next-iteration` was reached, with the parameters it supplied.
    NextIteration(Vec<(ExpandedName, XpathValue)>),
}

impl Flow {
    fn is_normal(&self) -> bool {
        matches!(self, Self::Normal)
    }
}

impl Transformer {
    /// Runs a sequence constructor.
    ///
    /// # Errors
    ///
    /// Any dynamic error the instructions raise.
    pub(crate) fn run_sequence(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        // `xsl:on-empty` and `xsl:on-non-empty` ask about the *other* instructions of the same
        // sequence constructor, so their emptiness is settled before anything is written.
        if instructions
            .iter()
            .any(|instruction| matches!(instruction, Instruction::OnEmpty { .. }))
        {
            let empty = self.probe_emptiness(instructions, context)?;
            return self.run_gated(instructions, context, out, empty);
        }
        self.run_plain(instructions, context, out)
    }

    /// Runs the instructions that are not `xsl:on-empty`/`xsl:on-non-empty` into a throwaway
    /// emitter, to learn whether they produce anything.
    fn probe_emptiness(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
    ) -> XsltResult<bool> {
        let ordinary: Vec<&Instruction> = instructions
            .iter()
            .filter(|instruction| !matches!(instruction, Instruction::OnEmpty { .. }))
            .collect();
        // An attribute or namespace node counts as content for `xsl:on-empty`, but writing one
        // at the top of the free-standing probe tree is not allowed; seeing that refusal is
        // itself the answer.
        let mut attributes_written = false;
        let mut probe = Emitter::tree();
        let quiet = self.runtime.messages.borrow().len();
        let mut scoped = context.clone();
        for instruction in ordinary {
            if let Instruction::Variable(declaration) = instruction {
                let value = self.variable_value(declaration, &scoped)?;
                scoped.bindings = scoped.bindings.bind(declaration.name.clone(), value);
                continue;
            }
            match self.run_instruction(instruction, &scoped, &mut probe) {
                Ok(_) => {}
                Err(error) if error.is("XTDE0410") || error.is("XTDE0420") => {
                    attributes_written = true;
                }
                Err(error) => return Err(error),
            }
        }
        // The probe is not part of the transformation, so anything it said is forgotten.
        self.runtime.messages.borrow_mut().truncate(quiet);
        // Nothing reaching the emitter at all settles the question without building a tree.
        if probe.is_empty() {
            return Ok(!attributes_written);
        }
        // Otherwise what settles it is the *content* the instructions contributed, which is
        // what reached the probe tree. Asking the tree rather than the emitter matters for a
        // document node with no children — `xsl:sequence`/`xsl:copy-of` over an empty temporary
        // tree — which is an item the emitter records but which writes nothing, and so leaves
        // the containing sequence constructor empty (the suite's on-empty-012, on-non-empty-012
        // and on-non-empty-014).
        let (tree, _) = probe.finish()?;
        Ok(!attributes_written && tree.root_children().is_empty())
    }

    fn run_gated(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        out: &mut Emitter,
        empty: bool,
    ) -> XsltResult<Flow> {
        let mut context = context.clone();
        for instruction in instructions {
            if let Instruction::OnEmpty {
                expect_empty,
                select,
                content,
            } = instruction
            {
                if *expect_empty == empty {
                    let flow = self.emit_selected(select.as_ref(), content, &context, out)?;
                    if !flow.is_normal() {
                        return Ok(flow);
                    }
                }
                continue;
            }
            if let Instruction::Variable(declaration) = instruction {
                let value = self.variable_value(declaration, &context)?;
                context.bindings = context.bindings.bind(declaration.name.clone(), value);
                continue;
            }
            let flow = self.run_instruction(instruction, &context, out)?;
            if !flow.is_normal() {
                return Ok(flow);
            }
        }
        Ok(Flow::Normal)
    }

    fn run_plain(
        &mut self,
        instructions: &[Instruction],
        context: &Context,
        out: &mut Emitter,
    ) -> XsltResult<Flow> {
        let mut context = context.clone();
        for instruction in instructions {
            if let Instruction::Variable(declaration) = instruction {
                let value = self.variable_value(declaration, &context)?;
                context.bindings = context.bindings.bind(declaration.name.clone(), value);
                continue;
            }
            let flow = self.run_instruction(instruction, &context, out)?;
            if !flow.is_normal() {
                return Ok(flow);
            }
        }
        Ok(Flow::Normal)
    }

    /// The value a variable declaration binds.
    ///
    /// # Errors
    ///
    /// Any dynamic error the declaration raises.
    pub(crate) fn variable_value(
        &mut self,
        declaration: &VariableDecl,
        context: &Context,
    ) -> XsltResult<XpathValue> {
        let value = self.bound_value(
            declaration.select.as_ref(),
            &declaration.content,
            declaration.as_type.is_some(),
            declaration.base_uri.as_deref(),
            context,
        )?;
        match &declaration.as_type {
            // XSLT 3.0 §9.2: the value of an `xsl:variable` with `as` is converted to the
            // required type, and a failure is `XTTE0570`.
            Some(required) => required.convert(
                value,
                "XTTE0570",
                &format!("the value of the variable `{}`", declaration.name),
            ),
            None => Ok(value),
        }
    }

    /// The value a `select` attribute or a sequence constructor produces.
    ///
    /// Without an `as` attribute the constructor builds a *tree*, which is what XSLT 1.0 calls
    /// a result tree fragment; with one it builds a sequence.
    ///
    /// `base` is the static base URI of the binding element, which the tree the constructor
    /// builds takes as its own — [XSLT 3.0 §9.2](https://www.w3.org/TR/xslt-30/#temporary-trees).
    ///
    /// # Errors
    ///
    /// Any dynamic error the expression or the constructor raises.
    pub(crate) fn bound_value(
        &mut self,
        select: Option<&oxixml_xpath::CompiledXPath>,
        content: &[Instruction],
        typed: bool,
        base: Option<&str>,
        context: &Context,
    ) -> XsltResult<XpathValue> {
        match select {
            Some(select) => self.evaluate(select, context),
            None if content.is_empty() && typed => Ok(XpathValue::from(Sequence::empty())),
            None if typed => {
                let (_, sequence) = self.run_to_sequence(content, base, context)?;
                Ok(XpathValue::from(sequence))
            }
            None => {
                let tree = self.run_to_tree(content, base, context)?;
                Ok(XpathValue::from(Sequence::from(tree.root())))
            }
        }
    }

    /// Calls an `xsl:function`.
    ///
    /// A function declared `cache="yes"` is memoized, but only for a call the engine can prove
    /// observes nothing but its arguments: every argument must be a sequence of atomic values,
    /// which rules out node identity and document order, and the table lives only for the current
    /// transformation, over which every other context-dependent value a body may read
    /// (`current-dateTime()`, a global variable, a loaded document) is stable. A wrong cache hit
    /// would be a correctness bug, so any call that does not meet the test is simply evaluated,
    /// which the specification permits because `cache` is only a hint.
    ///
    /// # Errors
    ///
    /// `XPST0017` when no such function is declared — XSLT 3.0 §10.3 makes a call on an
    /// undeclared stylesheet function that static error, and the compiler normally rejects it
    /// before a transformation starts, so this branch is defensive — plus any dynamic error the
    /// body raises.
    pub(crate) fn call_user_function(
        &mut self,
        name: &ExpandedName,
        arity: usize,
        arguments: &[Sequence],
    ) -> XsltResult<Sequence> {
        let ir = Arc::clone(&self.runtime.ir);
        let Some(function) = ir.functions.get(&(name.clone(), arity)) else {
            return Err(XsltError::dynamic(
                "XPST0017",
                format!("no xsl:function named `{name}` of arity {arity} is declared"),
            ));
        };
        // A cacheable call is keyed on its arguments, atomic values throughout. A hit returns the
        // memoized result without re-entering the body — so a recursive `cache="yes"` function
        // such as Fibonacci runs in linear time rather than exponential.
        let cache_key = if function.cache {
            atomic_arguments(arguments)
        } else {
            None
        };
        if let Some(key) = &cache_key {
            if let Some(cached) = self
                .runtime
                .function_cache
                .borrow()
                .get(&(name.clone(), arity))
                .and_then(|table| table.get(key))
            {
                return Ok(cached.clone());
            }
        }
        let mut bindings = Bindings::default();
        for (parameter, value) in function.params.iter().zip(arguments.iter()) {
            let mut value = XpathValue::from(value.clone());
            // XSLT 3.0 §10.3 makes a call on a stylesheet function an ordinary XPath function
            // call, whose arguments are converted by the function conversion rules; a failure
            // is therefore the XPath type error `XPTY0004`. (XSLT 2.0 gave this its own code,
            // `XTTE0790`, which XSLT 3.0 dropped — the error catalogue no longer lists it, and
            // the suite forked error-0790a into an `XSLT20` case expecting `XTTE0790` and an
            // `XSLT30+` case, error-0790a3, expecting `XPTY0004`.)
            if let Some(required) = &parameter.as_type {
                value = required.convert(
                    value,
                    "XPTY0004",
                    &format!("the argument for the parameter `{}`", parameter.name),
                )?;
            }
            bindings = bindings.bind(parameter.name.clone(), value);
        }
        let mut context = Context::new(None);
        context.bindings = bindings;
        self.enter()?;
        let result = self.without_regex_groups(|engine| {
            engine.run_to_sequence(&function.content, None, &context)
        });
        self.leave();
        let sequence = result?.1;
        let output = match &ir
            .functions
            .get(&(name.clone(), arity))
            .and_then(|f| f.as_type.clone())
        {
            // XSLT 3.0 §10.3: the body's result is converted to the `as` type, and a failure
            // is `XTTE0780`.
            Some(required) => required
                .convert(
                    XpathValue::from(sequence),
                    "XTTE0780",
                    &format!("the result of the function `{name}`"),
                )?
                .into_sequence()?,
            None => sequence,
        };
        // Only a successful result is memoized — an error is never cached, so a call that failed
        // is retried and fails again identically. A result that *builds nodes* is memoized too,
        // and that is the whole point of the declaration rather than an oversight:
        // [XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions-caching) makes
        // `new-each-time="no"` the promise that two calls with the same arguments return the
        // same nodes, so serving the second from the table is what the stylesheet asked for.
        // `new-each-time="maybe"` — the default, and the only state in which node identity is
        // left to the processor — never sets `cache`, so an undeclared function still yields a
        // fresh node per call. The suite's function-1025 counts the distinct nodes ten calls
        // over seven distinct arguments produce and requires seven.
        if let Some(key) = cache_key {
            let mut cache = self.runtime.function_cache.borrow_mut();
            let table = cache.entry((name.clone(), arity)).or_default();
            // The table is a hint, not a promise: past a generous per-function ceiling it
            // stops growing rather than becoming an unbounded memory sink, and the calls
            // beyond it simply recompute — always a correct answer.
            if table.len() < MAX_FUNCTION_CACHE_ENTRIES {
                table.insert(key, output.clone());
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Parameters, Stylesheet};
    use oxixml_infoset::Tree;

    /// A stack large enough to hold a recursion many levels deep: each `xsl:function` call is a
    /// tall stack of interpreter frames, so a Fibonacci that recurses seventy deep needs room the
    /// default 2 MiB test thread does not have. The conformance harness spawns the same way.
    const DEEP_STACK: usize = 64 * 1024 * 1024;

    /// Compiles a stylesheet whose body is `functions_and_templates` and transforms `<in/>` on a
    /// thread with a deep stack, returning the string result.
    fn transform(functions_and_templates: &str) -> XsltResult<String> {
        let source = functions_and_templates.to_owned();
        std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(move || {
                let xsl = format!(
                    r#"<xsl:stylesheet version="3.0"
                         xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                         xmlns:f="urn:f">
                         <xsl:output omit-xml-declaration="yes"/>
                         {source}
                       </xsl:stylesheet>"#
                );
                let compiled = Stylesheet::compile_str(&xsl)?;
                let tree = Tree::parse_str("<in/>")
                    .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
                let parameters = Parameters::new().with_stack_size(DEEP_STACK);
                Ok(compiled
                    .transform_with(&tree.root(), &parameters)?
                    .into_string())
            })
            .expect("the deep-stack thread starts")
            .join()
            .expect("the deep-stack thread finishes")
    }

    /// A recursive Fibonacci `xsl:function`, its `cache` attribute set to `cache`, returning
    /// `f:fib($n)`.
    fn fibonacci(cache: &str, n: u32) -> XsltResult<String> {
        transform(&format!(
            r#"<xsl:function name="f:fib" as="xs:integer" cache="{cache}">
                 <xsl:param name="n" as="xs:integer"/>
                 <xsl:sequence select="if ($n lt 2) then $n else f:fib($n - 1) + f:fib($n - 2)"/>
               </xsl:function>
               <xsl:template match="/"><xsl:value-of select="f:fib({n})"/></xsl:template>"#
        ))
    }

    #[test]
    fn a_cached_function_agrees_with_the_uncached_one() -> XsltResult<()> {
        // Small enough that the exponential un-memoized form still terminates promptly.
        for n in [0, 1, 2, 7, 20] {
            assert_eq!(fibonacci("no", n)?, fibonacci("yes", n)?, "fib({n})");
        }
        Ok(())
    }

    #[test]
    fn caching_makes_an_exponential_recursion_tractable() -> XsltResult<()> {
        // `f:fib(70)` issues on the order of 10^14 calls without memoization — the test could not
        // finish. With `cache="yes"` each of the 71 argument values is computed once, so it
        // returns at once, and the answer proves the cache never served a wrong result.
        assert_eq!(fibonacci("yes", 70)?, "190392490709135");
        Ok(())
    }

    #[test]
    fn an_undeclared_function_yields_a_fresh_node_each_call() -> XsltResult<()> {
        // `new-each-time` defaults to `maybe`, which leaves node identity to the processor; this
        // one builds a fresh node per call, so `$a | $b` — a union, which counts two distinct
        // element nodes as two and one shared node as one — is 2.
        let out = transform(
            r#"<xsl:function name="f:make">
                 <xsl:element name="e"/>
               </xsl:function>
               <xsl:template match="/">
                 <xsl:variable name="a" select="f:make()"/>
                 <xsl:variable name="b" select="f:make()"/>
                 <xsl:value-of select="count($a | $b)"/>
               </xsl:template>"#,
        )?;
        assert_eq!(out, "2");
        Ok(())
    }

    /// `function-1025`: `new-each-time="no"` promises that two calls with the same arguments
    /// return the *same* nodes, so ten calls over seven distinct arguments build seven nodes.
    #[test]
    fn new_each_time_no_shares_the_nodes_between_equal_calls() -> XsltResult<()> {
        let out = transform(
            r#"<xsl:function name="f:test" as="element()" new-each-time="no">
                 <xsl:param name="n" as="xs:integer"/>
                 <e><xsl:value-of select="$n"/></e>
               </xsl:function>
               <xsl:template match="/">
                 <xsl:variable name="nodes" as="element()*">
                   <xsl:for-each select="(1,4,6,8,3,5,6,2,1,3)">
                     <xsl:sequence select="f:test(.)"/>
                   </xsl:for-each>
                 </xsl:variable>
                 <xsl:value-of select="count($nodes | ())"/>
               </xsl:template>"#,
        )?;
        assert_eq!(out, "7");
        Ok(())
    }

    /// The negative control: `new-each-time="yes"` states the opposite promise, so the same ten
    /// calls build ten distinct nodes.
    #[test]
    fn new_each_time_yes_keeps_every_node_distinct() -> XsltResult<()> {
        let out = transform(
            r#"<xsl:function name="f:test" as="element()" new-each-time="yes">
                 <xsl:param name="n" as="xs:integer"/>
                 <e><xsl:value-of select="$n"/></e>
               </xsl:function>
               <xsl:template match="/">
                 <xsl:variable name="nodes" as="element()*">
                   <xsl:for-each select="(1,4,6,8,3,5,6,2,1,3)">
                     <xsl:sequence select="f:test(.)"/>
                   </xsl:for-each>
                 </xsl:variable>
                 <xsl:value-of select="count($nodes | ())"/>
               </xsl:template>"#,
        )?;
        assert_eq!(out, "10");
        Ok(())
    }

    #[test]
    fn caching_keys_on_the_whole_argument_so_distinct_calls_stay_distinct() -> XsltResult<()> {
        // A collapsed key would make every call return the first result; each square must be its
        // own.
        let out = transform(
            r#"<xsl:function name="f:sq" as="xs:integer" cache="yes">
                 <xsl:param name="n" as="xs:integer"/>
                 <xsl:sequence select="$n * $n"/>
               </xsl:function>
               <xsl:template match="/">
                 <xsl:value-of select="f:sq(3), f:sq(4), f:sq(3), f:sq(5)" separator=" "/>
               </xsl:template>"#,
        )?;
        assert_eq!(out, "9 16 9 25");
        Ok(())
    }
}
