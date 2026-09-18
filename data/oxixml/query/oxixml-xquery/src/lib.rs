//! `oxixml-xquery` — XQuery 3.1, in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the query language Saxon
//! occupies elsewhere: the whole of [XQuery 3.1](https://www.w3.org/TR/xquery-31/) — FLWOR
//! expressions, node constructors, the query prolog, library modules, `typeswitch`, `switch`,
//! `try`/`catch` — evaluated over the XDM of [`oxixml_infoset`], with the function library of
//! [`oxixml_fn`] behind it.
//!
//! # One engine, not two
//!
//! XQuery is a superset of XPath, and this crate treats it as one. The parser records, for every
//! expression, the byte range it covers and whether that range contains any XQuery-only syntax.
//! A range that is **pure XPath** is compiled by [`oxixml_xpath`] and evaluated by it: every path
//! expression, comparison, cast, type test and function call therefore behaves exactly as that
//! engine's own conformance suite says it does, with no second implementation to drift. Only the
//! constructs XPath does not have get an evaluator here, and an XQuery-only operand inside an
//! otherwise-XPath expression is lifted into a variable so that the rest can still be delegated:
//!
//! ```text
//! count((<a/>, <b/>))   →   count($_h0)   with   $_h0 = the two constructors
//! ```
//!
//! # Example
//!
//! ```
//! use oxixml_infoset::{AtomicValue, Tree};
//! use oxixml_xquery::{DynamicQueryContext, StaticQueryContext, XQuery};
//!
//! let library = Tree::parse_str(
//!     "<library>\
//!        <book year='1965'><title>Dune</title><price>9.99</price></book>\
//!        <book year='1815'><title>Emma</title><price>5.50</price></book>\
//!        <book year='1851'><title>Moby-Dick</title><price>7.25</price></book>\
//!      </library>",
//! )?;
//!
//! // A FLWOR expression with a `where`, an `order by` and a node constructor.
//! let compiled = XQuery::compile(
//!     "<catalogue count='{count(//book)}'>{
//!        for $b in //book
//!        let $year := xs:integer($b/@year)
//!        where $year < $after
//!        order by $year descending
//!        return <entry year='{$year}'>{ string($b/title) }</entry>
//!      }</catalogue>",
//!     StaticQueryContext::new().with_variable("after"),
//! )?;
//!
//! let context = DynamicQueryContext::new()
//!     .with_context_item(library.root())
//!     .with_variable("after", AtomicValue::integer(1900_i64));
//! let result = compiled.execute(&context)?;
//! assert_eq!(result.len(), 1);
//!
//! // Grouping, with the non-grouping variables collected per group.
//! let grouped = XQuery::compile_default(
//!     "string-join(
//!        for $n in 1 to 6
//!        group by $parity := $n mod 2
//!        order by $parity
//!        return concat($parity, ':', count($n)),
//!      ' ')",
//! )?;
//! assert_eq!(
//!     grouped.execute_to_string(&DynamicQueryContext::new())?,
//!     "0:3 1:3"
//! );
//!
//! // A prolog: declared functions, declared variables and `try`/`catch`.
//! let prolog = XQuery::compile_default(
//!     r#"declare variable $limit := 5;
//!        declare function local:fact($n as xs:integer) as xs:integer {
//!          if ($n le 1) then 1 else $n * local:fact($n - 1)
//!        };
//!        try { local:fact($limit) } catch * { -1 }"#,
//! )?;
//! assert_eq!(prolog.execute_to_string(&DynamicQueryContext::new())?, "120");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What is here
//!
//! | Concern | Type |
//! |---|---|
//! | Compiling | [`XQuery::compile`], [`CompiledQuery`] |
//! | Compile-time environment | [`StaticQueryContext`] |
//! | Run-time environment | [`DynamicQueryContext`] |
//! | Prolog-settable components | [`BoundarySpace`], [`ConstructionMode`], [`OrderingMode`], [`EmptyOrder`], [`CopyNamespaces`] |
//! | Library modules | [`ModuleResolver`], [`InMemoryModules`] |
//! | Errors | [`XQueryError`], [`XQueryErrorCode`] |
//!
//! Values, items, types and function items are [`oxixml_xpath`]'s: a query returns an
//! [`oxixml_infoset::Sequence`], or an [`oxixml_xpath::XpathValue`] when the result may hold a
//! function item.
//!
//! # Modules
//!
//! A library module is reached through a [`ModuleResolver`], which the application installs.
//! Nothing is importable until it does — a Pure-Rust library does not read the filesystem or the
//! network behind its caller's back — and an `import module` with no resolver is the static error
//! `XQST0059`. [`InMemoryModules`] is the resolver most callers want.
//!
//! # Conformance
//!
//! The crate is tested against the W3C *QT3* suite (FOTS), compiling every case as XQuery 3.1.
//! On the last measured run the engine passed **99.7%** of the 3 092 attempted cases in the test
//! sets that cover the XQuery-only grammar — the constructors, the FLWOR clauses and the prolog —
//! and **99.4%** of the 29 407 cases it attempts across the whole suite. Set
//! `OXIXML_XML_SUITES_DIR` to a directory holding the suite to run it; the conformance tests
//! skip silently when it is unset.
//!
//! The residue is dominated by things this crate cannot fix from within its own directory:
//!
//! * `xs:decimal` and `xs:integer` are fixed-width here, so the suite's twenty-odd-digit
//!   literals overflow where the specification's unbounded `xs:integer` would not;
//! * `fn:name` and the serializer derive a name's prefix from the in-scope namespaces rather
//!   than reading the one the node's name retains, so a *parentless* attribute constructed with
//!   a prefixed `xs:QName` serializes without it;
//! * a compiled XPath expression is evaluated against the *caller's* static context, so a
//!   namespace declaration attribute does not reach an `xs:QName` cast inside the constructor's
//!   content;
//! * `fn:format-dateTime`'s timezone names and a handful of JSON edge cases, which belong to
//!   `oxixml-fn`;
//! * the schema-aware cases.
//!
//! # What this engine does not do
//!
//! * **The XQuery Update Facility.** `insert`, `delete`, `replace`, `rename`, `copy … modify`
//!   and `transform` are not implemented, and a query that uses them is a syntax error. Nothing
//!   in the data model here is mutable, so updating primitives would need a pending update list
//!   and a snapshot discipline that this crate deliberately does not have.
//! * **Schema awareness.** `validate` is the static error `XQST0075`, `import schema` is recorded
//!   but not honoured, and a constructed element carries `xs:anyType` (or `xs:untyped` under
//!   `declare construction strip`) rather than a validated type.
//! * **Full-text.** The XQuery and XPath Full Text extension is out of scope.
//! * **`declare option output:…`.** Serialization parameters declared in the prolog are recorded
//!   but do not reach [`oxixml_fn::xmlio`], which the caller drives itself.
//!
//! # Static analysis
//!
//! The compiler hands name resolution inside an expression to [`oxixml_xpath`], and adds three
//! static passes of its own over what the parser noticed on the way through: a reference to a
//! variable no declaration and no binder in the module introduces is `XPST0008`, a call to a
//! `%private` function of an imported library module is `XPST0017`, and an annotation assertion
//! in a namespace the specification reserves is `XQST0045`. The variable check is deliberately
//! coarse about *where* a name is bound — a name bound anywhere in its module is accepted
//! everywhere in it — because anything finer would need a scope-resolution pass this compiler
//! does not run; being coarse can only miss a static error, never invent one.
//!
//! # Recursion and the stack
//!
//! A recursive declared function consumes native stack, and running a thread out of it aborts
//! the process — an outcome neither the caller nor a `try`/`catch` can intercept. Two ceilings
//! stop that from happening, and both report `XPDY0130`.
//!
//! The first is a nesting depth counted in *calls* — the unit the query is written in — which is
//! the limit [XQuery 3.1 §2.3.4](https://www.w3.org/TR/xquery-31/#id-dynamic-error-list) lets a
//! processor impose. It is [`DEFAULT_MAX_CALL_DEPTH`] on a thread whose stack the caller did not
//! declare, and rises with a larger declared one, because a caller that runs a query on a
//! 256 MiB thread has said it expects deep recursion.
//!
//! The second is the native stack itself, which is usually the one that bites: a call costs
//! several kilobytes of stack in an optimized build and tens of kilobytes in a build without
//! optimization, so a thread with Rust's default 2 MiB stack carries far fewer than
//! [`DEFAULT_MAX_CALL_DEPTH`] of them. The evaluator therefore measures the stack it has
//! consumed and stops while there is still room, against the figure
//! [`DynamicQueryContext::with_stack_size`] supplies — [`DEFAULT_STACK_SIZE`] when the caller
//! says nothing. A query meant to recurse deeply is executed on a thread spawned with a larger
//! stack, and that stack is declared here.
//!
//! Both budgets are [`oxixml_xpath`]'s, so a call routed through the XPath engine and one the
//! query evaluator makes itself cost one level between them rather than one each. `XPDY0130` may
//! also arrive from that crate for a reason of its own — a function item recursing inside an
//! expression — so the code alone does not say which limit was met; the message does.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod api;
mod ast;
mod compile;
mod context;
mod error;
mod eval;
mod lexer;
mod module;
mod nodes;
mod parser;
mod prolog;
mod serialization;

pub use api::{CompiledQuery, XQuery};
pub use context::{
    BoundarySpace, ConstructionMode, CopyNamespaces, DynamicQueryContext, ERROR_NAMESPACE,
    EmptyOrder, LOCAL_NAMESPACE, OUTPUT_NAMESPACE, OrderingMode, StaticQueryContext,
};
pub use error::{XQueryError, XQueryErrorCode, XQueryResult};
pub use eval::{DEFAULT_MAX_CALL_DEPTH, DEFAULT_STACK_SIZE};
pub use module::{InMemoryModules, ModuleResolver};

/// Re-export of [`oxixml_fn::FN_NAMESPACE`], the namespace of the `fn:` functions.
pub use oxixml_fn::FN_NAMESPACE;
/// Re-export of [`oxixml_infoset::XSD_NAMESPACE`], the namespace of the `xs:` types.
pub use oxixml_infoset::XSD_NAMESPACE;
/// Re-export of [`oxixml_xpath::DocumentResolver`], which supplies what `fn:doc` returns.
pub use oxixml_xpath::DocumentResolver;
/// Re-export of [`oxixml_xpath::XpathValue`], the result of
/// [`CompiledQuery::execute_value`] — a sequence that may hold function items.
pub use oxixml_xpath::XpathValue;
