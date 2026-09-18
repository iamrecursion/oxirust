//! Compiling a stylesheet module tree into the executable form the engine runs.

pub(crate) mod decl;
mod env;
mod instr;
mod ir;
mod statics;

pub(crate) use env::{Avt, AvtPart};
pub use env::{StaticEnv, XsltVersion};
pub(crate) use ir::*;

use crate::error::{XsltError, XsltResult};
use crate::names::{XSLT_NAMESPACE, is_xslt};
use crate::resolve::Resolver;
use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};
use oxixml_qname::XML_NAMESPACE;
use oxixml_xpath::SequenceType;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A stylesheet module flattened into its top-level declarations.
struct Module {
    declarations: Vec<(NodeRef, StaticEnv)>,
    /// The import precedence of the lowest module in this one's import subtree, which is the
    /// bottom of the band `xsl:apply-imports` searches — see
    /// [`StylesheetIr::import_subtree_floors`](ir::StylesheetIr::import_subtree_floors).
    import_subtree_floor: usize,
}

/// What the module that asked for a module does with the declarations it expands to.
enum Fold {
    /// The principal module, which is the result of the expansion.
    Principal,
    /// `xsl:import`: a separate module of lower import precedence.
    Import,
    /// `xsl:include`: folded into the including module at the point of inclusion.
    Include,
}

/// One module part-way through expansion, held on the expander's own stack.
struct Expansion {
    /// The children still to be looked at.
    children: std::vec::IntoIter<NodeRef>,
    /// The environment the children are read in.
    env: StaticEnv,
    /// The declarations found so far, including those folded in from `xsl:include`.
    declarations: Vec<(NodeRef, StaticEnv)>,
    /// The URI this module was loaded from, dropped from the visited chain once it finishes.
    loaded: Option<String>,
    fold: Fold,
    /// The *import group* the module's own declarations belong to — see
    /// [`statics::StaticBindings`]. A module and everything it includes share one; an imported
    /// module gets a fresh one, and so has a different import precedence.
    group: usize,
    /// How many modules were already numbered when this one started expanding. Every module
    /// numbered from here on, until this frame finishes, is a descendant of it in the import
    /// tree, so this is its import subtree's floor.
    floor: usize,
}

impl Expansion {
    /// Starts expanding `element`.
    ///
    /// # Errors
    ///
    /// Whatever the module element's own attributes raise — `XTSE0125` for an unrecognized
    /// `default-collation`, say.
    fn begin(
        element: &NodeRef,
        env: &StaticEnv,
        loaded: Option<String>,
        fold: Fold,
        group: usize,
        floor: usize,
    ) -> XsltResult<Self> {
        if !is_stylesheet_element(element) {
            // A simplified stylesheet: the whole literal result element is one template, and it
            // has no module structure to walk into.
            return Ok(Self {
                children: Vec::new().into_iter(),
                env: env.clone(),
                declarations: vec![(element.clone(), env.clone())],
                loaded,
                fold,
                group,
                floor,
            });
        }
        // The module element carries the defaults its declarations are read under —
        // `default-collation`, `xpath-default-namespace`, `expand-text`, `default-mode`,
        // `version`. An imported or included module starts from a fresh environment, so
        // without this its `xsl:stylesheet` attributes would reach nothing
        // ([XSLT 3.0 §3.6](https://www.w3.org/TR/xslt-30/#stylesheet-element)); the suite's
        // collations-0127 imports a module that sets `default-collation` on exactly this
        // element.
        let module_env = element_env(element, env)?;
        // XSLT 3.0 §3.14.1: "If the xsl:stylesheet or xsl:transform element itself is effectively
        // excluded, the effect is to exclude all the children of the xsl:stylesheet or
        // xsl:transform element, but not the xsl:stylesheet or xsl:transform element or its
        // attributes" — so the module still contributes its defaults, and declares nothing.
        // `use-when-0116` includes a module whose element says `use-when="false()"`.
        let children = if instr::use_when_holds(element, &module_env)? {
            element.children()
        } else {
            Vec::new()
        };
        Ok(Self {
            children: children.into_iter(),
            env: module_env,
            declarations: Vec::new(),
            loaded,
            fold,
            group,
            floor,
        })
    }
}

/// The highest import precedence each key was declared at.
///
/// Every "do the declarations that win on import precedence agree?" check needs this first, and
/// needs it in one pass: a stylesheet can carry thousands of `xsl:output` or `xsl:decimal-format`
/// attributes, and rescanning the whole list once per entry would make compiling one quadratic in
/// the number of declarations.
fn winning_precedences<K: Eq + std::hash::Hash>(
    entries: impl Iterator<Item = (K, usize)>,
) -> HashMap<K, usize> {
    let mut winning: HashMap<K, usize> = HashMap::new();
    for (key, precedence) in entries {
        winning
            .entry(key)
            .and_modify(|highest| *highest = (*highest).max(precedence))
            .or_insert(precedence);
    }
    winning
}

/// A `character` to `string` table declared by `xsl:character-map`.
pub(crate) type CharacterMap = std::collections::BTreeMap<char, String>;

/// The compiler's working state.
pub(crate) struct Compiler {
    pub(crate) ir: StylesheetIr,
    pub(crate) resolver: Option<Arc<dyn Resolver>>,
    pub(crate) precedence: usize,
    pub(crate) order: usize,
    /// Every `xsl:character-map`, by name.
    pub(crate) character_maps: std::collections::HashMap<ExpandedName, CharacterMap>,
    /// `(output format, character map names)` pairs, resolved once every map is known.
    pub(crate) pending_character_maps: Vec<(Option<ExpandedName>, Vec<ExpandedName>)>,
    /// `use-character-maps` written on a character map itself.
    pub(crate) pending_character_map_uses: Vec<(ExpandedName, Vec<ExpandedName>)>,
    /// The template rules written with `mode="#all"`, which every mode the package uses
    /// picks up once they are all known.
    pub(crate) all_mode_rules: Vec<usize>,
    /// Every named mode an `xsl:apply-templates` selects. A mode nothing declares and no
    /// `xsl:template` writes is still a mode "the package uses" when an instruction names it, so
    /// the `mode="#all"` rules belong to it too (XSLT 3.0 §6.4).
    pub(crate) referenced_modes: Vec<ModeKey>,
    /// `(mode, import precedence)` for every `xsl:mode` declaration, so that two at the same
    /// precedence can be reported as `XTSE0545`.
    pub(crate) mode_declarations: Vec<(ModeKey, usize)>,
    /// `(error code, kind, key, import precedence)` for every declaration whose name must be
    /// unique at the highest import precedence that declares it.
    pub(crate) declarations: Vec<(&'static str, &'static str, String, usize)>,
    /// Every `use-attribute-sets` reference, checked once all the sets are known.
    pub(crate) attribute_set_references: Vec<ExpandedName>,
    /// `(format, attribute, value, import precedence)` for every attribute written on an
    /// `xsl:decimal-format`, so that a conflict at the winning precedence can be reported.
    pub(crate) decimal_format_attributes: Vec<(Option<ExpandedName>, String, String, usize)>,
    /// The same, for `xsl:output`.
    pub(crate) output_attributes: Vec<(Option<ExpandedName>, String, String, usize)>,
    /// `(template, parameters)` for every `xsl:call-template`: the name it calls, and every
    /// non-tunnel `xsl:with-param` name *that one call site* supplies — kept per call site,
    /// rather than flattened across every call to the same template, so that one call site's
    /// missing `required="yes"` parameter is not masked by a sibling call site that does
    /// supply it.
    pub(crate) call_template_params: Vec<(ExpandedName, Vec<ExpandedName>)>,
    /// The `input-type-annotations` value each module declares, for `XTSE0265`.
    pub(crate) input_type_annotations: Vec<String>,
    /// Whether an element whose effective version is 1.0 may be evaluated at all, from
    /// [`StaticOptions::with_backwards_compatibility`](crate::StaticOptions::with_backwards_compatibility).
    pub(crate) backwards_compatibility: bool,
    /// `(name, select text, the environment it was written in)` for every global variable or
    /// parameter with a `select` attribute, so that `check_global_self_reference` can re-read
    /// each one once every global name is known.
    pub(crate) global_selects: Vec<(ExpandedName, String, StaticEnv)>,
    /// `(name, as written)` for every `xsl:result-document/@format` that named no output
    /// definition yet, checked once every declaration has been compiled.
    pub(crate) result_document_formats: Vec<(ExpandedName, String)>,
    /// `((use, as as written), import precedence)` for every `xsl:global-context-item`, so that
    /// a second one in a module, or two modules that disagree, can be reported as `XTSE3087`.
    pub(crate) global_context_items: Vec<((ContextItemUse, Option<String>), usize)>,
    /// The values the caller supplied for the stylesheet's *static* parameters, from
    /// [`StaticOptions::with_static_parameter`](crate::StaticOptions::with_static_parameter).
    pub(crate) static_parameters: HashMap<ExpandedName, oxixml_xpath::XpathValue>,
    /// Every static variable and parameter with the value the compilation settled on, kept for
    /// the compiled stylesheet to bind again when it runs.
    pub(crate) bound_statics: Vec<(ExpandedName, oxixml_xpath::XpathValue)>,
}

impl Compiler {
    pub(crate) fn new(resolver: Option<Arc<dyn Resolver>>) -> Self {
        Self {
            ir: StylesheetIr::default(),
            resolver,
            precedence: 0,
            order: 0,
            character_maps: std::collections::HashMap::new(),
            pending_character_maps: Vec::new(),
            pending_character_map_uses: Vec::new(),
            all_mode_rules: Vec::new(),
            referenced_modes: Vec::new(),
            mode_declarations: Vec::new(),
            declarations: Vec::new(),
            attribute_set_references: Vec::new(),
            decimal_format_attributes: Vec::new(),
            output_attributes: Vec::new(),
            call_template_params: Vec::new(),
            input_type_annotations: Vec::new(),
            backwards_compatibility: true,
            global_selects: Vec::new(),
            result_document_formats: Vec::new(),
            global_context_items: Vec::new(),
            static_parameters: HashMap::new(),
            bound_statics: Vec::new(),
        }
    }

    /// Records every namespace prefix declared anywhere in a module, and the module's
    /// `input-type-annotations`.
    fn record_module(&mut self, element: &NodeRef) {
        if let Some(value) = attribute(element, "input-type-annotations") {
            let value = value.trim().to_owned();
            if value == "strip" || value == "preserve" {
                self.input_type_annotations.push(value);
            }
        }
        let mut stack = vec![element.clone()];
        while let Some(node) = stack.pop() {
            for binding in node.namespaces() {
                if let Some(name) = binding.node_name() {
                    self.ir
                        .declared_prefixes
                        .insert(name.local_name().to_owned());
                }
            }
            for child in node.children() {
                if child.node_kind() == NodeKind::Element {
                    stack.push(child);
                }
            }
        }
    }

    /// Records a declaration whose name must be unique at its highest import precedence.
    pub(crate) fn declare(
        &mut self,
        code: &'static str,
        kind: &'static str,
        key: impl Into<String>,
    ) {
        let precedence = self.precedence;
        self.declarations.push((code, kind, key.into(), precedence));
    }

    /// Compiles a whole stylesheet, starting from the document node or document element of the
    /// principal module.
    ///
    /// # Errors
    ///
    /// `XTSE0165` when the root is not a stylesheet, plus every static error the declarations
    /// raise.
    pub(crate) fn compile(&mut self, root: &NodeRef, base_uri: Option<String>) -> XsltResult<()> {
        let element = document_element(root).ok_or_else(|| {
            XsltError::stat1c("XTSE0165", "a stylesheet must have a document element")
        })?;
        // A stylesheet packaged as `xsl:package` needs the whole packaging model, a documented
        // gap; it is reported under this crate's own code rather than mis-compiled as a
        // simplified stylesheet.
        if xslt_local_name(&element) == Some("package") {
            return Err(XsltError::stat1c(
                "OXIXML0001",
                "`xsl:package` is not implemented",
            ));
        }
        let mut env = StaticEnv::new(XsltVersion::V10);
        // The static parameters the caller supplied are in force before the first declaration is
        // read: they can decide, through `use-when` and shadow attributes, which declarations
        // the stylesheet even has.
        env.statics()
            .borrow_mut()
            .set_supplied(std::mem::take(&mut self.static_parameters));
        env.set_base_uri(base_uri.or_else(|| element.base_uri()));
        // A static expression may call `fn:transform`, which needs the resolver and the base URI
        // this compilation is using and has no runtime to read them from.
        let _compiling = crate::run::functions::enter_compilation(
            crate::run::functions::CompilationEnvironment {
                base_uri: env.base_uri().map(str::to_owned),
                resolver: self.resolver.clone(),
            },
        );
        let env = element_env(&element, &env)?;
        self.ir.version = env.version();
        self.ir.default_collation = env.default_collation().map(str::to_owned);

        crate::validate::validate_module(&element)?;
        self.record_module(&element);
        let mut modules = Vec::new();
        let mut visited = HashSet::new();
        let principal = self.expand(&element, &env, &mut modules, &mut visited)?;
        modules.push(principal);
        // The import tree, in the one form `xsl:apply-imports` needs it: each stylesheet level's
        // subtree is the band of precedences from its floor up to its own.
        self.ir.import_subtree_floors = modules
            .iter()
            .map(|module| module.import_subtree_floor)
            .collect();

        // A call to a stylesheet function must compile wherever it is written, so every
        // `xsl:function` is discovered before any expression is.
        self.bound_statics = env.statics().borrow().values();
        let user_functions = Arc::new(scan_user_functions(&modules)?);
        // `xsl:output` and `xsl:character-map` are compiled first, whatever order the
        // declarations were written in, so that an `xsl:result-document` can be resolved
        // against the named output definitions — character maps and all — however late they
        // were declared.
        for (precedence, module) in modules.iter().enumerate() {
            self.precedence = precedence;
            for (node, env) in &module.declarations {
                if matches!(xslt_local_name(node), Some("output" | "character-map")) {
                    decl::compile_declaration(self, node, env)?;
                }
            }
        }
        self.resolve_character_maps()?;
        for (precedence, mut module) in modules.into_iter().enumerate() {
            self.precedence = precedence;
            for (node, env) in &mut module.declarations {
                if matches!(xslt_local_name(node), Some("output" | "character-map")) {
                    continue;
                }
                env.set_user_functions(Arc::clone(&user_functions));
                self.order += 1;
                decl::compile_declaration(self, node, env)?;
            }
        }
        self.finish()
    }

    /// Expands one module: its imports become separate modules of lower precedence, its
    /// includes are folded into it.
    ///
    /// A module tree is walked with an explicit stack rather than by recursion. `xsl:include`
    /// chains are a normal way to layer a large stylesheet, and a chain has no length a
    /// specification bounds, so the depth of the tree must not be the depth of the native
    /// stack: expanding a chain of any length costs the same few hundred bytes here.
    fn expand(
        &mut self,
        element: &NodeRef,
        env: &StaticEnv,
        imports: &mut Vec<Module>,
        visited: &mut HashSet<String>,
    ) -> XsltResult<Module> {
        let floor = imports.len();
        let mut stack = vec![Expansion::begin(
            element,
            env,
            None,
            Fold::Principal,
            0,
            floor,
        )?];
        let mut groups = 1_usize;
        let mut principal = Module {
            declarations: Vec::new(),
            import_subtree_floor: floor,
        };
        while !stack.is_empty() {
            let next = stack.last_mut().and_then(|frame| frame.children.next());
            let Some(child) = next else {
                // The module is finished: it leaves the chain currently being expanded, and its
                // declarations go wherever the module that asked for it puts them.
                let Some(finished) = stack.pop() else { break };
                if let Some(loaded) = finished.loaded {
                    visited.remove(&loaded);
                }
                let module = Module {
                    declarations: finished.declarations,
                    import_subtree_floor: finished.floor,
                };
                match finished.fold {
                    Fold::Principal => principal = module,
                    Fold::Import => imports.push(module),
                    Fold::Include => {
                        if let Some(parent) = stack.last_mut() {
                            parent.declarations.extend(module.declarations);
                        }
                    }
                }
                continue;
            };
            if child.node_kind() != NodeKind::Element {
                continue;
            }
            let child_env = match stack.last() {
                Some(frame) => element_env(&child, &frame.env)?,
                None => break,
            };
            let local = xslt_local_name(&child);
            if !matches!(local, Some("import") | Some("include")) {
                let group = stack.last().map_or(0, |frame| frame.group);
                // XSLT 3.0 §9.6 binds a static variable while the module tree is expanded, in
                // stylesheet tree order, so that the next declaration's static expressions —
                // and every shadow attribute after it — can already see its value. The
                // declaration stays in the module: "static variables and parameters are global
                // variables and can be used in the same way as other global variables", so it
                // is compiled into the IR as well.
                statics::collect_declaration(&child, &child_env, group)?;
                if let Some(frame) = stack.last_mut() {
                    frame.declarations.push((child, child_env));
                }
                continue;
            }
            // §3.14.1 excludes the element "together with all the nodes having that element as
            // an ancestor", and for `xsl:include`/`xsl:import` that is the whole module it names:
            // `use-when-0220` includes a module under `use-when="false()"` and expects none of
            // its declarations, whatever the module's own `use-when` says.
            if !instr::use_when_holds(&child, &child_env)? {
                continue;
            }
            let importing = local == Some("import");
            let Some(href) = child_env.shadowed_attribute(&child, "href")? else {
                return Err(XsltError::stat1c(
                    "XTSE0165",
                    "xsl:import and xsl:include require an href attribute",
                ));
            };
            // `visited` records the chain currently being expanded, not every module ever
            // loaded: XSLT 3.0 §3.10 forbids a module that *contains itself*, and says nothing
            // against two modules that each import a third. The entry is dropped when the
            // module's own subtree finishes, which keeps a diamond legal while a cycle is still
            // caught on the way in.
            let module = self.load_module(&href, &child_env, visited, importing)?;
            let Some((root, module_env)) = module else {
                return Err(XsltError::stat1c(
                    "XTSE0165",
                    format!("the stylesheet module `{href}` could not be retrieved"),
                ));
            };
            let Some(module_element) = document_element(&root) else {
                return Err(XsltError::stat1c(
                    "XTSE0165",
                    format!("the stylesheet module `{href}` has no document element"),
                ));
            };
            crate::validate::validate_module(&module_element)?;
            self.record_module(&module_element);
            let loaded = module_env.base_uri().map(str::to_owned);
            let (fold, group) = if importing {
                groups += 1;
                (Fold::Import, groups - 1)
            } else {
                (Fold::Include, stack.last().map_or(0, |frame| frame.group))
            };
            stack.push(Expansion::begin(
                &module_element,
                &module_env,
                loaded,
                fold,
                group,
                imports.len(),
            )?);
        }
        Ok(principal)
    }

    fn load_module(
        &self,
        href: &str,
        env: &StaticEnv,
        visited: &mut HashSet<String>,
        importing: bool,
    ) -> XsltResult<Option<(NodeRef, StaticEnv)>> {
        let Some(resolver) = &self.resolver else {
            return Ok(None);
        };
        let resolved = resolver.resolve_uri(href, env.base_uri());
        if !visited.insert(resolved.clone()) {
            // XSLT 3.0 §3.10 gives the two directions separate codes: `XTSE0180` for a module
            // that includes itself, `XTSE0210` for one that imports itself.
            let code = if importing { "XTSE0210" } else { "XTSE0180" };
            return Err(XsltError::stat1c(
                code,
                format!("the stylesheet module `{resolved}` refers to itself"),
            ));
        }
        let Some(node) = resolver.stylesheet(&resolved) else {
            return Ok(None);
        };
        let mut module_env = env.for_module(env.version());
        module_env.set_base_uri(Some(resolved));
        Ok(Some((node, module_env)))
    }

    /// Folds every `xsl:character-map` into the output definitions that name it.
    ///
    /// This runs after the `xsl:output` and `xsl:character-map` declarations of every module
    /// have been read but *before* any instruction is compiled, because `xsl:result-document`
    /// captures the output definition it names as it compiles: `output-0308` names a definition
    /// whose whole contribution is the character map, and would otherwise capture the
    /// definition a step too early.
    ///
    /// # Errors
    ///
    /// `XTSE1590` for a reference to a map that no declaration defines, and `XTSE1585` for a
    /// cycle among the maps.
    fn resolve_character_maps(&mut self) -> XsltResult<()> {
        self.check_character_maps()?;
        // A character map may itself use other character maps.
        for (owner, used) in std::mem::take(&mut self.pending_character_map_uses) {
            let mut merged = CharacterMap::new();
            for name in used {
                if let Some(map) = self.character_maps.get(&name) {
                    merged.extend(map.iter().map(|(key, value)| (*key, value.clone())));
                }
            }
            if let Some(map) = self.character_maps.get(&owner) {
                merged.extend(map.iter().map(|(key, value)| (*key, value.clone())));
            }
            self.character_maps.insert(owner, merged);
        }
        // XSLT 3.0 §26.1 puts `use-character-maps` among the attributes whose values are
        // *combined* rather than made to agree: the effective value of one output definition is
        // the union of the maps every contributing `xsl:output` names. So each declaration adds
        // to what the previous ones settled, and where two of them map the same character the
        // later declaration wins — which is what `character-map-005` asserts of a module with
        // two unnamed `xsl:output` elements naming two maps that disagree about `c`.
        for (format, names) in std::mem::take(&mut self.pending_character_maps) {
            let mut merged = match &format {
                None => self.ir.output.character_map.clone(),
                Some(name) => self
                    .ir
                    .named_outputs
                    .get(name)
                    .map(|properties| properties.character_map.clone())
                    .unwrap_or_default(),
            };
            for name in names {
                if let Some(map) = self.character_maps.get(&name) {
                    merged.extend(map.iter().map(|(key, value)| (*key, value.clone())));
                }
            }
            match format {
                None => self.ir.output.character_map = merged,
                Some(name) => {
                    if let Some(properties) = self.ir.named_outputs.get(&name) {
                        let mut updated = (**properties).clone();
                        updated.character_map = merged;
                        self.ir.named_outputs.insert(name, Arc::new(updated));
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies the finishing touches: sorting the template rules and recording whether any
    /// whitespace stripping is in force.
    fn finish(&mut self) -> XsltResult<()> {
        // XSLT 3.0 §4.3: two modules of one package may not disagree about whether the type
        // annotations of the source are stripped.
        if self
            .input_type_annotations
            .iter()
            .any(|value| value == "strip")
            && self
                .input_type_annotations
                .iter()
                .any(|value| value == "preserve")
        {
            return Err(XsltError::stat1c(
                "XTSE0265",
                "one module specifies input-type-annotations=\"strip\" and another \"preserve\"",
            ));
        }
        for (name, written) in std::mem::take(&mut self.result_document_formats) {
            if !self.ir.named_outputs.contains_key(&name) {
                return Err(XsltError::stat1c(
                    "XTDE1460",
                    format!("`{written}` names no xsl:output declaration"),
                ));
            }
        }
        // A rule written with `mode="#all"` belongs to every mode the package uses, which is
        // only known once every declaration has been compiled (XSLT 3.0 §6.4). A mode nothing
        // declares and no `xsl:template` names is one of them as soon as an `xsl:apply-templates`
        // selects it: `match-245` writes three such modes and nothing but `mode="#all"` rules.
        // The buckets are created only when there is something to put in them, so that a mode
        // named by an instruction and served by no rule at all stays the `XTDE0045` it was.
        if !self.all_mode_rules.is_empty() {
            self.ir.modes.entry(ModeKey::Default).or_default();
            for key in std::mem::take(&mut self.referenced_modes) {
                self.ir.modes.entry(key).or_default();
            }
            let rules = std::mem::take(&mut self.all_mode_rules);
            for mode in self.ir.modes.values_mut() {
                mode.rules.extend(rules.iter().copied());
            }
        }
        self.ir.strips_any_space = self.ir.space_controls.iter().any(|control| control.strip);
        let precedence: Vec<(usize, f64, usize)> = self
            .ir
            .templates
            .iter()
            .map(|template| {
                (
                    template.import_precedence,
                    template.priority,
                    template.declaration_order,
                )
            })
            .collect();
        self.check_attribute_set_cycles()?;
        self.check_mode_declarations()?;
        self.check_unique_declarations()?;
        self.check_attribute_set_references()?;
        self.check_namespace_aliases()?;
        self.check_decimal_formats()?;
        self.check_space_controls()?;
        self.check_output_declarations()?;
        self.check_call_template_params()?;
        self.check_global_self_reference()?;
        for mode in self.ir.modes.values_mut() {
            mode.rules.sort_by(|left, right| {
                let a = precedence.get(*left).copied().unwrap_or_default();
                let b = precedence.get(*right).copied().unwrap_or_default();
                b.0.cmp(&a.0)
                    .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
                    .then(b.2.cmp(&a.2))
            });
        }
        Ok(())
    }

    /// Rejects two `xsl:output` declarations that give one serialization parameter different
    /// values at the winning import precedence.
    ///
    /// # Errors
    ///
    /// `XTSE1560`, from [XSLT 3.0 §26.1](https://www.w3.org/TR/xslt-30/#serialization).
    fn check_output_declarations(&self) -> XsltResult<()> {
        // These two accumulate rather than conflict.
        let conflicting =
            |attribute: &str| !matches!(attribute, "cdata-section-elements" | "use-character-maps");
        let winning = winning_precedences(
            self.output_attributes
                .iter()
                .filter(|(_, attribute, _, _)| conflicting(attribute))
                .map(|(name, attribute, _, precedence)| ((name, attribute.as_str()), *precedence)),
        );
        let mut agreed: HashMap<(&Option<ExpandedName>, &str), &String> = HashMap::new();
        for (name, attribute, value, precedence) in &self.output_attributes {
            let key = (name, attribute.as_str());
            if !conflicting(attribute) || winning.get(&key) != Some(precedence) {
                continue;
            }
            if agreed.entry(key).or_insert(value) != &value {
                return Err(XsltError::stat1c(
                    "XTSE1560",
                    format!("two `xsl:output` declarations disagree about `{attribute}`"),
                ));
            }
        }
        Ok(())
    }

    /// Rejects an `xsl:with-param` of `xsl:call-template` that names no parameter of the
    /// template it calls, and — statically knowable exactly because `xsl:call-template` names
    /// its target — that every `required="yes"` non-tunnel parameter of that template is among
    /// them.
    ///
    /// A *tunnel* parameter is not checked here: it need not appear as a sibling `with-param`
    /// at all, since it can instead be inherited from an enclosing `xsl:apply-templates` or
    /// `xsl:call-template`'s own tunnel set, which this per-call-site view cannot see.
    ///
    /// # Errors
    ///
    /// `XTSE0680` for a supplied name the template does not declare, `XTSE0690` for a
    /// `required="yes"` parameter no `xsl:with-param` at the call site supplies — both
    /// [XSLT 3.0 §10.1.1](https://www.w3.org/TR/xslt-30/#passing-parameters).
    fn check_call_template_params(&self) -> XsltResult<()> {
        for (template, supplied) in &self.call_template_params {
            let Some(index) = self.ir.named_templates.get(template) else {
                continue;
            };
            let Some(declared) = self.ir.templates.get(*index) else {
                continue;
            };
            for parameter in supplied {
                let known = declared
                    .params
                    .iter()
                    .any(|param| !param.tunnel && param.name == *parameter);
                if !known {
                    return Err(XsltError::stat1c(
                        "XTSE0680",
                        format!("the template `{template}` has no parameter named `{parameter}`"),
                    ));
                }
            }
            for param in declared.params.iter() {
                if param.required && !param.tunnel && !supplied.contains(&param.name) {
                    return Err(XsltError::stat1c(
                        "XTSE0690",
                        format!(
                            "the call to template `{template}` does not supply its required \
                             parameter `{}`",
                            param.name
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Rejects a global variable or parameter whose own `select` refers to itself.
    ///
    /// [XSLT 3.0 §9.9](https://www.w3.org/TR/xslt-30/#variable-scope): "A global variable
    /// binding element is visible everywhere in the containing package (including other
    /// stylesheet modules) **except within the `xsl:variable` or `xsl:param` element itself**
    /// and any region where it is shadowed by another variable binding." A declaration that
    /// names itself therefore refers to a variable that is not in scope, which is the *static*
    /// error `XPST0008` — not the dynamic circularity `XTDE0640`, which §9.11 reserves for a
    /// dependency that genuinely runs through other declarations. The suite settles the same
    /// question twice: variable-0120 (an overriding global that tries to read the one it
    /// overrides) requires `XPST0008`, and error-0640b accepts either code for the plain
    /// self-reference, its catalogue entry calling `XPST0008` "arguably preferable".
    ///
    /// The judgement is left to the XPath parser rather than made by inspecting the syntax
    /// tree here: each `select` is re-compiled with every *other* global name in scope and its
    /// own name out of scope, so range variables that rebind the name (`for $v in … return $v`)
    /// are handled by the same code that handles them everywhere else. Only an `XPST0008`
    /// naming the declaration's own variable is reported; anything else the strict scope
    /// rejects is left to the run-time behaviour the engine already has.
    ///
    /// # Errors
    ///
    /// `XPST0008` when a global declaration's `select` refers to the variable it declares.
    fn check_global_self_reference(&self) -> XsltResult<()> {
        let mut names: Vec<ExpandedName> = self
            .ir
            .globals
            .iter()
            .map(|global| global.name.clone())
            .collect();
        let positions: HashMap<ExpandedName, usize> = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        // A name that no expression can mention, which stands in for the variable being checked
        // while its own declaration is compiled. Rebuilding the visible list per declaration
        // instead — one clone of every global name, once per global — is what made compiling a
        // stylesheet with thousands of them quadratic.
        let hidden = ExpandedName::new(Some("urn:x-oxixml:hidden"), "global");
        for (name, text, env) in &self.global_selects {
            // Only an expression that mentions a variable at all can mention itself, and
            // re-compiling one costs a static context holding every global name — so the ones
            // that plainly cannot are not compiled a second time.
            if !text.contains('$') {
                continue;
            }
            let index = positions.get(name).copied();
            let displaced = index
                .and_then(|index| names.get_mut(index))
                .map(|slot| std::mem::replace(slot, hidden.clone()));
            let outcome = env.compile_expression_in_scope(text, &names);
            if let (Some(index), Some(displaced)) = (index, displaced)
                && let Some(slot) = names.get_mut(index)
            {
                *slot = displaced;
            }
            let Err(error) = outcome else {
                continue;
            };
            if error.is("XPST0008") && error.message().contains(&format!("${name}")) {
                return Err(XsltError::stat1c(
                    "XPST0008",
                    format!(
                        "the global variable `${name}` is not in scope within its own \
                         declaration, so `{}` cannot refer to it",
                        text.trim()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Rejects a `use-character-maps` naming a map the stylesheet does not declare, or a
    /// character map that uses itself.
    ///
    /// # Errors
    ///
    /// `XTSE1590` for a dangling reference, `XTSE1600` for a cycle
    /// ([XSLT 3.0 §20.1](https://www.w3.org/TR/xslt-30/#character-maps)).
    fn check_character_maps(&self) -> XsltResult<()> {
        for (_, used) in &self.pending_character_map_uses {
            for name in used {
                if !self.character_maps.contains_key(name) {
                    return Err(XsltError::stat1c(
                        "XTSE1590",
                        format!("no `xsl:character-map` is named `{name}`"),
                    ));
                }
            }
        }
        for (_, used) in &self.pending_character_maps {
            for name in used {
                if !self.character_maps.contains_key(name) {
                    return Err(XsltError::stat1c(
                        "XTSE1590",
                        format!("no `xsl:character-map` is named `{name}`"),
                    ));
                }
            }
        }
        let mut edges: HashMap<&ExpandedName, Vec<&ExpandedName>> = HashMap::new();
        for (owner, used) in &self.pending_character_map_uses {
            edges.entry(owner).or_default().extend(used.iter());
        }
        check_reference_cycles(
            self.pending_character_map_uses
                .iter()
                .map(|(owner, _)| owner),
            &edges,
            "XTSE1600",
            "character map",
        )
    }

    /// Rejects a name test claimed by both `xsl:strip-space` and `xsl:preserve-space`.
    ///
    /// # Errors
    ///
    /// `XTSE0270`, which [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#strip) raises when "the
    /// same NameTest appears in both an xsl:strip-space and an xsl:preserve-space declaration
    /// [with] the same import precedence". Two name tests are the same when they match the same
    /// set of names, which is what `SpaceTest` equality decides — an `ExpandedName` compares by
    /// namespace and local name, so `b` and `Q{}b` are one test, as `strip-space-028` requires.
    fn check_space_controls(&self) -> XsltResult<()> {
        for (index, control) in self.ir.space_controls.iter().enumerate() {
            let clash = self.ir.space_controls[..index].iter().find(|earlier| {
                earlier.strip != control.strip
                    && earlier.import_precedence == control.import_precedence
                    && earlier.test == control.test
            });
            if clash.is_some() {
                return Err(XsltError::stat1c(
                    "XTSE0270",
                    "one name test appears in both xsl:strip-space and xsl:preserve-space at \
                     the same import precedence",
                ));
            }
        }
        Ok(())
    }

    /// Rejects a decimal format whose declarations disagree, or whose characters overlap.
    ///
    /// # Errors
    ///
    /// `XTSE1290` when two declarations at the winning import precedence give one attribute
    /// different values, `XTSE1295` and `XTSE1300` from the format's own consistency rules.
    fn check_decimal_formats(&self) -> XsltResult<()> {
        let winning = winning_precedences(
            self.decimal_format_attributes
                .iter()
                .map(|(name, attribute, _, precedence)| ((name, attribute.as_str()), *precedence)),
        );
        let mut agreed: HashMap<(&Option<ExpandedName>, &str), &String> = HashMap::new();
        for (name, attribute, value, precedence) in &self.decimal_format_attributes {
            let key = (name, attribute.as_str());
            if winning.get(&key) != Some(precedence) {
                continue;
            }
            if agreed.entry(key).or_insert(value) != &value {
                return Err(XsltError::stat1c(
                    "XTSE1290",
                    format!("two `xsl:decimal-format` declarations disagree about `{attribute}`"),
                ));
            }
        }
        for format in self.ir.decimal_formats.values() {
            decl::check_decimal_format(format)?;
        }
        Ok(())
    }

    /// Rejects a name declared twice at the highest import precedence that declares it.
    ///
    /// # Errors
    ///
    /// The code each declaration recorded: `XTSE0630` for a global variable, `XTSE0660` for a
    /// named template, `XTSE0770` for a stylesheet function, `XTSE1580` for a character map.
    /// A conflict at a lower precedence is resolved by the higher one and is not an error.
    fn check_unique_declarations(&self) -> XsltResult<()> {
        // `(highest precedence seen, how many declarations at it)`, by kind and name. Keeping
        // it in a map rather than a list is what stops a stylesheet with thousands of global
        // variables from costing a scan of every earlier one per declaration.
        let mut seen: HashMap<(&str, &str), (usize, usize)> = HashMap::new();
        for (code, kind, key, precedence) in &self.declarations {
            let entry = seen.entry((kind, key.as_str())).or_insert((*precedence, 0));
            match precedence.cmp(&entry.0) {
                std::cmp::Ordering::Greater => *entry = (*precedence, 1),
                std::cmp::Ordering::Equal => entry.1 += 1,
                std::cmp::Ordering::Less => {}
            }
            if entry.1 > 1 {
                return Err(XsltError::stat1c(
                    *code,
                    format!("`{key}` is declared twice at the same import precedence"),
                ));
            }
        }
        Ok(())
    }

    /// Rejects a `use-attribute-sets` naming a set the stylesheet does not declare.
    ///
    /// # Errors
    ///
    /// `XTSE0710`, which [XSLT 3.0 §10.2](https://www.w3.org/TR/xslt-30/#attribute-sets)
    /// raises for a dangling reference.
    fn check_attribute_set_references(&self) -> XsltResult<()> {
        for name in &self.attribute_set_references {
            if !self.ir.attribute_sets.contains_key(name) {
                return Err(XsltError::stat1c(
                    "XTSE0710",
                    format!("no `xsl:attribute-set` is named `{name}`"),
                ));
            }
        }
        Ok(())
    }

    /// Rejects two `xsl:namespace-alias` declarations that map the same namespace to different
    /// targets at the same import precedence.
    ///
    /// # Errors
    ///
    /// `XTSE0810`, from [XSLT 3.0 §11.1.5](https://www.w3.org/TR/xslt-30/#literal-result-element).
    fn check_namespace_aliases(&self) -> XsltResult<()> {
        let winning = winning_precedences(
            self.ir
                .namespace_aliases
                .iter()
                .map(|(literal, _, precedence)| (literal, *precedence)),
        );
        let mut agreed: HashMap<&String, &String> = HashMap::new();
        for (literal, target, precedence) in &self.ir.namespace_aliases {
            if winning.get(literal) != Some(precedence) {
                continue;
            }
            if agreed.entry(literal).or_insert(target) != &target {
                return Err(XsltError::stat1c(
                    "XTSE0810",
                    format!("`{literal}` is aliased to two namespaces at the same precedence"),
                ));
            }
        }
        Ok(())
    }

    /// Rejects a mode with two `xsl:mode` declarations at its highest import precedence.
    ///
    /// # Errors
    ///
    /// `XTSE0545`, which [XSLT 3.0 §6.6.1](https://www.w3.org/TR/xslt-30/#declaring-modes)
    /// raises when the declaration that wins on import precedence is not unique. A conflict at
    /// a lower precedence is resolved by the higher one and is not an error.
    fn check_mode_declarations(&self) -> XsltResult<()> {
        // As in `check_unique_declarations`: `(highest precedence seen, how many at it)`, kept
        // by mode so that the check costs one pass rather than one scan per declaration.
        let mut seen: HashMap<&ModeKey, (usize, usize)> = HashMap::new();
        for (mode, precedence) in &self.mode_declarations {
            let entry = seen.entry(mode).or_insert((*precedence, 0));
            match precedence.cmp(&entry.0) {
                std::cmp::Ordering::Greater => *entry = (*precedence, 1),
                std::cmp::Ordering::Equal => entry.1 += 1,
                std::cmp::Ordering::Less => {}
            }
        }
        // The count is only conclusive once the whole pass is done: a mode may be declared more
        // than once at a lower precedence and still be legal when a single declaration at a
        // higher precedence resolves the conflict, and that winning declaration can appear after
        // the duplicates in declaration order. Comparing mid-pass would reject the legal case.
        if seen.values().any(|(_, count)| *count > 1) {
            return Err(XsltError::stat1c(
                "XTSE0545",
                "a mode has two `xsl:mode` declarations at the same import precedence",
            ));
        }
        Ok(())
    }

    /// Rejects an `xsl:attribute-set` that uses itself, directly or through others.
    ///
    /// # Errors
    ///
    /// `XTSE0720`, which [XSLT 3.0 §10.2](https://www.w3.org/TR/xslt-30/#attribute-sets) makes
    /// a static error.
    fn check_attribute_set_cycles(&self) -> XsltResult<()> {
        let mut edges: HashMap<&ExpandedName, Vec<&ExpandedName>> = HashMap::new();
        for (name, definitions) in &self.ir.attribute_sets {
            let entry = edges.entry(name).or_default();
            for definition in definitions {
                entry.extend(definition.uses.iter());
            }
        }
        check_reference_cycles(
            self.ir.attribute_sets.keys(),
            &edges,
            "XTSE0720",
            "attribute set",
        )
    }
}

/// One step of a cycle walk: a name whose references are about to be followed, or one whose
/// references have all been checked.
enum Reference<'a> {
    Enter(&'a ExpandedName),
    Leave(&'a ExpandedName),
}

/// Rejects a cycle in a *name uses name* graph.
///
/// Both `use-attribute-sets` and `use-character-maps` need this, and neither may cost a native
/// stack frame per link: nothing bounds how long a chain of sets each using the next may be, so
/// the walk keeps its own stack. A name already proved acyclic is remembered, which is what
/// keeps a graph where many names reach the same one from being walked once per path to it.
///
/// # Errors
///
/// `code` when a name is reached again while it is still on the path from a root to it.
fn check_reference_cycles<'a>(
    roots: impl IntoIterator<Item = &'a ExpandedName>,
    edges: &HashMap<&'a ExpandedName, Vec<&'a ExpandedName>>,
    code: &'static str,
    what: &str,
) -> XsltResult<()> {
    let mut settled: HashSet<&ExpandedName> = HashSet::new();
    let mut on_path: HashSet<&ExpandedName> = HashSet::new();
    let mut stack: Vec<Reference<'a>> = Vec::new();
    for root in roots {
        stack.push(Reference::Enter(root));
        while let Some(step) = stack.pop() {
            match step {
                Reference::Leave(name) => {
                    on_path.remove(name);
                    settled.insert(name);
                }
                Reference::Enter(name) => {
                    if settled.contains(name) {
                        continue;
                    }
                    if !on_path.insert(name) {
                        return Err(XsltError::stat1c(
                            code,
                            format!("the {what} `{name}` references itself"),
                        ));
                    }
                    stack.push(Reference::Leave(name));
                    for next in edges.get(name).into_iter().flatten() {
                        stack.push(Reference::Enter(next));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Collects the name and arity of every `xsl:function` the modules declare.
fn scan_user_functions(modules: &[Module]) -> XsltResult<Vec<FunctionSignature>> {
    let mut out: Vec<FunctionSignature> = Vec::new();
    for module in modules {
        for (node, env) in &module.declarations {
            if xslt_local_name(node) != Some("function") {
                continue;
            }
            // The name may be supplied by a shadow attribute — `_name="{$f}"` — which is
            // expanded here exactly as the declaration itself expands it, so that a call
            // written before the declaration resolves against the same name.
            let Some(lexical) = env.shadowed_attribute(node, "name")? else {
                continue;
            };
            let name = env.resolve_attribute_name(&lexical)?;
            let parameters: Vec<NodeRef> = node
                .children()
                .into_iter()
                .filter(|child| {
                    child.node_kind() == NodeKind::Element
                        && xslt_local_name(child) == Some("param")
                })
                .collect();
            let arity = parameters.len();
            if out
                .iter()
                .any(|entry| entry.name == name && entry.arity == arity)
            {
                continue;
            }
            let declared = declared_signature(&parameters, node, env)?;
            out.push(FunctionSignature {
                name,
                arity,
                declared,
            });
        }
    }
    Ok(out)
}

/// The parameter and result types an `xsl:function` declares, or `None` when one of them is a
/// sequence type this processor does not recognize.
///
/// An absent `as` is `item()*` on both sides —
/// [XSLT 3.0 §10.3](https://www.w3.org/TR/xslt-30/#stylesheet-functions): "if the `as` attribute
/// is omitted, the default is `item()*`" — so a function that declares nothing still publishes a
/// signature, and `local:f#1 instance of function(item()*) as item()*` holds for it.
fn declared_signature(
    parameters: &[NodeRef],
    function: &NodeRef,
    env: &StaticEnv,
) -> XsltResult<Option<(Vec<SequenceType>, SequenceType)>> {
    let mut types = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        match declared_type(parameter, env)? {
            Some(declared) => types.push(declared),
            None => return Ok(None),
        }
    }
    Ok(declared_type(function, env)?.map(|result| (types, result)))
}

/// The type one `as` attribute denotes, `item()*` when there is none, and `None` when the text
/// is a sequence type naming something unrecognized.
fn declared_type(element: &NodeRef, env: &StaticEnv) -> XsltResult<Option<SequenceType>> {
    let Some(text) = env.shadowed_attribute(element, "as")? else {
        return Ok(Some(crate::typing::RequiredType::any_sequence()));
    };
    Ok(crate::typing::RequiredType::compile(env, &text)?
        .sequence_type()
        .cloned())
}

/// The document element of `root`, which may itself already be an element.
pub(crate) fn document_element(root: &NodeRef) -> Option<NodeRef> {
    match root.node_kind() {
        NodeKind::Element => Some(root.clone()),
        NodeKind::Document => root
            .children()
            .into_iter()
            .find(|child| child.node_kind() == NodeKind::Element),
        _ => None,
    }
}

/// Whether `element` is `xsl:stylesheet` or `xsl:transform`.
pub(crate) fn is_stylesheet_element(element: &NodeRef) -> bool {
    matches!(
        xslt_local_name(element),
        Some("stylesheet") | Some("transform")
    )
}

/// The local name of an element in the XSLT namespace.
pub(crate) fn xslt_local_name(element: &NodeRef) -> Option<&'static str> {
    let name = element.node_name()?;
    if name.namespace() != Some(XSLT_NAMESPACE) {
        return None;
    }
    crate::names::xslt_element(name.local_name())
}

/// Whether `text` is made up entirely of the characters
/// [XML 1.0 §2.3](https://www.w3.org/TR/xml/#sec-common-syn) calls white space.
///
/// This is deliberately not `str::trim().is_empty()`. Rust trims the Unicode `White_Space`
/// property, which is a strictly larger set: it counts U+00A0 NO-BREAK SPACE, the ideographic
/// space and the whole run of typographic spaces. XML recognizes only #x20, #x9, #xD and #xA, so
/// a stylesheet that writes `&#xa0;` as the content of an element is writing *text*, and
/// [XSLT 3.0 §4.3](https://www.w3.org/TR/xslt-30/#stripping-whitespace) — which strips only
/// "whitespace text nodes" — must keep it.
pub(crate) fn is_xml_whitespace(text: &str) -> bool {
    text.chars()
        .all(|character| matches!(character, '\u{20}' | '\u{9}' | '\u{d}' | '\u{a}'))
}

/// The value of an unprefixed attribute.
pub(crate) fn attribute(element: &NodeRef, name: &str) -> Option<String> {
    element.attributes().into_iter().find_map(|attribute| {
        let attribute_name = attribute.node_name()?;
        (attribute_name.namespace().is_none() && attribute_name.local_name() == name)
            .then(|| attribute.string_value())
    })
}

/// The value of an attribute in the XSLT namespace, such as `xsl:version` on a literal result
/// element.
pub(crate) fn xslt_attribute(element: &NodeRef, name: &str) -> Option<String> {
    element.attributes().into_iter().find_map(|attribute| {
        let attribute_name = attribute.node_name()?;
        (attribute_name.namespace() == Some(XSLT_NAMESPACE) && attribute_name.local_name() == name)
            .then(|| attribute.string_value())
    })
}

/// Builds the static environment an element sees, from `parent` plus everything the element
/// itself declares.
///
/// # Errors
///
/// Any static error the attributes raise.
pub(crate) fn element_env(element: &NodeRef, parent: &StaticEnv) -> XsltResult<StaticEnv> {
    let mut env = parent.clone();
    let bindings: Vec<(Option<String>, String)> = element
        .namespaces()
        .into_iter()
        .map(|namespace| {
            (
                namespace.node_name().map(|n| n.local_name().to_owned()),
                namespace.string_value(),
            )
        })
        .collect();
    // The namespace axis reports every binding *in scope* for the element, so a prefix the
    // parent had and this element does not was undeclared here — `xmlns=""` for the default
    // namespace (XML Namespaces 1.0 §6.2), `xmlns:p=""` for a prefixed one under 1.1. Removing
    // it matters because an unprefixed `name` attribute expands against the default namespace in
    // scope where it is written: `namespace-3109` writes `<xsl:element name="foo" xmlns=""/>`
    // under a module whose default namespace is `http://testguys.com/` and wants `foo` in no
    // namespace at all.
    env.undeclare_namespaces_absent_from(bindings.iter().map(|(prefix, _)| prefix.as_deref()));
    env.declare_namespaces(
        bindings
            .iter()
            .map(|(prefix, uri)| (prefix.as_deref(), uri.as_str())),
    );
    // XSLT 3.0 §3.9: the effective version comes from the `[xsl:]version` attribute, "excluding
    // the version attribute on an xsl:output element", whose value names a version of the
    // *output method* and has nothing to do with the XSLT language level.
    //
    // Each of the settings read here is a no-namespace attribute of an XSLT element whenever the
    // element is one, so §3.14.2's shadow spelling reaches all of them: `shadow-003` writes
    // `_version="{system-property('xsl:version')}"` on the module element and `shadow-006` writes
    // `_xpath-default-namespace="{$ONE}"` on a template. The `xsl:`-prefixed spelling a *literal
    // result element* uses is not shadowable — §3.14.2 excludes attributes in a namespace — so
    // that one is read as written.
    let version = match env.shadowed_attribute(element, "version")? {
        Some(version) if is_xslt(element) && xslt_local_name(element) != Some("output") => {
            Some(version)
        }
        _ => None,
    };
    if let Some(version) = version {
        env::check_version_text(&version)?;
        env.set_version_text(&version);
    }
    if let Some(version) = xslt_attribute(element, "version") {
        env::check_version_text(&version)?;
        env.set_version_text(&version);
    }
    // On an element that is not itself in the XSLT namespace the attribute has to be spelled
    // `xsl:xpath-default-namespace`: an unprefixed one belongs to the literal result element and
    // is copied to the result, saying nothing about the expressions written inside it.
    // `xpath-default-namespace-0104` writes both spellings on two sibling result elements and
    // expects only the second to select anything.
    let default_namespace = match env.shadowed_attribute(element, "xpath-default-namespace")? {
        Some(uri) if is_xslt(element) => Some(uri),
        _ => xslt_attribute(element, "xpath-default-namespace"),
    };
    if let Some(uri) = default_namespace {
        env.set_default_element_namespace((!uri.is_empty()).then_some(uri));
    }
    let default_collation = match env.shadowed_attribute(element, "default-collation")? {
        Some(collation) if is_xslt(element) => Some(collation),
        _ => xslt_attribute(element, "default-collation"),
    };
    if let Some(collation) = default_collation {
        // XSLT 3.0 §5.4.4: the attribute holds a list of collation URIs and the first the
        // processor recognizes wins; it is `XTSE0125` if it recognizes none of them.
        let chosen = collation
            .split_whitespace()
            .find(|uri| crate::run::is_recognized_collation(uri));
        let Some(chosen) = chosen else {
            return Err(XsltError::stat1c(
                "XTSE0125",
                format!("`{collation}` names no collation this processor recognizes"),
            ));
        };
        env.set_default_collation(Some(chosen.to_owned()));
    }
    // The shadow spelling supplies a computed value; the plain one is read literally, because
    // XSLT 3.0 §5.7 gives `[xsl:]expand-text` the value space of `xs:boolean` and not that of an
    // attribute value template. `cvt-008` writes `expand-text="{$yesOrNo}"` and `cvt-009` writes
    // `expand-text="TRUE"`, and both are `XTSE0020` rather than a silent "no".
    let expand = match env.shadowed_attribute(element, "expand-text")? {
        Some(expand) if is_xslt(element) => Some(expand),
        _ => xslt_attribute(element, "expand-text"),
    };
    if let Some(expand) = expand {
        let trimmed = expand.trim();
        let Some(expands) = crate::validate::boolean_value(trimmed) else {
            return Err(XsltError::stat1c(
                "XTSE0020",
                format!("`{trimmed}` is not a value for the `expand-text` attribute"),
            ));
        };
        env.set_expand_text(expands);
    }
    let default_mode = match env.shadowed_attribute(element, "default-mode")? {
        Some(mode) if is_xslt(element) => Some(mode),
        _ => xslt_attribute(element, "default-mode"),
    };
    if let Some(mode) = default_mode {
        let trimmed = mode.trim();
        env.set_default_mode(match trimmed {
            "#unnamed" | "#default" => None,
            name => Some(env.resolve_attribute_name(name)?),
        });
    }
    let base = element.attributes().into_iter().find_map(|attribute| {
        let name = attribute.node_name()?;
        (name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "base")
            .then(|| attribute.string_value())
    });
    if let Some(base) = base {
        // XML Base §3 applies RFC 3986 §5.1.1 to the attribute itself: an `xml:base` that is a
        // relative reference is resolved against the base URI already in scope, so
        // `xml:base="Z/../Z"` under `.../d/e/` names `.../d/e/Z`. `resolve-uri-022` reads the
        // result back out with `static-base-uri()`.
        let resolved = crate::resolve::resolve_uri(&base, env.base_uri());
        env.set_base_uri(Some(resolved));
    } else if env.base_uri().is_none()
        && let Some(uri) = element.base_uri()
    {
        env.set_base_uri(Some(uri));
    }
    Ok(env)
}

/// The mode reference an absent `mode` attribute stands for, given the `default-mode` in
/// scope — [XSLT 3.0 §6.6.1](https://www.w3.org/TR/xslt-30/#default-mode).
pub(crate) fn default_mode_reference(env: &StaticEnv) -> ModeRef {
    match env.default_mode() {
        Some(name) => ModeRef::Named(name.clone()),
        None => ModeRef::Default,
    }
}

/// The mode key an absent `mode` attribute stands for.
pub(crate) fn default_mode_key(env: &StaticEnv) -> ModeKey {
    match env.default_mode() {
        Some(name) => ModeKey::Named(name.clone()),
        None => ModeKey::Default,
    }
}

/// Splits an attribute on whitespace.
pub(crate) fn tokens(value: &str) -> Vec<&str> {
    value.split_whitespace().collect()
}

/// Resolves a whitespace-separated list of `QName`s.
///
/// # Errors
///
/// `XTSE0280` when a prefix is not bound.
pub(crate) fn resolve_names(value: &str, env: &StaticEnv) -> XsltResult<Vec<ExpandedName>> {
    tokens(value)
        .into_iter()
        .map(|token| env.resolve_element_name(token))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn attributes_are_read_by_local_name() -> XsltResult<()> {
        let tree = Tree::parse_str(r#"<e a="1" xmlns:x="urn:x" x:b="2"/>"#)?;
        let element = tree.document_element().expect("an element");
        assert_eq!(attribute(&element, "a").as_deref(), Some("1"));
        assert_eq!(attribute(&element, "b"), None);
        Ok(())
    }

    #[test]
    fn the_document_element_is_found_from_either_end() -> XsltResult<()> {
        let tree = Tree::parse_str("<a/>")?;
        let root = tree.root();
        let element = tree.document_element().expect("an element");
        assert_eq!(document_element(&root), Some(element.clone()));
        assert_eq!(document_element(&element), Some(element));
        Ok(())
    }

    #[test]
    fn tokens_split_on_whitespace() {
        assert_eq!(tokens(" a  b\nc "), ["a", "b", "c"]);
        assert!(tokens("   ").is_empty());
    }
}
