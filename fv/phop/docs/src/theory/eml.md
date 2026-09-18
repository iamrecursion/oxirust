# The EML operator

The EML operator is

```text
eml(x, y) = exp(x) − ln(y).
```

Odrzywołek (2026, arXiv:2603.21852) proves that, together with the constant `1`, this single
binary primitive is *functionally complete* for the elementary functions: every polynomial,
rational, exponential, logarithmic, trigonometric, and hyperbolic function — and arbitrary
compositions — can be written as a finite binary tree of `eml`. Just as NAND is functionally
complete for Boolean logic, EML is functionally complete for continuous elementary mathematics.

This collapses the grammar of expressions to the single production

```text
S → 1 | eml(S, S).
```

phop delegates the EML abstract syntax tree, evaluation, canonicalization, and LaTeX rendering
to the [`oxieml`](https://github.com/cool-japan/oxieml) crate.
