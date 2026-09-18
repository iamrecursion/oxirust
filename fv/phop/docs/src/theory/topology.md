# Differentiable topology

Conventional symbolic regression searches a *heterogeneous* operator set
`{+, −, ×, ÷, exp, ln, sin, …}`. Because each node may be a different operator, the choice of
"which operator goes here" is discrete, and structure search is evolutionary or sequence-based —
no gradient flows through the structure itself.

EML removes this obstacle. Every internal node is the **same** operator, so the only remaining
categorical choice is *which source feeds each leaf*: one of the input variables or a learnable
constant. phop relaxes that choice with the **Gumbel-Softmax** estimator and anneals the
temperature from exploratory (near-uniform) to near-discrete, so Adam can descend through the
tree structure jointly with the numeric parameters. At inference the leaves are hardened by
`argmax` into a concrete tree.

This is the phop contribution: differentiable learning of *structure and parameters together*
over a population of EML trees — feasible precisely because EML is operator-homogeneous.

## Learning the tree's depth and shape

Leaf selection relaxes *which source feeds a fixed-depth leaf*. phop can also make the tree's
**depth and shape** differentiable (`gated.rs`, `discover_gated`). Over a maximal complete tree,
every node additionally holds a soft "leaf" value, and each *non-root* internal node carries an
**expand/terminate gate** `σ(z_n)`:

```text
val(n) = leaf(n) + σ(z_n) · ( eml(val(2n+1), val(2n+2)) − leaf(n) ).
```

With `σ → 0` the node *terminates* into its own leaf and its subtree is pruned; with `σ → 1` it
*expands* into `eml(L, R)`. A complexity penalty `λ·Σ σ(z_n)` pressures the search toward shallow
trees, and each gate is hardened by a `0.5` threshold to read off a concrete tree of learned depth.

A subtlety unique to EML: there is **no identity/skip element** — no constant `b` makes
`exp(a) − ln(b) = a` for all `a` — so the DARTS trick of zeroing an edge to prune cannot be ported.
A subtree can only be collapsed by *terminating its node into a leaf*, which is exactly what the
gate does. In practice the root is kept always-expanded and deeper gates start *terminated* (a
depth curriculum), so the optimizer expands a node only when the data rewards it — avoiding the
credit-assignment collapse that a naive soft forest over guarded `exp`/`ln` falls into.
