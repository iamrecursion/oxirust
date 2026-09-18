# The autograd forward pass

`scirs2-autograd` is a define-by-run graph: you build the computation inside `ag::run` /
`env.run`, and tensors are graph nodes bound to the context.

phop builds an EML tree's forward pass from the primitives `exp`, `ln`, `clip`, `sub`, with
**guarded** evaluation so that `exp` cannot overflow (argument clamped) and `ln` is never
applied to a non-positive value (argument clamped to a small positive epsilon). Data — feature
columns, a ones-column, and targets — enters through **fed placeholders** rather than constant
nodes, which keeps the graph free of const-generating ops and is the efficient idiomatic path.

Constant leaves and Gumbel logits are kept as true scalars (`shape []`) so the library's
scalar-broadcast paths are used. Training uses Adam from `scirs2-autograd`.
