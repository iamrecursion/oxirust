# Security Policy

## Supported Versions

OxiNum follows a rolling `0.x` release line. Only the **latest published
`0.x.y` release** on crates.io (covering `oxinum` and its sub-crates —
`oxinum-core`, `oxinum-int`, `oxinum-float`, `oxinum-rational`,
`oxinum-complex`) is supported with security fixes. Please upgrade to the
latest release before reporting an issue if possible.

## Reporting a Vulnerability

If you believe you have found a security vulnerability, **please do not
file a public GitHub issue** — public issues are indexed and searchable,
which could expose users before a fix ships. Instead, report privately by
emailing:

**info@kitasan.io**

Please include a description of the issue and its impact, reproduction
steps (a minimal code sample is ideal), the affected crate(s) and
version(s), and any suggested remediation.

Reports are triaged privately. We will acknowledge receipt, investigate,
and work with you on a coordinated disclosure timeline once a fix is
available. Credit is given to reporters who wish to be acknowledged.

## Scope Note

OxiNum is a general-purpose numerics library and is explicitly **not
constant-time** — it must never be used for secret-dependent (cryptographic)
computation, and timing side-channel reports are out of scope. Panics,
memory-safety issues, and incorrect results on untrusted input are in scope.

## Maintainer

COOLJAPAN OU (Team Kitasan)
