# Security Policy

## Supported versions

Only the latest published `0.1.x` release receives fixes. `rankfusion` is
pre-1.0: there are no long-term support branches yet.

## Reporting a vulnerability

Report privately through GitHub's
[security advisories](https://github.com/Lsh0x/rankfusion/security/advisories/new)
form. Please do **not** open a public issue for an unfixed vulnerability.

Expect an acknowledgement within 7 days and an assessment within 30. Fixed
issues are published as a GitHub Security Advisory, with a RustSec advisory
filed when the crate is affected as published on crates.io.

## Threat model

`rankfusion` is a pure computation library: no I/O, no network, no
deserialization of untrusted input by default, and `#![forbid(unsafe_code)]`
crate-wide. Its inputs are candidate lists that the calling application has
already retrieved.

In scope:

- Panics, unbounded memory growth, or non-termination reachable from
  library inputs (candidate lists, weights, scores, `k` values).
- Incorrect results that a caller could rely on for an access-control or
  filtering decision.
- Unsoundness in any dependency reachable from a default or optional feature.

Out of scope:

- The relevance quality of a ranking, or a fusion configuration producing
  results you consider wrong.
- NaN or infinite scores propagating to fused scores. This is the documented
  garbage-in/garbage-out policy (see `src/core.rs`): ordering stays total and
  deterministic, and validating inputs is the caller's responsibility.
- Attacks requiring the caller to deserialize attacker-controlled data into
  the crate's types via the optional `serde` feature — validate before
  deserializing.
