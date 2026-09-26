# Security notes

## Threat model

kredo executes third-party neural-network graphs locally and serves an HTTP
API. The security goals are: only verified weights run, the API exposes no
more than the decision surface, and network exposure is explicit.

## Supply chain (weights)

- Built-in third-party models pin an exact upstream commit **and** sha256
  per file; a mismatch aborts the pull. Digests live in
  `crates/kredo-registry/src/library.rs` and are covered by unit tests
  (`builds_hf_urls` asserts pinning).
- Trained models ship embedded in the binary (`embedded.rs`), so the
  `kredo:triage` graph is the exact artifact that passed the parity gate —
  there is no runtime fetch at all.
- Manifest schema versioning: manifests carry `schema: 1`; unknown future
  schemas are rejected with an upgrade hint.
- ONNX graphs are code. Only pull models from authors you trust; kredo does
  not sandbox graph execution.

## API

- `KREDO_API_KEY` enables bearer auth, compared in constant time (`subtle`).
  The key (or client IP) also keys the rate limiter (`KREDO_RATE_LIMIT`,
  req/min; 429 with `retry-after`).
- Requests are size-capped (`KREDO_BODY_LIMIT`, default 1 MiB) and time-bound
  (`KREDO_REQUEST_TIMEOUT`, default 30 s → 503).
- Binding a non-loopback address without TLS/auth logs a warning at startup.

## Transport

- `KREDO_TLS_CERT` + `KREDO_TLS_KEY` enable HTTPS (rustls). For networked
  deployments, terminate TLS here or at the load balancer — never serve
  plain HTTP on a public interface.

## Data

- Decision states are processed in memory and never persisted; the blob
  store contains only model artifacts. Logs record metadata (paths, ids,
  timings), never state contents.
