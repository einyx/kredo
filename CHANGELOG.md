# Changelog

All notable changes to kredo are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning follows
[SemVer](https://semver.org/).

## [0.1.0] - unreleased

### Added
- Daemon with TypeSafe-compatible `/v1/systemone`, `/v1/decisions`,
  `/v1/models` and native `/api/*` endpoints (decide, pull with NDJSON
  progress, tags, show, ps, stop, delete).
- CLI: `serve`, `start` (supervised), `stop`, `status`, `run`, `pull`,
  `list`, `ps`, `show`, `rm`, `cp`, `unload`, `create`, `verify`.
- ONNX Runtime inference: zero-shot pairwise (`kredo:en`,
  `kredo:multilingual`, `nli`) and native multi-head (trained models).
- Router model with script/language detection (`kredo` → `kredo:en` or
  `kredo:multilingual`).
- Registry: library manifests with pinned Hugging Face revisions and
  sha256 digests, content-addressed blob store, resumable pulls,
  embedded offline models.
- Verification: `kredo verify` re-runs packaging-time golden fixtures;
  `KREDO_REQUIRE_VERIFIED=1` refuses unverified models; manifest
  provenance (dataset, seed, pipeline, per-head eval metrics).
- Production hardening: graceful shutdown, pid file, health/readiness
  probes, Prometheus metrics, request IDs, structured JSON logs,
  per-model concurrency pools, request timeouts, body limits,
  constant-time bearer auth, token-bucket rate limiting, rustls TLS,
  CORS toggle.
- convert/ pipeline: dataset generation (synthetic triage; real
  support-ticket corpus), multi-head training, ONNX export, hard parity
  gate (ONNX vs PyTorch ≤ 1e-4), packaging with digests.
- Docker image, docker-compose, GitHub Actions CI.
