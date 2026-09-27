# Changelog

All notable changes to kredo are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning follows
[SemVer](https://semver.org/).

## [0.3.0] - 2026-09-27

### Added
- **Shadow mode**: `kredo shadow start/stop/status/report` — evaluate a
  candidate model against live traffic without serving it; agreement
  report with per-question breakdown, latency comparison and disagreement
  samples; `--min-agreement` for CI gates. `kredo promote/demote` switch
  the daemon default (tags stay immutable, routing is config).
- **Per-head temperature calibration**: `Head.temperature` /
  pairwise temperature in the manifest; `kredo calibrate --eval set.jsonl`
  grid-searches T per head on labeled data, re-records the verification
  fixture. Argmax (accuracy) is preserved — only confidence becomes honest.
- **Per-model Prometheus metrics**: `kredo_decisions_by_model_total`,
  `kredo_model_inference_duration_seconds{model=…}`, shadow agreement
  counters and latency histogram.
- **Grafana alert-triage example** (`examples/grafana-webhook`): webhook
  relay that decides page/no-page, subsystem and customer impact on alerts.

### Fixed
- **`kredo:en` read the wrong NLI logit**: the manifest pinned
  `{0: entailment, …}` but the upstream model config is
  `{0: contradiction, 1: entailment, 2: neutral}` — urgency/entailment
  scores were near-zero for every input. Re-pull `kredo:en` to pick up the
  corrected manifest.

## [0.2.3] - 2026-09-27

### Fixed
- Docker image did not start (`GLIBC_2.39 not found`): build stage
  `rust:1-slim` is now trixie-based, so the runtime base moved from
  `debian:bookworm-slim` to `debian:trixie-slim` to match.

### Changed
- `kredo-mcp` reads its version from `package.json` instead of a hardcoded
  constant, so release bumps cannot drift.

## [0.2.2] - 2026-09-27

### Changed
- npm package moved to GitHub Packages as `@einyx/kredo-mcp` (publishes with
  the built-in `GITHUB_TOKEN`; see README for one-time client setup).
- Docker build installs `g++` (release image link needs `libstdc++`).

## [0.2.1] - 2026-09-27

### Fixed
- npm release step skipped cleanly when `NPM_TOKEN` is not configured.
- Docker build installs `pkg-config`/`libssl-dev` for rustls.
- Better error (and UI note) when arbitrary questions are sent to a trained
  multi-head model: the built-in question set is listed in the message, and
  the playground shows trained heads read-only.

## [0.2.0] - 2026-09-27

### Added
- `kredo ui`: embedded decision playground served at `/ui` — model picker
  with verification seals, structured question editor, animated calibrated
  probability bars, provenance panel. Fully offline (fonts, icons and logic
  ship inside the binary).
- `kredo mcp`: Model Context Protocol stdio server exposing
  `kredo_decide`, `kredo_list_models` and `kredo_describe_model`; tool
  descriptions surface verification status.
- `kredo-mcp` npm package: `npx @einyx/kredo-mcp install` registers the kredo MCP
  server with Claude Code, Claude Desktop and opencode; `npx @einyx/kredo-mcp`
  runs the server, downloading the release binary on first use.

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
