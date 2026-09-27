# kredo

**Run decision models you can verify.**

A decision model reads a *state* (a message, an email, a ticket, any JSON)
plus typed questions (`choice`, `score`, `noul`) and returns calibrated
probabilities in a single forward pass, in milliseconds. It never generates
text.

kredo pulls these models by name, **verifies them on your machine against
packaging-time golden fixtures**, and serves them from a local daemon that
speaks the TypeSafe-compatible `/v1/systemone` wire format — so existing
clients work by changing one environment variable.

Inspired by [Ollaya](https://github.com/ollaya-dev/ollaya) ("Ollama for
decision models"); kredo's focus is the verification story: every trained
model ships with provenance, a parity gate, and a fixture you can re-run.

```
kredo pull kredo:triage
kredo run kredo:triage "I was charged twice for my subscription this month and want a refund."
```

```
intent            a refund      ████████████████ 0.99
is_urgent         ░░░░░░░░░░░░░ 0.00
frustration       1.32 / 3      ███████░░░░░░ 0.44
refund_requested  ██████████████ 0.99
churn_risk        ░░░░░░░░░░░░░ 0.01
```

## Why "verify"

Small classifiers are easy to get subtly wrong: a re-exported ONNX graph, a
tokenizer mismatch, or a re-released upstream checkpoint can silently change
decisions. kredo closes that loop:

| Step | Gate |
|---|---|
| Training | `convert/` pipeline, seeded and reproducible |
| Export | **parity gate** — ONNX must match PyTorch within 1e-4 on the eval distribution |
| Packaging | fixture digest + sha256 of every file recorded in the manifest |
| Install | `kredo verify <model>` re-runs the fixture cases locally; exit code is machine-checkable |
| Serving | `KREDO_REQUIRE_VERIFIED=1` refuses models that never passed |

Every trained model manifest carries `provenance` (dataset, seed, pipeline
version, per-head eval metrics) and `verification` (fixture digest,
tolerance, case count).

## Models

| Model | What it is | Eval |
|---|---|---|
| `kredo:triage` | Trained multi-head triage model (BERT-tiny, 17 MB, embedded — works offline) | intent/urgent/refund acc 1.00, churn 0.84 |
| `kredo:support` | Real support-ticket routing (queue + urgency), trained on [customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | queue acc 0.50, urgency 0.71 |
| `kredo:en` | English zero-shot (DistilBERT MNLI cross-encoder, pinned HF commit) | zero-shot |
| `kredo:multilingual` | 100+ languages (mDeBERTa-v3 XNLI cross-encoder, pinned) | zero-shot |
| `kredo` | Router: picks `kredo:en` or `kredo:multilingual` by script/language | — |

Eval numbers live in the manifest (`kredo show kredo:triage`) and are
re-checked by `kredo verify`. The support model's queue accuracy is honest
about real data being hard: 10 queues, 0.50 accuracy versus a ~0.30 majority
baseline — this is what real corpus training looks like at 17 MB, and we
publish the number rather than hiding it.

Third-party weights are never re-hosted: they stream from the author's
Hugging Face repository at a pinned commit, verified by sha256.

## Install

From source:

```
cargo install --path crates/kredo
```

Prebuilt binaries and a `curl | sh` installer are attached to GitHub
Releases. Docker: `docker run -p 21435:21435 ghcr.io/einyx/kredo`.

## Playground

Start the daemon and open the embedded UI — no frontend build, no network
beyond the model files, fonts and icons ship inside the binary:

```sh
kredo serve            # then open http://127.0.0.1:21435/ui
# or
kredo ui               # same thing, prints the URL
```

Pick a model, inspect its verification seal and provenance, edit structured
questions (zero-shot models accept arbitrary ones; trained models show their
built-in head set read-only), and run decisions with animated calibrated
probability bars.

## For agents

Expose kredo to Claude Code, Claude Desktop, opencode and other MCP clients
in one line:

```sh
npx @einyx/kredo-mcp install   # registers the kredo MCP server everywhere
```

The package is published to GitHub Packages, so if you've never used the
`@einyx` scope, point npx at it first (any GitHub account can read; a
classic PAT with `read:packages` works):

```sh
# ~/.npmrc
@einyx:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=<YOUR_GITHUB_TOKEN>
```

Tools: `kredo_decide`, `kredo_list_models`, `kredo_describe_model` — tool
descriptions surface verification status, so agents prefer verified models.

## The API

TypeSafe-compatible: `POST /v1/systemone`, `POST /v1/decisions`,
`GET /v1/models`. Native: `POST /api/decide` (routing + timings),
`POST /api/pull` (NDJSON progress), `/api/tags`, `/api/show`, `/api/ps`,
`/api/stop`, `/api/delete`, plus `/healthz`, `/readyz`, `/metrics`
(Prometheus). Full reference: [docs/api.md](docs/api.md).

```
curl -s localhost:21435/v1/systemone -d '{"state":"I need a refund today or I cancel"}'
```

## Production

Graceful shutdown, supervised restart, health/readiness probes, Prometheus
metrics, request IDs and JSON logs, per-model concurrency pools, request
timeouts and body limits, constant-time bearer auth, rate limiting, rustls
TLS. Deployment: [docs/runbook.md](docs/runbook.md). Threat model:
[docs/security.md](docs/security.md).

Configuration is via environment variables: `KREDO_HOST`, `KREDO_MODELS`,
`KREDO_KEEP_ALIVE`, `KREDO_DEVICE`, `KREDO_API_KEY`, `KREDO_REQUIRE_VERIFIED`
and others — see [docs/api.md](docs/api.md).

## Building your own model

The `convert/` pipeline takes a question set and a dataset to a verified,
embedded or pinned artifact:

```sh
cd convert
uv run python -m kredo_convert.support          # dataset
uv run python -m kredo_convert.train --data data/support.jsonl --set support --out out/support
uv run python -m kredo_convert.export --model out/support --out out/support.onnx
uv run python -m kredo_convert.parity --model out/support --onnx out/support.onnx --set support
uv run python -m kredo_convert.package --set support
```

Details: [convert/README.md](convert/README.md).

## Repository

| Path | What |
|---|---|
| `crates/kredo` | The binary: CLI, daemon lifecycle, `verify` |
| `crates/kredo-server` | HTTP API, scheduler, verification enforcement |
| `crates/kredo-api` | Wire types; contract is [docs/api.md](docs/api.md) |
| `crates/kredo-registry` | Manifests, blob store, resumable verified pulls |
| `crates/kredo-decision` | Question schema, calibration, answers |
| `crates/kredo-runner` | ONNX Runtime engines + verification runner |
| `crates/kredo-lang` | Script/language detection for routers |
| `convert/` | Train → export → parity → package pipeline |

## Development

```
cargo test --workspace
cargo build --release -p kredo --features kredo-runner/cuda   # CUDA build
```

Golden-fixture tests require pulling the model and run with
`cargo test -p kredo-runner -- --ignored`.

## Status & roadmap

kredo is early (0.1). Roadmap: CUDA exercised on hardware, GGUF decoder
models, more real-data models with published evals, calibration
(temperature scaling) per head.

## License

Apache-2.0. Third-party models keep their own licenses; see
[docs/security.md](docs/security.md) for supply-chain guarantees.
