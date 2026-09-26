# Operations runbook

## Deploy

### Local (single user)

```
cargo install --path crates/kredo
kredo serve          # foreground
kredo start          # foreground, supervised (restart with backoff)
```

For auto-start, register a LaunchAgent/systemd unit running `kredo serve`.

### Server

```
docker compose up -d
```

Set `KREDO_API_KEY` before exposing beyond localhost. For TLS, mount certs and
set `KREDO_TLS_CERT` / `KREDO_TLS_KEY` (compose example in docker-compose.yml).

Behind a load balancer, `/healthz` is the liveness probe and `/readyz` the
readiness probe; both are unauthenticated by design.

## Model lifecycle

```
kredo pull kredo:triage    # embedded, offline, byte-pinned
kredo pull kredo:en        # streams from HF at a pinned commit, sha256-verified
kredo ps                  # what is resident + keep-alive expiry
kredo unload kredo:en      # drop from memory
kredo rm kredo:en          # drop from disk
```

Models load lazily on first request and unload after `KREDO_KEEP_ALIVE`
seconds of inactivity. Pulls are resumable and verified; a digest mismatch
fails the pull, never serves weights.

## Rebuilding a trained model

See `convert/README.md`. The parity gate (ONNX vs PyTorch ≤ 1e-4 over the
eval distribution) must pass before packaging. After packaging, refresh the
digests in `crates/kredo-registry/src/library.rs` and re-run the ignored
golden tests:

```
KREDO_MODELS=... cargo test -p kredo-runner -- --ignored
```

## Monitoring

- `GET /metrics` — Prometheus: request/error counters, latency histograms
  (`kredo_request_duration_seconds`, `kredo_inference_duration_seconds`),
  resident-model gauge, rate-limit rejections.
- Structured logs: `KREDO_LOG_FORMAT=json` gives one JSON event per request
  with `request_id`, `path`, `status`, `elapsed_ms`. Correlate via the
  `x-request-id` response header.

## Capacity

- Each model runs one ONNX Runtime session; inferences are serialized per
  session (`KREDO_MAX_CONCURRENCY` permits per model, default 4; requests wait
  up to 30 s then receive a busy error).
- A `kredo:triage`-sized model (17 MB, BERT-tiny) is single-digit
  milliseconds per decision on a laptop core. The pairwise `kredo:en` model
  runs one forward pass per option — budget accordingly.

## Incident playbook

| Symptom | Check | Action |
|---|---|---|
| 404 on decide | `kredo list` | `kredo pull <model>` |
| Busy errors / slow | `/metrics` inference histogram | raise `KREDO_MAX_CONCURRENCY` or scale replicas |
| 429s | `kredo_rate_limited_total` | raise `KREDO_RATE_LIMIT` or fix client |
| Daemon won't start | `~/.kredo/kredo.pid` | `kredo stop`, remove stale pid, restart |
| Pull digest mismatch | upstream re-release | verify upstream, update pinned digest |
