# The kredo API

Default endpoint: `http://127.0.0.1:21435` (configurable with `KREDO_HOST`).

## TypeSafe-compatible endpoints

### `POST /v1/systemone`

```json
{
  "state": "I was charged twice this month and want a refund.",
  "questions": [
    { "id": "intent", "type": "choice", "prompt": "What does the user want?",
      "options": ["a refund", "a bug report", "praise"] },
    { "id": "frustration", "type": "score", "prompt": "How frustrated?",
      "min": 0, "max": 3 },
    { "id": "is_urgent", "type": "noul", "prompt": "This message is urgent." }
  ],
  "model": "kredo:en"
}
```

`questions` may be omitted to use the model's built-in set. `state` may be a
string or arbitrary JSON. Response:

```json
{
  "model": "kredo",
  "elapsed_ms": 12.4,
  "answers": [
    { "id": "intent", "type": "choice",
      "probabilities": [ {"label": "a refund", "p": 0.99}, ... ] },
    { "id": "frustration", "type": "score",
      "score": {"value": 0.4, "normalized": 0.13, "min": 0, "max": 3} },
    { "id": "is_urgent", "type": "noul", "p": 0.01 }
  ]
}
```

### `POST /v1/decisions`

Batch form: `{"model": "...", "states": [...], "questions": [...]}` → array
of systemone responses.

### `GET /v1/models`

Array of `{"name", "model", "modified_at", "size", "questions"}`.

## Native endpoints

- `POST /api/decide` — `{model, state, questions}` → answers plus
  `{routing: {resolved, script, language, route_ms}, load_ms, infer_ms}`.
- `POST /api/pull` — `{model}`; streams NDJSON progress:
  `{"status": "downloading onnx/model.onnx", "total": 267958787, "completed": 12345}`.
- `GET /api/tags` — pulled models.
- `POST /api/show` — `{model}` → manifest details and question set.
- `GET /api/ps` — resident models and keep-alive expiry.
- `POST /api/stop` — `{model}` → unload from memory.
- `POST /api/delete` — `{model}` → unload and remove from disk.

## Configuration

| Variable | Default | What |
|---|---|---|
| `KREDO_HOST` | `127.0.0.1:21435` | Daemon bind address (client base URL) |
| `KREDO_MODELS` | `~/.kredo/models` | Model store root |
| `KREDO_KEEP_ALIVE` | `300` | Seconds a loaded model stays resident |
| `KREDO_DEVICE` | `cpu` | `cpu` or `cuda` |
| `KREDO_API_KEY` | unset | Require `Authorization: Bearer <key>` (constant-time check) |
| `KREDO_MAX_CONCURRENCY` | `4` | Max concurrent inferences per model |
| `KREDO_REQUEST_TIMEOUT` | `30` | Per-request timeout in seconds (→ 503) |
| `KREDO_RATE_LIMIT` | `0` (off) | Requests per minute per client (→ 429) |
| `KREDO_TLS_CERT` / `KREDO_TLS_KEY` | unset | Serve HTTPS (rustls) |
| `KREDO_LOG_FORMAT` | `text` | `text` or `json` |
| `KREDO_CORS` | `0` | `1` allows cross-origin requests |
| `KREDO_BODY_LIMIT` | `1048576` | Max request body bytes |
| `KREDO_PID_FILE` | `~/.kredo/kredo.pid` | Daemon pid file |
| `KREDO_DEBUG_LOGITS` | unset | Print raw logits per pair to stderr |

## Operations

Endpoints for orchestration (all unauthenticated by design):

- `GET /healthz` — liveness (`ok`).
- `GET /readyz` — `{"ready": true, "models": N}`.
- `GET /version` — `{"version": "..."}`.
- `GET /metrics` — Prometheus text exposition.

See docs/runbook.md for deployment and docs/security.md for the threat
model.
