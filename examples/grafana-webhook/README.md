# Grafana → kredo alert triage

Decide what to do with system alerts using a small decision model instead
of an LLM: page the on-call or not, which subsystem is responsible, whether
customers are impacted. Fully offline, ~10 ms per decision, deterministic
enough to put in a runbook.

```
Grafana alert ──webhook──▶ relay.py ──/v1/systemone──▶ kredo ──▶ page / route / suppress
```

## Try it in 60 seconds

```sh
kredo serve &                                  # or: kredo start
kredo pull kredo:en                            # zero-shot, one-time (~250 MB)
python3 relay.py --selftest                    # decide on a synthetic alert
python3 relay.py                               # listen on :9911
```

Point a Grafana contact point (Webhook) at `http://relay-host:9911/grafana`.
Every alert comes back triaged:

```json
{
  "alert": "HighCPUUsage",
  "page": true,
  "page_p": 0.91,
  "customer_impact_p": 0.62,
  "likely_domain": "database",
  "domain_p": 0.44,
  "model": "kredo:en",
  "elapsed_ms": 14.2
}
```

## Honest scope (measured, not vibes)

- **Domain routing works zero-shot**: "database connections exhausted" →
  `database`, cert expiry → `third-party dependency`, CPU → `storage/disk`.
- **Page/no-page does not** — zero-shot NLI on alert-shaped text is noisy in
  both directions (a 30-day cert expiry can look more urgent than a
  saturated connection pool). The zero-shot `page_worthy` score is a prior,
  not a decision. For paging, train a small severity model on your closed
  incidents (the convert pipeline) and run `kredo calibrate` — that is the
  production path, and it is exactly why kredo ships verification,
  calibration and shadow mode: a model that pages humans must earn it.

## Knobs (env)

| Variable | Default | Meaning |
|---|---|---|
| `KREDO_URL` | `http://127.0.0.1:21435` | kredo daemon |
| `KREDO_API_KEY` | — | bearer key when the daemon requires one |
| `KREDO_MODEL` | `kredo:en` | any tag; a trained `kredo:alerts` drops in unchanged |
| `PAGE_THRESHOLD` | `0.8` | P(page-worthy) at or above this pages |
| `PORT` | `9911` | relay listen port |

## Production shape

- The questions in `relay.py` are the whole "prompt" — versioned in git,
  reviewed like config, not hidden in a prompt string.
- Calibrate on your closed incidents: dump `title/labels/resolved-by` rows
  to JSONL and run `kredo calibrate` so the page/snooze threshold is an
  honest probability, not a vibe.
- The Prometheus series (`kredo_decisions_by_model_total`,
  `kredo_shadow_agree_total`) go on the same Grafana dashboard as the
  alerts: shadow-evaluate a retrained severity model against live alerts
  (`kredo shadow start kredo:alerts-v2`) before it ever pages anyone.
