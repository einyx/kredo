#!/usr/bin/env python3
"""Grafana -> kredo alert-triage relay.

Receives Grafana alert webhooks, asks kredo decision questions about the
alert (page-worthiness, domain, severity) and prints/forwards the enriched
result. Zero external dependencies (stdlib only).

Configure in Grafana: Alerting -> Contact points -> Webhook
    URL: http://relay-host:9911/grafana

Run:
    python3 relay.py                      # expects kredo on 127.0.0.1:21435
    KREDO_URL=... PAGE_THRESHOLD=0.8 python3 relay.py

The relay decides with a zero-shot model (kredo:en) using custom questions,
so it works out of the box; swap MODEL to a trained severity model once you
have one (e.g. kredo:alerts) — the contract does not change.
"""

import json
import os
import sys
import urllib.request

KREDO_URL = os.environ.get("KREDO_URL", "http://127.0.0.1:21435")
KREDO_KEY = os.environ.get("KREDO_API_KEY", "")
MODEL = os.environ.get("KREDO_MODEL", "kredo:en")
LISTEN_PORT = int(os.environ.get("PORT", "9911"))
# P(should page) at or above this pages the on-call.
PAGE_THRESHOLD = float(os.environ.get("PAGE_THRESHOLD", "0.8"))

QUESTIONS = [
    {"id": "page_worthy", "type": "noul",
     "prompt": "This alert requires waking up a human right now"},
    {"id": "domain", "type": "choice",
     "prompt": "Which system is most likely responsible for this alert?",
     "options": ["database", "network", "application code",
                 "infrastructure / capacity", "third-party dependency",
                 "storage / disk"]},
    {"id": "customer_impact", "type": "noul",
     "prompt": "This alert means customers are actively affected"},
]


def kredo_decide(state: str) -> dict:
    body = json.dumps({
        "model": MODEL,
        "state": state,
        "questions": QUESTIONS,
    }).encode()
    req = urllib.request.Request(
        f"{KREDO_URL}/v1/systemone", data=body,
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {KREDO_KEY}"} if KREDO_KEY else {})})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def alert_to_state(alert: dict) -> str:
    """Flatten a Grafana alert into the text a decision model reads."""
    labels = " ".join(f"{k}={v}" for k, v in (alert.get("labels") or {}).items())
    title = alert.get("title") or alert.get("labels", {}).get("alertname", "alert")
    parts = [f"Alert: {title}"]
    if labels:
        parts.append(f"Labels: {labels}")
    if alert.get("message") or alert.get("annotations", {}).get("summary"):
        parts.append(f"Summary: {alert.get('message') or alert['annotations']['summary']}")
    status = alert.get("status", "firing")
    parts.append(f"Status: {status}")
    return ". ".join(parts)


def triage(alert: dict) -> dict:
    resp = kredo_decide(alert_to_state(alert))
    answers = {a["id"]: a for a in resp.get("answers", [])}
    page_p = answers.get("page_worthy", {}).get("p", 0.0)
    impact = answers.get("customer_impact", {}).get("p", 0.0)
    domain_top = max(answers.get("domain", {}).get("probabilities", [{"label": "?", "p": 0}]),
                     key=lambda x: x["p"])
    return {
        "alert": alert.get("title", alert.get("labels", {}).get("alertname", "?")),
        "page": page_p >= PAGE_THRESHOLD,
        "page_p": round(page_p, 3),
        "customer_impact_p": round(impact, 3),
        "likely_domain": domain_top["label"],
        "domain_p": round(domain_top["p"], 3),
        "model": resp.get("model"),
        "elapsed_ms": resp.get("elapsed_ms"),
    }


def handler(payload: dict) -> dict:
    """Grafana v9+ sends {"alerts": [...]} (v8 sends one object)."""
    alerts = payload.get("alerts") or [payload]
    return {"triage": [triage(a) for a in alerts]}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print(json.dumps(triage({
            "title": "HighCPUUsage",
            "status": "firing",
            "labels": {"severity": "critical", "host": "db-01", "team": "data"},
            "annotations": {"summary": "CPU above 95% for 10 minutes"},
        }), indent=2))
        sys.exit(0)

    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Relay(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                out = handler(payload)
                code = 200
            except Exception as e:  # noqa: BLE001 — relay must never 500-loop Grafana
                out, code = {"error": str(e)}, 502
            body = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # quiet
            pass

    print(f"grafana-kredo relay on :{LISTEN_PORT} -> {KREDO_URL} ({MODEL})", flush=True)
    HTTPServer(("0.0.0.0", LISTEN_PORT), Relay).serve_forever()
