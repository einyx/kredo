//! `kredo mcp`: Model Context Protocol stdio server.
//!
//! Exposes kredo decision models as tools to MCP clients (Claude Code,
//! Claude Desktop, Cursor, …). Speaks newline-delimited JSON-RPC 2.0;
//! requests are proxied to the kredo daemon, so auth, scheduling and
//! verification enforcement stay centralized.

use serde_json::{json, Value};
use std::sync::LazyLock;

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "kredo";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

static TOOLS: LazyLock<Value> = LazyLock::new(|| {
    json!({
        "tools": [
            {
                "name": "kredo_decide",
                "description": "Ask a kredo decision model to answer typed questions (choice/score/noul) about a state, returning calibrated probabilities in one forward pass. Prefer models whose verification status is 'verified'.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "description": "Model name, e.g. kredo:triage"},
                        "state": {"type": "string", "description": "The text/JSON state to decide on"},
                        "questions": {
                            "type": "array",
                            "description": "Optional typed questions; omit to use the model's built-in set",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string", "enum": ["choice", "score", "noul"]},
                                    "prompt": {"type": "string"},
                                    "options": {"type": "array", "items": {"type": "string"}},
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                },
                                "required": ["id", "type", "prompt"]
                            }
                        }
                    },
                    "required": ["model", "state"]
                }
            },
            {
                "name": "kredo_list_models",
                "description": "List pulled kredo decision models with verification status and eval metrics. Prefer 'verified' models.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "kredo_describe_model",
                "description": "Describe a kredo decision model: its built-in question set, training provenance (dataset, metrics) and verification record.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"]
                }
            }
        ]
    })
});

/// Serve MCP over stdio until EOF.
pub async fn run() -> anyhow::Result<()> {
    let base = super::base_url();
    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);
    let mut line = String::new();
    loop {
        line.clear();
        let n = tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut line).await?;
        if n == 0 {
            return Ok(());
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(id) = req.get("id").cloned() else {
            continue; // notification: no response
        };
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
            })),
            "tools/list" => Ok(TOOLS.clone()),
            "tools/call" => call_tool(&base, &params).await,
            other => Err(format!("unknown method {other}")),
        };
        let response = match result {
            Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}),
            Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32603, "message": e}}),
        };
        println!("{response}");
    }
}

async fn call_tool(base: &str, params: &Value) -> Result<Value, String> {
    let name = params
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or("missing tool name")?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let text = match name {
        "kredo_decide" => {
            let model = args
                .get("model")
                .and_then(|m| m.as_str())
                .ok_or("missing model")?
                .to_string();
            let state = args
                .get("state")
                .and_then(|s| s.as_str())
                .ok_or("missing state")?
                .to_string();
            let questions: Vec<kredo_api::Question> = args
                .get("questions")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or_default();
            let resp = super::client::decide(base, &model, &state, Some(questions))
                .await
                .map_err(|e| e.to_string())?;
            format_answers(&resp)
        }
        "kredo_list_models" => {
            let tags = super::client::tags(base).await.map_err(|e| e.to_string())?;
            let mut lines = vec!["model | verified | metrics".to_string()];
            for m in tags.models {
                let show = super::client::show(base, &m.model).await.ok();
                let (verified, metrics) = match &show {
                    Some(s) => {
                        let v = s
                            .details
                            .get("verified")
                            .and_then(|b| b.as_bool())
                            .unwrap_or(false);
                        let ms = s
                            .provenance
                            .as_ref()
                            .and_then(|p| p.get("metrics"))
                            .map(|m| m.to_string())
                            .unwrap_or_default();
                        (v, ms)
                    }
                    None => (false, String::new()),
                };
                lines.push(format!(
                    "{} | {} | {}",
                    m.model,
                    if verified { "verified" } else { "unverified" },
                    metrics
                ));
            }
            lines.join("\n")
        }
        "kredo_describe_model" => {
            let model = args
                .get("model")
                .and_then(|m| m.as_str())
                .ok_or("missing model")?
                .to_string();
            let show = super::client::show(base, &model)
                .await
                .map_err(|e| e.to_string())?;
            serde_json::to_string_pretty(&show).map_err(|e| e.to_string())?
        }
        other => return Err(format!("unknown tool {other}")),
    };

    Ok(json!({
        "content": [{"type": "text", "text": text}]
    }))
}

fn format_answers(resp: &kredo_api::DecideResponse) -> String {
    let mut out = Vec::new();
    for a in &resp.answers {
        match a.kind {
            kredo_api::QuestionKind::Choice => {
                if let Some(top) = a.top() {
                    let dist = a
                        .probabilities
                        .iter()
                        .map(|p| format!("{} {:.3}", p.label, p.p))
                        .collect::<Vec<_>>()
                        .join("; ");
                    out.push(format!("{}: {} ({dist})", a.id, top.label));
                }
            }
            kredo_api::QuestionKind::Score => {
                if let Some(s) = &a.score {
                    out.push(format!(
                        "{}: {:.2} / {:.0} (normalized {:.2})",
                        a.id, s.value, s.max, s.normalized
                    ));
                }
            }
            kredo_api::QuestionKind::Noul => {
                if let Some(p) = a.p {
                    out.push(format!("{}: {:.3}", a.id, p));
                }
            }
        }
    }
    let routing = &resp.routing;
    out.push(format!(
        "[model {} resolved in {:.1} ms]",
        routing.resolved, resp.infer_ms
    ));
    out.join("\n")
}
