//! Thin HTTP client for the kredo daemon.

use anyhow::Result;
use kredo_api::{
    DecideRequest, DecideResponse, PsResponse, PullProgress, PullRequest, Question, ShowRequest,
    ShowResponse, SystemOneRequest, SystemOneResponse, TagsResponse,
};

fn url(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

fn http() -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(key) = std::env::var("KREDO_API_KEY") {
        if !key.is_empty() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")) {
                headers.insert(reqwest::header::AUTHORIZATION, v);
            }
        }
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .unwrap_or_default()
}

pub async fn ping(base: &str) -> Result<()> {
    http()
        .get(url(base, "/"))
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn version(base: &str) -> Result<serde_json::Value> {
    Ok(http()
        .get(url(base, "/version"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

pub async fn decide(
    base: &str,
    model: &str,
    state: &str,
    questions: Option<Vec<Question>>,
) -> Result<DecideResponse> {
    let body = DecideRequest {
        model: model.into(),
        state: serde_json::Value::String(state.into()),
        questions: questions.unwrap_or_default(),
    };
    let resp = http()
        .post(url(base, "/api/decide"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}

/// Run via the TypeSafe-compatible endpoint (used when questions are custom).
#[allow(dead_code)]
pub async fn systemone(
    base: &str,
    model: Option<&str>,
    state: &str,
    questions: Vec<Question>,
) -> Result<SystemOneResponse> {
    let body = SystemOneRequest {
        state: serde_json::Value::String(state.into()),
        questions,
        model: model.map(String::from),
    };
    let resp = http()
        .post(url(base, "/v1/systemone"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}

pub async fn pull(
    base: &str,
    model: &str,
    on_progress: &mut dyn FnMut(PullProgress),
) -> Result<()> {
    let body = PullRequest {
        model: model.into(),
        insecure: false,
    };
    let resp = http()
        .post(url(base, "/api/pull"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    let mut buf = Vec::new();
    let mut last_status = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            if let Ok(p) = serde_json::from_slice::<PullProgress>(&line) {
                last_status = p.status.clone();
                on_progress(p);
            }
        }
    }
    if last_status.starts_with("error") {
        anyhow::bail!("pull failed: {last_status}");
    }
    Ok(())
}

pub async fn tags(base: &str) -> Result<TagsResponse> {
    Ok(http()
        .get(url(base, "/api/tags"))
        .send()
        .await?
        .json()
        .await?)
}

pub async fn ps(base: &str) -> Result<PsResponse> {
    Ok(http()
        .get(url(base, "/api/ps"))
        .send()
        .await?
        .json()
        .await?)
}

pub async fn show(base: &str, model: &str) -> Result<ShowResponse> {
    let resp = http()
        .post(url(base, "/api/show"))
        .json(&ShowRequest {
            model: model.into(),
        })
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}

pub async fn delete(base: &str, model: &str) -> Result<()> {
    http()
        .post(url(base, "/api/delete"))
        .json(&ShowRequest {
            model: model.into(),
        })
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn stop(base: &str, model: &str) -> Result<()> {
    http()
        .post(url(base, "/api/stop"))
        .json(&ShowRequest {
            model: model.into(),
        })
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn shadow_start(base: &str, model: &str) -> Result<kredo_api::ShadowStatus> {
    let resp = http()
        .post(url(base, "/api/shadow"))
        .json(&kredo_api::ShadowRequest {
            model: Some(model.into()),
        })
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}

pub async fn shadow_stop(base: &str) -> Result<kredo_api::ShadowStatus> {
    let resp = http()
        .post(url(base, "/api/shadow"))
        .json(&kredo_api::ShadowRequest { model: None })
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}

pub async fn shadow_status(base: &str) -> Result<kredo_api::ShadowStatus> {
    Ok(http()
        .get(url(base, "/api/shadow"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

pub async fn shadow_report(base: &str) -> Result<kredo_api::ShadowReport> {
    Ok(http()
        .get(url(base, "/api/shadow/report"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

pub async fn promote(base: &str, model: Option<&str>) -> Result<kredo_api::ShadowStatus> {
    let resp = http()
        .post(url(base, "/api/promote"))
        .json(&kredo_api::PromoteRequest {
            model: model.map(String::from),
        })
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.json().await?)
}
