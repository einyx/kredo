//! Resumable pulls: fetch model files into the content-addressed blob store,
//! verify sha256 and persist a local manifest.

use crate::{LocalModel, Manifest, Registry};
use kredo_api::PullProgress;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

#[derive(Debug, thiserror::Error)]
pub enum PullError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("sha256 mismatch for {path}: expected {expected}, got {actual}")]
    DigestMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("registry: {0}")]
    Registry(#[from] crate::RegistryError),
}

/// Hugging Face resolve URL for a repo file at a revision.
fn file_url(manifest: &Manifest, spec: &crate::FileSpec) -> String {
    if let Some(url) = &spec.url {
        return url.clone();
    }
    let repo = manifest.source.repo.as_deref().unwrap_or_default();
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo, manifest.source.revision, spec.path
    )
}

fn now_rfc3339() -> String {
    // No chrono dependency; seconds-precision via SystemTime.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{} epoch-seconds", d.as_secs())
}

/// Progress callback: receives NDJSON-serializable progress events.
pub type OnProgress<'a> = dyn FnMut(PullProgress) + Send + Sync + 'a;

/// Pull (or resume pulling) `manifest` into `registry`.
pub async fn pull(
    manifest: &Manifest,
    registry: &Registry,
    on_progress: &mut OnProgress<'_>,
) -> Result<LocalModel, PullError> {
    tokio::fs::create_dir_all(registry.blobs_dir()).await?;
    let client = reqwest::Client::builder()
        .user_agent("kredo/0.1 (+https://github.com/einyx/kredo)")
        .build()?;

    // Reuse blobs recorded by a previous pull of this model, even when the
    // library manifest does not pin digests.
    let previous = registry
        .get(&format!("{}:{}", manifest.name, manifest.tag))
        .ok();

    let mut digests = std::collections::BTreeMap::new();
    let mut total_size = 0u64;

    for spec in &manifest.source.files {
        let url = file_url(manifest, spec);
        on_progress(PullProgress {
            status: format!("pulling {}", spec.path),
            digest: spec.sha256.clone(),
            total: None,
            completed: Some(0),
        });

        // Fast path 0: artifact embedded in the binary (trained models).
        if let Some(bytes) = crate::embedded::blob(&manifest.name, &manifest.tag, &spec.path) {
            let digest = hex::encode(Sha256::digest(bytes));
            let dest = registry.blob_path(&digest);
            if !dest.exists() {
                tokio::fs::write(&dest, bytes).await?;
            }
            let size = bytes.len() as u64;
            digests.insert(spec.path.clone(), digest.clone());
            total_size += size;
            on_progress(PullProgress {
                status: format!("embedded {}", spec.path),
                digest: Some(digest),
                total: Some(size),
                completed: Some(size),
            });
            continue;
        }

        // Fast path 1: pinned digest already in the blob store.
        let mut reused: Option<(String, u64)> = None;
        if let Some(expected) = &spec.sha256 {
            let p = registry.blob_path(expected);
            if p.exists() {
                reused = Some((expected.clone(), std::fs::metadata(&p)?.len()));
            }
        }
        // Fast path 2: previously pulled digest for the same source path.
        if reused.is_none() {
            if let Some(prev) = &previous {
                if let Some(digest) = prev.digests.get(&spec.path) {
                    let p = registry.blob_path(digest);
                    if p.exists() {
                        reused = Some((digest.clone(), std::fs::metadata(&p)?.len()));
                    }
                }
            }
        }
        if let Some((digest, size)) = reused {
            digests.insert(spec.path.clone(), digest.clone());
            total_size += size;
            on_progress(PullProgress {
                status: format!("already present {}", spec.path),
                digest: Some(digest),
                total: Some(size),
                completed: Some(size),
            });
            continue;
        }

        let (digest, size) =
            download_resumable(&client, &url, &spec.path, registry, on_progress).await?;
        if let Some(expected) = &spec.sha256 {
            if &digest != expected {
                return Err(PullError::DigestMismatch {
                    path: spec.path.clone(),
                    expected: expected.clone(),
                    actual: digest,
                });
            }
        }
        digests.insert(spec.path.clone(), digest.clone());
        total_size += size;
        on_progress(PullProgress {
            status: format!("pulled {}", spec.path),
            digest: Some(digest),
            total: Some(size),
            completed: Some(size),
        });
    }

    let model = LocalModel {
        manifest: manifest.clone(),
        digests,
        size: total_size,
        modified_at: now_rfc3339(),
    };
    registry.save(&model)?;
    on_progress(PullProgress {
        status: "success".into(),
        digest: None,
        total: Some(total_size),
        completed: Some(total_size),
    });
    Ok(model)
}

/// Download `url` into the blob store with HTTP Range resume support.
/// Returns (sha256 hex, bytes on disk).
async fn download_resumable(
    client: &reqwest::Client,
    url: &str,
    label: &str,
    registry: &Registry,
    on_progress: &mut OnProgress<'_>,
) -> Result<(String, u64), PullError> {
    let tmp = partial_path(registry, url);
    let mut offset: u64 = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

    let mut request = client.get(url);
    if offset > 0 {
        request = request.header("Range", format!("bytes={offset}-"));
    }
    let resp = request.send().await?.error_for_status()?;

    // Server ignored the Range header: start over.
    if offset > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        offset = 0;
        tokio::fs::File::create(&tmp).await?;
    }

    let total: Option<u64> = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .map(|v: u64| v + offset);
    let content_range_total = resp
        .headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .and_then(|cr| cr.rsplit('/').next())
        .and_then(|v| v.parse::<u64>().ok());
    let total = content_range_total.or(total);

    on_progress(PullProgress {
        status: format!("downloading {label}"),
        digest: None,
        total,
        completed: Some(offset),
    });

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&tmp)
        .await?;

    // Hash only the newly downloaded bytes; a prior partial prefix has no
    // verified hash, so resume requires re-hashing the whole file afterwards.
    let mut hasher = Sha256::new();
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    let mut completed = offset;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        completed += chunk.len() as u64;
        on_progress(PullProgress {
            status: format!("downloading {label}"),
            digest: None,
            total,
            completed: Some(completed),
        });
    }
    file.flush().await?;
    drop(file);

    let digest = hex::encode(hasher.finalize());

    // If we resumed, re-hash the full file to be safe.
    if offset > 0 {
        let full = hash_file(&tmp).await?;
        return finalize(tmp, registry.blob_path(&full), full, label, registry).await;
    }

    finalize(tmp, registry.blob_path(&digest), digest, label, registry).await
}

async fn finalize(
    tmp: PathBuf,
    dest: PathBuf,
    digest: String,
    label: &str,
    _registry: &Registry,
) -> Result<(String, u64), PullError> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::rename(&tmp, &dest).await?;
    let size = tokio::fs::metadata(&dest).await?.len();
    tracing::debug!("pulled {label} -> sha256:{digest} ({size} bytes)");
    Ok((digest, size))
}

async fn hash_file(path: &PathBuf) -> Result<String, PullError> {
    let data = tokio::fs::read(path).await?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(hex::encode(hasher.finalize()))
}

fn partial_path(registry: &Registry, url: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let key = hex::encode(&hasher.finalize()[..8]);
    registry.blobs_dir().join(format!("{key}.part"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_hf_urls() {
        let m = crate::Manifest::from_library("kredo:en").unwrap();
        let spec = &m.source.files[0];
        assert_eq!(
            file_url(&m, spec),
            format!(
                "https://huggingface.co/Xenova/distilbert-base-uncased-mnli/resolve/{}/onnx/model.onnx",
                m.source.revision
            )
        );
        // kredo:en must be pinned to an exact upstream commit with digests.
        assert_ne!(m.source.revision, "main");
        assert!(spec.sha256.is_some());
    }
}
