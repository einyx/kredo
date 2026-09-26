//! Model names, manifests, the blob store and resumable pulls.
//!
//! Layout under the models root (`$KREDO_MODELS`, default `~/.kredo/models`):
//!
//! ```text
//! blobs/<sha256>              content-addressed files (graphs, weights, tokenizers)
//! manifests/<name>/<tag>.json pulled manifests (library manifest + local digests)
//! ```

pub mod embedded;
pub mod library;
pub mod pull;

use kredo_decision::DecisionSpec;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("model `{0}` not found (run `kredo pull {0}`)")]
    NotFound(String),
    #[error("invalid model name `{0}` (expected `name[:tag]`)")]
    InvalidName(String),
    #[error("manifest schema {0} not supported (this build supports {1}); upgrade kredo")]
    UnsupportedSchema(u32, u32),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("pull failed: {0}")]
    Pull(#[from] Box<pull::PullError>),
}

/// A single remote file of a model: an ONNX graph, safetensors weights or a
/// tokenizer. Files resolve from `url` when set (registry-hosted graphs), or
/// from the Hugging Face repository at a pinned revision. `sha256` is
/// verified when known; when absent, the digest is recorded after download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSpec {
    /// Path inside the source repository (e.g. `onnx/model.onnx`).
    pub path: String,
    /// Absolute URL override (registry-hosted ONNX graphs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Expected sha256 of the content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Whether this file must be resident on disk to run the model
    /// (tokenizers yes; optional auxiliary files no).
    #[serde(default = "yes")]
    pub required: bool,
}

fn yes() -> bool {
    true
}

/// Where a model's files come from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Hugging Face repository, e.g. `onnx-community/deberta-v3-base`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Pinned revision (commit sha or tag). Weight files always resolve
    /// against this revision.
    #[serde(default = "default_revision")]
    pub revision: String,
    pub files: Vec<FileSpec>,
}

fn default_revision() -> String {
    "main".into()
}

fn default_schema() -> u32 {
    1
}

/// Manifest schema version this build understands.
pub const SUPPORTED_SCHEMA: u32 = 1;

/// A router model delegates to concrete tags per script/language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Router {
    pub en: String,
    pub multilingual: String,
}

/// Where the model's training data and evaluation numbers come from.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Provenance {
    /// Human-readable dataset description.
    pub dataset: String,
    /// Generator/training seed, when deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// convert/ pipeline version or commit the model was built with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<String>,
    /// Per-head evaluation metrics, e.g. `intent.accuracy` = 0.97,
    /// `frustration.mae` = 0.41.
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
}

/// A passing parity/fixture run recorded at packaging time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    /// sha256 of the `verification.json` fixture file this refers to.
    pub fixture_digest: String,
    /// Max absolute deviation the fixture run allows.
    pub tolerance: f64,
    /// RFC-3339-ish timestamp of the verified run.
    pub verified_at: String,
    /// Number of fixture cases.
    pub cases: u32,
}

/// The registry/library manifest of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest schema version; only 1 is understood.
    #[serde(default = "default_schema")]
    pub schema: u32,
    pub name: String,
    pub tag: String,
    /// Short description.
    pub description: String,
    /// Inference engine: `onnx-pairwise` or `onnx-multihead`.
    pub engine: String,
    pub precision: String,
    pub max_seq_len: usize,
    pub source: Source,
    /// How head outputs map to questions.
    pub decision: DecisionSpec,
    /// Built-in question set, used when a request omits `questions`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub questions: Vec<kredo_api::Question>,
    /// Router delegation (router models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<Router>,
    /// Training provenance (trained models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// Recorded passing fixture/parity run (verified models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<Verification>,
}

impl Manifest {
    /// Resolve `name[:tag]` against the embedded library.
    pub fn from_library(spec: &str) -> Result<Manifest, RegistryError> {
        let (name, tag) = split_name_tag(spec)?;
        library::lookup(&name, &tag).ok_or(RegistryError::NotFound(spec.to_string()))
    }

    /// Fully-qualified model name, e.g. `kredo:en`.
    pub fn full_name(&self) -> String {
        format!("{}:{}", self.name, self.tag)
    }
}

/// Split `name[:tag]`, defaulting the tag to `latest`.
pub fn split_name_tag(spec: &str) -> Result<(String, String), RegistryError> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(RegistryError::InvalidName(spec.into()));
    }
    match spec.split_once(':') {
        Some((n, t)) if !n.is_empty() && !t.is_empty() => Ok((n.into(), t.into())),
        Some(_) => Err(RegistryError::InvalidName(spec.into())),
        None => Ok((spec.into(), "latest".into())),
    }
}

/// Local, materialized model: library manifest plus recorded digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    #[serde(flatten)]
    pub manifest: Manifest,
    /// File path -> blob digest.
    pub digests: BTreeMap<String, String>,
    /// Total bytes on disk.
    pub size: u64,
    /// Last-modified timestamp (RFC 3339).
    pub modified_at: String,
}

/// The on-disk registry.
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    pub fn open() -> Self {
        let root = std::env::var("KREDO_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
                home.join(".kredo").join("models")
            });
        Self { root }
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    pub fn manifests_dir(&self) -> PathBuf {
        self.root.join("manifests")
    }

    fn manifest_path(&self, name: &str, tag: &str) -> PathBuf {
        self.manifests_dir().join(name).join(format!("{tag}.json"))
    }

    /// List all pulled models.
    pub fn list(&self) -> Result<Vec<LocalModel>, RegistryError> {
        let mut out = Vec::new();
        let dir = self.manifests_dir();
        if !dir.exists() {
            return Ok(out);
        }
        for name_entry in std::fs::read_dir(&dir)? {
            let name_dir = name_entry?.path();
            if !name_dir.is_dir() {
                continue;
            }
            for tag_entry in std::fs::read_dir(&name_dir)? {
                let p = tag_entry?.path();
                if p.extension().is_some_and(|e| e == "json") {
                    let m: LocalModel = serde_json::from_slice(&std::fs::read(&p)?)?;
                    out.push(m);
                }
            }
        }
        out.sort_by_key(|m| m.manifest.full_name());
        Ok(out)
    }

    /// Load a pulled model by `name[:tag]`.
    pub fn get(&self, spec: &str) -> Result<LocalModel, RegistryError> {
        let (name, tag) = split_name_tag(spec)?;
        let p = self.manifest_path(&name, &tag);
        if !p.exists() {
            return Err(RegistryError::NotFound(spec.to_string()));
        }
        let model: LocalModel = serde_json::from_slice(&std::fs::read(&p)?)?;
        if model.manifest.schema != SUPPORTED_SCHEMA {
            return Err(RegistryError::UnsupportedSchema(
                model.manifest.schema,
                SUPPORTED_SCHEMA,
            ));
        }
        Ok(model)
    }

    /// Persist a pulled model.
    pub fn save(&self, model: &LocalModel) -> Result<(), RegistryError> {
        let p = self.manifest_path(&model.manifest.name, &model.manifest.tag);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&p, serde_json::to_vec_pretty(model)?)?;
        Ok(())
    }

    /// Remove a pulled model. Returns false when it wasn't present.
    pub fn remove(&self, spec: &str) -> Result<bool, RegistryError> {
        let (name, tag) = split_name_tag(spec)?;
        let p = self.manifest_path(&name, &tag);
        if p.exists() {
            std::fs::remove_file(&p)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Copy a pulled model to a new local name/tag.
    pub fn copy(&self, src: &str, dest_spec: &str) -> Result<LocalModel, RegistryError> {
        let mut model = self.get(src)?;
        let (name, tag) = split_name_tag(dest_spec)?;
        model.manifest.name = name;
        model.manifest.tag = tag;
        self.save(&model)?;
        Ok(model)
    }

    pub fn blob_path(&self, digest: &str) -> PathBuf {
        self.blobs_dir().join(format!("sha256-{digest}"))
    }

    /// Best-effort size of a blob (0 when missing).
    pub fn blob_size(&self, digest: &str) -> u64 {
        std::fs::metadata(self.blob_path(digest))
            .map(|m| m.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_names() {
        assert_eq!(
            split_name_tag("kredo:en").unwrap(),
            ("kredo".into(), "en".into())
        );
        assert_eq!(
            split_name_tag("kredo").unwrap(),
            ("kredo".into(), "latest".into())
        );
        assert!(split_name_tag(":tag").is_err());
        assert!(split_name_tag("name:").is_err());
    }

    #[test]
    fn library_resolves() {
        let m = Manifest::from_library("kredo:en").unwrap();
        assert_eq!(m.name, "kredo");
        let m = Manifest::from_library("nli").unwrap();
        assert_eq!(m.tag, "latest");
        assert!(Manifest::from_library("nope").is_err());
    }
}
