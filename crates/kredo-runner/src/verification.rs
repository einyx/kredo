//! Model verification: run golden fixture cases through the resident engine
//! and compare against expectations recorded at packaging time.
//!
//! The fixture (`verification.json`) is emitted by the convert/ pipeline
//! after the ONNX-vs-PyTorch parity gate. `kredo verify` re-runs it locally,
//! and `--require-verified` makes the daemon refuse models that don't pass.

use kredo_api::Question;
use kredo_registry::LocalModel;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixture {
    #[serde(default)]
    pub provenance: Option<kredo_registry::Provenance>,
    #[serde(default)]
    pub verification: Option<kredo_registry::Verification>,
    #[serde(default)]
    pub questions: Vec<Question>,
    pub cases: Vec<FixtureCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureCase {
    pub input: String,
    pub answers: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    pub input: String,
    pub passed: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub model: String,
    pub digest_ok: bool,
    pub cases_passed: u32,
    pub cases_total: u32,
    pub passed: bool,
    pub tolerance: f64,
    pub provenance: Option<kredo_registry::Provenance>,
    pub cases: Vec<CaseResult>,
}

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("model has no verification fixture (unverified provenance)")]
    NoFixture,
    #[error("fixture digest mismatch: manifest claims {expected}, file is {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error(transparent)]
    Runner(#[from] super::RunnerError),
    #[error(transparent)]
    Registry(#[from] kredo_registry::RegistryError),
    #[error("fixture: {0}")]
    Format(String),
}

/// Locate the fixture bytes for a pulled model: embedded artifact or a
/// downloaded blob recorded in the manifest digests.
fn fixture_bytes(model: &LocalModel, registry: &kredo_registry::Registry) -> Option<Vec<u8>> {
    if let Some(bytes) = kredo_registry::embedded::blob(
        &model.manifest.name,
        &model.manifest.tag,
        "verification.json",
    ) {
        return Some(bytes.to_vec());
    }
    if let Some(digest) = model.digests.get("verification.json") {
        if let Ok(bytes) = std::fs::read(registry.blob_path(digest)) {
            return Some(bytes);
        }
    }
    None
}

/// Verify `model` end-to-end. Loads the engine and re-runs every fixture
/// case against the recorded expectations.
pub fn verify(
    model: &LocalModel,
    registry: &kredo_registry::Registry,
) -> Result<Report, VerifyError> {
    let bytes = fixture_bytes(model, registry).ok_or(VerifyError::NoFixture)?;
    let fixture: Fixture =
        serde_json::from_slice(&bytes).map_err(|e| VerifyError::Format(e.to_string()))?;

    let actual_digest = hex::encode(Sha256::digest(&bytes));
    let digest_ok = match (&model.manifest.verification, &fixture.verification) {
        (Some(mv), Some(_)) => {
            // Case-insensitive hex compare on the recorded fixture digest.
            mv.fixture_digest.eq_ignore_ascii_case(&actual_digest)
        }
        _ => true,
    };

    let engine = super::load(model, registry, super::Device::Cpu)?;
    let tolerance = fixture
        .verification
        .as_ref()
        .map(|v| v.tolerance)
        .unwrap_or(1e-4);

    let mut case_results = Vec::new();
    let mut passed_cases = 0u32;
    for case in &fixture.cases {
        let answers = engine.decide(&case.input, &fixture.questions)?;
        let mut failures = Vec::new();
        for want in &case.answers {
            let id = want["id"].as_str().unwrap_or_default().to_string();
            let got = answers.iter().find(|a| a.id == id);
            let Some(got) = got else {
                failures.push(format!("{id}: no answer"));
                continue;
            };
            match want["type"].as_str().unwrap_or_default() {
                "choice" => {
                    for exp in want["value"].as_array().unwrap_or(&vec![]) {
                        let label = exp["label"].as_str().unwrap_or_default();
                        let p = exp["p"].as_f64().unwrap_or_default();
                        let got_p = got
                            .probabilities
                            .iter()
                            .find(|x| x.label == label)
                            .map(|x| x.p);
                        match got_p {
                            Some(gp) if (gp - p).abs() <= tolerance => {}
                            Some(gp) => {
                                failures.push(format!("{id}/{label}: got {gp:.6}, want {p:.6}"))
                            }
                            None => failures.push(format!("{id}/{label}: missing")),
                        }
                    }
                }
                "score" => {
                    let want_v = want["value"]["value"].as_f64().unwrap_or_default();
                    match &got.score {
                        Some(s) if (s.value - want_v).abs() <= tolerance => {}
                        Some(s) => {
                            failures.push(format!("{id}: got {:.6}, want {want_v:.6}", s.value))
                        }
                        None => failures.push(format!("{id}: no score")),
                    }
                }
                "noul" => {
                    let want_v = want["value"].as_f64().unwrap_or_default();
                    match got.p {
                        Some(gp) if (gp - want_v).abs() <= tolerance => {}
                        Some(gp) => failures.push(format!("{id}: got {gp:.6}, want {want_v:.6}")),
                        None => failures.push(format!("{id}: no p")),
                    }
                }
                other => failures.push(format!("{id}: unknown kind {other}")),
            }
        }
        let case_passed = failures.is_empty();
        if case_passed {
            passed_cases += 1;
        }
        case_results.push(CaseResult {
            input: case.input.clone(),
            passed: case_passed,
            failures,
        });
    }

    let total = fixture.cases.len() as u32;
    Ok(Report {
        model: model.manifest.full_name(),
        digest_ok,
        cases_passed: passed_cases,
        cases_total: total,
        passed: digest_ok && passed_cases == total,
        tolerance,
        provenance: fixture.provenance,
        cases: case_results,
    })
}

/// Render a report as human-readable terminal text.
pub fn render(report: &Report) -> String {
    let mut out = String::new();
    out.push_str(&format!("{} verification\n", report.model));
    out.push_str(&format!(
        "  fixture digest   {} (tolerance {:.0e})\n",
        if report.digest_ok { "ok" } else { "MISMATCH" },
        report.tolerance
    ));
    out.push_str(&format!(
        "  cases            {}/{} passed\n",
        report.cases_passed, report.cases_total
    ));
    if let Some(prov) = &report.provenance {
        out.push_str(&format!("  dataset          {}\n", prov.dataset));
        if let Some(pipeline) = &prov.pipeline {
            out.push_str(&format!("  pipeline         {pipeline}\n"));
        }
        let mut metrics: Vec<_> = prov.metrics.iter().collect();
        metrics.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in metrics {
            out.push_str(&format!("  {k:<17}{v:.4}\n"));
        }
    }
    for case in &report.cases {
        if !case.passed {
            out.push_str(&format!("  FAIL: {}\n", case.input));
            for f in &case.failures {
                out.push_str(&format!("    {f}\n"));
            }
        }
    }
    out.push_str(if report.passed {
        "VERIFIED\n"
    } else {
        "NOT VERIFIED\n"
    });
    out
}
