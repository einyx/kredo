//! `kredo calibrate`: fit per-head temperature scaling on a labeled eval set.
//!
//! Temperature scaling (Guo et al., 2017) is the simplest post-hoc
//! calibrator: logits are divided by a per-head temperature T before the
//! softmax/sigmoid. It preserves the argmax (accuracy is unchanged) while
//! making the reported probabilities honest. The search runs on the
//! daemon's decision outputs — no retraining, no Python.

use anyhow::{bail, Context, Result};
use kredo_api::{Question, QuestionKind};
use kredo_registry::Registry;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// One eval example: input text plus the gold label per question id.
#[derive(Debug, Deserialize)]
struct EvalCase {
    text: String,
    labels: BTreeMap<String, String>,
}

/// Grid-search result for one head.
struct Fit {
    temperature: f64,
    loss: f64,
}

const GRID: std::ops::RangeInclusive<f64> = 0.25f64..=4.0;
const GRID_STEP: f64 = 0.05;

/// Apply temperature to a distribution in probability space: softmax over
/// `ln(p) / T` reproduces temperature scaling exactly.
fn scaled_distribution(probs: &[f64], t: f64) -> Vec<f64> {
    let logits: Vec<f64> = probs.iter().map(|p| p.max(1e-12).ln()).collect();
    kredo_decision::softmax(&logits.iter().map(|z| z / t).collect::<Vec<_>>())
}

/// NLL of the gold label under the temperature-scaled distribution.
fn nll(probs: &[f64], t: f64, gold: usize) -> f64 {
    let scaled = scaled_distribution(probs, t);
    -(scaled[gold].max(1e-12).ln())
}

/// Fit one scalar temperature by minimizing the mean loss over samples.
fn fit_temperature<F>(loss_at: F) -> Result<Fit>
where
    F: Fn(f64) -> f64,
{
    let mut best = (f64::INFINITY, 1.0);
    let mut t = *GRID.start();
    while t <= *GRID.end() {
        let loss = loss_at(t);
        if loss < best.0 {
            best = (loss, t);
        }
        t += GRID_STEP;
    }
    if !best.0.is_finite() {
        bail!("calibration search produced no finite loss");
    }
    Ok(Fit {
        temperature: (best.1 * 100.0).round() / 100.0,
        loss: best.0,
    })
}

/// Run the eval set through the engine once and fit a temperature per head.
pub fn calibrate(model_spec: &str, eval_path: &str, dry_run: bool) -> Result<Vec<(String, f64)>> {
    let registry = Registry::open();
    let local = registry.get(model_spec)?;
    let questions: Vec<Question> = local.manifest.questions.clone();
    if questions.is_empty() {
        bail!("model has no built-in question set to calibrate");
    }

    let mut cases = Vec::new();
    for (i, line) in std::fs::read_to_string(eval_path)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        cases.push(
            serde_json::from_str::<EvalCase>(line).with_context(|| format!("line {}", i + 1))?,
        );
    }
    if cases.len() < 20 {
        bail!(
            "eval set too small ({} cases); need at least 20 for a stable fit",
            cases.len()
        );
    }

    let engine = kredo_runner::load(&local, &registry, kredo_runner::Device::Cpu)?;

    // Group eval cases by the gold label of each question.
    let mut per_question: BTreeMap<String, Vec<(Vec<f64>, usize)>> = BTreeMap::new();
    let mut scalar_cases: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new(); // (unit, target)
    let mut scored = 0usize;
    for case in &cases {
        let answers = engine.decide(&case.text, &questions)?;
        for a in &answers {
            let Some(gold) = case.labels.get(&a.id) else {
                continue;
            };
            match a.kind {
                QuestionKind::Choice => {
                    let probs: Vec<f64> = a.probabilities.iter().map(|p| p.p).collect();
                    let Some(gi) = a.probabilities.iter().position(|p| p.label == *gold) else {
                        bail!("gold label `{gold}` not in options for question `{}`", a.id);
                    };
                    per_question
                        .entry(a.id.clone())
                        .or_default()
                        .push((probs, gi));
                    scored += 1;
                }
                QuestionKind::Noul => {
                    let p = a.p.context("noul answer missing p")?;
                    let target = if gold.eq_ignore_ascii_case("yes") {
                        1.0
                    } else {
                        0.0
                    };
                    scalar_cases
                        .entry(a.id.clone())
                        .or_default()
                        .push((p, target));
                    scored += 1;
                }
                QuestionKind::Score => {
                    let s = a.score.as_ref().context("score answer missing score")?;
                    let target_val: f64 = gold.parse().with_context(|| {
                        format!("question `{}`: gold `{gold}` is not numeric", a.id)
                    })?;
                    let unit = (target_val - s.min) / (s.max - s.min);
                    scalar_cases
                        .entry(a.id.clone())
                        .or_default()
                        .push((s.normalized, unit.clamp(0.0, 1.0)));
                    scored += 1;
                }
            }
        }
    }
    if scored == 0 {
        bail!("no eval labels matched the model's question ids");
    }

    let mut fits: Vec<(String, f64)> = Vec::new();
    for q in &questions {
        if let Some(samples) = per_question.get(&q.id) {
            let fit = fit_temperature(|t| {
                samples.iter().map(|(p, g)| nll(p, t, *g)).sum::<f64>() / samples.len() as f64
            })?;
            println!(
                "{:<20} choice  T={:.2}  nll={:.4}  n={}",
                q.id,
                fit.temperature,
                fit.loss,
                samples.len()
            );
            fits.push((q.id.clone(), fit.temperature));
        } else if let Some(samples) = scalar_cases.get(&q.id) {
            let fit = fit_temperature(|t| {
                let mut s = 0.0;
                for (u, target) in samples {
                    let z = (u.clamp(1e-6, 1.0 - 1e-6) / (1.0 - u.clamp(1e-6, 1.0 - 1e-6))).ln();
                    let scaled = 1.0 / (1.0 + (-(z / t)).exp());
                    s += (scaled - target).powi(2);
                }
                s / samples.len() as f64
            })?;
            println!(
                "{:<20} scalar  T={:.2}  mse={:.4}  n={}",
                q.id,
                fit.temperature,
                fit.loss,
                samples.len()
            );
            fits.push((q.id.clone(), fit.temperature));
        }
    }

    if dry_run {
        println!("dry run — manifest unchanged");
        return Ok(fits);
    }

    // Write temperatures into the manifest decision spec.
    let mut local = local;
    let kredo_decision::HeadLayout::MultiHead { heads } = &mut local.manifest.decision.layout
    else {
        bail!("calibrate supports multi-head (trained) models; zero-shot models calibrate via the pairwise temperature in the Modelfile");
    };
    for (qid, t) in &fits {
        if let Some(h) = heads.iter_mut().find(|h| h.question == *qid) {
            h.temperature = Some(*t);
        }
    }

    // Re-record the verification fixture with the calibrated outputs so the
    // verification gate reflects the shipped (calibrated) behavior.
    let engine = kredo_runner::load(&local, &registry, kredo_runner::Device::Cpu)?;
    let mut case_records = Vec::new();
    for case in cases.iter().take(3) {
        let answers = engine.decide(&case.text, &questions)?;
        let answers: Vec<serde_json::Value> = answers
            .iter()
            .map(|a| serde_json::to_value(a).unwrap_or_default())
            .collect();
        case_records.push(serde_json::json!({ "input": case.text, "answers": answers }));
    }
    let fixture = serde_json::json!({
        "questions": questions,
        "cases": case_records,
    });
    let bytes = serde_json::to_vec_pretty(&fixture)?;
    let digest = hex::encode(Sha256::digest(&bytes));
    let blob = registry.blobs_dir().join(format!("sha256-{digest}"));
    std::fs::create_dir_all(registry.blobs_dir())?;
    std::fs::write(&blob, &bytes)?;
    if let Some(v) = &mut local.manifest.verification {
        v.fixture_digest = digest.clone();
        v.verified_at = chrono_now();
        v.cases = case_records.len() as u32;
    } else {
        local.manifest.verification = Some(kredo_registry::Verification {
            fixture_digest: digest.clone(),
            tolerance: 1e-4,
            verified_at: chrono_now(),
            cases: case_records.len() as u32,
        });
    }
    if let Some(vf) = local
        .manifest
        .source
        .files
        .iter_mut()
        .find(|f| f.path == "verification.json")
    {
        vf.sha256 = Some(digest);
    }
    if let Some(digests) = local.digests.get_mut("verification.json") {
        let new_digest = hex::encode(Sha256::digest(&bytes));
        *digests = new_digest;
    }
    registry.save(&local)?;
    println!(
        "calibrated {} — re-recorded verification fixture",
        local.manifest.full_name()
    );
    Ok(fits)
}

/// RFC 3339 UTC timestamp from the wall clock (no external deps).
fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, m, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}+00:00")
}
