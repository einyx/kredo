//! jeff: run open decision models locally.
//!
//! `jeff serve` runs the daemon; every other subcommand talks to it over
//! HTTP, starting it automatically when it is not running. `jeff start`
//! runs a supervised daemon that restarts on crash.

mod calibrate;
mod client;
mod mcp;
mod output;
mod pidfile;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use kredo_api::{Question, DEFAULT_PORT};
use kredo_registry::Registry;
use kredo_server::{Config, LogFormat};
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "kredo",
    version,
    about = "Run open decision models locally, the way Ollama runs LLMs."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the kredo daemon in the foreground.
    Serve,
    /// Start a supervised daemon: restarts with backoff on crash.
    Start,
    /// Stop a running daemon.
    Stop_,
    /// Show daemon status.
    Status,
    /// Pull a model by name from the library.
    Pull { model: String },
    /// Decide against a model: jeff run kredo:en "state text" [--questions q.json]
    Run {
        model: String,
        state: String,
        /// JSON file with a question array; omitted = model's built-ins.
        #[arg(long)]
        questions: Option<String>,
    },
    /// List pulled models.
    List,
    /// List resident (loaded) models.
    Ps,
    /// Show a model's details.
    Show { model: String },
    /// Remove a pulled model.
    Rm { model: String },
    /// Copy a model to a new name.
    Cp { source: String, dest: String },
    /// Unload a resident model.
    Unload { model: String },
    /// Expose kredo models as MCP tools over stdio (claude mcp add kredo -- kredo mcp).
    Mcp,
    /// Shadow-evaluate a candidate model against live traffic without
    /// serving it.
    Shadow {
        #[command(subcommand)]
        action: ShadowAction,
    },
    /// Make a model the daemon default (what bare /v1/systemone routes to).
    Promote { model: String },
    /// Clear the daemon default (back to the library router).
    Demote,
    /// Open the decision playground in your browser.
    Ui,
    /// Re-run the model's verification fixtures locally and print a report.
    Verify { model: String },
    /// Fit per-head temperature calibration on a labeled eval set (JSONL:
    /// {"text": "...", "labels": {"question_id": "gold label"}}).
    Calibrate {
        model: String,
        /// Path to the eval set (JSON Lines).
        #[arg(short = 'e', long = "eval")]
        eval: String,
        /// Print the fitted temperatures without saving.
        #[arg(long)]
        dry_run: bool,
    },
    /// Bake a question set into a new local model (Modelfile).
    Create {
        name: String,
        /// Path to the Modelfile.
        #[arg(short = 'f', long = "file")]
        file: String,
    },
}

#[derive(Subcommand)]
enum ShadowAction {
    /// Start shadowing a candidate model.
    Start { model: String },
    /// Stop shadowing.
    Stop,
    /// Show shadow + promotion status.
    Status,
    /// Print the shadow agreement report.
    Report {
        /// Exit non-zero when agreement is below this threshold (0-1).
        #[arg(long)]
        min_agreement: Option<f64>,
    },
}

fn base_url() -> String {
    std::env::var("KREDO_HOST").unwrap_or_else(|_| format!("http://127.0.0.1:{DEFAULT_PORT}"))
}

/// Spawn a detached `jeff serve` and wait for it to answer.
async fn spawn_daemon() -> Result<()> {
    let exe = std::env::current_exe().context("locate jeff binary")?;
    tokio::process::Command::new(exe)
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .process_group(0)
        .spawn()
        .context("spawn `jeff serve`")?;
    for _ in 0..50 {
        if client::ping(&base_url()).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bail!("daemon did not come up at {}", base_url())
}

/// Ensure the daemon is up; start a detached one when it isn't.
async fn ensure_daemon() -> Result<()> {
    if client::ping(&base_url()).await.is_ok() {
        return Ok(());
    }
    if let Some(pid) = pidfile::running_pid() {
        bail!(
            "daemon pid {pid} is registered but not answering; remove {} if stale",
            pidfile::pid_path().display()
        );
    }
    spawn_daemon().await
}

fn load_questions(path: &str) -> Result<Vec<Question>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

/// Parse a minimal Modelfile:
/// ```text
/// FROM kredo:en
/// QUESTIONS ./triage.json
/// PARAMETER precision fp32
/// ```
fn parse_modelfile(path: &str) -> Result<(String, Option<Vec<Question>>, Option<String>)> {
    let dir = std::path::Path::new(path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let mut from = None;
    let mut questions = None;
    let mut precision = None;
    for line in std::fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (kw, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match kw.to_ascii_uppercase().as_str() {
            "FROM" => from = Some(rest.trim().to_string()),
            "QUESTIONS" => {
                let qp = dir.join(rest.trim());
                questions = Some(load_questions(qp.to_str().unwrap_or(rest))?);
            }
            "PARAMETER" => {
                let (k, v) = rest
                    .split_once(char::is_whitespace)
                    .context("PARAMETER needs key value")?;
                if k.eq_ignore_ascii_case("precision") {
                    precision = Some(v.trim().to_string());
                }
            }
            other => bail!("unknown Modelfile instruction: {other}"),
        }
    }
    Ok((
        from.context("Modelfile missing FROM")?,
        questions,
        precision,
    ))
}

fn init_logging() {
    let cfg = Config::from_env();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "kredo_server=info".into());
    let fmt = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    match cfg.log_format {
        LogFormat::Json => fmt.json().init(),
        LogFormat::Text => fmt.init(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve => serve(false).await,
        Command::Start => serve(true).await,
        Command::Stop_ => {
            match pidfile::running_pid() {
                Some(pid) => {
                    // Graceful first, force after a delay.
                    let _ = client::ping(&base_url()).await;
                    unsafe { libc::kill(pid, libc::SIGTERM) };
                    for _ in 0..30 {
                        if pidfile::running_pid().is_none() {
                            println!("stopped (pid {pid})");
                            return Ok(());
                        }
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                    pidfile::release();
                    println!("force-stopped (pid {pid})");
                    Ok(())
                }
                None => {
                    println!("no running daemon");
                    Ok(())
                }
            }
        }
        Command::Status => match client::ping(&base_url()).await {
            Ok(()) => {
                let v = client::version(&base_url()).await?;
                let tags = client::tags(&base_url()).await?;
                println!("daemon: up ({})", base_url());
                println!("version: {}", v["version"].as_str().unwrap_or("?"));
                println!("models:  {}", tags.models.len());
                Ok(())
            }
            Err(_) => {
                match pidfile::running_pid() {
                    Some(pid) => println!("daemon: registered (pid {pid}) but not answering"),
                    None => println!("daemon: down"),
                }
                Ok(())
            }
        },
        Command::Pull { model } => {
            ensure_daemon().await?;
            client::pull(&base_url(), &model, &mut |p| {
                output::print_progress(&p);
            })
            .await?;
            println!("\npulled {model}");
            Ok(())
        }
        Command::Run {
            model,
            state,
            questions,
        } => {
            ensure_daemon().await?;
            let qs = match &questions {
                Some(p) => Some(load_questions(p)?),
                None => None,
            };
            let resp = client::decide(&base_url(), &model, &state, qs).await?;
            output::print_answers(&resp.answers);
            Ok(())
        }
        Command::List => {
            ensure_daemon().await?;
            let tags = client::tags(&base_url()).await?;
            if tags.models.is_empty() {
                println!("no models pulled; try `kredo pull jeff`");
                return Ok(());
            }
            println!("NAME                    SIZE       MODIFIED");
            for m in tags.models {
                println!(
                    "{:<22}{:<10}{}",
                    m.model,
                    output::human_bytes(m.size),
                    m.modified_at
                );
            }
            Ok(())
        }
        Command::Ps => {
            ensure_daemon().await?;
            let ps = client::ps(&base_url()).await?;
            if ps.models.is_empty() {
                println!("no models resident");
                return Ok(());
            }
            println!("NAME                    SIZE       EXPIRES");
            for m in ps.models {
                println!(
                    "{:<22}{:<10}{}",
                    m.model,
                    output::human_bytes(m.size),
                    m.expires_at
                );
            }
            Ok(())
        }
        Command::Show { model } => {
            ensure_daemon().await?;
            let show = client::show(&base_url(), &model).await?;
            println!("name        {}", show.model);
            println!("size        {}", output::human_bytes(show.size));
            println!("modified    {}", show.modified_at);
            for (k, v) in &show.details {
                println!("{:<12}{}", k, v);
            }
            println!("questions   {}", show.questions.len());
            Ok(())
        }
        Command::Rm { model } => {
            ensure_daemon().await?;
            client::delete(&base_url(), &model).await?;
            println!("removed {model}");
            Ok(())
        }
        Command::Cp { source, dest } => {
            let reg = Registry::open();
            let m = reg.copy(&source, &dest)?;
            println!("copied {} -> {}", source, m.manifest.full_name());
            Ok(())
        }
        Command::Unload { model } => {
            ensure_daemon().await?;
            client::stop(&base_url(), &model).await?;
            println!("unloaded {model}");
            Ok(())
        }
        Command::Mcp => mcp::run().await,
        Command::Shadow { action } => {
            ensure_daemon().await?;
            match action {
                ShadowAction::Start { model } => {
                    let st = client::shadow_start(&base_url(), &model).await?;
                    println!("shadowing {model} (promoted: {:?})", st.promoted);
                    println!("watch agreement with `kredo shadow report`");
                }
                ShadowAction::Stop => {
                    client::shadow_stop(&base_url()).await?;
                    println!("shadow evaluation stopped");
                }
                ShadowAction::Status => {
                    let st = client::shadow_status(&base_url()).await?;
                    println!("shadow:   {:?}", st.shadow);
                    println!("promoted: {:?}", st.promoted);
                }
                ShadowAction::Report { min_agreement } => {
                    let r = client::shadow_report(&base_url()).await?;
                    println!("candidate   {}", r.model);
                    println!("evaluated   {} requests", r.n);
                    println!("agreement   {:.1}% ({})", r.agreement * 100.0, r.agree);
                    println!(
                        "latency     base {:.1} ms vs shadow {:.1} ms",
                        r.base_ms, r.shadow_ms
                    );
                    for q in &r.questions {
                        println!("  {:<20}{}/{} agree", q.question, q.agree, q.n);
                    }
                    if !r.samples.is_empty() {
                        println!("recent disagreements:");
                        for s in r.samples.iter().rev().take(5) {
                            if let Some(input) = s["input"].as_str() {
                                println!("  · {}", input.chars().take(80).collect::<String>());
                            }
                        }
                    }
                    if let Some(min) = min_agreement {
                        if r.n == 0 || r.agreement < min {
                            anyhow::bail!(
                                "agreement {:.1}% below threshold {:.1}%",
                                r.agreement * 100.0,
                                min * 100.0
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        Command::Promote { model } => {
            ensure_daemon().await?;
            client::promote(&base_url(), Some(&model)).await?;
            println!("promoted {model} to daemon default");
            Ok(())
        }
        Command::Demote => {
            ensure_daemon().await?;
            client::promote(&base_url(), None).await?;
            println!("daemon default cleared (library router)");
            Ok(())
        }
        Command::Ui => {
            ensure_daemon().await?;
            let url = format!("{}/ui", base_url());
            #[cfg(target_os = "macos")]
            let opened = tokio::process::Command::new("open")
                .arg(&url)
                .status()
                .await;
            #[cfg(target_os = "linux")]
            let opened = tokio::process::Command::new("xdg-open")
                .arg(&url)
                .status()
                .await;
            #[cfg(target_os = "windows")]
            let opened = tokio::process::Command::new("cmd")
                .args(["/c", "start", "", &url])
                .status()
                .await;
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            let opened: Result<std::process::ExitStatus, _> =
                Ok(std::process::ExitStatus::default());
            if opened.map(|s| s.success()).unwrap_or(false) {
                println!("playground: {url}");
            } else {
                println!("open {url} in your browser");
            }
            Ok(())
        }
        Command::Verify { model } => {
            let reg = Registry::open();
            let local = reg.get(&model)?;
            let report = kredo_runner::verification::verify(&local, &reg)?;
            print!("{}", kredo_runner::verification::render(&report));
            if !report.passed {
                std::process::exit(1);
            }
            Ok(())
        }
        Command::Calibrate {
            model,
            eval,
            dry_run,
        } => {
            calibrate::calibrate(&model, &eval, dry_run)?;
            Ok(())
        }
        Command::Create { name, file } => {
            let (from, questions, precision) = parse_modelfile(&file)?;
            let mut base = Registry::open().get(&from)?;
            if let Some(qs) = questions {
                base.manifest.questions = qs;
            }
            if let Some(p) = precision {
                base.manifest.precision = p;
            }
            let (name, tag) = kredo_registry::split_name_tag(&name)?;
            base.manifest.name = name;
            base.manifest.tag = tag;
            base.modified_at = "just now".into();
            let full = base.manifest.full_name();
            Registry::open().save(&base)?;
            println!("created {full}");
            Ok(())
        }
    }
}

/// Foreground serve; when `supervised`, restart with exponential backoff.
async fn serve(supervised: bool) -> Result<()> {
    pidfile::acquire()?;
    let cfg = Config::from_env();
    if supervised {
        // A supervised daemon defaults to text logs unless JSON was asked for;
        // restarts must not inherit a swallowed panic.
        cfg.validate().map_err(anyhow::Error::msg)?;
        let mut delay = Duration::from_secs(1);
        loop {
            let result = kredo_server::serve(cfg.clone()).await;
            match result {
                Ok(()) => break,
                Err(e) => {
                    eprintln!("daemon exited: {e}; restarting in {:?}", delay);
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(Duration::from_secs(60));
                }
            }
        }
    } else {
        kredo_server::serve(cfg).await?;
    }
    pidfile::release();
    Ok(())
}
