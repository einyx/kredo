//! PID-file helpers for the daemon lifecycle.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

/// `~/.kredo/kredo.pid` (or under the models root).
pub fn pid_path() -> PathBuf {
    std::env::var("KREDO_PID_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".kredo")
                .join("kredo.pid")
        })
}

fn pid_alive(pid: i32) -> bool {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid, 0) == 0
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// Refuse to start when another daemon holds the PID file.
pub fn acquire() -> Result<()> {
    let path = pid_path();
    if let Ok(content) = std::fs::read_to_string(&path) {
        if let Ok(pid) = content.trim().parse::<i32>() {
            if pid_alive(pid) && pid != std::process::id() as i32 {
                bail!(
                    "kredo daemon already running (pid {pid}, {})",
                    path.display()
                );
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, std::process::id().to_string())
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Remove the PID file on shutdown (best-effort).
pub fn release() {
    let _ = std::fs::remove_file(pid_path());
}

/// Read the PID of a running daemon, if any.
pub fn running_pid() -> Option<i32> {
    let content = std::fs::read_to_string(pid_path()).ok()?;
    let pid = content.trim().parse::<i32>().ok()?;
    if pid_alive(pid) {
        Some(pid)
    } else {
        None
    }
}
