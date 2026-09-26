//! Terminal rendering: progress lines and bar-chart answers.

use kredo_api::{Answer, PullProgress};

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

pub fn print_progress(p: &PullProgress) {
    match (&p.total, &p.completed) {
        (Some(total), Some(done)) if *total > 0 => {
            let pct = (*done as f64 / *total as f64 * 100.0).min(100.0);
            eprint!("\r{} {:>5.1}%", p.status, pct);
        }
        _ => eprintln!("{}", p.status),
    }
}

fn bar(p: f64, width: usize) -> String {
    let filled = (p.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

pub fn print_answers(answers: &[Answer]) {
    for a in answers {
        match a.kind {
            kredo_api::QuestionKind::Choice => {
                if let Some(top) = a.top() {
                    println!(
                        "{:<18}{:<14}{} {:.2}",
                        a.id,
                        top.label,
                        bar(top.p, 16),
                        top.p
                    );
                }
            }
            kredo_api::QuestionKind::Score => {
                if let Some(s) = &a.score {
                    println!(
                        "{:<18}{:.2} / {:.0}   {} {:.2}",
                        a.id,
                        s.value,
                        s.max,
                        bar(s.normalized, 16),
                        s.normalized
                    );
                }
            }
            kredo_api::QuestionKind::Noul => {
                if let Some(p) = a.p {
                    println!("{:<18}{:>4} {} {:.2}", a.id, "", bar(p, 16), p);
                }
            }
        }
    }
}
