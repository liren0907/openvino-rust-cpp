use crate::examples::ExampleKind;
use crate::event::AppEvent;
use crate::parser::{self, ParsedLine, VideoMeta};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};

#[derive(Debug)]
pub enum TaskStatus {
    Running,
    Completed,
    Failed(String),
}

pub struct Task {
    pub id: usize,
    pub kind: ExampleKind,
    pub status: TaskStatus,
    pub log_lines: Vec<String>,
    pub video_meta: Option<VideoMeta>,
    pub current_frame: u32,
    pub total_frames: Option<u32>,
    child: Arc<Mutex<Option<tokio::process::Child>>>,
}

impl Task {
    pub fn progress_text(&self) -> String {
        match &self.status {
            TaskStatus::Completed => {
                let total = self.total_frames.unwrap_or(self.current_frame);
                format!("Done ({} frames)", total)
            }
            TaskStatus::Failed(msg) => format!("FAILED: {}", msg),
            TaskStatus::Running => {
                if let Some(total) = self.total_frames {
                    let pct = if total > 0 {
                        (self.current_frame as f64 / total as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    format!("Frame {} / {} ({}%)", self.current_frame, total, pct)
                } else {
                    format!("Frame {}", self.current_frame)
                }
            }
        }
    }

    pub fn progress_ratio(&self) -> f64 {
        match &self.status {
            TaskStatus::Completed => 1.0,
            TaskStatus::Failed(_) => 0.0,
            TaskStatus::Running => {
                if let Some(total) = self.total_frames {
                    if total > 0 {
                        (self.current_frame as f64 / total as f64).min(1.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, TaskStatus::Running)
    }

    pub fn push_log(&mut self, line: String) {
        if self.log_lines.len() >= 500 {
            self.log_lines.remove(0);
        }
        self.log_lines.push(line);
    }
}

pub struct TaskManager {
    pub tasks: Vec<Task>,
    next_id: usize,
    workspace_root: PathBuf,
}

impl TaskManager {
    pub fn new() -> Self {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = PathBuf::from(manifest_dir)
            .parent().expect("CARGO_MANIFEST_DIR has no parent (expected crates/)")
            .parent().expect("crates/ has no parent (expected workspace root)")
            .to_path_buf();

        Self {
            tasks: Vec::new(),
            next_id: 0,
            workspace_root,
        }
    }

    pub fn spawn(
        &mut self,
        kind: ExampleKind,
        config: &HashMap<String, String>,
        event_tx: mpsc::UnboundedSender<AppEvent>,
    ) -> Result<usize, String> {
        let binary_path = self
            .workspace_root
            .join("target")
            .join("debug")
            .join(kind.binary_name());

        if !binary_path.exists() {
            return Err(format!(
                "Binary not found: {}. Run: cargo build -p {}",
                binary_path.display(),
                kind.binary_name()
            ));
        }

        let mut cmd = Command::new(&binary_path);
        cmd.current_dir(&self.workspace_root);
        cmd.arg("--no-show");

        for (flag, value) in config {
            if value.is_empty() {
                continue;
            }
            cmd.arg(flag);
            cmd.arg(value);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;

        let task_id = self.next_id;
        self.next_id += 1;

        // Take stdout/stderr before wrapping child in Arc<Mutex>
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let child_handle = Arc::new(Mutex::new(Some(child)));

        // Spawn stdout reader
        if let Some(stdout) = stdout {
            let tx = event_tx.clone();
            let id = task_id;
            tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if tx
                        .send(AppEvent::TaskOutput {
                            task_id: id,
                            line,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }

        // Spawn stderr reader
        if let Some(stderr) = stderr {
            let tx = event_tx.clone();
            let id = task_id;
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send(AppEvent::TaskOutput {
                        task_id: id,
                        line: format!("[stderr] {}", line),
                    });
                }
            });
        }

        // Spawn exit watcher
        {
            let tx = event_tx;
            let id = task_id;
            let child_for_wait = child_handle.clone();
            tokio::spawn(async move {
                let mut guard = child_for_wait.lock().await;
                if let Some(ref mut c) = *guard {
                    let status = c.wait().await;
                    let code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
                    let _ = tx.send(AppEvent::TaskExited {
                        task_id: id,
                        code,
                    });
                }
            });
        }

        self.tasks.push(Task {
            id: task_id,
            kind,
            status: TaskStatus::Running,
            log_lines: Vec::new(),
            video_meta: None,
            current_frame: 0,
            total_frames: None,
            child: child_handle,
        });

        Ok(task_id)
    }

    pub fn kill_task(&self, task_id: usize) {
        if let Some(task) = self.tasks.iter().find(|t| t.id == task_id) {
            let child = task.child.clone();
            tokio::spawn(async move {
                if let Some(ref mut c) = *child.lock().await {
                    let _ = c.kill().await;
                }
            });
        }
    }

    pub fn kill_all(&self) {
        for task in &self.tasks {
            if task.is_running() {
                let child = task.child.clone();
                tokio::spawn(async move {
                    if let Some(ref mut c) = *child.lock().await {
                        let _ = c.kill().await;
                    }
                });
            }
        }
    }

    pub fn handle_output(&mut self, task_id: usize, line: String) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            let parsed = parser::parse_line(&line);
            match parsed {
                ParsedLine::VideoMeta(meta) => {
                    task.video_meta = Some(meta);
                }
                ParsedLine::Frame(n) => {
                    task.current_frame = n;
                }
                ParsedLine::Complete(total) => {
                    task.total_frames = Some(total);
                    task.current_frame = total;
                }
                ParsedLine::Other => {}
            }
            task.push_log(line);
        }
    }

    pub fn handle_exit(&mut self, task_id: usize, code: i32) {
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == task_id) {
            if code == 0 {
                task.status = TaskStatus::Completed;
            } else {
                task.status = TaskStatus::Failed(format!("Exit code: {}", code));
            }
        }
    }

    pub fn running_count(&self) -> usize {
        self.tasks.iter().filter(|t| t.is_running()).count()
    }

    pub fn completed_count(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, TaskStatus::Completed))
            .count()
    }
}
