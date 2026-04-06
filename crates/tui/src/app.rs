use crate::examples::{self, ExampleKind};
use crate::event::AppEvent;
use crate::task::TaskManager;
use crossterm::event::{KeyCode, KeyEvent};
use std::collections::HashMap;
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq)]
pub enum Screen {
    Menu,
    Config(ExampleKind),
    Dashboard,
}

pub struct MenuState {
    pub selected: usize,
    pub checked: [bool; 4],
}

pub struct ConfigState {
    pub kind: ExampleKind,
    pub field_index: usize,
    pub values: Vec<String>,
    pub editing: bool,
}

pub struct DashboardState {
    pub selected_task: usize,
    pub scroll_offset: u16,
}

pub struct AppState {
    pub screen: Screen,
    pub menu: MenuState,
    pub configs: HashMap<ExampleKind, Vec<String>>,
    pub config_state: Option<ConfigState>,
    pub dashboard: DashboardState,
    pub tasks: TaskManager,
    pub should_quit: bool,
    pub error_message: Option<String>,
}

impl AppState {
    pub fn new() -> Self {
        // Initialize configs with defaults
        let mut configs = HashMap::new();
        for kind in ExampleKind::ALL {
            let params = examples::params_for(kind);
            let values: Vec<String> = params.iter().map(|p| p.default.to_string()).collect();
            configs.insert(kind, values);
        }

        Self {
            screen: Screen::Menu,
            menu: MenuState {
                selected: 0,
                checked: [false; 4],
            },
            configs,
            config_state: None,
            dashboard: DashboardState {
                selected_task: 0,
                scroll_offset: 0,
            },
            tasks: TaskManager::new(),
            should_quit: false,
            error_message: None,
        }
    }

    pub fn update(&mut self, event: AppEvent, event_tx: &mpsc::UnboundedSender<AppEvent>) {
        // Clear error on any key press
        if matches!(event, AppEvent::Key(_)) {
            self.error_message = None;
        }

        match event {
            AppEvent::Key(key) => self.handle_key(key, event_tx),
            AppEvent::TaskOutput { task_id, line } => {
                self.tasks.handle_output(task_id, line);
            }
            AppEvent::TaskExited { task_id, code } => {
                self.tasks.handle_exit(task_id, code);
            }
            AppEvent::Tick => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent, event_tx: &mpsc::UnboundedSender<AppEvent>) {
        match &self.screen {
            Screen::Menu => self.handle_menu_key(key, event_tx),
            Screen::Config(_) => self.handle_config_key(key),
            Screen::Dashboard => self.handle_dashboard_key(key),
        }
    }

    fn handle_menu_key(&mut self, key: KeyEvent, event_tx: &mpsc::UnboundedSender<AppEvent>) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.tasks.kill_all();
                self.should_quit = true;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.menu.selected > 0 {
                    self.menu.selected -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.menu.selected < 3 {
                    self.menu.selected += 1;
                }
            }
            KeyCode::Char(' ') => {
                self.menu.checked[self.menu.selected] =
                    !self.menu.checked[self.menu.selected];
            }
            KeyCode::Enter => {
                let kind = ExampleKind::ALL[self.menu.selected];
                let params = examples::params_for(kind);
                let values = self.configs.get(&kind).cloned().unwrap_or_else(|| {
                    params.iter().map(|p| p.default.to_string()).collect()
                });
                self.config_state = Some(ConfigState {
                    kind,
                    field_index: 0,
                    values,
                    editing: false,
                });
                self.screen = Screen::Config(kind);
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.run_selected(event_tx);
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                if !self.tasks.tasks.is_empty() {
                    self.screen = Screen::Dashboard;
                }
            }
            _ => {}
        }
    }

    fn handle_config_key(&mut self, key: KeyEvent) {
        let state = match self.config_state.as_mut() {
            Some(s) => s,
            None => return,
        };

        let param_count = examples::params_for(state.kind).len();

        if state.editing {
            match key.code {
                KeyCode::Enter | KeyCode::Esc => {
                    state.editing = false;
                }
                KeyCode::Backspace => {
                    if !state.values[state.field_index].is_empty() {
                        state.values[state.field_index].pop();
                    }
                }
                KeyCode::Char(c) => {
                    state.values[state.field_index].push(c);
                }
                _ => {}
            }
        } else {
            match key.code {
                KeyCode::Esc => {
                    // Save values back to configs
                    if let Some(ref cs) = self.config_state {
                        self.configs.insert(cs.kind, cs.values.clone());
                    }
                    self.config_state = None;
                    self.screen = Screen::Menu;
                }
                KeyCode::Enter => {
                    // Save and go back
                    if let Some(ref cs) = self.config_state {
                        self.configs.insert(cs.kind, cs.values.clone());
                    }
                    self.config_state = None;
                    self.screen = Screen::Menu;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if state.field_index > 0 {
                        state.field_index -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                    if state.field_index < param_count - 1 {
                        state.field_index += 1;
                    }
                }
                KeyCode::Char('e') | KeyCode::Char('i') => {
                    state.editing = true;
                }
                _ => {}
            }
        }
    }

    fn handle_dashboard_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.screen = Screen::Menu;
            }
            KeyCode::Tab => {
                if !self.tasks.tasks.is_empty() {
                    self.dashboard.selected_task =
                        (self.dashboard.selected_task + 1) % self.tasks.tasks.len();
                    self.dashboard.scroll_offset = 0;
                }
            }
            KeyCode::Up => {
                if self.dashboard.scroll_offset > 0 {
                    self.dashboard.scroll_offset -= 1;
                }
            }
            KeyCode::Down => {
                self.dashboard.scroll_offset += 1;
            }
            KeyCode::Char('k') | KeyCode::Char('K') => {
                if let Some(task) = self.tasks.tasks.get(self.dashboard.selected_task) {
                    if task.is_running() {
                        self.tasks.kill_task(task.id);
                    }
                }
            }
            _ => {}
        }
    }

    fn run_selected(&mut self, event_tx: &mpsc::UnboundedSender<AppEvent>) {
        let mut any_spawned = false;

        for (i, &checked) in self.menu.checked.iter().enumerate() {
            if !checked {
                continue;
            }
            let kind = ExampleKind::ALL[i];
            let params = examples::params_for(kind);
            let Some(values) = self.configs.get(&kind) else {
                self.error_message = Some(format!("No configuration found for {:?}", kind));
                return;
            };

            // Build flag -> value map
            let mut config_map: HashMap<String, String> = HashMap::new();
            for (param, value) in params.iter().zip(values.iter()) {
                if !value.is_empty() {
                    config_map.insert(param.flag.to_string(), value.clone());
                }
            }

            match self.tasks.spawn(kind, &config_map, event_tx.clone()) {
                Ok(_) => {
                    any_spawned = true;
                }
                Err(e) => {
                    self.error_message = Some(e);
                    return;
                }
            }
        }

        if any_spawned {
            // Uncheck all after spawning
            self.menu.checked = [false; 4];
            self.dashboard.selected_task = self.tasks.tasks.len().saturating_sub(1);
            self.screen = Screen::Dashboard;
        } else {
            self.error_message = Some("No examples selected. Use [Space] to select.".into());
        }
    }
}
