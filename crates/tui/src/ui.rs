pub mod menu;
pub mod config;
pub mod dashboard;

use crate::app::{AppState, Screen};
use ratatui::Frame;

pub fn render(f: &mut Frame, state: &AppState) {
    let area = f.area();
    match state.screen {
        Screen::Menu => menu::render(f, area, state),
        Screen::Config(_) => config::render(f, area, state),
        Screen::Dashboard => dashboard::render(f, area, state),
    }
}
