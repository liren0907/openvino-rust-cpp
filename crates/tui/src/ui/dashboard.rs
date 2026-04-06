use crate::app::AppState;
use crate::task::TaskStatus;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph};
use ratatui::Frame;

pub fn render(f: &mut Frame, area: Rect, state: &AppState) {
    let main_chunks = Layout::horizontal([
        Constraint::Percentage(35),
        Constraint::Percentage(65),
    ])
    .split(area);

    render_task_list(f, main_chunks[0], state);
    render_log_panel(f, main_chunks[1], state);
}

fn render_task_list(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::vertical([
        Constraint::Min(5),    // task list with progress
        Constraint::Length(3), // help
    ])
    .split(area);

    let items: Vec<ListItem> = state
        .tasks
        .tasks
        .iter()
        .enumerate()
        .map(|(i, task)| {
            let is_selected = i == state.dashboard.selected_task;
            let marker = if is_selected { ">" } else { " " };

            let status_icon = match &task.status {
                TaskStatus::Running => "...",
                TaskStatus::Completed => " ok",
                TaskStatus::Failed(_) => "ERR",
            };

            let status_color = match &task.status {
                TaskStatus::Running => Color::Yellow,
                TaskStatus::Completed => Color::Green,
                TaskStatus::Failed(_) => Color::Red,
            };

            let style = if is_selected {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(vec![
                Span::styled(format!(" {} ", marker), style),
                Span::styled(format!("#{} ", task.id), Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:16}", task.kind.display_name()),
                    style,
                ),
                Span::styled(
                    format!("[{}]", status_icon),
                    Style::default().fg(status_color),
                ),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Tasks ")
            .borders(Borders::ALL),
    );
    f.render_widget(list, chunks[0]);

    // Render progress gauge for selected task
    let help = if state.tasks.tasks.is_empty() {
        Paragraph::new(" No tasks. [Esc] Back to menu")
    } else {
        let selected = state
            .tasks
            .tasks
            .get(state.dashboard.selected_task);
        if let Some(task) = selected {
            let gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL))
                .gauge_style(Style::default().fg(Color::Cyan))
                .ratio(task.progress_ratio())
                .label(task.progress_text());
            f.render_widget(gauge, chunks[1]);
            return;
        }
        Paragraph::new(Line::from(vec![
            Span::styled("[Tab]", Style::default().fg(Color::Green)),
            Span::raw(" Switch  "),
            Span::styled("[Esc]", Style::default().fg(Color::Green)),
            Span::raw(" Menu "),
        ]))
    };
    f.render_widget(
        help.block(Block::default().borders(Borders::ALL)),
        chunks[1],
    );
}

fn render_log_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let task = state
        .tasks
        .tasks
        .get(state.dashboard.selected_task);

    let (title, lines) = match task {
        Some(task) => {
            let title = format!(
                " Log: {} (#{}) ",
                task.kind.display_name(),
                task.id
            );
            let visible_height = area.height.saturating_sub(2) as usize; // borders
            let total_lines = task.log_lines.len();
            let scroll = state.dashboard.scroll_offset as usize;

            // Auto-scroll to bottom if not manually scrolled
            let start = if scroll == 0 {
                total_lines.saturating_sub(visible_height)
            } else {
                total_lines.saturating_sub(visible_height + scroll)
            };

            let lines: Vec<Line> = task.log_lines[start..]
                .iter()
                .take(visible_height)
                .map(|l| {
                    if l.starts_with("[stderr]") {
                        Line::from(Span::styled(l.as_str(), Style::default().fg(Color::Red)))
                    } else if l.starts_with("Frame") {
                        Line::from(Span::styled(l.as_str(), Style::default().fg(Color::White)))
                    } else if l.starts_with("  ") {
                        Line::from(Span::styled(l.as_str(), Style::default().fg(Color::DarkGray)))
                    } else {
                        Line::from(Span::styled(l.as_str(), Style::default().fg(Color::Cyan)))
                    }
                })
                .collect();
            (title, lines)
        }
        None => (" Log ".to_string(), vec![Line::from(" No task selected")]),
    };

    let log = Paragraph::new(lines).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL),
    );
    f.render_widget(log, area);
}
