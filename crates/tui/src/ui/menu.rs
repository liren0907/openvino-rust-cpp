use crate::app::AppState;
use crate::examples::ExampleKind;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Frame;

pub fn render(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::vertical([
        Constraint::Length(3),  // title
        Constraint::Min(8),    // list
        Constraint::Length(3), // status bar
    ])
    .split(area);

    // Title
    let title = Paragraph::new(" OpenVINO Vision TUI ")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Example list
    let items: Vec<ListItem> = ExampleKind::ALL
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let checked = if state.menu.checked[i] { "x" } else { " " };
            let marker = if i == state.menu.selected {
                ">"
            } else {
                " "
            };
            let style = if i == state.menu.selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {} [{}] ", marker, checked), style),
                Span::styled(kind.display_name(), style),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Select Examples ")
            .borders(Borders::ALL),
    );
    f.render_widget(list, chunks[1]);

    // Status bar
    let running = state.tasks.running_count();
    let completed = state.tasks.completed_count();
    let status_text = if let Some(ref err) = state.error_message {
        Line::from(Span::styled(
            format!(" {} ", err),
            Style::default().fg(Color::Red),
        ))
    } else {
        Line::from(vec![
            Span::raw(format!(
                " {} running, {} completed  ",
                running, completed
            )),
            Span::styled("[Enter]", Style::default().fg(Color::Green)),
            Span::raw(" Configure  "),
            Span::styled("[Space]", Style::default().fg(Color::Green)),
            Span::raw(" Toggle  "),
            Span::styled("[R]", Style::default().fg(Color::Green)),
            Span::raw(" Run  "),
            Span::styled("[D]", Style::default().fg(Color::Green)),
            Span::raw(" Dashboard  "),
            Span::styled("[Q]", Style::default().fg(Color::Green)),
            Span::raw(" Quit "),
        ])
    };

    let status =
        Paragraph::new(status_text).block(Block::default().borders(Borders::ALL));
    f.render_widget(status, chunks[2]);
}
