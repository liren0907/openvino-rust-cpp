use crate::app::AppState;
use crate::examples;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Frame;

pub fn render(f: &mut Frame, area: Rect, state: &AppState) {
    let config_state = match &state.config_state {
        Some(cs) => cs,
        None => return,
    };

    let params = examples::params_for(config_state.kind);

    let chunks = Layout::vertical([
        Constraint::Length(3), // title
        Constraint::Min(5),   // form
        Constraint::Length(3), // help
    ])
    .split(area);

    // Title
    let title = Paragraph::new(format!(
        " Configure: {} ",
        config_state.kind.display_name()
    ))
    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Form fields
    let items: Vec<ListItem> = params
        .iter()
        .enumerate()
        .map(|(i, param)| {
            let is_selected = i == config_state.field_index;
            let is_editing = is_selected && config_state.editing;

            let req_marker = if param.required { "*" } else { " " };
            let label = format!(" {} {:20}", req_marker, param.label);

            let value = &config_state.values[i];
            let value_display = if is_editing {
                format!("{}|", value)
            } else {
                value.clone()
            };

            let style = if is_selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            let value_style = if is_editing {
                Style::default().fg(Color::White).bg(Color::DarkGray)
            } else if is_selected {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Gray)
            };

            ListItem::new(Line::from(vec![
                Span::styled(label, style),
                Span::raw(" "),
                Span::styled(value_display, value_style),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Parameters ")
            .borders(Borders::ALL),
    );
    f.render_widget(list, chunks[1]);

    // Help bar
    let help_text = if config_state.editing {
        Line::from(vec![
            Span::raw(" Editing: "),
            Span::styled("[Enter/Esc]", Style::default().fg(Color::Green)),
            Span::raw(" Done  "),
            Span::styled("[Backspace]", Style::default().fg(Color::Green)),
            Span::raw(" Delete  "),
            Span::raw(" * = required "),
        ])
    } else {
        Line::from(vec![
            Span::styled("[i/e]", Style::default().fg(Color::Green)),
            Span::raw(" Edit field  "),
            Span::styled("[Tab/j/k]", Style::default().fg(Color::Green)),
            Span::raw(" Navigate  "),
            Span::styled("[Enter]", Style::default().fg(Color::Green)),
            Span::raw(" Accept  "),
            Span::styled("[Esc]", Style::default().fg(Color::Green)),
            Span::raw(" Back  "),
            Span::raw(" * = required "),
        ])
    };

    let help = Paragraph::new(help_text).block(Block::default().borders(Borders::ALL));
    f.render_widget(help, chunks[2]);
}
