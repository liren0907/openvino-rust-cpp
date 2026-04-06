use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug)]
#[allow(dead_code)]
pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

#[derive(Debug)]
pub enum ParsedLine {
    VideoMeta(VideoMeta),
    Frame(u32),
    Complete(u32),
    Other,
}

fn video_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^Video: (\d+)x(\d+) @ ([\d.]+) FPS").unwrap())
}

fn frame_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^Frame (\d+):").unwrap())
}

fn complete_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^Processing complete\. (\d+) frames processed\.").unwrap())
}

pub fn parse_line(line: &str) -> ParsedLine {
    if let Some(caps) = video_re().captures(line) {
        return ParsedLine::VideoMeta(VideoMeta {
            width: caps[1].parse().unwrap_or(0),
            height: caps[2].parse().unwrap_or(0),
            fps: caps[3].parse().unwrap_or(0.0),
        });
    }
    if let Some(caps) = complete_re().captures(line) {
        return ParsedLine::Complete(caps[1].parse().unwrap_or(0));
    }
    if let Some(caps) = frame_re().captures(line) {
        return ParsedLine::Frame(caps[1].parse().unwrap_or(0));
    }
    ParsedLine::Other
}
