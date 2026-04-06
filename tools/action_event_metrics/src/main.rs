use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Parser)]
#[command(about = "Calculate event-based action detection metrics")]
struct Args {
    /// Path to .json file with dumped tracks
    #[arg(short, long)]
    detections: String,

    /// Path to .xml file with annotation
    #[arg(short, long)]
    annotation: String,

    /// Min action duration (num frames)
    #[arg(long = "min_action_length", default_value_t = 30)]
    min_action_length: i64,

    /// Smooth window size (num frames)
    #[arg(long = "window_size", default_value_t = 30)]
    window_size: i64,
}

const UNDEFINED_ACTION_ID: i32 = 3;

fn action_name_to_id(name: &str) -> Option<i32> {
    match name {
        "sitting" | "listening" | "reading" | "writing" | "lie_on_the_desk" | "busy"
        | "in_group_discussions" => Some(0),
        "standing" => Some(1),
        "raising_hand" => Some(2),
        "__undefined__" => Some(UNDEFINED_ACTION_ID),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct BBoxDesc {
    id: i64,
    label: i32,
    det_conf: f64,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
}

#[derive(Clone, Debug)]
struct MatchDesc {
    gt: BBoxDesc,
    pred: Option<BBoxDesc>,
}

#[derive(Clone, Debug)]
struct Range {
    start: i64,
    end: i64,
    label: i32,
}

type FrameDetections = HashMap<i64, Vec<BBoxDesc>>;

// --- Loading ---

#[derive(Deserialize)]
struct DetectionsFile {
    data: Vec<DetectionRecord>,
}

#[derive(Deserialize)]
struct DetectionRecord {
    frame_id: serde_json::Value,
    label: serde_json::Value,
    det_conf: f64,
    rect: [f64; 4],
}

fn load_detections(file_path: &str) -> FrameDetections {
    let content = fs::read_to_string(file_path).expect("Failed to read detections file");
    let file: DetectionsFile = serde_json::from_str(&content).expect("Failed to parse detections");

    let pb = ProgressBar::new(file.data.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Extracting detections");

    let mut out: FrameDetections = HashMap::new();
    for det in &file.data {
        let frame_id = match &det.frame_id {
            serde_json::Value::Number(n) => n.as_i64().unwrap(),
            serde_json::Value::String(s) => s.parse().unwrap(),
            _ => panic!("Invalid frame_id"),
        };
        let label = match &det.label {
            serde_json::Value::Number(n) => n.as_i64().unwrap() as i32,
            serde_json::Value::String(s) => s.parse().unwrap(),
            _ => panic!("Invalid label"),
        };
        let [x, y, w, h] = det.rect;
        out.entry(frame_id).or_default().push(BBoxDesc {
            id: -1,
            label,
            det_conf: det.det_conf,
            xmin: x,
            ymin: y,
            xmax: x + w,
            ymax: y + h,
        });
        pb.inc(1);
    }
    pb.finish_and_clear();
    out
}

fn load_annotation(file_path: &str) -> FrameDetections {
    let content = fs::read_to_string(file_path).expect("Failed to read annotation file");
    let mut reader = Reader::from_str(&content);

    let mut detections: FrameDetections = HashMap::new();
    let mut ordered_track_id: i64 = -1;
    let mut current_track_id: Option<i64> = None;
    let mut is_person_track = false;

    // Per-box state
    let mut in_box = false;
    let mut box_frame_id: i64 = 0;
    let mut box_xtl: f64 = 0.0;
    let mut box_ytl: f64 = 0.0;
    let mut box_xbr: f64 = 0.0;
    let mut box_ybr: f64 = 0.0;
    let mut box_action: Option<String> = None;
    let mut in_action_attr = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"track" => {
                        is_person_track = false;
                        current_track_id = None;
                        let mut has_id = false;
                        for attr in e.attributes().flatten() {
                            match attr.key.local_name().as_ref() {
                                b"label" => {
                                    if attr.unescape_value().unwrap() == "person" {
                                        is_person_track = true;
                                    }
                                }
                                b"id" => {
                                    has_id = true;
                                    current_track_id = Some(
                                        attr.unescape_value()
                                            .unwrap()
                                            .parse()
                                            .unwrap(),
                                    );
                                }
                                _ => {}
                            }
                        }
                        if is_person_track {
                            ordered_track_id += 1;
                            if !has_id {
                                current_track_id = Some(ordered_track_id);
                            }
                        }
                    }
                    b"box" if is_person_track => {
                        in_box = true;
                        box_action = None;
                        for attr in e.attributes().flatten() {
                            match attr.key.local_name().as_ref() {
                                b"frame" => {
                                    box_frame_id =
                                        attr.unescape_value().unwrap().parse().unwrap();
                                }
                                b"xtl" => {
                                    box_xtl = attr.unescape_value().unwrap().parse().unwrap();
                                }
                                b"ytl" => {
                                    box_ytl = attr.unescape_value().unwrap().parse().unwrap();
                                }
                                b"xbr" => {
                                    box_xbr = attr.unescape_value().unwrap().parse().unwrap();
                                }
                                b"ybr" => {
                                    box_ybr = attr.unescape_value().unwrap().parse().unwrap();
                                }
                                _ => {}
                            }
                        }
                    }
                    b"attribute" if in_box => {
                        for attr in e.attributes().flatten() {
                            if attr.key.local_name().as_ref() == b"name"
                                && attr.unescape_value().unwrap() == "action"
                            {
                                in_action_attr = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_action_attr {
                    box_action = Some(e.unescape().unwrap().trim().to_string());
                    in_action_attr = false;
                }
            }
            Ok(Event::End(ref e)) => {
                let local = e.local_name();
                match local.as_ref() {
                    b"box" if in_box => {
                        in_box = false;
                        if box_frame_id <= 0 {
                            continue;
                        }
                        if let Some(ref action) = box_action {
                            if let Some(label) = action_name_to_id(action) {
                                let track_id = current_track_id.unwrap();
                                detections.entry(box_frame_id).or_default().push(BBoxDesc {
                                    id: track_id,
                                    label,
                                    det_conf: 1.0,
                                    xmin: box_xtl,
                                    ymin: box_ytl,
                                    xmax: box_xbr,
                                    ymax: box_ybr,
                                });
                            }
                        }
                    }
                    b"track" => {
                        is_person_track = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => panic!("XML parse error: {e}"),
            _ => {}
        }
    }

    println!("Loaded {} annotated frames.", detections.len());
    detections
}

// --- Matching ---

fn iou(a: &BBoxDesc, b: &BBoxDesc) -> f64 {
    let ix = (a.xmax.min(b.xmax) - a.xmin.max(b.xmin)).max(0.0);
    let iy = (a.ymax.min(b.ymax) - a.ymin.max(b.ymin)).max(0.0);
    let inter = ix * iy;
    let union = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin)
        - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

fn match_detections(
    predicted: &FrameDetections,
    gt: &FrameDetections,
    min_iou: f64,
) -> HashMap<i64, Vec<(usize, usize)>> {
    let mut all_matches: HashMap<i64, Vec<(usize, usize)>> = HashMap::new();
    let mut total_gt = 0usize;
    let mut matched_gt = 0usize;

    let pb = ProgressBar::new(gt.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Matching detections");

    for &frame_id in gt.keys() {
        let gt_bboxes = &gt[&frame_id];
        total_gt += gt_bboxes.len();

        let pred_bboxes = match predicted.get(&frame_id) {
            Some(p) => p,
            None => {
                all_matches.insert(frame_id, vec![]);
                pb.inc(1);
                continue;
            }
        };

        let mut sorted: Vec<(usize, &BBoxDesc)> = pred_bboxes.iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.det_conf.partial_cmp(&a.1.det_conf).unwrap());

        let mut visited = vec![false; gt_bboxes.len()];
        let mut matches = Vec::new();

        for &(pred_idx, pred_bbox) in &sorted {
            let mut best_overlap = 0.0;
            let mut best_gt = None;
            for (gt_idx, gt_bbox) in gt_bboxes.iter().enumerate() {
                if visited[gt_idx] {
                    continue;
                }
                let ov = iou(pred_bbox, gt_bbox);
                if ov > best_overlap {
                    best_overlap = ov;
                    best_gt = Some(gt_idx);
                }
            }
            if let Some(gt_idx) = best_gt {
                if best_overlap > min_iou {
                    visited[gt_idx] = true;
                    matches.push((gt_idx, pred_idx));
                    matched_gt += 1;
                    if matches.len() >= gt_bboxes.len() {
                        break;
                    }
                }
            }
        }

        all_matches.insert(frame_id, matches);
        pb.inc(1);
    }
    pb.finish_and_clear();

    println!(
        "Matched gt bbox: {} / {} ({:.2}%)",
        matched_gt,
        total_gt,
        100.0 * matched_gt as f64 / total_gt.max(1) as f64
    );

    all_matches
}

// --- Tracks ---

type Tracks = HashMap<i64, HashMap<i64, MatchDesc>>;

fn split_to_tracks(gt_data: &FrameDetections) -> Tracks {
    let mut tracks: Tracks = HashMap::new();
    for (&frame_id, bboxes) in gt_data {
        for bbox in bboxes {
            let md = MatchDesc {
                gt: bbox.clone(),
                pred: None,
            };
            tracks
                .entry(bbox.id)
                .or_default()
                .insert(frame_id, md);
        }
    }
    tracks
}

fn add_matched_predictions(
    tracks: &mut Tracks,
    all_matches: &HashMap<i64, Vec<(usize, usize)>>,
    predicted: &FrameDetections,
    gt: &FrameDetections,
) {
    for (&frame_id, matches) in all_matches {
        if matches.is_empty() {
            continue;
        }
        let gt_frame = &gt[&frame_id];
        let pred_frame = &predicted[&frame_id];
        for &(gt_idx, pred_idx) in matches {
            let track_id = gt_frame[gt_idx].id;
            if let Some(track) = tracks.get_mut(&track_id) {
                if let Some(md) = track.get_mut(&frame_id) {
                    md.pred = Some(pred_frame[pred_idx].clone());
                }
            }
        }
    }
}

// --- Event extraction ---

fn extract_events(
    frame_events: &[(i64, i32)],
    window_size: i64,
    min_length: i64,
    frame_limits: (i64, i64),
) -> Vec<Range> {
    // Smooth: merge frames into contiguous ranges
    let mut events: Vec<Range> = Vec::new();
    if !frame_events.is_empty() {
        let mut last = Range {
            start: frame_events[0].0,
            end: frame_events[0].0 + 1,
            label: frame_events[0].1,
        };
        for &(f, l) in &frame_events[1..] {
            if last.end + window_size - 1 >= f && last.label == l {
                last.end = f + 1;
            } else {
                events.push(last);
                last = Range {
                    start: f,
                    end: f + 1,
                    label: l,
                };
            }
        }
        events.push(last);
    }

    // Filter: remove short events
    events.retain(|e| e.end - e.start >= min_length);

    // Extrapolate: extend to frame limits
    events = match events.len() {
        0 => vec![],
        1 => vec![Range {
            start: frame_limits.0,
            end: frame_limits.1,
            label: events[0].label,
        }],
        n => {
            events[0].start = frame_limits.0;
            events[n - 1].end = frame_limits.1;
            events
        }
    };

    // Interpolate: fill gaps between events
    if events.len() > 1 {
        let mut interpolated = Vec::new();
        let mut last = events[0].clone();
        for ev in &events[1..] {
            let mid = (last.end + ev.start) / 2;
            last.end = mid;
            let mut cur = ev.clone();
            cur.start = mid;
            interpolated.push(last);
            last = cur;
        }
        interpolated.push(last);
        events = interpolated;
    }

    // Merge: consecutive events with same label
    if events.len() > 1 {
        let mut merged = Vec::new();
        let mut last = events[0].clone();
        for ev in &events[1..] {
            if last.end == ev.start && last.label == ev.label {
                last.end = ev.end;
            } else {
                merged.push(last);
                last = ev.clone();
            }
        }
        merged.push(last);
        events = merged;
    }

    events
}

fn match_events(gt_events: &[Range], pred_events: &[Range]) -> Vec<(usize, usize)> {
    if gt_events.is_empty() || pred_events.is_empty() {
        return vec![];
    }

    let mut matches = Vec::new();
    for (pred_id, pred) in pred_events.iter().enumerate() {
        let mut best_overlap: i64 = 0;
        let mut best_gt_id: Option<usize> = None;
        for (gt_id, gt) in gt_events.iter().enumerate() {
            let intersect = (gt.end.min(pred.end) - gt.start.max(pred.start)).max(0);
            let overlap = if gt.label != pred.label { 0 } else { intersect };
            if overlap > best_overlap {
                best_overlap = overlap;
                best_gt_id = Some(gt_id);
            }
        }
        if best_overlap > 0 {
            if let Some(gt_id) = best_gt_id {
                matches.push((gt_id, pred_id));
            }
        }
    }
    matches
}

fn process_tracks(
    all_tracks: &Tracks,
    window_size: i64,
    min_length: i64,
) -> HashMap<i64, (Vec<Range>, Vec<Range>)> {
    let mut out = HashMap::new();

    let pb = ProgressBar::new(all_tracks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap(),
    );
    pb.set_message("Extracting events");

    for (&track_id, track) in all_tracks {
        let mut frame_ids: Vec<i64> = track.keys().copied().collect();
        frame_ids.sort();
        let frame_limits = (*frame_ids.first().unwrap(), *frame_ids.last().unwrap() + 1);

        let gt_frame_events: Vec<(i64, i32)> = frame_ids
            .iter()
            .filter(|fi| track[fi].gt.label != UNDEFINED_ACTION_ID)
            .map(|fi| (*fi, track[fi].gt.label))
            .collect();

        let pred_frame_events: Vec<(i64, i32)> = frame_ids
            .iter()
            .filter(|fi| track[fi].pred.is_some())
            .map(|fi| (*fi, track[fi].pred.as_ref().unwrap().label))
            .collect();

        if gt_frame_events.is_empty() || pred_frame_events.is_empty() {
            pb.inc(1);
            continue;
        }

        let gt_events = extract_events(&gt_frame_events, window_size, min_length, frame_limits);
        let pred_events =
            extract_events(&pred_frame_events, window_size, min_length, frame_limits);

        out.insert(track_id, (gt_events, pred_events));
        pb.inc(1);
    }
    pb.finish_and_clear();
    out
}

fn calculate_metrics(all_tracks: &HashMap<i64, (Vec<Range>, Vec<Range>)>) -> (f64, f64) {
    let mut total_pred = 0usize;
    let mut valid_pred = 0usize;
    let mut total_gt = 0usize;
    let mut valid_gt = 0usize;

    for (gt_events, pred_events) in all_tracks.values() {
        let matches = match_events(gt_events, pred_events);
        total_pred += pred_events.len();
        total_gt += gt_events.len();

        if !matches.is_empty() {
            let mut matched_gt = vec![false; gt_events.len()];
            let mut matched_pred = vec![false; pred_events.len()];
            for &(gi, pi) in &matches {
                matched_gt[gi] = true;
                matched_pred[pi] = true;
            }
            valid_pred += matched_pred.iter().filter(|&&v| v).count();
            valid_gt += matched_gt.iter().filter(|&&v| v).count();
        }
    }

    let precision = if total_pred > 0 {
        valid_pred as f64 / total_pred as f64
    } else {
        0.0
    };
    let recall = if total_gt > 0 {
        valid_gt as f64 / total_gt as f64
    } else {
        0.0
    };

    (precision, recall)
}

fn main() {
    let args = Args::parse();

    assert!(
        std::path::Path::new(&args.detections).exists(),
        "Detections file not found: {}",
        args.detections
    );
    assert!(
        std::path::Path::new(&args.annotation).exists(),
        "Annotation file not found: {}",
        args.annotation
    );
    assert!(args.min_action_length > 0);
    assert!(args.window_size > 0);

    let detections = load_detections(&args.detections);
    let annotation = load_annotation(&args.annotation);

    let all_matches = match_detections(&detections, &annotation, 0.5);

    let mut tracks = split_to_tracks(&annotation);
    println!("Found {} tracks.", tracks.len());

    add_matched_predictions(&mut tracks, &all_matches, &detections, &annotation);

    let track_events = process_tracks(&tracks, args.window_size, args.min_action_length);

    let (precision, recall) = calculate_metrics(&track_events);
    println!(
        "\nPrecision: {:.3}%   Recall: {:.3}%",
        1e2 * precision,
        1e2 * recall
    );
}
