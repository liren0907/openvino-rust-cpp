use clap::Parser;
use openvino_vision::{
    ActionDetector, AnnotatedFrame, Color, Core, Detection, FaceDetector, Tracker, TrackerConfig,
    Video, VideoWriter,
};

#[derive(Parser)]
#[command(name = "example_tracking", about = "Detection + Tracking Example")]
struct Args {
    /// Input video file path
    #[arg(short, long)]
    input: String,

    /// Detection mode: "face" or "action"
    #[arg(long, default_value = "face")]
    mode: String,

    /// Face detection model (.xml) — required when --mode=face
    #[arg(long)]
    m_fd: Option<String>,

    /// Action detection model (.xml) — required when --mode=action
    #[arg(long)]
    m_act: Option<String>,

    /// Face detection confidence threshold
    #[arg(long, default_value_t = 0.6)]
    t_fd: f32,

    /// Action detection confidence threshold
    #[arg(long, default_value_t = 0.3)]
    t_ad: f32,

    /// Action recognition confidence threshold
    #[arg(long, default_value_t = 0.75)]
    t_ar: f32,

    /// Number of action classes (must match model: 0005=3, 0006=6)
    #[arg(long, default_value_t = 3)]
    num_actions: u32,

    /// Output video file path (annotated .mp4)
    #[arg(short, long)]
    output: Option<String>,

    /// Headless mode (no display window)
    #[arg(long, default_value_t = false)]
    no_show: bool,

    /// Maximum frames to read (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    read_limit: u32,
}

enum DetectorMode {
    Face(FaceDetector),
    Action(ActionDetector),
}

impl DetectorMode {
    fn enqueue(&mut self, frame: &openvino_vision::Frame<'_>) -> openvino_vision::Result<()> {
        match self {
            DetectorMode::Face(d) => d.enqueue(frame),
            DetectorMode::Action(d) => d.enqueue(frame),
        }
    }

    fn fetch_results(&mut self) -> openvino_vision::Result<Vec<Detection>> {
        match self {
            DetectorMode::Face(d) => d.fetch_results(),
            DetectorMode::Action(d) => {
                let actions = d.fetch_results()?;
                Ok(actions
                    .iter()
                    .map(|a| Detection {
                        x: a.x,
                        y: a.y,
                        width: a.width,
                        height: a.height,
                        confidence: a.detection_conf,
                        label: a.label,
                    })
                    .collect())
            }
        }
    }
}

fn main() -> openvino_vision::Result<()> {
    let args = Args::parse();

    // Validate mode and model args
    match args.mode.as_str() {
        "face" => {
            if args.m_fd.is_none() {
                eprintln!("Error: --m-fd is required when --mode=face");
                std::process::exit(1);
            }
        }
        "action" => {
            if args.m_act.is_none() {
                eprintln!("Error: --m-act is required when --mode=action");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!(
                "Error: unknown mode \"{}\". Use \"face\" or \"action\".",
                other
            );
            std::process::exit(1);
        }
    }

    println!("Initializing OpenVINO Core...");
    let core = Core::new()?;

    println!("Opening video: {}", args.input);
    let mut video = Video::open_with_options(&args.input, false, args.read_limit)?;
    println!(
        "Video: {}x{} @ {:.1} FPS",
        video.width(),
        video.height(),
        video.fps()
    );

    // Create detector based on mode
    let mut detector = match args.mode.as_str() {
        "face" => {
            let model = args.m_fd.as_ref().unwrap();
            println!("Loading face detector: {}", model);
            let d = FaceDetector::builder(model)
                .confidence(args.t_fd)
                .build(&core)?;
            DetectorMode::Face(d)
        }
        "action" => {
            let model = args.m_act.as_ref().unwrap();
            println!("Loading action detector: {}", model);
            let d = ActionDetector::builder(model)
                .det_threshold(args.t_ad)
                .act_threshold(args.t_ar)
                .num_actions(args.num_actions)
                .build(&core)?;
            DetectorMode::Action(d)
        }
        _ => unreachable!(),
    };

    // Create tracker with mode-appropriate config
    let tracker_config = match args.mode.as_str() {
        "face" => TrackerConfig {
            min_track_duration: 1,
            forget_delay: 150,
            affinity_thr: 0.8,
            bbox_heights_range: (10.0, 1080.0),
            drop_forgotten_tracks: false,
            ..TrackerConfig::default()
        },
        "action" => TrackerConfig {
            min_track_duration: 8,
            forget_delay: 150,
            affinity_thr: 0.9,
            averaging_window_size_for_rects: 5,
            bbox_heights_range: (10.0, 2160.0),
            drop_forgotten_tracks: false,
            ..TrackerConfig::default()
        },
        _ => unreachable!(),
    };
    let mut tracker = Tracker::new(&tracker_config)?;

    let mut writer = args
        .output
        .as_ref()
        .map(|path| VideoWriter::new(path, video.fps()))
        .transpose()?;

    if let Some(ref path) = args.output {
        println!("Output video: {}", path);
    }

    println!("Mode: {} detection + tracking", args.mode);
    println!("Starting processing (async double-buffering)...\n");
    let mut frame_num: i32 = 0;

    // Enqueue first frame for async detection
    {
        let frame = video.current_frame()?;
        detector.enqueue(&frame)?;
    }

    loop {
        // Fetch detection results (blocks until inference completes)
        let detections = detector.fetch_results()?;

        // Track detections on current frame
        let frame = video.current_frame()?;
        let tracked = tracker.process_frame(&detections, frame_num, &frame)?;

        if let Some(ref mut w) = writer {
            let mut annotated = AnnotatedFrame::from_frame(&frame)?;
            for obj in &tracked {
                annotated.draw_detection(
                    obj.x,
                    obj.y,
                    obj.width,
                    obj.height,
                    &format!("#{} {:.0}%", obj.object_id, obj.confidence * 100.0),
                    Color::YELLOW,
                )?;
            }
            w.write(&annotated)?;
        }

        drop(frame);

        // Read next frame and enqueue async detection
        let has_next = video.read()?;
        if has_next {
            let next_frame = video.current_frame()?;
            detector.enqueue(&next_frame)?;
        }

        // Print results
        println!("Frame {}: {} objects tracked", frame_num, tracked.len());
        for obj in &tracked {
            println!(
                "  Track #{}: ({}, {}) {}x{} conf={:.2}",
                obj.object_id, obj.x, obj.y, obj.width, obj.height, obj.confidence
            );
        }

        if !has_next {
            break;
        }
        frame_num += 1;
    }

    println!("\nProcessing complete. {} frames processed.", frame_num + 1);
    Ok(())
}
