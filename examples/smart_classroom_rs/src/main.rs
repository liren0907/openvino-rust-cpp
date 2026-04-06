use clap::Parser;
use ovi_vision::{
    ActionDetector, Core, Detection, FaceDetector, FaceGallery, Tracker, TrackerConfig, Video,
};

#[derive(Parser)]
#[command(name = "smart_classroom_rs", about = "Smart Classroom Rust Example")]
struct Args {
    /// Input video file path
    #[arg(short, long)]
    input: String,

    /// Face detection model (.xml)
    #[arg(long)]
    m_fd: String,

    /// Action detection model (.xml)
    #[arg(long)]
    m_act: String,

    /// Landmarks model (.xml)
    #[arg(long)]
    m_lm: String,

    /// Face re-identification model (.xml)
    #[arg(long)]
    m_reid: String,

    /// Face gallery JSON path
    #[arg(long, default_value = "")]
    fg: String,

    /// Face detection confidence threshold
    #[arg(long, default_value_t = 0.6)]
    t_fd: f32,

    /// Action detection confidence threshold
    #[arg(long, default_value_t = 0.3)]
    t_ad: f32,

    /// Action recognition confidence threshold
    #[arg(long, default_value_t = 0.75)]
    t_ar: f32,

    /// Headless mode (no display window)
    #[arg(long, default_value_t = false)]
    no_show: bool,

    /// Maximum frames to read (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    read_limit: u32,
}

fn main() -> ovi_vision::Result<()> {
    let args = Args::parse();

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

    println!("Loading face detector...");
    let mut face_detector = FaceDetector::builder(&args.m_fd)
        .confidence(args.t_fd)
        .build(&core)?;

    println!("Loading action detector...");
    let mut action_detector = ActionDetector::builder(&args.m_act)
        .det_threshold(args.t_ad)
        .act_threshold(args.t_ar)
        .build(&core)?;

    // Set up trackers
    let face_tracker_config = TrackerConfig {
        min_track_duration: 1,
        forget_delay: 150,
        affinity_thr: 0.8,
        bbox_heights_range: (10.0, 1080.0),
        drop_forgotten_tracks: false,
        max_num_objects_in_track: i32::MAX,
        averaging_window_size_for_labels: i32::MAX,
        ..TrackerConfig::default()
    };
    let mut face_tracker = Tracker::new(&face_tracker_config)?;

    let action_tracker_config = TrackerConfig {
        min_track_duration: 8,
        forget_delay: 150,
        affinity_thr: 0.9,
        averaging_window_size_for_rects: 5,
        bbox_heights_range: (10.0, 2160.0),
        drop_forgotten_tracks: false,
        max_num_objects_in_track: i32::MAX,
        ..TrackerConfig::default()
    };
    let mut action_tracker = Tracker::new(&action_tracker_config)?;

    // Optional: face gallery
    let mut gallery = if !args.fg.is_empty() && !args.m_lm.is_empty() && !args.m_reid.is_empty() {
        println!("Loading face gallery: {}", args.fg);
        let g = FaceGallery::builder(&args.fg, &args.m_fd, &args.m_lm, &args.m_reid)
            .build(&core)?;
        println!("Gallery loaded: {} identities", g.size());
        Some(g)
    } else {
        println!("Face gallery disabled (no -fg or models)");
        None
    };

    println!("Starting processing loop...");
    let mut frame_num: i32 = 0;

    // Enqueue first frame for async detection (both detectors run concurrently)
    {
        let frame = video.current_frame()?;
        face_detector.enqueue(&frame)?;
        action_detector.enqueue(&frame)?;
    } // frame dropped — video.read() is now allowed

    loop {
        // Fetch detection results (blocks until inference completes)
        let faces = face_detector.fetch_results()?;
        let actions = action_detector.fetch_results()?;

        // Get current frame for identification and tracking
        let frame = video.current_frame()?;

        // Identify faces BEFORE tracking — attach gallery IDs as labels
        let face_detections_with_ids: Vec<Detection> = if let Some(ref mut g) = gallery {
            let ids = g.identify_frame(&frame, &faces)?;
            faces
                .iter()
                .zip(ids.iter())
                .map(|(f, &id)| Detection { label: id, ..*f })
                .collect()
        } else {
            faces
        };

        // Track faces (labels are preserved by the tracker)
        let tracked_faces =
            face_tracker.process_frame(&face_detections_with_ids, frame_num, &frame)?;

        // Track actions (convert to Detection format)
        let action_detections: Vec<Detection> = actions
            .iter()
            .map(|a| Detection {
                x: a.x,
                y: a.y,
                width: a.width,
                height: a.height,
                confidence: a.detection_conf,
                label: a.label,
            })
            .collect();
        let tracked_actions =
            action_tracker.process_frame(&action_detections, frame_num, &frame)?;

        // Done with this frame's data
        drop(frame);

        // Read next frame and enqueue async detection before printing
        // (inference runs in the background while we do I/O below)
        let has_next = video.read()?;
        if has_next {
            let next_frame = video.current_frame()?;
            face_detector.enqueue(&next_frame)?;
            action_detector.enqueue(&next_frame)?;
        }

        // Print identified faces from tracked results
        if let Some(ref g) = gallery {
            for face in &tracked_faces {
                if face.label >= 0 {
                    let label = g.label(face.label)?;
                    println!(
                        "Frame {}: face #{} at ({},{}) = {}",
                        frame_num, face.object_id, face.x, face.y, label
                    );
                }
            }
        }

        // Print summary
        println!(
            "Frame {}: {} faces, {} actions tracked",
            frame_num,
            tracked_faces.len(),
            tracked_actions.len()
        );

        if !has_next {
            println!("End of video.");
            break;
        }
        frame_num += 1;
    }

    println!("Processing complete. {} frames processed.", frame_num + 1);
    Ok(())
}
