use clap::Parser;
use openvino_vision::{ActionDetector, AnnotatedFrame, Color, Core, Video, VideoWriter};

#[derive(Parser)]
#[command(name = "example_action_detection", about = "Action Detection Example")]
struct Args {
    /// Input video file path
    #[arg(short, long)]
    input: String,

    /// Action detection model (.xml)
    #[arg(long)]
    m_act: String,

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

fn main() -> openvino_vision::Result<()> {
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

    println!("Loading action detector: {}", args.m_act);
    let mut action_detector = ActionDetector::builder(&args.m_act)
        .det_threshold(args.t_ad)
        .act_threshold(args.t_ar)
        .num_actions(args.num_actions)
        .build(&core)?;

    let mut writer = args
        .output
        .as_ref()
        .map(|path| VideoWriter::new(path, video.fps()))
        .transpose()?;

    if let Some(ref path) = args.output {
        println!("Output video: {}", path);
    }

    println!("Starting processing...\n");
    let mut frame_num: i32 = 0;

    loop {
        let frame = video.current_frame()?;
        let actions = action_detector.detect_frame(&frame)?;

        if let Some(ref mut w) = writer {
            let action_names = ["sitting", "standing", "raising_hand", "turned", "writing", "unknown"];
            let mut annotated = AnnotatedFrame::from_frame(&frame)?;
            for action in &actions {
                let name = action_names
                    .get(action.label as usize)
                    .unwrap_or(&"?");
                annotated.draw_detection(
                    action.x,
                    action.y,
                    action.width,
                    action.height,
                    &format!("{} {:.0}%", name, action.detection_conf * 100.0),
                    Color::CYAN,
                )?;
            }
            w.write(&annotated)?;
        }

        drop(frame);

        println!("Frame {}: {} actions detected", frame_num, actions.len());
        for (i, action) in actions.iter().enumerate() {
            println!(
                "  Action {}: ({}, {}) {}x{} det_conf={:.2} label={}",
                i, action.x, action.y, action.width, action.height, action.detection_conf, action.label
            );
        }

        if !video.read()? {
            break;
        }
        frame_num += 1;
    }

    println!("\nProcessing complete. {} frames processed.", frame_num + 1);
    Ok(())
}
