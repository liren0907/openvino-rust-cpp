use clap::Parser;
use ovi_vision::{AnnotatedFrame, Color, Core, FaceDetector, Video, VideoWriter};

#[derive(Parser)]
#[command(name = "example_face_detection", about = "Face Detection Example")]
struct Args {
    /// Input video file path
    #[arg(short, long)]
    input: String,

    /// Face detection model (.xml)
    #[arg(long)]
    m_fd: String,

    /// Face detection confidence threshold
    #[arg(long, default_value_t = 0.6)]
    t_fd: f32,

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

    println!("Loading face detector: {}", args.m_fd);
    let mut face_detector = FaceDetector::builder(&args.m_fd)
        .confidence(args.t_fd)
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
        let faces = face_detector.detect_frame(&frame)?;

        if let Some(ref mut w) = writer {
            let mut annotated = AnnotatedFrame::from_frame(&frame)?;
            for face in &faces {
                annotated.draw_detection(
                    face.x,
                    face.y,
                    face.width,
                    face.height,
                    &format!("{:.0}%", face.confidence * 100.0),
                    Color::GREEN,
                )?;
            }
            w.write(&annotated)?;
        }

        drop(frame);

        println!("Frame {}: {} faces detected", frame_num, faces.len());
        for (i, face) in faces.iter().enumerate() {
            println!(
                "  Face {}: ({}, {}) {}x{} conf={:.2}",
                i, face.x, face.y, face.width, face.height, face.confidence
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
