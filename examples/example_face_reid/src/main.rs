use clap::Parser;
use openvino_vision::{AnnotatedFrame, Color, Core, FaceDetector, FaceGallery, Video, VideoWriter};

#[derive(Parser)]
#[command(name = "example_face_reid", about = "Face Re-Identification Example")]
struct Args {
    /// Input video file path
    #[arg(short, long)]
    input: String,

    /// Face detection model (.xml)
    #[arg(long)]
    m_fd: String,

    /// Landmarks model (.xml)
    #[arg(long)]
    m_lm: String,

    /// Face re-identification model (.xml)
    #[arg(long)]
    m_reid: String,

    /// Face gallery JSON path
    #[arg(long)]
    fg: String,

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

    println!("Loading face detector: {}", args.m_fd);
    let mut face_detector = FaceDetector::builder(&args.m_fd)
        .confidence(args.t_fd)
        .build(&core)?;

    println!("Loading face gallery: {}", args.fg);
    let mut gallery =
        FaceGallery::builder(&args.fg, &args.m_fd, &args.m_lm, &args.m_reid).build(&core)?;
    println!("Gallery loaded: {} identities", gallery.size());

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
    let mut total_identified: u32 = 0;

    loop {
        let frame = video.current_frame()?;
        let faces = face_detector.detect_frame(&frame)?;
        let ids = gallery.identify_frame(&frame, &faces)?;

        if let Some(ref mut w) = writer {
            let mut annotated = AnnotatedFrame::from_frame(&frame)?;
            for (face, &id) in faces.iter().zip(ids.iter()) {
                let label = if id >= 0 {
                    gallery.label(id)?
                } else {
                    "Unknown".to_string()
                };
                let color = if id >= 0 { Color::GREEN } else { Color::RED };
                annotated.draw_detection(
                    face.x,
                    face.y,
                    face.width,
                    face.height,
                    &label,
                    color,
                )?;
            }
            w.write(&annotated)?;
        }

        drop(frame);

        let identified_count = ids.iter().filter(|&&id| id >= 0).count();
        total_identified += identified_count as u32;

        println!(
            "Frame {}: {} faces detected, {} identified",
            frame_num,
            faces.len(),
            identified_count
        );
        for (i, (face, &id)) in faces.iter().zip(ids.iter()).enumerate() {
            let label = if id >= 0 {
                gallery.label(id)?
            } else {
                "Unknown".to_string()
            };
            println!(
                "  Face {}: ({}, {}) {}x{} conf={:.2} -> \"{}\"",
                i, face.x, face.y, face.width, face.height, face.confidence, label
            );
        }

        if !video.read()? {
            break;
        }
        frame_num += 1;
    }

    println!("\nProcessing complete. {} frames processed.", frame_num + 1);
    println!("Total identifications: {}", total_identified);
    Ok(())
}
