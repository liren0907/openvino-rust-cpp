use clap::Parser;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Create a face gallery JSON file from a directory of face images")]
struct Args {
    /// Path to the directory with face images
    #[arg(long = "face_db", default_value = "face_db")]
    face_db: PathBuf,

    /// Path to the output JSON file
    #[arg(long, default_value = "faces_gallery.json")]
    output: PathBuf,
}

#[derive(Serialize)]
struct Gallery {
    identities: Vec<Identity>,
}

#[derive(Serialize)]
struct Identity {
    label: String,
    images: Vec<String>,
}

fn main() {
    let args = Args::parse();

    if !args.face_db.is_dir() {
        eprintln!(
            "Error: {} is not a directory or does not exist.",
            args.face_db.display()
        );
        std::process::exit(1);
    }

    let mut identities = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(&args.face_db)
        .expect("Failed to read face_db directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let person_dir = entry.path();
        let label = person_dir
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();

        let mut images: Vec<String> = fs::read_dir(&person_dir)
            .expect("Failed to read person directory")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext.eq_ignore_ascii_case("jpg"))
            })
            .map(|e| {
                fs::canonicalize(e.path())
                    .expect("Failed to resolve absolute path")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        images.sort();

        if !images.is_empty() {
            identities.push(Identity { label, images });
        }
    }

    let gallery = Gallery { identities };

    let total_images: usize = gallery.identities.iter().map(|id| id.images.len()).sum();

    let json = serde_json::to_string_pretty(&gallery).expect("Failed to serialize JSON");
    fs::write(&args.output, json).expect("Failed to write output file");

    println!("Created gallery JSON at {}", args.output.display());
    println!(
        "Found {} identities with a total of {} images.",
        gallery.identities.len(),
        total_images
    );
}
