use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    // cpp/ and cpp/ffi/ are at the workspace root
    let workspace_root = manifest_dir
        .parent().expect("CARGO_MANIFEST_DIR has no parent (expected crates/)")
        .parent().expect("crates/ has no parent (expected workspace root)");
    let cpp_dir = workspace_root.join("cpp");
    let ffi_dir = cpp_dir.join("ffi");

    // --- Step 1: Generate CXX Bridge Files ---
    let crate_name = env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME not set");
    let generated_source_path = out_dir
        .join("cxxbridge")
        .join("sources")
        .join(&crate_name)
        .join("src/lib.rs.cc");

    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++17");

    let cxx_include_path = out_dir.join("cxxbridge").join("include");

    // --- Step 2: Build C++ via CMake ---
    // We build the ffi bridge + ovi_vision library together.
    // The ffi/CMakeLists.txt pulls in the parent cpp/ library.
    let cpp_lib_dst = cmake::Config::new(&ffi_dir)
        .define("CXX_INCLUDE_DIR", &cxx_include_path)
        .define("CXX_GENERATED_SOURCE", &generated_source_path)
        .define("OPENVINO_VISION_DIR", &cpp_dir)
        .build();

    // --- Step 3: Link static libraries ---
    println!(
        "cargo:rustc-link-search=native={}",
        cpp_lib_dst.join("lib").display()
    );
    // ovi_vision is built via add_subdirectory
    println!(
        "cargo:rustc-link-search=native={}",
        cpp_lib_dst.join("build").join("ovi_vision_build").display()
    );
    println!("cargo:rustc-link-lib=static=ovi_vision_ffi");
    println!("cargo:rustc-link-lib=static=ovi_vision");

    // --- Step 4: Link External Dependencies ---
    if cfg!(target_os = "macos") {
        let homebrew_lib = "/opt/homebrew/lib";
        println!("cargo:rustc-link-search=native={}", homebrew_lib);

        if let Ok(output) = std::process::Command::new("brew")
            .args(["--prefix", "openvino"])
            .output()
        {
            if output.status.success() {
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:rustc-link-search=native={}/lib", prefix);
            }
        }
        if let Ok(output) = std::process::Command::new("brew")
            .args(["--prefix", "opencv"])
            .output()
        {
            if output.status.success() {
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:rustc-link-search=native={}/lib", prefix);
            }
        }
    }

    println!("cargo:rustc-link-lib=dylib=opencv_core");
    println!("cargo:rustc-link-lib=dylib=opencv_highgui");
    println!("cargo:rustc-link-lib=dylib=opencv_imgproc");
    println!("cargo:rustc-link-lib=dylib=opencv_videoio");
    println!("cargo:rustc-link-lib=dylib=opencv_imgcodecs");
    println!("cargo:rustc-link-lib=dylib=openvino");

    // --- Step 5: Rerun conditions ---
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed={}", ffi_dir.display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("src").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("include").display());
}
