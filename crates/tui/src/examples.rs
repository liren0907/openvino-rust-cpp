#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ExampleKind {
    FaceDetection,
    ActionDetection,
    Tracking,
    FaceReid,
}

impl ExampleKind {
    pub const ALL: [ExampleKind; 4] = [
        ExampleKind::FaceDetection,
        ExampleKind::ActionDetection,
        ExampleKind::Tracking,
        ExampleKind::FaceReid,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            ExampleKind::FaceDetection => "Face Detection",
            ExampleKind::ActionDetection => "Action Detection",
            ExampleKind::Tracking => "Tracking (face)",
            ExampleKind::FaceReid => "Face Re-ID",
        }
    }

    pub fn binary_name(&self) -> &'static str {
        match self {
            ExampleKind::FaceDetection => "example-face-detection",
            ExampleKind::ActionDetection => "example-action-detection",
            ExampleKind::Tracking => "example-tracking",
            ExampleKind::FaceReid => "example-face-reid",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParamDef {
    pub flag: &'static str,
    pub label: &'static str,
    pub required: bool,
    pub default: &'static str,
}

pub fn params_for(kind: ExampleKind) -> Vec<ParamDef> {
    let mut params = vec![];
    match kind {
        ExampleKind::FaceDetection => {
            params.push(ParamDef { flag: "--input", label: "Input video", required: true, default: "assets/data/test1.mp4" });
            params.push(ParamDef { flag: "--m-fd", label: "Face detection model", required: true, default: "assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml" });
            params.push(ParamDef { flag: "--t-fd", label: "Confidence threshold", required: false, default: "0.6" });
            params.push(ParamDef { flag: "--output", label: "Output video (.mp4)", required: false, default: "output/face_detection.mp4" });
            params.push(ParamDef { flag: "--read-limit", label: "Frame limit (0=all)", required: false, default: "0" });
        }
        ExampleKind::ActionDetection => {
            params.push(ParamDef { flag: "--input", label: "Input video", required: true, default: "assets/data/test1.mp4" });
            params.push(ParamDef { flag: "--m-act", label: "Action detection model", required: true, default: "assets/intel/person-detection-action-recognition-0005/FP32/person-detection-action-recognition-0005.xml" });
            params.push(ParamDef { flag: "--t-ad", label: "Detection threshold", required: false, default: "0.3" });
            params.push(ParamDef { flag: "--t-ar", label: "Action threshold", required: false, default: "0.75" });
            params.push(ParamDef { flag: "--num-actions", label: "Num action classes", required: false, default: "3" });
            params.push(ParamDef { flag: "--output", label: "Output video (.mp4)", required: false, default: "output/action_detection.mp4" });
            params.push(ParamDef { flag: "--read-limit", label: "Frame limit (0=all)", required: false, default: "0" });
        }
        ExampleKind::Tracking => {
            params.push(ParamDef { flag: "--input", label: "Input video", required: true, default: "assets/data/test1.mp4" });
            params.push(ParamDef { flag: "--mode", label: "Mode (face/action)", required: false, default: "face" });
            params.push(ParamDef { flag: "--m-fd", label: "Face detection model", required: false, default: "assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml" });
            params.push(ParamDef { flag: "--t-fd", label: "Confidence threshold", required: false, default: "0.6" });
            params.push(ParamDef { flag: "--output", label: "Output video (.mp4)", required: false, default: "output/tracking.mp4" });
            params.push(ParamDef { flag: "--read-limit", label: "Frame limit (0=all)", required: false, default: "0" });
        }
        ExampleKind::FaceReid => {
            params.push(ParamDef { flag: "--input", label: "Input video", required: true, default: "assets/data/test1.mp4" });
            params.push(ParamDef { flag: "--m-fd", label: "Face detection model", required: true, default: "assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml" });
            params.push(ParamDef { flag: "--m-lm", label: "Landmarks model", required: true, default: "assets/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml" });
            params.push(ParamDef { flag: "--m-reid", label: "Re-ID model", required: true, default: "assets/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml" });
            params.push(ParamDef { flag: "--fg", label: "Face gallery JSON", required: true, default: "output/faces_gallery.json" });
            params.push(ParamDef { flag: "--t-fd", label: "Confidence threshold", required: false, default: "0.6" });
            params.push(ParamDef { flag: "--output", label: "Output video (.mp4)", required: false, default: "output/face_reid.mp4" });
            params.push(ParamDef { flag: "--read-limit", label: "Frame limit (0=all)", required: false, default: "0" });
        }
    }
    params
}
