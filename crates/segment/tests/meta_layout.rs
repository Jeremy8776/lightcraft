//! One SAM 3 checkpoint, two file layouts: the Hugging Face file and the same weights in Meta's original layout
//! (the file ComfyUI's SAM 3 nodes use) must give bit-identical results.
//!
//! Needs real files, so it is skipped (passes) without them:
//! `LIGHTCRAFT_SAM3_DIR=<Hugging Face model dir> LIGHTCRAFT_SAM3_META=<Meta-layout .safetensors> cargo test -p lightcraft-segment --release --test meta_layout -- --nocapture`
//! The tokenizer files are taken from the first directory; only the weights differ.

use std::path::{Path, PathBuf};

use lightcraft_segment::{Click, Layout, Sam3};

fn setup() -> Option<(PathBuf, PathBuf)> {
    let dir = PathBuf::from(std::env::var_os("LIGHTCRAFT_SAM3_DIR")?);
    let meta = PathBuf::from(std::env::var_os("LIGHTCRAFT_SAM3_META")?);
    (dir.is_dir() && meta.is_file()).then_some((dir, meta))
}

/// A model directory whose `model.safetensors` is `weights` (linked, not copied: it is 3.4 GB).
fn dir_with(weights: &Path, tokenizer_from: &Path, name: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir().join(format!("lc-meta-layout-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    for f in ["vocab.json", "merges.txt"] {
        std::fs::copy(tokenizer_from.join(f), dir.join(f)).ok()?;
    }
    let target = dir.join("model.safetensors");
    let _ = std::fs::remove_file(&target);
    std::fs::hard_link(weights, &target).ok()?;
    Some(dir)
}

/// A flat grey photo with a bright red disc and a blue square, so both prompt kinds have something to find.
fn picture(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - w as f32 * 0.35, y as f32 - h as f32 * 0.5);
            let px = if dx * dx + dy * dy < (h as f32 * 0.25).powi(2) {
                [220, 30, 30]
            } else if x > w * 6 / 10 && x < w * 9 / 10 && y > h / 4 && y < h * 3 / 4 {
                [30, 60, 220]
            } else {
                [110 + (x * 40 / w) as u8, 120, 125]
            };
            rgb[(y * w + x) * 3..][..3].copy_from_slice(&px);
        }
    }
    rgb
}

struct Run {
    backbone: Vec<f32>,
    clicks: (Vec<f32>, f32),
    text: Option<(Vec<f32>, f32)>,
}

fn run(dir: &Path) -> (Layout, Run) {
    let mut model = Sam3::load(dir).unwrap();
    let layout = model.layout();
    let (w, h) = (640, 480);
    let mut enc = model.encode(&picture(w, h), w, h).unwrap();
    let backbone = enc.backbone().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let p = model.segment_clicks(&mut enc, &[Click { x: 0.35, y: 0.5, positive: true }]).unwrap();
    let text = model.segment_text(&mut enc, "red circle", 0.3).unwrap().map(|p| (p.logits, p.score));
    (layout, Run { backbone, clicks: (p.logits, p.score), text })
}

#[test]
fn the_meta_layout_gives_bit_identical_results() {
    let Some((hf_dir, meta)) = setup() else {
        eprintln!("skipped: set LIGHTCRAFT_SAM3_DIR and LIGHTCRAFT_SAM3_META");
        return;
    };
    let Some(meta_dir) = dir_with(&meta, &hf_dir, "meta") else {
        eprintln!("skipped: could not link the Meta file next to the tokenizer (a different drive?)");
        return;
    };
    let (hf_layout, hf) = run(&hf_dir);
    let (meta_layout, other) = run(&meta_dir);
    let _ = std::fs::remove_dir_all(&meta_dir);
    assert_eq!(hf_layout, Layout::HuggingFace);
    assert_eq!(meta_layout, Layout::Meta);
    assert_eq!(hf.backbone, other.backbone, "backbone features differ");
    assert_eq!(hf.clicks.0, other.clicks.0, "click mask logits differ");
    assert_eq!(hf.clicks.1.to_bits(), other.clicks.1.to_bits(), "click score differs");
    let (a, b) = (hf.text.as_ref(), other.text.as_ref());
    assert_eq!(a.is_some(), b.is_some(), "text prompt found something in one layout only");
    if let (Some(a), Some(b)) = (a, b) {
        assert_eq!(a.0, b.0, "text mask logits differ");
        assert_eq!(a.1.to_bits(), b.1.to_bits(), "text score differs");
    }
    eprintln!("text prompt found a match: {}; the two layouts agree on every value", a.is_some());
}
