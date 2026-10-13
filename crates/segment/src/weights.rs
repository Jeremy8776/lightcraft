//! Safetensors weights read on demand.
//!
//! Tensors are read from the file when a module asks for them (positional reads, no memory
//! map: `unsafe` stays in `lightcraft-sysmem`), so building only the point-prompt model never
//! touches the text encoder's bytes.
//!
//! Two files hold the same SAM 3 weights under different names: the Hugging Face checkpoint this
//! crate's modules are written against, and Meta's original layout (the file ComfyUI's SAM 3 nodes
//! use). [`Weights::open`] recognises which one it was given. For a Meta file it answers the
//! Hugging Face names by finding, splitting, transposing or joining the Meta tensors
//! ([`crate::weights_layout`]), so one download serves both programs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;

use crate::weights_layout::{self, Kind};
use crate::{Error, Result};

struct Entry {
    /// `None` for a type this crate cannot read (the original checkpoint has a few complex tables
    /// the code recomputes); asking for such a tensor is an error, having one in the file is not.
    dtype: Option<DType>,
    dtype_name: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// Which naming the file uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Hugging Face `facebook/sam3`: the names the model code asks for.
    HuggingFace,
    /// Meta's original names (`detector.*`, `tracker.*`).
    Meta,
}

/// A tensor only the Meta layout has, and the Hugging Face layout does not.
const META_MARKER: &str = "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight";

/// An open safetensors file: its header, and the file to read tensor bytes from.
pub struct Weights {
    file: Mutex<File>,
    entries: HashMap<String, Entry>,
    data_start: u64,
    layout: Layout,
    /// The last fused q, k, v tensor read: the three parts are asked for one after another.
    fused: Mutex<Option<(String, Tensor)>>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        let mut file = File::open(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        let file_len = file.metadata().map_err(|e| Error::Model(format!("{}: {e}", path.display())))?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        let n = u64::from_le_bytes(len);
        if n > 100 << 20 || n.saturating_add(8) > file_len {
            return Err(Error::Model(format!("{}: not a safetensors file (header of {n} bytes)", path.display())));
        }
        let mut header = vec![0u8; n as usize];
        file.read_exact(&mut header).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|e| Error::Model(format!("{}: bad header: {e}", path.display())))?;
        let mut entries = HashMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype_name = v.get("dtype").and_then(|d| d.as_str()).unwrap_or("?").to_owned();
            let dtype = (dtype_name == "F32").then_some(DType::F32);
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
                .unwrap_or_default();
            let offsets: Vec<u64> =
                v.get("data_offsets").and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
            let [start, end] = offsets[..] else { return Err(Error::Model(format!("{name}: bad data_offsets"))) };
            if let Some(dtype) = dtype {
                let want = shape.iter().try_fold(dtype.size_in_bytes(), |a, d| a.checked_mul(*d));
                if end.checked_sub(start) != want.map(|w| w as u64) {
                    return Err(Error::Model(format!("{name}: size does not match its shape")));
                }
            }
            // the bytes must be in the file (a damaged or truncated download fails here, not later)
            if (8 + n).checked_add(end).is_none_or(|e| e > file_len) {
                return Err(Error::Model(format!("{}: tensor {name} lies outside the file (incomplete or damaged download?)", path.display())));
            }
            entries.insert(name, Entry { dtype, dtype_name, shape, start, end });
        }
        let layout = if entries.contains_key(META_MARKER) { Layout::Meta } else { Layout::HuggingFace };
        Ok(Arc::new(Self { file: Mutex::new(file), entries, data_start: 8 + n, layout, fused: Mutex::new(None) }))
    }

    /// Which naming the file uses.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Whether [`read`](Self::read) can answer for `name` (a Hugging Face name, whatever the file's layout).
    pub fn contains(&self, name: &str) -> bool {
        match self.layout {
            Layout::HuggingFace => self.entries.contains_key(name),
            Layout::Meta => match weights_layout::translate(name) {
                Some((_, Kind::Concat(parts))) => parts.iter().all(|p| self.entries.contains_key(p)),
                Some((meta, _)) => self.entries.contains_key(&meta),
                None => false,
            },
        }
    }

    /// Tensor `name` (a Hugging Face name) on the CPU, whatever the file's layout.
    pub fn read(&self, name: &str) -> Result<Tensor> {
        if self.layout == Layout::HuggingFace {
            return self.read_stored(name);
        }
        let (meta, kind) = weights_layout::translate(name).ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let bad = |what: &str| Error::Model(format!("{name}: {what}"));
        match kind {
            Kind::Direct => self.read_stored(&meta),
            Kind::Slice(part) => {
                let fused = self.read_fused(&meta)?;
                let rows = fused.dims().first().copied().unwrap_or(0);
                if part >= 3 || rows == 0 || rows % 3 != 0 {
                    return Err(bad(&format!("the fused tensor `{meta}` has {rows} rows and does not split in three")));
                }
                let third = rows / 3;
                Ok(fused.narrow(0, part * third, third)?.contiguous()?)
            }
            Kind::Transpose => Ok(self.read_stored(&meta)?.t()?.contiguous()?),
            Kind::DropLeading { dim, n } => {
                let t = self.read_stored(&meta)?;
                let len = t.dims().get(dim).copied().ok_or_else(|| bad("the tensor has too few dimensions"))?;
                if len <= n {
                    return Err(bad("nothing is left after dropping the leading entries"));
                }
                Ok(t.narrow(dim, n, len - n)?.contiguous()?)
            }
            Kind::Concat(parts) => {
                let tensors = parts.iter().map(|p| self.read_stored(p)).collect::<Result<Vec<_>>>()?;
                Ok(Tensor::cat(&tensors, 0)?)
            }
        }
    }

    fn read_fused(&self, meta: &str) -> Result<Tensor> {
        let mut cache = self.fused.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((name, tensor)) = cache.as_ref()
            && name == meta
        {
            return Ok(tensor.clone());
        }
        let tensor = self.read_stored(meta)?;
        *cache = Some((meta.to_owned(), tensor.clone()));
        Ok(tensor)
    }

    /// Tensor `name` as stored in the file, on the CPU.
    fn read_stored(&self, name: &str) -> Result<Tensor> {
        let e = self.entries.get(name).ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        if e.dtype.is_none() {
            return Err(Error::Model(format!("{name}: unsupported dtype {:?} (the SAM 3 checkpoint is F32)", e.dtype_name)));
        }
        // checked against the file's length in `open`
        let len = usize::try_from(e.end.saturating_sub(e.start)).map_err(|_| Error::Model(format!("{name}: too large")))?;
        let mut bytes = vec![0u8; len];
        {
            let mut f = self.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            f.seek(SeekFrom::Start(self.data_start + e.start)).map_err(|err| Error::Model(format!("{name}: {err}")))?;
            f.read_exact(&mut bytes).map_err(|err| Error::Model(format!("{name}: {err}")))?;
        }
        let v: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        Ok(Tensor::from_vec(v, e.shape.as_slice(), &Device::Cpu)?)
    }
}

/// [`Weights`] as a candle `VarBuilder` backend.
pub struct Backend(pub Arc<Weights>);

impl SimpleBackend for Backend {
    fn get(&self, s: Shape, name: &str, _h: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &s {
            candle_core::bail!("shape mismatch for {name}: expected {s:?}, got {:?}", t.shape())
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.0.read(name).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        t.to_dtype(dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, header: &str, data_len: usize) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("lc-weights-{name}-{}.safetensors", std::process::id()));
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header.as_bytes());
        b.extend(std::iter::repeat_n(0u8, data_len));
        std::fs::write(&p, b).unwrap();
        p
    }

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// A safetensors file of `(name, dtype, shape, bytes)`.
    fn write_tensors(name: &str, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (tensor, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(tensor.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(&header);
        b.extend_from_slice(&data);
        let p = std::env::temp_dir().join(format!("lc-weights-{name}-{}.safetensors", std::process::id()));
        std::fs::write(&p, b).unwrap();
        p
    }

    const MARK: &str = "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight";

    /// A few tensors of a checkpoint in Meta's layout, with easy numbers.
    fn meta_file(name: &str) -> std::path::PathBuf {
        let mut tensors = vec![
            (MARK, "F32", vec![1, 2], f32s(&[9.0, 9.5])),
            (
                "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
                "F32",
                vec![6, 2],
                f32s(&(0..12).map(|i| i as f32).collect::<Vec<_>>()),
            ),
            ("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.bias", "F32", vec![6], f32s(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0])),
            ("detector.backbone.language_backbone.encoder.text_projection", "F32", vec![3, 2], f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ("detector.backbone.vision_backbone.trunk.pos_embed", "F32", vec![1, 3, 2], f32s(&[0.0, 0.1, 1.0, 1.1, 2.0, 2.1])),
            ("detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis", "C64", vec![1, 2], vec![0u8; 16]),
            ("detector.transformer.decoder.layers.1.linear1.weight", "F32", vec![1, 2], f32s(&[7.0, 8.0])),
        ];
        for i in 0..4 {
            tensors.push((
                Box::leak(format!("tracker.sam_prompt_encoder.point_embeddings.{i}.weight").into_boxed_str()),
                "F32",
                vec![1, 2],
                f32s(&[i as f32, i as f32 + 0.5]),
            ));
        }
        write_tensors(name, &tensors)
    }

    #[test]
    fn a_checkpoint_in_meta_layout_serves_the_names_the_model_asks_for() {
        let w = Weights::open(&meta_file("meta-read")).unwrap();
        let q = |n: &str| values(&w.read(n).unwrap());
        let attn = "detector_model.vision_encoder.backbone.layers.0.attention";
        assert_eq!(q(&format!("{attn}.q_proj.weight")), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(q(&format!("{attn}.k_proj.weight")), [4.0, 5.0, 6.0, 7.0]);
        assert_eq!(q(&format!("{attn}.v_proj.weight")), [8.0, 9.0, 10.0, 11.0]);
        assert_eq!(w.read(&format!("{attn}.v_proj.weight")).unwrap().dims(), &[2, 2]);
        assert_eq!(q(&format!("{attn}.k_proj.bias")), [12.0, 13.0]);
        // the text projection is stored the other way round
        let proj = w.read("detector_model.text_encoder.text_projection.weight").unwrap();
        assert_eq!(proj.dims(), &[2, 3]);
        assert_eq!(values(&proj), [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        // the class token of the position embedding is not part of the Hugging Face one
        let pos = w.read("detector_model.vision_encoder.backbone.embeddings.position_embeddings").unwrap();
        assert_eq!(pos.dims(), &[1, 2, 2]);
        assert_eq!(values(&pos), [1.0, 1.1, 2.0, 2.1]);
        // four point embeddings become one
        let points = w.read("tracker_model.prompt_encoder.point_embed.weight").unwrap();
        assert_eq!(points.dims(), &[4, 2]);
        assert_eq!(values(&points), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        // a plain rename
        assert_eq!(q("detector_model.detr_decoder.layers.1.mlp.fc1.weight"), [7.0, 8.0]);
    }

    #[test]
    fn contains_follows_the_translation_and_the_tensors_that_exist() {
        let w = Weights::open(&meta_file("meta-contains")).unwrap();
        assert!(w.contains("detector_model.vision_encoder.backbone.layers.0.attention.k_proj.weight"));
        assert!(w.contains("tracker_model.prompt_encoder.point_embed.weight"));
        assert!(!w.contains("detector_model.vision_encoder.backbone.layers.1.attention.k_proj.weight"), "layer 1 is not in the file");
        assert!(!w.contains("detector_model.nothing.weight"));
        // a joined tensor needs every part
        let three = write_tensors(
            "meta-three-points",
            &[
                (MARK, "F32", vec![1], f32s(&[0.0])),
                ("tracker.sam_prompt_encoder.point_embeddings.0.weight", "F32", vec![1, 2], f32s(&[0.0; 2])),
                ("tracker.sam_prompt_encoder.point_embeddings.1.weight", "F32", vec![1, 2], f32s(&[0.0; 2])),
                ("tracker.sam_prompt_encoder.point_embeddings.2.weight", "F32", vec![1, 2], f32s(&[0.0; 2])),
            ],
        );
        let three = Weights::open(&three).unwrap();
        assert!(!three.contains("tracker_model.prompt_encoder.point_embed.weight"));
        assert!(three.read("tracker_model.prompt_encoder.point_embed.weight").is_err());
        assert!(!w.contains(MARK), "a Meta name is not a name the model asks for");
        assert!(w.read("detector_model.nothing.weight").unwrap_err().to_string().contains("missing tensor"));
        assert!(w.read("detector_model.vision_encoder.backbone.layers.1.attention.k_proj.weight").is_err());
    }

    #[test]
    fn a_fused_tensor_that_does_not_split_in_three_is_refused() {
        let p = write_tensors(
            "meta-bad-split",
            &[
                (MARK, "F32", vec![1], f32s(&[0.0])),
                ("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight", "F32", vec![4, 1], f32s(&[0.0; 4])),
            ],
        );
        let w = Weights::open(&p).unwrap();
        assert!(w.read("detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight").unwrap_err().to_string().contains("three"));
    }

    #[test]
    fn a_tensor_of_another_type_is_refused_when_asked_for_not_when_the_file_is_opened() {
        // the Meta file's complex tables open fine (the other tests) and are never asked for; any other type is an error on use
        let p = write_tensors(
            "half",
            &[("detector_model.a.weight", "F16", vec![2], vec![0u8; 4]), ("detector_model.b.weight", "F32", vec![1], f32s(&[1.0]))],
        );
        let w = Weights::open(&p).unwrap();
        let err = w.read("detector_model.a.weight").unwrap_err().to_string();
        assert!(err.contains("unsupported dtype"), "{err}");
        assert_eq!(values(&w.read("detector_model.b.weight").unwrap()), [1.0]);
    }

    #[test]
    fn the_layout_is_told_from_the_names() {
        assert_eq!(Weights::open(&meta_file("layout-meta")).unwrap().layout(), Layout::Meta);
        let hf = write_tensors("layout-hf", &[("detector_model.a.weight", "F32", vec![1], f32s(&[1.0]))]);
        assert_eq!(Weights::open(&hf).unwrap().layout(), Layout::HuggingFace);
    }

    #[test]
    fn a_hugging_face_checkpoint_is_read_as_it_always_was() {
        let p = write_tensors(
            "hf",
            &[
                ("detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight", "F32", vec![1, 2], f32s(&[1.0, 2.0])),
                ("detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight", "F32", vec![1, 2], f32s(&[3.0, 4.0])),
            ],
        );
        let w = Weights::open(&p).unwrap();
        assert_eq!(values(&w.read("detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight").unwrap()), [3.0, 4.0]);
        assert!(w.contains("detector_model.vision_encoder.backbone.layers.0.attention.q_proj.weight"));
        assert!(!w.contains("detector_model.vision_encoder.backbone.layers.0.attention.k_proj.weight"));
    }

    #[test]
    fn tensors_must_lie_inside_the_file() {
        let ok = write("ok", r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#, 16);
        let w = Weights::open(&ok).unwrap();
        assert_eq!(w.read("a").unwrap().dims(), &[2, 2]);
        // a truncated download: the header promises bytes that aren't there
        let short = write("short", r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#, 8);
        assert!(Weights::open(&short).err().unwrap().to_string().contains("outside the file"));
        // hostile offsets: huge, or beyond u64
        let huge = write("huge", r#"{"a":{"dtype":"F32","shape":[1073741824,1073741824],"data_offsets":[0,4611686018427387904]}}"#, 16);
        assert!(Weights::open(&huge).is_err());
        let wrap = write("wrap", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[18446744073709551612,18446744073709551616]}}"#, 16);
        assert!(Weights::open(&wrap).is_err());
        // a header longer than the file
        let p = std::env::temp_dir().join(format!("lc-weights-hdr-{}.safetensors", std::process::id()));
        std::fs::write(&p, 1000u64.to_le_bytes()).unwrap();
        assert!(Weights::open(&p).is_err());
        for f in [ok, short, huge, wrap, p] {
            let _ = std::fs::remove_file(f);
        }
    }
}
