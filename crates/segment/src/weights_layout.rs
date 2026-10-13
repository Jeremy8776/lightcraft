//! Where a Hugging Face tensor of the SAM 3 checkpoint lives in a checkpoint saved in Meta's layout.
//!
//! The same trained weights circulate in two files: `facebook/sam3` in Hugging Face format (what
//! this crate's modules ask for: `detector_model.*`, `tracker_model.*`, `tracker_neck.*`) and the
//! original Meta layout (`detector.*`, `tracker.*`) that ComfyUI's SAM 3 nodes read. They differ
//! only in names and in four mechanical ways: Meta fuses the q, k and v projections into one tensor,
//! stores the text projection transposed, keeps a class token in the vision position embedding,
//! and keeps the prompt point embeddings as four tensors. [`translate`] turns a Hugging Face name
//! into the Meta tensor to read and what to do with it, so one downloaded file serves both.
//!
//! The rules are data (`sam3_layout.tsv`, format described in its header). Nothing here reads a
//! file, allocates from input sizes, or panics.

use std::sync::OnceLock;

/// What to do with the Meta tensor to get the Hugging Face one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Use it as it is.
    Direct,
    /// Part `n` (0 q, 1 k, 2 v) of three equal parts along dim 0.
    Slice(usize),
    Transpose,
    /// Without its first `n` entries along dim `dim`.
    DropLeading {
        dim: usize,
        n: usize,
    },
    /// The named Meta tensors joined along dim 0 (the rule's own tensor name is unused).
    Concat(Vec<String>),
}

struct Rule {
    hf: String,
    meta: String,
    kind: Kind,
}

const TABLE: &str = include_str!("sam3_layout.tsv");

fn parse_rule(line: &str) -> Option<Rule> {
    if line.trim().is_empty() || line.starts_with('#') {
        return None;
    }
    let mut cols = line.split('\t');
    let (kind, hf, meta, arg) = (cols.next()?, cols.next()?, cols.next()?, cols.next().unwrap_or(""));
    let kind = match kind {
        "direct" => Kind::Direct,
        "slice" => Kind::Slice(arg.trim().parse().ok()?),
        "transpose" => Kind::Transpose,
        "drop" => {
            let (dim, n) = arg.trim().split_once(',')?;
            Kind::DropLeading { dim: dim.parse().ok()?, n: n.parse().ok()? }
        }
        "concat" => Kind::Concat(meta.split(',').map(str::to_owned).collect()),
        _ => return None,
    };
    Some(Rule { hf: hf.to_owned(), meta: meta.to_owned(), kind })
}

fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| TABLE.lines().filter_map(parse_rule).collect())
}

/// Numbers captured by a pattern, and the rest of the name if it ends in `*`.
fn matches(pattern: &str, name: &str) -> Option<(Vec<usize>, String)> {
    let mut caps = Vec::new();
    let mut names = name.split('.');
    let mut segments = pattern.split('.').peekable();
    while let Some(segment) = segments.next() {
        if segment == "*" {
            // only as the last segment, and it needs at least one segment of the name to take
            let rest: Vec<&str> = names.collect();
            return (segments.peek().is_none() && !rest.is_empty()).then(|| (caps, rest.join(".")));
        }
        let part = names.next()?;
        if segment == "#" {
            if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            caps.push(part.parse().ok()?);
        } else if segment != part {
            return None;
        }
    }
    names.next().is_none().then(|| (caps, String::new()))
}

/// `template` with its `{..}` placeholders filled in.
fn expand(template: &str, caps: &[usize], rest: &str) -> Option<String> {
    let mut out = String::with_capacity(template.len() + rest.len());
    let mut remaining = template;
    while let Some(start) = remaining.find('{') {
        out.push_str(remaining.get(..start)?);
        let end = remaining.get(start..)?.find('}')? + start;
        let inner = remaining.get(start + 1..end)?;
        if inner == "*" {
            out.push_str(rest);
        } else {
            // `n`, `n*k`, `n+c` or `n*k+c`
            let (product, add) = inner.split_once('+').unwrap_or((inner, "0"));
            let (index, times) = product.split_once('*').unwrap_or((product, "1"));
            let value = caps.get(index.parse::<usize>().ok()?)?;
            let number = value.checked_mul(times.parse::<usize>().ok()?)?.checked_add(add.parse::<usize>().ok()?)?;
            out.push_str(&number.to_string());
        }
        remaining = remaining.get(end + 1..)?;
    }
    out.push_str(remaining);
    Some(out)
}

/// The Meta tensor to read for the Hugging Face tensor `hf`, and what to do with it. For
/// [`Kind::Concat`] the returned name is empty and the parts are in the kind.
pub(crate) fn translate(hf: &str) -> Option<(String, Kind)> {
    for rule in rules() {
        if let Some((caps, rest)) = matches(&rule.hf, hf) {
            let meta = match rule.kind {
                Kind::Concat(_) => String::new(),
                _ => expand(&rule.meta, &caps, &rest)?,
            };
            return Some((meta, rule.kind.clone()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str) -> (String, Kind) {
        translate(name).unwrap_or_else(|| panic!("no rule for {name}"))
    }

    #[test]
    fn a_rename_keeps_the_layer_number() {
        assert_eq!(
            t("detector_model.detr_decoder.layers.3.mlp.fc1.weight"),
            ("detector.transformer.decoder.layers.3.linear1.weight".into(), Kind::Direct)
        );
        assert_eq!(
            t("detector_model.vision_encoder.backbone.layers.31.layer_norm2.bias").0,
            "detector.backbone.vision_backbone.trunk.blocks.31.norm2.bias"
        );
    }

    #[test]
    fn q_k_and_v_are_the_three_parts_of_the_fused_tensor() {
        let base = "detector_model.text_encoder.text_model.encoder.layers.7.self_attn";
        let fused = "detector.backbone.language_backbone.encoder.transformer.resblocks.7.attn.in_proj_weight";
        for (p, part) in [("q_proj", 0), ("k_proj", 1), ("v_proj", 2)] {
            assert_eq!(t(&format!("{base}.{p}.weight")), (fused.into(), Kind::Slice(part)));
        }
        assert_eq!(t(&format!("{base}.k_proj.bias")).0, fused.replace("weight", "bias"));
        assert_eq!(
            t("detector_model.vision_encoder.backbone.layers.2.attention.v_proj.weight"),
            ("detector.backbone.vision_backbone.trunk.blocks.2.attn.qkv.weight".into(), Kind::Slice(2))
        );
    }

    #[test]
    fn numbers_can_be_scaled_and_shifted() {
        let mm = "tracker.maskmem_backbone.mask_downsampler.encoder";
        assert_eq!(t("tracker_model.memory_encoder.mask_downsampler.layers.2.conv.weight").0, format!("{mm}.6.weight"));
        assert_eq!(t("tracker_model.memory_encoder.mask_downsampler.layers.2.layer_norm.weight").0, format!("{mm}.7.weight"));
        assert_eq!(t("tracker_model.memory_encoder.mask_downsampler.final_conv.bias").0, format!("{mm}.12.bias"));
        // the middle layer of a three layer head moves up by one, the ends become proj_in and proj_out
        assert_eq!(t("tracker_model.object_pointer_proj.layers.0.bias").0, "tracker.obj_ptr_proj.layers.1.bias");
        assert_eq!(t("tracker_model.object_pointer_proj.proj_in.bias").0, "tracker.obj_ptr_proj.layers.0.bias");
        assert_eq!(t("tracker_model.object_pointer_proj.proj_out.bias").0, "tracker.obj_ptr_proj.layers.2.bias");
        // the head number is kept, the layer number inside it moves
        assert_eq!(
            t("tracker_model.mask_decoder.output_hypernetworks_mlps.2.layers.0.weight").0,
            "tracker.sam_mask_decoder.output_hypernetworks_mlps.2.layers.1.weight"
        );
    }

    #[test]
    fn the_four_mechanical_differences() {
        assert_eq!(
            t("detector_model.text_encoder.text_projection.weight"),
            ("detector.backbone.language_backbone.encoder.text_projection".into(), Kind::Transpose)
        );
        assert_eq!(
            t("detector_model.vision_encoder.backbone.embeddings.position_embeddings"),
            ("detector.backbone.vision_backbone.trunk.pos_embed".into(), Kind::DropLeading { dim: 1, n: 1 })
        );
        let (_, kind) = t("tracker_model.prompt_encoder.point_embed.weight");
        let Kind::Concat(parts) = kind else { panic!("not a concat") };
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[3], "tracker.sam_prompt_encoder.point_embeddings.3.weight");
    }

    #[test]
    fn the_two_necks_read_their_own_convolutions() {
        assert_eq!(t("detector_model.vision_encoder.neck.fpn_layers.1.proj1.weight").0, "detector.backbone.vision_backbone.convs.1.conv_1x1.weight");
        assert_eq!(t("tracker_neck.fpn_layers.1.proj1.weight").0, "detector.backbone.vision_backbone.sam2_convs.1.conv_1x1.weight");
        assert_eq!(t("tracker_neck.fpn_layers.0.scale_layers.2.bias").0, "detector.backbone.vision_backbone.sam2_convs.0.dconv_2x2_1.bias");
    }

    #[test]
    fn names_with_no_rule_or_a_malformed_number_are_not_translated() {
        assert!(translate("detector_model.nothing.here.weight").is_none());
        assert!(translate("detector_model.detr_decoder.layers.x.mlp.fc1.weight").is_none());
        assert!(translate("detector_model.detr_decoder.layers.3.mlp.fc1").is_none(), "a leaf is required where the rule has one");
        assert!(translate("").is_none());
        assert!(translate("detector_model.detr_decoder.layers.+3.mlp.fc1.weight").is_none(), "a sign is not a digit");
        assert!(translate("detector_model.detr_decoder.layers..mlp.fc1.weight").is_none());
        assert!(translate("detector_model.detr_decoder.layers.3.mlp.fc1.weight.extra").is_some(), "a trailing * takes the rest");
        assert!(translate("detector_model.detr_decoder.presence_token.weight.extra").is_none(), "a literal rule takes no more segments");
        assert!(translate("detector.backbone.vision_backbone.trunk.patch_embed.proj.weight").is_none(), "Meta names are not Hugging Face names");
    }

    #[test]
    fn every_line_of_the_table_is_a_rule() {
        let lines = TABLE.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).count();
        assert_eq!(rules().len(), lines, "a line of sam3_layout.tsv does not parse");
        assert_eq!(lines, 203);
    }

    #[test]
    fn templates_only_use_numbers_their_pattern_captures() {
        for rule in rules() {
            let captures = rule.hf.split('.').filter(|s| *s == "#").count();
            let mut rest = rule.meta.as_str();
            while let Some(start) = rest.find('{') {
                let end = rest[start..].find('}').map(|e| start + e).expect("unterminated placeholder");
                let inner = &rest[start + 1..end];
                if inner != "*" {
                    let index: usize = inner.split(['*', '+']).next().unwrap().parse().expect("placeholder number");
                    assert!(index < captures, "{} uses {{{inner}}} but captures {captures}", rule.hf);
                } else {
                    assert!(rule.hf.ends_with(".*"), "{} uses {{*}} without a trailing *", rule.hf);
                }
                rest = &rest[end + 1..];
            }
        }
    }
}
