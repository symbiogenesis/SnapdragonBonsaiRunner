//! Closed static-SVG allowlist and resource bounds enforced before `usvg` conversion.
//!
//! All inputs arriving at the offline worker are treated as untrusted. Even though
//! `usvg` / `resvg` implements only static SVG features and runs inside an offline
//! MXC AppContainer, we explicitly reject DTDs, processing instructions, scripting,
//! event attributes, foreign namespaces, hyperlinks, raster/external image elements,
//! external or `data:` URLs, CSS `@`-rules and escapes, reference cycles, and
//! `<use>` / marker node-amplification bombs before conversion and again on the
//! resolved render tree.

use resvg::usvg::{self, roxmltree};
use std::collections::HashMap;

/// Maximum XML AST nodes accepted by `roxmltree` per SVG document.
pub(super) const MAX_XML_NODES: u32 = 4_096;
/// Maximum XML and `<use>`-expanded element nesting depth.
pub(super) const MAX_SVG_DEPTH: u32 = 32;
/// Maximum element visits when expanding `<use>` subtrees before `usvg` runs.
const MAX_EXPANDED_NODES: usize = 8_192;
/// Maximum `href` inheritance hops on paint servers, filters, and text paths.
const MAX_HREF_CHAIN: usize = 16;
/// Maximum entries permitted in an `feConvolveMatrix` kernel (`8x8`).
const MAX_CONVOLVE_ENTRIES: usize = 64;
/// Maximum `feConvolveMatrix` `order` dimension along either axis.
const MAX_CONVOLVE_ORDER: u32 = 8;
/// Maximum local fragment identifier byte length (`#<id>`).
const MAX_ID_BYTES: usize = 128;
/// Maximum resolved `usvg::Node` count across the tree and defs subroots.
const MAX_RENDER_NODES: usize = 16_384;
/// Maximum resolved `tiny_skia_path` segments across all rendered paths and glyphs.
const MAX_PATH_SEGMENTS: usize = 262_144;

const SVG_NS: &str = "http://www.w3.org/2000/svg";
const XLINK_NS: &str = "http://www.w3.org/1999/xlink";
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";

/// Sorted closed allowlist of static SVG element local names.
///
/// Explicitly omits `script`, `foreignObject`, `a`, `image`, `feImage`, `tref`,
/// `view`, `cursor`, SMIL animation tags (`animate*`, `set`, `discard`, `mpath`),
/// `handler`/`listener`, `metadata`, and all HTML/MathML/media elements.
pub(crate) const ALLOWED_ELEMENTS: &[&str] = &[
    "circle",
    "clipPath",
    "defs",
    "desc",
    "ellipse",
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feConvolveMatrix",
    "feDiffuseLighting",
    "feDisplacementMap",
    "feDistantLight",
    "feDropShadow",
    "feFlood",
    "feFuncA",
    "feFuncB",
    "feFuncG",
    "feFuncR",
    "feGaussianBlur",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "fePointLight",
    "feSpecularLighting",
    "feSpotLight",
    "feTile",
    "feTurbulence",
    "filter",
    "g",
    "line",
    "linearGradient",
    "marker",
    "mask",
    "path",
    "pattern",
    "polygon",
    "polyline",
    "radialGradient",
    "rect",
    "stop",
    "style",
    "svg",
    "switch",
    "symbol",
    "text",
    "textPath",
    "title",
    "tspan",
    "use",
];

/// Sorted closed allowlist of static SVG attribute local names (in no namespace).
///
/// Zero `on*` event attributes, zero HTML resource-loading attributes, and zero
/// CSS image-bearing properties (`mask-image`, `mask-border-source`, `color-profile`,
/// `shape-inside`, `shape-subtract`) are admitted.
pub(crate) const ALLOWED_ATTRIBUTES: &[&str] = &[
    "alignment-baseline",
    "amplitude",
    "aria-describedby",
    "aria-hidden",
    "aria-label",
    "aria-labelledby",
    "azimuth",
    "background-color",
    "baseFrequency",
    "baseProfile",
    "baseline-shift",
    "bias",
    "class",
    "clip",
    "clip-path",
    "clip-rule",
    "clipPathUnits",
    "color",
    "color-interpolation",
    "color-interpolation-filters",
    "color-rendering",
    "cx",
    "cy",
    "d",
    "diffuseConstant",
    "direction",
    "display",
    "divisor",
    "dominant-baseline",
    "dx",
    "dy",
    "edgeMode",
    "elevation",
    "enable-background",
    "exponent",
    "fill",
    "fill-opacity",
    "fill-rule",
    "filter",
    "filterUnits",
    "flood-color",
    "flood-opacity",
    "focusable",
    "font",
    "font-family",
    "font-feature-settings",
    "font-kerning",
    "font-optical-sizing",
    "font-size",
    "font-size-adjust",
    "font-stretch",
    "font-style",
    "font-synthesis",
    "font-variant",
    "font-variant-caps",
    "font-variant-east-asian",
    "font-variant-ligatures",
    "font-variant-numeric",
    "font-variant-position",
    "font-variation-settings",
    "font-weight",
    "fr",
    "fx",
    "fy",
    "glyph-orientation-horizontal",
    "glyph-orientation-vertical",
    "gradientTransform",
    "gradientUnits",
    "height",
    "href",
    "id",
    "image-rendering",
    "in",
    "in2",
    "inline-size",
    "intercept",
    "isolation",
    "k1",
    "k2",
    "k3",
    "k4",
    "kernelMatrix",
    "kernelUnitLength",
    "kerning",
    "lengthAdjust",
    "letter-spacing",
    "lighting-color",
    "limitingConeAngle",
    "line-height",
    "marker",
    "marker-end",
    "marker-mid",
    "marker-start",
    "markerHeight",
    "markerUnits",
    "markerWidth",
    "mask",
    "mask-border",
    "mask-border-mode",
    "mask-border-outset",
    "mask-border-repeat",
    "mask-border-slice",
    "mask-border-width",
    "mask-clip",
    "mask-composite",
    "mask-mode",
    "mask-origin",
    "mask-position",
    "mask-size",
    "mask-type",
    "maskContentUnits",
    "maskUnits",
    "mix-blend-mode",
    "mode",
    "numOctaves",
    "offset",
    "opacity",
    "operator",
    "order",
    "orient",
    "overflow",
    "paint-order",
    "path",
    "pathLength",
    "patternContentUnits",
    "patternTransform",
    "patternUnits",
    "points",
    "pointsAtX",
    "pointsAtY",
    "pointsAtZ",
    "preserveAlpha",
    "preserveAspectRatio",
    "primitiveUnits",
    "r",
    "radius",
    "refX",
    "refY",
    "requiredExtensions",
    "requiredFeatures",
    "result",
    "role",
    "rotate",
    "rx",
    "ry",
    "scale",
    "seed",
    "shape-image-threshold",
    "shape-margin",
    "shape-padding",
    "shape-rendering",
    "side",
    "slope",
    "space",
    "specularConstant",
    "specularExponent",
    "spreadMethod",
    "startOffset",
    "stdDeviation",
    "stitchTiles",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "style",
    "surfaceScale",
    "systemLanguage",
    "tableValues",
    "targetX",
    "targetY",
    "text-align",
    "text-align-last",
    "text-anchor",
    "text-decoration",
    "text-decoration-color",
    "text-decoration-fill",
    "text-decoration-line",
    "text-decoration-stroke",
    "text-decoration-style",
    "text-indent",
    "text-orientation",
    "text-overflow",
    "text-rendering",
    "text-underline-position",
    "textLength",
    "transform",
    "transform-box",
    "transform-origin",
    "type",
    "unicode-bidi",
    "unicode-range",
    "values",
    "vector-effect",
    "version",
    "viewBox",
    "visibility",
    "white-space",
    "width",
    "word-spacing",
    "writing-mode",
    "x",
    "x1",
    "x2",
    "xChannelSelector",
    "y",
    "y1",
    "y2",
    "yChannelSelector",
    "z",
];

fn has_safe_xml_chars(text: &str) -> bool {
    !text
        .chars()
        .any(|ch| ch.is_control() && !matches!(ch, '\t' | '\n' | '\r'))
}

fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_ID_BYTES
        && id
            .chars()
            .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

fn local_fragment_id(raw: &str) -> Option<&str> {
    let trimmed = raw.trim_matches(|ch: char| ch.is_ascii_whitespace());
    let id = trimmed.strip_prefix('#')?;
    is_valid_id(id).then_some(id)
}

fn contains_ascii_case_insensitive(haystack: &str, needle: &[u8]) -> bool {
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|window| window.eq_ignore_ascii_case(needle))
}

fn has_forbidden_scheme_or_script_token(value: &str) -> bool {
    [
        &b"javascript:"[..],
        &b"vbscript:"[..],
        &b"data:"[..],
        &b"file:"[..],
        &b"http:"[..],
        &b"https:"[..],
        &b"blob:"[..],
        &b"expression("[..],
        &b"-moz-binding"[..],
    ]
    .into_iter()
    .any(|token| contains_ascii_case_insensitive(value, token))
}

/// Validates that every `url(...)` function call in `input` resolves strictly to `#<id>`.
fn validate_url_functions(input: &str) -> bool {
    let bytes = input.as_bytes();
    let mut index = 0;
    while index + 4 <= bytes.len() {
        let is_boundary_before = index == 0
            || !(bytes[index - 1].is_ascii_alphanumeric()
                || matches!(bytes[index - 1], b'_' | b'-'));
        if is_boundary_before && bytes[index..index + 4].eq_ignore_ascii_case(b"url(") {
            let arg_start = index + 4;
            let Some(rel_close) = input.get(arg_start..).and_then(|rest| rest.find(')')) else {
                return false;
            };
            let arg_end = arg_start + rel_close;
            let Some(raw_arg) = input.get(arg_start..arg_end) else {
                return false;
            };
            let trimmed = raw_arg.trim_matches(|ch: char| ch.is_ascii_whitespace());
            let unquoted = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
                || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
            {
                if trimmed.len() < 2 {
                    return false;
                }
                &trimmed[1..trimmed.len() - 1]
            } else {
                trimmed
            };
            if unquoted.contains(['"', '\'', '(', '\\']) || local_fragment_id(unquoted).is_none() {
                return false;
            }
            index = arg_end + 1;
        } else {
            index += 1;
        }
    }
    true
}

/// Strips `/* ... */` comments (replacing each with `' '`) while rejecting unclosed
/// comments, CSS backslash escapes, `@`-rules, and any non-fragment `url(...)`.
fn validate_css_text(css: &str) -> bool {
    if !has_safe_xml_chars(css) || css.contains('\\') {
        return false;
    }
    let mut scrubbed = String::with_capacity(css.len());
    let mut chars = css.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(ch) = chars.next() {
        if let Some(delim) = quote {
            if ch == delim {
                quote = None;
            }
            scrubbed.push(ch);
            continue;
        }
        match ch {
            '"' | '\'' => {
                quote = Some(ch);
                scrubbed.push(ch);
            }
            '/' if chars.peek() == Some(&'*') => {
                let _ = chars.next();
                let mut closed = false;
                while let Some(comment_ch) = chars.next() {
                    if comment_ch == '*' && chars.peek() == Some(&'/') {
                        let _ = chars.next();
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return false;
                }
                scrubbed.push(' ');
            }
            '@' => return false,
            _ => scrubbed.push(ch),
        }
    }
    if quote.is_some() {
        return false;
    }
    !has_forbidden_scheme_or_script_token(&scrubbed) && validate_url_functions(&scrubbed)
}

fn element_allows_href(tag_name: &str) -> bool {
    matches!(
        tag_name,
        "use" | "linearGradient" | "radialGradient" | "pattern" | "filter" | "textPath"
    )
}

fn node_href<'a>(node: roxmltree::Node<'a, '_>) -> Option<&'a str> {
    node.attribute("href")
        .or_else(|| node.attribute((XLINK_NS, "href")))
}

fn validate_convolve_matrix(node: roxmltree::Node<'_, '_>) -> bool {
    if let Some(order) = node.attribute("order") {
        let mut count = 0_usize;
        for part in order
            .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
            .filter(|part| !part.is_empty())
        {
            count += 1;
            let Ok(dim) = part.parse::<u32>() else {
                return false;
            };
            if dim == 0 || dim > MAX_CONVOLVE_ORDER || count > 2 {
                return false;
            }
        }
    }
    if let Some(matrix) = node.attribute("kernelMatrix") {
        let entries = matrix
            .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
            .filter(|part| !part.is_empty())
            .count();
        if entries > MAX_CONVOLVE_ENTRIES {
            return false;
        }
    }
    true
}

fn validate_element_and_index<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    depth: u32,
    id_map: &mut HashMap<&'a str, roxmltree::Node<'a, 'input>>,
) -> bool {
    if depth > MAX_SVG_DEPTH {
        return false;
    }
    for ns in node.namespaces() {
        if !matches!(ns.uri(), SVG_NS | XLINK_NS | XML_NS) {
            return false;
        }
    }
    if !matches!(node.tag_name().namespace(), None | Some(SVG_NS)) {
        return false;
    }
    let tag_name = node.tag_name().name();
    if ALLOWED_ELEMENTS.binary_search(&tag_name).is_err() {
        return false;
    }

    for attr in node.attributes() {
        let attr_name = attr.name();
        let attr_val = attr.value();
        if !has_safe_xml_chars(attr_val) {
            return false;
        }
        match attr.namespace() {
            None => {
                if attr_name
                    .as_bytes()
                    .get(..2)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"on"))
                {
                    return false;
                }
                if ALLOWED_ATTRIBUTES.binary_search(&attr_name).is_err() {
                    return false;
                }
            }
            Some(XML_NS) => {
                if !matches!(attr_name, "space" | "lang") {
                    return false;
                }
            }
            Some(XLINK_NS) => {
                if attr_name != "href" {
                    return false;
                }
            }
            Some(_) => return false,
        }

        if attr_name == "href" {
            if !element_allows_href(tag_name) || local_fragment_id(attr_val).is_none() {
                return false;
            }
        } else if attr.namespace().is_none() && attr_name == "id" {
            let trimmed_id = attr_val.trim_matches(|ch: char| ch.is_ascii_whitespace());
            if trimmed_id != attr_val || !is_valid_id(trimmed_id) {
                return false;
            }
            id_map.entry(trimmed_id).or_insert(node);
        } else if attr.namespace().is_none() && attr_name == "style" {
            if matches!(tag_name, "title" | "desc" | "style") || !validate_css_text(attr_val) {
                return false;
            }
        } else if attr.namespace().is_none() && attr_name == "aria-label" {
            if has_forbidden_scheme_or_script_token(attr_val) {
                return false;
            }
        } else if attr_val.contains('\\')
            || has_forbidden_scheme_or_script_token(attr_val)
            || !validate_url_functions(attr_val)
        {
            return false;
        }
    }

    if matches!(tag_name, "title" | "desc") && node.children().any(|child| child.is_element()) {
        return false;
    }
    if tag_name == "style" {
        if let Some(style_type) = node.attribute("type") {
            let trimmed = style_type.trim_matches(|ch: char| ch.is_ascii_whitespace());
            if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("text/css") {
                return false;
            }
        }
        for child in node.children() {
            match child.node_type() {
                roxmltree::NodeType::Text => {
                    if let Some(text) = child.text()
                        && !validate_css_text(text)
                    {
                        return false;
                    }
                }
                roxmltree::NodeType::Comment => {}
                _ => return false,
            }
        }
        return true;
    }
    if tag_name == "feConvolveMatrix" && !validate_convolve_matrix(node) {
        return false;
    }

    for child in node.children() {
        match child.node_type() {
            roxmltree::NodeType::Element => {
                if !validate_element_and_index(child, depth + 1, id_map) {
                    return false;
                }
            }
            roxmltree::NodeType::Text => {
                if let Some(text) = child.text()
                    && !has_safe_xml_chars(text)
                {
                    return false;
                }
            }
            roxmltree::NodeType::Comment => {}
            roxmltree::NodeType::PI | roxmltree::NodeType::Root => return false,
        }
    }
    true
}

fn validate_non_use_href_chains<'a>(
    doc: &'a roxmltree::Document<'_>,
    id_map: &HashMap<&'a str, roxmltree::Node<'a, '_>>,
) -> bool {
    for node in doc.descendants().filter(|node| node.is_element()) {
        let tag_name = node.tag_name().name();
        if tag_name == "use" || !element_allows_href(tag_name) {
            continue;
        }
        let mut visited = [node.id(); MAX_HREF_CHAIN + 1];
        let mut hops = 0_usize;
        let mut current = node;
        while let Some(raw_href) = node_href(current) {
            let Some(target_id) = local_fragment_id(raw_href) else {
                return false;
            };
            let Some(&target) = id_map.get(target_id) else {
                break;
            };
            if hops >= MAX_HREF_CHAIN || visited[..=hops].contains(&target.id()) {
                return false;
            }
            hops += 1;
            visited[hops] = target.id();
            current = target;
        }
    }
    true
}

fn is_ancestor(candidate: roxmltree::Node<'_, '_>, mut node: roxmltree::Node<'_, '_>) -> bool {
    loop {
        if node.id() == candidate.id() {
            return true;
        }
        let Some(parent) = node.parent() else {
            return false;
        };
        node = parent;
    }
}

fn validate_use_expansion<'a>(
    node: roxmltree::Node<'a, '_>,
    depth: u32,
    id_map: &HashMap<&'a str, roxmltree::Node<'a, '_>>,
    active_targets: &mut Vec<roxmltree::NodeId>,
    budget: &mut usize,
) -> bool {
    if depth > MAX_SVG_DEPTH || *budget == 0 {
        return false;
    }
    *budget -= 1;

    if node.tag_name().name() == "use" {
        if let Some(raw_href) = node_href(node) {
            let Some(target_id) = local_fragment_id(raw_href) else {
                return false;
            };
            if let Some(&target) = id_map.get(target_id) {
                if active_targets.contains(&target.id()) || is_ancestor(target, node) {
                    return false;
                }
                active_targets.push(target.id());
                let ok = validate_use_expansion(target, depth + 1, id_map, active_targets, budget);
                let _ = active_targets.pop();
                if !ok {
                    return false;
                }
            }
        }
    }

    for child in node.children().filter(|child| child.is_element()) {
        if !validate_use_expansion(child, depth + 1, id_map, active_targets, budget) {
            return false;
        }
    }
    true
}

/// Validates raw XML AST against the closed static-SVG subset and expansion bounds.
pub(super) fn validate_untrusted_xml(doc: &roxmltree::Document<'_>) -> bool {
    let mut root_element = None;
    for child in doc.root().children() {
        match child.node_type() {
            roxmltree::NodeType::Element => {
                if root_element.replace(child).is_some() {
                    return false;
                }
            }
            roxmltree::NodeType::Comment => {}
            roxmltree::NodeType::Text => {
                if child.text().is_some_and(|text| !text.trim().is_empty()) {
                    return false;
                }
            }
            roxmltree::NodeType::PI | roxmltree::NodeType::Root => return false,
        }
    }
    let Some(svg_root) = root_element else {
        return false;
    };
    if svg_root.tag_name().name() != "svg" {
        return false;
    }

    let mut id_map = HashMap::new();
    if !validate_element_and_index(svg_root, 1, &mut id_map) {
        return false;
    }
    if !validate_non_use_href_chains(doc, &id_map) {
        return false;
    }

    let mut active_targets = Vec::new();
    let mut budget = MAX_EXPANDED_NODES;
    validate_use_expansion(svg_root, 1, &id_map, &mut active_targets, &mut budget)
}

fn validate_group_bounds(
    group: &usvg::Group,
    depth: u32,
    nodes_budget: &mut usize,
    segments_budget: &mut usize,
) -> bool {
    if depth > MAX_SVG_DEPTH + 4 {
        return false;
    }
    for child in group.children() {
        if *nodes_budget == 0 {
            return false;
        }
        *nodes_budget -= 1;
        match child {
            usvg::Node::Group(sub) => {
                if !validate_group_bounds(sub, depth + 1, nodes_budget, segments_budget) {
                    return false;
                }
            }
            usvg::Node::Path(path) => {
                let segs = path.data().len();
                if segs > *segments_budget {
                    return false;
                }
                *segments_budget -= segs;
            }
            usvg::Node::Text(text) => {
                if !validate_group_bounds(
                    text.flattened(),
                    depth + 1,
                    nodes_budget,
                    segments_budget,
                ) {
                    return false;
                }
            }
            // Raster and nested SVG `<image>` nodes are forbidden in the worker.
            usvg::Node::Image(_) => return false,
        }
    }
    true
}

/// Verifies post-conversion `usvg::Tree` invariants before CPU rasterization.
pub(super) fn validate_resolved_tree(tree: &usvg::Tree) -> bool {
    let mut nodes_budget = MAX_RENDER_NODES;
    let mut segments_budget = MAX_PATH_SEGMENTS;
    if !validate_group_bounds(tree.root(), 1, &mut nodes_budget, &mut segments_budget) {
        return false;
    }
    for clip in tree.clip_paths() {
        if !validate_group_bounds(clip.root(), 1, &mut nodes_budget, &mut segments_budget) {
            return false;
        }
    }
    for mask in tree.masks() {
        if !validate_group_bounds(mask.root(), 1, &mut nodes_budget, &mut segments_budget) {
            return false;
        }
    }
    for pattern in tree.patterns() {
        if !validate_group_bounds(pattern.root(), 1, &mut nodes_budget, &mut segments_budget) {
            return false;
        }
    }
    true
}
