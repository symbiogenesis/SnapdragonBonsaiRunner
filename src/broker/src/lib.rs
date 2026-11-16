//! M1 request policy, bounded HTTP I/O and explicit development orchestration.
//!
//! `reception` bounds one supplied stream while preserving header multiplicity.
//! `response` frames a bounded, already HTTP-decoded backend response.
//! `backend` sends canonical requests and decodes HTTP on supplied private streams.
//! `scheduler` bounds reservations and a single active request; `supervisor`
//! joins reception, exact counting, generation and disconnect cancellation.
//! `service` can explicitly bind a numeric loopback listener for development.
//! `runtime` supplies an explicit development runner with bounded concurrency.
//! `launcher` explicitly owns a pinned development backend on Windows, using
//! separate audited Win32 dependencies for process and credential operations.
//! Backend bypass prevention and daily OS isolation still require qualification.
//! This library alone is not a qualified HTTP or process-isolation boundary.
#![forbid(unsafe_code)]

pub mod artifacts;
pub mod backend;
mod configuration;
pub mod deadline;
#[cfg(windows)]
mod inference;
pub mod launch_policy;
#[cfg(windows)]
pub mod launcher;
#[cfg(windows)]
pub mod owner_shutdown;
#[cfg(windows)]
pub mod pipe_transport;
pub mod reception;
mod request_buffer;
pub mod response;
pub mod runtime;
pub mod scheduler;
pub mod service;
pub mod supervisor;
pub mod transport;
mod wire;

#[cfg(all(test, feature = "json-candidate"))]
mod json_candidate;
#[cfg(feature = "json-candidate")]
mod json_candidate_parser;

use http::{HeaderMap, HeaderValue, Method, Version, request::Parts, uri::PathAndQuery};
use serde::{Deserialize, Deserializer, Serialize, de};
use std::{borrow::Cow, fmt, marker::PhantomData};
use subtle::ConstantTimeEq;

mod chat_controls;

/// Largest accepted client or canonical backend request body in bytes.
///
/// The native UI matches this bound. The pinned native runtime also limits
/// worker frames and tokenizer input independently; a larger selected token
/// capacity does not authorize larger serialized requests.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;
/// Largest accepted multimodal client or canonical backend request body in bytes.
pub const MAX_MULTIMODAL_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Largest accepted number of image parts across all messages in one request.
pub const MAX_REQUEST_IMAGES: usize = 24;
/// Largest accepted decoded image payload size per `image_url` data URI.
pub const MAX_IMAGE_DATA_BYTES: usize = 12 * 1024 * 1024;
/// Largest accepted HTTP header block in bytes.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;
/// Largest accepted number of HTTP header fields, including duplicates.
pub const MAX_HEADERS: usize = 64;
/// Largest accepted number of messages in one chat history.
pub const MAX_MESSAGES: usize = 128;
/// Largest accepted number of text content parts in one message.
pub const MAX_CONTENT_PARTS: usize = 128;
/// Stable public alias for the broker's already selected and verified model.
pub const MODEL_ALIAS: &str = "bonsai";
/// Largest allowed completion token budget.
pub const MAX_COMPLETION_TOKENS: u32 = 4096;
/// Historical fixed context used by nonlaunching fixtures and legacy runtimes.
pub const CONTEXT_TOKENS: usize = 8192;
/// Context reserved for template accounting differences.
pub const TEMPLATE_MARGIN_TOKENS: usize = 64;

/// Declared inference capabilities of a verified broker profile.
///
/// Native speech recognition/synthesis belongs to the desktop client and does
/// not expand these model capabilities. Media parts remain rejected unless a
/// verified matching model/projector and vision context admission are active.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCapabilities {
    /// Stable public model alias; this never selects a caller-provided model.
    pub id: &'static str,
    /// Supported inference inputs, independent of native speech facilities.
    pub input_modalities: &'static [&'static str],
    /// Supported inference outputs, independent of native speech facilities.
    pub output_modalities: &'static [&'static str],
    /// Total prompt, completion and reserved context bound.
    pub context_tokens: usize,
    /// Maximum requested completion, including model reasoning.
    pub max_completion_tokens: u32,
}

/// Legacy default capabilities; use [`RequestPolicy::capabilities`] for a session.
pub const MODEL_CAPABILITIES: ModelCapabilities = ModelCapabilities {
    id: MODEL_ALIAS,
    input_modalities: &["text"],
    output_modalities: &["text"],
    context_tokens: CONTEXT_TOKENS,
    max_completion_tokens: MAX_COMPLETION_TOKENS,
};

/// Deliberately contains no input values, parser diagnostics, keys or prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The private endpoint port or credential format is invalid.
    Configuration,
    /// Header count, size, multiplicity or forwarding metadata is invalid.
    Headers,
    /// The private authorization credential is missing or invalid.
    Unauthorized,
    /// The Host header does not identify the private endpoint.
    Host,
    /// The supplied Origin header does not match the private endpoint.
    Origin,
    /// The HTTP method, version or request target is unsupported.
    Route,
    /// The request has unsupported or ambiguous HTTP framing.
    Framing,
    /// The request does not declare the supported JSON content type.
    ContentType,
    /// The declared, actual or canonical body length is invalid or exceeds the cap.
    BodySize,
    /// The body does not match the strict chat schema or role sequence.
    ChatSchema,
    /// Supplied text contains a pinned model control or user-defined spelling.
    ChatControlTokens,
    /// The requested completion token budget is outside the allowed range.
    CompletionLimit,
    /// The trusted counter could not provide a positive prompt token count.
    TokenCountUnavailable,
    /// The prompt, completion and template margin exceed the context budget.
    ContextLimit,
}

/// No Debug/Serialize implementation: credentials must not enter diagnostics.
pub struct RequestPolicy {
    authority: String,
    origin: String,
    authorization: [u8; 71], // "Bearer " + 64 lowercase hex characters
    pinned_abliteration: bool,
    context_tokens: usize,
    pub(crate) vision: bool,
}

impl Drop for RequestPolicy {
    fn drop(&mut self) {
        scrub_bytes(&mut self.authorization);
    }
}

pub(crate) fn scrub_bytes(bytes: &mut [u8]) {
    bytes.fill(0);
    std::hint::black_box(&*bytes);
}

pub(crate) fn bearer_header(key: &[u8]) -> Option<[u8; 71]> {
    if !configuration::valid_key(key) {
        return None;
    }
    let mut header = [0; 71];
    header[..7].copy_from_slice(b"Bearer ");
    header[7..].copy_from_slice(key);
    Some(header)
}

impl RequestPolicy {
    /// The launcher must supply 32 cryptographically random bytes encoded as
    /// lowercase hex. Format validation here cannot establish randomness or ACLs.
    ///
    /// # Errors
    /// Returns [`Rejection::Configuration`] for a privileged port or invalid key.
    pub fn new(port: u16, key: &str) -> Result<Self, Rejection> {
        let Some(authorization) = (port >= 1024)
            .then(|| bearer_header(key.as_bytes()))
            .flatten()
        else {
            return Err(Rejection::Configuration);
        };
        let authority = format!("127.0.0.1:{port}");
        Ok(Self {
            origin: format!("http://{authority}"),
            authority,
            authorization,
            pinned_abliteration: false,
            context_tokens: CONTEXT_TOKENS,
            vision: false,
        })
    }

    /// Construct a policy for an already selected and verified model context.
    ///
    /// The trusted owner must pass the same bound to runtime construction and
    /// readiness verification. Client requests cannot change this selection.
    ///
    /// # Errors
    /// Rejects invalid credentials, ports or contexts outside Swift metadata.
    pub fn with_context_tokens(
        port: u16,
        key: &str,
        context_tokens: usize,
    ) -> Result<Self, Rejection> {
        if !(1..=bonsai_cpu_policy::context::MODEL_CONTEXT_TOKENS).contains(&context_tokens) {
            return Err(Rejection::Configuration);
        }
        let mut policy = Self::new(port, key)?;
        policy.context_tokens = context_tokens;
        Ok(policy)
    }

    /// Enable or disable bounded multimodal image inputs for this session policy.
    #[must_use]
    pub const fn with_vision(mut self, vision: bool) -> Self {
        self.vision = vision;
        self
    }

    /// Maximum serialized client or canonical backend body length admitted.
    #[must_use]
    pub const fn max_body_bytes(&self) -> usize {
        if self.vision {
            MAX_MULTIMODAL_BODY_BYTES
        } else {
            MAX_BODY_BYTES
        }
    }

    /// Capabilities for the owner's actual immutable context selection.
    #[must_use]
    pub const fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            input_modalities: if self.vision {
                &["text", "image"]
            } else {
                &["text"]
            },
            context_tokens: self.context_tokens,
            ..MODEL_CAPABILITIES
        }
    }

    /// Check the parsed head before the transport accepts a bounded body.
    /// Only HTTP/1.1 origin-form POST /v1/chat/completions is supported.
    /// OPTIONS, readiness, UI, tokenization and administrative routes are closed.
    ///
    /// # Errors
    /// Returns a value-free [`Rejection`] for invalid credentials, headers, route,
    /// content type, framing or body length.
    pub fn check_head(&self, head: &Parts) -> Result<AuthorizedHead, Rejection> {
        self.check_headers(&head.headers, || {
            if head.method != Method::POST
                || head.version != Version::HTTP_11
                || head.uri.scheme().is_some()
                || head.uri.authority().is_some()
                || head.uri.path_and_query().map(PathAndQuery::as_str)
                    != Some("/v1/chat/completions")
            {
                return Err(Rejection::Route);
            }
            Ok(())
        })
    }

    pub(crate) fn check_wire_headers(
        &self,
        headers: &[httparse::Header<'_>],
    ) -> Result<AuthorizedHead, Rejection> {
        self.check_headers(headers, || Ok(()))
    }

    fn check_headers<H: HeaderSource + ?Sized>(
        &self,
        headers: &H,
        check_request_line: impl FnOnce() -> Result<(), Rejection>,
    ) -> Result<AuthorizedHead, Rejection> {
        if headers.count() > MAX_HEADERS
            || headers.encoded_bytes().is_none_or(|n| n > MAX_HEADER_BYTES)
        {
            return Err(Rejection::Headers);
        }
        let auth = headers
            .singleton("authorization")?
            .ok_or(Rejection::Unauthorized)?;
        if !bool::from(auth.ct_eq(self.authorization.as_slice())) {
            return Err(Rejection::Unauthorized);
        }
        if headers.singleton("host")? != Some(self.authority.as_bytes()) {
            return Err(Rejection::Host);
        }
        if let Some(origin) = headers.singleton("origin")?
            && origin != self.origin.as_bytes()
        {
            return Err(Rejection::Origin);
        }
        check_request_line()?;
        for name in [
            "transfer-encoding",
            "content-encoding",
            "trailer",
            "expect",
            "upgrade",
        ] {
            if headers.contains(name) {
                return Err(Rejection::Framing);
            }
        }
        if headers.has_forwarded() {
            return Err(Rejection::Headers);
        }
        if let Some(connection) = headers.singleton("connection")?
            && !connection.eq_ignore_ascii_case(b"close")
            && !connection.eq_ignore_ascii_case(b"keep-alive")
        {
            return Err(Rejection::Framing);
        }
        let content_type = headers
            .singleton("content-type")?
            .ok_or(Rejection::ContentType)?;
        if !content_type.eq_ignore_ascii_case(b"application/json")
            && !content_type.eq_ignore_ascii_case(b"application/json; charset=utf-8")
        {
            return Err(Rejection::ContentType);
        }
        let length = headers
            .singleton("content-length")?
            .ok_or(Rejection::Framing)?;
        let length = wire::content_length(length).map_err(|error| match error {
            wire::LengthError::Syntax => Rejection::Framing,
            wire::LengthError::Overflow => Rejection::BodySize,
        })?;
        if length == 0 || length > self.max_body_bytes() {
            return Err(Rejection::BodySize);
        }
        Ok(AuthorizedHead {
            length,
            pinned_abliteration: self.pinned_abliteration,
            context_tokens: self.context_tokens,
            vision: self.vision,
        })
    }
}

trait HeaderSource {
    fn count(&self) -> usize;
    fn encoded_bytes(&self) -> Option<usize>;
    fn singleton(&self, name: &'static str) -> Result<Option<&[u8]>, Rejection>;
    fn contains(&self, name: &'static str) -> bool;
    fn has_forwarded(&self) -> bool;
}

fn add_header_bytes(total: usize, name_len: usize, value_len: usize) -> Option<usize> {
    name_len
        .checked_add(value_len)
        .and_then(|pair| pair.checked_add(4))
        .and_then(|entry| total.checked_add(entry))
}

impl HeaderSource for HeaderMap {
    fn count(&self) -> usize {
        self.len()
    }

    fn encoded_bytes(&self) -> Option<usize> {
        self.iter().try_fold(0usize, |total, (name, value)| {
            add_header_bytes(total, name.as_str().len(), value.as_bytes().len())
        })
    }

    fn singleton(&self, name: &'static str) -> Result<Option<&[u8]>, Rejection> {
        singleton(self, name)
    }

    fn contains(&self, name: &'static str) -> bool {
        self.contains_key(name)
    }

    fn has_forwarded(&self) -> bool {
        self.keys()
            .any(|name| name.as_str() == "forwarded" || name.as_str().starts_with("x-forwarded-"))
    }
}

impl HeaderSource for [httparse::Header<'_>] {
    fn count(&self) -> usize {
        self.len()
    }

    fn encoded_bytes(&self) -> Option<usize> {
        self.iter().try_fold(0usize, |total, header| {
            add_header_bytes(total, header.name.len(), header.value.len())
        })
    }

    fn singleton(&self, name: &'static str) -> Result<Option<&[u8]>, Rejection> {
        wire::singleton_slice(self, name)
    }

    fn contains(&self, name: &'static str) -> bool {
        wire::has_header(self, name)
    }

    fn has_forwarded(&self) -> bool {
        self.iter().any(|header| {
            header.name.eq_ignore_ascii_case("forwarded")
                || header
                    .name
                    .as_bytes()
                    .split_first_chunk::<12>()
                    .is_some_and(|(prefix, _)| prefix.eq_ignore_ascii_case(b"x-forwarded-"))
        })
    }
}

fn singleton<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a [u8]>, Rejection> {
    let mut values = headers.get_all(name).iter();
    let first = values.next().map(HeaderValue::as_bytes);
    if values.next().is_some() {
        return Err(Rejection::Headers);
    }
    Ok(first)
}

/// Constructed only by successful head validation; does not retain the key.
pub struct AuthorizedHead {
    length: usize,
    pinned_abliteration: bool,
    context_tokens: usize,
    vision: bool,
}

impl AuthorizedHead {
    /// Validate a bounded JSON chat body and construct canonical backend bytes.
    ///
    /// # Errors
    /// Returns [`Rejection`] for a mismatched or oversized canonical body, invalid
    /// schema, unsupported message history, completion limit or serialization failure.
    pub fn prepare(self, body: &[u8]) -> Result<PendingChat, Rejection> {
        self.check_body_size(body)?;
        // Deserialize directly: maps/Value would silently erase duplicate keys.
        // Parser error strings can include private values; never propagate them.
        let Object(chat): Object<ClientChat<'_>> =
            serde_json::from_slice(body).map_err(|_| Rejection::ChatSchema)?;
        self.prepare_chat(&chat)
    }

    /// Qualify the pinned SIMD-JSON candidate against ordinary request parsing.
    ///
    /// Copies and pads the bounded immutable body because SIMD-JSON parses in
    /// place. The padded tape path avoids a second internal input copy. This
    /// compatibility entry point starts with fresh scratch; the qualification
    /// harness also measures repeated requests with reusable parser scratch.
    /// Schema, history, settings and canonical serialization use the production
    /// policy. Requests with a decoded `abliteration_scale` field, and candidate
    /// parsing failures, use the production parser to retain its exact numeric
    /// and rejection semantics. No service route selects this parser.
    ///
    /// # Errors
    /// Returns the same value-free rejections as [`Self::prepare`]. Allocation
    /// failures return [`Rejection::ChatSchema`].
    #[cfg(feature = "json-candidate")]
    pub fn prepare_simd_json(self, body: &[u8]) -> Result<PendingChat, Rejection> {
        json_candidate_parser::JsonCandidateScratch::default().prepare(self, body)
    }

    const fn max_body_bytes(&self) -> usize {
        if self.vision {
            MAX_MULTIMODAL_BODY_BYTES
        } else {
            MAX_BODY_BYTES
        }
    }

    const fn check_body_size(&self, body: &[u8]) -> Result<(), Rejection> {
        if body.len() != self.length || body.len() > self.max_body_bytes() {
            Err(Rejection::BodySize)
        } else {
            Ok(())
        }
    }

    fn prepare_chat(self, chat: &ClientChat<'_>) -> Result<PendingChat, Rejection> {
        if !valid_history(&chat.messages) || !valid_multimodal(&chat.messages, self.vision) {
            return Err(Rejection::ChatSchema);
        }
        if chat.messages.iter().any(|message| {
            message.content.contains_control_tokens()
                || message
                    .reasoning_content
                    .as_deref()
                    .is_some_and(chat_controls::contains)
        }) {
            return Err(Rejection::ChatControlTokens);
        }
        if !(1..=MAX_COMPLETION_TOKENS).contains(&chat.max_completion_tokens) {
            return Err(Rejection::CompletionLimit);
        }
        if chat.abliteration_scale.is_some() && !self.pinned_abliteration {
            return Err(Rejection::ChatSchema);
        }
        let lora = self.pinned_abliteration.then(|| {
            [PinnedLora {
                id: 0,
                scale: chat.abliteration_scale.unwrap_or(1.3),
            }]
        });
        let max_completion_tokens = chat.max_completion_tokens;
        let body = request_buffer::serialize_with_hint_and_limit(
            &BackendChat {
                messages: &chat.messages,
                max_completion_tokens,
                stream: chat.stream,
                lora,
                n: 1,
                cache_prompt: false,
                n_cache_reuse: 0,
                ignore_eos: false,
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                repeat_penalty: 1.0,
                chat_template_kwargs: ChatTemplateKwargs {
                    enable_thinking: chat.enable_thinking,
                    reasoning_effort: "xhigh",
                    preserve_thinking: true,
                },
            },
            self.length.saturating_add(256),
            self.max_body_bytes(),
        )?;
        Ok(PendingChat {
            body,
            max_completion_tokens,
            stream: chat.stream,
            context_tokens: self.context_tokens,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientChat<'a> {
    #[serde(default, rename = "model", deserialize_with = "deserialize_model")]
    _model: (),
    #[serde(borrow, deserialize_with = "deserialize_messages")]
    messages: Vec<Message<'a>>,
    #[serde(default = "default_cap")]
    max_completion_tokens: u32,
    #[serde(default = "default_stream")]
    stream: bool,
    #[serde(default)]
    enable_thinking: bool,
    #[serde(default, deserialize_with = "deserialize_abliteration_scale")]
    abliteration_scale: Option<f64>,
}

fn deserialize_abliteration_scale<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    // Deserialize the number directly so an explicit null cannot mean omission.
    // Validate before any narrower representation can round an excess to 2.0.
    let scale = f64::deserialize(deserializer)?;
    if !scale.is_finite() || !(0.0..=2.0).contains(&scale) {
        return Err(de::Error::custom("unsupported abliteration scale"));
    }
    Ok(Some(scale))
}

const fn default_cap() -> u32 {
    MAX_COMPLETION_TOKENS
}
const fn default_stream() -> bool {
    true
}

#[derive(Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(test, derive(Debug))]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: ImageUrl<'a> },
}

#[derive(Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(test, derive(Debug))]
struct ImageUrl<'a> {
    url: Cow<'a, str>,
}

#[derive(Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(test, derive(Debug))]
#[serde(untagged)]
enum MessageContent<'a> {
    Text(Cow<'a, str>),
    Parts(Vec<ContentPart<'a>>),
}

impl MessageContent<'_> {
    fn contains_control_tokens(&self) -> bool {
        match self {
            Self::Text(text) => chat_controls::contains(text),
            Self::Parts(parts) => parts.iter().any(|part| match part {
                ContentPart::Text { text } => chat_controls::contains(text),
                ContentPart::ImageUrl { .. } => false,
            }),
        }
    }
}

/// Text plus optional bounded base64 image data URLs when vision is active.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message<'a> {
    role: Role,
    // JSON text without escapes can be borrowed until canonical serialization;
    // escaped text still uses Serde's owned decoding without changing its bytes.
    #[serde(borrow, deserialize_with = "deserialize_content")]
    content: MessageContent<'a>,
    #[serde(
        default,
        borrow,
        deserialize_with = "deserialize_reasoning_content",
        skip_serializing_if = "Option::is_none"
    )]
    reasoning_content: Option<Cow<'a, str>>,
}

fn deserialize_model<'de, D: Deserializer<'de>>(deserializer: D) -> Result<(), D::Error> {
    struct ModelVisitor;
    impl de::Visitor<'_> for ModelVisitor {
        type Value = ();

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("the fixed public model alias")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<(), E> {
            if value == MODEL_ALIAS {
                Ok(())
            } else {
                Err(E::custom("unsupported model alias"))
            }
        }
    }
    deserializer.deserialize_str(ModelVisitor)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImageUrl<'a> {
    #[serde(borrow)]
    url: Cow<'a, str>,
}

fn deserialize_optional_text<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Cow<'de, str>>, D::Error> {
    #[derive(Deserialize)]
    struct Text<'a>(#[serde(borrow)] Cow<'a, str>);
    Text::deserialize(deserializer).map(|Text(text)| Some(text))
}

fn deserialize_optional_image_url<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Object<RawImageUrl<'de>>>, D::Error> {
    Object::<RawImageUrl<'de>>::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawContentPart<'a> {
    #[serde(borrow, rename = "type")]
    kind: Cow<'a, str>,
    #[serde(default, borrow, deserialize_with = "deserialize_optional_text")]
    text: Option<Cow<'a, str>>,
    #[serde(default, borrow, deserialize_with = "deserialize_optional_image_url")]
    image_url: Option<Object<RawImageUrl<'a>>>,
}

const fn base64_sextet(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn validate_image_data_url(url: &str) -> bool {
    let Some(encoded) = url
        .strip_prefix("data:image/png;base64,")
        .or_else(|| url.strip_prefix("data:image/jpeg;base64,"))
    else {
        return false;
    };
    let bytes = encoded.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return false;
    }
    let pad = if bytes.ends_with(b"==") {
        2
    } else if bytes.ends_with(b"=") {
        1
    } else {
        0
    };
    let core = &bytes[..bytes.len() - pad];
    if core.iter().any(|&byte| base64_sextet(byte).is_none()) {
        return false;
    }
    if pad > 0 {
        let Some(last_sextet) = core.last().copied().and_then(base64_sextet) else {
            return false;
        };
        let mask = if pad == 2 { 0x0f } else { 0x03 };
        if last_sextet & mask != 0 {
            return false;
        }
    }
    let decoded = (bytes.len() / 4) * 3 - pad;
    (1..=MAX_IMAGE_DATA_BYTES).contains(&decoded)
}

fn deserialize_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<MessageContent<'de>, D::Error> {
    struct ContentVisitor;
    impl<'de> de::Visitor<'de> for ContentVisitor {
        type Value = MessageContent<'de>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a string or bounded nonempty array of content objects")
        }

        fn visit_borrowed_str<E: de::Error>(self, value: &'de str) -> Result<Self::Value, E> {
            Ok(MessageContent::Text(Cow::Borrowed(value)))
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
            Ok(MessageContent::Text(Cow::Owned(value.to_owned())))
        }

        fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
            Ok(MessageContent::Text(Cow::Owned(value)))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            let mut parts = Vec::new();
            let mut has_image = false;
            while let Some(Object(part)) =
                sequence.next_element_seed(BoundedElement::<Object<RawContentPart<'de>>> {
                    within_limit: parts.len() < MAX_CONTENT_PARTS,
                    marker: PhantomData,
                })?
            {
                match part.kind.as_ref() {
                    "text" => {
                        let Some(text) = part.text else {
                            return Err(de::Error::custom("missing text field"));
                        };
                        if part.image_url.is_some() {
                            return Err(de::Error::custom("unexpected image_url on text part"));
                        }
                        parts.push(ContentPart::Text { text });
                    }
                    "image_url" => {
                        if part.text.is_some() {
                            return Err(de::Error::custom("unexpected text on image_url part"));
                        }
                        let Some(Object(RawImageUrl { url })) = part.image_url else {
                            return Err(de::Error::custom("missing image_url field"));
                        };
                        if !validate_image_data_url(&url) {
                            return Err(de::Error::custom("unsupported image data URL"));
                        }
                        has_image = true;
                        parts.push(ContentPart::ImageUrl {
                            image_url: ImageUrl { url },
                        });
                    }
                    _ => return Err(de::Error::custom("unsupported content part")),
                }
            }
            if parts.is_empty() {
                return Err(de::Error::custom("empty content parts"));
            }
            if !has_image {
                let mut combined = String::new();
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        combined.push_str(&text);
                    }
                }
                return Ok(MessageContent::Text(Cow::Owned(combined)));
            }
            Ok(MessageContent::Parts(parts))
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

fn deserialize_reasoning_content<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Cow<'de, str>>, D::Error> {
    // An absent field defaults to None; a present field must be text, even when
    // empty. Deserializing Option directly would also accept an unsupported null.
    // The derived wrapper enables Serde's borrowing support for Cow while
    // retaining owned decoding for escaped JSON strings.
    #[derive(Deserialize)]
    struct Text<'a>(#[serde(borrow)] Cow<'a, str>);

    Text::deserialize(deserializer).map(|Text(text)| Some(text))
}

#[derive(Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Role {
    System,
    User,
    Assistant,
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A derived unit enum also accepts externally tagged objects in JSON.
        struct RoleVisitor;
        impl de::Visitor<'_> for RoleVisitor {
            type Value = Role;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a role string")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Role, E> {
                match value {
                    "system" => Ok(Role::System),
                    "user" => Ok(Role::User),
                    "assistant" => Ok(Role::Assistant),
                    _ => Err(E::custom("unsupported role")),
                }
            }
        }
        deserializer.deserialize_str(RoleVisitor)
    }
}

// Serde's derived structs accept positional arrays. Demand an object, then
// delegate its streaming map directly to the derive without losing duplicate keys.
#[derive(Clone)]
struct Object<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Object<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ObjectVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> de::Visitor<'de> for ObjectVisitor<T> {
            type Value = Object<T>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }
            fn visit_map<A: de::MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                T::deserialize(de::value::MapAccessDeserializer::new(map)).map(Object)
            }
        }
        deserializer.deserialize_map(ObjectVisitor(PhantomData))
    }
}

// SeqAccess probes the sequence end before invoking this seed. Exact-limit
// arrays therefore finish normally, while an excess element is rejected before
// its fields, escaped strings or nested containers can be deserialized.
struct BoundedElement<T> {
    within_limit: bool,
    marker: PhantomData<T>,
}

impl<'de, T: Deserialize<'de>> de::DeserializeSeed<'de> for BoundedElement<T> {
    type Value = T;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<T, D::Error> {
        if !self.within_limit {
            return Err(de::Error::custom("array element limit exceeded"));
        }
        T::deserialize(deserializer)
    }
}

fn deserialize_messages<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Message<'de>>, D::Error> {
    struct MessagesVisitor;

    impl<'de> de::Visitor<'de> for MessagesVisitor {
        type Value = Vec<Message<'de>>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded array of message objects")
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            let mut messages = Vec::new();
            while let Some(Object(message)) =
                sequence.next_element_seed(BoundedElement::<Object<Message<'de>>> {
                    within_limit: messages.len() < MAX_MESSAGES,
                    marker: PhantomData,
                })?
            {
                messages.push(message);
            }
            Ok(messages)
        }
    }

    deserializer.deserialize_seq(MessagesVisitor)
}

fn valid_history(messages: &[Message<'_>]) -> bool {
    if messages.is_empty()
        || messages.len() > MAX_MESSAGES
        || messages
            .iter()
            .any(|message| message.reasoning_content.is_some() && message.role != Role::Assistant)
    {
        return false;
    }
    let turns = match messages {
        [first, rest @ ..] if first.role == Role::System => rest,
        _ => messages,
    };
    !turns.len().is_multiple_of(2)
        && turns.iter().enumerate().all(|(i, m)| {
            m.role
                == if i.is_multiple_of(2) {
                    Role::User
                } else {
                    Role::Assistant
                }
        })
}

fn valid_multimodal(messages: &[Message<'_>], vision: bool) -> bool {
    let mut total_images = 0usize;
    for message in messages {
        if let MessageContent::Parts(parts) = &message.content {
            if !vision || message.role != Role::User {
                return false;
            }
            for part in parts {
                if matches!(part, ContentPart::ImageUrl { .. }) {
                    total_images = total_images.saturating_add(1);
                    if total_images > MAX_REQUEST_IMAGES {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Exposes canonical bytes only to a trusted counter before context admission.
pub struct PendingChat {
    body: Vec<u8>,
    max_completion_tokens: u32,
    stream: bool,
    context_tokens: usize,
}

impl PendingChat {
    /// The trusted counter must send these exact canonical bytes to the pinned
    /// runtime's internal `/v1/chat/completions/input_tokens` endpoint. Generation
    /// must use the same immutable model/runtime/template and explicit thinking
    /// settings. [`backend::count_and_admit`] implements the supplied-stream exchange;
    /// bytes/characters are never tokens. The supervisor must prevent identity drift.
    ///
    /// # Errors
    /// Propagates the trusted counter's rejection, or returns
    /// [`Rejection::TokenCountUnavailable`] for zero tokens and
    /// [`Rejection::ContextLimit`] when the checked context budget is exceeded.
    pub fn admit(
        self,
        count_rendered_prompt: impl FnOnce(&[u8]) -> Result<usize, Rejection>,
    ) -> Result<AdmittedChat, Rejection> {
        let prompt = count_rendered_prompt(&self.body)?;
        if prompt == 0 {
            return Err(Rejection::TokenCountUnavailable);
        }
        let completion =
            usize::try_from(self.max_completion_tokens).map_err(|_| Rejection::ContextLimit)?;
        let total = prompt
            .checked_add(completion)
            .and_then(|n| n.checked_add(TEMPLATE_MARGIN_TOKENS));
        if total.is_none_or(|n| n > self.context_tokens) {
            return Err(Rejection::ContextLimit);
        }
        Ok(AdmittedChat {
            body: self.body,
            stream: self.stream,
        })
    }
}

/// Canonical body only: caller headers, credentials and JSON are never forwarded.
/// The backend adapter supplies its own fixed route/Host/key/Content-Length.
pub struct AdmittedChat {
    body: Vec<u8>,
    stream: bool,
}

impl AdmittedChat {
    /// Borrow the canonical request body admitted by the trusted token counter.
    #[must_use]
    pub fn backend_body(&self) -> &[u8] {
        &self.body
    }

    /// Retained from the canonical request, never inferred from backend headers.
    #[must_use]
    pub const fn streams_response(&self) -> bool {
        self.stream
    }
}

#[derive(Serialize)]
struct BackendChat<'a, 'b> {
    messages: &'a [Message<'b>],
    max_completion_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    lora: Option<[PinnedLora; 1]>,
    n: u8,
    cache_prompt: bool,
    n_cache_reuse: u8,
    ignore_eos: bool,
    temperature: f32,
    top_p: f32,
    top_k: u8,
    min_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    repeat_penalty: f32,
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Serialize)]
struct PinnedLora {
    id: u8,
    scale: f64,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
    reasoning_effort: &'static str,
    preserve_thinking: bool,
}

#[cfg(test)]
mod abliteration_tests;

#[cfg(test)]
mod tests {
    use super::{
        ClientChat, ContentPart, Cow, MAX_CONTENT_PARTS, MAX_MESSAGES, Message, MessageContent,
        Object, Role, valid_history, validate_image_data_url,
    };
    use std::assert_matches;

    #[test]
    fn every_short_role_history_matches_the_accepted_language() {
        // The complete accepted corpus through six messages, including the
        // optional system preamble. Enumerating other words catches misplaced
        // system messages, repeated speakers and assistant-prefill acceptance.
        let accepted = ["u", "su", "uau", "suau", "uauau", "suauau"];
        for length in 0..=6 {
            for mut word in 0..3_usize.pow(length) {
                let mut spelling = String::new();
                let mut messages = Vec::new();
                for _ in 0..length {
                    let (letter, role) = match word % 3 {
                        0 => ('s', Role::System),
                        1 => ('u', Role::User),
                        _ => ('a', Role::Assistant),
                    };
                    word /= 3;
                    spelling.push(letter);
                    messages.push(Message {
                        role,
                        content: MessageContent::Text(Cow::Borrowed("")),
                        reasoning_content: None,
                    });
                }
                assert_eq!(
                    valid_history(&messages),
                    accepted.contains(&spelling.as_str()),
                    "history {spelling:?}"
                );
            }
        }
    }

    #[test]
    fn message_limit_rejects_excess_before_deserializing_fields() -> Result<(), serde_json::Error> {
        let message = r#"{"role":"user","content":"kept"}"#;
        let messages = vec![message; MAX_MESSAGES].join(",");
        let prefix = format!("{{\"messages\":[{messages}");
        let body = format!("{prefix}]}}");
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_str(&body)?;
        assert_eq!(chat.messages.len(), MAX_MESSAGES);
        assert!(
            chat.messages
                .iter()
                .all(|message| message.content == MessageContent::Text(Cow::Borrowed("kept")))
        );
        let large = format!(
            r#"{{"role":"user","content":"{}"}}"#,
            r"\u0061".repeat(8192)
        );
        for excess in [
            "null",
            "[]",
            r#"{"content":"\u"#,
            r#"{"unknown":[[["#,
            &large,
        ] {
            let body = format!("{prefix},{excess}]}}");
            let result = serde_json::from_str::<Object<ClientChat<'_>>>(&body);
            assert!(
                matches!(result, Err(error) if error.to_string().contains("array element limit exceeded"))
            );
        }
        let within_limit = format!("{{\"messages\":[{message},null]}}");
        let result = serde_json::from_str::<Object<ClientChat<'_>>>(&within_limit);
        assert!(
            matches!(result, Err(error) if !error.to_string().contains("array element limit exceeded"))
        );
        Ok(())
    }

    #[test]
    fn content_part_limit_rejects_excess_before_deserializing_fields()
    -> Result<(), serde_json::Error> {
        let part = r#"{"type":"text","text":"x"}"#;
        let parts = vec![part; MAX_CONTENT_PARTS].join(",");
        let prefix = format!("{{\"messages\":[{{\"role\":\"user\",\"content\":[{parts}");
        let body = format!("{prefix}]}}]}}");
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_str(&body)?;
        assert_eq!(
            chat.messages[0].content,
            MessageContent::Text(Cow::Owned("x".repeat(MAX_CONTENT_PARTS)))
        );
        let large = format!(r#"{{"type":"text","text":"{}"}}"#, r"\u0078".repeat(8192));
        for excess in ["null", "[]", r#"{"text":"\u"#, r#"{"unknown":[[["#, &large] {
            let body = format!("{prefix},{excess}]}}]}}");
            let result = serde_json::from_str::<Object<ClientChat<'_>>>(&body);
            assert!(
                matches!(result, Err(error) if error.to_string().contains("array element limit exceeded"))
            );
        }
        let within_limit =
            format!("{{\"messages\":[{{\"role\":\"user\",\"content\":[{part},null]}}]}}");
        let result = serde_json::from_str::<Object<ClientChat<'_>>>(&within_limit);
        assert!(
            matches!(result, Err(error) if !error.to_string().contains("array element limit exceeded"))
        );
        Ok(())
    }

    #[test]
    fn message_content_borrows_plain_text_and_decodes_escapes() -> Result<(), serde_json::Error> {
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_slice(
            br#"{"messages":[{"role":"user","content":"plain text"},{"role":"assistant","content":"line\n\u96ea"}]}"#,
        )?;
        assert_matches!(
            chat.messages[0].content,
            MessageContent::Text(Cow::Borrowed("plain text"))
        );
        assert_matches!(
            chat.messages[1].content,
            MessageContent::Text(Cow::Owned(_))
        );
        assert_eq!(
            chat.messages[1].content,
            MessageContent::Text(Cow::Borrowed("line\n雪"))
        );
        Ok(())
    }

    #[test]
    fn image_data_url_validator_enforces_mime_prefix_and_canonical_base64()
    -> Result<(), serde_json::Error> {
        assert!(validate_image_data_url("data:image/png;base64,eA=="));
        assert!(validate_image_data_url("data:image/jpeg;base64,AAAA"));
        for bad in [
            "",
            "http://127.0.0.1/a.png",
            "data:image/gif;base64,AAAA",
            "data:image/png;base64,",
            "data:image/png;base64,AAA",
            "data:image/png;base64,AAA!",
            "data:image/png;base64,eB==",
            "data:image/png;base64,AA=A",
        ] {
            assert!(!validate_image_data_url(bad), "unexpectedly accepted {bad}");
        }
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_slice(
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,eA=="}}]}]}"#,
        )?;
        assert!(matches!(
            chat.messages[0].content,
            MessageContent::Parts(ref parts)
                if matches!(parts.as_slice(), [ContentPart::Text { .. }, ContentPart::ImageUrl { .. }])
        ));
        Ok(())
    }

    #[test]
    fn message_reasoning_borrows_plain_text_and_decodes_escapes() -> Result<(), serde_json::Error> {
        let Object(chat): Object<ClientChat<'_>> = serde_json::from_slice(
            br#"{"messages":[{"role":"assistant","content":"","reasoning_content":"plain text"},{"role":"assistant","content":"","reasoning_content":"line\n\u96ea"}]}"#,
        )?;
        assert_matches!(
            chat.messages[0].reasoning_content,
            Some(Cow::Borrowed("plain text"))
        );
        assert_matches!(chat.messages[1].reasoning_content, Some(Cow::Owned(_)));
        assert_eq!(
            chat.messages[1].reasoning_content.as_deref(),
            Some("line\n雪")
        );
        Ok(())
    }

    #[test]
    fn unified_header_checks_reject_mismatched_auth_lengths_and_scrub_bytes()
    -> Result<(), super::Rejection> {
        use super::{Rejection, RequestPolicy};

        let key = "a".repeat(64);
        let policy = RequestPolicy::new(8080, &key)?;
        for bad_auth in [
            "",
            "Bearer short",
            &format!("Bearer {key}0"),
            &format!("Bearer {}b", "a".repeat(63)),
        ] {
            let head = http::Request::builder()
                .method(http::Method::POST)
                .uri("/v1/chat/completions")
                .header("host", "127.0.0.1:8080")
                .header("authorization", bad_auth)
                .header("content-type", "application/json")
                .header("content-length", "2")
                .body(())
                .map_err(|_| Rejection::Configuration)?
                .into_parts()
                .0;
            assert_eq!(
                policy.check_head(&head).err(),
                Some(Rejection::Unauthorized)
            );

            let headers = [
                httparse::Header {
                    name: "Host",
                    value: b"127.0.0.1:8080",
                },
                httparse::Header {
                    name: "Authorization",
                    value: bad_auth.as_bytes(),
                },
                httparse::Header {
                    name: "Content-Type",
                    value: b"application/json",
                },
                httparse::Header {
                    name: "Content-Length",
                    value: b"2",
                },
            ];
            assert_eq!(
                policy.check_wire_headers(&headers).err(),
                Some(Rejection::Unauthorized)
            );
        }

        let mut buf = [0x5Au8; 71];
        super::scrub_bytes(&mut buf);
        assert_eq!(buf, [0u8; 71]);
        Ok(())
    }
}
