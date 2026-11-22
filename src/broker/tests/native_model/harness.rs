use super::*;
use std::{
    cell::Cell,
    ops::Deref,
    path::PathBuf,
    rc::Rc,
    sync::{
        LazyLock,
        atomic::{AtomicU64, Ordering},
    },
};

static EVIDENCE_SEQ: AtomicU64 = AtomicU64::new(0);

static RUNTIME_LOCK: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../../config/prism-inference.lock.json")).unwrap()
});
static MODEL_LOCK: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!(
        "../../../../config/swift-assessment.lock.json"
    ))
    .unwrap()
});
static ADAPTER_LOCK: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../../config/swift-lora.lock.json")).unwrap()
});
static PTQ1_EXPORT_LOCK: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../../../config/swift.ptq1-export.json")).unwrap()
});

pub(super) fn runtime_lock() -> &'static Value {
    &RUNTIME_LOCK
}

pub(super) fn model_lock() -> &'static Value {
    &MODEL_LOCK
}

pub(super) fn adapter_lock() -> &'static Value {
    &ADAPTER_LOCK
}

pub(super) fn ptq1_export_lock() -> &'static Value {
    &PTQ1_EXPORT_LOCK
}

pub(super) struct TempEvidence(PathBuf);

impl TempEvidence {
    pub(super) fn new(prefix: &str) -> Self {
        let seq = EVIDENCE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("{prefix}-{}-{seq}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Deref for TempEvidence {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for TempEvidence {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) struct DropFlag(Rc<Cell<bool>>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

pub(super) fn drop_flag() -> (DropFlag, Rc<Cell<bool>>) {
    let flag = Rc::new(Cell::new(false));
    (DropFlag(flag.clone()), flag)
}

pub(super) fn run_local(test: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(test);
}

pub(super) fn run_local_set(test: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(LocalSet::new().run_until(test));
}

pub(super) fn run_timed(limit: Duration, test: impl Future<Output = ()>) {
    run_local_set(async {
        timeout(limit, test).await.unwrap();
    });
}

pub(super) fn run_paused(test: impl Future<Output = ()>) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .unwrap()
        .block_on(LocalSet::new().run_until(test));
}

pub(super) async fn read_fixture_request(peer: &mut BufReader<TcpStream>) -> Value {
    let mut length = None;
    loop {
        let line = prefix_line(peer, 16 * 1024).await;
        if let Some(value) = line.strip_prefix("Content-Length: ") {
            length = Some(value.trim().parse::<usize>().unwrap());
        }
        if line == "\r\n" {
            break;
        }
    }
    let mut body = vec![0; length.unwrap()];
    peer.read_exact(&mut body).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

pub(super) fn chunked_sse(payload: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{payload}\r\n0\r\n\r\n",
        payload.len()
    )
    .into_bytes()
}

pub(super) struct ParsedStream {
    pub(super) summary: StreamSummary,
    pub(super) visible_content: String,
    pub(super) prompt_tokens: Option<u64>,
}

pub(super) fn parse_stream_response(bytes: &[u8]) -> ParsedStream {
    let payload = stream_payload(bytes);
    let payload = std::str::from_utf8(&payload).unwrap();
    let mut terminal = false;
    let mut done = false;
    let mut finish_reason = String::new();
    let mut visible_content = String::new();
    let mut completion_tokens = None;
    let mut prompt_tokens = None;
    for line in payload
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        assert!(!done, "No event may follow DONE.");
        if line == "[DONE]" {
            assert!(terminal);
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(line).unwrap();
        assert!(event.get("error").is_none());
        if let Some(tokens) = event["usage"]["prompt_tokens"].as_u64() {
            assert!(prompt_tokens.replace(tokens).is_none());
        }
        if let Some(tokens) = event["usage"]["completion_tokens"].as_u64() {
            assert!(completion_tokens.replace(tokens).is_none());
        }
        let choices = event["choices"].as_array().unwrap();
        for choice in choices {
            assert!(
                !terminal,
                "No generation event may follow a terminal reason."
            );
            if let Some(text) = choice["delta"]["content"].as_str() {
                visible_content.push_str(text);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                assert!(!terminal);
                assert!(["length", "stop"].contains(&reason));
                terminal = true;
                reason.clone_into(&mut finish_reason);
            }
        }
    }
    assert!(terminal && done);
    ParsedStream {
        summary: StreamSummary {
            finish_reason,
            content_bytes: visible_content.len(),
            completion_tokens,
        },
        visible_content,
        prompt_tokens,
    }
}

pub(super) fn prompt_request(prompt: &str, scale: f64, limit: u32) -> Value {
    let mut body = request(json!([{"role":"user","content":prompt}]), true, limit);
    body["enable_thinking"] = json!(false);
    body["abliteration_scale"] = json!(scale);
    body
}
