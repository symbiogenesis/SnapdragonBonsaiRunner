//! Regression coverage for private backend HTTP exchanges and body decoding.

use bonsai_broker::{
    AdmittedChat, MAX_HEADER_BYTES, MAX_HEADERS, PendingChat, Rejection,
    backend::{BackendError, BackendPolicy, count_and_admit, request_chat},
    response::MAX_RESPONSE_BYTES,
};
use std::{collections::BTreeMap, io, time::Duration};
use tokio::{
    io::AsyncReadExt,
    time::{Instant, timeout},
};

use crate::support::{
    BACKEND_KEY, CLIENT_KEY, COUNT_BODY as COUNT, MockStream as Stream, backend_policy as policy,
    deadline, http_chunked_response as chunked_response, http_length_response as length_response,
    pending_chat, run,
};

const CONTEXT_ERROR: &[u8] = br#"{"error":{"code":"context_length_exceeded"}}"#;

fn pending(stream: bool) -> PendingChat {
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"A scene.\nKeep rain & snow ☃."}}],"max_completion_tokens":128,"stream":{stream}}}"#
    );
    pending_chat(body.as_bytes())
}

fn admitted(stream: bool) -> AdmittedChat {
    pending(stream).admit(|_| Ok(100)).unwrap()
}

fn context_response(body: &[u8], chunked: bool) -> Vec<u8> {
    let framing = if chunked {
        "Transfer-Encoding: chunked\r\n".to_owned()
    } else {
        format!("Content-Length: {}\r\n", body.len())
    };
    let mut bytes = format!(
        "HTTP/1.1 413 Content Too Large\r\nContent-Type: application/json\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n{framing}\r\n"
    ).into_bytes();
    if chunked {
        bytes.extend_from_slice(format!("{:x}\r\n", body.len()).as_bytes());
        bytes.extend_from_slice(body);
        bytes.extend_from_slice(b"\r\n0\r\n\r\n");
    } else {
        bytes.extend_from_slice(body);
    }
    bytes
}

fn request_parts<'a>(bytes: &'a [u8], path: &str) -> (BTreeMap<String, String>, &'a [u8]) {
    let split = bytes
        .array_windows()
        .position(|part| part == b"\r\n\r\n")
        .unwrap()
        + 4;
    let mut lines = std::str::from_utf8(&bytes[..split]).unwrap().split("\r\n");
    assert_eq!(lines.next().unwrap(), format!("POST {path} HTTP/1.1"));
    let mut headers = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line.split_once(':').unwrap();
        assert!(
            headers
                .insert(name.to_ascii_lowercase(), value.trim().into())
                .is_none()
        );
    }
    assert_eq!(headers["host"], "127.0.0.1:8081");
    assert_eq!(headers["authorization"], format!("Bearer {BACKEND_KEY}"));
    assert_eq!(headers["content-type"], "application/json");
    assert_eq!(headers["connection"], "close");
    assert_eq!(headers["content-length"], (bytes.len() - split).to_string());
    for name in [
        "origin",
        "forwarded",
        "x-forwarded-for",
        "cookie",
        "transfer-encoding",
    ] {
        assert!(!headers.contains_key(name));
    }
    assert!(
        !bytes
            .array_windows::<{ CLIENT_KEY.len() }>()
            .any(|part| part == CLIENT_KEY.as_bytes())
    );
    (headers, &bytes[split..])
}

#[test]
fn configuration_rejects_invalid_private_backend_authority_or_key() {
    for port in [0, 80, 1023] {
        assert!(matches!(
            BackendPolicy::new(port, BACKEND_KEY),
            Err(BackendError::Configuration)
        ));
    }
    for key in ["", "short", &"G".repeat(64), &"a".repeat(65)] {
        assert!(matches!(
            BackendPolicy::new(8081, key),
            Err(BackendError::Configuration)
        ));
    }
}

#[test]
fn exact_count_and_generation_use_identical_canonical_bytes_and_private_headers() {
    run(async {
        for stream in [false, true] {
            let mut counter = Stream::new(length_response(COUNT, false), 1);
            counter.write_fragment = 2;
            let counter_state = counter.state.clone();
            let chat = count_and_admit(counter, pending(stream), &policy(), deadline())
                .await
                .unwrap();
            assert!(counter_state.borrow().dropped);
            assert_eq!(chat.backend_body(), admitted(stream).backend_body());

            let backend = Stream::new(length_response(b"{}", stream), 3);
            let backend_state = backend.state.clone();
            let response = request_chat(backend, &chat, &policy(), deadline())
                .await
                .unwrap();
            assert!(!backend_state.borrow().dropped);
            drop(response);
            assert!(backend_state.borrow().dropped);

            let count_state = counter_state.borrow();
            let generation_state = backend_state.borrow();
            let (_, count_body) =
                request_parts(&count_state.written, "/v1/chat/completions/input_tokens");
            let (_, generation_body) =
                request_parts(&generation_state.written, "/v1/chat/completions");
            assert_eq!(count_body, chat.backend_body());
            assert_eq!(count_body, generation_body);
        }
    });
}

#[test]
fn exact_count_accepts_an_escaped_object_tag() {
    run(async {
        let count = br#"{"input_tokens":100,"object":"response.\u0069nput_tokens"}"#;
        let chat = count_and_admit(
            Stream::new(length_response(count, false), 1),
            pending(false),
            &policy(),
            deadline(),
        )
        .await
        .unwrap();
        assert_eq!(chat.backend_body(), admitted(false).backend_body());
    });
}

#[test]
fn fragmented_framed_bodies_finish_without_waiting_for_peer_eof() {
    run(async {
        for fragment in [1, 2, 7, MAX_HEADER_BYTES] {
            for stream in [false, true] {
                let expected = b"{\"text\":\"rain\\n\"}\n\0\xff";
                let mut encoded = Vec::new();
                for part in [
                    &expected[..1],
                    &expected[1..5],
                    &expected[5..expected.len() - 2],
                    &expected[expected.len() - 2..],
                ] {
                    encoded.extend_from_slice(format!("{:X}\r\n", part.len()).as_bytes());
                    encoded.extend_from_slice(part);
                    encoded.extend_from_slice(b"\r\n");
                }
                encoded.extend_from_slice(b"0\r\n\r\n");
                for bytes in [
                    length_response(expected, stream),
                    chunked_response(&encoded, stream),
                ] {
                    let mut backend = Stream::new(bytes, fragment);
                    backend.stall_after_input = true;
                    let state = backend.state.clone();
                    let mut body = request_chat(backend, &admitted(stream), &policy(), deadline())
                        .await
                        .unwrap()
                        .into_body();
                    let start = Instant::now();
                    let mut actual = Vec::new();
                    timeout(Duration::from_millis(1), body.read_to_end(&mut actual))
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(Instant::now(), start);
                    assert!(!state.borrow().dropped);
                    let mut byte = [0];
                    assert_eq!(body.read(&mut byte).await.unwrap(), 0);
                    drop(body);
                    assert!(state.borrow().dropped);
                }
            }
        }
    });
}

#[test]
fn zero_length_and_zero_chunk_bodies_have_immediate_clean_eof() {
    run(async {
        for bytes in [
            length_response(b"", false),
            chunked_response(b"0\r\n\r\n", false),
        ] {
            let mut backend = Stream::new(bytes, 1);
            backend.stall_after_input = true;
            let mut body = request_chat(backend, &admitted(false), &policy(), deadline())
                .await
                .unwrap()
                .into_body();
            let mut actual = Vec::new();
            assert_eq!(body.read_to_end(&mut actual).await.unwrap(), 0);
            assert_eq!(actual, [0_u8; 0]);
        }
    });
}

#[test]
fn content_length_does_not_consume_a_later_response_or_wait_for_close() {
    run(async {
        let first = length_response(b"{}", false);
        let mut bytes = first.clone();
        bytes.extend_from_slice(b"HTTP/1.1 200 OK\r\nprivate: later\r\n\r\n");
        let backend = Stream::new(bytes, 1);
        let state = backend.state.clone();
        let mut body = request_chat(backend, &admitted(false), &policy(), deadline())
            .await
            .unwrap()
            .into_body();
        let mut actual = Vec::new();
        body.read_to_end(&mut actual).await.unwrap();
        assert_eq!(actual, b"{}");
        assert_eq!(state.borrow().read_bytes, first.len());
        drop(body);
        assert!(state.borrow().dropped);
    });
}

#[test]
fn ambiguous_or_unsupported_response_metadata_and_framing_fail_closed() {
    run(async {
        for (status, headers) in [
            (
                "HTTP/1.1 201 Created",
                "Content-Type: application/json\r\nContent-Length: 2\r\n",
            ),
            (
                "HTTP/1.1 100 Continue",
                "Content-Type: application/json\r\nContent-Length: 2\r\n",
            ),
            (
                "HTTP/1.0 200 OK",
                "Content-Type: application/json\r\nContent-Length: 2\r\n",
            ),
            ("HTTP/1.1 200 OK", "Content-Length: 2\r\n"),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: text/event-stream\r\nContent-Length: 2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\ncontent-type: application/json\r\nContent-Length: 2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Encoding: identity\r\nContent-Length: 2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nTrailer: X-End\r\nContent-Length: 2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nUpgrade: websocket\r\nContent-Length: 2\r\n",
            ),
            ("HTTP/1.1 200 OK", "Content-Type: application/json\r\n"),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Length: 2\r\ncontent-length: 2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nTransfer-Encoding: gzip, chunked\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nTransfer-Encoding: chunked\r\ntransfer-encoding: chunked\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Length: +2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Length: 2, 2\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Length: 99999999999999999999999999999999999999\r\n",
            ),
            (
                "HTTP/1.1 200 OK",
                "Content-Type: application/json\r\nContent-Length: 4194305\r\n",
            ),
        ] {
            let mut backend = Stream::new(format!("{status}\r\n{headers}\r\n").into_bytes(), 1);
            backend.stall_after_input = true;
            let state = backend.state.clone();
            let start = Instant::now();
            assert!(
                request_chat(backend, &admitted(false), &policy(), deadline())
                    .await
                    .is_err()
            );
            assert_eq!(
                Instant::now(),
                start,
                "rejection must not wait for the body"
            );
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn raw_response_head_syntax_and_bounds_are_enforced() {
    run(async {
        let good = String::from_utf8(length_response(b"", false)).unwrap();
        for bytes in [
            format!("\r\n{good}"),
            good.replacen("HTTP/1.1 200 OK\r\n", "HTTP/1.1 200 OK\n", 1),
            good.replace("Content-Type:", "Content-Type :"),
            good.replace("Content-Type:", " Content-Type:"),
            good.replace("Content-Length:", "\tContent-Length:"),
            good.replace(
                "Content-Type: application/json",
                "Content-Type: application\rjson",
            ),
        ] {
            assert!(
                request_chat(
                    Stream::new(bytes.into_bytes(), 1),
                    &admitted(false),
                    &policy(),
                    deadline()
                )
                .await
                .is_err()
            );
        }
        let base = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 0\r\nX-Pad: \r\n\r\n";
        let exact = base.replace(
            "X-Pad: ",
            &format!("X-Pad: {}", " ".repeat(MAX_HEADER_BYTES - base.len())),
        );
        assert_eq!(exact.len(), MAX_HEADER_BYTES);
        assert!(
            request_chat(
                Stream::new(exact.clone().into_bytes(), usize::MAX),
                &admitted(false),
                &policy(),
                deadline()
            )
            .await
            .is_ok()
        );
        assert!(matches!(
            request_chat(
                Stream::new(
                    exact.replacen("X-Pad: ", "X-Pad:  ", 1).into_bytes(),
                    usize::MAX
                ),
                &admitted(false),
                &policy(),
                deadline()
            )
            .await,
            Err(BackendError::HeadLimit)
        ));
        let extra = "X-Value: x\r\n".repeat(MAX_HEADERS - 2);
        let exact = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 0\r\n{extra}\r\n"
        );
        assert!(
            request_chat(
                Stream::new(exact.into_bytes(), 7),
                &admitted(false),
                &policy(),
                deadline()
            )
            .await
            .is_ok()
        );
        let excessive = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 0\r\n{extra}X-Value: x\r\n\r\n"
        );
        assert!(matches!(
            request_chat(
                Stream::new(excessive.into_bytes(), 7),
                &admitted(false),
                &policy(),
                deadline()
            )
            .await,
            Err(BackendError::HeadLimit)
        ));
    });
}

#[test]
fn body_read_errors_close_the_stream_immediately_and_remain_sticky() {
    run(async {
        for input in [
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n"
                .to_vec(),
            chunked_response(b"", false),
            chunked_response(b"1\r\n", false),
        ] {
            let mut backend = Stream::new(input, usize::MAX);
            backend.fail_read_after_input = Some(io::ErrorKind::ConnectionReset);
            let state = backend.state.clone();
            let mut body = request_chat(backend, &admitted(false), &policy(), deadline())
                .await
                .unwrap()
                .into_body();
            let error = body.read(&mut [0]).await.unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::Other);
            assert!(state.borrow().dropped);
            let polls = state.borrow().read_polls;
            let error = body.read(&mut [0]).await.unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert_eq!(state.borrow().read_polls, polls);
        }
    });
}

#[test]
fn truncated_content_length_is_a_sticky_error_instead_of_clean_eof() {
    run(async {
        for fragment in [1, usize::MAX] {
            let backend = Stream::new(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 3\r\n\r\n{}"
                    .to_vec(),
                fragment,
            );
            let state = backend.state.clone();
            let mut body = request_chat(backend, &admitted(false), &policy(), deadline())
                .await
                .unwrap()
                .into_body();
            let mut actual = Vec::new();
            assert!(body.read_to_end(&mut actual).await.is_err());
            assert_eq!(actual, b"{}");
            assert!(body.read(&mut [0]).await.is_err());
            drop(body);
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn malformed_or_truncated_chunks_never_become_clean_eof() {
    run(async {
        for fragment in [1, usize::MAX] {
            for payload in [
                b"".as_slice(),
                b"1",
                b"1\r",
                b"1\nX\r\n0\r\n\r\n",
                b"g\r\n",
                b"+1\r\n",
                b" 1\r\n",
                b"1;extension=x\r\na\r\n0\r\n\r\n",
                b"2\r\na",
                b"1\r\na",
                b"1\r\na\r",
                b"1\r\naXX0\r\n\r\n",
                b"1\r\na\r\n",
                b"0\r\n",
                b"0\r\n\r",
                b"0\r\nX-Trailer: secret\r\n\r\n",
                b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\r\n",
            ] {
                let backend = Stream::new(chunked_response(payload, false), fragment);
                let state = backend.state.clone();
                let mut body = request_chat(backend, &admitted(false), &policy(), deadline())
                    .await
                    .unwrap()
                    .into_body();
                assert!(body.read_to_end(&mut Vec::new()).await.is_err());
                assert!(
                    state.borrow().dropped,
                    "framing error closes the stream immediately"
                );
                assert!(body.read(&mut [0]).await.is_err());
                drop(body);
                assert!(state.borrow().dropped);
            }
        }
    });
}

#[test]
fn decoded_body_and_chunk_metadata_have_separate_finite_limits() {
    run(async {
        for length in [MAX_RESPONSE_BYTES, MAX_RESPONSE_BYTES + 1] {
            let mut payload = format!("{length:X}\r\n").into_bytes();
            payload.resize(payload.len() + length, b'x');
            payload.extend_from_slice(b"\r\n0\r\n\r\n");
            let backend = Stream::new(chunked_response(&payload, true), usize::MAX);
            let mut body = request_chat(backend, &admitted(true), &policy(), deadline())
                .await
                .unwrap()
                .into_body();
            let mut actual = Vec::new();
            let result = body.read_to_end(&mut actual).await;
            if length == MAX_RESPONSE_BYTES {
                result.unwrap();
                assert_eq!(actual.len(), length);
            } else {
                assert!(result.is_err());
                assert!(actual.len() <= MAX_RESPONSE_BYTES);
            }
        }
        // Every data byte is legal, but enough tiny chunks exhaust metadata.
        let mut payload = b"1\r\nx\r\n".repeat(14_000);
        payload.extend_from_slice(b"0\r\n\r\n");
        let mut body = request_chat(
            Stream::new(chunked_response(&payload, true), usize::MAX),
            &admitted(true),
            &policy(),
            deadline(),
        )
        .await
        .unwrap()
        .into_body();
        let mut actual = Vec::new();
        assert!(body.read_to_end(&mut actual).await.is_err());
        assert!(actual.len() < 14_000);
    });
}

#[test]
fn exact_count_schema_rejects_missing_duplicate_unknown_or_noninteger_fields() {
    run(async {
        for bytes in [
            b"".as_slice(), b"[]", b"null", b"{}",
            br#"{"input_tokens":100}"#,
            br#"{"object":"response.input_tokens"}"#,
            br#"{"input_tokens":100,"object":"other"}"#,
            br#"{"input_tokens":100,"object":"response.input_tokens","extra":1}"#,
            br#"{"input_tokens":100,"input_tokens":100,"object":"response.input_tokens"}"#,
            br#"{"input_tokens":100,"object":"response.input_tokens","object":"response.input_tokens"}"#,
            br#"{"input_tokens":-1,"object":"response.input_tokens"}"#,
            br#"{"input_tokens":1.0,"object":"response.input_tokens"}"#,
            br#"{"input_tokens":"100","object":"response.input_tokens"}"#,
            br#"{"input_tokens":null,"object":"response.input_tokens"}"#,
            br#"{"input_tokens":100,"object":{"response.input_tokens":null}}"#,
            br#"{"input_tokens":999999999999999999999999999999,"object":"response.input_tokens"}"#,
            br#"{"input_tokens":100,"object":"response.input_tokens"}{}"#,
        ] {
            let backend = Stream::new(length_response(bytes, false), 3);
            let state = backend.state.clone();
            assert!(matches!(count_and_admit(backend, pending(false), &policy(), deadline()).await, Err(BackendError::CountSchema)));
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn typed_private_context_rejection_is_framed_bounded_and_count_route_only() {
    run(async {
        for chunked in [false, true] {
            for fragment in [1, 3, usize::MAX] {
                for streaming in [false, true] {
                    let mut backend =
                        Stream::new(context_response(CONTEXT_ERROR, chunked), fragment);
                    backend.stall_after_input = true;
                    let state = backend.state.clone();
                    assert!(matches!(
                        count_and_admit(backend, pending(streaming), &policy(), deadline()).await,
                        Err(BackendError::Admission(Rejection::ContextLimit))
                    ));
                    assert!(state.borrow().dropped);
                    assert!(
                        state
                            .borrow()
                            .written
                            .starts_with(b"POST /v1/chat/completions/input_tokens HTTP/1.1\r\n")
                    );
                }
            }
            let backend = Stream::new(context_response(CONTEXT_ERROR, chunked), 1);
            let state = backend.state.clone();
            assert!(matches!(
                request_chat(backend, &admitted(false), &policy(), deadline()).await,
                Err(BackendError::Metadata)
            ));
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn typed_private_context_rejection_requires_exact_object_schema_and_code() {
    run(async {
        for body in [
            b"".as_slice(),b"[]",b"null",b"{}",br#"{"error":[]}"#,
            br#"{"error":{"code":"other"}}"#, br#"{"error":{"code":413}}"#,
            br#"{"error":{"code":"context_length_exceeded","message":"private input"}}"#,
            br#"{"error":{"code":"context_length_exceeded"},"input_tokens":9000}"#,
            br#"{"error":{"code":"context_length_exceeded","code":"context_length_exceeded"}}"#,
            br#"{"error":{"code":"context_length_exceeded"},"error":{"code":"context_length_exceeded"}}"#,
            br#"{"error":{"code":"context_length_exceeded"}}{}"#,
        ] {
            let backend = Stream::new(context_response(body,false),3);
            let state = backend.state.clone();
            assert!(matches!(count_and_admit(backend,pending(false),&policy(),deadline()).await,Err(BackendError::CountSchema)));
            assert!(state.borrow().dropped);
        }
        assert!(matches!(
            count_and_admit(
                Stream::new(length_response(CONTEXT_ERROR, false), 1),
                pending(false),
                &policy(),
                deadline()
            )
            .await,
            Err(BackendError::CountSchema)
        ));
    });
}

#[test]
fn typed_private_context_rejection_rejects_wrong_status_metadata_and_ambiguous_framing() {
    run(async {
        let original = String::from_utf8(context_response(CONTEXT_ERROR, false)).unwrap();
        for (from, to) in [
            ("413 Content Too Large", "400 Bad Request"),
            ("413 Content Too Large", "401 Unauthorized"),
            ("413 Content Too Large", "429 Too Many Requests"),
            ("413 Content Too Large", "500 Internal Server Error"),
            (
                "Content-Type: application/json",
                "Content-Type: text/event-stream",
            ),
            (
                "Content-Type: application/json",
                "Content-Type: application/json\r\nContent-Encoding: identity",
            ),
            ("Connection: close\r\n", ""),
            ("Connection: close", "Connection: keep-alive"),
            ("Cache-Control: no-store\r\n", ""),
            ("Cache-Control: no-store", "Cache-Control: public"),
            (
                "Cache-Control: no-store",
                "Cache-Control: no-store\r\nCache-Control: no-store",
            ),
            ("X-Content-Type-Options: nosniff\r\n", ""),
            (
                "X-Content-Type-Options: nosniff",
                "X-Content-Type-Options: nosniff\r\nX-Content-Type-Options: nosniff",
            ),
            (
                "Content-Length:",
                "Transfer-Encoding: chunked\r\nContent-Length:",
            ),
        ] {
            let backend = Stream::new(original.replace(from, to).into_bytes(), 1);
            let state = backend.state.clone();
            let result = count_and_admit(backend, pending(false), &policy(), deadline()).await;
            assert!(result.is_err() && !matches!(result, Err(BackendError::Admission(_))));
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn typed_private_context_rejection_requires_complete_body_within_original_deadline() {
    run(async {
        for chunked in [false, true] {
            let mut truncated = context_response(CONTEXT_ERROR, chunked);
            let _ = truncated.pop();
            let backend = Stream::new(truncated.clone(), 3);
            let state = backend.state.clone();
            assert!(matches!(
                count_and_admit(backend, pending(false), &policy(), deadline()).await,
                Err(BackendError::Body)
            ));
            assert!(state.borrow().dropped);
            let mut stalled = Stream::new(truncated, 3);
            stalled.stall_after_input = true;
            let state = stalled.state.clone();
            assert!(matches!(
                count_and_admit(
                    stalled,
                    pending(false),
                    &policy(),
                    Instant::now() + Duration::from_secs(1)
                )
                .await,
                Err(BackendError::Deadline)
            ));
            assert!(state.borrow().dropped);
            for length in [4096, 4097] {
                let mut body = CONTEXT_ERROR.to_vec();
                body.resize(length, b' ');
                let result = count_and_admit(
                    Stream::new(context_response(&body, chunked), usize::MAX),
                    pending(false),
                    &policy(),
                    deadline(),
                )
                .await;
                if length == 4096 {
                    assert!(matches!(
                        result,
                        Err(BackendError::Admission(Rejection::ContextLimit))
                    ));
                } else {
                    assert!(matches!(result, Err(BackendError::CountSchema)));
                }
            }
        }
    });
}

#[test]
fn exact_count_applies_checked_context_admission_and_bounded_collection() {
    run(async {
        let zero = br#"{"input_tokens":0,"object":"response.input_tokens"}"#;
        assert!(matches!(
            count_and_admit(
                Stream::new(length_response(zero, false), 1),
                pending(false),
                &policy(),
                deadline()
            )
            .await,
            Err(BackendError::Admission(Rejection::TokenCountUnavailable))
        ));
        for count in [8001usize, usize::MAX] {
            let bytes = format!(r#"{{"input_tokens":{count},"object":"response.input_tokens"}}"#);
            let backend = Stream::new(length_response(bytes.as_bytes(), false), 7);
            assert!(matches!(
                count_and_admit(backend, pending(false), &policy(), deadline()).await,
                Err(BackendError::Admission(Rejection::ContextLimit))
            ));
        }
        for length in [4096, 4097] {
            let mut bytes = COUNT.to_vec();
            bytes.resize(length, b' ');
            let backend = Stream::new(length_response(&bytes, false), usize::MAX);
            let state = backend.state.clone();
            let result = count_and_admit(backend, pending(false), &policy(), deadline()).await;
            if length == 4096 {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err());
            }
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn request_write_errors_and_flush_failure_drop_the_owned_stream() {
    run(async {
        for case in 0..3 {
            let mut backend = Stream::new(length_response(b"{}", false), 1);
            backend.fail_write = case == 0;
            backend.zero_write = case == 1;
            backend.fail_flush = case == 2;
            let state = backend.state.clone();
            assert!(matches!(
                request_chat(backend, &admitted(false), &policy(), deadline()).await,
                Err(BackendError::Io)
            ));
            assert!(state.borrow().dropped);
            assert_eq!(state.borrow().read_polls, 0);
        }
    });
}

#[test]
fn request_writes_and_flush_share_one_absolute_five_second_budget() {
    run(async {
        for case in 0..3 {
            let mut backend = Stream::new(length_response(b"{}", false), 1);
            backend.stall_write = case == 0;
            backend.stall_flush = case == 1;
            if case == 2 {
                backend.write_fragment = 1;
                backend.write_delay = Some(Duration::from_secs(1));
            }
            let state = backend.state.clone();
            let start = Instant::now();
            assert!(matches!(
                request_chat(backend, &admitted(false), &policy(), deadline()).await,
                Err(BackendError::Deadline)
            ));
            assert_eq!(Instant::now() - start, Duration::from_secs(5));
            assert!(state.borrow().dropped);
            assert_eq!(state.borrow().read_polls, 0);
        }
    });
}

#[test]
fn supervisor_deadline_bounds_slow_head_and_body_without_reset_on_progress() {
    run(async {
        for slow_drip in [false, true] {
            let mut backend = Stream::new(
                if slow_drip {
                    length_response(b"{}", false)
                } else {
                    Vec::new()
                },
                1,
            );
            backend.stall_after_input = true;
            backend.read_delay = slow_drip.then_some(Duration::from_secs(1));
            let state = backend.state.clone();
            let start = Instant::now();
            assert!(matches!(
                request_chat(
                    backend,
                    &admitted(false),
                    &policy(),
                    start + Duration::from_secs(2)
                )
                .await,
                Err(BackendError::Deadline)
            ));
            assert_eq!(Instant::now() - start, Duration::from_secs(2));
            assert!(state.borrow().dropped);
        }
        let mut backend = Stream::new(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n"
                .to_vec(),
            usize::MAX,
        );
        backend.stall_after_input = true;
        let state = backend.state.clone();
        let start = Instant::now();
        let mut body = request_chat(
            backend,
            &admitted(false),
            &policy(),
            start + Duration::from_secs(2),
        )
        .await
        .unwrap()
        .into_body();
        assert!(body.read_to_end(&mut Vec::new()).await.is_err());
        assert_eq!(Instant::now() - start, Duration::from_secs(2));
        assert!(body.read(&mut [0]).await.is_err());
        drop(body);
        assert!(state.borrow().dropped);
    });
}

#[test]
fn expiry_rejects_ready_input_and_operations_becoming_ready_at_the_deadline() {
    run(async {
        for counter in [false, true] {
            let backend = Stream::new(length_response(COUNT, false), usize::MAX);
            let state = backend.state.clone();
            if counter {
                assert!(matches!(
                    count_and_admit(backend, pending(false), &policy(), Instant::now()).await,
                    Err(BackendError::Deadline)
                ));
            } else {
                assert!(matches!(
                    request_chat(backend, &admitted(false), &policy(), Instant::now()).await,
                    Err(BackendError::Deadline)
                ));
            }
            assert!(state.borrow().dropped);
            assert_eq!(state.borrow().read_polls, 0);
            assert_eq!(state.borrow().write_polls, 0);
        }
        for delayed_write in [false, true] {
            let mut backend = Stream::new(length_response(b"{}", false), usize::MAX);
            if delayed_write {
                backend.write_delay = Some(Duration::from_secs(2));
            } else {
                backend.read_delay = Some(Duration::from_secs(2));
            }
            let state = backend.state.clone();
            let start = Instant::now();
            assert!(matches!(
                request_chat(
                    backend,
                    &admitted(false),
                    &policy(),
                    start + Duration::from_secs(2)
                )
                .await,
                Err(BackendError::Deadline)
            ));
            assert_eq!(Instant::now() - start, Duration::from_secs(2));
            assert!(state.borrow().dropped);
        }
    });
}

#[test]
fn cancelling_request_or_body_collection_drops_the_owned_connection() {
    run(async {
        for blocked_write in [false, true] {
            let mut backend = Stream::new(Vec::new(), 1);
            backend.stall_write = blocked_write;
            backend.stall_after_input = true;
            let state = backend.state.clone();
            assert!(
                timeout(
                    Duration::from_millis(1),
                    request_chat(backend, &admitted(false), &policy(), deadline())
                )
                .await
                .is_err()
            );
            assert!(state.borrow().dropped);
        }
        let mut backend = Stream::new(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n"
                .to_vec(),
            usize::MAX,
        );
        backend.stall_after_input = true;
        let state = backend.state.clone();
        let mut body = request_chat(backend, &admitted(false), &policy(), deadline())
            .await
            .unwrap()
            .into_body();
        assert!(
            timeout(Duration::from_millis(1), async move {
                body.read_to_end(&mut Vec::new()).await
            })
            .await
            .is_err()
        );
        assert!(state.borrow().dropped);
    });
}
