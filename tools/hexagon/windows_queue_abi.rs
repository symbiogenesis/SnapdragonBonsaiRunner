//! Standalone native-layout comparison with the separately compiled public C headers.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::io::{self, Write};
use std::mem::offset_of;
use std::process::ExitCode;

#[repr(C)]
struct Buffer {
    fd: u32,
    size: u32,
    offset: u32,
    flags: u32,
    address: u64,
}

#[repr(C)]
struct Tensor {
    data: u64,
    size: u32,
    flags: u32,
    kind: u32,
    buffer: u16,
    tensor: u16,
    dimensions: [u32; 4],
    strides: [u32; 4],
}

#[repr(C)]
struct HtpBuffer {
    base: u64,
    size: u64,
    flags: u32,
    fd: u32,
}

#[repr(C)]
struct Operation {
    opcode: u32,
    flags: u32,
    params: [i32; 16],
    kernel_params: [i32; 32],
    source: [u16; 10],
    destination: [u16; 4],
    padding: [u16; 2],
}

#[repr(C)]
struct Request {
    sequence: u64,
    buffers: u32,
    tensors: u32,
    operations: u32,
    traces: u32,
}

#[repr(C)]
struct Response {
    sequence: u64,
    cycles_start: u64,
    cycles_stop: u64,
    status: u32,
    buffers: u32,
    tensors: u32,
    operations: u32,
    microseconds: u32,
    traces: [u32; 11],
}

#[repr(C)]
struct Profile {
    opcode: u32,
    microseconds: u32,
    cycles_start: u32,
    cycles_stop: u32,
    counters: [u32; 8],
}

fn layout<T>(offsets: &[(&str, usize)]) -> String {
    let mut fields = format!(
        "\"bytes\":{},\"alignment\":{}",
        size_of::<T>(),
        align_of::<T>()
    );
    for (name, offset) in offsets {
        fields.push_str(&format!(",\"{name}\":{offset}"));
    }
    format!("{{{fields}}}")
}

fn layouts() -> String {
    let observed = [
        ("result", layout::<i32>(&[])),
        ("queue", layout::<*mut std::ffi::c_void>(&[])),
        (
            "queueBuffer",
            layout::<Buffer>(&[
                ("fd", offset_of!(Buffer, fd)),
                ("size", offset_of!(Buffer, size)),
                ("offset", offset_of!(Buffer, offset)),
                ("flags", offset_of!(Buffer, flags)),
                ("ptr", offset_of!(Buffer, address)),
                ("address", offset_of!(Buffer, address)),
            ]),
        ),
        (
            "tensor",
            layout::<Tensor>(&[
                ("data", offset_of!(Tensor, data)),
                ("size", offset_of!(Tensor, size)),
                ("flags", offset_of!(Tensor, flags)),
                ("type", offset_of!(Tensor, kind)),
                ("bi", offset_of!(Tensor, buffer)),
                ("ti", offset_of!(Tensor, tensor)),
                ("ne", offset_of!(Tensor, dimensions)),
                ("nb", offset_of!(Tensor, strides)),
            ]),
        ),
        (
            "buffer",
            layout::<HtpBuffer>(&[
                ("base", offset_of!(HtpBuffer, base)),
                ("size", offset_of!(HtpBuffer, size)),
                ("flags", offset_of!(HtpBuffer, flags)),
                ("fd", offset_of!(HtpBuffer, fd)),
            ]),
        ),
        (
            "operation",
            layout::<Operation>(&[
                ("opcode", offset_of!(Operation, opcode)),
                ("flags", offset_of!(Operation, flags)),
                ("params", offset_of!(Operation, params)),
                ("kernel_params", offset_of!(Operation, kernel_params)),
                ("src", offset_of!(Operation, source)),
                ("dst", offset_of!(Operation, destination)),
                ("pad", offset_of!(Operation, padding)),
            ]),
        ),
        (
            "request",
            layout::<Request>(&[
                ("seq", offset_of!(Request, sequence)),
                ("n_bufs", offset_of!(Request, buffers)),
                ("n_tensors", offset_of!(Request, tensors)),
                ("n_ops", offset_of!(Request, operations)),
                ("n_traces", offset_of!(Request, traces)),
            ]),
        ),
        (
            "response",
            layout::<Response>(&[
                ("seq", offset_of!(Response, sequence)),
                ("cycles_start", offset_of!(Response, cycles_start)),
                ("cycles_stop", offset_of!(Response, cycles_stop)),
                ("status", offset_of!(Response, status)),
                ("n_bufs", offset_of!(Response, buffers)),
                ("n_tensors", offset_of!(Response, tensors)),
                ("n_ops", offset_of!(Response, operations)),
                ("usecs", offset_of!(Response, microseconds)),
                ("n_traces", offset_of!(Response, traces)),
            ]),
        ),
        (
            "profile",
            layout::<Profile>(&[
                ("opcode", offset_of!(Profile, opcode)),
                ("usecs", offset_of!(Profile, microseconds)),
                ("cycles_start", offset_of!(Profile, cycles_start)),
                ("cycles_stop", offset_of!(Profile, cycles_stop)),
                ("pmu", offset_of!(Profile, counters)),
            ]),
        ),
    ];
    observed
        .into_iter()
        .map(|(name, value)| format!("\"{name}\":{value}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> ExitCode {
    if std::env::args_os().len() != 1 || !cfg!(all(windows, target_arch = "aarch64")) {
        return ExitCode::FAILURE;
    }
    let buffer = Buffer {
        fd: 0xfedc_ba98,
        size: 0x1020_3040,
        offset: 0x0102_0304,
        flags: 4 | 16 | 128,
        address: 0x8123_4567_89ab_cdef,
    };
    // Encode the declared fields independently; no unsafe object byte reads or copied C output.
    let bytes: Vec<u8> = [buffer.fd, buffer.size, buffer.offset, buffer.flags]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(buffer.address.to_le_bytes())
        .collect();
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    let output = format!(
        concat!(
            "{{\"schemaVersion\":2,\"kind\":\"windows-arm64-Rust-queue-abi-control\",",
            "\"queueBuffer\":{{\"bytes\":{},\"alignment\":{},\"fdOffset\":{},",
            "\"sizeOffset\":{},\"offsetOffset\":{},\"flagsOffset\":{},\"addressOffset\":{}}},",
            "\"htp\":{{\"tensorBytes\":{},\"bufferBytes\":{},\"operationBytes\":{},",
            "\"requestBytes\":{},\"responseBytes\":{},\"profileBytes\":{}}},",
            "\"infiniteTimeout\":{},\"bufferHex\":\"{}\",\"layouts\":{{{}}},",
            "\"constants\":{{\"packetMessage\":1,\"packetBuffers\":2,",
            "\"reference\":4,\"dereference\":8,\"flushSender\":16,",
            "\"invalidateSender\":32,\"flushRecipient\":64,\"invalidateRecipient\":128,",
            "\"multiplyMatrix\":4,\"float32\":0,\"q2_0\":42,\"ok\":1}},",
            "\"driverLoaded\":false,\"deviceExecuted\":false,\"profileEligible\":false}}\n"
        ),
        size_of::<Buffer>(),
        align_of::<Buffer>(),
        offset_of!(Buffer, fd),
        offset_of!(Buffer, size),
        offset_of!(Buffer, offset),
        offset_of!(Buffer, flags),
        offset_of!(Buffer, address),
        size_of::<Tensor>(),
        size_of::<HtpBuffer>(),
        size_of::<Operation>(),
        size_of::<Request>(),
        size_of::<Response>(),
        size_of::<Profile>(),
        u32::MAX,
        hex,
        layouts(),
    );
    match io::stdout().lock().write_all(output.as_bytes()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => ExitCode::FAILURE,
    }
}
