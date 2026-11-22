//! One fixed original-PQ2 integer projection, owned by a supervised child.

use super::{Api, Id, MAX_DEVICES, check, invalid};
use crate::{
    AdrenoFamily, OpenClDevice, OpenClKernelKind, OpenClPq2Projection, OpenClProjectionSelection,
};
use std::ffi::{CStr, c_char, c_void};
use std::io;

type Context = unsafe extern "system" fn(
    *const isize,
    u32,
    *const Id,
    Option<unsafe extern "system" fn(*const c_char, *const c_void, usize, *mut c_void)>,
    *mut c_void,
    *mut i32,
) -> Id;
type Queue = unsafe extern "system" fn(Id, Id, u64, *mut i32) -> Id;
type Buffer = unsafe extern "system" fn(Id, u64, usize, *mut c_void, *mut i32) -> Id;
type Program =
    unsafe extern "system" fn(Id, u32, *const *const c_char, *const usize, *mut i32) -> Id;
type Build = unsafe extern "system" fn(
    Id,
    u32,
    *const Id,
    *const c_char,
    Option<unsafe extern "system" fn(Id, *mut c_void)>,
    *mut c_void,
) -> i32;
type Kernel = unsafe extern "system" fn(Id, *const c_char, *mut i32) -> Id;
type Argument = unsafe extern "system" fn(Id, u32, usize, *const c_void) -> i32;
type Enqueue = unsafe extern "system" fn(
    Id,
    Id,
    u32,
    *const usize,
    *const usize,
    *const usize,
    u32,
    *const Id,
    *mut Id,
) -> i32;
type Read = unsafe extern "system" fn(
    Id,
    Id,
    u32,
    usize,
    usize,
    *mut c_void,
    u32,
    *const Id,
    *mut Id,
) -> i32;
type Release = unsafe extern "system" fn(Id) -> i32;

// No optional extensions, floating arithmetic, subgroup widths or vendor binaries.
// One work-item preserves one original 128-symbol PQ2 integer reduction. The host
// retains the original half scales and CPU-prepared Q8_K scale/FMA order.
pub(crate) const SOURCE: &str = r"
__kernel void pq2_q8k_block_dots(__global const uchar *weights,
                                __global const char *codes,
                                __global int *output) {
    const size_t block = get_global_id(0);
    const size_t offset = block * 34 + 2;
    const size_t activation = (block % 40) * 128;
    int sum = 0;
    for (uint lane = 0; lane < 128; ++lane) {
        const uint symbol = (weights[offset + lane / 4] >> (2 * (lane % 4))) & 3;
        sum += (convert_int(symbol) - 1) * convert_int(codes[activation + lane]);
    }
    output[block] = sum;
}
";

const GEMV_4ROW_SOURCE: &str = r"
__kernel void pq2_q8k_gemv_4row(__global const uchar *weights,
                                __global const char *codes,
                                __global int *output,
                                const uint blocks_per_row) {
    const size_t gid = get_global_id(0);
    const size_t col_block = gid % blocks_per_row;
    const size_t row_base = (gid / blocks_per_row) * 4;
    const size_t act_base = col_block * 128;
    const size_t r0 = (row_base * blocks_per_row + col_block) * 34 + 2;
    const size_t r1 = ((row_base + 1) * blocks_per_row + col_block) * 34 + 2;
    const size_t r2 = ((row_base + 2) * blocks_per_row + col_block) * 34 + 2;
    const size_t r3 = ((row_base + 3) * blocks_per_row + col_block) * 34 + 2;
    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    for (uint lane = 0; lane < 128; ++lane) {
        const int act = convert_int(codes[act_base + lane]);
        const uint byte_off = lane / 4;
        const uint shift = 2 * (lane % 4);
        sum0 += (convert_int((weights[r0 + byte_off] >> shift) & 3U) - 1) * act;
        sum1 += (convert_int((weights[r1 + byte_off] >> shift) & 3U) - 1) * act;
        sum2 += (convert_int((weights[r2 + byte_off] >> shift) & 3U) - 1) * act;
        sum3 += (convert_int((weights[r3 + byte_off] >> shift) & 3U) - 1) * act;
    }
    output[row_base * blocks_per_row + col_block] = sum0;
    output[(row_base + 1) * blocks_per_row + col_block] = sum1;
    output[(row_base + 2) * blocks_per_row + col_block] = sum2;
    output[(row_base + 3) * blocks_per_row + col_block] = sum3;
}
";

const GEMV_8ROW_SOURCE: &str = r"
__kernel void pq2_q8k_gemv_8row(__global const uchar *weights,
                                __global const char *codes,
                                __global int *output,
                                const uint blocks_per_row) {
    const size_t gid = get_global_id(0);
    const size_t col_block = gid % blocks_per_row;
    const size_t row_base = (gid / blocks_per_row) * 8;
    const size_t act_base = col_block * 128;
    int sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint lane = 0; lane < 128; ++lane) {
        const int act = convert_int(codes[act_base + lane]);
        const uint byte_off = lane / 4;
        const uint shift = 2 * (lane % 4);
        for (uint r = 0; r < 8; ++r) {
            const size_t offset = ((row_base + r) * blocks_per_row + col_block) * 34 + 2;
            const uint symbol = (weights[offset + byte_off] >> shift) & 3U;
            sums[r] += (convert_int(symbol) - 1) * act;
        }
    }
    for (uint r = 0; r < 8; ++r) {
        output[(row_base + r) * blocks_per_row + col_block] = sums[r];
    }
}
";

const PREFILL_4COL_SOURCE: &str = r"
__kernel void pq2_q8k_prefill_4col(__global const uchar *weights,
                                   __global const char *codes,
                                   __global int *output,
                                   const uint blocks_per_row,
                                   const uint total_blocks) {
    const size_t gid = get_global_id(0);
    const size_t block = gid % total_blocks;
    const size_t col_block = block % blocks_per_row;
    const size_t token_base = (gid / total_blocks) * 4;
    const size_t weight_base = block * 34 + 2;
    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    for (uint lane = 0; lane < 128; ++lane) {
        const uint symbol = (weights[weight_base + lane / 4] >> (2 * (lane % 4))) & 3U;
        const int weight = convert_int(symbol) - 1;
        sum0 += weight * convert_int(codes[((token_base + 0) * blocks_per_row + col_block) * 128 + lane]);
        sum1 += weight * convert_int(codes[((token_base + 1) * blocks_per_row + col_block) * 128 + lane]);
        sum2 += weight * convert_int(codes[((token_base + 2) * blocks_per_row + col_block) * 128 + lane]);
        sum3 += weight * convert_int(codes[((token_base + 3) * blocks_per_row + col_block) * 128 + lane]);
    }
    output[(token_base + 0) * total_blocks + block] = sum0;
    output[(token_base + 1) * total_blocks + block] = sum1;
    output[(token_base + 2) * total_blocks + block] = sum2;
    output[(token_base + 3) * total_blocks + block] = sum3;
}
";

const FWHT_LOCAL_1024_SOURCE: &str = r"
__kernel void fwht_local_1024(__global const uchar *weights,
                              __global const char *codes,
                              __global int *output) {
    (void)weights;
    __local int scratch[1024];
    const size_t token = get_group_id(0);
    const size_t lid = get_local_id(0);
    const __global uchar *src = (const __global uchar *)codes;
    for (uint k = 0; k < 4; ++k) {
        const size_t idx = lid + k * 256;
        const size_t byte_offset = (token * 1024 + idx) * 4;
        const uint raw = convert_uint(src[byte_offset])
            | (convert_uint(src[byte_offset + 1]) << 8)
            | (convert_uint(src[byte_offset + 2]) << 16)
            | (convert_uint(src[byte_offset + 3]) << 24);
        scratch[idx] = as_int(raw);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 1; stride < 1024; stride *= 2) {
        for (uint b = 0; b < 2; ++b) {
            const size_t pair = lid + b * 256;
            const size_t pos = (pair / stride) * (stride * 2) + (pair % stride);
            const int a = scratch[pos];
            const int c = scratch[pos + stride];
            scratch[pos] = a + c;
            scratch[pos + stride] = a - c;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint k = 0; k < 4; ++k) {
        const size_t idx = lid + k * 256;
        output[token * 1024 + idx] = scratch[idx];
    }
}
";

pub(crate) const fn source_for(kind: OpenClKernelKind) -> &'static str {
    match kind {
        OpenClKernelKind::BlockDots => SOURCE,
        OpenClKernelKind::Gemv4Row => GEMV_4ROW_SOURCE,
        OpenClKernelKind::Gemv8Row => GEMV_8ROW_SOURCE,
        OpenClKernelKind::Prefill4Col => PREFILL_4COL_SOURCE,
        OpenClKernelKind::FwhtLocal1024 => FWHT_LOCAL_1024_SOURCE,
    }
}

const fn entry_cstr(kind: OpenClKernelKind) -> &'static CStr {
    match kind {
        OpenClKernelKind::BlockDots => c"pq2_q8k_block_dots",
        OpenClKernelKind::Gemv4Row => c"pq2_q8k_gemv_4row",
        OpenClKernelKind::Gemv8Row => c"pq2_q8k_gemv_8row",
        OpenClKernelKind::Prefill4Col => c"pq2_q8k_prefill_4col",
        OpenClKernelKind::FwhtLocal1024 => c"fwht_local_1024",
    }
}

struct Compute {
    context: Context,
    queue: Queue,
    buffer: Buffer,
    program: Program,
    build: Build,
    kernel: Kernel,
    argument: Argument,
    enqueue: Enqueue,
    read: Read,
    finish: Release,
    release_context: Release,
    release_queue: Release,
    release_buffer: Release,
    release_program: Release,
    release_kernel: Release,
}

impl Compute {
    fn load(api: &Api) -> io::Result<Self> {
        use super::super::library::load_symbol;
        Ok(Self {
            context: load_symbol!(api._library, c"clCreateContext", Context),
            queue: load_symbol!(api._library, c"clCreateCommandQueue", Queue),
            buffer: load_symbol!(api._library, c"clCreateBuffer", Buffer),
            program: load_symbol!(api._library, c"clCreateProgramWithSource", Program),
            build: load_symbol!(api._library, c"clBuildProgram", Build),
            kernel: load_symbol!(api._library, c"clCreateKernel", Kernel),
            argument: load_symbol!(api._library, c"clSetKernelArg", Argument),
            enqueue: load_symbol!(api._library, c"clEnqueueNDRangeKernel", Enqueue),
            read: load_symbol!(api._library, c"clEnqueueReadBuffer", Read),
            finish: load_symbol!(api._library, c"clFinish", Release),
            release_context: load_symbol!(api._library, c"clReleaseContext", Release),
            release_queue: load_symbol!(api._library, c"clReleaseCommandQueue", Release),
            release_buffer: load_symbol!(api._library, c"clReleaseMemObject", Release),
            release_program: load_symbol!(api._library, c"clReleaseProgram", Release),
            release_kernel: load_symbol!(api._library, c"clReleaseKernel", Release),
        })
    }
}

struct Owner<'a> {
    api: &'a Compute,
    context: Id,
    queue: Id,
    program: Id,
    kernel: Id,
    buffers: [Id; 3],
    submitted: bool,
}

impl Owner<'_> {
    fn drain(&mut self) {
        if self.submitted {
            // SAFETY: The queue remains live. A successful fence ends all
            // asynchronous use before releasing any object or unloading code.
            if unsafe { (self.api.finish)(self.queue) } != 0 {
                // Retain both this owner and any caller readback storage while
                // terminal completion is unknown, until the supervisor exits us.
                loop {
                    std::thread::park();
                }
            }
            self.submitted = false;
        }
    }

    fn close(&mut self) -> [i32; 7] {
        self.drain();
        let mut codes = [0; 7];
        let [first, second, third] = &mut self.buffers;
        for (index, (id, release)) in [
            (&mut self.kernel, self.api.release_kernel),
            (&mut self.program, self.api.release_program),
            (third, self.api.release_buffer),
            (second, self.api.release_buffer),
            (first, self.api.release_buffer),
            (&mut self.queue, self.api.release_queue),
            (&mut self.context, self.api.release_context),
        ]
        .into_iter()
        .enumerate()
        {
            if !id.is_null() {
                // SAFETY: Each owned object is released once after a terminal
                // fence, while the implementation and context remain live.
                codes[index] = unsafe { release(*id) };
                *id = std::ptr::null_mut();
            }
        }
        codes
    }
}

impl Drop for Owner<'_> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn object(id: Id, code: i32) -> io::Result<()> {
    check(code)?;
    if id.is_null() { Err(invalid()) } else { Ok(()) }
}

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const BLOCK_DOTS_FIXED_COLUMNS: usize = 5120;
const BLOCK_DOTS_BLOCKS_PER_ROW: usize = 40;
const FWHT_LOCAL_WORK_SIZE: usize = 256;
const FWHT_COLUMNS: usize = 1024;

struct LaunchSpec<'a> {
    selection: &'a OpenClProjectionSelection,
    family: AdrenoFamily,
    source: &'static str,
    entry_point: &'static CStr,
    extra_uint_args: [Option<u32>; 2],
    global_work: usize,
    local_work: Option<usize>,
    output_len: usize,
    output_bytes: usize,
    weights: &'a [u8],
    codes: &'a [u8],
}

fn selected_for_family(
    api: &Api,
    selection: &OpenClProjectionSelection,
    family: AdrenoFamily,
) -> io::Result<(Id, OpenClDevice)> {
    let (platforms, count) = api.platform_ids(false)?;
    let mut selected = None;
    let mut remaining = MAX_DEVICES;
    for platform in &platforms[..count] {
        let (devices, count) = api.platform_gpus(*platform, &mut remaining)?;
        if count == 0 {
            continue;
        }
        let name = api.string(*platform, 0x0902, true)?;
        for id in &devices[..count] {
            let device = api.device(*id)?;
            if name == selection.platform_name
                && device.name == selection.device_name
                && device.driver_version == selection.driver_version
                && device.vendor_id == selection.vendor_id
            {
                if selected.is_some()
                    || !device.available
                    || !device.compiler_available
                    || device.address_bits != 64
                    || AdrenoFamily::from_device_name(&device.name) != Some(family)
                {
                    return Err(invalid());
                }
                selected = Some((*id, device));
            }
        }
    }
    let unavailable = match family {
        AdrenoFamily::X1_85 => "exact Adreno X1-85 OpenCL selection unavailable",
        AdrenoFamily::X2_90 => "exact Adreno X2 OpenCL selection unavailable",
    };
    selected.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, unavailable))
}

fn validate_buffer_caps(
    weights_len: usize,
    codes_len: usize,
    output_len: usize,
) -> io::Result<usize> {
    let output_bytes = output_len
        .checked_mul(size_of::<i32>())
        .ok_or_else(invalid)?;
    if weights_len == 0
        || codes_len == 0
        || output_bytes == 0
        || weights_len > MAX_BUFFER_BYTES
        || codes_len > MAX_BUFFER_BYTES
        || output_bytes > MAX_BUFFER_BYTES
    {
        return Err(invalid());
    }
    Ok(output_bytes)
}

fn build_spec<'a>(
    selection: &'a OpenClProjectionSelection,
    family: AdrenoFamily,
    kind: OpenClKernelKind,
    rows: usize,
    columns: usize,
    tokens: usize,
    weights: &'a [u8],
    codes: &'a [u8],
) -> io::Result<LaunchSpec<'a>> {
    let (extra_uint_args, global_work, local_work, output_len) = match kind {
        OpenClKernelKind::BlockDots => {
            let expected_weights = rows
                .checked_mul(BLOCK_DOTS_BLOCKS_PER_ROW)
                .and_then(|blocks| blocks.checked_mul(34))
                .ok_or_else(invalid)?;
            let work = rows
                .checked_mul(BLOCK_DOTS_BLOCKS_PER_ROW)
                .ok_or_else(invalid)?;
            if rows == 0
                || columns != BLOCK_DOTS_FIXED_COLUMNS
                || tokens != 1
                || weights.len() != expected_weights
                || codes.len() != BLOCK_DOTS_FIXED_COLUMNS
            {
                return Err(invalid());
            }
            ([None, None], work, None, work)
        }
        OpenClKernelKind::Gemv4Row | OpenClKernelKind::Gemv8Row => {
            let row_tile = if kind == OpenClKernelKind::Gemv4Row {
                4
            } else {
                8
            };
            if rows == 0
                || !rows.is_multiple_of(row_tile)
                || columns == 0
                || !columns.is_multiple_of(128)
                || tokens != 1
            {
                return Err(invalid());
            }
            let blocks_per_row = columns / 128;
            let blocks_per_row_u32 = u32::try_from(blocks_per_row).map_err(|_| invalid())?;
            let total_blocks = rows.checked_mul(blocks_per_row).ok_or_else(invalid)?;
            let expected_weights = total_blocks.checked_mul(34).ok_or_else(invalid)?;
            if weights.len() != expected_weights || codes.len() != columns {
                return Err(invalid());
            }
            let work = (rows / row_tile)
                .checked_mul(blocks_per_row)
                .ok_or_else(invalid)?;
            ([Some(blocks_per_row_u32), None], work, None, total_blocks)
        }
        OpenClKernelKind::Prefill4Col => {
            if rows == 0
                || columns == 0
                || !columns.is_multiple_of(128)
                || tokens == 0
                || !tokens.is_multiple_of(4)
            {
                return Err(invalid());
            }
            let blocks_per_row = columns / 128;
            let total_blocks = rows.checked_mul(blocks_per_row).ok_or_else(invalid)?;
            let expected_weights = total_blocks.checked_mul(34).ok_or_else(invalid)?;
            let expected_codes = tokens.checked_mul(columns).ok_or_else(invalid)?;
            let output_len = tokens.checked_mul(total_blocks).ok_or_else(invalid)?;
            let blocks_per_row_u32 = u32::try_from(blocks_per_row).map_err(|_| invalid())?;
            let total_blocks_u32 = u32::try_from(total_blocks).map_err(|_| invalid())?;
            if weights.len() != expected_weights || codes.len() != expected_codes {
                return Err(invalid());
            }
            let work = (tokens / 4).checked_mul(total_blocks).ok_or_else(invalid)?;
            (
                [Some(blocks_per_row_u32), Some(total_blocks_u32)],
                work,
                None,
                output_len,
            )
        }
        OpenClKernelKind::FwhtLocal1024 => {
            let expected_codes = tokens
                .checked_mul(FWHT_COLUMNS * size_of::<i32>())
                .ok_or_else(invalid)?;
            let output_len = tokens.checked_mul(FWHT_COLUMNS).ok_or_else(invalid)?;
            let work = tokens
                .checked_mul(FWHT_LOCAL_WORK_SIZE)
                .ok_or_else(invalid)?;
            if rows != 1
                || columns != FWHT_COLUMNS
                || tokens == 0
                || weights.len() != 4
                || codes.len() != expected_codes
            {
                return Err(invalid());
            }
            ([None, None], work, Some(FWHT_LOCAL_WORK_SIZE), output_len)
        }
    };
    let output_bytes = validate_buffer_caps(weights.len(), codes.len(), output_len)?;
    Ok(LaunchSpec {
        selection,
        family,
        source: source_for(kind),
        entry_point: entry_cstr(kind),
        extra_uint_args,
        global_work,
        local_work,
        output_len,
        output_bytes,
        weights,
        codes,
    })
}

pub(crate) fn run(
    selection: &OpenClProjectionSelection,
    weights: &[u8],
    codes: &[u8],
) -> io::Result<OpenClPq2Projection> {
    run_with_kernel(
        selection,
        AdrenoFamily::X2_90,
        OpenClKernelKind::BlockDots,
        17_408,
        BLOCK_DOTS_FIXED_COLUMNS,
        1,
        weights,
        codes,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Matches the public parameterized OpenCL dispatch boundary without bundling a parameter struct."
)]
pub(crate) fn run_with_kernel(
    selection: &OpenClProjectionSelection,
    family: AdrenoFamily,
    kind: OpenClKernelKind,
    rows: usize,
    columns: usize,
    tokens: usize,
    weights: &[u8],
    codes: &[u8],
) -> io::Result<OpenClPq2Projection> {
    super::validate_environment()?;
    let spec = build_spec(
        selection, family, kind, rows, columns, tokens, weights, codes,
    )?;
    execute(&spec)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep the single audited creation, submission, readback and terminal-fence sequence together so object ownership remains reviewable."
)]
fn execute(spec: &LaunchSpec<'_>) -> io::Result<OpenClPq2Projection> {
    let output_bytes = spec.output_bytes;
    let api = Api::load()?;
    let (device_id, device) = selected_for_family(&api, spec.selection, spec.family)?;
    if [spec.weights.len(), spec.codes.len(), output_bytes]
        .iter()
        .any(|&size| {
            u64::try_from(size).map_or(true, |size| size > device.maximum_allocation_bytes)
        })
    {
        return Err(invalid());
    }
    let functions = Compute::load(&api)?;
    let mut owner = Owner {
        api: &functions,
        context: std::ptr::null_mut(),
        queue: std::ptr::null_mut(),
        program: std::ptr::null_mut(),
        kernel: std::ptr::null_mut(),
        buffers: [std::ptr::null_mut(); 3],
        submitted: false,
    };
    let mut code = 0;
    // SAFETY: One selected live device; no callback/user pointer is installed.
    owner.context = unsafe {
        (functions.context)(
            std::ptr::null(),
            1,
            &raw const device_id,
            None,
            std::ptr::null_mut(),
            &raw mut code,
        )
    };
    object(owner.context, code)?;
    // SAFETY: Context/device are live and associated. Zero selects an in-order
    // queue without profiling, optional flags or out-of-order execution.
    owner.queue = unsafe { (functions.queue)(owner.context, device_id, 0, &raw mut code) };
    object(owner.queue, code)?;
    for (index, bytes) in [spec.weights, spec.codes].into_iter().enumerate() {
        // SAFETY: COPY_HOST_PTR copies the bounded live input during this call;
        // no host address remains borrowed by the asynchronous queue afterward.
        owner.buffers[index] = unsafe {
            (functions.buffer)(
                owner.context,
                4 | 32,
                bytes.len(),
                bytes.as_ptr().cast_mut().cast(),
                &raw mut code,
            )
        };
        object(owner.buffers[index], code)?;
    }
    // SAFETY: Kernel-write-only object, fixed checked size and no host pointer.
    owner.buffers[2] = unsafe {
        (functions.buffer)(
            owner.context,
            2,
            output_bytes,
            std::ptr::null_mut(),
            &raw mut code,
        )
    };
    object(owner.buffers[2], code)?;
    let source = spec.source.as_ptr().cast::<c_char>();
    let length = spec.source.len();
    // SAFETY: One source string with an explicit byte length, copied by OpenCL.
    owner.program = unsafe {
        (functions.program)(
            owner.context,
            1,
            &raw const source,
            &raw const length,
            &raw mut code,
        )
    };
    object(owner.program, code)?;
    // SAFETY: Synchronous build for the same live device. Fixed options install
    // no callback and request OpenCL C 1.2 without fast-math/reassociation flags.
    check(unsafe {
        (functions.build)(
            owner.program,
            1,
            &raw const device_id,
            c"-cl-std=CL1.2".as_ptr(),
            None,
            std::ptr::null_mut(),
        )
    })?;
    // SAFETY: Fixed kernel name from the owned successfully built program.
    owner.kernel =
        unsafe { (functions.kernel)(owner.program, spec.entry_point.as_ptr(), &raw mut code) };
    object(owner.kernel, code)?;
    for (index, buffer) in (0_u32..3).zip(&owner.buffers) {
        // SAFETY: OpenCL copies one live cl_mem handle of the exact native size.
        check(unsafe {
            (functions.argument)(
                owner.kernel,
                index,
                size_of::<Id>(),
                (&raw const *buffer).cast(),
            )
        })?;
    }
    for (slot, extra) in (3_u32..5).zip(spec.extra_uint_args) {
        if let Some(value) = extra {
            // SAFETY: OpenCL copies one 32-bit unsigned scalar argument by value.
            check(unsafe {
                (functions.argument)(
                    owner.kernel,
                    slot,
                    size_of::<u32>(),
                    (&raw const value).cast(),
                )
            })?;
        }
    }
    let local_ptr = spec
        .local_work
        .as_ref()
        .map_or(std::ptr::null(), |size| &raw const *size);
    // SAFETY: One-dimensional checked work range and validated optional local size;
    // all referenced device buffers stay owned until queue drain.
    check(unsafe {
        (functions.enqueue)(
            owner.queue,
            owner.kernel,
            1,
            std::ptr::null(),
            &raw const spec.global_work,
            local_ptr,
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    })?;
    owner.submitted = true;
    let mut dots = vec![0_i32; spec.output_len];
    // SAFETY: Blocking read into the full aligned i32 array. Its lifetime spans
    // this call and the following terminal queue fence, including error paths.
    let read = unsafe {
        (functions.read)(
            owner.queue,
            owner.buffers[2],
            1,
            0,
            output_bytes,
            dots.as_mut_ptr().cast(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    // Drain before an error can drop the host readback array.
    owner.drain();
    check(read)?;
    let cleanup_codes = owner.close();
    Ok(OpenClPq2Projection {
        loader_path: api.path,
        device,
        integer_dots: dots,
        cleanup_codes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_selection(device_name: &str) -> OpenClProjectionSelection {
        OpenClProjectionSelection {
            platform_name: "Qualcomm Snapdragon".into(),
            device_name: device_name.into(),
            driver_version: "OpenCL 3.0 QUALCOMM".into(),
            vendor_id: 0x5143,
        }
    }

    #[test]
    fn adreno_family_recognition_covers_x1_85_and_x2_90_and_rejects_ambiguous() {
        assert_eq!(
            AdrenoFamily::from_device_name("Qualcomm(R) Adreno(TM) X1-85 GPU"),
            Some(AdrenoFamily::X1_85)
        );
        assert_eq!(
            AdrenoFamily::from_device_name("Qualcomm(R) Adreno(TM) X2-90 GPU"),
            Some(AdrenoFamily::X2_90)
        );
        assert_eq!(
            AdrenoFamily::from_device_name("Adreno X1"),
            Some(AdrenoFamily::X1_85)
        );
        assert_eq!(
            AdrenoFamily::from_device_name("Adreno X2"),
            Some(AdrenoFamily::X2_90)
        );
        for rejected in [
            "",
            "Adreno GPU",
            "Adreno X1-85 X2-90",
            "NVIDIA GeForce X1",
            "Adreno\nX1-85",
        ] {
            assert_eq!(AdrenoFamily::from_device_name(rejected), None);
        }
        assert_eq!(
            AdrenoFamily::from_device_name(&"Adreno X1 ".repeat(60)),
            None
        );
    }

    #[test]
    fn opencl_kernel_suite_preserves_block_dots_and_stays_pure_cl12_integer() {
        assert_eq!(
            crate::opencl_pq2_kernel_source_for(OpenClKernelKind::BlockDots),
            crate::opencl_pq2_kernel_source()
        );
        let kinds = [
            OpenClKernelKind::BlockDots,
            OpenClKernelKind::Gemv4Row,
            OpenClKernelKind::Gemv8Row,
            OpenClKernelKind::Prefill4Col,
            OpenClKernelKind::FwhtLocal1024,
        ];
        for kind in kinds {
            let source = crate::opencl_pq2_kernel_source_for(kind);
            assert!(source.contains(kind.entry_point()));
            let lower = source.to_ascii_lowercase();
            for forbidden in ["float", "double", "half", "cl_khr"] {
                assert!(
                    !lower.contains(forbidden),
                    "{:?} contains {forbidden}",
                    kind
                );
            }
        }
    }

    #[test]
    fn opencl_kernel_geometry_validation_rejects_unaligned_and_oversized_before_icd() {
        let selection = dummy_selection("Qualcomm(R) Adreno(TM) X1-85 GPU");
        let empty = &[];
        let four = [0_u8; 4];
        let act_256 = vec![0_u8; 256];
        let weights_4x2 = vec![0_u8; 4 * 2 * 34];

        assert!(
            build_spec(
                &selection,
                AdrenoFamily::X1_85,
                OpenClKernelKind::Gemv4Row,
                3,
                256,
                1,
                &weights_4x2[..3 * 2 * 34],
                &act_256,
            )
            .is_err()
        );
        assert!(
            build_spec(
                &selection,
                AdrenoFamily::X1_85,
                OpenClKernelKind::Gemv8Row,
                4,
                256,
                1,
                &weights_4x2,
                &act_256,
            )
            .is_err()
        );
        assert!(
            build_spec(
                &selection,
                AdrenoFamily::X1_85,
                OpenClKernelKind::Gemv4Row,
                4,
                192,
                1,
                empty,
                empty,
            )
            .is_err()
        );
        assert!(
            build_spec(
                &selection,
                AdrenoFamily::X1_85,
                OpenClKernelKind::Prefill4Col,
                4,
                256,
                3,
                &weights_4x2,
                &vec![0_u8; 3 * 256],
            )
            .is_err()
        );
        assert!(
            build_spec(
                &selection,
                AdrenoFamily::X1_85,
                OpenClKernelKind::FwhtLocal1024,
                1,
                512,
                1,
                &four,
                &vec![0_u8; 4096],
            )
            .is_err()
        );
        assert!(
            build_spec(
                &selection,
                AdrenoFamily::X1_85,
                OpenClKernelKind::FwhtLocal1024,
                1,
                1024,
                0,
                &four,
                empty,
            )
            .is_err()
        );

        let ok_gemv4 = build_spec(
            &selection,
            AdrenoFamily::X1_85,
            OpenClKernelKind::Gemv4Row,
            4,
            256,
            1,
            &weights_4x2,
            &act_256,
        );
        assert!(ok_gemv4.is_ok());
        let fwht_codes = vec![0_u8; 2 * 4096];
        let ok_fwht = build_spec(
            &selection,
            AdrenoFamily::X1_85,
            OpenClKernelKind::FwhtLocal1024,
            1,
            1024,
            2,
            &four,
            &fwht_codes,
        );
        assert!(ok_fwht.is_ok());

        let public_err = crate::opencl_pq2_projection_with_kernel(
            &selection,
            AdrenoFamily::X1_85,
            OpenClKernelKind::Gemv4Row,
            3,
            256,
            1,
            &weights_4x2[..3 * 2 * 34],
            &act_256,
        );
        assert!(public_err.is_err());
    }
}
