enable f16;

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    return f32(src0[src0_idx_base + offset]) * f32(src1[src1_idx_base + offset]);
}

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,

    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<f16>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total = params.m * params.n * params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (global_id.x >= total) {
        return;
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = global_id.x / dst3_stride;
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst3_rem = global_id.x % dst3_stride;

    let dst2_idx = dst3_rem / dst2_stride;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let dst2_rem = dst3_rem % dst2_stride;

    let row = dst2_rem / params.n;
    let col = dst2_rem % params.n;

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12 + row * params.stride_11;

    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k/1; i = i + 1u) {
        sum += multiply_add(src0_idx_base, src1_idx_base, i);
    }
    dst[params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row * params.n + col] = sum;
}