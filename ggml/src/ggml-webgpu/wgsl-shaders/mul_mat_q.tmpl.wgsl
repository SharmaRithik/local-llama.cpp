#define(VARIANTS)
[
  {
    "SHADER_SUFFIX": "q4_0_vec",
    "REPLS": {
      "SRC0_TYPE" : "q4_0",
      "SRC1_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
      "BLOCK_SIZE": 32,
    },
    "DECLS": ["BYTE_HELPERS", "Q4_0_T"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(SRC1_F32_VEC)
fn zero_val_src1() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn store_val(acc: array<array<f32, TILE_Y>, TILE_X>, tx: u32, ty: u32) -> vec4<f32> {
    return vec4<f32>(acc[tx][ty], acc[tx][ty + 1], acc[tx][ty + 2], acc[tx][ty + 3]);
}

fn mul_acc(src0_val: {{SRC0_TYPE}}, src1_val: vec4<f32>) -> f32 {
    return dot(vec4<f32>(src0_val), src1_val);
}
#enddecl(SRC1_F32_VEC)

#decl(SRC1_F32)
fn zero_val_src1() -> f32 {
    return 0.0;
}

fn store_val(acc: array<array<f32, TILE_Y>, TILE_X>, tx: u32, ty: u32) -> f32 {
    return acc[tx][ty];
}

fn mul_acc(src0_val: {{SRC0_TYPE}}, src1_val: f32) -> f32 {
    return f32(src0_val) * src1_val;
}
#enddecl(SRC1_F32)

#end(DECLS)

#define(SHADER)
enable f16;

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

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // N rows, K columns.
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // M rows, K columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{SRC1_TYPE}}>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

DECLS

fn get_local_x(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_Y;
}
fn get_local_y(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_Y;
}

// Warning: cannot be overrides, must match values in ggml-webgpu.cpp
const TILE_X = 4u;
// must be multiple of 4 for vec4 loads
const TILE_Y = 4u;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override TILE_K: u32;

override TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;
override BLOCKS = max(1, TILE_K/{{BLOCK_SIZE}}); // the number of blocks we need to store at least TILE_K elements per thread. Note that since TILE_K may be less than BLOCK_SIZE, we need at least room for 1 block. Otherwise, TILE_K must be divisible by BLOCK_SIZE, or a clean fraction if it, or things will get weird.
override TILE_SRC0_SHMEM = BLOCKS * WORKGROUP_SIZE_Y * TILE_Y;
override TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_X * TILE_X;

var<workgroup> src0_shmem: array<{{SRC0_TYPE}}, TILE_SRC0_SHMEM>; // stores tiles of quantized weights without dequantizing
var<workgroup> src1_shmem: array<{{SRC1_TYPE}}, TILE_SRC1_SHMEM/{{VEC_SIZE}}>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

    let thread_id = local_id.x;
    let local_x = get_local_x(thread_id);
    let local_y = get_local_y(thread_id);

    let wg_linear = global_id.x / TOTAL_WORKGROUP_SIZE;

    let wg_x_count = (params.m + WORKGROUP_SIZE_X * TILE_X - 1u) / (WORKGROUP_SIZE_X * TILE_X);
    let wg_y_count = (params.n + WORKGROUP_SIZE_Y * TILE_Y - 1u) / (WORKGROUP_SIZE_Y * TILE_Y);
    let wg_per_matrix = wg_x_count * wg_y_count;

    let batch_idx = wg_linear / wg_per_matrix;

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_y = wg_in_batch % wg_y_count;
    let wg_x = wg_in_batch / wg_y_count;

    let output_row_base = wg_x * WORKGROUP_SIZE_X * TILE_X + local_x * TILE_X;
    let output_col_base = wg_y * WORKGROUP_SIZE_Y * TILE_Y + local_y * TILE_Y;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    var acc: array<array<f32, TILE_Y>, TILE_X>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        // we only load new blocks of src0 when the k_outer is at a block boundary
        if (k_outer % {{BLOCK_SIZE}} == 0) {
            // load src0 tile
            // need to figure out mapping from blocks to where they should be stored here
            for (var block_idx = thread_id; block_idx < TILE_SRC0_SHMEM; block_idx += TOTAL_WORKGROUP_SIZE) {
                let block_col = block_idx / BLOCKS;
                let block_start_idx = block_idx * {{BLOCK_SIZE}};
                let tile_col = block_start_idx / TILE_K;
                let tile_k = block_start_idx % TILE_K;
                let global_col = wg_y * WORKGROUP_SIZE_Y * TILE_Y + tile_col;
                let global_k = k_outer + tile_k;
                let src0_idx = src0_batch_offset + global_col * params.stride_01 + global_k;
            src0_shmem[block_idx] = select( // taking a slight performance hit to avoid oob
              zero_val_src0(),
              src0[src0_idx],
              global_col < params.n && global_k < params.k);
        }

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_row = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_row = wg_x * WORKGROUP_SIZE_X * TILE_X + tile_row;
            let global_k = k_outer + tile_k;

            let src1_idx = src1_batch_offset + global_row * params.stride_11 + global_k;
            src1_shmem[elem_idx/{{VEC_SIZE}}] = select(
              zero_val_src1(),
              src1[src1_idx/{{VEC_SIZE}}],
              global_row < params.m && global_k < params.k);
        }

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner += {{VEC_SIZE}}) {
            var src0_tile: array<{{SRC0_TYPE}}, TILE_Y>;
            for (var ty = 0u; ty < TILE_Y; ty++) {
                let src0_col = local_y * TILE_Y + ty;
                let src0_idx = k_inner + src0_col * TILE_K;
                src0_tile[ty] = src0_shmem[src0_idx/{{VEC_SIZE}}];
            }
            for (var tx = 0u; tx < TILE_X; tx++) {
                let src1_row = local_x * TILE_X + tx;
                let src1_idx = src1_row * TILE_K + k_inner;
                let src1_vec = src1_shmem[src1_idx/{{VEC_SIZE}}];
                for (var ty = 0u; ty < TILE_Y; ty++) {
                      acc[tx][ty] += mul_acc(src0_tile[ty], src1_vec);
                }
            }
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tx = 0u; tx < TILE_X; tx++) {
        let global_row = output_row_base + tx;
        if (global_row < params.m) {
            for (var ty = 0u; ty < TILE_Y; ty += {{VEC_SIZE}}) {
                let global_col = output_col_base + ty;
                if (global_col < params.n) {
                    let dst_idx = dst_batch_offset + global_row * params.n + global_col;
                    dst[dst_idx/{{VEC_SIZE}}] = store_val(acc, tx, ty);
                }
            }
        }
    }
}

#end(SHADER)
