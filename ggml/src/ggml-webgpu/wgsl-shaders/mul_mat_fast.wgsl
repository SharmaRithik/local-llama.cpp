enable f16;

const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 8u;
const TOTAL_WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;
const TILE_X: u32 = 4u;
const TILE_Y: u32 = 4u;
const TILE_K: u32 = 32u;
const VECTOR_SIZE: u32 = 4u;

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

@group(0) @binding(0) var<storage, read_write> src0: array<vec4<f16>>; // N rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<vec4<f32>>; // M rows, K columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

var<workgroup> A_shared: array<vec4<f16>, (WORKGROUP_SIZE_Y * TILE_Y * TILE_K)/VECTOR_SIZE>;
var<workgroup> B_shared: array<vec4<f32>, (WORKGROUP_SIZE_X * TILE_X * TILE_K)/VECTOR_SIZE>;

fn get_local_x(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_Y;
}
fn get_local_y(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_Y;
}

fn compute_vec4(a: vec4<f16>, b: vec4<f32>) -> f32 {
    let a_f32 = vec4<f32>(f32(a.x), f32(a.y), f32(a.z), f32(a.w));
    return dot(a_f32, b);
}

//override const WORKGROUP_SIZE_X: u32;
//override const WORKGROUP_SIZE_Y: u32;
//override const TILE_X: u32;
//override const TILE_Y: u32;
//override const TILE_K: u32;
//override const VECTOR_SIZE: u32;

//const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;

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

    var acc: array<array<f32, TILE_Y>, TILE_X>;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        let a_tile_size = TILE_K * WORKGROUP_SIZE_Y * TILE_Y;
        let a_loads_per_thread = ((a_tile_size + TOTAL_WORKGROUP_SIZE - 1u) / TOTAL_WORKGROUP_SIZE) / 4;

        for (var load_idx = 0u; load_idx < a_loads_per_thread; load_idx++) {
            let elem_idx = ((thread_id / 4) * 4) + (thread_id % 4) * TOTAL_WORKGROUP_SIZE + load_idx * 4 * TOTAL_WORKGROUP_SIZE;
//            if (elem_idx < a_tile_size) {
                let tile_col = elem_idx / TILE_K;
                let tile_k = elem_idx % TILE_K;
                let global_col = wg_y * WORKGROUP_SIZE_Y * TILE_Y + tile_col;
                let global_k = k_outer + tile_k;

//                if (global_col < params.n && global_k < params.k) {
                    let src0_idx = src0_batch_offset + global_col * params.stride_01 + global_k;
                    A_shared[elem_idx/4] = src0[src0_idx/4];
//                }
//            }
        }

        let b_tile_size = WORKGROUP_SIZE_X * TILE_X * TILE_K;
        let b_loads_per_thread = ((b_tile_size + TOTAL_WORKGROUP_SIZE - 1u) / TOTAL_WORKGROUP_SIZE) / 4;

        for (var load_idx = 0u; load_idx < b_loads_per_thread; load_idx++) {
            let elem_idx = ((thread_id / 4) * 4) + (thread_id % 4) * TOTAL_WORKGROUP_SIZE + load_idx * 4 * TOTAL_WORKGROUP_SIZE;
//            if (elem_idx < b_tile_size) {
                let tile_row = elem_idx / TILE_K;
                let tile_k = elem_idx % TILE_K;
                let global_row = wg_x * WORKGROUP_SIZE_X * TILE_X + tile_row;
                let global_k = k_outer + tile_k;

//                if (global_row < params.m && global_k < params.k) {
                    let src1_idx = src1_batch_offset + global_row * params.stride_11 + global_k;
                    B_shared[elem_idx/4] = src1[src1_idx/4];
//                }
//            }
        }

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

                for (var k_inner = 0u; k_inner < k_end / 4u; k_inner++) {
                    var a_r_tile: array<vec4<f16>, TILE_Y>;
                    for (var ty = 0u; ty < TILE_Y; ty++) {
                      let a_col = local_y * TILE_Y + ty;
//                      if (output_col_base + ty < params.n) {
                        let a_idx = k_inner * 4 + a_col * TILE_K;
                        a_r_tile[ty] = A_shared[a_idx/4];
//                      }
                    }
                    for (var tx = 0u; tx < TILE_X; tx++) {
                        let b_row = local_x * TILE_X + tx;
//                        if (output_row_base + tx < params.m) {
                            let b_idx = b_row * TILE_K + k_inner * 4u;
                            let b_vec = B_shared[b_idx/4];

                            for (var ty = 0u; ty < TILE_Y; ty++) {
//                                if (output_col_base + ty < params.n) {
                                    acc[tx][ty] += compute_vec4(a_r_tile[ty], b_vec);
//                                }
                            }
//                        }
                    }
                }

        workgroupBarrier();
    }

        let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

        for (var tx = 0u; tx < TILE_X; tx++) {
            let global_row = output_row_base + tx;
            if (global_row < params.m) {
                for (var ty = 0u; ty < TILE_Y; ty += 4) {
                    let global_col = output_col_base + ty;
                    if (global_col < params.n) {
                        let dst_idx = dst_batch_offset + global_row * params.n + global_col;
                        dst[dst_idx/4] = vec4<f32>(acc[tx][ty], acc[tx][ty + 1], acc[tx][ty + 2], acc[tx][ty + 3]);
                    }
                }
            }
        }
}