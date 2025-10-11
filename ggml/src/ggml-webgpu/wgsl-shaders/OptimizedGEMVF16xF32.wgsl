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

@group(0) @binding(0) var<storage, read_write> src0: array<f16>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const TOTAL_WORKGROUP_SIZE: u32 = 256u;
const TILE_K: u32 = 16u;

const WPT_X: u32 = 4u;
const WPT_Y: u32 = 4u;

const VECTOR_WIDTH: u32 = 4u;

const TILE_SIZE_X: u32 = WORKGROUP_SIZE_X * WPT_X;
const TILE_SIZE_Y: u32 = WORKGROUP_SIZE_Y * WPT_Y;

var<workgroup> tile_src0: array<f32, TILE_K * TILE_SIZE_X>;
var<workgroup> tile_src1: array<f32, TILE_K * TILE_SIZE_Y>;

fn get_local_x(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_X;
}

fn get_local_y(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_X;
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let local_x = get_local_x(thread_id);
    let local_y = get_local_y(thread_id);
    
    let wg_linear = global_id.x / TOTAL_WORKGROUP_SIZE;
    
    let wg_x_count = (params.n + TILE_SIZE_X - 1u) / TILE_SIZE_X;
    let wg_y_count = (params.m + TILE_SIZE_Y - 1u) / TILE_SIZE_Y;
    let wg_per_matrix = wg_x_count * wg_y_count;
    
    let batch_idx = wg_linear / wg_per_matrix;
    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_x = wg_in_batch % wg_x_count;
    let wg_y = wg_in_batch / wg_x_count;
    
    let output_row_base = wg_y * TILE_SIZE_Y;
    let output_col_base = wg_x * TILE_SIZE_X;
    
    let thread_row_base = local_y * WPT_Y;
    let thread_col_base = local_x * WPT_X;
    
    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;
    
    var sums: array<f32, WPT_X * WPT_Y>;
    for (var i: u32 = 0u; i < WPT_X * WPT_Y; i++) {
        sums[i] = 0.0;
    }
    
    let num_tiles = (params.k + TILE_K - 1u) / TILE_K;
    
    for (var tile: u32 = 0u; tile < num_tiles; tile = tile + 1u) {
        let k_start = tile * TILE_K;
        
        for (var i = thread_id; i < TILE_K * TILE_SIZE_X; i += TOTAL_WORKGROUP_SIZE) {
            let k_offset = i / TILE_SIZE_X;
            let col_offset = i % TILE_SIZE_X;
            
            let k_idx = k_start + k_offset;
            let global_col_idx = output_col_base + col_offset;
            
            if (k_idx < params.k && global_col_idx < params.n) {
                let src0_idx = params.offset_src0 + src03_idx * params.stride_03 + 
                               src02_idx * params.stride_02 + global_col_idx * params.stride_01 + k_idx;
                tile_src0[i] = f32(src0[src0_idx]);
            } else {
                tile_src0[i] = 0.0;
            }
        }
        
        for (var i = thread_id; i < TILE_K * TILE_SIZE_Y; i += TOTAL_WORKGROUP_SIZE) {
            let k_offset = i / TILE_SIZE_Y;
            let row_offset = i % TILE_SIZE_Y;
            
            let k_idx = k_start + k_offset;
            let global_row_idx = output_row_base + row_offset;
            
            if (k_idx < params.k && global_row_idx < params.m) {
                let src1_idx = params.offset_src1 + src13_idx * params.stride_13 + 
                               src12_idx * params.stride_12 + global_row_idx * params.stride_11 + k_idx;
                tile_src1[i] = src1[src1_idx];
            } else {
                tile_src1[i] = 0.0;
            }
        }
        
        workgroupBarrier();
        
        let k_end = min(k_start + TILE_K, params.k);
        let k_vec_end = (k_end - k_start) / VECTOR_WIDTH * VECTOR_WIDTH;
        
        for (var k: u32 = 0u; k < k_vec_end; k += VECTOR_WIDTH) {
            for (var wy: u32 = 0u; wy < WPT_Y; wy++) {
                if (VECTOR_WIDTH == 4u) {
                    let base_idx = k * TILE_SIZE_Y + thread_row_base + wy;
                    let src1_vec = vec4<f32>(
                        tile_src1[base_idx],
                        tile_src1[base_idx + TILE_SIZE_Y],
                        tile_src1[base_idx + TILE_SIZE_Y * 2u],
                        tile_src1[base_idx + TILE_SIZE_Y * 3u]
                    );
                    
                    for (var wx: u32 = 0u; wx < WPT_X; wx++) {
                        let base_idx_0 = k * TILE_SIZE_X + thread_col_base + wx;
                        let src0_vec = vec4<f32>(
                            tile_src0[base_idx_0],
                            tile_src0[base_idx_0 + TILE_SIZE_X],
                            tile_src0[base_idx_0 + TILE_SIZE_X * 2u],
                            tile_src0[base_idx_0 + TILE_SIZE_X * 3u]
                        );
                        sums[wy * WPT_X + wx] += dot(src0_vec, src1_vec);
                    }
                } else if (VECTOR_WIDTH == 2u) {
                    let base_idx = k * TILE_SIZE_Y + thread_row_base + wy;
                    let src1_vec = vec2<f32>(
                        tile_src1[base_idx],
                        tile_src1[base_idx + TILE_SIZE_Y]
                    );
                    
                    for (var wx: u32 = 0u; wx < WPT_X; wx++) {
                        let base_idx_0 = k * TILE_SIZE_X + thread_col_base + wx;
                        let src0_vec = vec2<f32>(
                            tile_src0[base_idx_0],
                            tile_src0[base_idx_0 + TILE_SIZE_X]
                        );
                        sums[wy * WPT_X + wx] += dot(src0_vec, src1_vec);
                    }
                } else {
                    let src1_val = tile_src1[k * TILE_SIZE_Y + thread_row_base + wy];
                    for (var wx: u32 = 0u; wx < WPT_X; wx++) {
                        let src0_val = tile_src0[k * TILE_SIZE_X + thread_col_base + wx];
                        sums[wy * WPT_X + wx] += src0_val * src1_val;
                    }
                }
            }
        }
        
        for (var k: u32 = k_vec_end; k < k_end - k_start; k++) {
            for (var wy: u32 = 0u; wy < WPT_Y; wy++) {
                let src1_val = tile_src1[k * TILE_SIZE_Y + thread_row_base + wy];
                for (var wx: u32 = 0u; wx < WPT_X; wx++) {
                    let src0_val = tile_src0[k * TILE_SIZE_X + thread_col_base + wx];
                    sums[wy * WPT_X + wx] += src0_val * src1_val;
                }
            }
        }
        
        workgroupBarrier();
    }
    
    for (var wy: u32 = 0u; wy < WPT_Y; wy++) {
        for (var wx: u32 = 0u; wx < WPT_X; wx++) {
            let global_row = output_row_base + thread_row_base + wy;
            let global_col = output_col_base + thread_col_base + wx;
            
            if (global_row < params.m && global_col < params.n) {
                let dst_idx = params.offset_dst + dst3_idx * dst3_stride + 
                              dst2_idx * params.m * params.n + global_row * params.n + global_col;
                dst[dst_idx] = sums[wy * WPT_X + wx];
            }
        }
    }
}