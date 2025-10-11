enable f16;
const WORKGROUP_SIZE_X: u32 = 4u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const TOTAL_WORKGROUP_SIZE: u32 = 64u;
const TILE_X: u32 = 1u;
const TILE_Y: u32 = 1u;
const TILE_K: u32 = 16u;
const VECTOR_WIDTH: u32 = 4u;

fn load_src0_scalar(idx: u32) -> f32 {
    return f32(src0[idx]);
}

fn load_src1_scalar(idx: u32) -> f32 {
    return f32(src1[idx]);
}

fn zero_scalar() -> f32 {
    return 0.0;
}

fn compute_scalar(a: f32, b: f32) -> f32 {
    return a * b;
}

fn load_a_vec2_shared(shared_idx: u32) -> vec2<f32> {
    return vec2<f32>(
        A_shared[shared_idx],
        A_shared[shared_idx + 1u]
    );
}

fn load_b_vec2_shared(shared_idx: u32, stride: u32) -> vec2<f32> {
    return vec2<f32>(
        B_shared[shared_idx],
        B_shared[shared_idx + stride]
    );
}

fn zero_vec2() -> vec2<f32> {
    return vec2<f32>(0.0, 0.0);
}

fn compute_vec2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return dot(a, b);
}

fn load_a_vec4_shared(shared_idx: u32) -> vec4<f32> {
    return vec4<f32>(
        A_shared[shared_idx],
        A_shared[shared_idx + 1u],
        A_shared[shared_idx + 2u],
        A_shared[shared_idx + 3u]
    );
}

fn load_b_vec4_shared(shared_idx: u32, stride: u32) -> vec4<f32> {
    return vec4<f32>(
        B_shared[shared_idx],
        B_shared[shared_idx + stride],
        B_shared[shared_idx + 2u * stride],
        B_shared[shared_idx + 3u * stride]
    );
}

fn zero_vec4() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn compute_vec4(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(a, b);
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

@group(0) @binding(0) var<storage, read_write> src0: array<f16>; // N rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<f16>; // M rows, K columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

var<workgroup> A_shared: array<f32, 256>;
var<workgroup> B_shared: array<f32, 64>;

fn get_local_x(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_X;
}

fn get_local_y(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_X;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let thread_id = local_id.x;
    let local_x = get_local_x(thread_id);
    let local_y = get_local_y(thread_id);
    
    let wg_linear = global_id.x / TOTAL_WORKGROUP_SIZE;
    
    let wg_x_count = (params.n + WORKGROUP_SIZE_X * TILE_X - 1u) / (WORKGROUP_SIZE_X * TILE_X);
    let wg_y_count = (params.m + WORKGROUP_SIZE_Y * TILE_Y - 1u) / (WORKGROUP_SIZE_Y * TILE_Y);
    let wg_per_matrix = wg_x_count * wg_y_count;
    
    let batch_idx = wg_linear / wg_per_matrix;
    
    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_x = wg_in_batch % wg_x_count;
    let wg_y = wg_in_batch / wg_x_count;
    
    let output_row_base = wg_y * WORKGROUP_SIZE_Y * TILE_Y + local_y * TILE_Y;
    let output_col_base = wg_x * WORKGROUP_SIZE_X * TILE_X + local_x * TILE_X;
    
    let in_bounds = output_row_base < params.m && output_col_base < params.n;
    
    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;
    
    var acc: array<array<f32, TILE_X>, TILE_Y>;
    if (in_bounds) {
        for (var ty = 0u; ty < TILE_Y; ty++) {
            for (var tx = 0u; tx < TILE_X; tx++) {
                acc[ty][tx] = 0.0;
            }
        }
    }
    
    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;
    
    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {
        
        let a_tile_size = WORKGROUP_SIZE_Y * TILE_Y * TILE_K;
        let a_loads_per_thread = (a_tile_size + TOTAL_WORKGROUP_SIZE - 1u) / TOTAL_WORKGROUP_SIZE;
        
        for (var load_idx = 0u; load_idx < a_loads_per_thread; load_idx++) {
            let elem_idx = thread_id + load_idx * TOTAL_WORKGROUP_SIZE;
            if (elem_idx < a_tile_size) {
                let tile_row = elem_idx / TILE_K;
                let tile_k = elem_idx % TILE_K;
                let global_row = wg_y * WORKGROUP_SIZE_Y * TILE_Y + tile_row;
                let global_k = k_outer + tile_k;
                
                if (global_row < params.m && global_k < params.k) {
                    let src1_idx = src1_batch_offset + global_row * params.stride_11 + global_k;
                    A_shared[elem_idx] = load_src1_scalar(src1_idx);
                } else {
                    A_shared[elem_idx] = zero_scalar();
                }
            }
        }
        
        let b_tile_size = TILE_K * WORKGROUP_SIZE_X * TILE_X;
        let b_loads_per_thread = (b_tile_size + TOTAL_WORKGROUP_SIZE - 1u) / TOTAL_WORKGROUP_SIZE;
        
        for (var load_idx = 0u; load_idx < b_loads_per_thread; load_idx++) {
            let elem_idx = thread_id + load_idx * TOTAL_WORKGROUP_SIZE;
            if (elem_idx < b_tile_size) {
                let tile_k = elem_idx / (WORKGROUP_SIZE_X * TILE_X);
                let tile_col = elem_idx % (WORKGROUP_SIZE_X * TILE_X);
                let global_col = wg_x * WORKGROUP_SIZE_X * TILE_X + tile_col;
                let global_k = k_outer + tile_k;
                
                if (global_col < params.n && global_k < params.k) {
                    let src0_idx = src0_batch_offset + global_col * params.stride_01 + global_k;
                    B_shared[elem_idx] = load_src0_scalar(src0_idx);
                } else {
                    B_shared[elem_idx] = zero_scalar();
                }
            }
        }
        
        workgroupBarrier();
        
        let k_end = min(TILE_K, params.k - k_outer);
        
        if (in_bounds) {
            if (VECTOR_WIDTH == 4u && k_end >= 4u) {
                for (var k_inner = 0u; k_inner < k_end / 4u; k_inner++) {
                    for (var ty = 0u; ty < TILE_Y; ty++) {
                        let a_row = local_y * TILE_Y + ty;
                        if (output_row_base + ty < params.m) {
                            let a_idx = a_row * TILE_K + k_inner * 4u;
                            let a_vec = load_a_vec4_shared(a_idx);
                            
                            for (var tx = 0u; tx < TILE_X; tx++) {
                                let b_col = local_x * TILE_X + tx;
                                if (output_col_base + tx < params.n) {
                                    let b_idx = k_inner * 4u * (WORKGROUP_SIZE_X * TILE_X) + b_col;
                                    let b_vec = load_b_vec4_shared(b_idx, WORKGROUP_SIZE_X * TILE_X);
                                    acc[ty][tx] += compute_vec4(a_vec, b_vec);
                                }
                            }
                        }
                    }
                }
                
                for (var k_inner = (k_end / 4u) * 4u; k_inner < k_end; k_inner++) {
                    for (var ty = 0u; ty < TILE_Y; ty++) {
                        let a_row = local_y * TILE_Y + ty;
                        if (output_row_base + ty < params.m) {
                            let a_idx = a_row * TILE_K + k_inner;
                            let a_val = A_shared[a_idx];
                            
                            for (var tx = 0u; tx < TILE_X; tx++) {
                                let b_col = local_x * TILE_X + tx;
                                if (output_col_base + tx < params.n) {
                                    let b_idx = k_inner * (WORKGROUP_SIZE_X * TILE_X) + b_col;
                                    let b_val = B_shared[b_idx];
                                    acc[ty][tx] += compute_scalar(a_val, b_val);
                                }
                            }
                        }
                    }
                }
                
            } else if (VECTOR_WIDTH == 2u && k_end >= 2u) {
                for (var k_inner = 0u; k_inner < k_end / 2u; k_inner++) {
                    for (var ty = 0u; ty < TILE_Y; ty++) {
                        let a_row = local_y * TILE_Y + ty;
                        if (output_row_base + ty < params.m) {
                            let a_idx = a_row * TILE_K + k_inner * 2u;
                            let a_vec = load_a_vec2_shared(a_idx);
                            
                            for (var tx = 0u; tx < TILE_X; tx++) {
                                let b_col = local_x * TILE_X + tx;
                                if (output_col_base + tx < params.n) {
                                    let b_idx = k_inner * 2u * (WORKGROUP_SIZE_X * TILE_X) + b_col;
                                    let b_vec = load_b_vec2_shared(b_idx, WORKGROUP_SIZE_X * TILE_X);
                                    acc[ty][tx] += compute_vec2(a_vec, b_vec);
                                }
                            }
                        }
                    }
                }
                
                for (var k_inner = (k_end / 2u) * 2u; k_inner < k_end; k_inner++) {
                    for (var ty = 0u; ty < TILE_Y; ty++) {
                        let a_row = local_y * TILE_Y + ty;
                        if (output_row_base + ty < params.m) {
                            let a_idx = a_row * TILE_K + k_inner;
                            let a_val = A_shared[a_idx];
                            
                            for (var tx = 0u; tx < TILE_X; tx++) {
                                let b_col = local_x * TILE_X + tx;
                                if (output_col_base + tx < params.n) {
                                    let b_idx = k_inner * (WORKGROUP_SIZE_X * TILE_X) + b_col;
                                    let b_val = B_shared[b_idx];
                                    acc[ty][tx] += compute_scalar(a_val, b_val);
                                }
                            }
                        }
                    }
                }
                
            } else {
                for (var k_inner = 0u; k_inner < k_end; k_inner++) {
                    for (var ty = 0u; ty < TILE_Y; ty++) {
                        let a_row = local_y * TILE_Y + ty;
                        if (output_row_base + ty < params.m) {
                            let a_idx = a_row * TILE_K + k_inner;
                            let a_val = A_shared[a_idx];
                            
                            for (var tx = 0u; tx < TILE_X; tx++) {
                                let b_col = local_x * TILE_X + tx;
                                if (output_col_base + tx < params.n) {
                                    let b_idx = k_inner * (WORKGROUP_SIZE_X * TILE_X) + b_col;
                                    let b_val = B_shared[b_idx];
                                    acc[ty][tx] += compute_scalar(a_val, b_val);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        workgroupBarrier();
    }
    
    if (in_bounds) {
        let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;
        
        for (var ty = 0u; ty < TILE_Y; ty++) {
            let global_row = output_row_base + ty;
            if (global_row < params.m) {
                for (var tx = 0u; tx < TILE_X; tx++) {
                    let global_col = output_col_base + tx;
                    if (global_col < params.n) {
                        let dst_idx = dst_batch_offset + global_row * params.n + global_col;
                        dst[dst_idx] = acc[ty][tx];
                    }
                }
            }
        }
    }
}


