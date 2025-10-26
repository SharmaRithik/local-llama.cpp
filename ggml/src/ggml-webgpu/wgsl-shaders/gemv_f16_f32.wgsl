enable f16;

// Optimized GEMV shader for F16xF32 matrix-vector multiplication
// Handles both M=1 (row vector * matrix) and N=1 (matrix * column vector) cases
// Uses vectorized memory access and shared memory for better performance

const WORKGROUP_SIZE: u32 = 256u;  // Larger workgroup for better occupancy
const VECTOR_WIDTH: u32 = 4u;       // Process 4 elements at a time with vec4
const TILE_K: u32 = 128u;           // Tile size along K dimension for cache efficiency
const OUTPUTS_PER_WG: u32 = 16u;    // Each workgroup computes 16 outputs (written as 4x vec4) - OPTIMAL

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

@group(0) @binding(0) var<storage, read_write> src0: array<vec4<f16>>; // Matrix (N x K in vec4s)
@group(0) @binding(1) var<storage, read_write> src1: array<vec4<f32>>; // Vector (M x K or K in vec4s)
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;  // Result vector (vec4 for bandwidth)

@group(0) @binding(3) var<uniform> params: MulMatParams;

// Shared memory for collaborative loading and reduction
var<workgroup> shared_vector: array<vec4<f32>, TILE_K/4>;  // Cache vector tile
var<workgroup> partial_sums: array<f32, WORKGROUP_SIZE>;   // For reduction (4 groups)

// Helper function for vectorized dot product
fn dot_vec4_f16_f32(a: vec4<f16>, b: vec4<f32>) -> f32 {
    return f32(a.x) * b.x + f32(a.y) * b.y + f32(a.z) * b.z + f32(a.w) * b.w;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let thread_id = local_id.x;

    // Handle batch dimensions
    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let output_elements = select(params.n, params.m, params.n == 1u);

    // Each workgroup computes OUTPUTS_PER_WG consecutive outputs (written as vec4)
    // Using 2D dispatch to avoid exceeding 65535 limit per dimension
    // wg_linear = wg_id.y * 65535 + wg_id.x
    let wg_linear = wg_id.y * 65535u + wg_id.x;
    let output_vec4_groups = (output_elements + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_vec4_groups;
    let output_vec4_idx = wg_linear % output_vec4_groups;
    let base_output_idx = output_vec4_idx * OUTPUTS_PER_WG;

    // Which of the 16 outputs does this thread belong to?
    let threads_per_output = WORKGROUP_SIZE / OUTPUTS_PER_WG;  // 256/16 = 16
    let output_offset = thread_id / threads_per_output;  // 0-15
    let thread_in_group = thread_id % threads_per_output;  // 0-15

    if (batch_idx >= total_batches) {
        return;
    }

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;

    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    // Case 1: M == 1 (result is a row vector: 1 x N)
    // Each workgroup computes OUTPUTS_PER_WG (16) consecutive output elements
    // 256 threads split into 16 groups of 16, each group computes one output
    if (params.n == 1u) {
        let output_col_base = base_output_idx;
        let output_col = output_col_base + output_offset;

        // Check bounds but don't early return (must hit all barriers)
        let is_valid = output_col < params.m;

        var local_sum = 0.0;
        let k_vec = params.k / VECTOR_WIDTH;

        // Each thread processes multiple K elements and accumulates
        for (var k_tile = 0u; k_tile < k_vec; k_tile += TILE_K/VECTOR_WIDTH) {
            let tile_size = min(TILE_K/VECTOR_WIDTH, k_vec - k_tile);

            // Cooperatively load vector tile into shared memory (all threads)
            for (var i = thread_id; i < tile_size; i += WORKGROUP_SIZE) {
                let k_idx = (k_tile + i) * VECTOR_WIDTH;
                if (k_idx < params.k) {
                    let src1_idx = params.offset_src1 + src13_idx * params.stride_13 +
                                   src12_idx * params.stride_12 + k_idx;
                    shared_vector[i] = src1[src1_idx / VECTOR_WIDTH];
                }
            }

            workgroupBarrier();

            // Each sub-group of 16 threads computes its own output
            // thread_in_group = 0-15, provides stride for this sub-group
            let threads_per_output = WORKGROUP_SIZE / OUTPUTS_PER_WG;  // 16

            if (is_valid) {
                for (var i = thread_in_group; i < tile_size; i += threads_per_output) {
                    let k_idx = (k_tile + i) * VECTOR_WIDTH;
                    // Removed redundant k_idx < params.k check (guaranteed by tile_size)
                    let src0_idx = params.offset_src0 + src03_idx * params.stride_03 +
                                   src02_idx * params.stride_02 + output_col * params.stride_01 + k_idx;
                    let a = src0[src0_idx / VECTOR_WIDTH];
                    let b = shared_vector[i];
                    local_sum += dot_vec4_f16_f32(a, b);
                }
            }

            workgroupBarrier();
        }

        // Handle remaining elements (K % VECTOR_WIDTH)
        if (is_valid) {
            let k_remainder_start = k_vec * VECTOR_WIDTH;
            if (thread_in_group < (params.k - k_remainder_start)) {
                let k_idx = k_remainder_start + thread_in_group;
                let src0_idx = params.offset_src0 + src03_idx * params.stride_03 +
                               src02_idx * params.stride_02 + output_col * params.stride_01 + k_idx;
                let src1_idx = params.offset_src1 + src13_idx * params.stride_13 +
                               src12_idx * params.stride_12 + k_idx;
                // Read individual elements (last vec4 might be partial)
                let vec_idx = k_idx / VECTOR_WIDTH;
                let elem_idx = k_idx % VECTOR_WIDTH;
                let a_vec = src0[src0_idx / VECTOR_WIDTH];
                let b_vec = src1[src1_idx / VECTOR_WIDTH];
                local_sum += f32(a_vec[elem_idx]) * b_vec[elem_idx];
            }
        }

        // Store partial sums and reduce within each sub-group (16 threads per output)
        partial_sums[thread_id] = local_sum;
        workgroupBarrier();

        // Reduce within each sub-group: 16 threads → 1 result
        // Each sub-group occupies 16 consecutive slots in partial_sums
        let group_base = output_offset * (WORKGROUP_SIZE / OUTPUTS_PER_WG);  // 0, 16, 32, ..., 240

        // Reduction for 16 threads: 16 → 8 → 4 → 2 → 1 (loop version for correctness)
        for (var stride = 8u; stride > 0u; stride = stride / 2u) {
            if (thread_in_group < stride) {
                partial_sums[group_base + thread_in_group] += partial_sums[group_base + thread_in_group + stride];
            }
            workgroupBarrier();
        }

        // First thread of each sub-group has the result
        // Threads 0, 16, 32, 48, ... 240 hold the 16 output values
        if (thread_id == 0u && output_col_base < params.m) {
            // Gather 16 results and write as 4 vec4s
            let result_vec0 = vec4<f32>(partial_sums[0], partial_sums[16], partial_sums[32], partial_sums[48]);
            let result_vec1 = vec4<f32>(partial_sums[64], partial_sums[80], partial_sums[96], partial_sums[112]);
            let result_vec2 = vec4<f32>(partial_sums[128], partial_sums[144], partial_sums[160], partial_sums[176]);
            let result_vec3 = vec4<f32>(partial_sums[192], partial_sums[208], partial_sums[224], partial_sums[240]);

            let dst_idx = params.offset_dst + dst3_idx * dst3_stride +
                          dst2_idx * dst2_stride + output_col_base;
            dst[dst_idx / VECTOR_WIDTH] = result_vec0;
            dst[dst_idx / VECTOR_WIDTH + 1u] = result_vec1;
            dst[dst_idx / VECTOR_WIDTH + 2u] = result_vec2;
            dst[dst_idx / VECTOR_WIDTH + 3u] = result_vec3;
        }
    }
    // Case 2: N == 1 (result is a column vector: M x 1)
    // Each workgroup computes OUTPUTS_PER_WG (16) consecutive output elements
    // 256 threads split into 16 groups of 16, each group computes one output
    else if (params.m == 1u) {
        let output_row_base = base_output_idx;
        let output_row = output_row_base + output_offset;

        // Check bounds but don't early return (must hit all barriers)
        let is_valid = output_row < params.n;

        var local_sum = 0.0;
        let k_vec = params.k / VECTOR_WIDTH;

        // Each thread processes multiple K elements and accumulates
        for (var k_tile = 0u; k_tile < k_vec; k_tile += TILE_K/VECTOR_WIDTH) {
            let tile_size = min(TILE_K/VECTOR_WIDTH, k_vec - k_tile);

            // Cooperatively load vector tile into shared memory (all threads)
            // Note: In this case, src0 is the vector input
            for (var i = thread_id; i < tile_size; i += WORKGROUP_SIZE) {
                let k_idx = (k_tile + i) * VECTOR_WIDTH;
                if (k_idx < params.k) {
                    let src0_idx = params.offset_src0 + src03_idx * params.stride_03 +
                                   src02_idx * params.stride_02 + k_idx;
                    shared_vector[i] = vec4<f32>(src0[src0_idx / VECTOR_WIDTH]);
                }
            }

            workgroupBarrier();

            // Each sub-group of 16 threads computes its own output
            // thread_in_group = 0-15, provides stride for this sub-group
            let threads_per_output = WORKGROUP_SIZE / OUTPUTS_PER_WG;  // 16

            if (is_valid) {
                for (var i = thread_in_group; i < tile_size; i += threads_per_output) {
                    let k_idx = (k_tile + i) * VECTOR_WIDTH;
                    // Removed redundant k_idx < params.k check (guaranteed by tile_size)
                    let src1_idx = params.offset_src1 + src13_idx * params.stride_13 +
                                   src12_idx * params.stride_12 + output_row * params.stride_11 + k_idx;
                    let a = shared_vector[i];  // from src0
                    let b = src1[src1_idx / VECTOR_WIDTH];
                    local_sum += dot(a, b);
                }
            }

            workgroupBarrier();
        }

        // Handle remaining elements (K % VECTOR_WIDTH)
        if (is_valid) {
            let k_remainder_start = k_vec * VECTOR_WIDTH;
            if (thread_in_group < (params.k - k_remainder_start)) {
                let k_idx = k_remainder_start + thread_in_group;
                let src0_idx = params.offset_src0 + src03_idx * params.stride_03 +
                               src02_idx * params.stride_02 + k_idx;
                let src1_idx = params.offset_src1 + src13_idx * params.stride_13 +
                               src12_idx * params.stride_12 + output_row * params.stride_11 + k_idx;
                let vec_idx = k_idx / VECTOR_WIDTH;
                let elem_idx = k_idx % VECTOR_WIDTH;
                let a_vec = src0[src0_idx / VECTOR_WIDTH];
                let b_vec = src1[src1_idx / VECTOR_WIDTH];
                local_sum += f32(a_vec[elem_idx]) * b_vec[elem_idx];
            }
        }

        // Store partial sums and reduce within each sub-group (16 threads per output)
        partial_sums[thread_id] = local_sum;
        workgroupBarrier();

        // Reduce within each sub-group: 16 threads → 1 result
        // Each sub-group occupies 16 consecutive slots in partial_sums
        let group_base = output_offset * (WORKGROUP_SIZE / OUTPUTS_PER_WG);  // 0, 16, 32, ..., 240

        // Reduction for 16 threads: 16 → 8 → 4 → 2 → 1 (loop version for correctness)
        for (var stride = 8u; stride > 0u; stride = stride / 2u) {
            if (thread_in_group < stride) {
                partial_sums[group_base + thread_in_group] += partial_sums[group_base + thread_in_group + stride];
            }
            workgroupBarrier();
        }

        // First thread of each sub-group has the result
        // Threads 0, 16, 32, 48, ... 240 hold the 16 output values
        if (thread_id == 0u && output_row_base < params.n) {
            // Gather 16 results and write as 4 vec4s
            let result_vec0 = vec4<f32>(partial_sums[0], partial_sums[16], partial_sums[32], partial_sums[48]);
            let result_vec1 = vec4<f32>(partial_sums[64], partial_sums[80], partial_sums[96], partial_sums[112]);
            let result_vec2 = vec4<f32>(partial_sums[128], partial_sums[144], partial_sums[160], partial_sums[176]);
            let result_vec3 = vec4<f32>(partial_sums[192], partial_sums[208], partial_sums[224], partial_sums[240]);

            let dst_idx = params.offset_dst + dst3_idx * dst3_stride +
                          dst2_idx * dst2_stride + output_row_base * params.m;
            dst[dst_idx / VECTOR_WIDTH] = result_vec0;
            dst[dst_idx / VECTOR_WIDTH + 1u] = result_vec1;
            dst[dst_idx / VECTOR_WIDTH + 2u] = result_vec2;
            dst[dst_idx / VECTOR_WIDTH + 3u] = result_vec3;
        }
    }
}