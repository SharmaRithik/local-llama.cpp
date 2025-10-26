#define(VARIANTS)
[
  {
    "SHADER_SUFFIX": "f32_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f32>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
    },
    "DECLS": ["SHMEM_VEC"]
  },
  {
    "SHADER_SUFFIX": "f32_f32",
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f32",
      "DST_TYPE" : "f32",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SHMEM_SCALAR"]
  },
  {
    "SHADER_SUFFIX": "f16_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
    },
    "DECLS": ["SHMEM_VEC"]
  },
  {
    "SHADER_SUFFIX": "f16_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE" : "f32",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SHMEM_SCALAR"]
  },
  {
    "SHADER_SUFFIX": "f16_f16_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f16>",
      "DST_TYPE" : "vec4<f32>",
      "VEC_SIZE" : "4",
    },
    "DECLS": ["SHMEM_VEC"]
  },
  {
    "SHADER_SUFFIX": "f16_f16",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f16",
      "DST_TYPE" : "f32",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SHMEM_SCALAR"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(SHMEM_VEC)
fn zero_val_src0() -> {{SRC0_TYPE}} {
    return {{SRC0_TYPE}}(0.0, 0.0, 0.0, 0.0);
}

fn store_src0_shmem(val: {{SRC0_TYPE}}, idx: u32) {
    src0_shmem[idx] = f16(val.x);
    src0_shmem[idx + 1] = f16(val.y);
    src0_shmem[idx + 2] = f16(val.z);
    src0_shmem[idx + 3] = f16(val.w);
}

fn zero_val_src1() -> {{SRC1_TYPE}} {
    return {{SRC1_TYPE}}(0.0, 0.0, 0.0, 0.0);
}

fn store_src1_shmem(val: {{SRC1_TYPE}}, idx: u32) {
    src1_shmem[idx] = f16(val.x);
    src1_shmem[idx + 1] = f16(val.y);
    src1_shmem[idx + 2] = f16(val.z);
    src1_shmem[idx + 3] = f16(val.w);
}

fn store_dst(shmem_idx: u32, dst_idx: u32) {
    dst[dst_idx] = vec4<f32>(
        f32(src0_shmem[shmem_idx]),
        f32(src0_shmem[shmem_idx + 1]),
        f32(src0_shmem[shmem_idx + 2]),
        f32(src0_shmem[shmem_idx + 3])
    );
}
#enddecl(SHMEM_VEC)

#decl(SHMEM_SCALAR)
fn zero_val_src0() -> {{SRC0_TYPE}} {
    return 0.0;
}

fn store_src0_shmem(val: {{SRC0_TYPE}}, idx: u32) {
    src0_shmem[idx] = f16(val);
}

fn zero_val_src1() -> {{SRC1_TYPE}} {
    return 0.0;
}

fn store_src1_shmem(val: {{SRC1_TYPE}}, idx: u32) {
    src1_shmem[idx] = f16(val);
}

fn store_dst(shmem_idx: u32, dst_idx: u32) {
    dst[dst_idx] = f32(src0_shmem[shmem_idx]);
}
#enddecl(SHMEM_SCALAR)

#end(DECLS)

#define(SHADER)
diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable chromium_experimental_subgroup_matrix;

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

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{DST_TYPE}}>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

DECLS

override SUBGROUP_M: u32;
override SUBGROUP_MATRIX_M_SIZE: u32;
override SUBGROUP_N: u32;
override SUBGROUP_MATRIX_N_SIZE: u32;
override SUBGROUP_SIZE: u32;

// Note: must match values in ggml-webgpu.cpp
const SUBGROUP_MATRIX_M = 4u;
const SUBGROUP_MATRIX_N = 2u;

override TILE_K: u32;
// Note: we assume TILE_K is divisible by SUBGROUP_MATRIX_K;
override SUBGROUP_MATRIX_K_SIZE: u32;

override WG_M_SG_TILE_SIZE = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
override WG_N_SG_TILE_SIZE = SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

override TOTAL_WORKGROUP_SIZE = SUBGROUP_M * SUBGROUP_N * SUBGROUP_SIZE;
override TILE_SRC0_SHMEM = TILE_K * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
override TILE_SRC1_SHMEM = TILE_K * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

override SG_MAT_ACCUM_SHMEM = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_M_SIZE * SUBGROUP_MATRIX_N_SIZE;

// We reuse src0_shmem for accumulation matrices
override SHMEM_SIZE = max(TILE_SRC0_SHMEM, SG_MAT_ACCUM_SHMEM);

// Note: apparently current dawn doesn't like override constant shared memory size along with subgroup matrix loads
//var<workgroup> src0_shmem: array<f32, SHMEM_SIZE>;
//var<workgroup> src1_shmem: array<f32, TILE_SRC1_SHMEM>;
var<workgroup> src0_shmem: array<f16, 2048>;
var<workgroup> src1_shmem: array<f16, 1024>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32) {

    let thread_id = local_id.x;
    let subgroup_m = subgroup_id % SUBGROUP_M;
    let subgroup_n = subgroup_id / SUBGROUP_M;

    let wg_linear = global_id.x / TOTAL_WORKGROUP_SIZE;

    let wg_m_count = (params.m + WG_M_SG_TILE_SIZE - 1) / WG_M_SG_TILE_SIZE;
    let wg_n_count = (params.n + WG_N_SG_TILE_SIZE - 1) / WG_N_SG_TILE_SIZE;
    let wg_per_matrix = wg_m_count * wg_n_count;

    let batch_idx = wg_linear / wg_per_matrix;

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

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

    var acc_sg_mat : array<array<subgroup_matrix_result<f16, 8, 8>, SUBGROUP_MATRIX_N>, SUBGROUP_MATRIX_M>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_m = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_m = wg_m * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE + tile_m;
            let global_k = k_outer + tile_k;
            let src0_idx = src0_batch_offset + global_m * params.stride_01 + global_k;
            let src0_val = select( // taking a slight performance hit to avoid oob
                zero_val_src0(),
                src0[src0_idx/{{VEC_SIZE}}],
                global_m < params.m && global_k < params.k);
            store_src0_shmem(src0_val, elem_idx);
        }

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_n = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_n = wg_n * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE + tile_n;
            let global_k = k_outer + tile_k;

            let src1_idx = src1_batch_offset + global_n * params.stride_11 + global_k;
            let src1_val = select(
                zero_val_src1(),
                src1[src1_idx/{{VEC_SIZE}}],
                global_n < params.n && global_k < params.k);
            store_src1_shmem(src1_val, elem_idx);
        }

        workgroupBarrier();

        for (var k_inner = 0u; k_inner < TILE_K; k_inner += SUBGROUP_MATRIX_K_SIZE) {

            let src0_shmem_idx_base = subgroup_m * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE * TILE_K + k_inner;
            var src0_sg_mats: array<subgroup_matrix_left<f16, 8, 8>, SUBGROUP_MATRIX_M>;
            for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                src0_sg_mats[m] = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(
                    &src0_shmem,
                    src0_shmem_idx_base + m * SUBGROUP_MATRIX_M_SIZE * TILE_K,
                    false,
                    TILE_K
                );
            }

            let src1_shmem_idx_base = subgroup_n * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE * TILE_K + k_inner;
            for (var n = 0u; n < SUBGROUP_MATRIX_N; n++) {
                let src1_sg_mat = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(
                    &src1_shmem,
                    src1_shmem_idx_base + n * SUBGROUP_MATRIX_N_SIZE * TILE_K,
                    true,
                    TILE_K
                );
                for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                    acc_sg_mat[m][n] = subgroupMatrixMultiplyAccumulate(src0_sg_mats[m], src1_sg_mat, acc_sg_mat[m][n]);
                }
            }
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;


    // Stage the subgroup matrix tiles into shared memory
    // This uses WG_M_SG_TILE_SIZE as the stride (number of columns in the workgroup tile).
    let WG_TILE_STRIDE = WG_M_SG_TILE_SIZE;
    let tile_row_base_local = subgroup_n * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;
    let tile_col_base_local = subgroup_m * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;

    for (var n = 0u; n < SUBGROUP_MATRIX_N; n++) {
        for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
            let local_row = tile_row_base_local + n * SUBGROUP_MATRIX_N_SIZE;
            let local_col = tile_col_base_local + m * SUBGROUP_MATRIX_M_SIZE;
            let out_base = local_row * WG_TILE_STRIDE + local_col;
            subgroupMatrixStore(&src0_shmem, out_base, acc_sg_mat[m][n], true, WG_TILE_STRIDE);
        }
    }

    workgroupBarrier();

    // Cooperative write: iterate over the entire workgroup tile
    let tile_rows = WG_N_SG_TILE_SIZE;
    let tile_cols = WG_M_SG_TILE_SIZE;
    let total_tile_elems = tile_rows * tile_cols;
    let tile_dst_row_base = wg_n * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;
    let tile_dst_col_base = wg_m * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;

    for (var idx = thread_id * {{VEC_SIZE}}; idx < total_tile_elems; idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
        let local_row = idx / WG_TILE_STRIDE;
        let local_col = idx % WG_TILE_STRIDE;

        let global_row = tile_dst_row_base + local_row;
        let global_col = tile_dst_col_base + local_col;

        if (global_row < params.n && global_col < params.m) {
            let dst_idx = dst_batch_offset + global_row * params.m + global_col;
            store_dst(idx, dst_idx/{{VEC_SIZE}});
        }
    }
}

#end(SHADER)
