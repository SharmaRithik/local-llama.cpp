#define(VARIANTS)
[
  {
    "SHADER_SUFFIX": "f32_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f32>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE" : "vec4<f32>",
      "SHMEM_TYPE" : "vec4<f16>",
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
      "SHMEM_TYPE" : "f16",
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
      "SHMEM_TYPE" : "vec4<f16>",
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
      "SHMEM_TYPE" : "f16",
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
      "SHMEM_TYPE" : "vec4<f16>",
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
      "SHMEM_TYPE" : "f16",
      "VEC_SIZE" : "1",
    },
    "DECLS": ["SHMEM_SCALAR"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(SHMEM_VEC)
fn store_shmem(val: vec4<f16>, idx: u32) {
    shmem[idx] = val.x;
    shmem[idx + 1] = val.y;
    shmem[idx + 2] = val.z;
    shmem[idx + 3] = val.w;
}

fn store_dst(shmem_idx: u32, dst_idx: u32) {
    dst[dst_idx] = vec4<f32>(
        f32(shmem[shmem_idx]),
        f32(shmem[shmem_idx + 1]),
        f32(shmem[shmem_idx + 2]),
        f32(shmem[shmem_idx + 3])
    );
}
#enddecl(SHMEM_VEC)

#decl(SHMEM_SCALAR)
fn store_shmem(val: f16, idx: u32) {
    shmem[idx] = val;
}

fn store_dst(shmem_idx: u32, dst_idx: u32) {
    dst[dst_idx] = f32(shmem[shmem_idx]);
}
#enddecl(SHMEM_SCALAR)

#end(DECLS)

#define(SHADER)
diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable subgroups;
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

// Note: These are string interpolated at build time, cannot use override constants due to limitations in
// current Dawn version type definitions/matrix load requirements for constant memory sizes.
const SUBGROUP_M = {{WEBGPU_SUBGROUP_M}}u;
const SUBGROUP_N = {{WEBGPU_SUBGROUP_N}}u;
// For portability we assume the max subgroup size, meaning some subgroups will be masked out if the
// runtime subgroup size is smaller.
const MAX_SUBGROUP_SIZE = {{WEBGPU_MAX_SUBGROUP_SIZE}}u;

const EXPECTED_SUBGROUPS = SUBGROUP_M * SUBGROUP_N;

const SUBGROUP_MATRIX_M_SIZE = {{WEBGPU_SG_MAT_M_SIZE}}u;
const SUBGROUP_MATRIX_N_SIZE = {{WEBGPU_SG_MAT_N_SIZE}}u;
const SUBGROUP_MATRIX_K_SIZE = {{WEBGPU_SG_MAT_K_SIZE}}u;

const SUBGROUP_MATRIX_M = {{WEBGPU_SUBGROUP_MATRIX_M}}u;
const SUBGROUP_MATRIX_N = {{WEBGPU_SUBGROUP_MATRIX_N}}u;

const TILE_K = {{WEBGPU_TILE_K}}u;

const WG_M_SG_TILE_SIZE = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
const WG_N_SG_TILE_SIZE = SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

const TOTAL_WORKGROUP_SIZE = SUBGROUP_M * SUBGROUP_N * MAX_SUBGROUP_SIZE;
const TILE_SRC0_SHMEM = TILE_K * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
const TILE_SRC1_SHMEM = TILE_K * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

const SG_MAT_ACCUM_SHMEM = SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_M_SIZE * SUBGROUP_MATRIX_N_SIZE;

// We reuse shmem for accumulation matrices
const SHMEM_SIZE = max(TILE_SRC0_SHMEM + TILE_SRC1_SHMEM, SG_MAT_ACCUM_SHMEM);

var<workgroup> shmem: array<f16, SHMEM_SIZE>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32) {

    let thread_id = local_id.x;
    let subgroup_m = subgroup_id % SUBGROUP_M;
    let subgroup_n = subgroup_id / SUBGROUP_M;

    let wg_m_count = (params.m + WG_M_SG_TILE_SIZE - 1) / WG_M_SG_TILE_SIZE;
    let wg_n_count = (params.n + WG_N_SG_TILE_SIZE - 1) / WG_N_SG_TILE_SIZE;
    let wg_per_matrix = wg_m_count * wg_n_count;

    let batch_idx = wg_id.x / wg_per_matrix;

    let wg_in_batch = wg_id.x % wg_per_matrix;
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

    var acc_sg_mat : array<array<subgroup_matrix_result<f16, SUBGROUP_MATRIX_N_SIZE, SUBGROUP_MATRIX_M_SIZE>, SUBGROUP_MATRIX_N>, SUBGROUP_MATRIX_M>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC0_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_m = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_m = wg_m * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE + tile_m;
            let global_k = k_outer + tile_k;
            let src0_idx = src0_batch_offset + global_m * params.stride_01 + global_k;
            let src0_val = select( // taking a slight performance hit to avoid oob
                {{SRC0_TYPE}}(0.0),
                src0[src0_idx/{{VEC_SIZE}}],
                global_m < params.m && global_k < params.k);
            store_shmem({{SHMEM_TYPE}}(src0_val), elem_idx);
        }

        for (var elem_idx = thread_id * {{VEC_SIZE}}; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
            let tile_n = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_n = wg_n * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE + tile_n;
            let global_k = k_outer + tile_k;

            let src1_idx = src1_batch_offset + global_n * params.stride_11 + global_k;
            let src1_val = select(
                {{SRC1_TYPE}}(0.0),
                src1[src1_idx/{{VEC_SIZE}}],
                global_n < params.n && global_k < params.k);
            store_shmem({{SHMEM_TYPE}}(src1_val), TILE_SRC0_SHMEM + elem_idx);
        }

        workgroupBarrier();

        if (subgroup_id < EXPECTED_SUBGROUPS) {

            for (var k_inner = 0u; k_inner < TILE_K; k_inner += SUBGROUP_MATRIX_K_SIZE) {

                let src0_shmem_idx_base = subgroup_m * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE * TILE_K + k_inner;
                var src0_sg_mats: array<subgroup_matrix_left<f16, SUBGROUP_MATRIX_K_SIZE, SUBGROUP_MATRIX_M_SIZE>, SUBGROUP_MATRIX_M>;
                for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                    src0_sg_mats[m] = subgroupMatrixLoad<subgroup_matrix_left<f16, SUBGROUP_MATRIX_K_SIZE, SUBGROUP_MATRIX_M_SIZE>>(
                        &shmem,
                        src0_shmem_idx_base + m * SUBGROUP_MATRIX_M_SIZE * TILE_K,
                        false,
                        TILE_K
                    );
                }

                let src1_shmem_idx_base = subgroup_n * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE * TILE_K + k_inner;
                for (var n = 0u; n < SUBGROUP_MATRIX_N; n++) {
                    let src1_sg_mat = subgroupMatrixLoad<subgroup_matrix_right<f16, SUBGROUP_MATRIX_N_SIZE, SUBGROUP_MATRIX_K_SIZE>>(
                        &shmem,
                        TILE_SRC0_SHMEM + src1_shmem_idx_base + n * SUBGROUP_MATRIX_N_SIZE * TILE_K,
                        true,
                        TILE_K
                    );
                    for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                        acc_sg_mat[m][n] = subgroupMatrixMultiplyAccumulate(src0_sg_mats[m], src1_sg_mat, acc_sg_mat[m][n]);
                    }
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

    if (subgroup_id < EXPECTED_SUBGROUPS) { // 2-5% performance hit :(
        for (var n = 0u; n < SUBGROUP_MATRIX_N; n++) {
            for (var m = 0u; m < SUBGROUP_MATRIX_M; m++) {
                let local_row = tile_row_base_local + n * SUBGROUP_MATRIX_N_SIZE;
                let local_col = tile_col_base_local + m * SUBGROUP_MATRIX_M_SIZE;
                let out_base = local_row * WG_TILE_STRIDE + local_col;
                subgroupMatrixStore(&shmem, out_base, acc_sg_mat[m][n], true, WG_TILE_STRIDE);
            }
        }
    }

    workgroupBarrier();

    // Cooperative write: iterate over the entire workgroup tile
    let tile_rows = WG_N_SG_TILE_SIZE;
    let tile_cols = WG_M_SG_TILE_SIZE;
    let total_tile_elems = tile_rows * tile_cols;
    let tile_dst_row_base = wg_m * SUBGROUP_M * SUBGROUP_MATRIX_M * SUBGROUP_MATRIX_M_SIZE;
    let tile_dst_col_base = wg_n * SUBGROUP_N * SUBGROUP_MATRIX_N * SUBGROUP_MATRIX_N_SIZE;

    for (var idx = thread_id * {{VEC_SIZE}}; idx < total_tile_elems; idx += TOTAL_WORKGROUP_SIZE * {{VEC_SIZE}}) {
        let local_row = idx % WG_TILE_STRIDE;
        let local_col = idx / WG_TILE_STRIDE;

        let global_row = tile_dst_row_base + local_row;
        let global_col = tile_dst_col_base + local_col;

        if (global_col < params.n && global_row < params.m) {
            let dst_idx = dst_batch_offset + global_col * params.m + global_row;
            store_dst(idx, dst_idx/{{VEC_SIZE}});
        }
    }
}

#end(SHADER)
