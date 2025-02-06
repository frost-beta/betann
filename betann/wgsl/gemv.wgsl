if ($enable_f16) {
  enable f16;
}

alias dtype = $dtype;

const work_per_row: u32 = $work_per_row;
const work_per_col: u32 = 4;
const workgroup_size_row: u32 = $workgroup_size_row;
const workgroup_size_col: u32 = 32;

@group(0) @binding(0) var<storage, read_write> out: array<dtype>;
@group(0) @binding(1) var<storage, read> mat: array<dtype>;
@group(0) @binding(2) var<uniform> mat_rows: u32;
@group(0) @binding(3) var<uniform> mat_cols: u32;
@group(0) @binding(4) var<storage, read> vec: array<dtype>;

var<workgroup> workgroup_result: array<dtype, workgroup_size_row *
                                              work_per_row *
                                              workgroup_size_col>;

@compute @workgroup_size(workgroup_size_row, workgroup_size_col)
fn gemv(@builtin(workgroup_id) tid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  const block_size_row = work_per_row * workgroup_size_row;
  const block_size_col = work_per_col * workgroup_size_col;

  // The row worked on.
  var out_row = tid.x * block_size_row + lid.x * work_per_row;

  // Offset of current batch.
  let out_offset = tid.z * mat_rows + out_row;
  let mat_offset = tid.z * mat_rows * mat_cols + out_row * mat_cols;
  let vec_offset = tid.z * mat_cols;

  // Per-thread result and intermediates.
  var result: array<dtype, work_per_row>;
  var intermediate: array<dtype, work_per_col>;
  var coefficients: array<dtype, work_per_col>;

  // Loop over vector.
  for (var block = 0u; block < mat_cols / block_size_col; block++) {
    let col = lid.y * work_per_col + block * block_size_col;
    // Load vector.
    load_unsafe(&coefficients, &vec, vec_offset + col);

    // Work.
    for (var r = 0u; r < work_per_row; r++) {
      // Load row.
      load_unsafe(&intermediate, &mat, mat_offset + r * mat_cols + col);

      // Accumulate results.
      for (var c = 0u; c < work_per_col; c++) {
        if ($dtype_is_floating) {
          result[r] = fma(intermediate[c], coefficients[c], result[r]);
        } else {
          result[r] += intermediate[c] * coefficients[c];
        }
      }
    }
  }

  // Leftover in cols.
  let leftover = mat_cols % block_size_col;
  if (leftover != 0) {
    let col = (mat_cols - leftover) + lid.y * work_per_col;
    load_safe(&coefficients, &vec, vec_offset + mat_cols, vec_offset + col);
    for (var r = 0u; r < work_per_row; r++) {
      load_safe(&intermediate,
                &mat,
                mat_offset + mat_rows * mat_cols,
                mat_offset + r * mat_cols + col);
      for (var c = 0u; c < work_per_col; c++) {
        if ($dtype_is_floating) {
          result[r] = fma(intermediate[c], coefficients[c], result[r]);
        } else {
          result[r] += intermediate[c] * coefficients[c];
        }
      }
    }
  }

  // Write to shared memory.
  for (var r = 0u; r < work_per_row; r++) {
    let idx = (lid.x * work_per_row + r) * workgroup_size_col + lid.y;
    workgroup_result[idx] = result[r];
  }

  // Workgroup accumulations.
  workgroupBarrier();
  for (var r = 0u; r < work_per_row; r++) {
    let idx = (lid.x * work_per_row + r) * workgroup_size_col + lid.y;
    for (var delta = workgroup_size_col / 2; delta >= 1; delta >>= 1) {
      if (lid.y < delta) {
        workgroup_result[idx] += workgroup_result[idx + delta];
      }
      workgroupBarrier();
    }
  }

  // Write output.
  for (var r = 0u; r < work_per_row; r++) {
    if (out_row + r < mat_rows) {
      let idx = (lid.x * work_per_row + r) * workgroup_size_col;
      out[out_offset + r] = workgroup_result[idx];
    }
  }
}

fn load_unsafe(dst: ptr<function, array<dtype, work_per_col>>,
               src: ptr<storage, array<dtype>>,
               offset: u32) {
  for (var i = 0u; i < work_per_col; i++) {
    dst[i] = src[offset + i];
  }
}

fn load_safe(dst: ptr<function, array<dtype, work_per_col>>,
             src: ptr<storage, array<dtype>>,
             src_size: u32,
             offset: u32) {
  for (var i = 0u; i < work_per_col; i++) {
    dst[i] = select(0, src[offset + i], offset + i < src_size);
  }
}
