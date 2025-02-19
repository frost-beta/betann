if ($enable_f16) {
  enable f16;
}
if ($enable_subgroups) {
  enable subgroups;
}
if ($enable_subgroups_f16) {
  enable subgroups_f16;
}

alias dtype = $dtype;

// Workload per thread.
const row_work_per_thread: u32 = $row_work_per_thread;
const col_work_per_thread: u32 = $col_work_per_thread;
// Each workgroup works on (workgroup_size_row * row_work_per_thread) rows and all
// columns.
// Each thread works on (mat_cols / workgroup_size_col) columns.
const workgroup_size_row: u32 = $group_count;
const workgroup_size_col: u32 = 32;

@group(0) @binding(0) var<storage, read_write> out: array<dtype>;
@group(0) @binding(1) var<storage, read> mat: array<dtype>;
@group(0) @binding(2) var<uniform> mat_rows: u32;
@group(0) @binding(3) var<uniform> mat_cols: u32;
@group(0) @binding(4) var<uniform> mat_row_stride: u32;
@group(0) @binding(5) var<storage, read> batch_strides_mat: array<u32>;
@group(0) @binding(6) var<storage, read> vec: array<dtype>;
@group(0) @binding(7) var<storage, read> batch_strides_vec: array<u32>;
if (!$contiguous) {
  @group(0) @binding(8) var<storage, read> batch_shape: array<u32>;
}

if ($enable_subgroups) {
  // When enable_subgroups we only need results from each subgroup, however
  // as we don't know the subgroup_size we have to assume minimum size.
  const workgroup_result_cols = workgroup_size_col / $subgroup_min_size;
} else {
  const workgroup_result_cols = workgroup_size_col;
}
// The workgroup_result collects results from each thread.
var<workgroup> workgroup_result: array<dtype, workgroup_size_row *
                                              row_work_per_thread *
                                              workgroup_result_cols>;

@compute @workgroup_size(workgroup_size_col, workgroup_size_row)
fn gemv(if ($enable_subgroups) {
          @builtin(subgroup_size) subgroup_size: u32,
        }
        @builtin(workgroup_id) tid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  const block_size_row = row_work_per_thread * workgroup_size_row;
  const block_size_col = col_work_per_thread * workgroup_size_col;

  // The row worked on.
  let out_row = tid.x * block_size_row + lid.y * row_work_per_thread;

  // Offset of current batch.
  let out_offset = tid.z * mat_rows + out_row;
  if ($contiguous) {
    var mat_offset = tid.z * batch_strides_mat[0];
    let vec_offset = tid.z * batch_strides_vec[0];
  } else {
    var mat_offset = coord_to_index(tid.z, &batch_shape, &batch_strides_mat);
    let vec_offset = coord_to_index(tid.z, &batch_shape, &batch_strides_vec);
  }
  mat_offset += out_row * mat_row_stride;

  // Per-thread result and intermediates.
  var result: array<dtype, row_work_per_thread>;
  var coefficient: array<dtype, col_work_per_thread>;
  var intermediate: array<dtype, col_work_per_thread>;

  // Loop over vector.
  for (var block = 0u; block < mat_cols / block_size_col; block++) {
    let col = lid.x * col_work_per_thread + block * block_size_col;
    // Load vector.
    load_unsafe(&coefficient, &vec, vec_offset + col);

    // Work.
    for (var r = 0u; r < row_work_per_thread; r++) {
      // Load row.
      load_unsafe(&intermediate, &mat, mat_offset + r * mat_row_stride + col);

      // Accumulate results.
      for (var c = 0u; c < col_work_per_thread; c++) {
        if ($dtype_is_floating) {
          result[r] = fma(intermediate[c], coefficient[c], result[r]);
        } else {
          result[r] += intermediate[c] * coefficient[c];
        }
      }
    }
  }

  // Leftover in cols.
  let leftover = mat_cols % block_size_col;
  if (leftover != 0) {
    let col = (mat_cols - leftover) + lid.x * col_work_per_thread;
    load_safe(&coefficient,
              &vec,
              vec_offset + mat_cols,
              vec_offset + col);
    for (var r = 0u; r < row_work_per_thread; r++) {
      load_safe(&intermediate,
                &mat,
                mat_offset + r * mat_row_stride + mat_cols,
                mat_offset + r * mat_row_stride + col);
      for (var c = 0u; c < col_work_per_thread; c++) {
        if ($dtype_is_floating) {
          result[r] = fma(intermediate[c], coefficient[c], result[r]);
        } else {
          result[r] += intermediate[c] * coefficient[c];
        }
      }
    }
  }

  if ($enable_subgroups) {
    // Subgroup accumulations.
    for (var r = 0u; r < row_work_per_thread; r++) {
      if (subgroup_size <= workgroup_size_col) {
        result[r] = subgroupAdd(result[r]);
      } else {
        for (var delta = workgroup_size_col / 2; delta >= 1; delta >>= 1) {
          result[r] += subgroupShuffleDown(result[r], delta);
        }
      }
    }

    // Workgroup accumulations.
    for (var delta = workgroup_size_col / subgroup_size; delta > 1; delta /= subgroup_size) {
      // Write first lane's result to shared memory.
      if (lid.x % subgroup_size == 0) {
        for (var r = 0u; r < row_work_per_thread; r++) {
          let idx = (lid.y * row_work_per_thread + r) * workgroup_result_cols;
          workgroup_result[idx + lid.x / subgroup_size] = result[r];
        }
      }

      // Subgroup accumulations.
      workgroupBarrier();
      for (var r = 0u; r < row_work_per_thread; r++) {
        let idx = (lid.y * row_work_per_thread + r) * workgroup_result_cols;
        result[r] = select(0, workgroup_result[idx + lid.x], lid.x < delta);
        result[r] = subgroupAdd(result[r]);
      }
    }
  } else {
    // Write to shared memory.
    for (var r = 0u; r < row_work_per_thread; r++) {
      let idx = (lid.y * row_work_per_thread + r) * workgroup_size_col + lid.x;
      workgroup_result[idx] = result[r];
    }

    // Workgroup accumulations.
    workgroupBarrier();
    for (var r = 0u; r < row_work_per_thread; r++) {
      let idx = (lid.y * row_work_per_thread + r) * workgroup_size_col + lid.x;
      for (var delta = workgroup_size_col / 2; delta >= 1; delta >>= 1) {
        if (lid.x < delta) {
          workgroup_result[idx] += workgroup_result[idx + delta];
        }
        workgroupBarrier();
      }
    }
  }

  // Write output.
  if (lid.x != 0 || out_row >= mat_rows) {
    return;
  }
  for (var r = 0u; r < row_work_per_thread; r++) {
    if (out_row + r < mat_rows) {
      if ($enable_subgroups) {
        out[out_offset + r] = result[r];
      } else {
        let idx = (lid.y * row_work_per_thread + r) * workgroup_result_cols;
        out[out_offset + r] = workgroup_result[idx];
      }
    }
  }
}

fn load_unsafe(dst: ptr<function, array<dtype, col_work_per_thread>>,
               src: ptr<storage, array<dtype>>,
               offset: u32) {
  for (var i = 0u; i < col_work_per_thread; i++) {
    dst[i] = src[offset + i];
  }
}

fn load_safe(dst: ptr<function, array<dtype, col_work_per_thread>>,
             src: ptr<storage, array<dtype>>,
             src_size: u32,
             offset: u32) {
  for (var i = 0u; i < col_work_per_thread; i++) {
    dst[i] = select(0, src[offset + i], offset + i < src_size);
  }
}

// include utils.wgsl
