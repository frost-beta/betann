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
// A group consists one (on mac) or more subgroups, and a workgroup consists of
// group_count of groups.
// Each workgroup works on (group_cols * group_count) cols, and all rows.
// Each thread works on (mat_cols / group_rows) rows.
const group_count: u32 = $group_count;
const group_rows: u32 = $group_rows;
const group_cols: u32 = $group_cols;

const group_size: u32 = group_rows * group_cols;
const rows_per_workgroup = row_work_per_thread * group_rows;
const cols_per_workgroup = col_work_per_thread * group_cols * group_count;

@group(0) @binding(0) var<storage, read_write> out: array<dtype>;
@group(0) @binding(1) var<storage, read> mat: array<dtype>;
@group(0) @binding(2) var<uniform> mat_rows: u32;
@group(0) @binding(3) var<uniform> mat_cols: u32;
@group(0) @binding(4) var<storage, read> vec: array<dtype>;
if (!$contiguous) {
  @group(0) @binding(5) var<storage, read> batch_shape: array<u32>;
  @group(0) @binding(6) var<storage, read> batch_strides_mat: array<u32>;
  @group(0) @binding(7) var<storage, read> batch_strides_vec: array<u32>;
}

if (!$enable_subgroups) {
  var<workgroup> workgroup_result: array<dtype, group_rows * cols_per_workgroup>;
}

@compute @workgroup_size(group_size, group_count)
fn gemvt(@builtin(workgroup_id) tid: vec3<u32>,
         @builtin(local_invocation_id) lid: vec3<u32>) {
  // Position in the group and workgroup.
  let row_in_group = lid.x / group_cols;
  let col_in_group = lid.x % group_cols;
  let row_in_workgroup = row_in_group;
  let col_in_workgroup = lid.y * group_cols + col_in_group;

  // The col worked on.
  let out_col = tid.x * cols_per_workgroup + col_in_workgroup * col_work_per_thread;

  // Offset of current batch.
  let out_offset = tid.z * mat_cols + out_col;
  if ($contiguous) {
    let mat_offset = tid.z * mat_rows * mat_cols;
    let vec_offset = tid.z * mat_rows;
  } else {
    let mat_offset = coord_to_index(tid.z, &batch_shape, &batch_strides_mat);
    let vec_offset = coord_to_index(tid.z, &batch_shape, &batch_strides_vec);
  }

  // Per-thread result and intermediates.
  var result: array<dtype, col_work_per_thread>;
  var coefficient: array<dtype, row_work_per_thread>;
  var intermediate: array<dtype, col_work_per_thread>;

  // Loop over vector.
  for (var block = 0u; block < mat_rows / rows_per_workgroup; block++) {
    let row = block * rows_per_workgroup + row_in_workgroup * row_work_per_thread;
    // Load vector.
    for (var r = 0u; r < row_work_per_thread; r++) {
      coefficient[r] = vec[vec_offset + row + r];
    }

    // Work.
    for (var r = 0u; r < row_work_per_thread; r++) {
      // Load col.
      for (var c = 0u; c < col_work_per_thread; c++) {
        intermediate[c] = mat[mat_offset + (row + r) * mat_cols + out_col + c];
      }

      // Accumulate results.
      for (var c = 0u; c < col_work_per_thread; c++) {
        if ($dtype_is_floating) {
          result[c] = fma(intermediate[c], coefficient[r], result[c]);
        } else {
          result[c] += intermediate[c] * coefficient[r];
        }
      }
    }
  }

  // Leftover in rows.
  let leftover = mat_rows % rows_per_workgroup;
  if (leftover != 0) {
    let row = mat_rows - leftover + row_in_workgroup * row_work_per_thread;
    for (var r = 0u; r < row_work_per_thread && row + r < mat_rows; r++) {
      coefficient[r] = vec[vec_offset + row + r];
      for (var c = 0u; c < col_work_per_thread && out_col + c < mat_cols; c++) {
        intermediate[c] = mat[mat_offset + (row + r) * mat_cols + out_col + c];
      }
      for (var c = 0u; c < col_work_per_thread && out_col + c < mat_cols; c++) {
        if ($dtype_is_floating) {
          result[c] = fma(intermediate[c], coefficient[r], result[c]);
        } else {
          result[c] += intermediate[c] * coefficient[r];
        }
      }
    }
  }

  if ($enable_subgroups) {
    // Subgroup accumulations.
    for (var c = 0u; c < col_work_per_thread; c++) {
      for (var delta = group_rows / 2; delta >= 1; delta >>= 1) {
        result[c] += subgroupShuffleDown(result[c], delta * group_cols);
      }
    }
  } else {
    // Write to shared memory.
    for (var c = 0u; c < col_work_per_thread; c++) {
      let idx = row_in_workgroup * cols_per_workgroup +
                col_in_workgroup * col_work_per_thread +
                c;
      workgroup_result[idx] = result[c];
    }

    // Workgroup accumulations.
    workgroupBarrier();
    for (var c = 0u; c < col_work_per_thread; c++) {
      let idx = row_in_workgroup * cols_per_workgroup +
                col_in_workgroup * col_work_per_thread +
                c;
      for (var delta = group_rows / 2; delta >= 1; delta >>= 1) {
        if (row_in_group < delta) {
          workgroup_result[idx] += workgroup_result[idx + delta * cols_per_workgroup];
        }
        workgroupBarrier();
      }
    }
  }

  // Write output.
  if (row_in_workgroup != 0 || out_col >= mat_cols) {
    return;
  }
  for (var c = 0u; c < col_work_per_thread; c++) {
    if (out_col + c < mat_cols) {
      if ($enable_subgroups) {
        out[out_offset + c] = result[c];
      } else {
        let idx = col_in_workgroup * col_work_per_thread + c;
        out[out_offset + c] = workgroup_result[idx];
      }
    }
  }
}

// include utils.wgsl
