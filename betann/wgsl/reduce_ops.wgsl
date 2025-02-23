// include constants.wgsl

// The initial values for reduction ops.
const initial_value_and = output_dtype(1);
const initial_value_or = output_dtype(0);
const initial_value_sum = output_dtype(0);
const initial_value_prod = output_dtype(1);
const initial_value_min = max_value_$output_dtype;
const initial_value_max = min_value_$output_dtype;

fn get_initial_value_$op() -> output_dtype {
  return initial_value_$op;
}

// Reduce the row.
fn row_reduce(total: ptr<function, output_dtype>,
              lid: u32,
              input: ptr<storage, array<input_dtype>>,
              row_offset: u32,
              row_size: u32,
              workgroup_size: u32,
              work_per_thread: u32) {
  // Loop over row.
  let block_size = workgroup_size * work_per_thread;
  for (var block = 0u; block < row_size / block_size; block++) {
    let idx = block * block_size + lid * work_per_thread;
    for (var i = 0u; i < work_per_thread; i++) {
      *total = reduce_op_$op(output_dtype(input[row_offset + idx + i]), *total);
    }
  }

  // Leftover in row.
  let leftover = row_size % block_size;
  if (leftover != 0) {
    let idx = (row_size - leftover) + lid * work_per_thread;
    for (var i = 0u; i < work_per_thread && idx + i < row_size; i++) {
      *total = reduce_op_$op(output_dtype(input[row_offset + idx + i]), *total);
    }
  }
}

// Reduce results from workgroup threads to total.
fn workgroup_reduce(total: ptr<function, output_dtype>,
                    lid: u32,
                    workgroup_size: u32,
                    subgroup_gid: u32,
                    subgroup_size: u32) {
  if ($enable_subgroups) {
    // Subgroup reduction.
    *total = reduce_subgroup_op_$op(*total);

    // Workgroup reduction.
    // FIXME(zcbenz): Must convert delta to f32 for comparison, possible Metal bug.
    for (var delta = workgroup_size / subgroup_size; f32(delta) > 1f; delta /= subgroup_size) {
      // Write first lane's result to shared memory.
      if (subgroup_gid == 0) {
        workgroup_totals[lid / subgroup_size] = *total;
      }

      // Subgroup reduction.
      workgroupBarrier();
      *total = select(initial_value_$op, workgroup_totals[lid], lid < delta);
      *total = reduce_subgroup_op_$op(*total);
    }
  } else {
    // Write to shared memory.
    workgroup_totals[lid] = *total;

    // Workgroup reduction.
    workgroupBarrier();
    for (var delta = workgroup_size / 2; delta >= 1; delta >>= 1) {
      if (lid < delta) {
        workgroup_totals[lid] = reduce_op_$op(workgroup_totals[lid],
                                              workgroup_totals[lid + delta]);
      }
      workgroupBarrier();
    }

    if (lid == 0) {
      *total = workgroup_totals[0];
    }
  }
}

// The reduction ops.
fn reduce_op_and(a: output_dtype, b: output_dtype) -> output_dtype {
  return output_dtype(bool(a) && bool(b));
}

fn reduce_op_or(a: output_dtype, b: output_dtype) -> output_dtype {
  return output_dtype(bool(a) || bool(b));
}

fn reduce_op_sum(a: output_dtype, b: output_dtype) -> output_dtype {
  return a + b;
}

fn reduce_op_prod(a: output_dtype, b: output_dtype) -> output_dtype {
  return a * b;
}

fn reduce_op_min(a: output_dtype, b: output_dtype) -> output_dtype {
  return min(a, b);
}

fn reduce_op_max(a: output_dtype, b: output_dtype) -> output_dtype {
  return max(a, b);
}

if ($enable_subgroups) {
  fn reduce_subgroup_op_and(v: output_dtype) -> output_dtype {
    return output_dtype(subgroupAll(bool(v)));
  }

  fn reduce_subgroup_op_or(v: output_dtype) -> output_dtype {
    return output_dtype(subgroupAny(bool(v)));
  }

  fn reduce_subgroup_op_sum(v: output_dtype) -> output_dtype {
    return subgroupAdd(v);
  }

  fn reduce_subgroup_op_prod(v: output_dtype) -> output_dtype {
    return subgroupMul(v);
  }

  fn reduce_subgroup_op_min(v: output_dtype) -> output_dtype {
    return subgroupMin(v);
  }

  fn reduce_subgroup_op_max(v: output_dtype) -> output_dtype {
    return subgroupMax(v);
  }
}
