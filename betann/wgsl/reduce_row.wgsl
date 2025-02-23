if ($enable_f16) {
  enable f16;
}
if ($enable_subgroups) {
  enable subgroups;
}
if ($enable_subgroups_f16) {
  enable subgroups_f16;
}

alias output_dtype = $output_dtype;
alias input_dtype = $input_dtype;

const workgroup_size: u32 = $workgroup_size;
const work_per_thread: u32 = 4;
if ($use_fast_index) {
  // The depth of loops to cache for coord_to_index_next.
  const coord_cache_size: u32 = $coord_cache_size;
}

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<uniform> num_outputs: u32;
@group(0) @binding(2) var<storage, read> input: array<input_dtype>;
@group(0) @binding(3) var<uniform> row_size: u32;
@group(0) @binding(4) var<uniform> non_row_reductions: u32;
@group(0) @binding(5) var<storage, read> non_reduction_shape: array<u32>;
@group(0) @binding(6) var<storage, read> non_reduction_strides: array<u32>;
@group(0) @binding(7) var<storage, read> reduction_shape: array<u32>;
@group(0) @binding(8) var<storage, read> reduction_strides: array<u32>;

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_row_1d_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= num_outputs) {
    return;
  }

  // The index in non-reduction dimensions.
  var input_offset = coord_to_index(gid.x, &non_reduction_shape, &non_reduction_strides);

  if ($use_fast_index) {
    // Stateful computation of index in reduction dimensions.
    var index_state: coord_to_index_state;
    coord_to_index_init(&index_state, arrayLength(&reduction_shape));
  }

  // Reduce by rows.
  var total = get_initial_value_$op();
  for (var r = 0u; r < non_row_reductions; r++) {
    if ($use_fast_index) {
      let row_offset = input_offset + coord_to_index_result(&index_state);
      coord_to_index_next(&index_state, &reduction_shape, &reduction_strides);
    } else {
      let row_offset = input_offset + coord_to_index(r, &reduction_shape, &reduction_strides);
    }
    row_reduce(&total, 0, &input, row_offset, row_size, 1, work_per_thread);
  }

  output[gid.x] = total;
}

if ($enable_subgroups) {
  var<workgroup> workgroup_totals: array<output_dtype, workgroup_size / $subgroup_min_size>;
} else {
  var<workgroup> workgroup_totals: array<output_dtype, workgroup_size>;
}

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_row_2d_$op(if ($enable_subgroups) {
                       @builtin(subgroup_size) subgroup_size: u32,
                       @builtin(subgroup_invocation_id) subgroup_gid: u32,
                     }
                     @builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
  // The index in non-reduction dimensions.
  var input_offset = coord_to_index(gid.y, &non_reduction_shape, &non_reduction_strides);

  if ($use_fast_index) {
    // Stateful computation of index in reduction dimensions.
    var index_state: coord_to_index_state;
    coord_to_index_init(&index_state, arrayLength(&reduction_shape));
  }

  // Reduce rows per thread.
  var total = get_initial_value_$op();
  for (var r = 0u; r < non_row_reductions; r++) {
    if ($use_fast_index) {
      let row_offset = input_offset + coord_to_index_result(&index_state);
      coord_to_index_next(&index_state, &reduction_shape, &reduction_strides);
    } else {
      let row_offset = input_offset + coord_to_index(r, &reduction_shape, &reduction_strides);
    }
    row_reduce(&total, lid.x, &input, row_offset, row_size, workgroup_size, work_per_thread);
  }

  // Reduce across the workgroup.
  if ($enable_subgroups) {
    workgroup_reduce(&total, lid.x, workgroup_size, subgroup_gid, subgroup_size);
  } else {
    workgroup_reduce(&total, lid.x, workgroup_size, 0, 0);
  }

  // Write output.
  if (lid.x == 0 && gid.y < num_outputs) {
    output[gid.y] = total;
  }
}

if ($use_fast_index) {
  // Compute index faster by caching the results.
  struct coord_to_index_state {
    contiguous: bool,
    dim: array<u32, coord_cache_size>,
    coord: array<u32, coord_cache_size>,
    index: array<u32, coord_cache_size>,
  };

  fn coord_to_index_init(state: ptr<function, coord_to_index_state>, ndim: u32) {
    state.contiguous = ndim < 2;
    for (var i = 0u; i < coord_cache_size; i++) {
      state.dim[i] = max(ndim - i, 0);
    }
  }

  fn coord_to_index_result(state: ptr<function, coord_to_index_state>) -> u32 {
    return state.index[0];
  }

  fn coord_to_index_next(state: ptr<function, coord_to_index_state>,
                         shape: ptr<storage, array<u32>>,
                         strides: ptr<storage, array<u32>>) {
    // Increase coordinate in last dimension, and carry to previous ones.
    for (var i = 0u; i < coord_cache_size; i++) {
      let dim = state.dim[i];
      if (dim == 0) {  // unused cache entry
        break;
      }
      if (i == coord_cache_size - 1) {  // the frist dimension
        if (state.contiguous) {
          state.index[i] += strides[0];
        } else {
          state.coord[i]++;
          if (dim > 1) {
            state.index[i] = coord_to_index_ndim(state.coord[i], shape, strides, i32(dim));
          } else {
            state.index[i] += strides[0];
          }
        }
      } else {  // non-contiguous dimensions
        state.coord[i]++;
        state.index[i] += strides[dim - 1];
        // Only continue the loop when need to carry to the N-1 dimension.
        if (state.coord[i] < shape[dim - 1]) {
          break;
        }
      }
    }
    // Iterate backwards and ignore first dimension.
    for (var i = coord_cache_size - 2; i32(i) >= 0; i--) {
      let dim = state.dim[i];
      if (dim == 0) {
        continue;
      }
      // Carry the index from N-1 dimension.
      if (state.coord[i] >= shape[dim - 1]) {
        state.coord[i] = 0;
        state.index[i] = state.index[i + 1];
      }
    }
  }
}

// include reduce_ops.wgsl
// include utils.wgsl
