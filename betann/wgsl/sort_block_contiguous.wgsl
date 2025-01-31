enable f16;

alias dtype = $0;

const num_threads: u32 = 256;
const work_per_thread: u32 = 8;

const n_per_block = num_threads * work_per_thread;

@group(0) @binding(0) var<storage, read_write> out: array<dtype>;
@group(0) @binding(1) var<uniform> size_sorted_axis: u32;
@group(0) @binding(2) var<uniform> out_stride_sorted_axis: u32;
@group(0) @binding(3) var<uniform> out_stride_segment_axis: u32;
@group(0) @binding(4) var<storage, read> input: array<dtype>;
@group(0) @binding(5) var<uniform> input_stride_sorted_axis: u32;
@group(0) @binding(6) var<uniform> input_stride_segment_axis: u32;

var<workgroup> workgroup_vals: array<dtype, n_per_block>;

@compute @workgroup_size(num_threads, 1, 1)
fn sort_block_c(@builtin(workgroup_id) tid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
  // Copy into workgroup memory.
  let input_offset = tid.y * input_stride_segment_axis;
  for (var i = lid.x; i < n_per_block; i += num_threads) {
    workgroup_vals[i] = select(dtype_max_value(),
                               input[input_offset + i * input_stride_sorted_axis],
                               i < size_sorted_axis);
  }

  // Sort elements in workgroup.
  workgroupBarrier();
  sort_in_workgroup(size_sorted_axis, lid);
  workgroupBarrier();

  // Write output.
  let out_offset = tid.y * out_stride_segment_axis;
  for (var i = lid.x; i < size_sorted_axis; i += num_threads) {
    out[out_offset + i * out_stride_sorted_axis] = workgroup_vals[i];
  }
}

fn sort_in_workgroup(size_sorted_axis: u32, lid: vec3<u32>) {
  // Load from shared memory.
  var vals: array<dtype, work_per_thread>;
  let idx = lid.x * work_per_thread;
  for (var i: u32 = 0; i < work_per_thread; i++) {
    vals[i] = workgroup_vals[idx + i];
  }

  // Per-thread odd-even sort.
  if (idx < size_sorted_axis) {
    for (var i: u32 = 0; i < work_per_thread; i++) {
      for (var j: u32 = i & 1; j < work_per_thread; j += 2) {
        if (compare_op(vals[j + 1], vals[j])) {
          let tmp = vals[j];
          vals[j] = vals[j + 1];
          vals[j + 1] = tmp;
        }
      }
    }
  }

  // Do merges using threadgroup memory.
  for (var merge_threads: u32 = 2;
       merge_threads <= num_threads;
       merge_threads *= 2) {
    // Update threadgroup memory.
    workgroupBarrier();
    for (var i: u32 = 0; i < work_per_thread; i++) {
      workgroup_vals[idx + i] = vals[i];
    }
    workgroupBarrier();

    // Split threads into merge groups and lanes.
    let merge_group = lid.x / merge_threads;
    let merge_lane = lid.x % merge_threads;

    let sort_size = work_per_thread * merge_threads;
    let sort_start = work_per_thread * merge_threads * merge_group;

    // Elements are sorted:
    // workgroup_vals[a_start] with a_size
    // workgroup_vals[b_start] with b_size
    var a_start = sort_start;
    var a_size = sort_size / 2;
    var b_start = a_start + a_size;
    var b_size = a_size;

    // Find a partition of merge elements.
    let sort_median = work_per_thread * merge_lane;
    let part = merge_partition(a_start, a_size, b_start, b_size, sort_median);

    a_start += part;
    b_start += sort_median - part;
    a_size -= part;
    b_size -= sort_median - part;

    // Merge starting at the partition and store results in thread registers.
    merge_step(a_start, a_size, b_start, b_size, &vals);
  }

  // Write out.
  workgroupBarrier();
  for (var i: u32 = 0; i < work_per_thread; i++) {
    workgroup_vals[idx + i] = vals[i];
  }
}

fn merge_partition(a_offset: u32, a_size: u32,
                   b_offset: u32, b_size: u32,
                   sort_median: u32) -> u32 {
  var a_start = select(0, sort_median - b_size, sort_median > b_size);
  var a_end = min(sort_median, a_size);

  while (a_start < a_end) {
    let middle = a_start + (a_end - a_start) / 2;
    let a = workgroup_vals[a_offset + middle];
    let b = workgroup_vals[b_offset + sort_median - 1 - middle];

    if (compare_op(b, a)) {
      a_end = middle;
    } else {
      a_start = middle + 1;
    }
  }

  return a_end;
}

fn merge_step(a_offset: u32, a_size: u32,
              b_offset: u32, b_size: u32,
              vals: ptr<function, array<dtype, work_per_thread>>) {
  var a_idx: u32 = 0;
  var b_idx: u32 = 0;

  for (var i: u32 = 0; i < work_per_thread; i++) {
    let a = workgroup_vals[a_offset + a_idx];
    let b = workgroup_vals[b_offset + b_idx];
    let pred = (b_idx < b_size) && (a_idx >= a_size || compare_op(b, a));

    vals[i] = select(a, b, pred);

    a_idx += u32(!pred);
    b_idx += u32(pred);
  }
}

fn compare_op(a: dtype, b: dtype) -> bool {
  return a < b;
}

const i32_max = 0x7fffffffi;
const u32_max = 0xffffffffu;
const f32_max = 0x1.fffffep+127f;
const f16_max = 0x1.ffcp+15h;

fn dtype_max_value() -> dtype {
  return $0_max;
}
