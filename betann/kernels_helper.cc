#include "betann/kernels_helper.h"

namespace betann {

bool EnableSubgroups(Device& device, bool enableF16, bool disableSubgroups) {
  bool enableSubgroups = !disableSubgroups && device.SupportsSubgroups();
  if (enableSubgroups && enableF16) {
    enableSubgroups = device.SupportsF16() && device.SupportsSubgroupsF16();
  }
  return enableSubgroups;
}

VariablesMap GetCapacityVariables(Device& device,
                                  bool enableF16,
                                  bool disableSubgroups) {
  bool enableSubgroups = EnableSubgroups(device, enableF16, disableSubgroups);
  bool enableSubgroupsF16 = false;
  return {
    {"enable_f16", enableF16 ? device.SupportsF16() : false},
    {"enable_subgroups", enableSubgroups},
    {"enable_subgroups_f16", enableF16 && enableSubgroups},
#ifdef __APPLE__
    {"subgroup_min_size", 32u},
#else
    {"subgroup_min_size", 4u},
#endif
  };
}

}  // namespace betann
