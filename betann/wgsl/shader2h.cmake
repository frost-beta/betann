include("${BETANN_SOURCE_ROOT}/betann/wgsl/bin2h.cmake")
foreach(source ${BETANN_WGSL_SOURCES})
  list(APPEND BETANN_WGSL_SOURCES_ABS "${BETANN_SOURCE_ROOT}/${source}")
endforeach()
bin2h(SOURCE_FILES ${BETANN_WGSL_SOURCES_ABS}
      NULL_TERMINATE
      VARIABLE_NAME "wgsl_source"
      HEADER_NAMESPACE "betann"
      HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/gen/wgsl_sources.h")
