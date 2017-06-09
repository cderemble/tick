
set_source_files_properties(SOURCE simulation_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE simulation_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/src"
    )

swig_add_library(simulation LANGUAGE python SOURCES simulation_module.i TYPE SHARED)
swig_link_libraries(simulation ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target} ${base_target} ${crandom_target})

target_include_directories(${SWIG_MODULE_simulation_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../base/array/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../base/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../random/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    )

target_sources(${SWIG_MODULE_simulation_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/pp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/poisson.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/inhomogeneous_poisson.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_kernels/hawkes_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_kernels/hawkes_kernel_time_func.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_kernels/hawkes_kernel_power_law.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_kernels/hawkes_kernel_exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_kernels/hawkes_kernel_sum_exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_baselines/constant_baseline.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_baselines/timefunction_baseline.cpp"
    )
