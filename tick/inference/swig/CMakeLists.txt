
set_source_files_properties(SOURCE inference_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE inference_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/src"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../optim/model/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../optim/model/src"
    )

swig_add_library(inference LANGUAGE python SOURCES inference_module.i TYPE SHARED)
swig_link_libraries(inference ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target} ${base_target})

target_sources(${SWIG_MODULE_inference_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_conditional_law.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_em.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_adm4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_basis_kernels.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_sumgaussians.cpp"
    )
