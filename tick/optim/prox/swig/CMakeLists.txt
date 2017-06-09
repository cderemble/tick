
set_source_files_properties(SOURCE prox_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE prox_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../base/src"
    )
add_definitions("-DBUILDING_DLL")

swig_add_library(prox LANGUAGE python SOURCES prox_module.i TYPE SHARED)
swig_link_libraries(prox ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target} ${base_target})

target_include_directories(${SWIG_MODULE_prox_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../base/array/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../base/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    )

target_sources(${SWIG_MODULE_prox_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_separable.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_zero.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_positive.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_l2sq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_l1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_l1w.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_tv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_elasticnet.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_sorted_l1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_multi.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_equality.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/prox_slope.cpp"
    )

set(prox_target ${SWIG_MODULE_prox_REAL_NAME} PARENT_SCOPE)
