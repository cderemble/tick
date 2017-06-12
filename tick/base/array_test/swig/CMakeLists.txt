
set_source_files_properties(SOURCE array_test.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE array_test.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../array/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../src")

add_definitions("-DBUILDING_DLL")

swig_add_library(array_test LANGUAGE python SOURCES array_test.i TYPE SHARED)
swig_link_libraries(array_test ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target} ${base_target})

target_sources(${SWIG_MODULE_array_test_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/array_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/typemap_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/varraycontainer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/sbasearray_container.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/performance_test.cpp"
    )
