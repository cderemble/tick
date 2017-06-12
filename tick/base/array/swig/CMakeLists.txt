
set_source_files_properties(SOURCE array_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE array_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../swig")

add_definitions("-DBUILDING_DLL")

swig_add_library(array LANGUAGE python SOURCES array_module.i TYPE SHARED)
swig_link_libraries(array ${PYTHON_LIBRARIES} ${NUMPY_LIB})

target_sources(${SWIG_MODULE_array_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/alloc.cpp"
    )

set(array_target ${SWIG_MODULE_array_REAL_NAME} PARENT_SCOPE)
