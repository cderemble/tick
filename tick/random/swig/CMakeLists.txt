
set_source_files_properties(SOURCE crandom.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE crandom.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/src"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../src"
    )

add_definitions("-DBUILDING_DLL")

swig_add_library(crandom LANGUAGE python SOURCES crandom.i TYPE SHARED)
swig_link_libraries(crandom ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target})

target_sources(${SWIG_MODULE_crandom_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/rand.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/test_rand.cpp"
    )

set(crandom_target ${SWIG_MODULE_crandom_REAL_NAME} PARENT_SCOPE)
