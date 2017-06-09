
set_source_files_properties(SOURCE base_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE base_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-I${CMAKE_CURRENT_SOURCE_DIR}/../src")
add_definitions("-DBUILDING_DLL")

swig_add_library(base LANGUAGE python SOURCES base_module.i TYPE SHARED)
swig_link_libraries(base ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target})

target_include_directories(${SWIG_MODULE_base_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../array/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    )

target_sources(${SWIG_MODULE_base_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/math/normal_distribution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/math/t2exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/exceptions_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/interruption.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/time_func.cpp")

set(base_target ${SWIG_MODULE_base_REAL_NAME} PARENT_SCOPE)
