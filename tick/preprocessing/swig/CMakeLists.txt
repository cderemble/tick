
set_source_files_properties(SOURCE preprocessing_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE preprocessing_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../base/src"
    )

swig_add_library(preprocessing LANGUAGE python SOURCES preprocessing_module.i TYPE SHARED)
swig_link_libraries(preprocessing ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target})

target_sources(${SWIG_MODULE_preprocessing_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/sparse_longitudinal_features_product.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/longitudinal_features_lagger.cpp"
    )
