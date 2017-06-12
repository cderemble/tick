
set_source_files_properties(SOURCE solver_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE solver_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../base/src"
    )

swig_add_library(solver LANGUAGE python SOURCES solver_module.i TYPE SHARED)
swig_link_libraries(solver ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target} ${base_target} ${crandom_target} ${prox_target})

target_sources(${SWIG_MODULE_solver_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/sto_solver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/sgd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/svrg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/sdca.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/adagrad.cpp"
    )
