
set_source_files_properties(SOURCE model_module.i PROPERTIES CPLUSPLUS ON)
if(WIN32)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set_source_files_properties(SOURCE model_module.i PROPERTIES SWIG_FLAGS "-D_WIN64")
    endif()
endif()

set(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS}
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../base/swig"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../base/src"
    )

swig_add_library(model LANGUAGE python SOURCES model_module.i TYPE SHARED)
swig_link_libraries(model ${PYTHON_LIBRARIES} ${NUMPY_LIB} ${array_target} ${base_target} ${crandom_target})

target_sources(${SWIG_MODULE_model_REAL_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/model_labels_features.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/model_generalized_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/model_generalized_linear_with_intercepts.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/model_lipschitz.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_fixed_expkern_loglik.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_fixed_expkern_leastsq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_fixed_sumexpkern_leastsq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/hawkes_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/linreg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/linreg_with_intercepts.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/logreg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/poisreg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/coxreg_partial_lik.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/variants/hawkes_fixed_sumexpkern_leastsq_list.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/variants/hawkes_fixed_expkern_leastsq_list.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/variants/hawkes_fixed_expkern_loglik_list.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/variants/hawkes_leastsq_list.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/base/hawkes_single.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/base/hawkes_list.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/base/hawkes_model.cpp"
    )
