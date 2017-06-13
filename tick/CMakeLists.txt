cmake_minimum_required(VERSION 2.8)
project(tick)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CBLAS)

if (${CBLAS_FOUND})
    add_definitions(-DTICK_CBLAS_AVAILABLE)

    set(EXTRA_TEST_LIBS ${CBLAS_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CBLAS_LINKER_FLAGS}")

    message(STATUS "BLAS Libraries: " ${CBLAS_LIBRARIES})
    message(STATUS "BLAS Linker flags: " ${CBLAS_LINKER_FLAGS})
endif()

find_package(Backtrace QUIET)
configure_file(backtrace.in ${CMAKE_CURRENT_SOURCE_DIR}/base/src/backtrace.h)

if(MSVC)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS -wd4373 -wd4250)
        add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
        add_definitions(-D_USE_MATH_DEFINES)
endif()
add_definitions(-Wall)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/base/src
        ${CMAKE_CURRENT_SOURCE_DIR}/base/array/src
        ${CMAKE_CURRENT_SOURCE_DIR}/base/array_test/src
        ${CMAKE_CURRENT_SOURCE_DIR}/random/src
        ${CMAKE_CURRENT_SOURCE_DIR}/simulation/src
        ${CMAKE_CURRENT_SOURCE_DIR}/optim/model/src
        ${CMAKE_CURRENT_SOURCE_DIR}/optim/prox/src
        ${CMAKE_CURRENT_SOURCE_DIR}/optim/solver/src
        ${CMAKE_CURRENT_SOURCE_DIR}/inference/src
        ${CMAKE_CURRENT_SOURCE_DIR}/preprocessing/src
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal/include)

add_subdirectory(base/src)
add_subdirectory(base/array/src)
add_subdirectory(base/array_test/src)
add_subdirectory(random/src)
add_subdirectory(simulation/src)
add_subdirectory(optim/model/src)
add_subdirectory(optim/prox/src)
add_subdirectory(optim/solver/src)
add_subdirectory(inference/src)
add_subdirectory(preprocessing/src)

set(EXTRA_LIBS ${EXTRA_LIBS} array base random array_test simulation model prox
        solver inference)

find_package(GTest)

if (${GTEST_FOUND})
    enable_testing()
    message(STATUS "Testing in C++ enabled")

    find_package (Threads REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    if(APPLE)
        find_library(ACCELERATE_LIBRARY Accelerate)
        mark_as_advanced(ACCELERATE_LIBRARY)
        set(EXTRA_TEST_LIBS ${EXTRA_TEST_LIBS} ${ACCELERATE_LIBRARY})
    endif(APPLE)

    add_subdirectory(base/tests/src)
    add_subdirectory(base/array/tests/src)
    add_subdirectory(optim/model/tests/src)
    add_subdirectory(simulation/tests/src)

    add_custom_target(check
            COMMAND base/tests/src/tick_test_base
            COMMAND base/array/tests/src/tick_test_array
            COMMAND base/array/tests/src/tick_test_varray
            COMMAND optim/model/tests/src/tick_test_model
            COMMAND simulation/tests/src/tick_test_hawkes
            )
else()
    message(STATUS "Testing in C++ NOT enabled")

    add_custom_target(check COMMAND
            echo "Could not run C++ checks - gtest framework not found"
            )
endif()


if(BUILD_SWIG)
    add_definitions(-DPYTHON_LINK -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
    add_definitions(-DSWIG_PYTHON_INTERPRETER_NO_DEBUG)

    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})
    
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib().replace('\\\\', '/'))" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(NUMPY_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/numpy/core/include)
    if(MSVC)
        set(NUMPY_LIB ${PYTHON_SITE_PACKAGES}/numpy/core/lib/npymath.lib)
    else()
        set(NUMPY_LIB)
    endif()
    include_directories(${NUMPY_INCLUDE_DIR})

    add_subdirectory(base/array/swig)
    add_subdirectory(base/swig)
    add_subdirectory(base/array_test/swig)
    add_subdirectory(random/swig)
    add_subdirectory(simulation/swig)
    add_subdirectory(optim/model/swig)
    add_subdirectory(optim/prox/swig)
    add_subdirectory(optim/solver/swig)
    add_subdirectory(preprocessing/swig)
    add_subdirectory(inference/swig)
endif()
