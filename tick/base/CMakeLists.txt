cmake_minimum_required(VERSION 2.8)
project(mytick)

add_definitions("-DBUILDING_DLL" "-DPYTHON_LINK" "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
if(MSVC)
    add_definitions("-D_SCL_SECURE_NO_WARNINGS" "-wd4373" "-wd4250")
    add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
endif()

add_definitions(-DSWIG_PYTHON_INTERPRETER_NO_DEBUG)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib().replace('\\\\', '/'))" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
set(NUMPY_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/numpy/core/include)
set(NUMPY_LIB ${PYTHON_SITE_PACKAGES}/numpy/core/lib/npymath.lib)
include_directories(${NUMPY_INCLUDE_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cereal/include")


add_subdirectory(array/swig)
add_subdirectory(swig)
add_subdirectory(array_test/swig)
