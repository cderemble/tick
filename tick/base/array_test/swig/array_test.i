%module array_test

%include defs.i

%{
#include "tick_python.h"
#include "varraycontainer.h"
#include "array_test.h"
#include "typemap_test.h"
#include "sbasearray_container.h"
%}

%import(module="tick.base.array.build.array") array_module.i

%include "array_test.h"
%include "typemap_test.h"

class VarrayUser {
public:
    VArrayDoublePtr varrayPtr;
    VarrayUser() {};
    std::int64_t nRef();
    void setArray(VarrayContainer vcc);
};

class VarrayContainer {
public:
    VArrayDoublePtr varrayPtr;
    VarrayContainer() {};
    std::int64_t nRef();
    void initVarray();
    void initVarray(int size);
};

%include performance_test.i

void test_sbasearray_container_new(SBaseArrayDoublePtr a);
void test_sbasearray_container_clear();
double test_sbasearray_container_compute();

void test_sbasearray2d_container_new(SBaseArrayDouble2dPtr a);
void test_sbasearray2d_container_clear();
double test_sbasearray2d_container_compute();
