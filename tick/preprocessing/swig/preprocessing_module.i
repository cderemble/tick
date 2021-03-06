%module preprocessing

%include defs.i
%include std_shared_ptr.i
%include serialization.i

%shared_ptr(SparseLongitudinalFeaturesProduct);
%shared_ptr(LongitudinalFeaturesLagger);

%{
#include "tick_python.h"
%}

%import(module="tick.base") base_module.i

%include sparse_longitudinal_features_product.i
%include longitudinal_features_lagger.i