%module prox

%include defs.i
%include std_shared_ptr.i

%shared_ptr(Prox);
%shared_ptr(ProxSeparable);
%shared_ptr(ProxZero);
%shared_ptr(ProxPositive);
%shared_ptr(ProxL2Sq);
%shared_ptr(ProxL1);
%shared_ptr(ProxL1w);
%shared_ptr(ProxTV);
%shared_ptr(ProxElasticNet);
%shared_ptr(ProxSlope);
%shared_ptr(ProxMulti);
%shared_ptr(ProxEquality);

%{
#include "tick_python.h"
%}

%import(module="tick.base") base_module.i

%include "prox.i"

%include "prox_separable.i"

%include prox_zero.i

%include prox_positive.i

%include prox_l2sq.i

%include prox_l1.i

%include prox_l1w.i

%include prox_tv.i

%include prox_elasticnet.i

%include prox_slope.i

%include prox_multi.i

%include prox_equality.i
