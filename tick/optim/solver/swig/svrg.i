%include <std_shared_ptr.i>

%{
#include "svrg.h"
#include "model.h"
%}

class SVRG : public StoSolver {

public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };

    SVRG(ulong epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed,
         VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last);

    void solve();

    void set_step(double step);

    VarianceReductionMethod get_variance_reduction();

    void set_variance_reduction(VarianceReductionMethod variance_reduction);
};
