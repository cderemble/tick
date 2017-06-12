%include <std_shared_ptr.i>

%{
#include "sgd.h"
#include "model.h"
%}

class SGD : public StoSolver {

public:

    SGD(ulong epoch_size,
        double tol,
        RandType rand_type,
        double step,
        int seed);

    inline void set_step(double step);

    inline double get_step() const;

    void solve();
};
