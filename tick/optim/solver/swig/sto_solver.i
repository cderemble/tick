
%include <std_shared_ptr.i>

%{
#include "sto_solver.h"
#include "model.h"
%}


// Type of randomness used when sampling at random data points
enum class RandType {
    unif = 0,
    perm
};


class StoSolver {
    // Base abstract for a stochastic solver

public:

    StoSolver(ulong epoch_size,
              double tol,
              RandType rand_type);

    virtual void solve();

    virtual void get_minimizer(ArrayDouble &out);

    virtual void get_iterate(ArrayDouble &out);

    virtual void set_starting_iterate(ArrayDouble &new_iterate);

    inline void set_tol(double tol);
    inline double get_tol() const;

    inline void set_epoch_size(ulong epoch_size);
    inline ulong get_epoch_size() const;

    inline void set_rand_type(RandType rand_type);
    inline RandType get_rand_type() const;

    inline void set_rand_max(ulong rand_max);
    inline ulong get_rand_max() const;

    virtual void set_model(std::shared_ptr<Model> model);

    virtual void set_prox(std::shared_ptr<Prox> prox);

    void set_seed(int seed);

};
