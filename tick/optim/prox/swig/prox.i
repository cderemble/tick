%{
#include "prox.h"
%}


class Prox {

public:

    Prox(double strength);

    Prox(double strength,
         ulong start,
         ulong end);

    virtual double value(ArrayDouble &coeffs);

    virtual void call(ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out);

    virtual void call(ArrayDouble &coeffs,
                      ArrayDouble &step,
                      ArrayDouble &out);

    inline virtual void set_strength(double strength);

    inline virtual double get_strength() const;

    inline virtual void set_start_end(ulong start,
                                      ulong end);
};

typedef std::shared_ptr<Prox> ProxPtr;
