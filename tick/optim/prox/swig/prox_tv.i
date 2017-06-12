%{
#include "prox_tv.h"
%}


class ProxTV : public Prox {


public:

    ProxTV(double strength,
           bool positive);

    ProxTV(double strength,
           ulong start,
           ulong end,
           bool positive);

    inline virtual void set_positive(bool positive);
};
