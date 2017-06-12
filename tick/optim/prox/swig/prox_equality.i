%{
#include "prox_equality.h"
%}


class ProxEquality : public Prox {


public:

    ProxEquality(double strength, bool positive);

    ProxEquality(double strength,
                 ulong start,
                 ulong end,
                 bool positive);

    inline virtual void set_positive(bool positive);
};
