%{
#include "prox_zero.h"
%}


class ProxZero : public Prox {

public:

    ProxZero(double strength);

    ProxZero(double strength,
             ulong start,
             ulong end);
};
