%{
#include "prox_separable.h"
%}


class ProxSeparable : public Prox {

public:

    ProxSeparable(double strength);

    ProxSeparable(double strength,
                  ulong start,
                  ulong end);
};
