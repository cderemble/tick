%{
#include "prox_separable.h"
%}

class ProxSeparable : public Prox {
 public:
   ProxSeparable(double strength,
                 bool positive);

   ProxSeparable(double strength,
                 unsigned long start,
                 unsigned long end,
                 bool positive);
};
