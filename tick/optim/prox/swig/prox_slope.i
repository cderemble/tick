%{
#include "prox_slope.h"
%}

enum class WeightsType {
    bh = 0,
    oscar
};

class ProxSlope : public Prox {
 public:
   ProxSlope(double lambda,
             double fdr,
             bool positive);

   ProxSlope(double lambda, double fdr,
             unsigned long start,
             unsigned long end,
             bool positive);

   inline double get_fdr() const;

   inline void set_fdr(double fdr);

   inline double get_weight_i(unsigned long i);
};
