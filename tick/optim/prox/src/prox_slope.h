#ifndef TICK_OPTIM_PROX_SRC_PROX_SLOPE_H_
#define TICK_OPTIM_PROX_SRC_PROX_SLOPE_H_

#include "prox.h"
#include "prox_sorted_l1.h"

class ProxSlope : public ProxSortedL1 {
 protected:
  double fdr;
  void compute_weights(void) override;

 public:
  ProxSlope(double strength,
            double fdr,
            bool positive);

  ProxSlope(double strength,
            double fdr,
            ulong start,
            ulong end,
            bool positive);

  const std::string get_class_name() const override;

  inline double get_fdr() const {
    return fdr;
  }

  inline void set_fdr(double fdr) {
    if (fdr != this->fdr) {
      weights_ready = false;
    }
    this->fdr = fdr;
  }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_SLOPE_H_
