#ifndef TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_
#define TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_

#include "prox_separable.h"

class ProxL2Sq : public ProxSeparable {
 public:
  ProxL2Sq(double strength,
           bool positive);

  ProxL2Sq(double strength,
           ulong start,
           ulong end,
           bool positive);

  const std::string get_class_name() const override;

  double value(double x) const override;

  double call(double x,
              double step) const override;

  // Repeat n_times the prox on coordinate i
  double call(double x,
              double step,
              ulong n_times) const override;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_
