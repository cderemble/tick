#ifndef TICK_OPTIM_PROX_SRC_PROX_ZERO_H_
#define TICK_OPTIM_PROX_SRC_PROX_ZERO_H_

#include "prox_separable.h"

class ProxZero : public ProxSeparable {
 public:
  explicit ProxZero(double strength);

  ProxZero(double strength,
           ulong start,
           ulong end);

  const std::string get_class_name() const override;

  double call(double x,
              double step) const override;

  double call(double x,
              double step,
              ulong n_times) const override;

  double value(const ArrayDouble &coeffs,
               ulong start,
               ulong end) override;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_ZERO_H_
