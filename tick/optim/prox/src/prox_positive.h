#ifndef TICK_OPTIM_PROX_SRC_PROX_POSITIVE_H_
#define TICK_OPTIM_PROX_SRC_PROX_POSITIVE_H_

#include "prox_separable.h"

class ProxPositive : public ProxSeparable {
 public:
  explicit ProxPositive(double strength);

  ProxPositive(double strength,
               ulong start,
               ulong end);

  const std::string get_class_name() const override;

  double call(double x,
              double step) const override;

  // Repeat n_times the prox on coordinate i
  double call(double x,
              double step,
              ulong n_times) const override;

  // Override value, only this value method should be called
  double value(const ArrayDouble &coeffs,
               ulong start,
               ulong end) override;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_POSITIVE_H_
