#ifndef TICK_OPTIM_PROX_SRC_PROX_SEPARABLE_H_
#define TICK_OPTIM_PROX_SRC_PROX_SEPARABLE_H_

#include "prox.h"

class ProxSeparable : public Prox {
 public:
  ProxSeparable(double strength,
                bool positive);

  ProxSeparable(double strength,
                ulong start,
                ulong end,
                bool positive);

  const std::string get_class_name() const override;

  const bool is_separable() const override;

  void call(const ArrayDouble &coeffs,
            const ArrayDouble &step,
            ArrayDouble &out) override;

  void call(const ArrayDouble &coeffs,
            double step,
            ArrayDouble &out,
            ulong start,
            ulong end) override;

  void call(const ArrayDouble &coeffs,
            const ArrayDouble &step,
            ArrayDouble &out,
            ulong start,
            ulong end) override;

  double call(double x,
              double step) const override;

  double call(double x,
              double step,
              ulong n_times) const override;

  // Compute the prox on the i-th coordinate only
  void call(ulong i,
            const ArrayDouble &coeffs,
            double step,
            ArrayDouble &out) const override;

  // Repeat n_times the prox on coordinate i
  void call(ulong i,
            const ArrayDouble &coeffs,
            double step,
            ArrayDouble &out,
            ulong n_times) const override;

  double value(double x) const override;

  double value(const ArrayDouble &coeffs,
               ulong start,
               ulong end) override;

  // Compute the value given by the i-th coordinate only (multiplication by
  // lambda must not be done here)
  double value(ulong i,
               const ArrayDouble &coeffs) const override;
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_SEPARABLE_H_
