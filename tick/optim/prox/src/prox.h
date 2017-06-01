#ifndef TICK_OPTIM_PROX_SRC_PROX_H_
#define TICK_OPTIM_PROX_SRC_PROX_H_
#include <memory>
#include "base.h"
#include <string>


// TODO: all prox have a positive flag.
// TODO: use of final keyword
class Prox {
 protected:
  // Weight of the proximal operator
  double strength;
  // Flag to know if proximal operator concerns only a part of the vector
  bool has_range;
  // If range is restricted it will be applied from index start to index end
  ulong start, end;
  // Do we apply a non-negativity constraint ?
  bool positive;

 public:
  Prox(double strength,
       bool positive);

  Prox(double strength,
       ulong start,
       ulong end,
       bool positive);

  virtual const std::string get_class_name() const;

  virtual const bool is_separable() const;

  virtual void call(const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out);

  virtual void call(const ArrayDouble &coeffs,
                    const ArrayDouble &step,
                    ArrayDouble &out);

  virtual void call(const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out,
                    ulong start,
                    ulong end);

  virtual void call(const ArrayDouble &coeffs,
                    const ArrayDouble &step,
                    ArrayDouble &out,
                    ulong start,
                    ulong end);

  virtual double call(double x,
                      double step) const;

  virtual double call(double x,
                      double step,
                      ulong n_times) const;

  // Compute the prox on the i-th coordinate only
  virtual void call(ulong i,
                    const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out) const;

  // Repeat n_times the prox on coordinate i
  virtual void call(ulong i,
                    const ArrayDouble &coeffs,
                    double step,
                    ArrayDouble &out,
                    ulong n_times) const;

  virtual double value(const ArrayDouble &coeffs);

  virtual double value(const ArrayDouble &coeffs,
                       ulong start,
                       ulong end);

  virtual double value(double x) const;

  // Compute the value given by the i-th coordinate only (multiplication by
  // lambda must not be done here)
  virtual double value(ulong i,
                       const ArrayDouble &coeffs) const;

  virtual double get_strength() const;

  virtual void set_strength(double strength);

  virtual ulong get_start() const;

  virtual ulong get_end() const;

  virtual void set_start_end(ulong start,
                             ulong end);

  virtual bool get_positive() const;

  virtual void set_positive(bool positive);
};

typedef std::shared_ptr<Prox> ProxPtr;

#endif  // TICK_OPTIM_PROX_SRC_PROX_H_
