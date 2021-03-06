//
// Created by Martin Bompaire on 26/10/15.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_H_
#define TICK_OPTIM_PROX_SRC_PROX_H_

#include <memory>
#include <string>

#include "base.h"

class DLL_PUBLIC Prox {
 protected:
    // Weight of the proximal operator
    double strength;

    // Flag to know if proximal operator concerns only a part of the vector
    bool has_range;

    // If range is restricted it will be applied from index start to index end
    ulong start, end;

 public:
    explicit Prox(double strength);

    Prox(double strength, ulong start, ulong end);

    virtual const std::string get_class_name() const;

    virtual double value(ArrayDouble &coeffs);

    virtual double _value(ArrayDouble &coeffs,
                          ulong start,
                          ulong end);

    virtual void call(ArrayDouble &coeffs,
                      double step,
                      ArrayDouble &out);

    virtual void call(ArrayDouble &coeffs,
                      ArrayDouble &step,
                      ArrayDouble &out);

    virtual void _call(ArrayDouble &coeffs,
                       double step,
                       ArrayDouble &out,
                       ulong start,
                       ulong end);

    virtual void set_strength(double strength);

    virtual double get_strength() const;

    virtual void set_start_end(ulong start, ulong end);

    ulong get_start();
    ulong get_end();
};

#if defined(_MSC_VER)
template class DLL_PUBLIC std::shared_ptr<Prox>;
#endif

typedef std::shared_ptr<Prox> ProxPtr;

#endif  // TICK_OPTIM_PROX_SRC_PROX_H_
