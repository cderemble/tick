
#ifndef MLPP_PROX_BINARSITY_H
#define MLPP_PROX_BINARSITY_H


#include <sarray.h>
#include "prox.h"


class ProxBinarsity: public Prox {

protected:

    bool positive;
    ulong n_blocks;

    SArrayULongPtr blocks_start;
    SArrayULongPtr blocks_length;

    // A vector that contains the prox for each block
    std::vector<ProxTV> blocks_prox;

    // Tells us if the prox is ready (with correctly allocated sub-prox for each blocks).
    // This is mainly necessary when the user changes the range from python
    bool ready;

    void prepare();

public:

    ProxBinarsity(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                  bool positive);

    ProxBinarsity(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                  ulong start, ulong end, bool positive);

    const std::string get_class_name() const;

    double _value(ArrayDouble &coeffs, ulong start, ulong end);

    void _call(ArrayDouble &coeffs, double step, ArrayDouble &out, ulong start, ulong end);

    inline void set_positive(bool positive)
    {
        this->positive = positive;
    }

    // We overload set_start_end here, since we'd need to update blocks_prox when it's changed
    inline void set_start_end(ulong start, ulong end) {
        if ((start != this->start) or (end != this->end)) {
            // If we change the range, we need to compute again the weights
            ready = false;
        }
        this->has_range = true;
        this->start = start;
        this->end = end;
    }
};


#endif //MLPP_PROX_BINARSITY_H
