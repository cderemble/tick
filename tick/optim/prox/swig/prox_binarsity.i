%{
#include "prox_binarsity.h"
%}


class ProxBinarsity: public Prox {

public:

    ProxBinarsity(double strength, SArrayULongPtr blocks_start,
                  SArrayULongPtr blocks_length, bool positive);

    ProxBinarsity(double strength, SArrayULongPtr blocks_start,
                  SArrayULongPtr blocks_length, ulong start,
                  ulong end, bool positive);

    inline void set_positive(bool positive);
};
